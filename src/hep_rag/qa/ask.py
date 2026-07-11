"""Ask grounded questions over the HEP literature corpus."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hep_rag.search.common import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_RERANK_BASE_URL,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL,
    SearchHit,
    post_json,
)
from hep_rag.search.pgvector_search import (
    DEFAULT_DATABASE_URL,
    DEFAULT_RRF_K,
    DEFAULT_TABLE,
    database_url_from_args,
    import_psycopg,
    retrieve_hits,
    validate_identifier,
)


DEFAULT_QA_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_QA_BASE_URL = "http://localhost:8003/v1"
SYSTEM_PROMPT = """You are a careful high-energy-physics research assistant.
Answer the question using only the supplied literature evidence.

Requirements:
- Cite factual statements with the paper source number, such as [1] or [2].
- Do not cite a source that does not support the statement.
- End every factual paragraph or bullet with at least one supporting citation.
- Distinguish established results from proposals, projections, and comparisons.
- Preserve important qualifications and uncertainty.
- Do not introduce mechanisms, numbers, or conclusions that are not explicit in the evidence.
- If the evidence is insufficient, say what cannot be established from it.
- Use abstracts for each paper's scope and the labeled passages for technical details.
- Keep the answer focused and under 300 words. Avoid repeating the same point.
- Write clear Markdown. Do not add a separate references section; one is appended automatically.
"""


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    paper_id: str
    chunk_index: int
    chunk_type: str
    title: str | None
    section_path: list[str]
    token_count: int
    text: str
    source_url: str | None


@dataclass(frozen=True)
class PaperEvidence:
    source_number: int
    paper_id: str
    title: str | None
    source_url: str | None
    abstract: EvidenceChunk | None
    passages: list[EvidenceChunk]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question = read_question(args)
    print("Retrieving and reranking evidence...", file=sys.stderr)
    hits = retrieve_hits(question, args)
    if not hits:
        raise SystemExit("No evidence was retrieved for this question.")

    evidence = load_paper_evidence(hits, args)
    context, included_evidence = build_context(evidence, args.max_context_tokens)
    print(
        f"Answering from {len(included_evidence)} papers and "
        f"{sum(len(paper.passages) for paper in included_evidence)} passages...",
        file=sys.stderr,
    )
    answer = generate_answer(question, context, args)

    if args.json:
        print(
            json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "sources": [asdict(paper) for paper in included_evidence],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_answer(answer, included_evidence))
    if args.show_context:
        print("\n# Packed Evidence\n", file=sys.stderr)
        print(context, file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to answer.")
    parser.add_argument("--question-file", type=Path)
    parser.add_argument("--database-url", help="Postgres URL. Defaults to $DATABASE_URL.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--retrieval", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--bm25-candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--max-chunks-per-paper", type=int, default=2)
    parser.add_argument("--neighbor-window", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=5500)
    parser.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hnsw-ef-search", type=int)
    parser.add_argument("--query-vector", type=Path)

    parser.add_argument("--backend", choices=["openai-compatible"], default="openai-compatible")
    parser.add_argument("--model")
    parser.add_argument("--base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--api-key")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt",
        default=(
            "Instruct: Given a physics literature search query, retrieve relevant "
            "passages that answer the query\nQuery: "
        ),
    )
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-base-url", default=DEFAULT_RERANK_BASE_URL)
    parser.add_argument("--rerank-endpoint", default="/rerank")
    parser.add_argument("--rerank-api-key")
    parser.add_argument("--rerank-timeout", type=float, default=120.0)
    parser.add_argument("--rerank-max-length", type=int)
    parser.add_argument("--rerank-instruction", default=DEFAULT_RERANK_INSTRUCTION)

    parser.add_argument("--qa-model", default=DEFAULT_QA_MODEL)
    parser.add_argument("--qa-base-url", default=DEFAULT_QA_BASE_URL)
    parser.add_argument("--qa-api-key")
    parser.add_argument("--qa-timeout", type=float, default=180.0)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    parser.set_defaults(query_file=None, quiet=True)
    return parser


def read_question(args: argparse.Namespace) -> str:
    if args.question_file:
        question = args.question_file.read_text(encoding="utf-8").strip()
    elif args.question:
        question = args.question.strip()
    elif sys.stdin.isatty():
        question = input("Question: ").strip()
    else:
        question = sys.stdin.read().strip()
    if not question:
        raise SystemExit("Provide a question as an argument, file, or standard input.")
    return question


def load_paper_evidence(
    hits: list[SearchHit],
    args: argparse.Namespace,
) -> list[PaperEvidence]:
    if args.neighbor_window < 0:
        raise SystemExit("--neighbor-window cannot be negative")
    psycopg = import_psycopg()
    try:
        from psycopg.rows import dict_row
    except ImportError as error:
        raise SystemExit("Missing dependency: psycopg. Install with `pip install -e '.[pgvector]'`.") from error

    table = validate_identifier(args.table)
    paper_ids = list(dict.fromkeys(hit.paper_id for hit in hits))
    chunk_ids = [hit.chunk_id for hit in hits]
    sql = f"""
        WITH seeds AS (
            SELECT paper_id, chunk_index
            FROM {table}
            WHERE chunk_id = ANY(%(chunk_ids)s)
        )
        SELECT DISTINCT
            chunks.chunk_id, chunks.paper_id, chunks.chunk_index, chunks.chunk_type,
            chunks.title, chunks.section_path, chunks.token_count, chunks.text,
            chunks.source_url
        FROM {table} AS chunks
        WHERE chunks.paper_id = ANY(%(paper_ids)s)
          AND (
              chunks.chunk_type = 'abstract'
              OR chunks.section_path @> '["Abstract"]'::jsonb
              OR EXISTS (
                  SELECT 1 FROM seeds
                  WHERE seeds.paper_id = chunks.paper_id
                    AND chunks.chunk_index BETWEEN
                        seeds.chunk_index - %(neighbor_window)s
                        AND seeds.chunk_index + %(neighbor_window)s
              )
          )
        ORDER BY chunks.paper_id, chunks.chunk_index
    """
    with psycopg.connect(database_url_from_args(args), row_factory=dict_row) as conn:
        rows = conn.execute(
            sql,
            {
                "paper_ids": paper_ids,
                "chunk_ids": chunk_ids,
                "neighbor_window": args.neighbor_window,
            },
        ).fetchall()

    chunks = [evidence_chunk(row) for row in rows]
    by_paper: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        by_paper.setdefault(chunk.paper_id, []).append(chunk)
    rank_by_chunk = {hit.chunk_id: hit.rank for hit in hits}

    papers = []
    for source_number, paper_id in enumerate(paper_ids, start=1):
        paper_chunks = by_paper.get(paper_id, [])
        abstract = next((chunk for chunk in paper_chunks if is_abstract_chunk(chunk)), None)
        passages = [chunk for chunk in paper_chunks if not is_abstract_chunk(chunk)]
        passages.sort(
            key=lambda chunk: (
                min(
                    (
                        abs(chunk.chunk_index - selected.chunk_index)
                        for selected in passages
                        if selected.chunk_id in rank_by_chunk
                    ),
                    default=0,
                ),
                rank_by_chunk.get(chunk.chunk_id, 10_000),
                chunk.chunk_index,
            )
        )
        representative = abstract or (passages[0] if passages else None)
        papers.append(
            PaperEvidence(
                source_number=source_number,
                paper_id=paper_id,
                title=(representative.title if representative else None),
                source_url=(representative.source_url if representative else None),
                abstract=abstract,
                passages=passages,
            )
        )
    return papers


def evidence_chunk(row: dict[str, Any]) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=str(row["chunk_id"]),
        paper_id=str(row["paper_id"]),
        chunk_index=int(row["chunk_index"]),
        chunk_type=str(row["chunk_type"]),
        title=str(row["title"]) if row.get("title") is not None else None,
        section_path=[str(part) for part in row.get("section_path", [])],
        token_count=int(row["token_count"]),
        text=str(row["text"]).strip(),
        source_url=str(row["source_url"]) if row.get("source_url") is not None else None,
    )


def is_abstract_chunk(chunk: EvidenceChunk) -> bool:
    return chunk.chunk_type == "abstract" or (
        bool(chunk.section_path) and chunk.section_path[0].strip().lower() == "abstract"
    )


def build_context(
    papers: list[PaperEvidence],
    max_tokens: int,
) -> tuple[str, list[PaperEvidence]]:
    if max_tokens < 500:
        raise SystemExit("--max-context-tokens must be at least 500")
    mandatory_blocks = []
    used_tokens = 0
    for paper in papers:
        header = render_paper_header(paper)
        abstract_text = paper.abstract.text if paper.abstract else "(Abstract unavailable in corpus.)"
        mandatory = f"{header}\nAbstract:\n{abstract_text}\n"
        mandatory_blocks.append(mandatory)
        used_tokens += estimate_tokens(mandatory)
    if used_tokens > max_tokens:
        raise SystemExit(
            "The selected-paper abstracts exceed --max-context-tokens. "
            "Reduce --top-k or increase the context budget."
        )

    passage_blocks: dict[str, list[str]] = {paper.paper_id: [] for paper in papers}
    included_passages: dict[str, list[EvidenceChunk]] = {
        paper.paper_id: [] for paper in papers
    }
    for paper in papers:
        for passage in paper.passages:
            section = " > ".join(passage.section_path) or "(section unavailable)"
            block = f"Passage, section {section}:\n{passage.text}\n"
            block_tokens = estimate_tokens(block)
            if used_tokens + block_tokens > max_tokens:
                continue
            passage_blocks[paper.paper_id].append(block)
            included_passages[paper.paper_id].append(passage)
            used_tokens += block_tokens

    rendered_papers = []
    included_papers = []
    for paper, mandatory in zip(papers, mandatory_blocks):
        rendered_papers.append(
            "\n".join([mandatory, *passage_blocks[paper.paper_id]]).strip()
        )
        included_papers.append(
            PaperEvidence(
                source_number=paper.source_number,
                paper_id=paper.paper_id,
                title=paper.title,
                source_url=paper.source_url,
                abstract=paper.abstract,
                passages=included_passages[paper.paper_id],
            )
        )
    return "\n\n".join(rendered_papers), included_papers


def render_paper_header(paper: PaperEvidence) -> str:
    return (
        f"[SOURCE {paper.source_number}]\n"
        f"Paper: {paper.title or '(title unavailable)'}\n"
        f"arXiv: {paper.paper_id}"
    )


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 3.5))


def generate_answer(question: str, context: str, args: argparse.Namespace) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nLiterature evidence:\n{context}",
        },
    ]
    answer = request_chat_completion(messages, args)
    allowed_sources = {
        int(value) for value in re.findall(r"\[SOURCE (\d+)\]", context)
    }
    if not has_valid_citations(answer, allowed_sources):
        messages.extend(
            [
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the draft with inline citations. End every factual "
                        "paragraph or bullet with one or more citations in [n] format, "
                        f"using only these source numbers: {sorted(allowed_sources)}. "
                        "Keep it under 300 words and remove unsupported claims."
                    ),
                },
            ]
        )
        answer = request_chat_completion(messages, args)
    return answer


def request_chat_completion(
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    payload = {
        "model": args.qa_model,
        "messages": messages,
        "max_tokens": args.max_answer_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 20,
        "presence_penalty": args.presence_penalty,
    }
    body = post_json(
        args.qa_base_url.rstrip("/") + "/chat/completions",
        payload,
        api_key=args.qa_api_key,
        timeout=args.qa_timeout,
    )
    return parse_chat_answer(body)


def has_valid_citations(answer: str, allowed_sources: set[int]) -> bool:
    citations = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    return bool(citations) and citations <= allowed_sources


def parse_chat_answer(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit("Unexpected QA response: missing choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise SystemExit("Unexpected QA response: missing answer text.")
    return content.strip()


def format_answer(answer: str, papers: list[PaperEvidence]) -> str:
    lines = ["# Answer", "", answer, "", "## Sources", ""]
    for paper in papers:
        url = f"https://arxiv.org/abs/{paper.paper_id}"
        sections = sorted(
            {
                " > ".join(passage.section_path)
                for passage in paper.passages
                if passage.section_path
            }
        )
        section_note = f" — {', '.join(sections)}" if sections else ""
        lines.append(
            f"[{paper.source_number}] {paper.title or paper.paper_id} "
            f"(arXiv:{paper.paper_id}){section_note}\n    {url}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
