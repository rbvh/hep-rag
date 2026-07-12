"""Run the paper-level retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hep_rag.eval.metrics import aggregate_metrics, score_query
from hep_rag.eval.models import PaperCandidate, QueryMetrics
from hep_rag.eval.report import render_markdown
from hep_rag.search.common import SearchHit, embed_queries, load_jsonl, rerank_hits
from hep_rag.search.config import (
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_TABLE,
    EmbeddingConfig,
    RerankConfig,
    RetrievalConfig,
    database_url,
)
from hep_rag.search.retrieval import (
    configure_vector_search,
    import_psycopg,
    retrieval_uses_vector,
    search_chunks,
    validate_identifier,
    vector_to_pgvector,
)

DEFAULT_EVAL_SET = Path("data/eval/retrieval_queries.jsonl")
DEFAULT_PAPERS = Path("data/processed/papers.jsonl")
DEFAULT_REPORT_DIR = Path("data/eval/reports")
DEFAULT_KS = (1, 3, 5, 10, 20, 40)
DEFAULT_TOP_PAPERS = 40
DEFAULT_RERANK_PAPERS = 100


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_rows = load_eval_rows(args.eval_set, args.limit)
    paper_metadata = load_paper_metadata(DEFAULT_PAPERS)
    retrieval = RetrievalConfig(mode=args.retrieval, candidate_k=args.candidate_k)
    embedding = (
        EmbeddingConfig.from_artifacts(args.embedding_dir)
        if retrieval_uses_vector(retrieval.mode)
        else EmbeddingConfig()
    )
    rerank = RerankConfig.from_environment() if args.rerank else None
    top_papers = max(max(DEFAULT_KS), DEFAULT_TOP_PAPERS)

    query_vectors = (
        embed_queries([str(row["query"]) for row in eval_rows], embedding)
        if retrieval_uses_vector(retrieval.mode)
        else None
    )
    psycopg = import_psycopg()
    try:
        from psycopg.rows import dict_row
    except ImportError as error:
        raise SystemExit("Install Postgres support with `pip install -e '.[pgvector]'`.") from error

    table = validate_identifier(DEFAULT_TABLE)
    query_metrics: list[QueryMetrics] = []
    with psycopg.connect(database_url(args.database_url), row_factory=dict_row) as conn:
        configure_vector_search(conn, retrieval)
        for index, eval_row in enumerate(eval_rows):
            vector = vector_to_pgvector(query_vectors[index]) if query_vectors is not None else None
            hits = search_chunks(conn, table, str(eval_row["query"]), vector, retrieval)
            if rerank is not None:
                hits = unique_paper_hits(hits, max(top_papers, DEFAULT_RERANK_PAPERS))
                hits = rerank_hits(str(eval_row["query"]), hits, rerank)
            candidates = collapse_hits_by_paper(hits, top_papers)
            metrics = score_query(eval_row, candidates, list(DEFAULT_KS))
            query_metrics.append(metrics)
            print(
                f"{metrics.query_id}: MAP={metrics.average_precision:.3f} "
                f"MRR={metrics.mrr:.3f} "
                f"targets={len(metrics.target_ranks)}/{metrics.target_count}"
            )

    aggregate = aggregate_metrics(query_metrics, list(DEFAULT_KS))
    payload = {
        "config": {
            "eval_set": str(args.eval_set),
            "embedding_dir": str(args.embedding_dir),
            "candidate_k": args.candidate_k,
            "retrieval": args.retrieval,
            "k": DEFAULT_KS,
            "exact": True,
            "rerank": args.rerank,
            "rerank_model": rerank.model if rerank else None,
        },
        "aggregate": asdict(aggregate),
        "queries": [asdict(metric) for metric in query_metrics],
    }
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEFAULT_REPORT_DIR / "retrieval_metrics.json"
    markdown_path = DEFAULT_REPORT_DIR / "retrieval_metrics.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_markdown(aggregate, query_metrics, paper_metadata, list(DEFAULT_KS)),
        encoding="utf-8",
    )
    print(f"Wrote JSON metrics: {json_path}")
    print(f"Wrote Markdown metrics: {markdown_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--database-url", help="Defaults to $DATABASE_URL.")
    parser.add_argument(
        "--retrieval",
        choices=["vector", "lexical", "hybrid"],
        default="vector",
    )
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def load_eval_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    for index, row in enumerate(rows, start=1):
        if not row.get("query_id"):
            raise ValueError(f"Missing query_id in {path}:{index}")
        if not row.get("query"):
            raise ValueError(f"Missing query in {path}:{index}")
    return rows


def load_paper_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["paper_id"]): row for row in load_jsonl(path)}


def unique_paper_hits(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    if limit < 1:
        raise ValueError("paper limit must be at least 1")
    unique = []
    seen = set()
    for hit in hits:
        if hit.paper_id in seen:
            continue
        seen.add(hit.paper_id)
        unique.append(hit)
        if len(unique) == limit:
            break
    return unique


def collapse_hits_by_paper(hits: list[SearchHit], limit: int) -> list[PaperCandidate]:
    return [
        PaperCandidate(
            paper_rank=index,
            hit_rank=hit.rank,
            final_score=hit.final_score,
            paper_id=hit.paper_id,
            title=hit.title,
            living_review_categories=hit.living_review_categories,
            chunk_id=hit.chunk_id,
            section_path=hit.section_path,
            text=hit.text,
            source_url=hit.source_url,
        )
        for index, hit in enumerate(unique_paper_hits(hits, limit), start=1)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
