"""Shared search helpers for Postgres retrieval and reranking."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hep_rag.embed.build_embeddings import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    default_out_dir,
    embed_with_openai_compatible,
    normalize_rows,
)


DEFAULT_EMBEDDING_DIR = default_out_dir(DEFAULT_MODEL)
DEFAULT_EMBEDDING_BASE_URL = DEFAULT_BASE_URL
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_INSTRUCTION = (
    "Given a physics literature search query, retrieve relevant passages that answer the query"
)
DEFAULT_RERANK_BASE_URL = "http://localhost:8002"


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    chunk_id: str
    paper_id: str
    title: str | None
    section_path: list[str]
    living_review_categories: list[list[str]]
    chunk_type: str
    token_count: int
    text: str
    source_url: str | None
    vector_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


def read_query(args: argparse.Namespace) -> str:
    if getattr(args, "query_vector", None):
        return args.query.strip() if args.query else ""
    if args.query_file:
        return args.query_file.read_text(encoding="utf-8").strip()
    if args.query:
        return args.query.strip()
    raise SystemExit("Provide query text as an argument or with --query-file.")


def embed_query(query: str, args: argparse.Namespace, embedding_config: dict[str, Any]):
    if args.query_vector:
        try:
            import numpy as np
        except ImportError as error:
            raise SystemExit(
                "Missing dependency: numpy. Install with `pip install -e '.[pgvector]'`."
            ) from error
        vector = np.load(args.query_vector).astype("float32", copy=False)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        return vector

    embed_args = argparse.Namespace(
        backend=args.backend,
        model=args.model or embedding_config.get("model") or DEFAULT_MODEL,
        batch_size=args.batch_size,
        normalize=args.normalize,
        prompt=args.prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        quiet=getattr(args, "quiet", False),
    )
    vector = embed_with_openai_compatible([query], embed_args)
    return normalize_rows(vector) if args.normalize else vector


def rerank_hits(query: str, hits: list[SearchHit], args: argparse.Namespace) -> list[SearchHit]:
    if not hits:
        return []
    if not query:
        raise SystemExit("Reranking requires query text, not only --query-vector.")

    return rerank_hits_with_vllm(query, hits, args)


def rerank_hits_with_vllm(
    query: str,
    hits: list[SearchHit],
    args: argparse.Namespace,
) -> list[SearchHit]:
    payload: dict[str, Any] = {
        "model": args.rerank_model,
        "query": query,
        "documents": [hit.text for hit in hits],
        "top_n": len(hits),
    }
    if getattr(args, "rerank_instruction", None):
        payload["instruction"] = args.rerank_instruction
    if getattr(args, "rerank_max_length", None):
        payload["truncate_prompt_tokens"] = args.rerank_max_length

    body = post_json(
        rerank_url(args),
        payload,
        api_key=getattr(args, "rerank_api_key", None),
        timeout=float(getattr(args, "rerank_timeout", 120.0)),
    )
    scored_hits = []
    for index, score in parse_rerank_scores(body, expected_count=len(hits)).items():
        hit = hits[index]
        rerank_score = float(score)
        scored_hits.append(
            replace(
                hit,
                score=rerank_score,
                vector_score=hit.vector_score if hit.vector_score is not None else hit.score,
                rerank_score=rerank_score,
            )
        )
    scored_hits.sort(key=lambda hit: hit.rerank_score or float("-inf"), reverse=True)
    return [replace(hit, rank=index + 1) for index, hit in enumerate(scored_hits)]


def rerank_url(args: argparse.Namespace) -> str:
    base_url = getattr(args, "rerank_base_url", DEFAULT_RERANK_BASE_URL).rstrip("/")
    endpoint = getattr(args, "rerank_endpoint", "/rerank") or "/rerank"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return base_url + endpoint


def post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"vLLM request failed for {url}: {details}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"vLLM request failed for {url}: {error}") from error


def parse_rerank_scores(body: dict[str, Any], expected_count: int) -> dict[int, float]:
    rows = body.get("results") or body.get("data")
    if not isinstance(rows, list):
        raise SystemExit(
            "Unexpected vLLM rerank response: expected a `results` or `data` list."
        )

    scores: dict[int, float] = {}
    for default_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit("Unexpected vLLM rerank response: result rows must be objects.")
        raw_index = row.get("index", row.get("corpus_id", default_index))
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"Invalid rerank result index: {raw_index!r}") from error
        if index < 0 or index >= expected_count:
            raise SystemExit(f"Rerank result index out of range: {index}")
        raw_score = (
            row.get("relevance_score")
            if row.get("relevance_score") is not None
            else row.get("score")
        )
        if raw_score is None:
            raise SystemExit("Unexpected vLLM rerank response: missing relevance score.")
        scores[index] = float(raw_score)

    if len(scores) != expected_count:
        missing = sorted(set(range(expected_count)) - set(scores))
        raise SystemExit(f"vLLM rerank response omitted scores for document indexes: {missing}")
    return scores


def diversify_hits(
    hits: list[SearchHit],
    top_k: int,
    max_chunks_per_paper: int | None = None,
) -> list[SearchHit]:
    """Select top hits while optionally limiting repeated chunks per paper."""

    if max_chunks_per_paper is not None and max_chunks_per_paper < 1:
        raise SystemExit("--max-chunks-per-paper must be at least 1")

    selected: list[SearchHit] = []
    paper_counts: dict[str, int] = {}
    for hit in hits:
        if len(selected) >= top_k:
            break
        if max_chunks_per_paper is not None:
            count = paper_counts.get(hit.paper_id, 0)
            if count >= max_chunks_per_paper:
                continue
            paper_counts[hit.paper_id] = count + 1
        selected.append(hit)
    return [replace(hit, rank=index + 1) for index, hit in enumerate(selected)]


def format_hits(hits: list[SearchHit], max_text_chars: int = 900) -> str:
    blocks = []
    for hit in hits:
        section = " > ".join(hit.section_path) if hit.section_path else "(no section)"
        categories = "; ".join(" > ".join(category) for category in hit.living_review_categories)
        text = hit.text.strip()
        if len(text) > max_text_chars:
            text = text[: max_text_chars - 1].rstrip() + "..."
        lines = [
            f"## {hit.rank}. {hit.chunk_id}",
            f"Score: {hit.score:.4f}",
        ]
        if hit.rerank_score is not None:
            lines.append(f"Rerank score: {hit.rerank_score:.4f}")
        if hit.vector_score is not None:
            lines.append(f"Vector score: {hit.vector_score:.4f}")
        if hit.lexical_score is not None:
            lines.append(f"Lexical score: {hit.lexical_score:.4f}")
        if hit.rrf_score is not None:
            lines.append(f"RRF score: {hit.rrf_score:.4f}")
        lines.extend(
            [
                f"Paper: {hit.paper_id}",
                f"Title: {hit.title or '(unknown)'}",
                f"Section: {section}",
            ]
        )
        if categories:
            lines.append(f"Categories: {categories}")
        if hit.source_url:
            lines.append(f"Source: {hit.source_url}")
        lines.extend(["", text])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
