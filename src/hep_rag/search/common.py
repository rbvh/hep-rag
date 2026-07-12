"""Shared search helpers for Postgres retrieval and reranking."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hep_rag.embed.build_embeddings import embed_with_openai_compatible, normalize_rows
from hep_rag.search.config import EmbeddingConfig, RerankConfig


@dataclass(frozen=True)
class SearchHit:
    rank: int
    final_score: float
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


def embed_queries(queries: list[str], config: EmbeddingConfig):
    from argparse import Namespace

    request = Namespace(
        model=config.model,
        batch_size=config.batch_size,
        prompt=config.prompt,
        base_url=config.base_url,
        api_key=config.api_key,
        quiet=True,
        truncate_prompt_tokens=None,
    )
    vectors = embed_with_openai_compatible(queries, request)
    return normalize_rows(vectors) if config.normalize else vectors


def rerank_hits(
    query: str,
    hits: list[SearchHit],
    config: RerankConfig,
) -> list[SearchHit]:
    if not hits:
        return []
    if not query:
        raise SystemExit("Reranking requires query text, not only --query-vector.")

    return rerank_hits_with_vllm(query, hits, config)


def rerank_hits_with_vllm(
    query: str,
    hits: list[SearchHit],
    config: RerankConfig,
) -> list[SearchHit]:
    payload: dict[str, Any] = {
        "model": config.model,
        "query": query,
        "documents": [hit.text for hit in hits],
        "top_n": len(hits),
    }
    if config.instruction:
        payload["instruction"] = config.instruction
    if config.max_length:
        payload["truncate_prompt_tokens"] = config.max_length

    body = post_json(
        rerank_url(config),
        payload,
        api_key=config.api_key,
        timeout=config.timeout,
    )
    scored_hits = []
    for index, score in parse_rerank_scores(body, expected_count=len(hits)).items():
        hit = hits[index]
        rerank_score = float(score)
        scored_hits.append(
            replace(
                hit,
                final_score=rerank_score,
                rerank_score=rerank_score,
            )
        )
    scored_hits.sort(key=lambda hit: hit.rerank_score or float("-inf"), reverse=True)
    return [replace(hit, rank=index + 1) for index, hit in enumerate(scored_hits)]


def rerank_url(config: RerankConfig) -> str:
    base_url = config.base_url.rstrip("/")
    endpoint = config.endpoint or "/rerank"
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
        raise SystemExit("Unexpected vLLM rerank response: expected a `results` or `data` list.")

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
