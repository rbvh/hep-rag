"""Shared search helpers for Postgres retrieval and reranking."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hep_rag.embed.build_embeddings import (
    DEFAULT_MODEL,
    default_out_dir,
    embed_with_openai_compatible,
    embed_with_sentence_transformers,
    normalize_rows,
    sentence_transformer_model_kwargs,
)


DEFAULT_EMBEDDING_DIR = default_out_dir(DEFAULT_MODEL)
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_INSTRUCTION = (
    "Given a physics literature search query, retrieve relevant passages that answer the query"
)


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

    if not args.embed_in_process:
        return embed_query_subprocess(query, args, embedding_config)

    embed_args = argparse.Namespace(
        backend=args.backend,
        model=args.model or embedding_config.get("model") or DEFAULT_MODEL,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        normalize=args.normalize,
        prompt=args.prompt,
        prompt_name=args.prompt_name,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    text = args.prompt + query if args.backend == "openai-compatible" and args.prompt else query
    if args.backend == "sentence-transformers":
        return embed_with_sentence_transformers([text], embed_args)
    if args.backend == "openai-compatible":
        vector = embed_with_openai_compatible([text], embed_args)
        return normalize_rows(vector) if args.normalize else vector
    raise ValueError(f"Unknown backend: {args.backend}")


def embed_query_subprocess(
    query: str, args: argparse.Namespace, embedding_config: dict[str, Any]
):
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: numpy. Install with `pip install -e '.[pgvector]'`."
        ) from error

    model = args.model or embedding_config.get("model") or DEFAULT_MODEL
    with tempfile.TemporaryDirectory(prefix="hep-rag-query-") as tmp_dir:
        out_path = Path(tmp_dir) / "query.npy"
        command = [
            sys.executable,
            "-m",
            "hep_rag.embed.embed_query",
            query,
            "--out",
            str(out_path),
            "--backend",
            args.backend,
            "--model",
            model,
            "--batch-size",
            str(args.batch_size),
            "--max-seq-length",
            str(args.max_seq_length),
            "--torch-dtype",
            args.torch_dtype,
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.prompt_name:
            command.extend(["--prompt-name", args.prompt_name])
        if args.prompt:
            command.extend(["--prompt", args.prompt])
        if not args.trust_remote_code:
            command.append("--no-trust-remote-code")
        if not args.normalize:
            command.append("--no-normalize")
        if args.backend == "openai-compatible":
            command.extend(["--base-url", args.base_url])
            if args.api_key:
                command.extend(["--api-key", args.api_key])

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            details = "\n".join(
                part for part in (error.stdout.strip(), error.stderr.strip()) if part
            )
            message = f"Query embedding subprocess failed: {error}"
            if details:
                message += f"\n{details}"
            raise SystemExit(message) from error
        return np.load(out_path).astype("float32", copy=False)


def rerank_hits(query: str, hits: list[SearchHit], args: argparse.Namespace) -> list[SearchHit]:
    if not hits:
        return []
    if not query:
        raise SystemExit("Reranking requires query text, not only --query-vector.")

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install with "
            "`pip install -e '.[embeddings,pgvector]'`."
        ) from error

    model_kwargs = sentence_transformer_model_kwargs(args.rerank_torch_dtype)
    model = CrossEncoder(
        args.rerank_model,
        device=args.rerank_device or args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs or None,
        max_length=args.rerank_max_length,
        prompts={"physics": args.rerank_instruction},
        default_prompt_name="physics",
    )
    pairs = [(query, hit.text) for hit in hits]
    scores = model.predict(
        pairs,
        batch_size=args.rerank_batch_size,
        show_progress_bar=bool(args.rerank_progress),
    )

    reranked = []
    for hit, score in zip(hits, scores):
        rerank_score = float(score)
        reranked.append(
            replace(
                hit,
                score=rerank_score,
                vector_score=hit.vector_score if hit.vector_score is not None else hit.score,
                rerank_score=rerank_score,
            )
        )
    reranked.sort(key=lambda hit: hit.rerank_score or float("-inf"), reverse=True)
    return [replace(hit, rank=index + 1) for index, hit in enumerate(reranked)]


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
