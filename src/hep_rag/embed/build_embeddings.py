"""Build embedding matrices for chunk JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hep_rag.search.documents import retrieval_document_text

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_BATCH_SIZE = 1
DEFAULT_BASE_URL = "http://localhost:8001/v1"


@dataclass(frozen=True)
class EmbeddingRow:
    row_index: int
    chunk_id: str
    paper_id: str
    chunk_index: int
    chunk_type: str
    section_path: list[str]
    token_count: int
    text_sha1: str


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or default_out_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.chunks, args.limit)
    texts = [
        embedding_text(chunk, include_metadata_context=args.include_metadata_context)
        for chunk in chunks
    ]

    if args.dry_run:
        print(f"Would embed {len(chunks)} chunks")
        print(f"Backend: {args.backend}")
        print(f"Model: {args.model}")
        print(f"Base URL: {args.base_url}")
        print(f"Batch size: {args.batch_size}")
        print(f"Truncate prompt tokens: {args.truncate_prompt_tokens or '(none)'}")
        print(f"Prompt: {args.prompt or '(none)'}")
        print(f"Output directory: {out_dir}")
        if texts:
            print("First embedding text:")
            print(texts[0][:1000])
        return 0

    if not chunks:
        raise SystemExit(f"No chunks found in {args.chunks}")

    started_at = time.time()
    vectors = embed_with_openai_compatible(texts, args)
    if args.normalize:
        vectors = normalize_rows(vectors)

    rows = [
        embedding_row(index, chunk, text)
        for index, (chunk, text) in enumerate(zip(chunks, texts, strict=True))
    ]
    write_outputs(
        out_dir=out_dir,
        vectors=vectors,
        rows=rows,
        config={
            "backend": args.backend,
            "model": args.model,
            "base_url": args.base_url,
            "chunks_path": str(args.chunks),
            "count": len(rows),
            "dimension": int(vectors.shape[1]),
            "normalize": args.normalize,
            "include_metadata_context": args.include_metadata_context,
            "batch_size": args.batch_size,
            "truncate_prompt_tokens": args.truncate_prompt_tokens,
            "prompt": args.prompt,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
    )

    print(f"Embedded chunks: {len(rows)}")
    print(f"Embedding dimension: {vectors.shape[1]}")
    print(f"Output directory: {out_dir}")
    print(f"Embeddings: {out_dir / 'embeddings.npy'}")
    print(f"Rows: {out_dir / 'rows.jsonl'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Input chunks JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Defaults to data/embeddings/<model-name>.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Embedding model name or server-side model id.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai-compatible"],
        default="openai-compatible",
        help="Embedding backend. Only OpenAI-compatible HTTP/vLLM serving is supported.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding request batch size sent to the HTTP server.",
    )
    parser.add_argument("--limit", type=int, help="Embed at most this many chunks.")
    parser.add_argument(
        "--truncate-prompt-tokens",
        type=int,
        help="Ask the embedding server to truncate inputs to this many tokens.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings before writing them.",
    )
    parser.add_argument(
        "--include-metadata-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend title, section, and Living Review categories to each embedded text.",
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Optional prefix prepended to each input before sending it to the "
            "OpenAI-compatible embedding endpoint."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL for the embedding server.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY"),
        help="Bearer token for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and first embedded text without loading a model.",
    )
    return parser


def load_chunks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as chunks_file:
        for line in chunks_file:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


def embedding_text(chunk: dict[str, Any], include_metadata_context: bool = True) -> str:
    if not include_metadata_context:
        return str(chunk.get("text") or "").strip()
    return retrieval_document_text(chunk, style="dense")


def embed_with_openai_compatible(texts: list[str], args: argparse.Namespace):
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: numpy. Install with "
            "`pip install -e '.[embeddings]'` inside the conda env."
        ) from error

    embeddings: list[list[float]] = []
    url = args.base_url.rstrip("/") + "/embeddings"
    for start in range(0, len(texts), args.batch_size):
        batch = [
            embedding_input_text(text, args.prompt)
            for text in texts[start : start + args.batch_size]
        ]
        payload_body: dict[str, Any] = {"model": args.model, "input": batch}
        truncate_prompt_tokens = getattr(args, "truncate_prompt_tokens", None)
        if truncate_prompt_tokens is not None:
            payload_body["truncate_prompt_tokens"] = truncate_prompt_tokens
            payload_body["truncation_side"] = "right"
        payload = json.dumps(payload_body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers=openai_compatible_headers(args.api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310 - user URL.
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise SystemExit(f"Embedding request failed for {url}: {error}") from error

        batch_data = sorted(body["data"], key=lambda item: item.get("index", 0))
        embeddings.extend(item["embedding"] for item in batch_data)
        if not getattr(args, "quiet", False):
            print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)}")
    return np.asarray(embeddings, dtype="float32")


def embedding_input_text(text: str, prompt: str | None = None) -> str:
    return f"{prompt}{text}" if prompt else text


def openai_compatible_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def normalize_rows(vectors):
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: numpy. Install with "
            "`pip install -e '.[embeddings]'` inside the conda env."
        ) from error

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embedding_row(index: int, chunk: dict[str, Any], text: str) -> EmbeddingRow:
    return EmbeddingRow(
        row_index=index,
        chunk_id=str(chunk["chunk_id"]),
        paper_id=str(chunk["paper_id"]),
        chunk_index=int(chunk["chunk_index"]),
        chunk_type=str(chunk["chunk_type"]),
        section_path=[str(part) for part in chunk.get("section_path", [])],
        token_count=int(chunk["token_count"]),
        text_sha1=hashlib.sha1(text.encode("utf-8")).hexdigest(),
    )


def write_outputs(out_dir: Path, vectors, rows: list[EmbeddingRow], config: dict[str, Any]) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: numpy. Install with "
            "`pip install -e '.[embeddings]'` inside the conda env."
        ) from error

    np.save(out_dir / "embeddings.npy", vectors.astype("float32", copy=False))
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as rows_file:
        for row in rows:
            rows_file.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_out_dir(model: str) -> Path:
    return Path("data/embeddings") / safe_name(model)


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return name or "embeddings"


if __name__ == "__main__":
    raise SystemExit(main())
