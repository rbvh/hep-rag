"""Embed a retrieval query with the same model used for chunk embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hep_rag.embed.build_embeddings import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL,
    embed_with_openai_compatible,
    embed_with_sentence_transformers,
    normalize_rows,
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query = query_text(args)
    texts = [query]

    if args.dry_run:
        print(f"Would embed query with backend: {args.backend}")
        print(f"Model: {args.model}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_seq_length or '(model default)'}")
        print(f"Torch dtype: {args.torch_dtype}")
        print(f"Prompt name: {args.prompt_name or '(none)'}")
        print(f"Prompt: {args.prompt or '(none)'}")
        print("Query text:")
        print(query)
        return 0

    if args.backend == "sentence-transformers":
        vector = embed_with_sentence_transformers(texts, args)
    elif args.backend == "openai-compatible":
        vector = embed_with_openai_compatible(texts, args)
        if args.normalize:
            vector = normalize_rows(vector)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if args.out:
        write_vector(args.out, vector)
        print(f"Wrote query embedding: {args.out}")
        return 0

    payload = {
        "model": args.model,
        "backend": args.backend,
        "prompt": args.prompt,
        "prompt_name": args.prompt_name,
        "query": query,
        "dimension": int(vector.shape[1]),
        "embedding": vector[0].astype(float).tolist(),
    }
    print(json.dumps(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Query text. Omit when using --query-file.")
    parser.add_argument("--query-file", type=Path, help="Read query text from a file.")
    parser.add_argument("--out", type=Path, help="Write the query vector as a .npy file.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "openai-compatible"],
        default="sentence-transformers",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="SentenceTransformer max sequence length. Use 0 to keep the model default.",
    )
    parser.add_argument("--device", help="Torch device for sentence-transformers, e.g. mps/cpu/cuda.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help=(
            "Torch dtype passed to the underlying Transformers model for the "
            "sentence-transformers backend. `auto` leaves the library default."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prompt-name",
        default="query",
        help=(
            "SentenceTransformer prompt name. Qwen recommends query prompting for "
            "queries and no prompt for retrieval documents."
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Explicit prompt string. For OpenAI-compatible/vLLM backends, this is "
            "prepended to the query text because prompt_name is SentenceTransformer-specific."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL for the openai-compatible backend.",
    )
    parser.add_argument(
        "--api-key",
        help="Bearer token for the OpenAI-compatible backend.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def query_text(args: argparse.Namespace) -> str:
    if args.query_file:
        text = args.query_file.read_text(encoding="utf-8").strip()
    elif args.query:
        text = args.query.strip()
    else:
        raise SystemExit("Provide query text as an argument or with --query-file.")

    if args.backend == "openai-compatible" and args.prompt:
        return args.prompt + text
    return text


def write_vector(path: Path, vector: Any) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: numpy. Install with "
            "`pip install -e '.[embeddings]'` inside the conda env."
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector.astype("float32", copy=False))


if __name__ == "__main__":
    raise SystemExit(main())
