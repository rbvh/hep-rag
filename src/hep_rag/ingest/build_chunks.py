"""Build first-pass paragraph chunks from downloaded Living Review sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hep_rag.chunking.chunker import chunk_paper
from hep_rag.ingest.schema import PaperRecord, ParseErrorRecord
from hep_rag.ingest.source_parser import parse_latex_source


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.inspect_dir is not None:
        args.inspect_dir.mkdir(parents=True, exist_ok=True)

    records = list(load_manifest(args.manifest))
    if args.arxiv_id:
        records = [record for record in records if record.get("arxiv_id") == args.arxiv_id]
    if args.limit is not None:
        records = records[: args.limit]

    papers_path = args.out_dir / "papers.jsonl"
    chunks_path = args.out_dir / "chunks.jsonl"
    errors_path = args.out_dir / "parse_errors.jsonl"
    parsed_count = 0
    chunk_count = 0
    error_count = 0
    inspect_count = 0

    with (
        papers_path.open("w", encoding="utf-8") as papers_file,
        chunks_path.open("w", encoding="utf-8") as chunks_file,
        errors_path.open("w", encoding="utf-8") as errors_file,
    ):
        for record in records:
            if not should_parse_record(record):
                continue

            try:
                paper = parse_latex_source(record)
                chunks = chunk_paper(
                    paper,
                    min_tokens=args.min_tokens,
                    target_tokens=args.target_tokens,
                    max_tokens=args.max_tokens,
                )
            except Exception as error:  # noqa: BLE001 - keep corpus jobs moving.
                error_count += 1
                errors_file.write(
                    json.dumps(
                        asdict(
                            ParseErrorRecord(
                                paper_id=optional_string(record.get("arxiv_id")),
                                bib_key=optional_string(record.get("bib_key")),
                                extracted_path=optional_string(record.get("extracted_path")),
                                status=optional_string(record.get("status")) or "unknown",
                                error=f"{type(error).__name__}: {error}",
                            )
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue

            parsed_count += 1
            chunk_count += len(chunks)
            papers_file.write(
                json.dumps(
                    asdict(
                        PaperRecord(
                            paper_id=paper.paper_id,
                            bib_key=paper.bib_key,
                            title=paper.title,
                            authors=paper.authors,
                            year=paper.year,
                            primary_class=paper.primary_class,
                            source_url=paper.source_url,
                            source_path=paper.source_path,
                            extracted_path=paper.extracted_path,
                            main_tex_path=paper.main_tex_path,
                            living_review_categories=paper.living_review_categories,
                            abstract=paper.abstract,
                            paragraph_count=len(paper.paragraphs),
                            chunk_count=len(chunks),
                        )
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            for chunk in chunks:
                chunks_file.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")

            if args.inspect_dir is not None and inspect_count < args.inspect_limit:
                write_inspection_markdown(args.inspect_dir, paper, chunks)
                inspect_count += 1

    print(f"Parsed papers: {parsed_count}")
    print(f"Wrote chunks: {chunk_count}")
    print(f"Parse errors: {error_count}")
    print(f"Papers: {papers_path}")
    print(f"Chunks: {chunks_path}")
    print(f"Errors: {errors_path}")
    if args.inspect_dir is not None:
        print(f"Inspection files: {args.inspect_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/living_review/manifest.jsonl"),
        help="Living Review source manifest.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for papers.jsonl, chunks.jsonl, and parse_errors.jsonl.",
    )
    parser.add_argument("--limit", type=int, help="Parse at most this many manifest rows.")
    parser.add_argument("--arxiv-id", help="Parse only one arXiv ID.")
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--target-tokens", type=int, default=350)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument(
        "--inspect-dir",
        type=Path,
        default=Path("data/processed/inspect"),
        help="Directory for human-readable per-paper chunk inspection markdown.",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=10,
        help="Maximum number of inspection files to write.",
    )
    return parser


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as manifest:
        return [json.loads(line) for line in manifest if line.strip()]


def should_parse_record(record: dict[str, object]) -> bool:
    if record.get("status") not in {"downloaded_and_extracted", "already_extracted"}:
        return False
    extracted_path = record.get("extracted_path")
    if not extracted_path:
        return False
    return Path(str(extracted_path)).exists()


def write_inspection_markdown(inspect_dir: Path, paper, chunks) -> None:
    path = inspect_dir / f"{paper.paper_id}.md"
    lines = [
        f"# {paper.paper_id}",
        "",
        f"Title: {paper.title or ''}",
        f"Bib key: {paper.bib_key}",
        f"Main TeX: `{paper.main_tex_path}`",
        f"Paragraphs: {len(paper.paragraphs)}",
        f"Chunks: {len(chunks)}",
        "",
        "Categories:",
    ]
    for category in paper.living_review_categories:
        lines.append(f"- {' > '.join(category)}")
    lines.extend(["", "## Chunks", ""])

    for chunk in chunks:
        lines.extend(
            [
                f"### {chunk.chunk_id}",
                "",
                f"Section: {' > '.join(chunk.section_path)}",
                f"Type: {chunk.chunk_type}",
                f"Tokens: {chunk.token_count}",
                f"Paragraph IDs: {', '.join(chunk.paragraph_ids)}",
                "",
                chunk.text,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
