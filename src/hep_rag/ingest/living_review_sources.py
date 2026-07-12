"""Download arXiv LaTeX sources for the HEPML Living Review corpus."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from hep_rag.ingest.bibtex import BibEntry, parse_bibtex
from hep_rag.ingest.living_review_categories import (
    LivingReviewCategoryIndex,
    parse_living_review_categories,
)

LIVING_REVIEW_BIB_URL = (
    "https://raw.githubusercontent.com/iml-wg/HEPML-LivingReview/master/HEPML.bib"
)
LIVING_REVIEW_TEX_URL = (
    "https://raw.githubusercontent.com/iml-wg/HEPML-LivingReview/master/HEPML.tex"
)
DEFAULT_USER_AGENT = "hep-rag-lab/0.1 (mailto:replace-with-your-email@example.com)"
ARXIV_SOURCE_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


@dataclass(frozen=True)
class SourceRecord:
    bib_key: str
    entry_type: str
    title: str | None
    authors: str | None
    year: str | None
    arxiv_id: str | None
    primary_class: str | None
    source_url: str | None
    status: str
    living_review_categories: list[list[str]]
    source_path: str | None = None
    extracted_path: str | None = None
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bib_text = load_bibliography(args.bib_path, args.bib_url, args.user_agent)
    bib_output_path = out_dir / "HEPML.bib"
    bib_output_path.write_text(bib_text, encoding="utf-8")

    tex_text = load_living_review_tex(args.tex_path, args.tex_url, args.user_agent)
    tex_output_path = out_dir / "HEPML.tex"
    tex_output_path.write_text(tex_text, encoding="utf-8")
    category_index = parse_living_review_categories(tex_text)
    write_category_index(category_index, out_dir / "living_review_categories.json")

    entries = parse_bibtex(bib_text)
    arxiv_entries = [entry for entry in entries if arxiv_id_from_entry(entry)]

    print(f"Parsed {len(entries)} BibTeX entries")
    print(f"Found {len(arxiv_entries)} entries with arXiv identifiers")
    print(f"Parsed categories for {len(category_index.assignments_by_key)} BibTeX keys")

    selected = arxiv_entries[: args.limit] if args.limit is not None else arxiv_entries
    if args.dry_run:
        for entry in selected[: args.preview]:
            arxiv_id = arxiv_id_from_entry(entry)
            category_count = len(category_index.paths_for_key(entry.key))
            print(
                f"{entry.key}: {arxiv_id} - {entry.fields.get('title', '')} "
                f"[{category_count} categories]"
            )
        if len(selected) > args.preview:
            print(f"... {len(selected) - args.preview} more selected entries")
        return 0

    manifest_path = out_dir / "manifest.jsonl"
    records: list[SourceRecord] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, entry in enumerate(selected, start=1):
            record = download_entry_source(
                entry,
                out_dir=out_dir,
                user_agent=args.user_agent,
                category_index=category_index,
                force=args.force,
            )
            records.append(record)
            manifest.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            manifest.flush()
            print(f"[{index}/{len(selected)}] {record.arxiv_id}: {record.status}")

            if index < len(selected) and args.sleep > 0:
                time.sleep(args.sleep)

        if args.record_skipped:
            selected_keys = {entry.key for entry in selected}
            for entry in entries:
                if entry.key in selected_keys or arxiv_id_from_entry(entry):
                    continue
                record = source_record_for_entry(
                    entry,
                    status="skipped_no_arxiv_id",
                    source_url=None,
                    category_index=category_index,
                )
                manifest.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    write_summary(records, out_dir / "summary.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bib-url",
        default=LIVING_REVIEW_BIB_URL,
        help="URL for the Living Review BibTeX file.",
    )
    parser.add_argument(
        "--bib-path",
        type=Path,
        help="Use a local BibTeX file instead of downloading the Living Review bibliography.",
    )
    parser.add_argument(
        "--tex-url",
        default=LIVING_REVIEW_TEX_URL,
        help="URL for the Living Review LaTeX file containing category assignments.",
    )
    parser.add_argument(
        "--tex-path",
        type=Path,
        help="Use a local HEPML.tex file instead of downloading the Living Review LaTeX.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/living_review"),
        help="Output directory for bibliography, manifest, and source files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Download at most this many arXiv source entries.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between arXiv source requests.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent. Replace the default email before large crawls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract sources that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print selected entries without downloading sources.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of entries to show in dry-run mode.",
    )
    parser.add_argument(
        "--record-skipped",
        action="store_true",
        help="Append non-arXiv bibliography entries to the manifest as skipped.",
    )
    return parser


def load_bibliography(bib_path: Path | None, bib_url: str, user_agent: str) -> str:
    if bib_path is not None:
        return bib_path.read_text(encoding="utf-8")

    request = urllib.request.Request(bib_url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def load_living_review_tex(tex_path: Path | None, tex_url: str, user_agent: str) -> str:
    if tex_path is not None:
        return tex_path.read_text(encoding="utf-8")

    request = urllib.request.Request(tex_url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def arxiv_id_from_entry(entry: BibEntry) -> str | None:
    archive = entry.fields.get("archiveprefix", "")
    eprint = entry.fields.get("eprint", "").strip()
    if not eprint:
        return None
    if archive and archive.lower() != "arxiv":
        return None
    if not ARXIV_ID_RE.match(eprint):
        return None
    return eprint


def download_entry_source(
    entry: BibEntry,
    out_dir: Path,
    user_agent: str,
    category_index: LivingReviewCategoryIndex,
    force: bool = False,
) -> SourceRecord:
    arxiv_id = arxiv_id_from_entry(entry)
    if arxiv_id is None:
        return source_record_for_entry(
            entry,
            status="skipped_no_arxiv_id",
            source_url=None,
            category_index=category_index,
        )

    source_url = ARXIV_SOURCE_URL.format(arxiv_id=arxiv_id)
    paper_dir = out_dir / "sources" / safe_id(arxiv_id)
    extracted_dir = paper_dir / "extracted"
    marker_path = paper_dir / "source.tar.gz"

    if not force and extracted_dir.exists():
        return source_record_for_entry(
            entry,
            status="already_extracted",
            source_url=source_url,
            category_index=category_index,
            source_path=str(existing_source_path(paper_dir) or marker_path),
            extracted_path=str(extracted_dir),
        )

    if force and paper_dir.exists():
        shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = download_source_archive(source_url, paper_dir, user_agent)
        status, extracted_path = unpack_source(downloaded_path, extracted_dir)
        return source_record_for_entry(
            entry,
            status=status,
            source_url=source_url,
            category_index=category_index,
            source_path=str(downloaded_path),
            extracted_path=str(extracted_path) if extracted_path else None,
        )
    except urllib.error.HTTPError as error:
        return source_record_for_entry(
            entry,
            status="download_failed",
            source_url=source_url,
            category_index=category_index,
            error=f"HTTP {error.code}: {error.reason}",
        )
    except Exception as error:  # noqa: BLE001 - manifest should preserve per-paper failures.
        return source_record_for_entry(
            entry,
            status="failed",
            source_url=source_url,
            category_index=category_index,
            error=f"{type(error).__name__}: {error}",
        )


def download_source_archive(source_url: str, paper_dir: Path, user_agent: str) -> Path:
    request = urllib.request.Request(source_url, headers={"User-Agent": user_agent})
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        NamedTemporaryFile("wb", dir=paper_dir, delete=False) as temp_file,
    ):
        shutil.copyfileobj(response, temp_file)
        temp_path = Path(temp_file.name)

    kind = detect_source_kind(temp_path)
    target_path = paper_dir / filename_for_kind(kind)
    temp_path.replace(target_path)
    return target_path


def detect_source_kind(path: Path) -> str:
    prefix = path.read_bytes()[:8]
    if prefix.startswith(b"%PDF"):
        return "pdf"
    if zipfile.is_zipfile(path):
        return "zip"
    if tarfile.is_tarfile(path):
        return "tar"
    if prefix.startswith(b"\x1f\x8b"):
        return "gzip"
    return "tex"


def filename_for_kind(kind: str) -> str:
    return {
        "pdf": "source.pdf",
        "zip": "source.zip",
        "tar": "source.tar.gz",
        "gzip": "source.tex.gz",
        "tex": "source.tex",
    }[kind]


def unpack_source(source_path: Path, extracted_dir: Path) -> tuple[str, Path | None]:
    kind = detect_source_kind(source_path)
    if kind == "pdf":
        return "downloaded_pdf_only", None

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True)

    if kind == "zip":
        with zipfile.ZipFile(source_path) as archive:
            safe_extract_zip(archive, extracted_dir)
        return "downloaded_and_extracted", extracted_dir

    if kind == "tar":
        with tarfile.open(source_path) as archive:
            safe_extract_tar(archive, extracted_dir)
        return "downloaded_and_extracted", extracted_dir

    if kind == "gzip":
        with gzip.open(source_path, "rb") as compressed:
            (extracted_dir / "source.tex").write_bytes(compressed.read())
        return "downloaded_and_extracted", extracted_dir

    shutil.copy2(source_path, extracted_dir / "source.tex")
    return "downloaded_and_extracted", extracted_dir


def safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    archive.extractall(destination)


def safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.namelist():
        target = (destination / member).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe zip member path: {member}")
    archive.extractall(destination)


def source_record_for_entry(
    entry: BibEntry,
    status: str,
    source_url: str | None,
    category_index: LivingReviewCategoryIndex,
    source_path: str | None = None,
    extracted_path: str | None = None,
    error: str | None = None,
) -> SourceRecord:
    return SourceRecord(
        bib_key=entry.key,
        entry_type=entry.entry_type,
        title=entry.fields.get("title"),
        authors=entry.fields.get("author"),
        year=entry.fields.get("year"),
        arxiv_id=arxiv_id_from_entry(entry),
        primary_class=entry.fields.get("primaryclass"),
        source_url=source_url,
        status=status,
        living_review_categories=category_index.paths_for_key(entry.key),
        source_path=source_path,
        extracted_path=extracted_path,
        error=error,
    )


def existing_source_path(paper_dir: Path) -> Path | None:
    for filename in (
        "source.tar.gz",
        "source.tex.gz",
        "source.zip",
        "source.tex",
        "source.pdf",
    ):
        path = paper_dir / filename
        if path.exists():
            return path
    return None


def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def write_summary(records: list[SourceRecord], path: Path) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    path.write_text(
        json.dumps(
            {"total": len(records), "status_counts": counts},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_category_index(category_index: LivingReviewCategoryIndex, path: Path) -> None:
    assignments_by_key = {
        bib_key: [list(assignment.path) for assignment in assignments]
        for bib_key, assignments in sorted(category_index.assignments_by_key.items())
    }
    descriptions = [
        {"path": list(path_key), "description": description}
        for path_key, description in sorted(category_index.descriptions_by_path.items())
    ]
    path.write_text(
        json.dumps(
            {
                "assignments_by_key": assignments_by_key,
                "descriptions": descriptions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
