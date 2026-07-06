"""Shared data records for ingestion and chunking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedParagraph:
    paragraph_id: str
    section_path: list[str]
    text: str
    kind: str = "paragraph"


@dataclass(frozen=True)
class ParsedPaper:
    paper_id: str
    bib_key: str
    title: str | None
    authors: str | None
    year: str | None
    primary_class: str | None
    source_url: str | None
    source_path: str | None
    extracted_path: str
    main_tex_path: str
    living_review_categories: list[list[str]]
    abstract: str | None
    paragraphs: list[ParsedParagraph] = field(default_factory=list)


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    bib_key: str
    title: str | None
    authors: str | None
    year: str | None
    primary_class: str | None
    source_url: str | None
    source_path: str | None
    extracted_path: str
    main_tex_path: str
    living_review_categories: list[list[str]]
    abstract: str | None
    paragraph_count: int
    chunk_count: int


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    paper_id: str
    bib_key: str
    title: str | None
    authors: str | None
    year: str | None
    primary_class: str | None
    source_url: str | None
    main_tex_path: str
    living_review_categories: list[list[str]]
    section_path: list[str]
    chunk_index: int
    chunk_type: str
    text: str
    token_count: int
    paragraph_ids: list[str]


@dataclass(frozen=True)
class ParseErrorRecord:
    paper_id: str | None
    bib_key: str | None
    extracted_path: str | None
    status: str
    error: str
