"""Turn a preprocessed LaTeX paper into section-aware paragraphs."""

from __future__ import annotations

import re
from pathlib import Path

from hep_rag.ingest.latex_source import expand_inputs, find_main_tex_file, strip_comments
from hep_rag.ingest.latex_text import (
    clean_latex_text,
    extract_captions,
    split_paragraphs,
    text_for_paragraph_splitting,
    useful_paragraph,
)
from hep_rag.ingest.schema import ParsedPaper, ParsedParagraph

SECTION_LEVELS = {
    "section": 0,
    "subsection": 1,
    "subsubsection": 2,
    "paragraph": 3,
}
SECTION_RE = re.compile(
    r"\\(section|subsection|subsubsection|paragraph)\*?"
    r"(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    re.DOTALL,
)
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\{document\}")
END_DOCUMENT_RE = re.compile(r"\\end\{document\}")
ABSTRACT_ENV_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)
ABSTRACT_CMD_RE = re.compile(r"\\abstract\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
ABSTRACT_HEADING_RE = re.compile(
    r"(?:\{\\bf\s+Abstract\}|\\noindent\s*\{\\bf\s+Abstract\}|\\textbf\{Abstract\})"
    r"(.*?)(?=\\clearpage|\\newpage|\\tableofcontents|\\section\*?\{)",
    re.DOTALL,
)
TITLE_RE = re.compile(
    r"\\title(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    re.DOTALL,
)


def parse_latex_source(record: dict[str, object]) -> ParsedPaper:
    """Parse one manifest row into a paragraph-level paper representation."""

    extracted_path = Path(str(record["extracted_path"]))
    main_tex_path = find_main_tex_file(extracted_path)
    raw_tex = main_tex_path.read_text(encoding="utf-8", errors="replace")
    tex = strip_comments(expand_inputs(raw_tex, main_tex_path.parent))

    title = clean_latex_text(extract_first(TITLE_RE, tex)) or clean_latex_text(
        value_or_none(record, "title")
    )
    abstract = extract_abstract(tex)
    body = drop_front_matter(document_body(tex))
    paragraphs = parse_body_paragraphs(body, paper_id=str(record["arxiv_id"]))

    if abstract:
        paragraphs.insert(
            0,
            ParsedParagraph(
                paragraph_id=f"{record['arxiv_id']}:abstract:p000",
                section_path=["Abstract"],
                text=abstract,
                kind="abstract",
            ),
        )

    return ParsedPaper(
        paper_id=str(record["arxiv_id"]),
        bib_key=str(record["bib_key"]),
        title=title,
        authors=value_or_none(record, "authors"),
        year=value_or_none(record, "year"),
        primary_class=value_or_none(record, "primary_class"),
        source_url=value_or_none(record, "source_url"),
        source_path=value_or_none(record, "source_path"),
        extracted_path=str(extracted_path),
        main_tex_path=str(main_tex_path),
        living_review_categories=list(record.get("living_review_categories", [])),
        abstract=abstract,
        paragraphs=paragraphs,
    )


def extract_abstract(tex: str) -> str | None:
    raw = (
        extract_first(ABSTRACT_ENV_RE, tex)
        or extract_first(ABSTRACT_CMD_RE, tex)
        or extract_first(ABSTRACT_HEADING_RE, tex)
    )
    return clean_latex_text(raw) if raw else None


def extract_first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def document_body(tex: str) -> str:
    begin_match = BEGIN_DOCUMENT_RE.search(tex)
    if begin_match:
        tex = tex[begin_match.end() :]
    end_match = END_DOCUMENT_RE.search(tex)
    if end_match:
        tex = tex[: end_match.start()]
    return tex


def drop_front_matter(text: str) -> str:
    first_section = SECTION_RE.search(text)
    if first_section:
        text = text[first_section.start() :]
    for marker in ("\\bibliography", "\\begin{thebibliography}", "\\appendix"):
        marker_index = text.find(marker)
        if marker_index != -1:
            text = text[:marker_index]
    return text


def parse_body_paragraphs(body: str, paper_id: str) -> list[ParsedParagraph]:
    section_path: list[str] = ["Body"]
    paragraphs: list[ParsedParagraph] = []
    cursor = 0
    paragraph_index = 0
    caption_index = 0

    for match in SECTION_RE.finditer(body):
        pre_section_text = body[cursor : match.start()]
        paragraph_index = append_section_paragraphs(
            paragraphs,
            pre_section_text,
            paper_id,
            section_path,
            paragraph_index,
        )
        caption_index = append_caption_paragraphs(
            paragraphs,
            pre_section_text,
            paper_id,
            section_path,
            caption_index,
        )
        command = match.group(1)
        title = clean_latex_text(match.group(2))
        level = SECTION_LEVELS[command]
        section_path = section_path[:level]
        section_path.append(title or command.title())
        cursor = match.end()

    final_text = body[cursor:]
    append_section_paragraphs(
        paragraphs,
        final_text,
        paper_id,
        section_path,
        paragraph_index,
    )
    append_caption_paragraphs(
        paragraphs,
        final_text,
        paper_id,
        section_path,
        caption_index,
    )
    return paragraphs


def append_section_paragraphs(
    paragraphs: list[ParsedParagraph],
    text: str,
    paper_id: str,
    section_path: list[str],
    start_index: int,
) -> int:
    if non_retrieval_section(section_path):
        return start_index

    cleaned = text_for_paragraph_splitting(text)
    index = start_index
    for paragraph in split_paragraphs(cleaned):
        if not useful_paragraph(paragraph):
            continue
        paragraphs.append(
            ParsedParagraph(
                paragraph_id=f"{paper_id}:p{index:05d}",
                section_path=list(section_path),
                text=paragraph,
            )
        )
        index += 1
    return index


def append_caption_paragraphs(
    captions: list[ParsedParagraph],
    text: str,
    paper_id: str,
    section_path: list[str],
    start_index: int,
) -> int:
    if non_retrieval_section(section_path):
        return start_index

    index = start_index
    for caption in extract_captions(text):
        if not caption:
            continue
        captions.append(
            ParsedParagraph(
                paragraph_id=f"{paper_id}:caption:p{index:05d}",
                section_path=list(section_path),
                text=caption,
                kind="caption",
            )
        )
        index += 1
    return index


def non_retrieval_section(section_path: list[str]) -> bool:
    if not section_path:
        return False
    title = re.sub(r"\s+", " ", section_path[-1].strip().lower())
    if title in {
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
    }:
        return True
    return bool(re.fullmatch(r"the .+ collaboration", title))


def value_or_none(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    return None if value is None else str(value)
