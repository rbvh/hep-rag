"""Paragraph-first chunking for parsed papers."""

from __future__ import annotations

import hashlib
import re

from hep_rag.ingest.schema import ParsedPaper, ParsedParagraph, TextChunk


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
DISPLAY_MATH_RE = re.compile(
    r"\\begin\{(?:equation\*?|align\*?|eqnarray\*?|gather\*?|multline\*?|split)\}"
    r"|\\\[|\$\$"
)


def chunk_paper(
    paper: ParsedPaper,
    min_tokens: int = 120,
    target_tokens: int = 350,
    max_tokens: int = 600,
) -> list[TextChunk]:
    """Create one section-aware chunk per parsed paragraph or caption."""

    chunks: list[TextChunk] = []
    chunk_index = 0

    for paragraph in paper.paragraphs:
        paragraph_tokens = estimate_tokens(paragraph.text)
        if paragraph_tokens > max_tokens and not contains_display_math(paragraph.text):
            for split_text in split_long_text(paragraph.text, max_tokens):
                chunks.append(
                    make_chunk(
                        paper,
                        paragraph,
                        [split_text],
                        [paragraph.paragraph_id],
                        chunk_index,
                    )
                )
                chunk_index += 1
            continue

        chunks.append(
            make_chunk(
                paper,
                paragraph,
                [paragraph.text],
                [paragraph.paragraph_id],
                chunk_index,
            )
        )
        chunk_index += 1

    return chunks


def contains_display_math(text: str) -> bool:
    return bool(DISPLAY_MATH_RE.search(text))


def split_long_text(text: str, max_tokens: int) -> list[str]:
    sentences = SENTENCE_BOUNDARY_RE.split(text)
    chunks: list[str] = []
    pending: list[str] = []
    pending_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if pending and pending_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(pending).strip())
            pending = []
            pending_tokens = 0
        pending.append(sentence)
        pending_tokens += sentence_tokens

    if pending:
        chunks.append(" ".join(pending).strip())
    return chunks


def make_chunk(
    paper: ParsedPaper,
    paragraph: ParsedParagraph,
    texts: list[str],
    paragraph_ids: list[str],
    chunk_index: int,
) -> TextChunk:
    text = "\n\n".join(texts)
    chunk_id = stable_chunk_id(paper.paper_id, paragraph.section_path, chunk_index, text)
    return TextChunk(
        chunk_id=chunk_id,
        paper_id=paper.paper_id,
        bib_key=paper.bib_key,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        primary_class=paper.primary_class,
        source_url=paper.source_url,
        main_tex_path=paper.main_tex_path,
        living_review_categories=paper.living_review_categories,
        section_path=paragraph.section_path,
        chunk_index=chunk_index,
        chunk_type=paragraph.kind,
        text=text,
        token_count=estimate_tokens(text),
        paragraph_ids=paragraph_ids,
    )


def stable_chunk_id(
    paper_id: str, section_path: list[str], chunk_index: int, text: str
) -> str:
    payload = "\n".join([paper_id, "/".join(section_path), str(chunk_index), text[:500]])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{paper_id}:chunk:{chunk_index:05d}:{digest}"


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))
