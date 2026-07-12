"""Canonical text representations for retrieval documents."""

from __future__ import annotations

from typing import Any, Literal


def retrieval_document_text(
    chunk: dict[str, Any],
    style: Literal["dense", "lexical"] = "dense",
) -> str:
    """Render the same metadata fields for dense or lexical retrieval."""

    title = str(chunk.get("title") or "").strip()
    section = " > ".join(str(part) for part in chunk.get("section_path") or [])
    categories = [
        " > ".join(str(part) for part in category)
        for category in chunk.get("living_review_categories") or []
    ]
    text = str(chunk.get("text") or "").strip()

    if style == "dense":
        metadata = []
        if title:
            metadata.append(f"Title: {title}")
        if section:
            metadata.append(f"Section: {section}")
        if categories:
            metadata.append(f"Living Review categories: {'; '.join(categories)}")
        return "\n".join([*metadata, "", text]).strip() if metadata else text

    parts = [title, section, *categories, text]
    return "\n".join(part for part in parts if part)
