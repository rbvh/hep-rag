"""Small BibTeX parser for corpus metadata ingestion.

This is intentionally conservative and dependency-free. It handles the parts of
BibTeX needed by the HEPML Living Review bibliography: entry types, keys, and
field values delimited by braces, quotes, or bare tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]
    raw: str


class BibTeXParseError(ValueError):
    """Raised when a BibTeX entry cannot be parsed."""


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse BibTeX text into entries.

    The parser skips comments and preamble text outside ``@...`` entries.
    """

    entries: list[BibEntry] = []
    index = 0
    while True:
        at_index = text.find("@", index)
        if at_index == -1:
            break
        entry, index = _parse_entry(text, at_index)
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_entry(text: str, at_index: int) -> tuple[BibEntry | None, int]:
    cursor = at_index + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1

    type_start = cursor
    while cursor < len(text) and (text[cursor].isalnum() or text[cursor] in "_-"):
        cursor += 1
    entry_type = text[type_start:cursor].strip().lower()
    if not entry_type:
        return None, at_index + 1

    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] not in "{(":
        return None, cursor

    open_char = text[cursor]
    close_char = "}" if open_char == "{" else ")"
    content_start = cursor + 1
    content_end = _find_matching_delimiter(text, cursor, open_char, close_char)
    raw = text[at_index : content_end + 1]
    content = text[content_start:content_end]

    key, fields_text = _split_key_and_fields(content)
    if entry_type in {"comment", "preamble", "string"}:
        return None, content_end + 1

    fields = _parse_fields(fields_text)
    return BibEntry(entry_type=entry_type, key=key, fields=fields, raw=raw), content_end + 1


def _find_matching_delimiter(
    text: str, start: int, open_char: str, close_char: str
) -> int:
    depth = 0
    in_quote = False
    escaped = False

    for cursor in range(start, len(text)):
        char = text[cursor]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return cursor

    raise BibTeXParseError(f"Unclosed BibTeX entry starting at offset {start}")


def _split_key_and_fields(content: str) -> tuple[str, str]:
    depth = 0
    in_quote = False
    escaped = False

    for cursor, char in enumerate(content):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            return content[:cursor].strip(), content[cursor + 1 :]

    return content.strip(), ""


def _parse_fields(fields_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    cursor = 0
    while cursor < len(fields_text):
        cursor = _skip_space_and_commas(fields_text, cursor)
        if cursor >= len(fields_text):
            break

        name_start = cursor
        while cursor < len(fields_text) and (
            fields_text[cursor].isalnum() or fields_text[cursor] in "_-"
        ):
            cursor += 1
        name = fields_text[name_start:cursor].strip().lower()
        if not name:
            cursor += 1
            continue

        cursor = _skip_whitespace(fields_text, cursor)
        if cursor >= len(fields_text) or fields_text[cursor] != "=":
            raise BibTeXParseError(f"Expected '=' after field {name!r}")
        cursor += 1
        cursor = _skip_whitespace(fields_text, cursor)

        value, cursor = _parse_value(fields_text, cursor)
        fields[name] = _clean_value(value)

    return fields


def _skip_space_and_commas(text: str, cursor: int) -> int:
    while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
        cursor += 1
    return cursor


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _parse_value(text: str, cursor: int) -> tuple[str, int]:
    if cursor >= len(text):
        return "", cursor

    if text[cursor] == "{":
        end = _find_matching_delimiter(text, cursor, "{", "}")
        return text[cursor + 1 : end], end + 1

    if text[cursor] == '"':
        return _parse_quoted_value(text, cursor)

    start = cursor
    while cursor < len(text) and text[cursor] != ",":
        cursor += 1
    return text[start:cursor].strip(), cursor


def _parse_quoted_value(text: str, cursor: int) -> tuple[str, int]:
    start = cursor + 1
    cursor += 1
    escaped = False
    while cursor < len(text):
        char = text[cursor]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return text[start:cursor], cursor + 1
        cursor += 1

    raise BibTeXParseError(f"Unclosed quoted BibTeX value at offset {start - 1}")


def _clean_value(value: str) -> str:
    return " ".join(value.strip().split())
