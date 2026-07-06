"""Parse HEPML Living Review topic assignments from ``HEPML.tex``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


CITE_RE = re.compile(r"\\cite\{([^}]*)\}")
TEXTIT_RE = re.compile(r"\\textit\{(.+)\}")


@dataclass(frozen=True)
class LivingReviewCategoryAssignment:
    """A Living Review category assigned to a bibliography key."""

    path: tuple[str, ...]


@dataclass(frozen=True)
class LivingReviewCategoryIndex:
    """Category metadata extracted from the Living Review topic tree."""

    assignments_by_key: dict[str, list[LivingReviewCategoryAssignment]]
    descriptions_by_path: dict[tuple[str, ...], str] = field(default_factory=dict)

    def paths_for_key(self, bib_key: str) -> list[list[str]]:
        """Return category paths for a BibTeX key as JSON-friendly lists."""

        return [
            list(assignment.path)
            for assignment in self.assignments_by_key.get(bib_key, [])
        ]


def parse_living_review_categories(tex: str) -> LivingReviewCategoryIndex:
    """Parse bibliography-key category assignments from ``HEPML.tex``.

    The Living Review encodes its taxonomy as nested ``itemize`` lists. Items
    with citations assign every cited BibTeX key to the current item path.
    """

    assignments_by_key: dict[str, list[LivingReviewCategoryAssignment]] = {}
    descriptions_by_path: dict[tuple[str, ...], str] = {}
    stack: list[str] = []
    itemize_level = 0
    current_path: tuple[str, ...] | None = None

    for raw_line in tex.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        if "\\begin{itemize}" in line:
            itemize_level += line.count("\\begin{itemize}")

        if "\\item" in line:
            title = extract_item_title(line)
            if title:
                level_index = max(itemize_level - 1, 0)
                stack = stack[:level_index]
                stack.append(title)
                current_path = tuple(stack)

                cite_keys = extract_cite_keys(line)
                for cite_key in cite_keys:
                    add_assignment(
                        assignments_by_key,
                        cite_key,
                        LivingReviewCategoryAssignment(path=current_path),
                    )

        elif current_path and "\\textit{" in line:
            description = extract_textit(line)
            if description:
                descriptions_by_path[current_path] = description

        if "\\end{itemize}" in line:
            itemize_level -= line.count("\\end{itemize}")
            itemize_level = max(itemize_level, 0)
            stack = stack[:itemize_level]
            current_path = tuple(stack) if stack else None

    return LivingReviewCategoryIndex(
        assignments_by_key=assignments_by_key,
        descriptions_by_path=descriptions_by_path,
    )


def add_assignment(
    assignments_by_key: dict[str, list[LivingReviewCategoryAssignment]],
    cite_key: str,
    assignment: LivingReviewCategoryAssignment,
) -> None:
    existing = assignments_by_key.setdefault(cite_key, [])
    if assignment not in existing:
        existing.append(assignment)


def extract_cite_keys(line: str) -> list[str]:
    """Extract BibTeX keys from all citation commands in a line."""

    keys: list[str] = []
    for match in CITE_RE.finditer(line):
        keys.extend(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def extract_item_title(line: str) -> str | None:
    """Extract and lightly normalize the category title from an item line."""

    if "\\item" not in line:
        return None

    text = line.split("\\item", 1)[1].strip()
    text = text.split("~\\cite", 1)[0]
    text = text.split("\\cite", 1)[0]
    text = text.split("\\\\", 1)[0]
    text = strip_latex_command_wrapper(text, "\\textbf")
    text = text.strip()
    text = text.removesuffix(".").strip()
    return clean_latex_text(text) or None


def extract_textit(line: str) -> str | None:
    """Extract the first ``\\textit{...}`` block from a line."""

    match = TEXTIT_RE.search(line)
    if not match:
        return None
    return clean_latex_text(match.group(1))


def strip_latex_command_wrapper(text: str, command: str) -> str:
    prefix = f"{command}{{"
    if not text.startswith(prefix):
        return text

    start = len(prefix)
    depth = 1
    for cursor in range(start, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:cursor] + text[cursor + 1 :]
    return text


def clean_latex_text(text: str) -> str:
    """Convert small bits of LaTeX markup into plain metadata text."""

    replacements = {
        r"\&": "&",
        r"\rightarrow": "->",
        r"\bar": "",
        r"\textit": "",
        r"\textbf": "",
        "`": "'",
        "''": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
