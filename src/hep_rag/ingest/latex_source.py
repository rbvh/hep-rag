"""Locate and preprocess files in an arXiv LaTeX source tree."""

from __future__ import annotations

import re
from pathlib import Path

INPUT_RE = re.compile(r"\\(input|include)\{([^}]+)\}")
SECTION_RE = re.compile(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{")


def find_main_tex_file(extracted_path: Path) -> Path:
    tex_files = sorted(extracted_path.rglob("*.tex"))
    if not tex_files:
        raise FileNotFoundError(f"No .tex files found under {extracted_path}")

    scored: list[tuple[int, int, Path]] = []
    for path in tex_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scored.append((score_main_tex_candidate(path, text), path.stat().st_size, path))

    if not scored:
        raise FileNotFoundError(f"No readable .tex files found under {extracted_path}")
    return max(scored)[2]


def score_main_tex_candidate(path: Path, text: str) -> int:
    name = path.name.lower()
    score = 0
    if "\\documentclass" in text:
        score += 100
    if "\\begin{document}" in text:
        score += 100
    if "\\title" in text:
        score += 20
    if "\\begin{abstract}" in text or "\\abstract{" in text:
        score += 20
    score += 5 * len(SECTION_RE.findall(text))
    if name in {"main.tex", "paper.tex", "ms.tex", "article.tex"}:
        score += 20
    if any(part in name for part in ("author", "package", "definition", "macro")):
        score -= 30
    return score


def expand_inputs(
    text: str,
    root_dir: Path,
    visited: set[Path] | None = None,
    depth: int = 0,
) -> str:
    """Inline local TeX files referenced with ``input`` or ``include``."""

    if visited is None:
        visited = set()
    if depth > 10:
        return text

    def replace(match: re.Match[str]) -> str:
        raw_name = match.group(2).strip()
        if raw_name.lower().endswith((".bib", ".bbl", ".sty", ".cls")):
            return ""
        input_path = (root_dir / raw_name).with_suffix(".tex")
        if Path(raw_name).suffix:
            input_path = root_dir / raw_name
        input_path = input_path.resolve()
        if input_path in visited or not input_path.exists():
            return ""
        visited.add(input_path)
        child_text = input_path.read_text(encoding="utf-8", errors="replace")
        return expand_inputs(child_text, input_path.parent, visited, depth + 1)

    return INPUT_RE.sub(replace, text)


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped_line = strip_comment_from_line(line)
        if not stripped_line.strip() and line.lstrip().startswith("%"):
            continue
        lines.append(stripped_line)
    return "\n".join(lines)


def strip_comment_from_line(line: str) -> str:
    escaped = False
    for index, char in enumerate(line):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "%" and not escaped:
            return line[:index]
        escaped = False
    return line
