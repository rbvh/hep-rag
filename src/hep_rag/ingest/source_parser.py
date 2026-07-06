"""First-pass parser for arXiv LaTeX sources."""

from __future__ import annotations

import re
from pathlib import Path

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
INPUT_RE = re.compile(r"\\(input|include)\{([^}]+)\}")
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\{document\}")
END_DOCUMENT_RE = re.compile(r"\\end\{document\}")
ABSTRACT_ENV_RE = re.compile(
    r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL
)
ABSTRACT_CMD_RE = re.compile(
    r"\\abstract\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)
ABSTRACT_HEADING_RE = re.compile(
    r"(?:\{\\bf\s+Abstract\}|\\noindent\s*\{\\bf\s+Abstract\}|\\textbf\{Abstract\})"
    r"(.*?)(?=\\clearpage|\\newpage|\\tableofcontents|\\section\*?\{)",
    re.DOTALL,
)
TITLE_RE = re.compile(r"\\title(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
ENV_BLOCK_RE = re.compile(
    r"\\begin\{(figure|figure\*|table|table\*)\}.*?\\end\{\1\}",
    re.DOTALL,
)
LIST_ENV_RE = re.compile(
    r"\\begin\{(itemize|enumerate)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)
DISPLAY_MATH_RE = re.compile(
    r"\\begin\{("
    r"equation|equation\*|align|align\*|eqnarray|eqnarray\*|"
    r"gather|gather\*|multline|multline\*|split"
    r")\}"
    r".*?\\end\{\1\}",
    re.DOTALL,
)
BRACKET_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
DOLLAR_MATH_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
LABEL_RE = re.compile(r"\\label\s*\{[^}]*\}")
REFERENCE_RE = re.compile(
    r"\\(?:ref|eqref|autoref|cref|Cref|pageref)\*?(?:\[[^\]]*\])?\s*\{[^}]*\}"
)
ITEM_MARKER = "@@ITEM@@"
LIST_BEGIN_MARKER = "@@LIST_BEGIN@@"
LIST_END_MARKER = "@@LIST_END@@"
FLOAT_MARKER = "@@FLOAT@@"
FLOAT_MACROS = {
    "singlefigure": {"arg_count": 4, "caption_args": (2,)},
    "doublefigure": {"arg_count": 8, "caption_args": (2, 5, 6)},
}


def parse_latex_source(record: dict[str, object]) -> ParsedPaper:
    """Parse one manifest row into a paragraph-level paper representation."""

    extracted_path = Path(str(record["extracted_path"]))
    main_tex_path = find_main_tex_file(extracted_path)
    raw_tex = main_tex_path.read_text(encoding="utf-8", errors="replace")
    expanded_tex = expand_inputs(raw_tex, main_tex_path.parent)
    tex = strip_comments(expanded_tex)

    title = clean_latex_text(extract_first(TITLE_RE, tex)) or clean_latex_text(
        value_or_none(record, "title")
    )
    abstract = extract_abstract(tex)
    body = document_body(tex)
    body = drop_front_matter(body)
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
        score = score_main_tex_candidate(path, text)
        scored.append((score, path.stat().st_size, path))

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
    text: str, root_dir: Path, visited: set[Path] | None = None, depth: int = 0
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


def extract_abstract(tex: str) -> str | None:
    raw = (
        extract_first(ABSTRACT_ENV_RE, tex)
        or extract_first(ABSTRACT_CMD_RE, tex)
        or extract_first(ABSTRACT_HEADING_RE, tex)
    )
    return clean_latex_text(raw) if raw else None


def extract_first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


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
    if title in {"acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"}:
        return True
    return bool(re.fullmatch(r"the .+ collaboration", title))


def extract_captions(text: str) -> list[str]:
    captions: list[str] = []
    cursor = 0
    while True:
        start = text.find("\\caption", cursor)
        if start == -1:
            break
        brace_start = text.find("{", start)
        if brace_start == -1:
            break
        brace_end = find_matching_brace(text, brace_start)
        if brace_end is None:
            cursor = brace_start + 1
            continue
        cleaned = clean_latex_text(text[brace_start + 1 : brace_end])
        if cleaned:
            captions.append(cleaned)
        cursor = brace_end + 1
    captions.extend(extract_float_macro_captions(text))
    return captions


def extract_float_macro_captions(text: str) -> list[str]:
    captions: list[str] = []
    for macro_name, spec in FLOAT_MACROS.items():
        cursor = 0
        while True:
            start = find_macro_call(text, macro_name, cursor)
            if start == -1:
                break
            parsed = parse_macro_arguments(text, start, int(spec["arg_count"]))
            if parsed is None:
                cursor = start + len(macro_name) + 1
                continue
            end, args = parsed
            for arg_index in spec["caption_args"]:
                cleaned = clean_latex_text(args[arg_index])
                if cleaned:
                    captions.append(cleaned)
            cursor = end
    return captions


def find_matching_brace(text: str, start: int) -> int | None:
    depth = 0
    escaped = False
    for cursor in range(start, len(text)):
        char = text[cursor]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cursor
    return None


def text_for_paragraph_splitting(text: str) -> str:
    text = ENV_BLOCK_RE.sub(FLOAT_MARKER, text)
    text = strip_float_macros(text)
    text = re.sub(r"\s*" + re.escape(FLOAT_MARKER) + r"\s*", "\n", text)
    text = LIST_ENV_RE.sub(normalize_list_environment, text)
    text = re.sub(r"\n\s*\n\s*" + re.escape(LIST_BEGIN_MARKER), f"\n{LIST_BEGIN_MARKER}", text)
    text = re.sub(re.escape(LIST_END_MARKER) + r"\s*\n\s*\n", f"{LIST_END_MARKER}\n", text)
    text = re.sub(r"\\maketitle\b", "\n\n", text)
    text = re.sub(r"\\begin\{(itemize|enumerate)\}", "\n", text)
    text = re.sub(r"\\end\{(itemize|enumerate)\}", "\n", text)
    text = re.sub(r"\\item\b(?:\[[^\]]*\])?", f"\n{ITEM_MARKER} ", text)
    return text


def strip_float_macros(text: str) -> str:
    for macro_name, spec in FLOAT_MACROS.items():
        text = replace_macro_calls_with_marker(text, macro_name, int(spec["arg_count"]))
    return text


def replace_macro_calls_with_marker(text: str, macro_name: str, arg_count: int) -> str:
    pieces: list[str] = []
    cursor = 0
    while True:
        start = find_macro_call(text, macro_name, cursor)
        if start == -1:
            pieces.append(text[cursor:])
            break
        parsed = parse_macro_arguments(text, start, arg_count)
        if parsed is None:
            pieces.append(text[cursor : start + len(macro_name) + 1])
            cursor = start + len(macro_name) + 1
            continue
        end, _args = parsed
        pieces.append(text[cursor:start])
        pieces.append(FLOAT_MARKER)
        cursor = end
    return "".join(pieces)


def find_macro_call(text: str, macro_name: str, start: int = 0) -> int:
    pattern = "\\" + macro_name
    cursor = start
    while True:
        cursor = text.find(pattern, cursor)
        if cursor == -1:
            return -1
        before = text[cursor - 1] if cursor > 0 else ""
        after_index = cursor + len(pattern)
        after = text[after_index] if after_index < len(text) else ""
        if before != "\\" and not after.isalpha():
            return cursor
        cursor = after_index


def parse_macro_arguments(
    text: str, macro_start: int, arg_count: int
) -> tuple[int, list[str]] | None:
    cursor = macro_start
    while cursor < len(text) and text[cursor] != "\\":
        cursor += 1
    while cursor < len(text) and (text[cursor].isalpha() or text[cursor] == "\\"):
        cursor += 1
    cursor = skip_latex_space(text, cursor)
    if cursor < len(text) and text[cursor] == "[":
        optional_end = find_matching_square_bracket(text, cursor)
        if optional_end is None:
            return None
        cursor = skip_latex_space(text, optional_end + 1)

    args: list[str] = []
    for _index in range(arg_count):
        cursor = skip_latex_space(text, cursor)
        if cursor >= len(text) or text[cursor] != "{":
            return None
        end = find_matching_brace(text, cursor)
        if end is None:
            return None
        args.append(text[cursor + 1 : end])
        cursor = end + 1
    return cursor, args


def skip_latex_space(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def find_matching_square_bracket(text: str, start: int) -> int | None:
    depth = 0
    escaped = False
    for cursor in range(start, len(text)):
        char = text[cursor]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return cursor
    return None


def normalize_list_environment(match: re.Match[str]) -> str:
    content = match.group(2)
    raw_items = re.split(r"\\item\b(?:\[[^\]]*\])?", content)
    items = [item.strip() for item in raw_items[1:] if item.strip()]
    if not items:
        return "\n\n"
    list_text = "\n".join(f"{ITEM_MARKER} {item}" for item in items)
    return f"\n{LIST_BEGIN_MARKER}\n{list_text}\n{LIST_END_MARKER}\n"


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for raw in split_paragraph_blocks(text):
        paragraph = clean_latex_text(raw)
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def split_paragraph_blocks(text: str) -> list[str]:
    """Split prose paragraphs without cutting through display math blocks."""

    blocks: list[str] = []
    current: list[str] = []
    math_depth = 0
    in_bracket_math = False
    in_dollar_math = False

    for line in text.splitlines():
        stripped = line.strip()
        can_split = not math_depth and not in_bracket_math and not in_dollar_math
        if not stripped and can_split:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        current.append(line)
        math_depth += display_math_begin_count(line)
        math_depth = max(0, math_depth - display_math_end_count(line))

        if r"\[" in line:
            in_bracket_math = True
        if r"\]" in line:
            in_bracket_math = False

        dollar_count = line.count("$$")
        if dollar_count % 2:
            in_dollar_math = not in_dollar_math

    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def display_math_begin_count(text: str) -> int:
    return len(
        re.findall(
            r"\\begin\{(?:equation\*?|align\*?|eqnarray\*?|gather\*?|multline\*?|split)\}",
            text,
        )
    )


def display_math_end_count(text: str) -> int:
    return len(
        re.findall(
            r"\\end\{(?:equation\*?|align\*?|eqnarray\*?|gather\*?|multline\*?|split)\}",
            text,
        )
    )


def useful_paragraph(paragraph: str) -> bool:
    if len(paragraph.split()) < 8:
        return False
    return True


def clean_latex_text(text: str | None) -> str | None:
    if text is None:
        return None

    text, protected_math = protect_display_math(text)
    text = text.replace("~", " ")
    text = LABEL_RE.sub("", text)
    text = REFERENCE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(LIST_BEGIN_MARKER, "")
    text = text.replace(LIST_END_MARKER, "\n")
    text = text.replace(f"{ITEM_MARKER} ", "\n- ")
    text = text.replace(ITEM_MARKER, "\n- ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    for placeholder, math_text in protected_math.items():
        text = text.replace(placeholder, f"\n{math_text}\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def protect_display_math(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    matches: list[tuple[int, int, str]] = []

    for pattern in (DISPLAY_MATH_RE, BRACKET_MATH_RE, DOLLAR_MATH_RE):
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), match.group(0)))

    matches.sort(key=lambda item: item[0])
    pieces: list[str] = []
    cursor = 0
    for start, end, raw_math in matches:
        if start < cursor:
            continue
        placeholder = f"@@DISPLAY_MATH_{len(protected)}@@"
        pieces.append(text[cursor:start])
        pieces.append(placeholder)
        protected[placeholder] = clean_display_math(raw_math)
        cursor = end

    pieces.append(text[cursor:])
    return "".join(pieces), protected


def clean_display_math(text: str) -> str:
    text = LABEL_RE.sub("", text)
    text = REFERENCE_RE.sub("", text)
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def value_or_none(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    return str(value)
