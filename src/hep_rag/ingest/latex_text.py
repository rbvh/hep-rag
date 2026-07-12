"""Extract and normalize retrieval text from LaTeX fragments."""

from __future__ import annotations

import re

ENV_BLOCK_RE = re.compile(r"\\begin\{(figure|figure\*|table|table\*)\}.*?\\end\{\1\}", re.DOTALL)
LIST_ENV_RE = re.compile(r"\\begin\{(itemize|enumerate)\}(.*?)\\end\{\1\}", re.DOTALL)
DISPLAY_MATH_RE = re.compile(
    r"\\begin\{(equation|equation\*|align|align\*|eqnarray|eqnarray\*|"
    r"gather|gather\*|multline|multline\*|split)\}.*?\\end\{\1\}",
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
    text = re.sub(
        r"\n\s*\n\s*" + re.escape(LIST_BEGIN_MARKER),
        f"\n{LIST_BEGIN_MARKER}",
        text,
    )
    text = re.sub(
        re.escape(LIST_END_MARKER) + r"\s*\n\s*\n",
        f"{LIST_END_MARKER}\n",
        text,
    )
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
    text: str,
    macro_start: int,
    arg_count: int,
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
        if line.count("$$") % 2:
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
    return len(paragraph.split()) >= 8


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
