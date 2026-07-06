from hep_rag.chunking.chunker import chunk_paper
from hep_rag.ingest.schema import ParsedPaper, ParsedParagraph


def test_chunk_paper_keeps_adjacent_paragraphs_separate() -> None:
    paper = ParsedPaper(
        paper_id="2501.00001",
        bib_key="Test:2025abc",
        title="A Test Paper",
        authors="A. Author",
        year="2025",
        primary_class="hep-ph",
        source_url="https://arxiv.org/e-print/2501.00001",
        source_path="source.tar.gz",
        extracted_path="extracted",
        main_tex_path="extracted/main.tex",
        living_review_categories=[["Classification"]],
        abstract=None,
        paragraphs=[
            ParsedParagraph(
                paragraph_id="p1",
                section_path=["Introduction"],
                text=" ".join(["alpha"] * 70),
            ),
            ParsedParagraph(
                paragraph_id="p2",
                section_path=["Introduction"],
                text=" ".join(["beta"] * 70),
            ),
            ParsedParagraph(
                paragraph_id="p3",
                section_path=["Methods"],
                text=" ".join(["gamma"] * 70),
            ),
        ],
    )

    chunks = chunk_paper(paper, min_tokens=120, target_tokens=200, max_tokens=300)

    assert len(chunks) == 3
    assert chunks[0].section_path == ["Introduction"]
    assert chunks[0].paragraph_ids == ["p1"]
    assert chunks[1].section_path == ["Introduction"]
    assert chunks[1].paragraph_ids == ["p2"]
    assert chunks[2].section_path == ["Methods"]


def test_chunk_paper_keeps_oversized_equation_paragraph_together() -> None:
    paper = ParsedPaper(
        paper_id="2501.00001",
        bib_key="Test:2025abc",
        title="A Test Paper",
        authors="A. Author",
        year="2025",
        primary_class="hep-ph",
        source_url="https://arxiv.org/e-print/2501.00001",
        source_path="source.tar.gz",
        extracted_path="extracted",
        main_tex_path="extracted/main.tex",
        living_review_categories=[["Classification"]],
        abstract=None,
        paragraphs=[
            ParsedParagraph(
                paragraph_id="p1",
                section_path=["Methods"],
                text=(
                    " ".join(["context"] * 40)
                    + r" \begin{align} "
                    + " ".join(["x_i = y_i"] * 80)
                    + r" \end{align} "
                    + " ".join(["more"] * 40)
                ),
            ),
        ],
    )

    chunks = chunk_paper(paper, min_tokens=20, target_tokens=60, max_tokens=80)

    assert len(chunks) == 1
    assert chunks[0].paragraph_ids == ["p1"]
    assert r"\begin{align}" in chunks[0].text
    assert r"\end{align}" in chunks[0].text
