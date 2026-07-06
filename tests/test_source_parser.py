from pathlib import Path

from hep_rag.ingest.source_parser import (
    find_main_tex_file,
    parse_body_paragraphs,
    parse_latex_source,
    split_paragraphs,
    strip_comments,
    text_for_paragraph_splitting,
)


def test_parse_latex_source_with_inputs_and_sections(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "intro.tex").write_text(
        r"""
\section{Introduction}
This is the first paragraph of the introduction. It mentions normalizing flows
and event generation with enough words to be a useful paragraph.

This is a second paragraph with a citation~\cite{Example:2024abc} and a
reference to Eq.~\ref{eq:thing}. It should be extracted cleanly.
""",
        encoding="utf-8",
    )
    (extracted / "main.tex").write_text(
        r"""
\documentclass{article}
\title{A Test Paper}
\abstract{This abstract explains the paper in enough words to become a chunk later.}
\begin{document}
\maketitle
\input{intro}
\section{Methods}
\subsection{Architecture}
The method paragraph describes the architecture and preserves section hierarchy
for later retrieval metadata. This paragraph has enough words to survive.
\begin{equation}
E = mc^2
\label{eq:energy}
\end{equation}
\begin{figure}
\caption{A useful caption describing a figure with enough words for retrieval.}
\end{figure}
\bibliography{refs}
\end{document}
""",
        encoding="utf-8",
    )

    paper = parse_latex_source(
        {
            "arxiv_id": "2501.00001",
            "bib_key": "Test:2025abc",
            "title": "{Fallback Title}",
            "authors": "A. Author",
            "year": "2025",
            "primary_class": "hep-ph",
            "source_url": "https://arxiv.org/e-print/2501.00001",
            "source_path": "source.tar.gz",
            "extracted_path": str(extracted),
            "living_review_categories": [["Classification", "Representations"]],
        }
    )

    assert paper.title == "A Test Paper"
    assert paper.abstract.startswith("This abstract explains")
    assert find_main_tex_file(extracted) == extracted / "main.tex"
    assert [paragraph.section_path for paragraph in paper.paragraphs] == [
        ["Abstract"],
        ["Introduction"],
        ["Introduction"],
        ["Methods", "Architecture"],
        ["Methods", "Architecture"],
    ]
    assert r"\cite{Example:2024abc}" in paper.paragraphs[2].text
    assert "[ref:" not in paper.paragraphs[2].text
    assert "eq:thing" not in paper.paragraphs[2].text
    assert paper.paragraphs[-2].kind == "paragraph"
    assert "\n" + r"\begin{equation}" in paper.paragraphs[-2].text
    assert r"E = mc^2" in paper.paragraphs[-2].text
    assert r"E = mc^2" + "\n" + r"\end{equation}" in paper.paragraphs[-2].text
    assert r"\label{eq:energy}" not in paper.paragraphs[-2].text
    assert paper.paragraphs[-1].kind == "caption"
    assert paper.paragraphs[-1].text.startswith("A useful caption")


def test_itemize_environment_is_part_of_surrounding_paragraph() -> None:
    text = r"""
Lead-in paragraph with enough words to survive paragraph filtering and introduce:

\begin{itemize}
    \item first useful consequence with enough words to be meaningful
    \item second useful consequence with enough words to be meaningful
\end{itemize}

Closing paragraph with enough words to survive paragraph filtering.
"""

    paragraphs = split_paragraphs(text_for_paragraph_splitting(text))

    assert len(paragraphs) == 1
    assert paragraphs[0].startswith("Lead-in paragraph")
    assert "\n- first useful consequence" in paragraphs[0]
    assert "\n- second useful consequence" in paragraphs[0]
    assert "\nClosing paragraph" in paragraphs[0]
    assert "\n\nClosing paragraph" not in paragraphs[0]
    assert paragraphs[0].endswith("paragraph filtering.")


def test_inline_math_symbols_and_floats_do_not_split_paragraph() -> None:
    text = r"""
The scale $\xi$ becomes a dimensionless quantity $\xi/a$ which diverges
towards the continuum limit $a\rightarrow 0$.
\begin{figure}
\caption{A figure caption that should be extracted elsewhere.}
\end{figure}
The lattice spacing is determined by the coupling $\beta\rightarrow\infty$.
"""

    paragraphs = split_paragraphs(text_for_paragraph_splitting(text))

    assert len(paragraphs) == 1
    assert r"scale $\xi$ becomes" in paragraphs[0]
    assert r"quantity $\xi/a$" in paragraphs[0]
    assert r"$a\rightarrow 0$" in paragraphs[0]
    assert r"$\beta\rightarrow\infty$" in paragraphs[0]
    assert "figure caption" not in paragraphs[0]


def test_custom_figure_macros_drop_payloads_and_keep_captions() -> None:
    text = r"""
\section{Results}
The model comparison paragraph has enough scientific words before the figure.
\doublefigure[htbp]
{\begin{tikzpicture} \draw (0,0) -- (1,1); \end{tikzpicture}}{\LabPhi}
{First panel caption describes azimuthal spectra for retrieval.}
{\input{huge_plot_source.tex}}{\LabPhiRat}
{Second panel caption describes ratio spectra for retrieval.}
{Overall caption describes the figure and its physics conclusion.}
{fig:bulk_phi}
The paragraph continues after the custom figure macro with enough words.
"""

    paragraphs = parse_body_paragraphs(text, paper_id="2501.00001")

    assert paragraphs[0].kind == "paragraph"
    assert "model comparison paragraph" in paragraphs[0].text
    assert "paragraph continues" in paragraphs[0].text
    assert "tikzpicture" not in paragraphs[0].text
    assert "huge_plot_source" not in paragraphs[0].text
    assert [paragraph.kind for paragraph in paragraphs[1:]] == ["caption", "caption", "caption"]
    assert paragraphs[1].text.startswith("First panel caption")
    assert paragraphs[2].text.startswith("Second panel caption")
    assert paragraphs[3].text.startswith("Overall caption")


def test_non_retrieval_author_and_acknowledgement_sections_are_skipped() -> None:
    text = r"""
\section{Results}
This useful result paragraph has enough scientific words to be retained.
\section*{The CMS Collaboration}
\cmsinstitute{A Very Long Institute List} A. Author, B. Author, C. Author, D. Author.
\section{Acknowledgements}
We thank funding agencies and administrative teams for support over many years.
"""

    paragraphs = parse_body_paragraphs(text, paper_id="2501.00001")

    assert len(paragraphs) == 1
    assert paragraphs[0].section_path == ["Results"]
    assert "useful result paragraph" in paragraphs[0].text


def test_comment_only_lines_do_not_split_equation_paragraph() -> None:
    text = r"""
The loss is based on the reverse Kullback-Leibler divergence
%
\begin{equation}
D_{KL}(q||p) = \int q(U) [\log q(U) - \log p(U)]
\end{equation}
%
which is available because the target distribution is known.
"""

    paragraphs = split_paragraphs(text_for_paragraph_splitting(strip_comments(text)))

    assert len(paragraphs) == 1
    assert "reverse Kullback-Leibler divergence" in paragraphs[0]
    assert r"\begin{equation}" in paragraphs[0]
    assert "which is available" in paragraphs[0]
