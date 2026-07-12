from hep_rag.ingest.living_review_categories import parse_living_review_categories


def test_parse_nested_living_review_categories() -> None:
    tex = r"""
\begin{itemize}
\item \textbf{Classification}
    \begin{itemize}
        \item \textbf{Representations}
            \begin{itemize}
                \item \textbf{Graphs}~\cite{GraphPaper:2024abc,Shared:2025abc}
                \\\textit{A graph is a collection of nodes and edges.}
                \item \textbf{Sets (point clouds)}~\cite{SetPaper:2023abc,Shared:2025abc}
            \end{itemize}
    \end{itemize}
\item \textbf{Anomaly detection}~\cite{AnomalyPaper:2022abc}
\end{itemize}
"""

    index = parse_living_review_categories(tex)

    assert index.paths_for_key("GraphPaper:2024abc") == [
        ["Classification", "Representations", "Graphs"]
    ]
    assert index.paths_for_key("SetPaper:2023abc") == [
        ["Classification", "Representations", "Sets (point clouds)"]
    ]
    assert index.paths_for_key("Shared:2025abc") == [
        ["Classification", "Representations", "Graphs"],
        ["Classification", "Representations", "Sets (point clouds)"],
    ]
    assert index.paths_for_key("AnomalyPaper:2022abc") == [["Anomaly detection"]]
    assert (
        index.descriptions_by_path[("Classification", "Representations", "Graphs")]
        == "A graph is a collection of nodes and edges."
    )
