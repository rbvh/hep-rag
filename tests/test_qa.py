from argparse import Namespace

import pytest

from hep_rag.qa.ask import (
    EvidenceChunk,
    PaperEvidence,
    build_context,
    build_parser,
    format_answer,
    has_valid_citations,
    is_abstract_chunk,
    parse_chat_answer,
)


def chunk(
    paper_id: str,
    chunk_id: str,
    chunk_type: str,
    text: str,
    section: str,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        chunk_index=0,
        chunk_type=chunk_type,
        title=f"Paper {paper_id}",
        section_path=[section],
        token_count=20,
        text=text,
        source_url=None,
    )


def paper(number: int, paper_id: str) -> PaperEvidence:
    return PaperEvidence(
        source_number=number,
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        source_url=None,
        abstract=chunk(paper_id, f"{paper_id}:abstract", "abstract", f"Abstract {paper_id}", "Abstract"),
        passages=[chunk(paper_id, f"{paper_id}:passage", "paragraph", f"Passage {paper_id}", "Methods")],
    )


def test_context_reserves_every_paper_abstract() -> None:
    context, included = build_context([paper(1, "a"), paper(2, "b")], max_tokens=500)

    assert "Abstract a" in context
    assert "Abstract b" in context
    assert [item.paper_id for item in included] == ["a", "b"]


def test_abstract_detection_accepts_section_path_fallback() -> None:
    fallback = chunk("a", "a:0", "paragraph", "Abstract text", "Abstract")

    assert is_abstract_chunk(fallback)


def test_context_rejects_budget_too_small_for_abstracts() -> None:
    huge = paper(1, "a")
    huge = PaperEvidence(
        source_number=huge.source_number,
        paper_id=huge.paper_id,
        title=huge.title,
        source_url=huge.source_url,
        abstract=chunk("a", "a:abstract", "abstract", "long " * 1000, "Abstract"),
        passages=huge.passages,
    )

    with pytest.raises(SystemExit, match="abstracts exceed"):
        build_context([huge], max_tokens=500)


def test_parse_chat_answer_reads_openai_response() -> None:
    assert (
        parse_chat_answer({"choices": [{"message": {"content": "Grounded answer [1]."}}]})
        == "Grounded answer [1]."
    )


def test_citation_validation_requires_known_inline_sources() -> None:
    assert has_valid_citations("Grounded claim [1].", {1, 2})
    assert not has_valid_citations("Uncited claim.", {1, 2})
    assert not has_valid_citations("Unknown source [3].", {1, 2})


def test_format_answer_appends_numbered_sources() -> None:
    rendered = format_answer("Answer [1].", [paper(1, "2501.00001")])

    assert rendered.startswith("# Answer")
    assert "## Sources" in rendered
    assert "[1] Paper 2501.00001" in rendered
    assert "https://arxiv.org/abs/2501.00001" in rendered


def test_ask_parser_defaults_to_grounded_hybrid_rag() -> None:
    args: Namespace = build_parser().parse_args(["How does this work?"])

    assert args.retrieval == "hybrid"
    assert args.rerank is True
    assert args.top_k == 6
    assert args.neighbor_window == 0
    assert args.max_context_tokens == 5500
    assert args.qa_model == "Qwen/Qwen3.5-0.8B"
