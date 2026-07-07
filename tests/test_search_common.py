import pytest
import sys
import types
from argparse import Namespace

from hep_rag.search.common import (
    SearchHit,
    diversify_hits,
    format_hits,
    read_query,
    rerank_hits,
)


def test_format_hits_renders_vector_and_rerank_scores() -> None:
    hit = SearchHit(
        rank=1,
        score=7.25,
        vector_score=0.875,
        rerank_score=7.25,
        chunk_id="2501.00001:chunk:00000:abc",
        paper_id="2501.00001",
        title="A Test Paper",
        section_path=["Methods"],
        living_review_categories=[["Generative models", "Normalizing flows"]],
        chunk_type="paragraph",
        token_count=42,
        text="A retrieved chunk about normalizing flows.",
        source_url=None,
    )

    rendered = format_hits([hit])

    assert "2501.00001:chunk:00000:abc" in rendered
    assert "Score: 7.2500" in rendered
    assert "Rerank score: 7.2500" in rendered
    assert "Vector score: 0.8750" in rendered
    assert "Categories: Generative models > Normalizing flows" in rendered


def test_read_query_requires_query_text() -> None:
    class Args:
        query = None
        query_file = None

    with pytest.raises(SystemExit):
        read_query(Args())


def test_rerank_hits_sorts_by_reranker_score(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, pairs, **kwargs):
            assert pairs == [
                ("query", "less relevant"),
                ("query", "more relevant"),
            ]
            return [0.1, 2.0]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    hits = [
        SearchHit(
            rank=1,
            score=0.9,
            vector_score=0.9,
            rerank_score=None,
            chunk_id="a",
            paper_id="p",
            title=None,
            section_path=[],
            living_review_categories=[],
            chunk_type="paragraph",
            token_count=10,
            text="less relevant",
            source_url=None,
        ),
        SearchHit(
            rank=2,
            score=0.8,
            vector_score=0.8,
            rerank_score=None,
            chunk_id="b",
            paper_id="p",
            title=None,
            section_path=[],
            living_review_categories=[],
            chunk_type="paragraph",
            token_count=10,
            text="more relevant",
            source_url=None,
        ),
    ]
    args = Namespace(
        rerank_model="fake-reranker",
        rerank_device=None,
        device=None,
        trust_remote_code=True,
        rerank_torch_dtype="auto",
        rerank_max_length=None,
        rerank_instruction="judge relevance",
        rerank_batch_size=4,
        rerank_progress=False,
    )

    reranked = rerank_hits("query", hits, args)

    assert [hit.chunk_id for hit in reranked] == ["b", "a"]
    assert reranked[0].rank == 1
    assert reranked[0].score == 2.0
    assert reranked[0].vector_score == 0.8
    assert reranked[0].rerank_score == 2.0


def test_diversify_hits_limits_chunks_per_paper() -> None:
    hits = [
        SearchHit(
            rank=1,
            score=0.9,
            chunk_id="a1",
            paper_id="paper-a",
            title=None,
            section_path=[],
            living_review_categories=[],
            chunk_type="paragraph",
            token_count=10,
            text="first",
            source_url=None,
        ),
        SearchHit(
            rank=2,
            score=0.8,
            chunk_id="a2",
            paper_id="paper-a",
            title=None,
            section_path=[],
            living_review_categories=[],
            chunk_type="paragraph",
            token_count=10,
            text="second",
            source_url=None,
        ),
        SearchHit(
            rank=3,
            score=0.7,
            chunk_id="b1",
            paper_id="paper-b",
            title=None,
            section_path=[],
            living_review_categories=[],
            chunk_type="paragraph",
            token_count=10,
            text="third",
            source_url=None,
        ),
    ]

    diversified = diversify_hits(hits, top_k=2, max_chunks_per_paper=1)

    assert [hit.chunk_id for hit in diversified] == ["a1", "b1"]
    assert [hit.rank for hit in diversified] == [1, 2]


def test_diversify_hits_without_cap_returns_top_k() -> None:
    hits = [
        SearchHit(
            rank=index + 1,
            score=float(index),
            chunk_id=str(index),
            paper_id="same-paper",
            title=None,
            section_path=[],
            living_review_categories=[],
            chunk_type="paragraph",
            token_count=10,
            text="text",
            source_url=None,
        )
        for index in range(3)
    ]

    diversified = diversify_hits(hits, top_k=2)

    assert [hit.chunk_id for hit in diversified] == ["0", "1"]
