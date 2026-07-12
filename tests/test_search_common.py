import pytest

import hep_rag.search.common as search_common
from hep_rag.search.common import (
    SearchHit,
    diversify_hits,
    parse_rerank_scores,
    rerank_hits,
)
from hep_rag.search.config import RerankConfig


def test_rerank_hits_sorts_by_reranker_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post_json(url, payload, api_key=None, timeout=120.0):
        assert url == "http://reranker/rerank"
        assert payload["query"] == "query"
        assert payload["documents"] == ["less relevant", "more relevant"]
        return {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 2.0},
            ]
        }

    monkeypatch.setattr(search_common, "post_json", fake_post_json)
    hits = [
        SearchHit(
            rank=1,
            final_score=0.9,
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
            final_score=0.8,
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
    config = RerankConfig(
        model="fake-reranker",
        base_url="http://reranker",
        instruction="judge relevance",
    )

    reranked = rerank_hits("query", hits, config)

    assert [hit.chunk_id for hit in reranked] == ["b", "a"]
    assert reranked[0].rank == 1
    assert reranked[0].final_score == 2.0
    assert reranked[0].vector_score == 0.8
    assert reranked[0].rerank_score == 2.0


def test_parse_rerank_scores_accepts_vllm_results_response() -> None:
    scores = parse_rerank_scores(
        {
            "results": [
                {"index": 1, "relevance_score": 0.25},
                {"index": 0, "relevance_score": 0.75},
            ]
        },
        expected_count=2,
    )

    assert scores == {0: 0.75, 1: 0.25}


def test_parse_rerank_scores_accepts_data_response() -> None:
    scores = parse_rerank_scores(
        {
            "data": [
                {"index": 0, "score": -1.0},
                {"index": 1, "score": 3.5},
            ]
        },
        expected_count=2,
    )

    assert scores == {0: -1.0, 1: 3.5}


def test_parse_rerank_scores_rejects_missing_scores() -> None:
    with pytest.raises(SystemExit):
        parse_rerank_scores({"results": [{"index": 0, "relevance_score": 1.0}]}, 2)


def test_diversify_hits_limits_chunks_per_paper() -> None:
    hits = [
        SearchHit(
            rank=1,
            final_score=0.9,
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
            final_score=0.8,
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
            final_score=0.7,
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
            final_score=float(index),
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
