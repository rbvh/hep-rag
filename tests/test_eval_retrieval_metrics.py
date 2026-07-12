from hep_rag.eval.metrics import aggregate_metrics, average_precision_for_ranks, score_query
from hep_rag.eval.models import PaperCandidate
from hep_rag.eval.report import render_markdown
from hep_rag.eval.runner import build_parser, collapse_hits_by_paper, unique_paper_hits
from hep_rag.search.common import SearchHit


def candidate(rank: int, paper_id: str) -> PaperCandidate:
    return PaperCandidate(
        paper_rank=rank,
        hit_rank=rank,
        final_score=1.0 / rank,
        paper_id=paper_id,
        title=f"Title {paper_id}",
        living_review_categories=[],
        chunk_id=f"{paper_id}:chunk",
        section_path=[],
        text="text",
        source_url=None,
    )


def hit(rank: int, paper_id: str, score: float) -> SearchHit:
    return SearchHit(
        rank=rank,
        final_score=score,
        chunk_id=f"{paper_id}:chunk:{rank:05d}:abc",
        paper_id=paper_id,
        title=f"Title {paper_id}",
        section_path=["Methods"],
        living_review_categories=[["Classification", "Targets"]],
        chunk_type="paragraph",
        token_count=20,
        text=f"Text for {paper_id}",
        source_url=None,
        vector_score=score,
    )


def test_collapse_hits_by_paper_keeps_best_first_hit_per_paper() -> None:
    candidates = collapse_hits_by_paper(
        [
            hit(1, "paper-a", 0.9),
            hit(2, "paper-a", 0.8),
            hit(3, "paper-b", 0.7),
        ],
        limit=10,
    )

    assert [candidate.paper_id for candidate in candidates] == ["paper-a", "paper-b"]
    assert [candidate.paper_rank for candidate in candidates] == [1, 2]
    assert candidates[0].hit_rank == 1


def test_collapse_hits_by_paper_limits_unique_papers() -> None:
    candidates = collapse_hits_by_paper(
        [hit(1, "paper-a", 0.9), hit(2, "paper-b", 0.8)],
        limit=1,
    )

    assert [candidate.paper_id for candidate in candidates] == ["paper-a"]


def test_representative_hits_by_paper_keeps_first_hit_per_paper() -> None:
    representatives = unique_paper_hits(
        [
            hit(1, "paper-a", 0.9),
            hit(2, "paper-a", 0.8),
            hit(3, "paper-b", 0.7),
        ],
        limit=10,
    )

    assert [hit.paper_id for hit in representatives] == ["paper-a", "paper-b"]
    assert representatives[0].rank == 1


def test_representative_hits_by_paper_limits_papers() -> None:
    representatives = unique_paper_hits(
        [hit(1, "paper-a", 0.9), hit(2, "paper-b", 0.8)],
        limit=1,
    )

    assert [hit.paper_id for hit in representatives] == ["paper-a"]


def test_representative_hits_by_paper_rejects_non_positive_limit() -> None:
    try:
        unique_paper_hits([], limit=0)
    except ValueError as error:
        assert "paper limit" in str(error)
    else:
        raise AssertionError("Expected invalid rerank candidate count to stop cleanly")


def test_parser_defaults_to_vector_without_rerank() -> None:
    args = build_parser().parse_args([])

    assert args.retrieval == "vector"
    assert args.rerank is False
    assert args.candidate_k == 300


def test_average_precision_for_ranks() -> None:
    assert average_precision_for_ranks([1, 3], target_count=2) == (1 / 1 + 2 / 3) / 2
    assert average_precision_for_ranks([], target_count=2) == 0.0
    assert average_precision_for_ranks([1], target_count=0) == 0.0


def test_score_query_computes_paper_level_metrics() -> None:
    metric = score_query(
        {
            "query_id": "q1",
            "query": "query",
            "expected_result": "multi_paper",
            "relevant_paper_ids": ["paper-a", "paper-c"],
        },
        [
            candidate(1, "paper-a"),
            candidate(2, "paper-b"),
            candidate(3, "paper-c"),
        ],
        ks=[1, 2, 3],
    )

    assert metric.target_ranks == {"paper-a": 1, "paper-c": 3}
    assert metric.missing_target_ids == []
    assert metric.recall_at_k == {"1": 0.5, "2": 0.5, "3": 1.0}
    assert metric.precision_at_k == {"1": 1.0, "2": 0.5, "3": 2 / 3}
    assert metric.hit_at_k == {"1": 1.0, "2": 1.0, "3": 1.0}
    assert metric.mrr == 1.0
    assert metric.average_precision == (1 / 1 + 2 / 3) / 2


def test_score_query_tracks_no_target_non_targets() -> None:
    metric = score_query(
        {
            "query_id": "q2",
            "query": "negative query",
            "expected_result": "no_known_match",
            "relevant_paper_ids": [],
        },
        [candidate(1, "paper-a"), candidate(2, "paper-b")],
        ks=[1, 3],
    )

    assert metric.target_count == 0
    assert metric.recall_at_k == {"1": 0.0, "3": 0.0}
    assert metric.non_target_count_at_k == {"1": 1, "3": 2}


def test_aggregate_metrics_separates_target_and_no_target_queries() -> None:
    target_metric = score_query(
        {
            "query_id": "q1",
            "query": "query",
            "expected_result": "multi_paper",
            "relevant_paper_ids": ["paper-a"],
        },
        [candidate(1, "paper-a")],
        ks=[1],
    )
    no_target_metric = score_query(
        {
            "query_id": "q2",
            "query": "negative query",
            "expected_result": "no_known_match",
            "relevant_paper_ids": [],
        },
        [candidate(1, "paper-b")],
        ks=[1],
    )

    aggregate = aggregate_metrics([target_metric, no_target_metric], ks=[1])

    assert aggregate.query_count == 2
    assert aggregate.target_query_count == 1
    assert aggregate.no_target_query_count == 1
    assert aggregate.mean_recall_at_k == {"1": 1.0}
    assert aggregate.no_target_mean_non_target_count_at_k == {"1": 1}


def test_render_markdown_includes_summary_and_missing_targets() -> None:
    metric = score_query(
        {
            "query_id": "q1",
            "query": "query",
            "expected_result": "multi_paper",
            "relevant_paper_ids": ["paper-a", "paper-c"],
        },
        [candidate(1, "paper-a"), candidate(2, "paper-b")],
        ks=[1, 2],
    )
    aggregate = aggregate_metrics([metric], ks=[1, 2])

    markdown = render_markdown(
        aggregate,
        [metric],
        paper_metadata={"paper-c": {"title": "Missing Paper"}},
        ks=[1, 2],
    )

    assert "# Retrieval Metrics" in markdown
    assert "Mean average precision" in markdown
    assert "paper-c - Missing Paper" in markdown
