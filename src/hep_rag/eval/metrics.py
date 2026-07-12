"""Pure paper-level retrieval metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any

from hep_rag.eval.models import AggregateMetrics, PaperCandidate, QueryMetrics


def score_query(
    eval_row: dict[str, Any],
    candidates: list[PaperCandidate],
    ks: list[int],
) -> QueryMetrics:
    target_ids = [str(paper_id) for paper_id in eval_row.get("relevant_paper_ids") or []]
    target_set = set(target_ids)
    top_paper_ids = [candidate.paper_id for candidate in candidates]
    target_ranks = {
        paper_id: rank
        for rank, paper_id in enumerate(top_paper_ids, start=1)
        if paper_id in target_set
    }
    missing = [paper_id for paper_id in target_ids if paper_id not in target_ranks]
    recall_at_k = {}
    precision_at_k = {}
    hit_at_k = {}
    non_target_count_at_k = {}
    for k in ks:
        top_k = top_paper_ids[:k]
        found = len(target_set.intersection(top_k))
        recall_at_k[str(k)] = found / len(target_set) if target_set else 0.0
        precision_at_k[str(k)] = found / k if target_set else 0.0
        hit_at_k[str(k)] = 1.0 if found else 0.0
        non_target_count_at_k[str(k)] = sum(paper_id not in target_set for paper_id in top_k)

    relevant_ranks = sorted(target_ranks.values())
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
    return QueryMetrics(
        query_id=str(eval_row["query_id"]),
        query=str(eval_row["query"]),
        expected_result=str(eval_row.get("expected_result") or ""),
        target_count=len(target_set),
        retrieved_count=len(candidates),
        target_ranks=target_ranks,
        missing_target_ids=missing,
        top_paper_ids=top_paper_ids,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        hit_at_k=hit_at_k,
        mrr=1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        average_precision=average_precision_for_ranks(relevant_ranks, len(target_set)),
        first_relevant_rank=first_relevant_rank,
        non_target_count_at_k=non_target_count_at_k,
    )


def average_precision_for_ranks(relevant_ranks: list[int], target_count: int) -> float:
    if target_count == 0 or not relevant_ranks:
        return 0.0
    return (
        sum(found_count / rank for found_count, rank in enumerate(relevant_ranks, start=1))
        / target_count
    )


def aggregate_metrics(query_metrics: list[QueryMetrics], ks: list[int]) -> AggregateMetrics:
    target_queries = [metric for metric in query_metrics if metric.target_count > 0]
    no_target_queries = [metric for metric in query_metrics if metric.target_count == 0]
    return AggregateMetrics(
        query_count=len(query_metrics),
        target_query_count=len(target_queries),
        no_target_query_count=len(no_target_queries),
        mean_recall_at_k=mean_metric_at_k(target_queries, "recall_at_k", ks),
        mean_precision_at_k=mean_metric_at_k(target_queries, "precision_at_k", ks),
        mean_hit_at_k=mean_metric_at_k(target_queries, "hit_at_k", ks),
        mean_mrr=mean([metric.mrr for metric in target_queries]) if target_queries else 0.0,
        mean_average_precision=(
            mean([metric.average_precision for metric in target_queries]) if target_queries else 0.0
        ),
        no_target_mean_non_target_count_at_k={
            str(k): (
                mean([metric.non_target_count_at_k[str(k)] for metric in no_target_queries])
                if no_target_queries
                else 0.0
            )
            for k in ks
        },
    )


def mean_metric_at_k(
    query_metrics: list[QueryMetrics],
    metric_name: str,
    ks: list[int],
) -> dict[str, float]:
    if not query_metrics:
        return {str(k): 0.0 for k in ks}
    return {
        str(k): mean([getattr(metric, metric_name)[str(k)] for metric in query_metrics]) for k in ks
    }
