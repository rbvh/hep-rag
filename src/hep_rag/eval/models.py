"""Data models produced by retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperCandidate:
    paper_rank: int
    hit_rank: int
    final_score: float
    paper_id: str
    title: str | None
    living_review_categories: list[list[str]]
    chunk_id: str
    section_path: list[str]
    text: str
    source_url: str | None


@dataclass(frozen=True)
class QueryMetrics:
    query_id: str
    query: str
    expected_result: str
    target_count: int
    retrieved_count: int
    target_ranks: dict[str, int]
    missing_target_ids: list[str]
    top_paper_ids: list[str]
    recall_at_k: dict[str, float]
    precision_at_k: dict[str, float]
    hit_at_k: dict[str, float]
    mrr: float
    average_precision: float
    first_relevant_rank: int | None
    non_target_count_at_k: dict[str, int]


@dataclass(frozen=True)
class AggregateMetrics:
    query_count: int
    target_query_count: int
    no_target_query_count: int
    mean_recall_at_k: dict[str, float]
    mean_precision_at_k: dict[str, float]
    mean_hit_at_k: dict[str, float]
    mean_mrr: float
    mean_average_precision: float
    no_target_mean_non_target_count_at_k: dict[str, float]
