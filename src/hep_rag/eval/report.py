"""Markdown rendering for retrieval evaluations."""

from __future__ import annotations

from typing import Any

from hep_rag.eval.models import AggregateMetrics, QueryMetrics


def render_markdown(
    aggregate: AggregateMetrics,
    query_metrics: list[QueryMetrics],
    paper_metadata: dict[str, dict[str, Any]],
    ks: list[int],
) -> str:
    lines = [
        "# Retrieval Metrics",
        "",
        f"- Queries: `{aggregate.query_count}`",
        f"- Queries with targets: `{aggregate.target_query_count}`",
        f"- Queries without targets: `{aggregate.no_target_query_count}`",
        f"- Mean average precision: `{aggregate.mean_average_precision:.4f}`",
        f"- Mean reciprocal rank: `{aggregate.mean_mrr:.4f}`",
        "",
        "## Aggregate At K",
        "",
        "| k | Recall | Precision | Hit Rate | No-target retrieved papers |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in ks:
        key = str(k)
        lines.append(
            f"| {k} | {aggregate.mean_recall_at_k[key]:.4f} | "
            f"{aggregate.mean_precision_at_k[key]:.4f} | "
            f"{aggregate.mean_hit_at_k[key]:.4f} | "
            f"{aggregate.no_target_mean_non_target_count_at_k[key]:.2f} |"
        )
    lines.extend(["", "## Query Details", ""])
    for metric in query_metrics:
        lines.extend(render_query_metrics(metric, paper_metadata, ks))
    return "\n".join(lines).rstrip() + "\n"


def render_query_metrics(
    metric: QueryMetrics,
    paper_metadata: dict[str, dict[str, Any]],
    ks: list[int],
) -> list[str]:
    lines = [
        f"### {metric.query_id}: {metric.query}",
        "",
        f"- Expected: `{metric.expected_result}`",
        f"- Targets found: `{len(metric.target_ranks)}/{metric.target_count}`",
        f"- First relevant rank: `{metric.first_relevant_rank or 'none'}`",
        f"- AP: `{metric.average_precision:.4f}`",
        f"- MRR: `{metric.mrr:.4f}`",
        "",
        "| k | Recall | Precision | Hit |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for k in ks:
        key = str(k)
        lines.append(
            f"| {k} | {metric.recall_at_k[key]:.4f} | "
            f"{metric.precision_at_k[key]:.4f} | {metric.hit_at_k[key]:.0f} |"
        )
    lines.extend(["", "Top papers:"])
    for rank, paper_id in enumerate(metric.top_paper_ids[:10], start=1):
        marker = "TARGET" if paper_id in metric.target_ranks else "MISS"
        title = paper_metadata.get(paper_id, {}).get("title") or "(metadata not found)"
        lines.append(f"- {rank}. `{marker}` {paper_id} - {title}")
    if metric.missing_target_ids:
        lines.extend(["", "Missing targets:"])
        for paper_id in metric.missing_target_ids:
            title = paper_metadata.get(paper_id, {}).get("title") or "(metadata not found)"
            lines.append(f"- {paper_id} - {title}")
    lines.append("")
    return lines
