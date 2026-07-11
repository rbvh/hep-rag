"""Evaluate paper-level retrieval quality on the draft eval set."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from hep_rag.embed.build_embeddings import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL,
    embed_with_openai_compatible,
    embed_with_sentence_transformers,
    normalize_rows,
)
from hep_rag.search.common import (
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL,
    SearchHit,
    load_json,
    load_jsonl,
    rerank_hits,
)
from hep_rag.search.pgvector_search import (
    DEFAULT_RRF_K,
    DEFAULT_TABLE,
    configure_vector_search,
    database_url_from_args,
    import_psycopg,
    pg_rows_to_hits,
    retrieval_uses_vector,
    search_sql,
    search_params,
    validate_identifier,
    vector_to_pgvector,
)


DEFAULT_EVAL_SET = Path("data/eval/retrieval_queries.jsonl")
DEFAULT_PAPERS = Path("data/processed/papers.jsonl")
DEFAULT_JSON_OUT = Path("data/eval/reports/retrieval_metrics.json")
DEFAULT_MARKDOWN_OUT = Path("data/eval/reports/retrieval_metrics.md")
DEFAULT_KS = (1, 3, 5, 10, 20, 40)


@dataclass(frozen=True)
class PaperCandidate:
    paper_rank: int
    hit_rank: int
    score: float
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_rows = load_eval_rows(args.eval_set, limit=args.limit)
    paper_metadata = load_paper_metadata(args.papers)
    embedding_config = (
        load_json(args.embedding_dir / "config.json")
        if retrieval_uses_vector(args.retrieval)
        else {}
    )
    table = validate_identifier(args.table)
    ks = metric_cutoffs(args)
    top_papers = max(max(ks), args.top_papers)

    if args.dry_run:
        print(f"Eval rows: {len(eval_rows)}")
        print(f"Eval set: {args.eval_set}")
        print(f"Embedding dir: {args.embedding_dir}")
        print(f"Database: {database_url_from_args(args)}")
        print(f"Table: {table}")
        print(f"Candidate chunks per query: {args.candidate_k}")
        if args.retrieval in {"bm25", "hybrid"}:
            print(f"BM25 candidate chunks per query: {args.bm25_candidate_k or args.candidate_k}")
        print(f"Paper cutoff: {top_papers}")
        print(f"Metrics k: {','.join(str(k) for k in ks)}")
        print(f"Retrieval: {args.retrieval}")
        print(f"Rerank: {args.rerank}")
        if args.rerank:
            print(f"Rerank model: {args.rerank_model}")
            print(f"Rerank candidate papers: {args.rerank_candidate_papers}")
        print(f"Search mode: {'exact' if args.exact else 'approximate'}")
        if args.hnsw_ef_search is not None:
            print(f"HNSW ef_search: {args.hnsw_ef_search}")
        return 0

    query_vectors = (
        embed_query_vectors(eval_rows, args, embedding_config)
        if retrieval_uses_vector(args.retrieval)
        else None
    )
    psycopg = import_psycopg()
    try:
        from psycopg.rows import dict_row
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: psycopg. Install with `pip install -e '.[pgvector]'`."
        ) from error

    sql = search_sql(table, retrieval=args.retrieval)
    query_metrics: list[QueryMetrics] = []
    with psycopg.connect(database_url_from_args(args), row_factory=dict_row) as conn:
        if retrieval_uses_vector(args.retrieval):
            configure_vector_search(conn, exact=args.exact, hnsw_ef_search=args.hnsw_ef_search)
        for index, eval_row in enumerate(eval_rows):
            rows = conn.execute(
                sql,
                search_params(
                    query_vector=(
                        vector_to_pgvector(query_vectors[index])
                        if query_vectors is not None
                        else None
                    ),
                    query_text=str(eval_row["query"]),
                    top_k=args.candidate_k,
                    vector_top_k=args.candidate_k,
                    bm25_top_k=args.bm25_candidate_k or args.candidate_k,
                    rrf_k=args.rrf_k,
                ),
            ).fetchall()
            hits = pg_rows_to_hits(rows)
            if args.rerank:
                hits = representative_hits_by_paper(
                    hits,
                    max_papers=max(top_papers, args.rerank_candidate_papers),
                )
                hits = rerank_hits(str(eval_row["query"]), hits, args)
            candidates = collapse_hits_by_paper(
                hits,
                top_papers=top_papers,
            )
            metrics = score_query(eval_row, candidates, ks)
            query_metrics.append(metrics)
            print(
                f"{metrics.query_id}: MAP={metrics.average_precision:.3f} "
                f"MRR={metrics.mrr:.3f} targets={len(metrics.target_ranks)}/{metrics.target_count}"
            )

    aggregate = aggregate_metrics(query_metrics, ks)
    payload = {
        "config": {
            "eval_set": str(args.eval_set),
            "embedding_dir": str(args.embedding_dir),
            "table": table,
            "candidate_k": args.candidate_k,
            "bm25_candidate_k": args.bm25_candidate_k or args.candidate_k,
            "retrieval": args.retrieval,
            "rrf_k": args.rrf_k,
            "top_papers": top_papers,
            "k": ks,
            "exact": args.exact,
            "hnsw_ef_search": args.hnsw_ef_search,
            "rerank": args.rerank,
            "rerank_model": args.rerank_model if args.rerank else None,
            "rerank_candidate_papers": args.rerank_candidate_papers if args.rerank else None,
        },
        "aggregate": asdict(aggregate),
        "queries": [asdict(metric) for metric in query_metrics],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(
        render_markdown(aggregate, query_metrics, paper_metadata, ks),
        encoding="utf-8",
    )
    print(f"Wrote JSON metrics: {args.json_out}")
    print(f"Wrote Markdown metrics: {args.markdown_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--database-url", help="Postgres URL. Defaults to $DATABASE_URL.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing config.json for query model defaults.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=300,
        help="Number of vector chunk hits to fetch before collapsing to unique papers.",
    )
    parser.add_argument(
        "--bm25-candidate-k",
        type=int,
        help="Number of lexical hits to fetch for BM25 or hybrid retrieval. Defaults to --candidate-k.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["vector", "bm25", "hybrid"],
        default="vector",
        help="First-stage retrieval strategy.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="Reciprocal Rank Fusion constant for hybrid retrieval.",
    )
    parser.add_argument(
        "--top-papers",
        type=int,
        default=40,
        help="Minimum number of unique papers to retain for scoring.",
    )
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        default=None,
        help="Paper-level cutoff for metrics. Can be passed more than once.",
    )
    parser.add_argument(
        "--exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exact vector search. Pass --no-exact to allow ANN indexes.",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        help="Set hnsw.ef_search for --no-exact HNSW searches.",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rerank paper-representative candidates with a cross-encoder reranker.",
    )
    parser.add_argument(
        "--rerank-candidate-papers",
        type=int,
        default=100,
        help="Number of unique paper representatives to rerank before scoring.",
    )
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    parser.add_argument("--rerank-max-length", type=int, help="Optional reranker max sequence length.")
    parser.add_argument("--rerank-device", help="Torch device for reranker. Defaults to --device.")
    parser.add_argument(
        "--rerank-torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--rerank-instruction",
        default=DEFAULT_RERANK_INSTRUCTION,
        help="Instruction prompt passed to instruction-aware rerankers.",
    )
    parser.add_argument(
        "--rerank-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show reranker progress bar.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N eval rows.")
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "openai-compatible"],
        default="sentence-transformers",
    )
    parser.add_argument("--model", help="Query embedding model. Defaults to embedding config.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="SentenceTransformer max sequence length. Use 0 to keep the model default.",
    )
    parser.add_argument("--device", help="Torch device for sentence-transformers.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prompt-name", default="query")
    parser.add_argument("--prompt")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_eval_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    for index, row in enumerate(rows, start=1):
        if not row.get("query_id"):
            raise SystemExit(f"Missing query_id in {path}:{index}")
        if not row.get("query"):
            raise SystemExit(f"Missing query in {path}:{index}")
    return rows


def load_paper_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["paper_id"]): row for row in load_jsonl(path)}


def embed_query_vectors(
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    embedding_config: dict[str, Any],
) -> Any:
    embed_args = argparse.Namespace(
        backend=args.backend,
        model=args.model or embedding_config.get("model") or DEFAULT_MODEL,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        normalize=args.normalize,
        prompt=args.prompt,
        prompt_name=args.prompt_name,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    texts = [query_text_for_embedding(row["query"], embed_args) for row in eval_rows]
    if args.backend == "sentence-transformers":
        vectors = embed_with_sentence_transformers(texts, embed_args)
    elif args.backend == "openai-compatible":
        vectors = embed_with_openai_compatible(texts, embed_args)
        if args.normalize:
            vectors = normalize_rows(vectors)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    return vectors.astype("float32", copy=False)


def query_text_for_embedding(query: str, args: argparse.Namespace) -> str:
    if args.backend == "openai-compatible" and args.prompt:
        return args.prompt + query
    return query


def representative_hits_by_paper(hits: list[SearchHit], max_papers: int) -> list[SearchHit]:
    if max_papers < 1:
        raise SystemExit("--rerank-candidate-papers must be at least 1")
    representatives = []
    seen: set[str] = set()
    for hit in hits:
        if hit.paper_id in seen:
            continue
        seen.add(hit.paper_id)
        representatives.append(hit)
        if len(representatives) >= max_papers:
            break
    return representatives


def collapse_hits_by_paper(
    hits: list[SearchHit],
    top_papers: int,
) -> list[PaperCandidate]:
    candidates: list[PaperCandidate] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.paper_id in seen:
            continue
        seen.add(hit.paper_id)
        candidates.append(
            PaperCandidate(
                paper_rank=len(candidates) + 1,
                hit_rank=hit.rank,
                score=hit.score,
                paper_id=hit.paper_id,
                title=hit.title,
                living_review_categories=hit.living_review_categories,
                chunk_id=hit.chunk_id,
                section_path=hit.section_path,
                text=hit.text,
                source_url=hit.source_url,
            )
        )
        if len(candidates) >= top_papers:
            break
    return candidates


def metric_cutoffs(args: argparse.Namespace) -> list[int]:
    return sorted(set(args.k or DEFAULT_KS))


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
        non_target_count_at_k[str(k)] = len([paper_id for paper_id in top_k if paper_id not in target_set])

    relevant_ranks = sorted(target_ranks.values())
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    average_precision = average_precision_for_ranks(relevant_ranks, len(target_set))
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
        mrr=mrr,
        average_precision=average_precision,
        first_relevant_rank=first_relevant_rank,
        non_target_count_at_k=non_target_count_at_k,
    )


def average_precision_for_ranks(relevant_ranks: list[int], target_count: int) -> float:
    if target_count == 0 or not relevant_ranks:
        return 0.0
    precision_sum = 0.0
    for found_count, rank in enumerate(relevant_ranks, start=1):
        precision_sum += found_count / rank
    return precision_sum / target_count


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
            mean([metric.average_precision for metric in target_queries])
            if target_queries
            else 0.0
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
        str(k): mean([getattr(metric, metric_name)[str(k)] for metric in query_metrics])
        for k in ks
    }


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
            "| "
            f"{k} | {aggregate.mean_recall_at_k[key]:.4f} | "
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


if __name__ == "__main__":
    raise SystemExit(main())
