"""Postgres retrieval queries and execution."""

from __future__ import annotations

from typing import Any

from hep_rag.search.common import SearchHit, diversify_hits, embed_queries, rerank_hits
from hep_rag.search.config import EmbeddingConfig, RerankConfig, RetrievalConfig, RetrievalMode


def retrieve_hits(
    query: str,
    *,
    database_url: str,
    table: str,
    retrieval: RetrievalConfig,
    embedding: EmbeddingConfig,
    rerank: RerankConfig | None = None,
) -> list[SearchHit]:
    """Run first-stage retrieval, optional reranking, and diversification."""

    psycopg = import_psycopg()
    try:
        from psycopg.rows import dict_row
    except ImportError as error:
        raise SystemExit("Install Postgres support with `pip install -e '.[pgvector]'`.") from error

    query_vector = None
    if retrieval_uses_vector(retrieval.mode):
        query_vector = vector_to_pgvector(embed_queries([query], embedding)[0])

    table = validate_identifier(table)
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        configure_vector_search(conn, retrieval)
        hits = search_chunks(conn, table, query, query_vector, retrieval)

    if rerank is not None:
        hits = rerank_hits(query, hits, rerank)
    return diversify_hits(hits, retrieval.final_k, retrieval.max_chunks_per_paper)


def search_chunks(
    conn: Any,
    table: str,
    query_text: str,
    query_vector: str | None,
    config: RetrievalConfig,
) -> list[SearchHit]:
    params = search_params(query_vector, query_text, config)
    return pg_rows_to_hits(conn.execute(search_sql(table, config.mode), params).fetchall())


def search_params(
    query_vector: str | None,
    query_text: str,
    config: RetrievalConfig,
) -> dict[str, Any]:
    if config.candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")
    if config.lexical_k < 1:
        raise ValueError("lexical_candidate_k must be at least 1")
    if config.rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if retrieval_uses_vector(config.mode) and query_vector is None:
        raise ValueError(f"{config.mode} retrieval requires a query vector")
    return {
        "query": query_vector,
        "query_text": query_text,
        "top_k": config.candidate_k,
        "vector_top_k": config.candidate_k,
        "lexical_top_k": config.lexical_k,
        "rrf_k": config.rrf_k,
    }


def configure_vector_search(conn: Any, config: RetrievalConfig) -> None:
    if not retrieval_uses_vector(config.mode):
        return
    if config.hnsw_ef_search is not None and config.hnsw_ef_search < 1:
        raise ValueError("hnsw_ef_search must be at least 1")
    if config.exact:
        if config.hnsw_ef_search is not None:
            raise ValueError("hnsw_ef_search only applies to approximate search")
        conn.execute("SET LOCAL enable_indexscan = off")
    elif config.hnsw_ef_search is not None:
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(config.hnsw_ef_search)}")


def retrieval_uses_vector(mode: RetrievalMode) -> bool:
    return mode in {"vector", "hybrid"}


def search_sql(table: str, mode: RetrievalMode = "vector") -> str:
    if mode == "vector":
        return vector_search_sql(table)
    if mode == "lexical":
        return lexical_search_sql(table)
    if mode == "hybrid":
        return hybrid_search_sql(table)
    raise ValueError(f"Unknown retrieval mode: {mode}")


def chunk_select_columns(alias: str = "chunks") -> str:
    columns = (
        "chunk_id",
        "paper_id",
        "title",
        "section_path",
        "living_review_categories",
        "chunk_type",
        "token_count",
        "text",
        "source_url",
    )
    return ",\n            ".join(f"{alias}.{column}" for column in columns)


def vector_search_sql(table: str) -> str:
    return f"""
        WITH query AS (SELECT %(query)s::vector AS embedding)
        SELECT
            {chunk_select_columns()},
            -(chunks.embedding <#> query.embedding) AS final_score,
            -(chunks.embedding <#> query.embedding) AS vector_score,
            NULL::double precision AS lexical_score,
            NULL::double precision AS rrf_score
        FROM {table} AS chunks, query
        ORDER BY chunks.embedding <#> query.embedding
        LIMIT %(top_k)s
    """


def lexical_search_sql(table: str) -> str:
    return f"""
        WITH query AS (
            SELECT websearch_to_tsquery('english'::regconfig, %(query_text)s) AS terms
        )
        SELECT
            {chunk_select_columns()},
            ts_rank_cd(chunks.search_vector, query.terms) AS final_score,
            NULL::double precision AS vector_score,
            ts_rank_cd(chunks.search_vector, query.terms) AS lexical_score,
            NULL::double precision AS rrf_score
        FROM {table} AS chunks, query
        WHERE query.terms @@ chunks.search_vector
        ORDER BY final_score DESC, chunks.chunk_id
        LIMIT %(top_k)s
    """


def hybrid_search_sql(table: str) -> str:
    return f"""
        WITH
        vector_query AS (SELECT %(query)s::vector AS embedding),
        lexical_query AS (
            SELECT websearch_to_tsquery('english'::regconfig, %(query_text)s) AS terms
        ),
        vector_hits AS (
            SELECT chunks.chunk_id,
                -(chunks.embedding <#> vector_query.embedding) AS vector_score,
                row_number() OVER (
                    ORDER BY chunks.embedding <#> vector_query.embedding
                ) AS vector_rank
            FROM {table} AS chunks, vector_query
            ORDER BY chunks.embedding <#> vector_query.embedding
            LIMIT %(vector_top_k)s
        ),
        lexical_hits AS (
            SELECT chunks.chunk_id,
                ts_rank_cd(chunks.search_vector, lexical_query.terms) AS lexical_score,
                row_number() OVER (
                    ORDER BY ts_rank_cd(chunks.search_vector, lexical_query.terms) DESC,
                        chunks.chunk_id
                ) AS lexical_rank
            FROM {table} AS chunks, lexical_query
            WHERE lexical_query.terms @@ chunks.search_vector
            ORDER BY lexical_score DESC, chunks.chunk_id
            LIMIT %(lexical_top_k)s
        ),
        combined AS (
            SELECT chunk_id FROM vector_hits
            UNION
            SELECT chunk_id FROM lexical_hits
        ),
        scored AS (
            SELECT combined.chunk_id,
                vector_hits.vector_score,
                lexical_hits.lexical_score,
                COALESCE(1.0 / (%(rrf_k)s + vector_hits.vector_rank), 0.0)
                    + COALESCE(1.0 / (%(rrf_k)s + lexical_hits.lexical_rank), 0.0)
                    AS rrf_score
            FROM combined
            LEFT JOIN vector_hits USING (chunk_id)
            LEFT JOIN lexical_hits USING (chunk_id)
        )
        SELECT
            {chunk_select_columns()},
            scored.rrf_score AS final_score,
            scored.vector_score,
            scored.lexical_score,
            scored.rrf_score
        FROM scored
        JOIN {table} AS chunks USING (chunk_id)
        ORDER BY final_score DESC, chunks.chunk_id
        LIMIT %(top_k)s
    """


def pg_rows_to_hits(rows: list[dict[str, Any]]) -> list[SearchHit]:
    hits = []
    for rank, row in enumerate(rows, start=1):
        hits.append(
            SearchHit(
                rank=rank,
                final_score=float(row["final_score"]),
                chunk_id=str(row["chunk_id"]),
                paper_id=str(row["paper_id"]),
                title=value_or_none(row, "title"),
                section_path=[str(part) for part in row.get("section_path", [])],
                living_review_categories=[
                    [str(part) for part in category]
                    for category in row.get("living_review_categories", [])
                ],
                chunk_type=str(row["chunk_type"]),
                token_count=int(row["token_count"]),
                text=str(row["text"]),
                source_url=value_or_none(row, "source_url"),
                vector_score=optional_float(row.get("vector_score")),
                lexical_score=optional_float(row.get("lexical_score")),
                rrf_score=optional_float(row.get("rrf_score")),
            )
        )
    return hits


def vector_to_pgvector(vector: Any) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in vector) + "]"


def validate_identifier(identifier: str) -> str:
    if not identifier or not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return identifier


def import_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as error:
        raise SystemExit("Install Postgres support with `pip install -e '.[pgvector]'`.") from error
    return psycopg


def value_or_none(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
