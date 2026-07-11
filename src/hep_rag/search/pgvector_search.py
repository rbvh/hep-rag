"""Load and query chunk embeddings with Postgres + pgvector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from hep_rag.embed.build_embeddings import DEFAULT_MAX_SEQ_LENGTH
from hep_rag.search.common import (
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL,
    SearchHit,
    diversify_hits,
    embed_query,
    format_hits,
    load_json,
    load_jsonl,
    read_query,
    rerank_hits,
)


DEFAULT_DATABASE_URL = "postgresql://hep_rag:hep_rag@localhost:5432/hep_rag"
DEFAULT_TABLE = "rag_chunks"
DEFAULT_RRF_K = 60


def load_main(argv: list[str] | None = None) -> int:
    args = load_parser().parse_args(argv)
    psycopg = import_psycopg()
    database_url = database_url_from_args(args)
    embedding_dir = args.embedding_dir
    embedding_config = load_json(embedding_dir / "config.json")
    rows = load_jsonl(embedding_dir / "rows.jsonl")
    chunks_path = args.chunks or Path(str(embedding_config["chunks_path"]))
    chunks_by_id = load_chunks_by_id(chunks_path)

    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: numpy. Install with `pip install -e '.[pgvector]'`."
        ) from error

    vectors = np.load(embedding_dir / "embeddings.npy").astype("float32", copy=False)
    if len(rows) != vectors.shape[0]:
        raise SystemExit(
            f"Row count mismatch: {len(rows)} rows for {vectors.shape[0]} vectors"
        )

    table = validate_identifier(args.table)
    dimension = int(vectors.shape[1])
    with psycopg.connect(database_url) as conn:
        create_schema(conn, table=table, dimension=dimension, recreate=args.recreate)
        if args.schema_only:
            print(f"Created pgvector schema: {table}")
            return 0

        inserted = 0
        for batch in batched(
            iter_chunk_records(rows, chunks_by_id, vectors),
            args.batch_size,
        ):
            insert_chunk_batch(conn, table, batch)
            inserted += len(batch)
            print(f"Loaded {inserted}/{len(rows)} chunks")

        create_metadata_indexes(conn, table)
        if args.vector_index != "none":
            create_vector_index(conn, table, args.vector_index)

    print(f"Loaded chunks: {inserted}")
    print(f"Table: {table}")
    print(f"Database: {redact_database_url(database_url)}")
    return 0


def load_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load embeddings into Postgres + pgvector.")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing embeddings.npy, rows.jsonl, and config.json.",
    )
    parser.add_argument("--chunks", type=Path, help="Chunks JSONL path.")
    parser.add_argument("--database-url", help="Postgres URL. Defaults to $DATABASE_URL.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop and recreate the destination table before loading.",
    )
    parser.add_argument(
        "--vector-index",
        choices=["none", "hnsw", "ivfflat"],
        default="none",
        help="Vector index to create after loading. Defaults to exact sequential search.",
    )
    return parser


def search_main(argv: list[str] | None = None) -> int:
    args = search_parser().parse_args(argv)
    psycopg = import_psycopg()
    try:
        from psycopg.rows import dict_row
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: psycopg. Install with `pip install -e '.[pgvector]'`."
        ) from error

    query = read_query(args)
    query_vector = None
    if retrieval_uses_vector(args.retrieval):
        embedding_config = load_json(args.embedding_dir / "config.json")
        vector = embed_query(query, args, embedding_config)
        query_vector = vector_to_pgvector(vector[0])
    table = validate_identifier(args.table)
    sql = search_sql(table, retrieval=args.retrieval)
    database_url = database_url_from_args(args)
    retrieval_limit = retrieval_candidate_limit(args)

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        if retrieval_uses_vector(args.retrieval):
            configure_vector_search(conn, exact=args.exact, hnsw_ef_search=args.hnsw_ef_search)
        rows = conn.execute(
            sql,
            search_params(
                query_vector=query_vector,
                query_text=query,
                top_k=retrieval_limit,
                vector_top_k=retrieval_limit,
                bm25_top_k=args.bm25_candidate_k or args.candidate_k,
                rrf_k=args.rrf_k,
            ),
        ).fetchall()

    hits = pg_rows_to_hits(rows)
    if args.rerank:
        hits = rerank_hits(query, hits, args)
    hits = diversify_hits(hits, args.top_k, args.max_chunks_per_paper)
    if args.json:
        from dataclasses import asdict

        print(json.dumps([asdict(hit) for hit in hits], indent=2, sort_keys=True))
    else:
        print(format_hits(hits, max_text_chars=args.max_text_chars))
    return 0


def search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search chunks in Postgres + pgvector.")
    parser.add_argument("query", nargs="?", help="Query text. Omit when using --query-file.")
    parser.add_argument("--query-file", type=Path, help="Read query text from a file.")
    parser.add_argument("--query-vector", type=Path, help="Use a precomputed query vector .npy file.")
    parser.add_argument("--database-url", help="Postgres URL. Defaults to $DATABASE_URL.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing config.json for query model defaults.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of final hits to return.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Number of vector hits to fetch before reranking or hybrid fusion.",
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
        "--max-chunks-per-paper",
        type=int,
        help="Limit the final results to at most this many chunks per paper.",
    )
    parser.add_argument("--max-text-chars", type=int, default=900)
    parser.add_argument("--json", action="store_true")
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
        help="Rerank vector candidates with a cross-encoder reranker.",
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
    parser.add_argument(
        "--embed-in-process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Embed query in this process instead of the default subprocess.",
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "openai-compatible"],
        default="sentence-transformers",
    )
    parser.add_argument("--model", help="Query embedding model. Defaults to embedding config.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--device", help="Torch device for sentence-transformers, e.g. mps/cpu/cuda.")
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
        help="L2-normalize the query embedding before searching.",
    )
    parser.add_argument("--prompt-name", default="query")
    parser.add_argument("--prompt")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key")
    return parser


def retrieval_candidate_limit(args: argparse.Namespace) -> int:
    if args.rerank or args.max_chunks_per_paper:
        return max(args.candidate_k, args.top_k)
    return args.top_k


def retrieval_uses_vector(retrieval: str) -> bool:
    return retrieval in {"vector", "hybrid"}


def search_params(
    query_vector: str | None,
    query_text: str,
    top_k: int,
    vector_top_k: int,
    bm25_top_k: int,
    rrf_k: int,
) -> dict[str, Any]:
    if top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if vector_top_k < 1:
        raise SystemExit("--candidate-k must be at least 1")
    if bm25_top_k < 1:
        raise SystemExit("--bm25-candidate-k must be at least 1")
    if rrf_k < 1:
        raise SystemExit("--rrf-k must be at least 1")
    return {
        "query": query_vector,
        "query_text": query_text,
        "top_k": top_k,
        "vector_top_k": vector_top_k,
        "bm25_top_k": bm25_top_k,
        "rrf_k": rrf_k,
    }


def configure_vector_search(
    conn: Any,
    exact: bool = True,
    hnsw_ef_search: int | None = None,
) -> None:
    if hnsw_ef_search is not None and hnsw_ef_search < 1:
        raise SystemExit("--hnsw-ef-search must be at least 1")
    if exact:
        if hnsw_ef_search is not None:
            raise SystemExit("--hnsw-ef-search only applies with --no-exact")
        conn.execute("SET LOCAL enable_indexscan = off")
        return
    if hnsw_ef_search is not None:
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(hnsw_ef_search)}")


def create_schema(conn: Any, table: str, dimension: int, recreate: bool = False) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if recreate:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            chunk_id TEXT PRIMARY KEY,
            row_index INTEGER UNIQUE NOT NULL,
            paper_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_type TEXT NOT NULL,
            title TEXT,
            authors TEXT,
            year TEXT,
            primary_class TEXT,
            section_path JSONB NOT NULL DEFAULT '[]'::jsonb,
            living_review_categories JSONB NOT NULL DEFAULT '[]'::jsonb,
            token_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            source_url TEXT,
            search_document TEXT NOT NULL DEFAULT '',
            search_vector TSVECTOR NOT NULL DEFAULT to_tsvector('english'::regconfig, ''::text),
            embedding vector({dimension}) NOT NULL
        )
        """
    )
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS search_document TEXT NOT NULL DEFAULT ''")
    conn.execute(
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS search_vector TSVECTOR "
        f"NOT NULL DEFAULT to_tsvector('english'::regconfig, ''::text)"
    )


def create_metadata_indexes(conn: Any, table: str) -> None:
    conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_paper_id_idx ON {table} (paper_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_chunk_type_idx ON {table} (chunk_type)")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_categories_gin_idx "
        f"ON {table} USING gin (living_review_categories)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_search_vector_gin_idx "
        f"ON {table} USING gin (search_vector)"
    )


def create_vector_index(conn: Any, table: str, index_type: str) -> None:
    if index_type == "hnsw":
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw_idx "
            f"ON {table} USING hnsw (embedding vector_ip_ops)"
        )
        return
    if index_type == "ivfflat":
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_ivfflat_idx "
            f"ON {table} USING ivfflat (embedding vector_ip_ops) WITH (lists = 100)"
        )
        return
    raise ValueError(f"Unknown vector index type: {index_type}")


def insert_chunk_batch(conn: Any, table: str, records: list[dict[str, Any]]) -> None:
    sql = f"""
        INSERT INTO {table} (
            chunk_id, row_index, paper_id, chunk_index, chunk_type, title,
            authors, year, primary_class, section_path, living_review_categories,
            token_count, text, source_url, search_document, search_vector, embedding
        )
        VALUES (
            %(chunk_id)s, %(row_index)s, %(paper_id)s, %(chunk_index)s,
            %(chunk_type)s, %(title)s, %(authors)s, %(year)s, %(primary_class)s,
            %(section_path)s::jsonb, %(living_review_categories)s::jsonb,
            %(token_count)s, %(text)s, %(source_url)s, %(search_document)s,
            to_tsvector('english'::regconfig, %(search_document)s),
            %(embedding)s::vector
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            row_index = EXCLUDED.row_index,
            paper_id = EXCLUDED.paper_id,
            chunk_index = EXCLUDED.chunk_index,
            chunk_type = EXCLUDED.chunk_type,
            title = EXCLUDED.title,
            authors = EXCLUDED.authors,
            year = EXCLUDED.year,
            primary_class = EXCLUDED.primary_class,
            section_path = EXCLUDED.section_path,
            living_review_categories = EXCLUDED.living_review_categories,
            token_count = EXCLUDED.token_count,
            text = EXCLUDED.text,
            source_url = EXCLUDED.source_url,
            search_document = EXCLUDED.search_document,
            search_vector = EXCLUDED.search_vector,
            embedding = EXCLUDED.embedding
    """
    with conn.cursor() as cursor:
        cursor.executemany(sql, records)
    conn.commit()


def iter_chunk_records(
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    vectors: Any,
) -> Iterable[dict[str, Any]]:
    for row, vector in zip(rows, vectors):
        chunk_id = str(row["chunk_id"])
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise SystemExit(f"Chunk metadata not found for {chunk_id}")
        yield {
            "chunk_id": chunk_id,
            "row_index": int(row["row_index"]),
            "paper_id": str(row["paper_id"]),
            "chunk_index": int(chunk["chunk_index"]),
            "chunk_type": str(row["chunk_type"]),
            "title": value_or_none(chunk, "title"),
            "authors": value_or_none(chunk, "authors"),
            "year": value_or_none(chunk, "year"),
            "primary_class": value_or_none(chunk, "primary_class"),
            "section_path": json.dumps(chunk.get("section_path") or []),
            "living_review_categories": json.dumps(
                chunk.get("living_review_categories") or []
            ),
            "token_count": int(row["token_count"]),
            "text": str(chunk.get("text") or ""),
            "source_url": value_or_none(chunk, "source_url"),
            "search_document": search_document_text(chunk),
            "embedding": vector_to_pgvector(vector),
        }


def search_document_text(chunk: dict[str, Any]) -> str:
    lines = []
    title = value_or_none(chunk, "title")
    if title:
        lines.append(title)
    section_path = chunk.get("section_path") or []
    if section_path:
        lines.append(" ".join(str(part) for part in section_path))
    categories = chunk.get("living_review_categories") or []
    for category in categories:
        lines.append(" ".join(str(part) for part in category))
    text = str(chunk.get("text") or "").strip()
    if text:
        lines.append(text)
    return "\n".join(lines)


def search_sql(table: str, retrieval: str = "vector") -> str:
    if retrieval == "vector":
        return vector_search_sql(table)
    if retrieval == "bm25":
        return bm25_search_sql(table)
    if retrieval == "hybrid":
        return hybrid_search_sql(table)
    raise ValueError(f"Unknown retrieval mode: {retrieval}")


def chunk_select_columns(alias: str = "chunks") -> str:
    return f"""
            {alias}.chunk_id,
            {alias}.paper_id,
            {alias}.title,
            {alias}.section_path,
            {alias}.living_review_categories,
            {alias}.chunk_type,
            {alias}.token_count,
            {alias}.text,
            {alias}.source_url
    """


def vector_search_sql(table: str) -> str:
    return f"""
        WITH query AS (SELECT %(query)s::vector AS embedding)
        SELECT
            {chunk_select_columns("chunks")},
            -(chunks.embedding <#> query.embedding) AS score,
            -(chunks.embedding <#> query.embedding) AS vector_score,
            NULL::double precision AS lexical_score,
            NULL::double precision AS rrf_score
        FROM {table} AS chunks, query
        ORDER BY chunks.embedding <#> query.embedding
        LIMIT %(top_k)s
    """


def bm25_search_sql(table: str) -> str:
    return f"""
        WITH query AS (
            SELECT websearch_to_tsquery('english'::regconfig, %(query_text)s) AS terms
        )
        SELECT
            {chunk_select_columns("chunks")},
            ts_rank_cd(chunks.search_vector, query.terms) AS score,
            NULL::double precision AS vector_score,
            ts_rank_cd(chunks.search_vector, query.terms) AS lexical_score,
            NULL::double precision AS rrf_score
        FROM {table} AS chunks, query
        WHERE query.terms @@ chunks.search_vector
        ORDER BY score DESC, chunks.chunk_id
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
            SELECT
                chunks.chunk_id,
                -(chunks.embedding <#> vector_query.embedding) AS vector_score,
                row_number() OVER (ORDER BY chunks.embedding <#> vector_query.embedding) AS vector_rank
            FROM {table} AS chunks, vector_query
            ORDER BY chunks.embedding <#> vector_query.embedding
            LIMIT %(vector_top_k)s
        ),
        lexical_hits AS (
            SELECT
                chunks.chunk_id,
                ts_rank_cd(chunks.search_vector, lexical_query.terms) AS lexical_score,
                row_number() OVER (
                    ORDER BY ts_rank_cd(chunks.search_vector, lexical_query.terms) DESC, chunks.chunk_id
                ) AS lexical_rank
            FROM {table} AS chunks, lexical_query
            WHERE lexical_query.terms @@ chunks.search_vector
            ORDER BY lexical_score DESC, chunks.chunk_id
            LIMIT %(bm25_top_k)s
        ),
        combined AS (
            SELECT chunk_id FROM vector_hits
            UNION
            SELECT chunk_id FROM lexical_hits
        )
        SELECT
            {chunk_select_columns("chunks")},
            (
                COALESCE(1.0 / (%(rrf_k)s + vector_hits.vector_rank), 0.0)
                + COALESCE(1.0 / (%(rrf_k)s + lexical_hits.lexical_rank), 0.0)
            ) AS score,
            vector_hits.vector_score AS vector_score,
            lexical_hits.lexical_score AS lexical_score,
            (
                COALESCE(1.0 / (%(rrf_k)s + vector_hits.vector_rank), 0.0)
                + COALESCE(1.0 / (%(rrf_k)s + lexical_hits.lexical_rank), 0.0)
            ) AS rrf_score
        FROM combined
        JOIN {table} AS chunks USING (chunk_id)
        LEFT JOIN vector_hits USING (chunk_id)
        LEFT JOIN lexical_hits USING (chunk_id)
        ORDER BY score DESC, chunks.chunk_id
        LIMIT %(top_k)s
    """


def pg_rows_to_hits(rows: list[dict[str, Any]]) -> list[SearchHit]:
    hits = []
    for rank, row in enumerate(rows, start=1):
        hits.append(
            SearchHit(
                rank=rank,
                score=float(row["score"]),
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
                vector_score=row_score(row, "vector_score", fallback_key="score"),
                lexical_score=(
                    float(row["lexical_score"]) if row.get("lexical_score") is not None else None
                ),
                rrf_score=float(row["rrf_score"]) if row.get("rrf_score") is not None else None,
            )
        )
    return hits


def load_chunks_by_id(path: Path) -> dict[str, dict[str, Any]]:
    chunks = {}
    with path.open(encoding="utf-8") as chunks_file:
        for line in chunks_file:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks[str(chunk["chunk_id"])] = chunk
    return chunks


def row_score(
    row: dict[str, Any],
    key: str,
    fallback_key: str | None = None,
) -> float | None:
    if key in row:
        value = row.get(key)
    elif fallback_key is not None:
        value = row.get(fallback_key)
    else:
        value = None
    return float(value) if value is not None else None


def vector_to_pgvector(vector: Any) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in vector) + "]"


def batched(records: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_identifier(identifier: str) -> str:
    if (
        not identifier
        or not identifier.replace("_", "").isalnum()
        or identifier[0].isdigit()
    ):
        raise SystemExit(f"Unsafe SQL identifier: {identifier}")
    return identifier


def database_url_from_args(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def redact_database_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    _credentials, host = rest.rsplit("@", 1)
    return f"{scheme}://***@{host}"


def import_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: psycopg. Install with `pip install -e '.[pgvector]'`."
        ) from error
    return psycopg


def value_or_none(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    return str(value)
