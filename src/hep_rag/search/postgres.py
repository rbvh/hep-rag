"""Create and populate the Postgres retrieval store."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hep_rag.search.common import load_json, load_jsonl
from hep_rag.search.config import DEFAULT_EMBEDDING_DIR, DEFAULT_TABLE, database_url
from hep_rag.search.documents import retrieval_document_text
from hep_rag.search.retrieval import import_psycopg, validate_identifier, vector_to_pgvector


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedding_config = load_json(args.embedding_dir / "config.json")
    rows = load_jsonl(args.embedding_dir / "rows.jsonl")
    chunks_path = args.chunks or Path(str(embedding_config["chunks_path"]))
    chunks_by_id = load_chunks_by_id(chunks_path)

    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("Install Postgres support with `pip install -e '.[pgvector]'`.") from error

    vectors = np.load(args.embedding_dir / "embeddings.npy").astype("float32", copy=False)
    if len(rows) != vectors.shape[0]:
        raise SystemExit(f"Row count mismatch: {len(rows)} rows for {vectors.shape[0]} vectors")

    psycopg = import_psycopg()
    table = validate_identifier(args.table)
    url = database_url(args.database_url)
    with psycopg.connect(url) as conn:
        create_schema(conn, table, int(vectors.shape[1]), recreate=args.recreate)
        inserted = 0
        for batch in batched(iter_chunk_records(rows, chunks_by_id, vectors), args.batch_size):
            insert_chunk_batch(conn, table, batch)
            inserted += len(batch)
            print(f"Loaded {inserted}/{len(rows)} chunks")
        create_metadata_indexes(conn, table)
        if args.vector_index != "none":
            create_vector_index(conn, table, args.vector_index)

    print(f"Loaded chunks: {inserted}")
    print(f"Table: {table}")
    print(f"Database: {redact_database_url(url)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--database-url", help="Defaults to $DATABASE_URL.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--vector-index", choices=["none", "hnsw", "ivfflat"], default="none")
    return parser


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
            search_document TEXT NOT NULL,
            search_vector TSVECTOR NOT NULL,
            embedding vector({dimension}) NOT NULL
        )
        """
    )
    migrate_schema(conn, table)


def migrate_schema(conn: Any, table: str) -> None:
    """Apply the small set of additive migrations supported by this prototype."""

    conn.execute(
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS search_document TEXT NOT NULL DEFAULT ''"
    )
    conn.execute(
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS search_vector TSVECTOR "
        "NOT NULL DEFAULT to_tsvector('english'::regconfig, ''::text)"
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
    elif index_type == "ivfflat":
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_ivfflat_idx "
            f"ON {table} USING ivfflat (embedding vector_ip_ops) WITH (lists = 100)"
        )
    else:
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
    for row, vector in zip(rows, vectors, strict=True):
        chunk_id = str(row["chunk_id"])
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Chunk metadata not found for {chunk_id}")
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
            "living_review_categories": json.dumps(chunk.get("living_review_categories") or []),
            "token_count": int(row["token_count"]),
            "text": str(chunk.get("text") or ""),
            "source_url": value_or_none(chunk, "source_url"),
            "search_document": retrieval_document_text(chunk, style="lexical"),
            "embedding": vector_to_pgvector(vector),
        }


def load_chunks_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(chunk["chunk_id"]): chunk for chunk in load_jsonl(path)}


def batched(records: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def redact_database_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    _credentials, host = rest.rsplit("@", 1)
    return f"{scheme}://***@{host}"


def value_or_none(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
