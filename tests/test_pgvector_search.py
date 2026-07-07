import pytest

from hep_rag.search.pgvector_search import (
    batched,
    pg_rows_to_hits,
    redact_database_url,
    retrieval_candidate_limit,
    search_sql,
    validate_identifier,
    vector_to_pgvector,
)


def test_vector_to_pgvector_formats_literal() -> None:
    assert vector_to_pgvector([1, 0.5, -0.25]) == "[1,0.5,-0.25]"


def test_validate_identifier_rejects_unsafe_table_names() -> None:
    assert validate_identifier("rag_chunks_2") == "rag_chunks_2"
    with pytest.raises(SystemExit):
        validate_identifier("rag_chunks; drop table rag_chunks")
    with pytest.raises(SystemExit):
        validate_identifier("2bad")


def test_search_sql_qualifies_embedding_column() -> None:
    sql = search_sql("rag_chunks")

    assert "chunks.embedding <#> query.embedding" in sql
    assert "FROM rag_chunks AS chunks" in sql


def test_pg_rows_to_hits_converts_database_rows() -> None:
    rows = [
        {
            "chunk_id": "2501.00001:chunk:00000:abc",
            "paper_id": "2501.00001",
            "title": "A Test Paper",
            "section_path": ["Methods"],
            "living_review_categories": [["Regression", "Lattice Gauge Theory"]],
            "chunk_type": "paragraph",
            "token_count": 123,
            "text": "A chunk about lattice gauge theory.",
            "source_url": "https://arxiv.org/e-print/2501.00001",
            "score": 0.8,
        }
    ]

    hits = pg_rows_to_hits(rows)

    assert hits[0].rank == 1
    assert hits[0].score == 0.8
    assert hits[0].vector_score == 0.8
    assert hits[0].title == "A Test Paper"
    assert hits[0].section_path == ["Methods"]
    assert hits[0].living_review_categories == [["Regression", "Lattice Gauge Theory"]]


def test_batched_groups_records() -> None:
    assert list(batched([{"i": 1}, {"i": 2}, {"i": 3}], 2)) == [
        [{"i": 1}, {"i": 2}],
        [{"i": 3}],
    ]


def test_redact_database_url_hides_credentials() -> None:
    assert (
        redact_database_url("postgresql://user:secret@localhost:5432/db")
        == "postgresql://***@localhost:5432/db"
    )


def test_retrieval_candidate_limit_expands_when_diversifying() -> None:
    class Args:
        top_k = 10
        candidate_k = 50
        rerank = False
        max_chunks_per_paper = 1

    assert retrieval_candidate_limit(Args()) == 50


def test_retrieval_candidate_limit_uses_top_k_for_plain_search() -> None:
    class Args:
        top_k = 10
        candidate_k = 50
        rerank = False
        max_chunks_per_paper = None

    assert retrieval_candidate_limit(Args()) == 10
