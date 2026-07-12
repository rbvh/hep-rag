import pytest

from hep_rag.search.config import RetrievalConfig
from hep_rag.search.documents import retrieval_document_text
from hep_rag.search.postgres import batched, build_parser, redact_database_url
from hep_rag.search.retrieval import (
    configure_vector_search,
    hybrid_search_sql,
    lexical_search_sql,
    pg_rows_to_hits,
    retrieval_uses_vector,
    search_params,
    search_sql,
    validate_identifier,
    vector_search_sql,
    vector_to_pgvector,
)


def test_vector_to_pgvector_formats_literal() -> None:
    assert vector_to_pgvector([1, 0.5, -0.25]) == "[1,0.5,-0.25]"


def test_validate_identifier_rejects_unsafe_table_names() -> None:
    assert validate_identifier("rag_chunks_2") == "rag_chunks_2"
    with pytest.raises(ValueError):
        validate_identifier("rag_chunks; drop table rag_chunks")
    with pytest.raises(ValueError):
        validate_identifier("2bad")


def test_search_sql_selects_retrieval_mode() -> None:
    assert search_sql("rag_chunks", "vector") == vector_search_sql("rag_chunks")
    assert search_sql("rag_chunks", "lexical") == lexical_search_sql("rag_chunks")
    assert search_sql("rag_chunks", "hybrid") == hybrid_search_sql("rag_chunks")


def test_lexical_search_uses_postgres_full_text_ranking() -> None:
    sql = lexical_search_sql("rag_chunks")

    assert "websearch_to_tsquery" in sql
    assert "chunks.search_vector" in sql
    assert "ts_rank_cd" in sql


def test_hybrid_search_uses_candidate_pools_and_rrf() -> None:
    sql = hybrid_search_sql("rag_chunks")

    assert "LIMIT %(vector_top_k)s" in sql
    assert "LIMIT %(lexical_top_k)s" in sql
    assert "%(rrf_k)s" in sql


def test_pg_rows_to_hits_preserves_score_provenance() -> None:
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
            "source_url": None,
            "final_score": 0.3,
            "vector_score": None,
            "lexical_score": 0.3,
            "rrf_score": None,
        }
    ]

    hit = pg_rows_to_hits(rows)[0]

    assert hit.final_score == 0.3
    assert hit.vector_score is None
    assert hit.lexical_score == 0.3


def test_search_params_always_uses_candidate_count_for_first_stage() -> None:
    config = RetrievalConfig(
        mode="hybrid",
        candidate_k=50,
        lexical_candidate_k=75,
        final_k=10,
    )

    params = search_params("[1,2]", "query", config)

    assert params["top_k"] == 50
    assert params["vector_top_k"] == 50
    assert params["lexical_top_k"] == 75


def test_configure_vector_search_disables_index_scan_for_exact() -> None:
    class Conn:
        def __init__(self) -> None:
            self.commands = []

        def execute(self, sql: str) -> None:
            self.commands.append(sql)

    conn = Conn()
    configure_vector_search(conn, RetrievalConfig(exact=True))

    assert conn.commands == ["SET LOCAL enable_indexscan = off"]


def test_retrieval_uses_vector() -> None:
    assert retrieval_uses_vector("vector")
    assert retrieval_uses_vector("hybrid")
    assert not retrieval_uses_vector("lexical")


def test_loader_defaults_to_exact_search_without_vector_index() -> None:
    assert build_parser().parse_args([]).vector_index == "none"


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


def test_document_renderer_uses_same_fields_for_dense_and_lexical() -> None:
    chunk = {
        "title": "Jet Tagging",
        "section_path": ["Methods", "Transformers"],
        "living_review_categories": [["Classification", "Targets"]],
        "text": "Particle transformer details.",
    }

    dense = retrieval_document_text(chunk, "dense")
    lexical = retrieval_document_text(chunk, "lexical")

    for expected in ("Jet Tagging", "Methods", "Transformers", "Classification", "Targets"):
        assert expected in dense
        assert expected in lexical
