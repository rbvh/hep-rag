import pytest

from hep_rag.search.pgvector_search import (
    bm25_search_sql,
    batched,
    configure_vector_search,
    hybrid_search_sql,
    load_parser,
    pg_rows_to_hits,
    redact_database_url,
    retrieval_candidate_limit,
    retrieval_uses_vector,
    search_document_text,
    search_parser,
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
    with pytest.raises(SystemExit):
        validate_identifier("rag_chunks; drop table rag_chunks")
    with pytest.raises(SystemExit):
        validate_identifier("2bad")


def test_search_sql_qualifies_embedding_column() -> None:
    sql = search_sql("rag_chunks")

    assert "chunks.embedding <#> query.embedding" in sql
    assert "FROM rag_chunks AS chunks" in sql


def test_search_sql_selects_retrieval_mode() -> None:
    assert search_sql("rag_chunks", retrieval="vector") == vector_search_sql("rag_chunks")
    assert search_sql("rag_chunks", retrieval="bm25") == bm25_search_sql("rag_chunks")
    assert search_sql("rag_chunks", retrieval="hybrid") == hybrid_search_sql("rag_chunks")
    with pytest.raises(ValueError):
        search_sql("rag_chunks", retrieval="bad")


def test_bm25_search_sql_uses_search_vector() -> None:
    sql = bm25_search_sql("rag_chunks")

    assert "websearch_to_tsquery" in sql
    assert "chunks.search_vector" in sql
    assert "ts_rank_cd" in sql


def test_hybrid_search_sql_uses_rrf() -> None:
    sql = hybrid_search_sql("rag_chunks")

    assert "vector_hits" in sql
    assert "lexical_hits" in sql
    assert "%(rrf_k)s" in sql


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
            "lexical_score": 0.3,
            "rrf_score": 0.1,
        }
    ]

    hits = pg_rows_to_hits(rows)

    assert hits[0].rank == 1
    assert hits[0].score == 0.8
    assert hits[0].vector_score == 0.8
    assert hits[0].lexical_score == 0.3
    assert hits[0].rrf_score == 0.1
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


def test_load_parser_defaults_to_no_vector_index() -> None:
    args = load_parser().parse_args([])

    assert args.vector_index == "none"


def test_search_parser_defaults_to_exact_search() -> None:
    args = search_parser().parse_args(["test query"])

    assert args.exact is True
    assert args.retrieval == "vector"
    assert args.rrf_k == 60


def test_configure_vector_search_disables_index_scan_for_exact() -> None:
    class Conn:
        def __init__(self) -> None:
            self.commands = []

        def execute(self, sql: str) -> None:
            self.commands.append(sql)

    conn = Conn()
    configure_vector_search(conn, exact=True)

    assert conn.commands == ["SET LOCAL enable_indexscan = off"]


def test_configure_vector_search_sets_hnsw_ef_search_for_approximate() -> None:
    class Conn:
        def __init__(self) -> None:
            self.commands = []

        def execute(self, sql: str) -> None:
            self.commands.append(sql)

    conn = Conn()
    configure_vector_search(conn, exact=False, hnsw_ef_search=500)

    assert conn.commands == ["SET LOCAL hnsw.ef_search = 500"]


def test_configure_vector_search_rejects_hnsw_ef_search_in_exact_mode() -> None:
    class Conn:
        def execute(self, sql: str) -> None:
            raise AssertionError("Should not execute SQL")

    with pytest.raises(SystemExit):
        configure_vector_search(Conn(), exact=True, hnsw_ef_search=500)


def test_retrieval_uses_vector() -> None:
    assert retrieval_uses_vector("vector")
    assert retrieval_uses_vector("hybrid")
    assert not retrieval_uses_vector("bm25")


def test_search_params_validates_candidates() -> None:
    params = search_params(
        query_vector="[1,2]",
        query_text="query",
        top_k=10,
        vector_top_k=20,
        bm25_top_k=30,
        rrf_k=60,
    )

    assert params["query"] == "[1,2]"
    assert params["query_text"] == "query"
    assert params["vector_top_k"] == 20
    assert params["bm25_top_k"] == 30
    assert params["rrf_k"] == 60

    with pytest.raises(SystemExit):
        search_params("[1,2]", "query", top_k=0, vector_top_k=20, bm25_top_k=30, rrf_k=60)
    with pytest.raises(SystemExit):
        search_params("[1,2]", "query", top_k=10, vector_top_k=0, bm25_top_k=30, rrf_k=60)
    with pytest.raises(SystemExit):
        search_params("[1,2]", "query", top_k=10, vector_top_k=20, bm25_top_k=0, rrf_k=60)
    with pytest.raises(SystemExit):
        search_params("[1,2]", "query", top_k=10, vector_top_k=20, bm25_top_k=30, rrf_k=0)


def test_search_document_text_includes_metadata_and_chunk_text() -> None:
    text = search_document_text(
        {
            "title": "Jet Tagging",
            "section_path": ["Methods", "Transformers"],
            "living_review_categories": [["Classification", "Targets"]],
            "text": "Particle transformer details.",
        }
    )

    assert "Jet Tagging" in text
    assert "Methods Transformers" in text
    assert "Classification Targets" in text
    assert "Particle transformer details." in text
