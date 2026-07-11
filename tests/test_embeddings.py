from hep_rag.embed import build_embeddings
from hep_rag.embed.build_embeddings import (
    embedding_row,
    embedding_input_text,
    embedding_text,
    safe_name,
)
from hep_rag.embed.embed_query import build_parser as build_query_parser
from hep_rag.embed.embed_query import query_text


def test_embedding_text_can_include_metadata_context() -> None:
    chunk = {
        "title": "A Test Paper",
        "section_path": ["Methods", "Results"],
        "living_review_categories": [["Regression", "PDFs"]],
        "text": "The actual chunk text with $q_T$.",
    }

    text = embedding_text(chunk, include_metadata_context=True)

    assert text.startswith("Title: A Test Paper")
    assert "Section: Methods > Results" in text
    assert "Living Review categories: Regression > PDFs" in text
    assert text.endswith("The actual chunk text with $q_T$.")


def test_embedding_text_can_be_chunk_text_only() -> None:
    chunk = {"text": "Only the chunk."}

    assert embedding_text(chunk, include_metadata_context=False) == "Only the chunk."


def test_embedding_row_tracks_chunk_identity_and_text_hash() -> None:
    chunk = {
        "chunk_id": "2501.00001:chunk:00001:abc",
        "paper_id": "2501.00001",
        "chunk_index": 1,
        "chunk_type": "paragraph",
        "section_path": ["Introduction"],
        "token_count": 10,
    }

    row = embedding_row(0, chunk, "embedded text")

    assert row.row_index == 0
    assert row.chunk_id == "2501.00001:chunk:00001:abc"
    assert row.section_path == ["Introduction"]
    assert len(row.text_sha1) == 40


def test_safe_name_makes_model_name_path_friendly() -> None:
    assert safe_name("Qwen/Qwen3-Embedding-0.6B") == "Qwen-Qwen3-Embedding-0.6B"


def test_build_embeddings_defaults_to_openai_compatible_backend() -> None:
    args = build_embeddings.build_parser().parse_args([])

    assert args.backend == "openai-compatible"
    assert args.batch_size == 1
    assert args.truncate_prompt_tokens is None


def test_build_embeddings_accepts_server_side_truncation() -> None:
    args = build_embeddings.build_parser().parse_args(
        ["--truncate-prompt-tokens", "4096"]
    )

    assert args.truncate_prompt_tokens == 4096


def test_query_parser_defaults_to_openai_compatible_backend() -> None:
    args = build_query_parser().parse_args(["what is TMD factorization?"])

    assert args.backend == "openai-compatible"
    assert query_text(args) == "what is TMD factorization?"


def test_query_text_keeps_prompt_separate_from_raw_query() -> None:
    args = build_query_parser().parse_args(
        [
            "--prompt",
            "Instruct: retrieve relevant HEP passages\nQuery: ",
            "normalizing flows",
        ]
    )

    assert query_text(args) == "normalizing flows"
    assert (
        embedding_input_text(query_text(args), args.prompt)
        == "Instruct: retrieve relevant HEP passages\nQuery: normalizing flows"
    )
