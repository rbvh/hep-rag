import torch

from hep_rag.embed import build_embeddings
from hep_rag.embed.build_embeddings import (
    embedding_row,
    embedding_text,
    safe_name,
    sentence_transformer_model_kwargs,
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


def test_sentence_transformer_model_kwargs_maps_torch_dtype() -> None:
    assert sentence_transformer_model_kwargs("auto") == {}
    assert sentence_transformer_model_kwargs("bfloat16") == {"torch_dtype": torch.bfloat16}


def test_build_embeddings_defaults_are_memory_conservative() -> None:
    args = build_embeddings.build_parser().parse_args([])

    assert args.batch_size == 1
    assert args.max_seq_length == 2048


def test_query_parser_defaults_to_qwen_query_prompt_name() -> None:
    args = build_query_parser().parse_args(["what is TMD factorization?"])

    assert args.prompt_name == "query"
    assert args.max_seq_length == 2048
    assert query_text(args) == "what is TMD factorization?"


def test_openai_compatible_query_can_prepend_explicit_prompt() -> None:
    args = build_query_parser().parse_args(
        [
            "--backend",
            "openai-compatible",
            "--prompt",
            "Instruct: retrieve relevant HEP passages\nQuery: ",
            "normalizing flows",
        ]
    )

    assert query_text(args) == "Instruct: retrieve relevant HEP passages\nQuery: normalizing flows"
