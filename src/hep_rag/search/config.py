"""Typed configuration shared by retrieval, evaluation, and QA."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hep_rag.embed.build_embeddings import DEFAULT_BASE_URL, DEFAULT_MODEL, default_out_dir

RetrievalMode = Literal["vector", "lexical", "hybrid"]

DEFAULT_DATABASE_URL = "postgresql://hep_rag:hep_rag@localhost:5432/hep_rag"
DEFAULT_TABLE = "rag_chunks"
DEFAULT_RRF_K = 60
DEFAULT_EMBEDDING_DIR = default_out_dir(DEFAULT_MODEL)
DEFAULT_EMBEDDING_BASE_URL = DEFAULT_BASE_URL
DEFAULT_QUERY_PROMPT = (
    "Instruct: Given a physics literature search query, retrieve relevant "
    "passages that answer the query\nQuery: "
)
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_INSTRUCTION = (
    "Given a physics literature search query, retrieve relevant passages that answer the query"
)
DEFAULT_RERANK_BASE_URL = "http://localhost:8002"


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    api_key: str | None = None
    prompt: str = DEFAULT_QUERY_PROMPT
    batch_size: int = 1
    normalize: bool = True

    @classmethod
    def from_artifacts(cls, embedding_dir: Path) -> EmbeddingConfig:
        import json

        path = embedding_dir / "config.json"
        artifact_config = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model=str(artifact_config.get("model") or DEFAULT_MODEL),
            base_url=os.environ.get("HEP_RAG_EMBEDDING_URL", DEFAULT_EMBEDDING_BASE_URL),
            api_key=os.environ.get("HEP_RAG_EMBEDDING_API_KEY"),
        )


@dataclass(frozen=True)
class RerankConfig:
    model: str = DEFAULT_RERANK_MODEL
    base_url: str = DEFAULT_RERANK_BASE_URL
    api_key: str | None = None
    endpoint: str = "/rerank"
    instruction: str = DEFAULT_RERANK_INSTRUCTION
    timeout: float = 120.0
    max_length: int | None = None

    @classmethod
    def from_environment(cls) -> RerankConfig:
        return cls(
            model=os.environ.get("HEP_RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            base_url=os.environ.get("HEP_RAG_RERANK_URL", DEFAULT_RERANK_BASE_URL),
            api_key=os.environ.get("HEP_RAG_RERANK_API_KEY"),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = "vector"
    candidate_k: int = 50
    lexical_candidate_k: int | None = None
    final_k: int = 10
    max_chunks_per_paper: int | None = None
    rrf_k: int = DEFAULT_RRF_K
    exact: bool = True
    hnsw_ef_search: int | None = None

    @property
    def lexical_k(self) -> int:
        return self.lexical_candidate_k or self.candidate_k


def database_url(explicit: str | None = None) -> str:
    return explicit or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
