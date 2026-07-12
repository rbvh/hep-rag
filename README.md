# HEP RAG Lab

An evaluated retrieval-augmented generation system over the papers catalogued by
the Living Review of Machine Learning for Particle Physics. The corpus is built
from arXiv LaTeX sources, searched through Postgres and pgvector, reranked with
Qwen3-Reranker, and answered with citations by a small local language model.

## What Is Included

The repository contains four internal layers:

1. LaTeX source ingestion and paragraph-first chunking.
2. Qwen3 document and query embeddings.
3. Exact vector, PostgreSQL full-text, and hybrid retrieval.
4. Paper-level evaluation and citation-grounded answer generation.

The installed interface intentionally has only three commands:

- `hep-rag-pg-load` loads prepared chunks and vectors into Postgres.
- `hep-rag-eval-retrieval` evaluates paper-level retrieval quality.
- `hep-rag-ask` retrieves evidence and generates a cited answer.

Corpus acquisition, parsing, and embedding remain importable internal tooling so
the dataset can be rebuilt, but they are not part of the application interface.

## Setup

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate hep-rag-lab
```

Start Postgres and the three local model services:

```bash
docker compose up -d --wait postgres vllm-embedder vllm-reranker vllm-qa
export DATABASE_URL=postgresql://hep_rag:hep_rag@localhost:5432/hep_rag
```

The services expose:

- `http://localhost:8001/v1/embeddings`: `Qwen/Qwen3-Embedding-0.6B`
- `http://localhost:8002/rerank`: `Qwen/Qwen3-Reranker-0.6B`
- `http://localhost:8003/v1/chat/completions`: `Qwen/Qwen3.5-0.8B`

Service locations and credentials can be overridden with:

```bash
export HEP_RAG_EMBEDDING_URL=http://localhost:8001/v1
export HEP_RAG_RERANK_URL=http://localhost:8002
export HEP_RAG_QA_URL=http://localhost:8003/v1
```

The corresponding API-key variables are `HEP_RAG_EMBEDDING_API_KEY`,
`HEP_RAG_RERANK_API_KEY`, and `HEP_RAG_QA_API_KEY`.

## Load The Corpus

Prepared artifacts live under:

```text
data/processed/chunks.jsonl
data/embeddings/Qwen-Qwen3-Embedding-0.6B/
  config.json
  embeddings.npy
  rows.jsonl
```

Load them into the `rag_chunks` table:

```bash
hep-rag-pg-load \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --recreate
```

The loader stores chunk text, paper metadata, Living Review categories, dense
vectors, and a PostgreSQL full-text search vector. Exact vector search is the
default. `--vector-index hnsw` or `--vector-index ivfflat` is available for
larger corpora.

## Evaluate Retrieval

Run the default exact vector benchmark:

```bash
hep-rag-eval-retrieval \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --candidate-k 300
```

Compare lexical or hybrid retrieval:

```bash
hep-rag-eval-retrieval --retrieval lexical
hep-rag-eval-retrieval --retrieval hybrid
```

Add paper-level reranking:

```bash
hep-rag-eval-retrieval --retrieval hybrid --rerank
```

Results are written to `data/eval/reports/retrieval_metrics.json` and
`data/eval/reports/retrieval_metrics.md`. Metrics include recall, precision,
hit rate, reciprocal rank, and average precision at the paper level.

The lexical mode uses PostgreSQL `websearch_to_tsquery` and `ts_rank_cd`. It is
a full-text ranking baseline, not an implementation of Okapi BM25. Hybrid mode
combines lexical and dense rankings with Reciprocal Rank Fusion.

## Ask A Question

```bash
hep-rag-ask "How are normalizing flows used for collider event generation?"
```

The question path is deliberately opinionated: hybrid retrieval, exact vector
search, Qwen reranking, at most two passages per paper, and up to six final
passages. Every selected paper contributes its abstract to the answer context.

Use `--show-context` to inspect the packed evidence or `--json` for structured
output:

```bash
hep-rag-ask --show-context "Can ML reduce negative weighted events?"
hep-rag-ask --json "Can matrix elements be evaluated on a GPU?"
```

## Retrieval Representation

Dense and lexical retrieval use one canonical document renderer containing the
paper title, section path, Living Review categories, and chunk text. Dense
documents retain field labels such as `Title:` and `Section:`; lexical documents
contain the same values without labels for PostgreSQL tokenization.

Retrieval results preserve separate vector, lexical, RRF, and reranker scores.
The final score records only the ranking stage that produced the final order.

## Development

Run the checks with:

```bash
ruff format --check .
ruff check .
pytest -q
```

The LaTeX pipeline is split by phase:

- `ingest/latex_source.py` locates source files, expands inputs, and removes comments.
- `ingest/latex_text.py` extracts captions and normalizes paragraphs and equations.
- `ingest/source_parser.py` tracks sections and assembles parsed papers.

Generated corpora, embeddings, databases, and evaluation reports are excluded
from version control. The hand-reviewed retrieval query set remains under
`data/eval/`.
