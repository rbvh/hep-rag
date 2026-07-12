# HEP RAG Lab

An evaluated RAG system for papers in the Living Review of Machine Learning for
Particle Physics. It uses Postgres with pgvector, Qwen embeddings and reranking,
and a small local language model for citation-grounded answers.

## Setup

```bash
conda env create -f environment.yml
conda activate hep-rag-lab
docker compose up -d --wait postgres vllm-embedder vllm-reranker vllm-qa
export DATABASE_URL=postgresql://hep_rag:hep_rag@localhost:5432/hep_rag
```

Prepared chunks and embeddings are expected under `data/processed/` and
`data/embeddings/`. Corpus preparation remains internal tooling; the installed
application exposes three commands.

## Load

Load prepared chunks and vectors into Postgres:

```bash
hep-rag-pg-load \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --recreate
```

Exact vector search is the default. The loader can optionally create an HNSW or
IVFFlat index with `--vector-index`.

## Evaluate

Run the paper-level retrieval benchmark:

```bash
hep-rag-eval-retrieval \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --candidate-k 300
```

Alternative configurations include:

```bash
hep-rag-eval-retrieval --retrieval lexical
hep-rag-eval-retrieval --retrieval hybrid --rerank
```

Reports are written to `data/eval/reports/`. Lexical retrieval uses PostgreSQL
full-text ranking; hybrid retrieval combines lexical and vector ranks with RRF.

## Ask

Retrieve evidence, rerank it, and generate a cited answer:

```bash
hep-rag-ask "How are normalizing flows used for collider event generation?"
```

Use `--show-context` to inspect the evidence or `--json` for structured output.
The default answer path uses hybrid retrieval, exact vector search, reranking,
and at most two passages per paper.

## Configuration

The Docker services default to:

- Embeddings: `http://localhost:8001/v1`
- Reranking: `http://localhost:8002`
- Answer generation: `http://localhost:8003/v1`

Override them with `HEP_RAG_EMBEDDING_URL`, `HEP_RAG_RERANK_URL`, and
`HEP_RAG_QA_URL`. Matching `*_API_KEY` variables are supported.

## Development

```bash
ruff format --check .
ruff check .
pytest -q
```
