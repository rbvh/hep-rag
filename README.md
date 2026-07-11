# HEP RAG Lab

This repository is the start of an evaluated RAG system for high-energy physics
papers. The first implemented step acquires arXiv source packages for the
papers listed in the HEPML Living Review bibliography.

## Setup

```bash
conda env create -f environment.yml
conda activate hep-rag-lab
```

## Living Review Source Download

Preview the current Living Review bibliography:

```bash
hep-rag-living-review-sources --dry-run --preview 20
```

Download a small smoke-test sample:

```bash
hep-rag-living-review-sources --limit 5 --sleep 3
```

Download all arXiv-backed entries:

```bash
hep-rag-living-review-sources --sleep 3
```

Outputs are written under `data/raw/living_review/`:

- `HEPML.bib`: the fetched bibliography snapshot.
- `HEPML.tex`: the fetched Living Review source snapshot used for category
  assignments.
- `living_review_categories.json`: category paths and category descriptions
  parsed from the Living Review topic tree.
- `manifest.jsonl`: one record per attempted paper.
- `summary.json`: status counts for the run.
- `sources/<arxiv_id>/`: raw arXiv source package plus extracted files when
  LaTeX source is available.

Each manifest row includes `living_review_categories`, a list of category paths
assigned to that paper. Papers may have multiple category paths.

Before doing a large crawl, replace the default `--user-agent` contact email.

## First-Pass Chunking

Build paragraph-first chunks from downloaded LaTeX sources:

```bash
hep-rag-build-chunks --limit 20
```

Outputs are written under `data/processed/`:

- `papers.jsonl`: one parsed-paper metadata row per paper.
- `chunks.jsonl`: one section-aware chunk per parsed paragraph or caption, with
  display equations kept inside the surrounding paragraph.
- `parse_errors.jsonl`: non-fatal source parsing failures.
- `inspect/<arxiv_id>.md`: human-readable chunks for manual source comparison.

To inspect one paper against its source:

```bash
hep-rag-build-chunks --arxiv-id 2604.13157 --inspect-limit 1
```

Then compare the `Main TeX` path listed in
`data/processed/inspect/2604.13157.md` with the extracted chunks below it.

## Embeddings

Install embedding dependencies inside the conda environment:

```bash
pip install -e ".[embeddings]"
```

For development and tests, install:

```bash
pip install -e ".[embeddings,pgvector,dev]"
```

Build local Qwen embeddings for the current chunks:

```bash
hep-rag-build-embeddings \
  --model Qwen/Qwen3-Embedding-0.6B \
  --backend sentence-transformers \
  --device mps \
  --torch-dtype float16 \
  --batch-size 1 \
  --max-seq-length 2048
```

The chunk embedding command treats chunks as retrieval documents, so it does
not pass `prompt_name="query"` by default. Qwen recommends query prompting on
the query side, not for document embeddings.

`--torch-dtype auto` is the default and leaves dtype selection to
SentenceTransformers/Transformers. Use `--torch-dtype bfloat16` or
`--torch-dtype float16` to force a lower-precision model load.
The command defaults to `--batch-size 1 --max-seq-length 2048`, which is
intentionally conservative for long-context Qwen models on laptop hardware.
Raise the batch size only after a small run succeeds.

Outputs are written under `data/embeddings/<model-name>/`:

- `embeddings.npy`: float32 matrix, one row per embedded chunk.
- `rows.jsonl`: row-to-chunk mapping aligned with `embeddings.npy`.
- `config.json`: model, backend, input file, dimensions, and run settings.

## Postgres + pgvector

For a deployable vector-database version, start a local pgvector Postgres:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://hep_rag:hep_rag@localhost:5432/hep_rag
```

Load the saved chunks and embeddings:

```bash
hep-rag-pg-load \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --recreate
```

This creates a `rag_chunks` table with chunk metadata, text, categories, and a
`vector(1024)` embedding column, plus a Postgres full-text `search_vector` used
for BM25-style lexical and hybrid retrieval. It does not create a vector index
by default; the corpus is small enough that exact search is simpler and better
for evaluation. To opt into approximate search, pass `--vector-index hnsw` when
loading and `--no-exact` when searching.

If you already loaded an older table, rerun `hep-rag-pg-load` so the lexical
search columns are added and populated.

Search through Postgres:

```bash
hep-rag-pg-search "normalizing flows for lattice gauge theory" \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --device mps \
  --torch-dtype float16 \
  --top-k 10 \
  --candidate-k 50 \
  --max-chunks-per-paper 2
```

Search is exact by default, even if a vector index exists on the table. For
approximate HNSW search, pass `--no-exact`; `--hnsw-ef-search 500` can be used
to trade speed for better recall.

Lexical and hybrid retrieval use Postgres full-text search:

```bash
hep-rag-pg-search "Particle Transformer sparse attention" \
  --retrieval hybrid \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --device mps \
  --torch-dtype float16 \
  --top-k 10 \
  --candidate-k 300 \
  --bm25-candidate-k 300 \
  --max-chunks-per-paper 2
```

Hybrid retrieval combines vector and BM25-style rankings with Reciprocal Rank
Fusion. BM25-only search is also available with `--retrieval bm25`.

Optionally rerank the top vector candidates with Qwen's reranker:

```bash
hep-rag-pg-search "normalizing flows for lattice gauge theory" \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --device mps \
  --torch-dtype float16 \
  --top-k 10 \
  --candidate-k 50 \
  --max-chunks-per-paper 2 \
  --rerank \
  --rerank-device mps \
  --rerank-torch-dtype float16
```

The vector score is retained in the output alongside the reranker score, which
helps debug whether errors come from first-stage retrieval or the reranker.
The `--max-chunks-per-paper` cap diversifies final results while still allowing
more than one useful passage from the same paper.

## Retrieval eval

Draft retrieval labels live in `data/eval/retrieval_queries.jsonl`. To score
the current retriever against those labels:

```bash
hep-rag-eval-retrieval \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --device mps \
  --torch-dtype float16 \
  --candidate-k 3000 \
  --top-papers 40
```

This writes `data/eval/reports/retrieval_metrics.json` and
`data/eval/reports/retrieval_metrics.md`. The command embeds the eval queries,
searches Postgres exactly by default, collapses chunk hits to unique papers,
then computes paper-level recall, precision, hit rate, MRR, and average
precision. Queries with no target papers are summarized separately by how many
non-target papers they retrieve.

To rerank the retrieved paper representatives:

```bash
hep-rag-eval-retrieval \
  --embedding-dir data/embeddings/Qwen-Qwen3-Embedding-0.6B \
  --device mps \
  --torch-dtype float16 \
  --candidate-k 3000 \
  --top-papers 40 \
  --rerank \
  --rerank-candidate-papers 100 \
  --rerank-device mps \
  --rerank-torch-dtype float16
```

To smoke-test configuration without loading the model:

```bash
hep-rag-build-embeddings --limit 3 --dry-run
```

If running a vLLM/OpenAI-compatible embedding server elsewhere, use:

```bash
pip install -e ".[gpu]"
vllm serve Qwen/Qwen3-Embedding-0.6B \
  --task embed \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Then build embeddings through the OpenAI-compatible endpoint:

```bash
hep-rag-build-embeddings \
  --backend openai-compatible \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-Embedding-0.6B
```

Embed a query locally with Qwen's default SentenceTransformer query prompt:

```bash
hep-rag-embed-query "What machine-learning methods reduce critical slowing down?" \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device mps \
  --torch-dtype float16 \
  --out data/embeddings/query.npy
```

For OpenAI-compatible or vLLM-style servers, `prompt_name` is not part of the
HTTP embedding API, so provide an explicit instruction prefix if needed:

```bash
hep-rag-embed-query "critical slowing down normalizing flows" \
  --backend openai-compatible \
  --prompt "Instruct: Given a physics retrieval query, retrieve relevant HEP passages\nQuery: "
```
