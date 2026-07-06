# Project Plan: Evaluated RAG System for High-Energy Physics Papers

## 1. Project Goal

Build a production-shaped retrieval-augmented generation system for a focused corpus of high-energy physics papers. The system should answer technical questions with cited evidence, abstain when the corpus does not support an answer, and report retrieval and answer-quality metrics.

The point is not to build another generic "chat with PDFs" demo. The point is to demonstrate that you can build and evaluate a reliable ML system over difficult technical documents: scientific language, equations, citations, long context, overlapping terminology, and highly specialized concepts.

Target CV bullet:

> Built an evaluated RAG system for high-energy physics papers with structured arXiv/PDF ingestion, hybrid retrieval, reranking, citation-grounded answers, abstention logic and latency benchmarking.

## 2. Why This Project Fits Your Profile

This project connects several parts of your background that are otherwise hard to show in one portfolio artifact:

- Deconcern: document AI, production ML, structured extraction, difficult-document fallback, model serving.
- Research: high-energy physics domain knowledge, scientific writing, paper-level reasoning.
- Industry gaps: RAG, retrieval evaluation, production APIs, observability, latency/cost tradeoffs.
- ML depth: embeddings, reranking, retrieval metrics, possible domain adaptation of retrievers.

It would be especially useful for roles involving production LLM systems, scientific ML, AI4Science, foundation-model applications, and technical-document intelligence.

## 3. Suggested Project Name

Working names:

- `hep-rag-lab`
- `paper-trail`
- `citation-grounded-hep-rag`
- `hep-retrieval-bench`

I would avoid using "arXiv" in the project name to stay clearly independent from arXiv branding. The arXiv API documentation asks independent projects to acknowledge arXiv data usage and avoid names or branding that imply endorsement.

## 4. Corpus Scope

Start narrow. A narrow corpus makes the system more credible, easier to evaluate, and less likely to feel like a toy.

Recommended first corpus:

- 100-300 papers in event generation, parton showers, normalizing flows, anomaly detection, and ML for high-energy physics.
- Include your own papers and papers they cite.
- Add standard papers in event generators, parton-shower algorithms, normalizing flows, and ML anomaly detection.

Possible expansion corpus:

- 500-1,000 papers from `hep-ph`, `hep-ex`, `hep-th`, and `cs.LG` cross-lists related to event generation and ML.
- Keep this as phase 2. The first version should be small enough that you can manually inspect failure modes.

## 5. System Capabilities

The system should support:

- Natural-language questions over a paper corpus.
- Answers grounded only in retrieved evidence.
- Citations to paper, section, page or paragraph, and URL.
- Explicit abstention when evidence is insufficient.
- Comparison questions across papers.
- Search by concept, author, title, year, or method.
- Retrieval diagnostics showing which chunks were selected and why.
- Benchmarking of retrieval quality, answer faithfulness, citation quality, latency and cost.

Example questions:

- "Which papers use normalizing flows for event generation?"
- "What is the motivation for using surjective flows in event generation?"
- "How do learned generative models differ from standard Monte Carlo event generators?"
- "Which papers discuss anomaly detection with graph neural networks?"
- "What assumptions does this paper make about permutation invariance?"
- "Does the corpus contain evidence that method X improves NLL accuracy?"
- "Compare the evaluation metrics used in these two papers."
- "Which paper introduced this specific sampling method?"

## 6. Architecture Overview

High-level flow:

```text
paper metadata/PDF/source
  -> ingestion
  -> structured document representation
  -> chunking
  -> embeddings + lexical index
  -> retrieval
  -> reranking
  -> evidence selection
  -> LLM answer generation
  -> citation validation
  -> evaluation + logging
```

Runtime query path:

```text
user question
  -> normalize/query enrich
  -> dense retrieval
  -> BM25 lexical retrieval
  -> hybrid merge
  -> cross-encoder reranking
  -> context packing
  -> LLM answer with citations
  -> output answer, citations, confidence and diagnostics
```

## 7. Data Acquisition

### 7.1 Metadata

Use the arXiv API for metadata and links. Store:

- arXiv ID
- title
- authors
- abstract
- categories
- publication date
- DOI/journal reference if available
- PDF URL
- source URL if available
- references if extracted later

Relevant source: arXiv API access documentation.

### 7.2 PDFs and Sources

Use two ingestion routes:

1. Preferred route: arXiv source if available.
   - LaTeX source usually preserves section structure, citations, equations, labels and bibliography more cleanly than PDFs.
   - More work upfront, but better chunk quality.

2. Fallback route: PDF parsing.
   - Use GROBID for scientific-paper structure and TEI output.
   - Use PyMuPDF for page-level extraction, coordinates, and PDF fallback.

Do not try to solve equation understanding in version 1. Preserve equations as text snippets or placeholders and focus on surrounding explanatory prose.

## 8. Document Representation

Create a normalized internal schema, for example:

```json
{
  "paper_id": "arxiv:2205.01697",
  "title": "Surjective Normalizing Flows for Event Generation and Density Estimation",
  "authors": ["Rob Verheyen"],
  "year": 2022,
  "url": "https://...",
  "sections": [
    {
      "section_id": "intro",
      "heading": "Introduction",
      "paragraphs": [
        {
          "paragraph_id": "intro.p3",
          "text": "...",
          "page": 2,
          "citations": ["..."],
          "equations_nearby": ["..."]
        }
      ]
    }
  ]
}
```

For each retrieval chunk, store:

- chunk ID
- text
- title
- authors
- year
- arXiv ID
- section heading
- page number if available
- paragraph IDs
- citation keys mentioned
- token count
- embedding vector

## 9. Chunking Strategy

Start with simple, inspectable chunking:

- Chunk by section-aware paragraphs.
- Aim for 250-500 tokens per chunk.
- Add 50-100 token overlap where needed.
- Keep title, abstract and section heading attached as metadata.
- Keep captions as separate chunks tagged as `caption`.
- Keep bibliography separate from main text, at least initially.

Important: do not chunk blindly every N tokens if it splits equations, definitions or method descriptions. The system will look much better if chunks correspond to meaningful scientific units.

Version 2 chunking:

- Add citation-context chunks: paragraphs around a citation plus cited-paper metadata.
- Add paper-level summary chunks from title + abstract + conclusion.
- Add method-result chunks for sections named "Method", "Results", "Experiments", "Validation", etc.

## 10. Retrieval System

### 10.1 Baseline 1: BM25

Implement a lexical baseline first. BM25 is still strong for technical terminology:

- exact phrases
- acronyms
- author names
- method names
- physics terms such as "Sudakov", "Lund plane", "parton shower", "NLL", "matching", "normalizing flow"

This baseline gives you a meaningful comparison and avoids the project becoming embedding-only.

### 10.2 Baseline 2: Dense Retrieval

Use an existing Sentence Transformers embedder first. Do not train an embedder from scratch.

Initial candidates:

- `BAAI/bge-base-en-v1.5`
- `intfloat/e5-base-v2`
- `mixedbread-ai/mxbai-embed-large-v1`
- a scientific-domain embedding model if one benchmarks well

Use local embedding inference if feasible. It makes the project stronger because you can measure latency and avoid relying entirely on API calls.

### 10.3 Vector Index

Start simple:

- FAISS for local vector search if you want a lightweight repo.
- Qdrant if you want a more production-shaped service with filtering and persistence.

Recommendation:

- Version 1: FAISS, because it is fast to build and easy to version.
- Version 2: Qdrant, if you want to show production service design and metadata filtering.

### 10.4 Hybrid Retrieval

Combine BM25 and dense retrieval using reciprocal rank fusion or score normalization.

Pipeline:

```text
top_k_bm25 = 50
top_k_dense = 50
merge candidates
deduplicate by chunk_id
rerank top 50
return top 5-10 evidence chunks
```

Hybrid retrieval is important because physics papers contain both semantic questions and exact technical vocabulary.

### 10.5 Reranking

Add a cross-encoder reranker after the first retrieval stage.

Why:

- Dense retrieval is fast but approximate.
- Cross-encoders are slower but better at judging whether a passage actually answers a question.
- This lets you discuss quality/latency tradeoffs, which is useful for industry roles.

Evaluate:

- BM25 only
- dense only
- hybrid without reranker
- hybrid with reranker

## 11. Answer Generation

The LLM should receive:

- the user question
- top evidence chunks
- chunk IDs
- citation metadata
- instructions to answer only from evidence
- instructions to abstain if evidence is insufficient

Output schema:

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "...",
      "paper_title": "...",
      "section": "...",
      "page": 4,
      "url": "..."
    }
  ],
  "confidence": "high | medium | low",
  "insufficient_evidence": false,
  "limitations": "..."
}
```

Prompt principles:

- No unsupported claims.
- Every factual claim should be traceable to one or more citations.
- If the evidence does not support an answer, say so.
- If evidence is partial or conflicting, state that explicitly.
- Prefer concise technical answers over long prose.

For the first version, use an API LLM or a strong local instruction model, whichever makes development easier. The project is about retrieval and evaluation, not proving you can host a frontier LLM.

## 12. Citation Validation

Do a lightweight validation pass after answer generation:

- Check that every cited chunk was actually included in the context.
- Check that every citation ID exists in the retrieval results.
- Reject or flag answers with no citations unless the answer is an abstention.
- Optionally ask a second model to mark which sentence is supported by which citation.

This is a small feature, but it makes the system feel much more serious than a plain demo.

## 13. Evaluation Design

Evaluation is the key differentiator. Build this early, not at the end.

### 13.1 Gold Question Set

Create 80-120 questions.

Question types:

- Fact lookup: answer is in one specific paper/section.
- Method explanation: requires a method paragraph.
- Comparison: requires evidence from two or more papers.
- Literature mapping: asks which papers discuss a concept.
- Citation trail: asks what a paper builds on or cites.
- Negative questions: answer is not in the corpus.
- Ambiguous questions: system should ask for clarification or state ambiguity.

For each question, store:

```json
{
  "question_id": "q001",
  "question": "...",
  "type": "fact_lookup",
  "gold_papers": ["arxiv:..."],
  "gold_chunks": ["..."],
  "gold_answer_notes": "...",
  "must_abstain": false
}
```

You can hand-label the gold chunks for the first 50 questions. This is not wasted work: it makes the project credible.

### 13.2 Retrieval Metrics

Report:

- Recall@5 and Recall@10: did the system retrieve a relevant chunk?
- MRR@10: how high was the first relevant result?
- nDCG@10: useful if you label graded relevance.
- query latency: retrieval and reranking time separately.

Compare:

- BM25
- dense retrieval
- hybrid retrieval
- hybrid + reranker

### 13.3 Answer Metrics

Report:

- citation precision: are cited chunks actually relevant?
- groundedness: are claims supported by citations?
- abstention precision/recall on unanswerable questions
- answer usefulness: manual 1-5 rating on a small eval set
- hallucination rate: fraction of answers with unsupported claims

For answer grading, start with manual grading for 30-50 examples. Optionally add LLM-as-judge later, but do not rely on it as the only evaluation.

### 13.4 Latency and Cost Metrics

Measure:

- embedding time per query
- BM25 retrieval time
- vector retrieval time
- reranking time
- LLM generation time
- end-to-end latency
- tokens per query
- approximate cost per query if using an API model

This lets the project speak to production ML roles, not only research roles.

## 14. API and Application

Build a minimal but real service.

### 14.1 FastAPI Backend

Endpoints:

- `POST /query`
  - input: question, filters, top_k, retrieval mode
  - output: answer, citations, diagnostics

- `GET /papers`
  - list indexed papers

- `GET /papers/{paper_id}`
  - metadata and chunks for one paper

- `POST /evaluate`
  - run evaluation on a selected benchmark split

- `GET /health`
  - service status

Use Pydantic schemas for request and response models.

### 14.2 Interface

Minimal options:

- CLI first.
- Then a simple Streamlit or small web UI.

UI should show:

- answer
- citations
- retrieved chunks
- scores
- latency
- retrieval mode
- "not enough evidence" cases

Do not spend too much time on frontend polish. The README and evaluation report matter more.

## 15. Observability and Logging

Log each query:

- timestamp
- question
- retrieval mode
- retrieved chunk IDs
- similarity/rerank scores
- answer
- citations
- latency by stage
- model names
- errors

Save logs locally as JSONL. This is enough for a portfolio project and shows production instincts.

Optional:

- Add Prometheus metrics if you want continuity with your Deconcern experience.
- Add simple dashboard summaries.

## 16. Testing

Tests should cover:

- metadata parsing
- PDF/source ingestion
- chunk generation
- deterministic chunk IDs
- index construction
- retrieval returns valid chunk IDs
- citation validation rejects invalid citations
- answer schema validation
- evaluation metric calculations
- API contract tests

Do not aim for exhaustive coverage. Aim for tests that prove this is more than a notebook.

## 17. Repository Structure

Suggested layout:

```text
hep-rag-lab/
  README.md
  pyproject.toml
  docker-compose.yml
  Dockerfile
  configs/
    corpus.yaml
    retrieval.yaml
    models.yaml
  data/
    raw/
    processed/
    indexes/
    eval/
  src/hep_rag/
    ingest/
      arxiv_client.py
      pdf_parser.py
      source_parser.py
      schema.py
    chunking/
      chunker.py
    retrieval/
      bm25.py
      dense.py
      hybrid.py
      rerank.py
    generation/
      prompts.py
      answerer.py
      citation_validation.py
    evaluation/
      datasets.py
      metrics.py
      run_eval.py
    api/
      main.py
      schemas.py
    cli.py
  tests/
  notebooks/
    error_analysis.ipynb
  reports/
    evaluation_report.md
```

## 18. Implementation Phases

### Phase 0: Project Definition

Duration: 0.5-1 day.

Deliverables:

- Choose corpus scope.
- Choose project name.
- Create repo skeleton.
- Write README project statement.
- Define success criteria.

Success criteria:

- You can explain in one paragraph why this is not a toy chatbot.

### Phase 1: Corpus and Ingestion

Duration: 3-5 days.

Tasks:

- Use arXiv API to collect metadata for a seed corpus.
- Download PDFs and, where available, source files.
- Parse 20-30 papers first.
- Implement normalized document schema.
- Store processed documents as JSONL or parquet.
- Inspect 10 parsed papers manually.

Deliverables:

- `data/processed/papers.jsonl`
- ingestion script
- parsing-quality notes in README

Success criteria:

- At least 50 papers parsed with title, abstract, sections and paragraphs.
- Chunk metadata includes paper ID, section and URL.

### Phase 2: Chunking and Baseline Search

Duration: 2-4 days.

Tasks:

- Implement section-aware chunking.
- Build BM25 index.
- Build dense embedding index.
- Implement command-line search.
- Inspect retrieval results for 20 example questions.

Deliverables:

- BM25 baseline
- dense retriever baseline
- command: `hep-rag search "question"`

Success criteria:

- For obvious questions, relevant chunks appear in top 10.
- You can compare BM25 and dense retrieval qualitatively.

### Phase 3: Hybrid Retrieval and Reranking

Duration: 3-5 days.

Tasks:

- Implement hybrid retrieval.
- Add reciprocal rank fusion or score normalization.
- Add cross-encoder reranking.
- Add metadata filters.
- Add retrieval diagnostics.

Deliverables:

- retrieval modes: `bm25`, `dense`, `hybrid`, `hybrid-rerank`
- score breakdown in query output

Success criteria:

- Hybrid retrieval visibly improves over either baseline on technical queries.
- You have examples where BM25 wins and examples where dense retrieval wins.

### Phase 4: Answer Generation with Citations

Duration: 3-5 days.

Tasks:

- Build context-packing logic.
- Write grounded-answer prompt.
- Implement structured output schema.
- Add citation validation.
- Add abstention behavior.
- Create a CLI query interface.

Deliverables:

- command: `hep-rag ask "question"`
- JSON answer with citations and diagnostics

Success criteria:

- Answers cite specific papers/sections.
- Unsupported questions trigger abstention rather than hallucination.

### Phase 5: Evaluation Benchmark

Duration: 5-8 days.

Tasks:

- Write 80-120 gold questions.
- Label gold papers/chunks for at least 50.
- Implement retrieval metrics.
- Implement answer-quality grading template.
- Run evaluation across retrieval modes.
- Analyze failure cases.

Deliverables:

- `data/eval/questions.jsonl`
- `reports/evaluation_report.md`
- tables comparing retrieval modes

Success criteria:

- Report includes Recall@5, MRR@10, citation precision, abstention behavior and latency.
- You can state a concrete result, e.g. "hybrid + reranking improved Recall@5 from X to Y over dense retrieval."

### Phase 6: Production Shape

Duration: 3-5 days.

Tasks:

- Add FastAPI service.
- Add Dockerfile.
- Add config files.
- Add JSONL query logging.
- Add tests for core components.
- Add simple UI or keep a polished CLI.

Deliverables:

- `docker compose up` runs the service.
- `/query` endpoint returns answer, citations and diagnostics.
- README has setup and example queries.

Success criteria:

- A reviewer can run the system locally and reproduce the demo.
- The service has clear schemas, logs and basic tests.

### Phase 7: Optional Domain Adaptation

Duration: 5-10 days. Only do this after the evaluated baseline exists.

Options:

1. Fine-tune an embedding model.
   - Create positive query-passage pairs from citation contexts, titles, abstracts and hand-labeled eval data.
   - Train with contrastive learning.
   - Compare retrieval metrics against the baseline embedder.

2. Train a small reranker.
   - Use labeled query/chunk pairs.
   - Add hard negatives from failed retrievals.
   - Compare reranking quality and latency.

3. Add citation graph retrieval.
   - Retrieve cited/citing papers around relevant chunks.
   - Useful for literature mapping questions.

Suggested CV bullet if successful:

> Fine-tuned a retrieval model on citation-derived supervision from high-energy physics papers, improving top-5 retrieval over a general-purpose embedding baseline.

## 19. Minimal Viable Version

If you want a version that can be completed quickly, do this:

- 100 papers.
- GROBID/PyMuPDF PDF parsing only.
- Section-aware chunks.
- BM25 + dense retrieval.
- Hybrid retrieval.
- 50 hand-written evaluation questions.
- LLM answers with citations.
- Retrieval metrics + latency metrics.
- FastAPI endpoint.
- Strong README.

This is enough to be CV-worthy.

## 20. Stretch Version

If the project goes well:

- arXiv source parsing.
- Citation graph.
- Cross-encoder reranker.
- Fine-tuned retriever.
- Qdrant service.
- Prometheus metrics.
- Small UI.
- Public demo with a limited corpus.
- Evaluation report with error taxonomy.

## 21. What Not To Do

Avoid:

- Starting with a fancy UI.
- Spending weeks training an embedder before there is an evaluation set.
- Building a generic chatbot with no metrics.
- Overclaiming that the system "understands physics".
- Trying to parse every equation perfectly.
- Indexing thousands of papers before the 100-paper version works well.
- Using only LLM-as-judge evaluation.

## 22. README Structure

The README should be written like a small engineering case study:

1. Problem statement.
2. Why high-energy physics papers are hard.
3. Architecture diagram.
4. Corpus description.
5. Retrieval methods.
6. Evaluation methodology.
7. Results table.
8. Example queries.
9. Failure cases.
10. How to run locally.
11. Next steps.

## 23. Example Evaluation Table

Target table format:

| Method | Recall@5 | Recall@10 | MRR@10 | Median latency | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 | TBD | TBD | TBD | TBD | Strong exact-term baseline |
| Dense | TBD | TBD | TBD | TBD | Better semantic matching |
| Hybrid | TBD | TBD | TBD | TBD | Expected best first-stage retrieval |
| Hybrid + rerank | TBD | TBD | TBD | TBD | Best quality, slower |

Answer-quality table:

| System | Citation precision | Abstention recall | Unsupported-claim rate | Median total latency |
| --- | ---: | ---: | ---: | ---: |
| Hybrid + LLM | TBD | TBD | TBD | TBD |
| Hybrid + rerank + LLM | TBD | TBD | TBD | TBD |

## 24. Possible Final CV Bullets

Conservative:

> Built an evaluated RAG system for high-energy physics papers with hybrid retrieval, citation-grounded answers, abstention logic and latency benchmarking.

More production-focused:

> Deployed a FastAPI-based RAG service for technical scientific papers with structured ingestion, hybrid retrieval, reranking, citation validation, query logging and Dockerized local deployment.

More research-focused:

> Benchmarked BM25, dense, hybrid and reranked retrieval over a high-energy physics paper corpus using hand-labeled scientific question-answer pairs.

If domain adaptation succeeds:

> Fine-tuned a retriever on citation-derived supervision from high-energy physics papers, improving top-k retrieval over a general-purpose embedding baseline.

## 25. Recommended Build Order

1. Create corpus of 100 papers.
2. Parse into structured JSON.
3. Chunk and inspect.
4. Build BM25 search.
5. Build dense search.
6. Add hybrid retrieval.
7. Write 50 evaluation questions.
8. Run retrieval metrics.
9. Add LLM answer generation.
10. Add citations and abstention.
11. Add answer evaluation.
12. Add FastAPI and Docker.
13. Write evaluation report.
14. Consider reranker or fine-tuning.

## 26. References and Useful Documentation

- [arXiv API access](https://info.arxiv.org/help/api/index.html)
- [GROBID documentation](https://grobid.readthedocs.io/en/latest/)
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/en/latest/)
- [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
- [Qdrant documentation](https://qdrant.tech/documentation/overview/)
- [BEIR information retrieval benchmark](https://github.com/beir-cellar/beir)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
