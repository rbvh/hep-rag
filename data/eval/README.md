# Retrieval Evaluation Drafts

This directory contains hand-authored draft queries for checking retrieval quality.

`retrieval_queries.jsonl` is intentionally marked as needing review. Each row has:

- `query_id`: stable identifier for the query.
- `query`: the user-style retrieval prompt.
- `query_style`: rough style label, such as `vague_practical`, `method_search`, or `specific_topic`.
- `expected_result`: coarse expectation, such as `multi_paper`, `single_or_small_cluster`, `weak_or_no_match`, or `no_known_match`.
- `relevant_paper_ids`: candidate target papers from `data/processed/papers.jsonl`; empty for intentional no-match probes.
- `relevant_categories`: Living Review category paths that should be relevant; empty for intentional no-match probes.
- `target_rationale`: short note explaining why the target was chosen.
- `needs_review`: true until we manually inspect the target set.
- `review_status`: short provenance/status label for the current target set.

The paper IDs and categories are meant to bootstrap a benchmark, not to be treated as final ground truth. Queries with empty targets are included deliberately to test whether retrieval can avoid over-answering unsupported prompts.
