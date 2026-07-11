"""HTTP service implementing Qwen3-Reranker's official vLLM scoring recipe."""

from __future__ import annotations

import math
import os
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RerankRequest(BaseModel):
    model: str | None = None
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)
    instruction: str = DEFAULT_INSTRUCTION
    truncate_prompt_tokens: int | None = Field(default=None, ge=1)


class Qwen3Reranker:
    def __init__(self) -> None:
        self.model_name = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        self.max_model_len = int(os.getenv("RERANK_MAX_MODEL_LEN", "1024"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.true_token = self._single_token_id("yes")
        self.false_token = self._single_token_id("no")
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )
        self.model = LLM(
            model=self.model_name,
            dtype=os.getenv("RERANK_DTYPE", "bfloat16"),
            max_model_len=self.max_model_len,
            gpu_memory_utilization=float(os.getenv("RERANK_GPU_MEMORY_UTILIZATION", "0.30")),
            enforce_eager=True,
            enable_prefix_caching=True,
        )
        self._lock = threading.Lock()

    def _single_token_id(self, text: str) -> int:
        token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise RuntimeError(f"Expected {text!r} to be one token, got {token_ids}")
        return token_ids[0]

    def _prompt(self, instruction: str, query: str, document: str, max_length: int) -> TokensPrompt:
        text = (
            f"{PREFIX}<Instruct>: {instruction}\n\n"
            f"<Query>: {query}\n\n"
            f"<Document>: {document}"
        )
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        content_limit = max_length - len(self.suffix_tokens)
        if content_limit < 1:
            raise ValueError(
                f"Maximum prompt length {max_length} is too short for the reranker suffix"
            )
        return TokensPrompt(prompt_token_ids=token_ids[:content_limit] + self.suffix_tokens)

    def score(
        self,
        query: str,
        documents: list[str],
        instruction: str,
        requested_max_length: int | None,
    ) -> tuple[list[float], int]:
        max_length = min(requested_max_length or self.max_model_len, self.max_model_len)
        prompts = [
            self._prompt(instruction, query, document, max_length)
            for document in documents
        ]
        prompt_tokens = sum(len(prompt["prompt_token_ids"]) for prompt in prompts)
        with self._lock:
            outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)

        scores: list[float] = []
        for output in outputs:
            logprobs: dict[int, Any] = output.outputs[0].logprobs[-1]
            true_logprob = (
                logprobs[self.true_token].logprob if self.true_token in logprobs else -10.0
            )
            false_logprob = (
                logprobs[self.false_token].logprob if self.false_token in logprobs else -10.0
            )
            true_score = math.exp(true_logprob)
            false_score = math.exp(false_logprob)
            scores.append(true_score / (true_score + false_score))
        return scores, prompt_tokens


app = FastAPI(title="Qwen3 vLLM Reranker", version="1.0")
reranker: Qwen3Reranker | None = None


def get_reranker() -> Qwen3Reranker:
    if reranker is None:
        raise HTTPException(status_code=503, detail="Reranker is still starting")
    return reranker


@app.get("/health")
def health() -> dict[str, str]:
    active_reranker = get_reranker()
    return {"status": "ok", "model": active_reranker.model_name}


@app.post("/rerank")
@app.post("/v1/rerank")
def rerank(request: RerankRequest) -> dict[str, Any]:
    active_reranker = get_reranker()
    if request.model and request.model != active_reranker.model_name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This service has {active_reranker.model_name!r} loaded, "
                f"not {request.model!r}"
            ),
        )
    if request.top_n is not None and request.top_n > len(request.documents):
        raise HTTPException(status_code=422, detail="top_n exceeds the number of documents")

    try:
        scores, prompt_tokens = active_reranker.score(
            request.query,
            request.documents,
            request.instruction,
            request.truncate_prompt_tokens,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    if request.top_n is not None:
        ranked = ranked[: request.top_n]
    results = [
        {
            "index": index,
            "document": {"text": request.documents[index]},
            "relevance_score": score,
        }
        for index, score in ranked
    ]
    return {
        "model": active_reranker.model_name,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(request.documents),
            "total_tokens": prompt_tokens + len(request.documents),
        },
        "results": results,
    }


if __name__ == "__main__":
    reranker = Qwen3Reranker()
    uvicorn.run(
        app,
        host=os.getenv("RERANK_HOST", "0.0.0.0"),
        port=int(os.getenv("RERANK_PORT", "8000")),
    )
