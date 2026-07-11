ARG VLLM_IMAGE=vllm/vllm-openai:v0.25.0
FROM ${VLLM_IMAGE}

COPY docker/vllm/reranker/server.py /app/reranker_server.py

ENTRYPOINT ["python3", "/app/reranker_server.py"]
