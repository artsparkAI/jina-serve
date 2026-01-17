# syntax=docker/dockerfile:1.7
# Jina Embeddings v4 vLLM Worker for RunPod Serverless
#
# Build: docker build -t jina-embeddings-v4-runpod .
# Get your Jina AI API key for free: https://jina.ai/?sui=apikey

FROM runpod/worker-v1-vllm:stable-cuda12.1.0

ENV MODEL_NAME="jinaai/jina-embeddings-v4-vllm-retrieval" \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.9 \
    DTYPE=float16 \
    VLLM_USE_V1=0 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir Pillow requests

COPY --link src/ /src/

WORKDIR /src

CMD ["python3", "-u", "handler.py"]
