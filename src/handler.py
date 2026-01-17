"""
RunPod Serverless Handler for Jina Embeddings v4 (vLLM Retrieval)

Serves jina-embeddings-v4-vllm-retrieval with an API identical to api.jina.ai.

Get your Jina AI API key for free: https://jina.ai/?sui=apikey
"""

import runpod
from engine import JinaEmbeddingEngine
from job_input import JobInput

# Initialize the embedding engine at startup
print("Initializing Jina Embeddings v4 engine...")
embedding_engine = JinaEmbeddingEngine()
print("Engine initialization complete.")


async def handler(job: dict) -> dict:
    """
    Async handler for embedding requests.

    Expected input format (matching Jina AI API):
    {
        "model": "jina-embeddings-v4",
        "input": ["text1", "text2", ...] or [{"text": "..."}, {"image": "..."}, ...],
        "task": "retrieval.query" | "retrieval.passage" | "text-matching" | ...,
        "embedding_type": "float" | "base64" | "binary" | "ubinary",
        "dimensions": 2048 (optional),
        "truncate": false (optional),
        "late_chunking": false (optional),
        "return_multivector": false (optional)
    }
    """
    try:
        job_input = JobInput(job["input"])
        result = await embedding_engine.encode(job_input)
        return result
    except Exception as e:
        return {
            "error": {
                "message": str(e),
                "type": type(e).__name__
            }
        }


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
