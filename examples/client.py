#!/usr/bin/env python3
"""
Example client for Jina Embeddings v4 RunPod Serverless API.

This client provides the same interface as the official Jina AI API,
but connects to your self-hosted RunPod endpoint.

Get your Jina AI API key for free: https://jina.ai/?sui=apikey

Usage:
    export RUNPOD_API_KEY="your-api-key"
    export RUNPOD_ENDPOINT_ID="your-endpoint-id"
    python client.py
"""

import os
import requests
import numpy as np
from typing import Union


class JinaEmbeddings:
    """
    Client for Jina Embeddings v4 on RunPod Serverless.

    API-compatible with api.jina.ai/v1/embeddings.
    """

    def __init__(
        self,
        api_key: str = None,
        endpoint_id: str = None,
        base_url: str = None,
    ):
        """
        Initialize the client.

        Args:
            api_key: RunPod API key (or set RUNPOD_API_KEY env var)
            endpoint_id: RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)
            base_url: Full base URL (overrides endpoint_id)
        """
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set RUNPOD_API_KEY env var or pass api_key param."
            )

        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
            if not endpoint_id:
                raise ValueError(
                    "Endpoint ID required. Set RUNPOD_ENDPOINT_ID env var or pass endpoint_id param."
                )
            self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    def embed(
        self,
        input: Union[str, list],
        model: str = "jina-embeddings-v4",
        task: str = "retrieval.passage",
        embedding_type: str = "float",
        dimensions: int = None,
        truncate: bool = False,
    ) -> dict:
        """
        Generate embeddings for the input.

        Args:
            input: Text string, list of strings, or list of dicts with text/image
            model: Model name (default: jina-embeddings-v4)
            task: Task type (retrieval.query, retrieval.passage, text-matching, etc.)
            embedding_type: Output format (float, base64, binary, ubinary)
            dimensions: Truncate to N dimensions (Matryoshka support)
            truncate: Truncate long inputs

        Returns:
            API response dict with embeddings
        """
        # Normalize input to list
        if isinstance(input, str):
            input = [input]

        payload = {
            "input": {
                "model": model,
                "input": input,
                "task": task,
                "embedding_type": embedding_type,
                "truncate": truncate,
            }
        }

        if dimensions:
            payload["input"]["dimensions"] = dimensions

        response = requests.post(
            f"{self.base_url}/runsync",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=120,
        )

        response.raise_for_status()
        result = response.json()

        # Extract the output from RunPod response wrapper
        if "output" in result:
            return result["output"]
        return result

    def embed_query(self, text: str, **kwargs) -> list:
        """Embed a search query."""
        result = self.embed(text, task="retrieval.query", **kwargs)
        return result["data"][0]["embedding"]

    def embed_documents(self, texts: list, **kwargs) -> list:
        """Embed documents for retrieval."""
        result = self.embed(texts, task="retrieval.passage", **kwargs)
        return [item["embedding"] for item in result["data"]]

    def embed_images(self, images: list, **kwargs) -> list:
        """Embed images (URLs or base64)."""
        input_data = [{"image": img} for img in images]
        result = self.embed(input_data, task="retrieval.passage", **kwargs)
        return [item["embedding"] for item in result["data"]]


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    """Example usage demonstration."""
    # Initialize client
    client = JinaEmbeddings()

    print("=" * 60)
    print("Jina Embeddings v4 - RunPod Serverless Client Example")
    print("=" * 60)

    # Example 1: Embed text
    print("\n1. Embedding single text...")
    result = client.embed("Hello, world!")
    print(f"   Embedding dimensions: {len(result['data'][0]['embedding'])}")
    print(f"   First 5 values: {result['data'][0]['embedding'][:5]}")

    # Example 2: Embed multiple texts
    print("\n2. Embedding multiple texts...")
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing deals with text and speech.",
    ]
    result = client.embed(texts, task="retrieval.passage")
    print(f"   Embedded {len(result['data'])} texts")
    print(f"   Total tokens used: {result['usage']['total_tokens']}")

    # Example 3: Query vs passage embeddings
    print("\n3. Semantic search example...")
    documents = [
        "Python is a programming language known for its simplicity.",
        "Machine learning algorithms learn from data.",
        "Cats are popular domestic pets.",
        "Neural networks are inspired by the human brain.",
    ]

    query = "What is artificial intelligence?"

    # Embed documents
    doc_embeddings = client.embed_documents(documents)

    # Embed query
    query_embedding = client.embed_query(query)

    # Calculate similarities
    print(f"\n   Query: '{query}'")
    print("   Document rankings:")
    similarities = [
        (doc, cosine_similarity(query_embedding, emb))
        for doc, emb in zip(documents, doc_embeddings)
    ]
    for doc, score in sorted(similarities, key=lambda x: -x[1]):
        print(f"   {score:.4f}: {doc[:50]}...")

    # Example 4: Different embedding types
    print("\n4. Different embedding types...")
    result_float = client.embed("Test text", embedding_type="float")
    result_base64 = client.embed("Test text", embedding_type="base64")

    print(f"   Float type: {type(result_float['data'][0]['embedding'])}")
    print(f"   Base64 type: {type(result_base64['data'][0]['embedding'])}")
    print(f"   Base64 preview: {result_base64['data'][0]['embedding'][:50]}...")

    # Example 5: Matryoshka dimensions
    print("\n5. Matryoshka dimension reduction...")
    for dim in [2048, 1024, 512, 256, 128]:
        result = client.embed("Test text", dimensions=dim)
        embedding = result["data"][0]["embedding"]
        print(f"   Dimensions={dim}: len={len(embedding)}")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
