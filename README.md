# Jina Embeddings v4 - RunPod Serverless Deployment

Self-hosted vLLM-based serving for [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) on RunPod Serverless with an API identical to `api.jina.ai`.

## Features

- **API Compatible**: Matches the Jina AI Embeddings API format exactly
- **Multimodal**: Supports text and image inputs
- **Task-Specific**: Supports `retrieval.query`, `retrieval.passage`, `text-matching`, `code.query`, `code.passage`
- **Matryoshka**: Dimension truncation support (128, 256, 512, 1024, 2048)
- **Multiple Formats**: Returns embeddings as float, base64, binary, or ubinary

## Quick Start

### 1. Build the Docker Image

```bash
# Clone this repository
git clone <repo-url>
cd jina-serve

# Build the image
docker build -t jina-embeddings-v4-runpod .

# Or with model baked in (larger image, faster cold start):
docker build \
  --build-arg MODEL_NAME=jinaai/jina-embeddings-v4-vllm-retrieval \
  -t jina-embeddings-v4-runpod .
```

### 2. Push to Docker Registry

```bash
# Tag for your registry
docker tag jina-embeddings-v4-runpod your-registry/jina-embeddings-v4:latest

# Push
docker push your-registry/jina-embeddings-v4:latest
```

### 3. Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create a new **Template**:
   - **Container Image**: `your-registry/jina-embeddings-v4:latest`
   - **Container Disk**: 50 GB (for model weights)
   - **Volume Disk**: 50 GB (optional, for caching)
   - **Volume Mount Path**: `/runpod-volume`
3. Create a new **Endpoint** using the template
4. Select GPU type (recommended: **A100 40GB** or **A10G 24GB**)

### 4. Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `jinaai/jina-embeddings-v4-vllm-retrieval` | HuggingFace model ID |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction to use |
| `DTYPE` | `float16` | Model dtype (float16, bfloat16) |
| `HF_TOKEN` | - | HuggingFace token (if needed) |

## API Usage

### Endpoint URL

```
https://api.runpod.ai/v2/{endpoint_id}/runsync
```

### Authentication

```bash
-H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

### Request Format

The API matches [Jina AI Embeddings API](https://jina.ai/embeddings/) exactly:

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "input": {
      "model": "jina-embeddings-v4",
      "input": ["Hello, world!", "How are you?"],
      "task": "retrieval.passage"
    }
  }'
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | No | `jina-embeddings-v4` | Model identifier |
| `input` | array | **Yes** | - | Array of strings or objects to embed |
| `task` | string | No | `retrieval.passage` | Task type (see below) |
| `embedding_type` | string/array | No | `float` | Output format |
| `dimensions` | int | No | `2048` | Truncate to dimensions (Matryoshka) |
| `truncate` | bool | No | `false` | Truncate long inputs |
| `late_chunking` | bool | No | `false` | Enable late chunking |
| `return_multivector` | bool | No | `false` | Return multi-vector embeddings |

### Task Types

| Task | Use Case |
|------|----------|
| `retrieval.query` | Encoding search queries |
| `retrieval.passage` | Encoding documents/passages to search |
| `text-matching` | Semantic text similarity |
| `code.query` | Code search queries |
| `code.passage` | Code snippets to search |

### Input Formats

**Text strings:**
```json
{
  "input": ["Hello world", "How are you?"]
}
```

**Text objects:**
```json
{
  "input": [{"text": "Hello world"}, {"text": "How are you?"}]
}
```

**Images (URL or base64):**
```json
{
  "input": [
    {"image": "https://example.com/image.jpg"},
    {"image": "iVBORw0KGgoAAAANSUhEUgAAAAUA..."}
  ]
}
```

**Mixed inputs:**
```json
{
  "input": [
    {"text": "A beautiful sunset"},
    {"image": "https://example.com/sunset.jpg"}
  ]
}
```

### Response Format

```json
{
  "model": "jina-embeddings-v4",
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.789, -0.012, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### Embedding Type Formats

**float (default):**
```json
{"embedding": [0.123, -0.456, 0.789, ...]}
```

**base64:**
```json
{"embedding": "zczMPQAAgL8AAIBP..."}
```

**binary/ubinary:**
```json
{"embedding": "gICAQEBA..."}
```

**Multiple formats:**
```json
{
  "embedding_type": ["float", "base64"],
  // Response:
  "embedding": {
    "float": [0.123, ...],
    "base64": "zczMPQ..."
  }
}
```

## Examples

### Python Client

```python
import requests

# Get your Jina AI API key for free: https://jina.ai/?sui=apikey
RUNPOD_API_KEY = "your-runpod-api-key"
ENDPOINT_ID = "your-endpoint-id"

def get_embeddings(texts, task="retrieval.passage"):
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "input": {
                "model": "jina-embeddings-v4",
                "input": texts,
                "task": task,
            }
        },
    )
    result = response.json()
    return [item["embedding"] for item in result["output"]["data"]]

# Encode documents
docs = ["The quick brown fox", "Machine learning is fascinating"]
doc_embeddings = get_embeddings(docs, task="retrieval.passage")

# Encode query
query_embedding = get_embeddings(["What is ML?"], task="retrieval.query")[0]
```

### Semantic Search

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get embeddings
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Cats are furry animals",
]
query = "What is artificial intelligence?"

doc_embs = get_embeddings(documents, task="retrieval.passage")
query_emb = get_embeddings([query], task="retrieval.query")[0]

# Rank by similarity
scores = [cosine_similarity(query_emb, doc) for doc in doc_embs]
ranked = sorted(zip(documents, scores), key=lambda x: -x[1])

for doc, score in ranked:
    print(f"{score:.4f}: {doc}")
```

### Image + Text Search

```python
# Encode mixed content
inputs = [
    {"text": "A photo of a cat"},
    {"image": "https://example.com/cat.jpg"},
    {"text": "Dogs are loyal pets"},
]

embeddings = get_embeddings(inputs, task="retrieval.passage")
```

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally with test input
cd src
python handler.py --test_input '{"input": {"model": "jina-embeddings-v4", "input": ["Hello world"]}}'

# Or start local API server
python handler.py --rp_serve_api --rp_api_port 8000

# Test with curl
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"model": "jina-embeddings-v4", "input": ["Hello world"]}}'
```

## GPU Requirements

| GPU | VRAM | Recommended |
|-----|------|-------------|
| A100 40GB | 40 GB | Yes |
| A100 80GB | 80 GB | Yes |
| A10G | 24 GB | Yes |
| L4 | 24 GB | Yes |
| RTX 4090 | 24 GB | Yes |
| RTX 3090 | 24 GB | Possible |
| T4 | 16 GB | No (insufficient VRAM) |

## Troubleshooting

### Out of Memory
- Reduce `GPU_MEMORY_UTILIZATION` to `0.8`
- Reduce `MAX_MODEL_LEN` to `4096`
- Use a GPU with more VRAM

### Slow Cold Start
- Bake the model into the Docker image during build
- Use a network volume to cache model weights

### Model Loading Fails
- Ensure `trust_remote_code=True` is set (handled by engine)
- Check HuggingFace token if model is gated
- Verify sufficient disk space for model weights

## License

This deployment code is provided under the MIT License. The Jina Embeddings v4 model is subject to its own license terms - see the [model card](https://huggingface.co/jinaai/jina-embeddings-v4) for details.
