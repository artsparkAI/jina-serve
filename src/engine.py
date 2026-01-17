"""
Jina Embeddings v4 Engine for vLLM.

Handles model loading, embedding generation, and post-processing.

Get your Jina AI API key for free: https://jina.ai/?sui=apikey
"""

import os
import base64
import time
from io import BytesIO
from typing import Union, List, Optional

import torch
import numpy as np
from PIL import Image
import requests

from job_input import JobInput


class JinaEmbeddingEngine:
    """
    vLLM-based embedding engine for Jina Embeddings v4.
    """

    def __init__(self):
        """Initialize the vLLM engine with Jina Embeddings v4."""
        from vllm import LLM
        from vllm.config import PoolerConfig

        self.model_name = os.environ.get(
            "MODEL_NAME", "jinaai/jina-embeddings-v4-vllm-retrieval"
        )
        self.max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
        self.gpu_memory_utilization = float(
            os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")
        )
        self.dtype = os.environ.get("DTYPE", "float16")

        print(f"Loading model: {self.model_name}")
        print(f"Max model length: {self.max_model_len}")
        print(f"GPU memory utilization: {self.gpu_memory_utilization}")

        start_time = time.time()

        # Set environment variable for vLLM v1 compatibility
        os.environ["VLLM_USE_V1"] = "0"

        self.llm = LLM(
            model=self.model_name,
            task="auto",
            override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False),
            dtype=self.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s on {self.device}")

    def _load_image(self, image_input: str) -> Image.Image:
        """Load image from URL or base64 string."""
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Assume base64 encoded
            image_data = base64.b64decode(image_input)
            return Image.open(BytesIO(image_data)).convert("RGB")

    def _prepare_prompts(
        self, job_input: JobInput
    ) -> tuple[list, list[str], list[int]]:
        """
        Prepare prompts for encoding.

        Returns: (prompts, input_types, estimated_tokens)
        """
        from vllm import TextPrompt

        prompts = []
        input_types = []
        estimated_tokens = []
        prefix = job_input.get_prompt_prefix()

        for item in job_input.get_inputs():
            if isinstance(item, str):
                # Plain text string
                prompts.append(TextPrompt(prompt=f"{prefix}{item}"))
                input_types.append("text")
                estimated_tokens.append(len(item) // 4 + 1)

            elif isinstance(item, dict):
                if "text" in item:
                    text = item["text"]
                    prompts.append(TextPrompt(prompt=f"{prefix}{text}"))
                    input_types.append("text")
                    estimated_tokens.append(len(text) // 4 + 1)

                elif "image" in item:
                    image = self._load_image(item["image"])
                    # Image prompt format for Jina v4 (Qwen2.5-VL based)
                    prompt_text = (
                        "<|im_start|>user\n"
                        "<|vision_start|><|image_pad|><|vision_end|>"
                        "Describe the image.<|im_end|>\n"
                    )
                    prompts.append(
                        TextPrompt(prompt=prompt_text, multi_modal_data={"image": image})
                    )
                    input_types.append("image")
                    estimated_tokens.append(256)  # Approximate for images

                elif "pdf" in item:
                    raise ValueError(
                        "PDF input is not yet supported. "
                        "Please extract text or convert pages to images first."
                    )

                else:
                    raise ValueError(f"Unknown input dict format: {list(item.keys())}")

            else:
                raise ValueError(f"Unknown input type: {type(item)}")

        return prompts, input_types, estimated_tokens

    def _pool_and_normalize(
        self, embeddings_tensor: torch.Tensor, dimensions: Optional[int] = None
    ) -> np.ndarray:
        """Pool token embeddings and L2 normalize."""
        # Mean pooling across tokens
        pooled = embeddings_tensor.sum(dim=0, dtype=torch.float32) / embeddings_tensor.shape[0]

        # L2 normalize
        normalized = torch.nn.functional.normalize(pooled, dim=-1)

        # Truncate dimensions if requested (Matryoshka support)
        if dimensions is not None and dimensions < normalized.shape[-1]:
            normalized = normalized[:dimensions]
            # Re-normalize after truncation
            normalized = torch.nn.functional.normalize(normalized, dim=-1)

        return normalized.cpu().numpy()

    def _convert_embedding(
        self, embedding: np.ndarray, embedding_type: str
    ) -> Union[list, str]:
        """Convert embedding to the requested format."""
        if embedding_type == "float":
            return embedding.tolist()

        elif embedding_type == "base64":
            # Pack as float32 and encode to base64
            return base64.b64encode(
                embedding.astype(np.float32).tobytes()
            ).decode("utf-8")

        elif embedding_type in ("binary", "ubinary"):
            # Binary quantization: sign of each dimension
            binary = np.packbits((embedding > 0).astype(np.uint8))
            return base64.b64encode(binary.tobytes()).decode("utf-8")

        else:
            return embedding.tolist()

    async def encode(self, job_input: JobInput) -> dict:
        """
        Encode inputs and return embeddings in Jina AI API format.

        Response format:
        {
            "model": "jina-embeddings-v4",
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [...]},
                ...
            ],
            "usage": {"prompt_tokens": N, "total_tokens": N}
        }
        """
        # Prepare prompts
        prompts, input_types, estimated_tokens = self._prepare_prompts(job_input)

        # Get embeddings from vLLM
        outputs = self.llm.encode(prompts)

        # Process outputs
        data = []
        embedding_types = job_input.get_embedding_types()
        dimensions = job_input.dimensions

        for idx, output in enumerate(outputs):
            # Get raw embeddings tensor
            raw_embeddings = torch.tensor(output.outputs.embedding)

            # Pool and normalize
            processed = self._pool_and_normalize(raw_embeddings, dimensions)

            # Convert to requested format(s)
            if len(embedding_types) == 1:
                embedding_data = self._convert_embedding(processed, embedding_types[0])
            else:
                # Return multiple formats as dict
                embedding_data = {
                    et: self._convert_embedding(processed, et) for et in embedding_types
                }

            data.append({
                "object": "embedding",
                "index": idx,
                "embedding": embedding_data,
            })

        # Calculate token usage
        total_tokens = sum(estimated_tokens)

        return {
            "model": job_input.model,
            "object": "list",
            "data": data,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
