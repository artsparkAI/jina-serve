"""
Job Input Parser for Jina AI API-compatible requests.

Get your Jina AI API key for free: https://jina.ai/?sui=apikey
"""

from dataclasses import dataclass, field
from typing import Union, List, Optional


@dataclass
class JobInput:
    """
    Parses and validates input matching the Jina AI Embeddings API format.

    Request body schema for jina-embeddings-v4:
    {
        "model": "jina-embeddings-v4",
        "input": [...],
        "embedding_type": "float" | "base64" | "binary" | "ubinary",
        "task": "retrieval.query" | "retrieval.passage" | "text-matching" | "code.query" | "code.passage",
        "dimensions": int (optional, for Matryoshka truncation),
        "late_chunking": bool (optional),
        "truncate": bool (optional),
        "return_multivector": bool (optional)
    }
    """

    model: str = "jina-embeddings-v4"
    input: Union[List[Union[str, dict]], str, dict] = field(default_factory=list)
    embedding_type: Union[str, List[str]] = "float"
    task: str = "retrieval.passage"
    dimensions: Optional[int] = None
    late_chunking: bool = False
    truncate: bool = False
    return_multivector: bool = False

    def __init__(self, raw_input: dict):
        """Parse raw job input into structured fields."""
        if not isinstance(raw_input, dict):
            raise ValueError("Input must be a dictionary")

        # Required field
        if "input" not in raw_input:
            raise ValueError("Missing required field: 'input'")

        self.model = raw_input.get("model", "jina-embeddings-v4")
        self.input = raw_input["input"]
        self.embedding_type = raw_input.get("embedding_type", "float")
        self.task = raw_input.get("task", "retrieval.passage")
        self.dimensions = raw_input.get("dimensions")
        self.late_chunking = raw_input.get("late_chunking", False)
        self.truncate = raw_input.get("truncate", False)
        self.return_multivector = raw_input.get("return_multivector", False)

        # Normalize input to list
        if not isinstance(self.input, list):
            self.input = [self.input]

        # Validate task
        valid_tasks = {
            "retrieval.query",
            "retrieval.passage",
            "text-matching",
            "code.query",
            "code.passage",
        }
        if self.task not in valid_tasks:
            raise ValueError(
                f"Invalid task: '{self.task}'. Must be one of: {', '.join(valid_tasks)}"
            )

        # Validate embedding_type
        valid_types = {"float", "base64", "binary", "ubinary"}
        if isinstance(self.embedding_type, list):
            for et in self.embedding_type:
                if et not in valid_types:
                    raise ValueError(f"Invalid embedding_type: '{et}'")
        elif self.embedding_type not in valid_types:
            raise ValueError(f"Invalid embedding_type: '{self.embedding_type}'")

        # Validate dimensions if provided
        if self.dimensions is not None:
            valid_dims = {128, 256, 512, 1024, 2048}
            if self.dimensions not in valid_dims:
                raise ValueError(
                    f"Invalid dimensions: {self.dimensions}. "
                    f"Must be one of: {sorted(valid_dims)}"
                )

    def get_inputs(self) -> List[Union[str, dict]]:
        """Return the list of inputs to encode."""
        return self.input

    def get_prompt_prefix(self) -> str:
        """Get the prompt prefix based on task type."""
        prefixes = {
            "retrieval.query": "Query: ",
            "retrieval.passage": "Passage: ",
            "text-matching": "",
            "code.query": "Query: ",
            "code.passage": "Passage: ",
        }
        return prefixes.get(self.task, "Passage: ")

    def get_embedding_types(self) -> List[str]:
        """Return embedding types as a list."""
        if isinstance(self.embedding_type, list):
            return self.embedding_type
        return [self.embedding_type]
