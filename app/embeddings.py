"""Local embeddings module using Sentence Transformers.

Generates dense 384-dimensional normalized vector embeddings locally on CPU
using the all-MiniLM-L6-v2 model (or configured alternative).
"""

import logging
from typing import List, Optional, Union

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Expected vector dimension for all-MiniLM-L6-v2
EMBEDDING_DIMENSION = 384
DEFAULT_DEVICE = "cpu"

# Module-level model cache to ensure the model is loaded only once
_cached_model: Optional[SentenceTransformer] = None


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model on the specified device.

    The model is loaded into memory once and reused across subsequent calls.
    The first invocation downloads the model files from Hugging Face if not cached.

    Args:
        model_name: Identifier of the SentenceTransformer model.
        device: Target execution device ('cpu' by default).

    Returns:
        Loaded SentenceTransformer instance.

    Raises:
        RuntimeError: If the model cannot be downloaded or loaded.
    """
    global _cached_model

    if _cached_model is None:
        logger.info(
            f"Loading embedding model '{model_name}' on device '{device}'. "
            "If this is the first run, model weights will be downloaded to local cache."
        )
        try:
            _cached_model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                f"Ensure internet connectivity for the initial download or check model name. "
                f"Error: {e}"
            )

    return _cached_model


def get_max_seq_length() -> int:
    """Return the maximum sequence length (in tokens) accepted by the model.

    Text longer than this limit will be truncated by the model's tokenizer.
    For all-MiniLM-L6-v2, this is typically 256 tokens (roughly 1,000 characters).
    """
    model = get_embedding_model()
    return int(getattr(model, "max_seq_length", 256))


def get_embeddings(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
) -> List[List[float]]:
    """Generate normalized dense embeddings for a batch of text chunks.

    Args:
        texts: List of text strings to embed.
        model: Optional pre-loaded or mock SentenceTransformer instance.

    Returns:
        List of 384-dimensional float vectors preserving input order.

    Raises:
        TypeError: If input is not a list or elements are not strings.
        ValueError: If resulting embedding dimensions do not match EMBEDDING_DIMENSION.
    """
    if not isinstance(texts, list):
        raise TypeError(f"Expected a list of strings, got {type(texts).__name__}.")

    if len(texts) == 0:
        return []

    # Validate that every element is a string
    for idx, item in enumerate(texts):
        if not isinstance(item, str):
            raise TypeError(
                f"Item at index {idx} is of type {type(item).__name__}; expected str."
            )

    encoder = model if model is not None else get_embedding_model()

    # Generate normalized embeddings; convert to numpy then Python list
    raw_embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
    )

    # Convert to pure Python float lists
    embeddings_list: List[List[float]] = [
        [float(val) for val in vector] for vector in raw_embeddings
    ]

    # Validate dimensions
    for idx, vec in enumerate(embeddings_list):
        if len(vec) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding at index {idx} has dimension {len(vec)}; "
                f"expected {EMBEDDING_DIMENSION}."
            )

    return embeddings_list


def get_query_embedding(
    query: str,
    model: Optional[SentenceTransformer] = None,
) -> List[float]:
    """Generate a normalized dense embedding for a single user query.

    Args:
        query: The user's search query string.
        model: Optional pre-loaded or mock SentenceTransformer instance.

    Returns:
        A single 384-dimensional float vector.

    Raises:
        TypeError: If query is not a string.
        ValueError: If query is empty or whitespace-only, or output dimension is invalid.
    """
    if not isinstance(query, str):
        raise TypeError(f"Query must be a string, got {type(query).__name__}.")

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty or whitespace-only.")

    encoder = model if model is not None else get_embedding_model()

    raw_embedding = encoder.encode(
        cleaned_query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vector: List[float] = [float(val) for val in raw_embedding]

    if len(query_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query embedding has dimension {len(query_vector)}; "
            f"expected {EMBEDDING_DIMENSION}."
        )

    return query_vector

