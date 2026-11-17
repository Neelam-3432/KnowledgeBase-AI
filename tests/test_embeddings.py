"""Unit tests for app.embeddings using mocked models.

Tests run offline without downloading the model or relying on internet access.
Tests cover:
- Single-text (query) embedding generation.
- Multiple-text batch embedding generation.
- Output dimension verification (384).
- Input/output ordering preservation.
- Empty input handling.
- Invalid input handling (non-list, non-string, empty query).
- Model caching (loaded only once).
- Query embedding generation using the same model.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import app.embeddings as emb_module
from app.embeddings import (
    EMBEDDING_DIMENSION,
    get_embedding_model,
    get_embeddings,
    get_max_seq_length,
    get_query_embedding,
)


@pytest.fixture(autouse=True)
def reset_cached_model():
    """Reset the module-level cached model before and after each test."""
    emb_module._cached_model = None
    yield
    emb_module._cached_model = None


def make_mock_vector(val: float = 0.1, dim: int = EMBEDDING_DIMENSION) -> np.ndarray:
    """Create a mock normalized numpy vector of the required dimension."""
    vec = np.full(dim, val, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1.0)


def test_batch_embeddings_generation_and_dimension():
    """Batch embeddings should return normalized 384-dimensional vectors."""
    mock_model = MagicMock()
    v1 = make_mock_vector(0.1)
    v2 = make_mock_vector(0.2)
    mock_model.encode.return_value = np.array([v1, v2])

    texts = ["First passage about AI.", "Second passage about ChromaDB."]
    embeddings = get_embeddings(texts, model=mock_model)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == EMBEDDING_DIMENSION
    assert len(embeddings[1]) == EMBEDDING_DIMENSION
    assert isinstance(embeddings[0], list)
    assert isinstance(embeddings[0][0], float)

    # Verify encode was called with correct parameters
    mock_model.encode.assert_called_once_with(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
    )


def test_input_output_ordering_preserved():
    """Output embeddings must preserve the exact ordering of input texts."""
    mock_model = MagicMock()
    # Create distinct vectors to distinguish them
    v1 = make_mock_vector(0.1)
    v2 = make_mock_vector(0.5)
    v3 = make_mock_vector(0.9)
    mock_model.encode.return_value = np.array([v1, v2, v3])

    texts = ["Alpha", "Beta", "Gamma"]
    embeddings = get_embeddings(texts, model=mock_model)

    assert len(embeddings) == 3
    # Check that vector 0 matches v1, vector 1 matches v2, vector 2 matches v3
    assert pytest.approx(embeddings[0][0]) == pytest.approx(float(v1[0]))
    assert pytest.approx(embeddings[1][0]) == pytest.approx(float(v2[0]))
    assert pytest.approx(embeddings[2][0]) == pytest.approx(float(v3[0]))


def test_empty_input_handling():
    """Passing an empty list of texts should return an empty list without calling model."""
    mock_model = MagicMock()
    embeddings = get_embeddings([], model=mock_model)

    assert embeddings == []
    mock_model.encode.assert_not_called()


def test_invalid_batch_input_types():
    """Invalid input types must raise TypeError."""
    mock_model = MagicMock()

    # Input not a list
    with pytest.raises(TypeError, match="Expected a list of strings"):
        get_embeddings("A single string instead of list", model=mock_model)  # type: ignore

    # Element inside list is not a string
    with pytest.raises(TypeError, match="expected str"):
        get_embeddings(["Valid string", 12345], model=mock_model)  # type: ignore


def test_single_query_embedding():
    """Single query embedding should return a 384-dimensional float vector."""
    mock_model = MagicMock()
    mock_vec = make_mock_vector(0.42)
    mock_model.encode.return_value = mock_vec

    query = "What is Retrieval-Augmented Generation?"
    vec = get_query_embedding(query, model=mock_model)

    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIMENSION
    assert isinstance(vec[0], float)

    mock_model.encode.assert_called_once_with(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def test_invalid_query_input():
    """Empty or non-string queries must raise appropriate errors."""
    mock_model = MagicMock()

    # Non-string input
    with pytest.raises(TypeError, match="Query must be a string"):
        get_query_embedding(None, model=mock_model)  # type: ignore

    # Empty or whitespace-only string
    with pytest.raises(ValueError, match="cannot be empty"):
        get_query_embedding("", model=mock_model)

    with pytest.raises(ValueError, match="cannot be empty"):
        get_query_embedding("   \t\n  ", model=mock_model)


def test_model_loaded_only_once():
    """The SentenceTransformer model must be cached and loaded only once."""
    fake_instance = MagicMock()
    fake_instance.max_seq_length = 256

    with patch("app.embeddings.SentenceTransformer", return_value=fake_instance) as mock_cls:
        m1 = get_embedding_model("test-model")
        m2 = get_embedding_model("test-model")

        # Must return identical cached object
        assert m1 is m2
        # Constructor called exactly once
        mock_cls.assert_called_once_with("test-model", device="cpu")


def test_model_loading_failure_handling():
    """If model loading fails, a clean RuntimeError must be raised."""
    with patch("app.embeddings.SentenceTransformer", side_effect=Exception("Connection refused")):
        with pytest.raises(RuntimeError, match="Failed to load embedding model"):
            get_embedding_model("non-existent-model")


def test_max_seq_length_exposure():
    """Max sequence length should be exposed correctly from the underlying model."""
    fake_instance = MagicMock()
    fake_instance.max_seq_length = 256

    with patch("app.embeddings.SentenceTransformer", return_value=fake_instance):
        seq_len = get_max_seq_length()
        assert seq_len == 256

