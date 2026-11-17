"""Unit tests for app.retriever.

Tests cover:
- Single-query retrieval.
- Top-K retrieval parameter constraints:
  - Rejection of booleans (TypeError).
  - Rejection of non-integers (TypeError).
  - Rejection of <= 0 integers (ValueError).
- Relevance ordering (results sorted by lowest cosine distance first).
- Integration with get_query_embedding.
- Empty collection handling and clean result logging.
- Validation of consistent ChromaDB response array lengths.
- Returned SearchResult dataclass fields.
"""

import logging
from unittest.mock import MagicMock, patch
import uuid
import chromadb
import pytest

from app.embeddings import EMBEDDING_DIMENSION
from app.retriever import SearchResult, retrieve


@pytest.fixture
def populated_collection():
    """Create an isolated in-memory collection populated with test vectors."""
    client = chromadb.EphemeralClient()
    col_name = f"test_retriever_{uuid.uuid4().hex[:8]}"
    col = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 3 distinct vectors with differing cosine similarities
    v_ai = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
    v_data = [0.8, 0.6] + [0.0] * (EMBEDDING_DIMENSION - 2)
    v_baking = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSION - 2)

    col.upsert(
        ids=["c_ai", "c_data", "c_baking"],
        documents=[
            "Artificial intelligence systems learn from experience.",
            "Data analysis and statistical patterns.",
            "Baking bread with flour, water, and yeast.",
        ],
        embeddings=[v_ai, v_data, v_baking],
        metadatas=[
            {"source": "ai.pdf", "page": 1, "doc_id": "d1"},
            {"source": "data.pdf", "page": 2, "doc_id": "d2"},
            {"source": "recipes.pdf", "page": 5, "doc_id": "d3"},
        ],
    )
    yield col
    try:
        client.delete_collection(col_name)
    except Exception:
        pass


@patch("app.retriever.get_query_embedding")
def test_single_query_retrieval(mock_query_emb, populated_collection):
    """Retrieve should return a list of SearchResult objects with correct fields."""
    mock_query_emb.return_value = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)

    results = retrieve("Tell me about artificial intelligence", top_k=1, collection=populated_collection)

    assert len(results) == 1
    top = results[0]
    assert isinstance(top, SearchResult)
    assert top.chunk_id == "c_ai"
    assert "Artificial intelligence" in top.text
    assert top.metadata["source"] == "ai.pdf"
    assert top.metadata["page"] == 1
    # Perfect match has cosine distance ~ 0.0
    assert abs(top.distance) < 1e-4


@patch("app.retriever.get_query_embedding")
def test_top_k_retrieval_and_clamping(mock_query_emb, populated_collection):
    """Retriever should honor top_k and clamp if top_k exceeds total chunks."""
    mock_query_emb.return_value = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)

    # Request top_k = 2
    res_2 = retrieve("AI", top_k=2, collection=populated_collection)
    assert len(res_2) == 2

    # Request top_k = 10 (exceeds collection count of 3) -> clamped to 3
    res_all = retrieve("AI", top_k=10, collection=populated_collection)
    assert len(res_all) == 3


@patch("app.retriever.get_query_embedding")
def test_relevance_ordering(mock_query_emb, populated_collection):
    """Results must be sorted ascending by distance (most relevant first)."""
    mock_query_emb.return_value = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)

    results = retrieve("AI query", top_k=3, collection=populated_collection)

    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].distance <= results[i + 1].distance

    assert results[0].chunk_id == "c_ai"
    assert results[-1].chunk_id == "c_baking"


def test_empty_database_handling(caplog):
    """Querying an empty database should return an empty list and log appropriately."""
    client = chromadb.EphemeralClient()
    empty_col = client.get_or_create_collection(f"empty_{uuid.uuid4().hex[:8]}")

    with caplog.at_level(logging.INFO):
        results = retrieve("any query", top_k=3, collection=empty_col)

    assert results == []
    assert "ChromaDB collection is empty. Returning 0 results for query: 'any query'" in caplog.text


@patch("app.retriever.get_query_embedding")
def test_empty_search_results_logging(mock_query_emb, caplog):
    """When query returns zero results, it should log an informative info message."""
    mock_query_emb.return_value = [0.1] * EMBEDDING_DIMENSION
    mock_col = MagicMock()
    mock_col.count.return_value = 5
    # Return empty response lists
    mock_col.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    with caplog.at_level(logging.INFO):
        results = retrieve("unmatched query", top_k=2, collection=mock_col)

    assert results == []
    assert "No matching document chunks found for query: 'unmatched query'" in caplog.text


def test_invalid_query_handling(populated_collection):
    """Invalid query strings must raise TypeError or ValueError."""
    with pytest.raises(TypeError, match="Query must be a string"):
        retrieve(None, collection=populated_collection)  # type: ignore

    with pytest.raises(TypeError, match="Query must be a string"):
        retrieve(12345, collection=populated_collection)  # type: ignore

    with pytest.raises(ValueError, match="cannot be empty"):
        retrieve("", collection=populated_collection)

    with pytest.raises(ValueError, match="cannot be empty"):
        retrieve("   \t  \n", collection=populated_collection)


def test_invalid_top_k_handling(populated_collection):
    """Invalid top_k types (including booleans) and non-positive numbers must raise appropriate errors."""
    # Booleans explicitly rejected
    with pytest.raises(TypeError, match="top_k must be an integer, got bool"):
        retrieve("Valid query", top_k=True, collection=populated_collection)  # type: ignore

    with pytest.raises(TypeError, match="top_k must be an integer, got bool"):
        retrieve("Valid query", top_k=False, collection=populated_collection)  # type: ignore

    # Non-integers rejected
    with pytest.raises(TypeError, match="top_k must be an integer"):
        retrieve("Valid query", top_k="5", collection=populated_collection)  # type: ignore

    with pytest.raises(TypeError, match="top_k must be an integer"):
        retrieve("Valid query", top_k=3.5, collection=populated_collection)  # type: ignore

    # Non-positive numbers rejected
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retrieve("Valid query", top_k=0, collection=populated_collection)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retrieve("Valid query", top_k=-5, collection=populated_collection)


@patch("app.retriever.get_query_embedding")
def test_inconsistent_response_lengths_raises_error(mock_query_emb):
    """If ChromaDB returns inconsistent list lengths across ids/documents/metadatas/distances, raise ValueError."""
    mock_query_emb.return_value = [0.1] * EMBEDDING_DIMENSION
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2

    # Inconsistent response: 2 ids, but only 1 document
    mock_collection.query.return_value = {
        "ids": [["c1", "c2"]],
        "documents": [["doc 1 only"]],
        "metadatas": [[{"page": 1}, {"page": 2}]],
        "distances": [[0.1, 0.2]],
    }

    with pytest.raises(ValueError, match="Inconsistent ChromaDB response lengths"):
        retrieve("Test query", top_k=2, collection=mock_collection)
