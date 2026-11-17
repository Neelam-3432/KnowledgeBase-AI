"""Unit tests for app.vector_store.

Uses in-memory ChromaDB EphemeralClient and mocked embeddings to test:
- Initialization and persistent path configuration.
- Collection creation, configuration (cosine space), and reuse.
- Adding single and multiple chunks with batching.
- Record updates and duplicate ID handling.
- Validation:
  - batch_size constraints (rejects booleans, non-ints, and <= 0).
  - embedding count mismatch from get_embeddings.
  - invalid embedding dimensions, non-list, and non-finite values (NaN/Inf/bool).
  - missing or non-string chunk_id and doc_id.
  - metadata logging for unsupported types vs safe omission of None.
- Stored chunk count, document index verification, and deletion.
- reset_collection handling of NotFoundError vs genuine errors.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import uuid
import chromadb
from chromadb.errors import InternalError, NotFoundError
import pytest

from app.document_loader import Document
from app.embeddings import EMBEDDING_DIMENSION
from app.vector_store import (
    add_documents,
    delete_document_by_doc_id,
    get_chroma_client,
    get_collection,
    get_document_count,
    is_document_indexed,
    reset_collection,
)


@pytest.fixture
def memory_collection():
    """Create a fresh isolated in-memory ChromaDB collection for each unit test."""
    client = chromadb.EphemeralClient()
    col_name = f"test_col_{uuid.uuid4().hex[:8]}"
    col = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )
    yield col
    try:
        client.delete_collection(col_name)
    except Exception:
        pass


def make_dummy_embedding(val: float = 0.1, dim: int = EMBEDDING_DIMENSION) -> list[float]:
    """Helper to return a dummy embedding vector."""
    return [val] * dim


def test_client_initialization_persistent(tmp_path: Path):
    """Client should initialize a persistent client at the specified path."""
    custom_path = tmp_path / "custom_chroma"
    client = get_chroma_client(persist_path=custom_path)
    assert hasattr(client, "get_or_create_collection")
    assert custom_path.exists()


def test_collection_creation_and_reuse():
    """Collection should be created with cosine space and reused on next call."""
    client = chromadb.EphemeralClient()
    col_name = f"col_{uuid.uuid4().hex[:8]}"
    col1 = get_collection(client=client, collection_name=col_name)
    assert col1.metadata.get("hnsw:space") == "cosine"

    # Re-retrieval should return the existing collection without error
    col2 = get_collection(client=client, collection_name=col_name)
    assert col1.name == col2.name


@patch("app.vector_store.get_embeddings")
def test_add_single_chunk(mock_embeddings, memory_collection):
    """Adding a single chunk should insert ID, text, embedding, and metadata."""
    mock_embeddings.return_value = [make_dummy_embedding(0.1)]

    doc = Document(
        text="Single chunk of knowledge.",
        metadata={
            "source": "manual.pdf",
            "page": 1,
            "chunk_id": "doc1_p1_c0",
            "doc_id": "doc1",
        },
    )

    added = add_documents([doc], collection=memory_collection)
    assert added == 1
    assert get_document_count(memory_collection) == 1

    stored = memory_collection.get(ids=["doc1_p1_c0"], include=["documents", "metadatas"])
    assert stored["documents"][0] == "Single chunk of knowledge."
    assert stored["metadatas"][0]["source"] == "manual.pdf"
    assert stored["metadatas"][0]["page"] == 1
    assert stored["metadatas"][0]["doc_id"] == "doc1"


@patch("app.vector_store.get_embeddings")
def test_add_multiple_chunks_with_batching(mock_embeddings, memory_collection):
    """Adding multiple chunks should batch them and insert all records."""
    docs = [
        Document(
            text=f"Chunk content number {i}",
            metadata={"source": "book.pdf", "page": 1, "chunk_id": f"hash_p1_c{i}", "doc_id": "hash"},
        )
        for i in range(5)
    ]
    mock_embeddings.side_effect = lambda texts: [make_dummy_embedding(0.1) for _ in texts]

    added = add_documents(docs, collection=memory_collection, batch_size=2)
    assert added == 5
    assert get_document_count(memory_collection) == 5


@patch("app.vector_store.get_embeddings")
def test_updating_records_and_duplicate_id_handling(mock_embeddings, memory_collection):
    """Upserting an existing ID should update the document rather than duplicate it."""
    mock_embeddings.return_value = [make_dummy_embedding(0.1)]

    doc_v1 = Document(
        text="Initial version of text.",
        metadata={"source": "notes.pdf", "page": 1, "chunk_id": "note_p1_c0", "doc_id": "note1"},
    )
    add_documents([doc_v1], collection=memory_collection)
    assert get_document_count(memory_collection) == 1

    # Insert updated version with identical chunk_id
    mock_embeddings.return_value = [make_dummy_embedding(0.2)]
    doc_v2 = Document(
        text="Updated revision of text.",
        metadata={"source": "notes.pdf", "page": 1, "chunk_id": "note_p1_c0", "doc_id": "note1"},
    )
    add_documents([doc_v2], collection=memory_collection)

    # Document count must still be 1 (no duplicates)
    assert get_document_count(memory_collection) == 1

    # Content must reflect updated text
    stored = memory_collection.get(ids=["note_p1_c0"], include=["documents"])
    assert stored["documents"][0] == "Updated revision of text."


def test_batch_size_validation(memory_collection):
    """batch_size must reject booleans, non-integers, and values <= 0."""
    doc = Document(text="T", metadata={"chunk_id": "c1", "doc_id": "d1"})

    # Booleans are rejected
    with pytest.raises(TypeError, match="batch_size must be an integer, got bool"):
        add_documents([doc], collection=memory_collection, batch_size=True)  # type: ignore

    with pytest.raises(TypeError, match="batch_size must be an integer, got bool"):
        add_documents([doc], collection=memory_collection, batch_size=False)  # type: ignore

    # Non-integers rejected
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        add_documents([doc], collection=memory_collection, batch_size="5")  # type: ignore

    with pytest.raises(TypeError, match="batch_size must be an integer"):
        add_documents([doc], collection=memory_collection, batch_size=2.5)  # type: ignore

    # <= 0 rejected
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        add_documents([doc], collection=memory_collection, batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        add_documents([doc], collection=memory_collection, batch_size=-3)


@patch("app.vector_store.get_embeddings")
def test_embedding_count_mismatch(mock_embeddings, memory_collection):
    """If get_embeddings returns fewer or more embeddings than inputs, raise ValueError."""
    # 2 input docs, but mock returns only 1 embedding
    mock_embeddings.return_value = [make_dummy_embedding(0.1)]

    docs = [
        Document(text="Text 1", metadata={"chunk_id": "c1", "doc_id": "d1"}),
        Document(text="Text 2", metadata={"chunk_id": "c2", "doc_id": "d1"}),
    ]

    with pytest.raises(ValueError, match="Embedding count mismatch: expected 2"):
        add_documents(docs, collection=memory_collection)


@patch("app.vector_store.get_embeddings")
def test_invalid_embedding_dimensions(mock_embeddings, memory_collection):
    """If embedding dimension is not 384, ValueError must be raised."""
    mock_embeddings.return_value = [[0.1] * 128]

    doc = Document(
        text="Sample text.",
        metadata={"chunk_id": "chk_1", "doc_id": "d1"},
    )

    with pytest.raises(ValueError, match="expected 384"):
        add_documents([doc], collection=memory_collection)


@patch("app.vector_store.get_embeddings")
def test_non_finite_embedding_values(mock_embeddings, memory_collection):
    """Embeddings containing NaN or Inf must raise ValueError."""
    bad_emb_nan = make_dummy_embedding(0.1)
    bad_emb_nan[5] = float("nan")
    mock_embeddings.return_value = [bad_emb_nan]

    doc = Document(text="Sample text.", metadata={"chunk_id": "c1", "doc_id": "d1"})
    with pytest.raises(ValueError, match="non-finite value 'nan'"):
        add_documents([doc], collection=memory_collection)

    bad_emb_inf = make_dummy_embedding(0.1)
    bad_emb_inf[10] = float("inf")
    mock_embeddings.return_value = [bad_emb_inf]
    with pytest.raises(ValueError, match="non-finite value 'inf'"):
        add_documents([doc], collection=memory_collection)


@patch("app.vector_store.get_embeddings")
def test_metadata_validation_and_unsupported_fields(mock_embeddings, memory_collection, caplog):
    """Unsupported metadata values (e.g. lists/dicts) must be logged and dropped; primitives kept."""
    mock_embeddings.return_value = [make_dummy_embedding(0.1)]

    doc = Document(
        text="Text with rich metadata.",
        metadata={
            "chunk_id": "c_meta",
            "doc_id": "doc_meta",
            "source": "paper.pdf",
            "page": 3,
            "score": 0.95,
            "is_valid": True,
            "nested_tags": ["ai", "rag"],       # Unsupported in ChromaDB
            "extra_info": {"author": "Alice"},    # Unsupported in ChromaDB
        },
    )

    with caplog.at_level(logging.WARNING):
        added = add_documents([doc], collection=memory_collection)

    assert added == 1
    # Verify warnings were logged for the dropped fields
    assert "Dropping unsupported metadata field 'nested_tags'" in caplog.text
    assert "Dropping unsupported metadata field 'extra_info'" in caplog.text

    # Verify stored metadata has primitive fields preserved and nested dropped
    stored = memory_collection.get(ids=["c_meta"], include=["metadatas"])
    stored_meta = stored["metadatas"][0]
    assert stored_meta["source"] == "paper.pdf"
    assert stored_meta["page"] == 3
    assert stored_meta["score"] == 0.95
    assert stored_meta["is_valid"] is True
    assert "nested_tags" not in stored_meta
    assert "extra_info" not in stored_meta


@patch("app.vector_store.get_embeddings")
def test_metadata_none_values_omitted_without_warning(mock_embeddings, memory_collection, caplog):
    """Optional metadata fields with None values should be omitted safely without noisy warnings."""
    mock_embeddings.return_value = [make_dummy_embedding(0.1)]

    doc = Document(
        text="Text with optional None metadata.",
        metadata={
            "chunk_id": "c_none",
            "doc_id": "doc_none",
            "source": "report.pdf",
            "optional_field": None,
        },
    )

    with caplog.at_level(logging.WARNING):
        added = add_documents([doc], collection=memory_collection)

    assert added == 1
    # Verify no warning was logged for the optional None field
    assert "optional_field" not in caplog.text

    stored = memory_collection.get(ids=["c_none"], include=["metadatas"])
    stored_meta = stored["metadatas"][0]
    assert stored_meta["chunk_id"] == "c_none"
    assert stored_meta["doc_id"] == "doc_none"
    assert stored_meta["source"] == "report.pdf"
    assert stored_meta["page"] == 1  # defaulted safely
    assert "optional_field" not in stored_meta


def test_missing_doc_id_raises_error(memory_collection):
    """Documents missing a valid 'doc_id' must raise ValueError."""
    doc_no_doc_id = Document(text="Valid text", metadata={"chunk_id": "c1"})
    with pytest.raises(ValueError, match="missing a valid 'doc_id'"):
        add_documents([doc_no_doc_id], collection=memory_collection)

    doc_empty_doc_id = Document(text="Valid text", metadata={"chunk_id": "c1", "doc_id": ""})
    with pytest.raises(ValueError, match="missing a valid 'doc_id'"):
        add_documents([doc_empty_doc_id], collection=memory_collection)


def test_invalid_metadata_type_raises_error(memory_collection):
    """Documents with non-dict metadata must raise TypeError."""
    doc_bad_meta = Document(text="Valid text", metadata="not_a_dict")  # type: ignore
    with pytest.raises(TypeError, match="invalid metadata; expected dict"):
        add_documents([doc_bad_meta], collection=memory_collection)


def test_empty_input_handling(memory_collection):
    """Empty document list should return 0 without error."""
    added = add_documents([], collection=memory_collection)
    assert added == 0
    assert get_document_count(memory_collection) == 0


def test_invalid_document_types(memory_collection):
    """Invalid inputs (non-list, non-Document, empty text, missing chunk_id) must raise errors."""
    with pytest.raises(TypeError, match="Expected a list of Document objects"):
        add_documents("not a list", collection=memory_collection)  # type: ignore

    with pytest.raises(TypeError, match="expected Document"):
        add_documents(["a string item"], collection=memory_collection)  # type: ignore

    # Empty text
    doc_empty_text = Document(text="", metadata={"chunk_id": "id1", "doc_id": "d1"})
    with pytest.raises(ValueError, match="empty text"):
        add_documents([doc_empty_text], collection=memory_collection)

    # Missing chunk_id
    doc_no_id = Document(text="Valid text", metadata={"doc_id": "d1"})
    with pytest.raises(ValueError, match="missing a valid 'chunk_id'"):
        add_documents([doc_no_id], collection=memory_collection)


@patch("app.vector_store.get_embeddings")
def test_is_document_indexed_and_deletion(mock_embeddings, memory_collection):
    """Check is_document_indexed and delete_document_by_doc_id functionality."""
    mock_embeddings.side_effect = lambda texts: [make_dummy_embedding(0.1) for _ in texts]

    doc1 = Document(text="T1", metadata={"chunk_id": "c1", "doc_id": "doc_A"})
    doc2 = Document(text="T2", metadata={"chunk_id": "c2", "doc_id": "doc_A"})
    doc3 = Document(text="T3", metadata={"chunk_id": "c3", "doc_id": "doc_B"})

    add_documents([doc1, doc2, doc3], collection=memory_collection)
    assert get_document_count(memory_collection) == 3

    assert is_document_indexed("doc_A", collection=memory_collection) is True
    assert is_document_indexed("doc_B", collection=memory_collection) is True
    assert is_document_indexed("doc_C", collection=memory_collection) is False

    # Delete doc_A
    deleted_count = delete_document_by_doc_id("doc_A", collection=memory_collection)
    assert deleted_count == 2
    assert get_document_count(memory_collection) == 1
    assert is_document_indexed("doc_A", collection=memory_collection) is False
    assert is_document_indexed("doc_B", collection=memory_collection) is True


def test_reset_collection_handles_not_found():
    """reset_collection should handle non-existent collections cleanly without error."""
    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = NotFoundError("Collection not found")
    mock_client.get_or_create_collection.return_value = MagicMock()

    col = reset_collection(collection_name="new_col", client=mock_client)
    assert col is not None
    mock_client.delete_collection.assert_called_once_with(name="new_col")


def test_reset_collection_raises_on_genuine_chroma_error():
    """reset_collection should re-raise unexpected ChromaError database errors."""
    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = InternalError("Fatal database corruption")

    with pytest.raises(InternalError, match="Fatal database corruption"):
        reset_collection(collection_name="corrupt_col", client=mock_client)
