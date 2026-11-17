"""Unit tests for app.text_splitter.

Tests cover:
- Text shorter than chunk size.
- Text exactly equal to chunk size.
- Text longer than chunk size.
- Overlap between consecutive chunks.
- Empty and whitespace-only text.
- Metadata preservation (source, page, total_pages, doc_id).
- Deterministic chunk IDs.
- Invalid chunk_size and chunk_overlap parameter validation.
"""

import pytest

from app.document_loader import Document
from app.text_splitter import split_documents, split_text, validate_chunk_parameters


def test_text_shorter_than_chunk_size():
    """Text shorter than chunk_size should return as a single chunk."""
    text = "Short text passage."
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0] == "Short text passage."


def test_text_exactly_equal_to_chunk_size():
    """Text exactly equal to chunk_size should return as a single chunk."""
    text = "A" * 50
    chunks = split_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) == 1
    assert len(chunks[0]) == 50
    assert chunks[0] == text


def test_text_longer_than_chunk_size():
    """Text longer than chunk_size should split into multiple chunks at word boundaries."""
    text = (
        "Artificial intelligence and machine learning allow computers to learn patterns. "
        "Retrieval-Augmented Generation enhances generative AI by attaching external memory. "
        "This ensures answers are grounded in actual facts rather than hallucinations."
    )
    chunks = split_text(text, chunk_size=80, chunk_overlap=15)

    assert len(chunks) > 1
    # Check that all chunks are non-empty and respect the approximate length
    for chunk in chunks:
        assert len(chunk) > 0
        assert len(chunk) <= 85  # Allowing small word boundary margin


def test_overlap_between_consecutive_chunks():
    """Consecutive chunks must share overlapping words."""
    sentence = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima."
    chunks = split_text(sentence, chunk_size=40, chunk_overlap=15)

    assert len(chunks) >= 2
    # Check that words from the end of chunk 0 appear in chunk 1
    words_chunk_0 = set(chunks[0].split())
    words_chunk_1 = set(chunks[1].split())
    overlap_words = words_chunk_0.intersection(words_chunk_1)
    assert len(overlap_words) > 0, "Consecutive chunks must share overlapping words."


def test_empty_and_whitespace_only_text():
    """Empty or whitespace-only inputs should produce an empty list of chunks."""
    assert split_text("", chunk_size=100, chunk_overlap=10) == []
    assert split_text("   \n\t  \n  ", chunk_size=100, chunk_overlap=10) == []


def test_invalid_parameters():
    """Invalid chunk sizes or overlaps must raise ValueError."""
    # chunk_size <= 0
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        validate_chunk_parameters(chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        validate_chunk_parameters(chunk_size=-10, chunk_overlap=5)

    # chunk_overlap < 0
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        validate_chunk_parameters(chunk_size=100, chunk_overlap=-1)

    # chunk_overlap >= chunk_size
    with pytest.raises(ValueError, match="strictly smaller"):
        validate_chunk_parameters(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="strictly smaller"):
        validate_chunk_parameters(chunk_size=100, chunk_overlap=150)


def test_split_documents_metadata_preservation():
    """Chunked documents must retain all page metadata and add chunk_id."""
    doc = Document(
        text="Sentence one about AI. Sentence two about vector search. Sentence three about ChromaDB.",
        metadata={
            "source": "handbook.pdf",
            "file_path": "/data/handbook.pdf",
            "page": 3,
            "total_pages": 10,
            "doc_id": "abc123hash",
        },
    )

    chunks = split_documents([doc], chunk_size=45, chunk_overlap=10)

    assert len(chunks) > 1

    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "handbook.pdf"
        assert chunk.metadata["file_path"] == "/data/handbook.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["total_pages"] == 10
        assert chunk.metadata["doc_id"] == "abc123hash"
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["chunk_id"] == f"abc123hash_p3_c{idx}"
        assert len(chunk.text.strip()) > 0


def test_deterministic_chunk_ids():
    """Processing the exact same document twice must produce identical chunk IDs."""
    doc = Document(
        text="Deterministic test passage for RAG pipeline ID consistency.",
        metadata={
            "source": "paper.pdf",
            "page": 1,
            "total_pages": 1,
            "doc_id": "sha256fixedvalue",
        },
    )

    run_1 = split_documents([doc], chunk_size=30, chunk_overlap=5)
    run_2 = split_documents([doc], chunk_size=30, chunk_overlap=5)

    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2):
        assert c1.metadata["chunk_id"] == c2.metadata["chunk_id"]
        assert c1.text == c2.text


def test_multiple_pages_split_independently():
    """Pages must be split independently so page attribution is never mixed."""
    p1 = Document(
        text="Page one text content.",
        metadata={"source": "book.pdf", "page": 1, "doc_id": "book_hash"},
    )
    p2 = Document(
        text="Page two text content.",
        metadata={"source": "book.pdf", "page": 2, "doc_id": "book_hash"},
    )

    chunks = split_documents([p1, p2], chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["chunk_id"] == "book_hash_p1_c0"
    assert chunks[1].metadata["page"] == 2
    assert chunks[1].metadata["chunk_id"] == "book_hash_p2_c0"

