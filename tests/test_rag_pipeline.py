"""Unit tests for app.rag_pipeline.

Tests cover:
- Document Ingestion:
  1. Successful PDF ingestion
  2. Empty PDF handling (zero-byte & scanned/blank)
  3. Invalid PDF path & extension handling
  4. Correct sequential order of ingestion operations
  5. Correct embedding and vector storage integration
  6. Duplicate ingestion prevention
  7. Metadata preservation throughout ingestion
  8. Ingestion failure handling (no swallowed exceptions)
- Question Answering:
  9. Successful question answering
  10. Exact retrieved SearchResult chunks passed to answer generator
  11. Empty retrieval skips Gemini API call completely
  12. Source and citation metadata preservation
  13. Retrieval failure handling
  14. Answer generation error handling
  15. Strict query and top_k parameter validation
  16. Pipeline status check
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch
import pytest

from app.answer_generator import GeneratedAnswer
from app.document_loader import Document
from app.rag_pipeline import (
    IngestionResult,
    RAGAnswer,
    answer_question,
    get_pipeline_status,
    ingest_document,
)
from app.retriever import SearchResult


# -----------------------------------------------------------------------------
# Document Ingestion Tests (1-8)
# -----------------------------------------------------------------------------

def test_successful_pdf_ingestion(tmp_path: Path):
    """1. Verify standard successful PDF ingestion end-to-end with mocks."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy valid content")

    mock_docs = [
        Document(text="Page 1 text", metadata={"source": "sample.pdf", "page": 1, "total_pages": 1, "doc_id": "hash123"})
    ]
    mock_chunks = [
        Document(text="Chunk 1 text", metadata={"source": "sample.pdf", "page": 1, "chunk_id": "hash123_p1_c0", "doc_id": "hash123"})
    ]

    with (
        patch("app.rag_pipeline.calculate_file_hash", return_value="hash123"),
        patch("app.rag_pipeline.is_document_indexed", return_value=False),
        patch("app.rag_pipeline.load_pdf", return_value=mock_docs) as mock_load,
        patch("app.rag_pipeline.split_documents", return_value=mock_chunks) as mock_split,
        patch("app.rag_pipeline.add_documents", return_value=1) as mock_add,
    ):
        result = ingest_document(pdf_file)

        assert isinstance(result, IngestionResult)
        assert result.status == "success"
        assert result.chunks_count == 1
        assert result.document_name == "sample.pdf"
        assert result.doc_id == "hash123"
        assert result.metadata["total_pages"] == 1

        mock_load.assert_called_once_with(pdf_file)
        mock_split.assert_called_once_with(mock_docs)
        mock_add.assert_called_once_with(mock_chunks, collection=mock_add.call_args.kwargs.get("collection"))


def test_empty_pdf_handling(tmp_path: Path):
    """2. Verify handling of zero-byte PDFs and scanned/blank PDFs with no text."""
    # Sub-case A: 0-byte file raises ValueError
    zero_file = tmp_path / "zero.pdf"
    zero_file.write_bytes(b"")

    with pytest.raises(ValueError, match="0 bytes"):
        ingest_document(zero_file)

    # Sub-case B: Valid PDF format but 0 extractable text (e.g. scanned image)
    blank_file = tmp_path / "blank.pdf"
    blank_file.write_bytes(b"%PDF-1.4 dummy scanned content")

    with (
        patch("app.rag_pipeline.calculate_file_hash", return_value="hash_blank"),
        patch("app.rag_pipeline.is_document_indexed", return_value=False),
        patch("app.rag_pipeline.load_pdf", return_value=[]),
        patch("app.rag_pipeline.split_documents") as mock_split,
        patch("app.rag_pipeline.add_documents") as mock_add,
    ):
        result = ingest_document(blank_file)

        assert result.status == "no_text_extracted"
        assert result.chunks_count == 0
        assert "scanned image or blank" in result.message
        mock_split.assert_not_called()
        mock_add.assert_not_called()


def test_invalid_pdf_path(tmp_path: Path):
    """3. Verify rejection of missing files, non-PDF files, and invalid path types."""
    # Missing file
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        ingest_document(missing)

    # Non-PDF extension
    text_file = tmp_path / "notes.txt"
    text_file.write_text("Hello world")
    with pytest.raises(ValueError, match="Expected '.pdf' extension"):
        ingest_document(text_file)

    # Invalid path types
    with pytest.raises(TypeError, match="must be a string or Path"):
        ingest_document(12345)  # type: ignore

    with pytest.raises(ValueError, match="cannot be empty"):
        ingest_document("   ")


def test_correct_order_of_ingestion_operations(tmp_path: Path):
    """4. Verify that hashing, duplicate check, loading, splitting, and storing execute in sequence."""
    pdf_file = tmp_path / "order_test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    call_order = []

    def mock_hash(path):
        call_order.append("hash")
        return "hash_order"

    def mock_is_indexed(doc_id, collection=None):
        call_order.append("is_indexed")
        return False

    def mock_load(path):
        call_order.append("load")
        return [Document(text="Text", metadata={"source": "order_test.pdf", "page": 1, "doc_id": "hash_order"})]

    def mock_split(docs):
        call_order.append("split")
        return [Document(text="Chunk", metadata={"source": "order_test.pdf", "page": 1, "doc_id": "hash_order", "chunk_id": "c1"})]

    def mock_add(chunks, collection=None):
        call_order.append("add")
        return len(chunks)

    with (
        patch("app.rag_pipeline.calculate_file_hash", side_effect=mock_hash),
        patch("app.rag_pipeline.is_document_indexed", side_effect=mock_is_indexed),
        patch("app.rag_pipeline.load_pdf", side_effect=mock_load),
        patch("app.rag_pipeline.split_documents", side_effect=mock_split),
        patch("app.rag_pipeline.add_documents", side_effect=mock_add),
    ):
        result = ingest_document(pdf_file)
        assert result.status == "success"
        assert call_order == ["hash", "is_indexed", "load", "split", "add"]


def test_embedding_and_vector_storage_integration(tmp_path: Path):
    """5. Verify that chunk documents produced by splitter are passed directly to vector store."""
    pdf_file = tmp_path / "storage_test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    mock_collection = MagicMock()
    mock_chunks = [
        Document(text="Chunk A", metadata={"source": "storage_test.pdf", "page": 1, "doc_id": "h1", "chunk_id": "h1_p1_c0"}),
        Document(text="Chunk B", metadata={"source": "storage_test.pdf", "page": 1, "doc_id": "h1", "chunk_id": "h1_p1_c1"}),
    ]

    with (
        patch("app.rag_pipeline.calculate_file_hash", return_value="h1"),
        patch("app.rag_pipeline.is_document_indexed", return_value=False),
        patch("app.rag_pipeline.load_pdf", return_value=[Document(text="A B", metadata={"doc_id": "h1", "source": "storage_test.pdf"})]),
        patch("app.rag_pipeline.split_documents", return_value=mock_chunks),
        patch("app.rag_pipeline.add_documents", return_value=2) as mock_add,
    ):
        result = ingest_document(pdf_file, collection=mock_collection)
        assert result.chunks_count == 2
        mock_add.assert_called_once_with(mock_chunks, collection=mock_collection)


def test_duplicate_ingestion_handling(tmp_path: Path):
    """6. Verify that duplicate ingestion is detected early, skipping loading and storage."""
    pdf_file = tmp_path / "duplicate.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy duplicate content")

    with (
        patch("app.rag_pipeline.calculate_file_hash", return_value="dup_hash"),
        patch("app.rag_pipeline.is_document_indexed", return_value=True) as mock_check,
        patch("app.rag_pipeline.load_pdf") as mock_load,
        patch("app.rag_pipeline.split_documents") as mock_split,
        patch("app.rag_pipeline.add_documents") as mock_add,
    ):
        result = ingest_document(pdf_file)

        assert result.status == "already_indexed"
        assert result.chunks_count == 0
        assert "already indexed" in result.message
        assert result.doc_id == "dup_hash"

        mock_check.assert_called_once()
        mock_load.assert_not_called()
        mock_split.assert_not_called()
        mock_add.assert_not_called()


def test_metadata_preservation(tmp_path: Path):
    """7. Verify metadata structure is preserved in IngestionResult."""
    pdf_file = tmp_path / "meta_test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    mock_docs = [
        Document(text="Page 1", metadata={"source": "meta_test.pdf", "page": 1, "total_pages": 5, "doc_id": "meta_hash"})
    ]

    with (
        patch("app.rag_pipeline.calculate_file_hash", return_value="meta_hash"),
        patch("app.rag_pipeline.is_document_indexed", return_value=False),
        patch("app.rag_pipeline.load_pdf", return_value=mock_docs),
        patch("app.rag_pipeline.split_documents", return_value=mock_docs),
        patch("app.rag_pipeline.add_documents", return_value=1),
    ):
        result = ingest_document(pdf_file)
        assert result.metadata["doc_id"] == "meta_hash"
        assert result.metadata["source"] == "meta_test.pdf"
        assert result.metadata["total_pages"] == 5
        assert result.metadata["chunks_count"] == 1


def test_ingestion_failure_handling(tmp_path: Path):
    """8. Verify that exceptions from underlying modules are not silently swallowed."""
    pdf_file = tmp_path / "corrupted.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 corrupt content")

    # Failure in load_pdf
    with (
        patch("app.rag_pipeline.calculate_file_hash", return_value="corrupt_hash"),
        patch("app.rag_pipeline.is_document_indexed", return_value=False),
        patch("app.rag_pipeline.load_pdf", side_effect=ValueError("Corrupted PDF structure")),
    ):
        with pytest.raises(ValueError, match="Corrupted PDF structure"):
            ingest_document(pdf_file)

    # Failure in add_documents
    with (
        patch("app.rag_pipeline.calculate_file_hash", return_value="db_error_hash"),
        patch("app.rag_pipeline.is_document_indexed", return_value=False),
        patch("app.rag_pipeline.load_pdf", return_value=[Document(text="txt", metadata={"source": "corrupted.pdf", "page": 1, "doc_id": "h"})]),
        patch("app.rag_pipeline.split_documents", return_value=[Document(text="txt", metadata={"source": "corrupted.pdf", "page": 1, "doc_id": "h", "chunk_id": "c1"})]),
        patch("app.rag_pipeline.add_documents", side_effect=RuntimeError("ChromaDB disk write failure")),
    ):
        with pytest.raises(RuntimeError, match="ChromaDB disk write failure"):
            ingest_document(pdf_file)


# -----------------------------------------------------------------------------
# Question Answering Tests (9-16)
# -----------------------------------------------------------------------------

def test_successful_question_answering():
    """9. Verify successful question answering flow returning RAGAnswer."""
    mock_chunks = [
        SearchResult(
            chunk_id="chunk_1",
            text="Python is an interpreted high-level language.",
            metadata={"source": "python_guide.pdf", "page": 3},
            distance=0.15,
        )
    ]
    mock_gen_answer = GeneratedAnswer(
        answer="Python is a high-level interpreted programming language.",
        sources=[{"source": "python_guide.pdf", "page": 3, "chunk_id": "chunk_1"}],
        model="gemini-3.8-flash",
    )

    with (
        patch("app.rag_pipeline.retrieve", return_value=mock_chunks) as mock_retrieve,
        patch("app.rag_pipeline.generate_answer", return_value=mock_gen_answer) as mock_generate,
    ):
        result = answer_question("What is Python?", top_k=3)

        assert isinstance(result, RAGAnswer)
        assert result.answer == "Python is a high-level interpreted programming language."
        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "python_guide.pdf"
        assert result.sources[0]["page"] == 3
        assert result.model == "gemini-3.8-flash"
        assert result.retrieved_chunks_count == 1
        assert result.filenames == ["python_guide.pdf"]
        assert result.pages == [3]

        mock_retrieve.assert_called_once_with(query="What is Python?", top_k=3, collection=mock_retrieve.call_args.kwargs.get("collection"))
        mock_generate.assert_called_once()


def test_retrieved_chunks_passed_correctly_to_answer_generator():
    """10. Verify SearchResult chunks from retriever are passed directly to answer generator."""
    mock_chunks = [
        SearchResult(chunk_id="c1", text="Chunk 1", metadata={"source": "a.pdf", "page": 1}, distance=0.1),
        SearchResult(chunk_id="c2", text="Chunk 2", metadata={"source": "b.pdf", "page": 2}, distance=0.2),
    ]

    with (
        patch("app.rag_pipeline.retrieve", return_value=mock_chunks),
        patch("app.rag_pipeline.generate_answer") as mock_generate,
    ):
        mock_generate.return_value = GeneratedAnswer(answer="Answer", sources=[], model="test-model")

        answer_question("Question text", top_k=2)

        # Check that mock_generate received exact mock_chunks
        mock_generate.assert_called_once()
        args, kwargs = mock_generate.call_args
        assert kwargs["chunks"] == mock_chunks
        assert kwargs["question"] == "Question text"


def test_empty_retrieval_bypasses_gemini():
    """11. Verify that empty retrieval skips the Gemini API call completely."""
    with (
        patch("app.rag_pipeline.retrieve", return_value=[]),
        patch("app.rag_pipeline.generate_answer") as mock_generate,
    ):
        result = answer_question("Any question?", top_k=5)

        # Gemini should NEVER be invoked when 0 chunks are found
        mock_generate.assert_not_called()
        assert result.retrieved_chunks_count == 0
        assert "No relevant documents were found" in result.answer
        assert result.sources == []


def test_source_metadata_preservation():
    """12. Verify source metadata properties (filenames, pages) function correctly."""
    mock_chunks = [
        SearchResult(chunk_id="c1", text="T1", metadata={"source": "doc1.pdf", "page": 1}, distance=0.1),
        SearchResult(chunk_id="c2", text="T2", metadata={"source": "doc2.pdf", "page": 4}, distance=0.2),
    ]
    mock_gen = GeneratedAnswer(
        answer="Summary",
        sources=[
            {"source": "doc1.pdf", "page": 1, "chunk_id": "c1"},
            {"source": "doc2.pdf", "page": 4, "chunk_id": "c2"},
        ],
        model="gemini-3.8-flash",
    )

    with (
        patch("app.rag_pipeline.retrieve", return_value=mock_chunks),
        patch("app.rag_pipeline.generate_answer", return_value=mock_gen),
    ):
        result = answer_question("Test query")
        assert result.filenames == ["doc1.pdf", "doc2.pdf"]
        assert result.pages == [1, 4]


def test_retrieval_failure_handling():
    """13. Verify that errors in the retriever module propagate without being swallowed."""
    with patch("app.rag_pipeline.retrieve", side_effect=RuntimeError("Vector index corrupted")):
        with pytest.raises(RuntimeError, match="Vector index corrupted"):
            answer_question("Query?")


def test_answer_generation_failure_handling():
    """14. Verify answer generator exceptions propagate cleanly."""
    mock_chunks = [SearchResult(chunk_id="c1", text="T1", metadata={}, distance=0.1)]
    with (
        patch("app.rag_pipeline.retrieve", return_value=mock_chunks),
        patch("app.rag_pipeline.generate_answer", side_effect=RuntimeError("Gemini service down")),
    ):
        with pytest.raises(RuntimeError, match="Gemini service down"):
            answer_question("Query?")


def test_invalid_question_and_top_k_validation():
    """15. Verify strict input validation for question and top_k parameters."""
    # Invalid question types
    with pytest.raises(TypeError, match="question must be a string"):
        answer_question(1234)  # type: ignore

    # Empty question
    with pytest.raises(ValueError, match="question cannot be empty"):
        answer_question("   ")

    # Boolean top_k
    with pytest.raises(TypeError, match="top_k must be an integer"):
        answer_question("Valid question?", top_k=True)  # type: ignore

    # Non-integer top_k
    with pytest.raises(TypeError, match="top_k must be an integer"):
        answer_question("Valid question?", top_k=3.5)  # type: ignore

    # Non-positive top_k
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        answer_question("Valid question?", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        answer_question("Valid question?", top_k=-5)


def test_get_pipeline_status():
    """16. Verify pipeline status reports total indexed chunks and model."""
    mock_coll = MagicMock()
    mock_coll.name = "test_collection"

    with patch("app.rag_pipeline.get_document_count", return_value=42):
        status = get_pipeline_status(collection=mock_coll)

        assert status["total_chunks_indexed"] == 42
        assert status["collection_name"] == "test_collection"
        assert "gemini_model" in status

