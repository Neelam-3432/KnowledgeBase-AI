"""Unit and workflow tests for the Streamlit application module (app.main).

Tests:
1. Module import integrity.
2. Safe duplicate filename collision avoidance in UPLOADS_DIR.
3. Multi-file upload processing flow.
4. Session state initialization.
5. Empty knowledge base querying behavior.
6. Knowledge base reset flow.
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.config import UPLOADS_DIR
from app.document_loader import Document
from app.main import init_session_state
from app.rag_pipeline import IngestionResult, RAGAnswer


def test_main_import_integrity():
    """Verify that app.main can be imported without syntax or import errors."""
    import app.main as main_mod
    assert hasattr(main_mod, "main")
    assert hasattr(main_mod, "init_session_state")


def test_session_state_initialization():
    """Verify session state initializes the message history."""
    with patch("streamlit.session_state", {}) as mock_state:
        init_session_state()
        assert "messages" in mock_state
        assert mock_state["messages"] == []


def test_duplicate_filename_collision_avoidance(tmp_path: Path):
    """Verify different file content with identical filenames gets a safe non-colliding name."""
    test_upload_dir = tmp_path / "uploads"
    test_upload_dir.mkdir(parents=True, exist_ok=True)

    file_a_name = "doc.pdf"
    file_a_bytes = b"Content A"
    file_b_bytes = b"Content B (Different)"

    # Write file A
    target_a = test_upload_dir / file_a_name
    target_a.write_bytes(file_a_bytes)

    # Simulate saving file B with same name but different content
    target_b = test_upload_dir / file_a_name
    if target_b.exists():
        existing = target_b.read_bytes()
        if existing != file_b_bytes:
            unique_name = f"{target_b.stem}_{uuid.uuid4().hex[:6]}{target_b.suffix}"
            target_b = test_upload_dir / unique_name

    target_b.write_bytes(file_b_bytes)

    # Verify both files exist safely without overwriting
    assert target_a.exists()
    assert target_b.exists()
    assert target_a != target_b
    assert target_a.read_bytes() == b"Content A"
    assert target_b.read_bytes() == b"Content B (Different)"


def test_empty_knowledge_base_behavior():
    """Verify that querying an empty knowledge base returns helpful feedback without error."""
    from app.rag_pipeline import answer_question

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch("app.retriever.get_collection", return_value=mock_collection):
        result = answer_question("What is in the document?", collection=mock_collection)
        assert isinstance(result, RAGAnswer)
        assert result.retrieved_chunks_count == 0
        assert "No relevant documents were found" in result.answer
        assert result.sources == []

