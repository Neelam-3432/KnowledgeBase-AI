"""Integration test simulating the Streamlit frontend workflow.

Tests the exact calls made by app/main.py:
1. Empty state querying and status check.
2. File saving to UPLOADS_DIR with collision avoidance.
3. Ingestion of multi-page PDF into ChromaDB.
4. Duplicate document upload prevention.
5. Live Groq Question Answering with real citations.
6. Knowledge base reset with verification.
"""

import sys
import uuid
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.test_document_loader import create_sample_pdf
from app.config import GROQ_API_KEY, GROQ_MODEL, UPLOADS_DIR
from app.rag_pipeline import answer_question, get_pipeline_status, ingest_document
from app.vector_store import reset_collection


def main():
    print("=" * 70)
    print("STREAMLIT FRONTEND WORKFLOW INTEGRATION TEST")
    print("=" * 70)

    # 1. Reset collection for clean starting state
    print("\n1. Resetting collection for baseline test...")
    reset_collection()
    initial_status = get_pipeline_status()
    print(f"   Initial chunks: {initial_status['total_chunks_indexed']}")
    assert initial_status["total_chunks_indexed"] == 0

    # 2. Test empty knowledge base query
    print("\n2. Querying empty knowledge base...")
    empty_ans = answer_question("What is AI?", top_k=3)
    print(f"   Response: '{empty_ans.answer}'")
    print(f"   Retrieved chunks: {empty_ans.retrieved_chunks_count}")
    print(f"   Sources: {empty_ans.sources}")
    assert empty_ans.retrieved_chunks_count == 0
    assert "No relevant documents were found" in empty_ans.answer
    assert empty_ans.sources == []

    # 3. Simulate PDF upload and file persistence in UPLOADS_DIR
    print("\n3. Simulating PDF upload into UPLOADS_DIR...")
    test_pdf_name = f"frontend_test_{uuid.uuid4().hex[:6]}.pdf"
    target_path = UPLOADS_DIR / test_pdf_name

    pdf_bytes = create_sample_pdf([
        "Page 1: Antigravity is an agentic AI assistant developed by Google DeepMind.",
        "Page 2: KnowledgeBase AI utilizes Groq LPU inference and ChromaDB persistent storage."
    ])
    target_path.write_bytes(pdf_bytes)
    print(f"   Saved PDF to: {target_path}")
    assert target_path.exists()

    # 4. Ingest document (as done by button in sidebar)
    print("\n4. Ingesting PDF via ingest_document()...")
    ingest_result = ingest_document(target_path)
    print(f"   Status: {ingest_result.status}")
    print(f"   Message: {ingest_result.message}")
    print(f"   Chunks indexed: {ingest_result.chunks_count}")
    assert ingest_result.status == "success"
    assert ingest_result.chunks_count == 2

    # 5. Check duplicate upload prevention
    print("\n5. Testing duplicate upload prevention...")
    dup_result = ingest_document(target_path)
    print(f"   Duplicate Status: {dup_result.status}")
    print(f"   Duplicate Message: {dup_result.message}")
    assert dup_result.status == "already_indexed"
    assert dup_result.chunks_count == 0

    # 6. Check updated pipeline status metrics
    updated_status = get_pipeline_status()
    print(f"\n6. Updated status metrics: {updated_status}")
    assert updated_status["total_chunks_indexed"] == 2

    # 7. Ask grounded question and verify citations
    query = "What storage and inference hardware does KnowledgeBase AI use?"
    print(f"\n7. Asking grounded question: '{query}'")
    rag_ans = answer_question(query, top_k=2)
    print(f"   Model: {rag_ans.model}")
    print(f"   Answer:\n{rag_ans.answer}")
    print(f"   Sources: {rag_ans.sources}")
    print(f"   Filenames: {rag_ans.filenames}")
    print(f"   Pages: {rag_ans.pages}")

    assert rag_ans.retrieved_chunks_count == 2
    assert len(rag_ans.sources) > 0
    assert target_path.name in rag_ans.filenames
    assert 2 in rag_ans.pages

    # 8. Reset database action (as done by danger zone in sidebar)
    print("\n8. Testing Reset Database action...")
    reset_collection()
    final_status = get_pipeline_status()
    print(f"   Final chunks count after reset: {final_status['total_chunks_indexed']}")
    assert final_status["total_chunks_indexed"] == 0

    # Clean up uploaded test PDF
    if target_path.exists():
        target_path.unlink()

    print("\n" + "=" * 70)
    print("ALL FRONTEND WORKFLOW INTEGRATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

