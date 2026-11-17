"""End-to-End Integration Smoke Test for app.rag_pipeline.

Performs a live test of:
1. Ingesting a real temporary PDF document into ChromaDB.
2. Verifying duplicate upload prevention.
3. Querying the knowledge base via answer_question.
4. Cleaning up test data afterwards.
"""

import sys
import uuid
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pypdf import PdfWriter

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.rag_pipeline import answer_question, get_pipeline_status, ingest_document
from app.vector_store import delete_document_by_doc_id, get_collection


def create_sample_pdf(file_path: Path) -> None:
    """Create a sample 2-page PDF file with factual text."""
    writer = PdfWriter()
    page1 = writer.add_blank_page(width=300, height=400)
    # Use pypdf to create pages or minimal structure
    from tests.test_document_loader import create_sample_pdf as generate_pdf_bytes

    content = generate_pdf_bytes([
        "KnowledgeBase AI is an end-to-end RAG application built with Streamlit and ChromaDB.",
        "It uses Sentence Transformers all-MiniLM-L6-v2 for generating 384-dimensional dense embeddings."
    ])
    file_path.write_bytes(content)


def main():
    print("=" * 65)
    print("End-to-End RAG Pipeline Smoke Test (rag_pipeline.py)")
    print("=" * 65)

    test_dir = Path("data") / "test_scratch"
    test_dir.mkdir(parents=True, exist_ok=True)
    sample_pdf = test_dir / f"smoke_test_{uuid.uuid4().hex[:6]}.pdf"

    try:
        # Step 1: Create sample PDF
        print(f"\n1. Creating test PDF: {sample_pdf.name}")
        create_sample_pdf(sample_pdf)

        # Step 2: Ingest PDF
        print("2. Ingesting PDF into knowledge base...")
        ingest_res = ingest_document(sample_pdf)
        print(f"   Status: {ingest_res.status}")
        print(f"   Chunks indexed: {ingest_res.chunks_count}")
        print(f"   Document ID: {ingest_res.doc_id[:16]}...")
        assert ingest_res.status == "success"
        assert ingest_res.chunks_count > 0

        # Step 3: Duplicate ingestion test
        print("3. Testing duplicate upload prevention...")
        dup_res = ingest_document(sample_pdf)
        print(f"   Status: {dup_res.status}")
        print(f"   Message: {dup_res.message}")
        assert dup_res.status == "already_indexed"
        assert dup_res.chunks_count == 0

        # Step 4: Pipeline Status
        status = get_pipeline_status()
        print(f"\n4. Pipeline Status: {status}")

        # Step 5: Question Answering
        query = "What embedding model does KnowledgeBase AI use?"
        print(f"\n5. Asking Question: '{query}'")
        rag_answer = answer_question(query, top_k=2)
        print(f"   Retrieved Chunks Count: {rag_answer.retrieved_chunks_count}")
        print(f"   Model: {rag_answer.model}")
        print(f"   Sources: {rag_answer.sources}")
        print(f"   Filenames: {rag_answer.filenames}")
        print(f"   Answer:\n{rag_answer.answer}\n")

        print("=" * 65)
        print("END-TO-END RAG PIPELINE VERIFICATION: SUCCESS")
        print("=" * 65)

    finally:
        # Clean up test artifacts
        if sample_pdf.exists():
            try:
                # Remove indexed chunks
                delete_document_by_doc_id(ingest_res.doc_id)
            except Exception:
                pass
            sample_pdf.unlink(missing_ok=True)
        if test_dir.exists():
            try:
                test_dir.rmdir()
            except Exception:
                pass


if __name__ == "__main__":
    main()
