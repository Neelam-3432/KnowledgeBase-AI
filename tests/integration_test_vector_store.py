"""Manual local integration test for ChromaDB vector store and retriever.

Uses an actual temporary persistent ChromaDB on disk and the real
all-MiniLM-L6-v2 Sentence Transformers embedding model:
- Initializes persistent storage on disk.
- Indexes real document chunks with 384-d normalized embeddings.
- Verifies persistence across client re-connections.
- Performs semantic retrieval and validates cosine distance ordering.

Run manually:
    .\\.venv\\Scripts\\python.exe tests/integration_test_vector_store.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path so 'app' is importable when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.document_loader import Document
from app.retriever import retrieve
from app.vector_store import (
    add_documents,
    get_chroma_client,
    get_collection,
    get_document_count,
    is_document_indexed,
)


def run_integration_test():
    print("=" * 65)
    print("Testing Real ChromaDB Persistent Storage & Semantic Retrieval")
    print("=" * 65)

    temp_dir = tempfile.mkdtemp(prefix="chroma_integration_test_")
    print(f"\n1. Initialized temporary persistent DB path:\n   {temp_dir}")

    try:
        # Step 1: Create persistent client and collection
        client = get_chroma_client(persist_path=temp_dir)
        collection = get_collection(client=client, collection_name="integration_docs")
        print(f"   Created collection '{collection.name}' with space: {collection.metadata.get('hnsw:space')}")

        # Step 2: Define sample chunks
        print("\n2. Staging document chunks for indexing...")
        chunks = [
            Document(
                text=(
                    "Agentic AI systems use reasoning loops, tool calling, and planning "
                    "to autonomously execute complex multi-step workflows."
                ),
                metadata={
                    "source": "agentic_ai_whitepaper.pdf",
                    "page": 1,
                    "total_pages": 4,
                    "doc_id": "sha256_agentic_001",
                    "chunk_id": "sha256_agentic_001_p1_c0",
                    "chunk_index": 0,
                },
            ),
            Document(
                text=(
                    "Retrieval-Augmented Generation (RAG) retrieves factual context from "
                    "a vector database and injects it into the prompt of a large language model."
                ),
                metadata={
                    "source": "agentic_ai_whitepaper.pdf",
                    "page": 2,
                    "total_pages": 4,
                    "doc_id": "sha256_agentic_001",
                    "chunk_id": "sha256_agentic_001_p2_c0",
                    "chunk_index": 0,
                },
            ),
            Document(
                text=(
                    "To bake artisan French bread, knead high-protein flour with water, "
                    "yeast, and salt, then allow the dough to ferment for 18 hours."
                ),
                metadata={
                    "source": "baking_cookbook.pdf",
                    "page": 14,
                    "total_pages": 50,
                    "doc_id": "sha256_baking_002",
                    "chunk_id": "sha256_baking_002_p14_c0",
                    "chunk_index": 0,
                },
            ),
        ]

        # Step 3: Embed and index chunks
        print("   Generating real 384-d embeddings and writing to disk...")
        indexed_count = add_documents(chunks, collection=collection)
        print(f"   Successfully indexed {indexed_count} chunks.")
        assert get_document_count(collection) == 3
        assert is_document_indexed("sha256_agentic_001", collection=collection) is True

        # Step 4: Verify persistence across fresh client re-connection
        print("\n3. Testing disk persistence across client re-initialization...")
        fresh_client = get_chroma_client(persist_path=temp_dir)
        fresh_collection = get_collection(client=fresh_client, collection_name="integration_docs")
        reloaded_count = get_document_count(fresh_collection)
        print(f"   Reloaded chunk count from disk: {reloaded_count}")
        assert reloaded_count == 3, f"Expected 3 chunks on reload, got {reloaded_count}"

        # Step 5: Test duplicate prevention (upserting same chunks should not duplicate)
        print("\n4. Testing duplicate upload prevention...")
        add_documents(chunks[:1], collection=fresh_collection)
        count_after_duplicate = get_document_count(fresh_collection)
        print(f"   Chunk count after re-indexing chunk 0: {count_after_duplicate}")
        assert count_after_duplicate == 3, "Duplicate chunk created a duplicate entry!"

        # Step 6: Query and evaluate semantic retrieval
        print("\n5. Executing semantic similarity search...")
        query = "How do autonomous AI agents plan actions and use tools?"
        print(f"   Query: \"{query}\"")
        results = retrieve(query, top_k=2, collection=fresh_collection)

        print(f"   Retrieved {len(results)} results:")
        for idx, res in enumerate(results, 1):
            print(f"\n   [Result {idx}] Cosine Distance: {res.distance:.4f}")
            print(f"   Chunk ID:  {res.chunk_id}")
            print(f"   Source:    {res.metadata.get('source')} (Page {res.metadata.get('page')})")
            print(f"   Text:      {res.text}")

        # Assertions
        assert len(results) == 2
        # Result 1 must be the agentic AI chunk
        assert results[0].chunk_id == "sha256_agentic_001_p1_c0"
        # Cosine distance for related topic should be significantly lower than unrelated
        assert results[0].distance < 0.8

        print("\n" + "=" * 65)
        print("ALL CHROMADB PERSISTENCE & RETRIEVAL CHECKS PASSED!")
        print("=" * 65)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nCleaned up temporary integration test directory.")


if __name__ == "__main__":
    try:
        run_integration_test()
    except Exception as err:
        print(f"\n[FAIL] Integration test failed: {err}", file=sys.stderr)
        sys.exit(1)

