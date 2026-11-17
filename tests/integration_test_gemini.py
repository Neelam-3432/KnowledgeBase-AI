"""Manual integration / smoke test for app.answer_generator with real Gemini API.

If GEMINI_API_KEY is configured with a valid key, this executes an actual call
to Google Gemini with sample retrieved chunks and prints the grounded answer.
If the API key is missing or a placeholder, it prints an informative notice and skips.

Run manually:
    .\\.venv\\Scripts\\python.exe tests/integration_test_gemini.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path so 'app' is importable when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.answer_generator import generate_answer, get_gemini_client
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.retriever import SearchResult


def run_gemini_smoke_test():
    print("=" * 65)
    print("Testing Real Gemini API Integration (answer_generator.py)")
    print("=" * 65)
    print(f"Configured Model: {GEMINI_MODEL}")

    # Check for placeholder or empty API key
    if (
        not GEMINI_API_KEY
        or not GEMINI_API_KEY.strip()
        or "your_" in GEMINI_API_KEY.lower()
        or "placeholder" in GEMINI_API_KEY.lower()
    ):
        print("\n[NOTICE] GEMINI_API_KEY is empty or a placeholder in .env.")
        print("To run the live Gemini smoke test, set your valid key in .env.")
        print("Skipping live API call. Unit tests verified all mock integration paths.")
        print("\n" + "=" * 65)
        print("GEMINI INTEGRATION MODULE STATUS: READY FOR CONFIGURATION")
        print("=" * 65)
        return

    print("\n1. Testing Gemini client authentication...")
    try:
        client = get_gemini_client()
        print("   Client initialized successfully.")
    except Exception as e:
        print(f"   [FAIL] Client initialization failed: {e}")
        return

    # 2. Stage sample retrieved chunks
    print("\n2. Staging sample retrieved document chunks...")
    sample_chunks = [
        SearchResult(
            chunk_id="whitepaper_p1_c0",
            text=(
                "Agentic AI systems autonomously pursue goals by breaking complex tasks "
                "into step-by-step plans, invoking external tools, and evaluating intermediate results."
            ),
            metadata={"source": "agentic_ai_overview.pdf", "page": 1, "chunk_id": "whitepaper_p1_c0"},
            distance=0.18,
        ),
        SearchResult(
            chunk_id="whitepaper_p2_c0",
            text=(
                "Retrieval-Augmented Generation (RAG) grounds agent responses by supplying "
                "factual context from local vector stores such as ChromaDB."
            ),
            metadata={"source": "agentic_ai_overview.pdf", "page": 2, "chunk_id": "whitepaper_p2_c0"},
            distance=0.25,
        ),
    ]
    print(f"   Staged {len(sample_chunks)} chunks from 'agentic_ai_overview.pdf'.")

    # 3. Send grounded question to Gemini
    question = "How do Agentic AI systems accomplish goals according to the documents?"
    print(f"\n3. Sending grounded prompt to Gemini ({GEMINI_MODEL})...")
    print(f"   Question: \"{question}\"")

    start_time = time.time()
    result = generate_answer(
        question=question,
        chunks=sample_chunks,
        client=client,
        model=GEMINI_MODEL,
    )
    elapsed = time.time() - start_time

    print(f"   Completed in {elapsed:.2f} seconds.")
    print("\n" + "-" * 65)
    print("GEMINI GENERATED ANSWER:")
    print("-" * 65)
    print(result.answer)
    print("-" * 65)
    print("\nPreserved Source Citations:")
    for s in result.sources:
        print(f"   - Source: {s['source']} (Page {s['page']}) [Chunk ID: {s['chunk_id']}]")

    print("\n" + "=" * 65)
    print("REAL GEMINI API SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_gemini_smoke_test()
    except Exception as err:
        print(f"\n[FAIL] Smoke test encountered an error: {err}", file=sys.stderr)
        sys.exit(1)

