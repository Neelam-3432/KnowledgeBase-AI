"""Live Integration Smoke Test for Groq LLM integration.

Verifies that:
1. GROQ_API_KEY and GROQ_MODEL are loaded from config / .env.
2. The Groq SDK client initializes successfully.
3. A real RAG answer generation request using generate_answer succeeds.
4. Non-empty grounded answer and source citations are returned.
5. If the API key is not configured, clearly skips the live test without failing.
6. The API key is NEVER exposed or printed.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.answer_generator import generate_answer, get_groq_client
from app.config import GROQ_API_KEY, GROQ_MODEL
from app.retriever import SearchResult


def main() -> int:
    print("=" * 65)
    print("Testing Real Groq API Integration (answer_generator.py)")
    print("=" * 65)
    print(f"Configured Model: {GROQ_MODEL}")

    # Check if key is configured (without printing the key)
    if (
        not GROQ_API_KEY
        or not GROQ_API_KEY.strip()
        or "your_" in GROQ_API_KEY.lower()
        or "placeholder" in GROQ_API_KEY.lower()
    ):
        print("\n[NOTICE] GROQ_API_KEY is empty or a placeholder in .env.")
        print("To run the live Groq test, set a valid GROQ_API_KEY in .env.")
        print("Skipping live API call. Unit tests verified all mock integration paths.")
        print("=" * 65)
        print("GROQ INTEGRATION STATUS: SKIPPED (NO CREDENTIALS)")
        print("=" * 65)
        return 0

    print("GROQ_API_KEY detected in configuration (masked).")

    # Construct sample retrieved passage for grounding test
    sample_chunk = SearchResult(
        chunk_id="groq_doc_p1_c0",
        text=(
            "KnowledgeBase AI uses Groq LPUs (Language Processing Units) for "
            "ultra-fast inference and high-throughput grounded question answering. "
            "Sentence Transformers generates 384-dimensional local embeddings."
        ),
        metadata={"source": "architecture_overview.pdf", "page": 1},
        distance=0.08,
    )

    test_question = "What hardware does KnowledgeBase AI use for fast inference?"
    print(f"\nSending grounded prompt to Groq model '{GROQ_MODEL}'...")
    print(f"Question: '{test_question}'")

    try:
        client = get_groq_client()
        result = generate_answer(
            question=test_question,
            chunks=[sample_chunk],
            client=client,
            model=GROQ_MODEL,
        )

        print("\n[RESPONSE RECEIVED FROM GROQ]:")
        print("-" * 65)
        print(result.answer)
        print("-" * 65)
        print(f"Model Used : {result.model}")
        print(f"Sources    : {result.sources}")

        # Verification
        assert result.answer, "Answer should not be empty."
        assert not result.answer.startswith("Groq API Error"), f"Received API error: {result.answer}"
        assert not result.answer.startswith("Configuration Error"), f"Received config error: {result.answer}"
        assert len(result.sources) == 1, "Should retain 1 source."
        assert result.sources[0]["source"] == "architecture_overview.pdf"
        assert result.sources[0]["page"] == 1

        print("\n" + "=" * 65)
        print("GROQ INTEGRATION TEST: SUCCESS (VERIFIED LIVE)")
        print("=" * 65)
        return 0

    except Exception as e:
        print(f"\n[FAILED] Live Groq integration call failed: {e}")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    sys.exit(main())

