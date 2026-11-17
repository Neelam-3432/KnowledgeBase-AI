"""Minimal isolated smoke test for Google Gemini API connectivity.

Isolates Gemini API availability from the RAG pipeline.
Does NOT load PDFs, text splitters, embeddings (Sentence Transformers),
or ChromaDB.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from google import genai
from google.genai.errors import APIError

from app.config import GEMINI_API_KEY, GEMINI_MODEL


def run_minimal_smoke_test() -> int:
    """Execute minimal prompt test against Gemini API."""
    print("=" * 65)
    print("Gemini API Minimal Isolated Smoke Test")
    print("=" * 65)
    print(f"Configured Model : {GEMINI_MODEL}")
    print("Test Prompt      : 'Reply with exactly OK.'")
    print("-" * 65)

    # 1. API key existence check (without exposing the key value)
    if (
        not GEMINI_API_KEY
        or not GEMINI_API_KEY.strip()
        or "your_" in GEMINI_API_KEY.lower()
        or "placeholder" in GEMINI_API_KEY.lower()
    ):
        print("\n[FAILED] Configuration Error:")
        print("  GEMINI_API_KEY is not configured or contains a placeholder.")
        print("  Please configure a valid API key in your .env file.")
        print("=" * 65)
        return 1

    # 2. Client initialization
    try:
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
    except Exception as init_err:
        print(f"\n[FAILED] Client Initialization Error: {init_err}")
        print("=" * 65)
        return 1

    # 3. API request execution
    try:
        print("Sending request to Gemini API...")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly OK.",
        )

        response_text = getattr(response, "text", None)
        if response_text and response_text.strip():
            print("\n[SUCCESS] Response received from Gemini:")
            print(f"  Response: {response_text.strip()}")
            print("=" * 65)
            return 0
        else:
            print("\n[WARNING] Gemini returned an empty response body.")
            print(f"  Raw response: {response}")
            print("=" * 65)
            return 1

    except APIError as api_err:
        status_code = getattr(api_err, "code", "Unknown")
        error_message = getattr(api_err, "message", str(api_err))
        print("\n[FAILED] Gemini APIError Encountered:")
        print(f"  HTTP Status Code : {status_code}")
        print(f"  Error Message    : {error_message}")
        print("=" * 65)
        return 1

    except Exception as generic_err:
        print(f"\n[FAILED] Unexpected Request / Network Error: {generic_err}")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    exit_code = run_minimal_smoke_test()
    sys.exit(exit_code)

