"""Isolated diagnostic smoke test for an alternative Gemini model.

Tests whether gemini-3.7-flash can generate content using the configured API key.
Does NOT modify .env, app/config.py, or any application files.
Does NOT import or load any RAG components (no PDFs, Sentence Transformers, or ChromaDB).
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from google import genai
from google.genai.errors import APIError

from app.config import GEMINI_API_KEY

# Local diagnostic model constant (isolated to this test)
ALTERNATIVE_MODEL = "gemini-3.7-flash"


def run_alternative_smoke_test() -> int:
    """Send minimal prompt to alternative Gemini model."""
    print("=" * 65)
    print("Gemini API Alternative Model Diagnostic Test")
    print("=" * 65)
    print(f"Target Model : {ALTERNATIVE_MODEL}")
    print("Prompt       : 'Reply with exactly OK.'")
    print("-" * 65)

    # 1. API key existence check (never expose the key value)
    if (
        not GEMINI_API_KEY
        or not GEMINI_API_KEY.strip()
        or "your_" in GEMINI_API_KEY.lower()
        or "placeholder" in GEMINI_API_KEY.lower()
    ):
        print("\n[FAILED] Configuration Error:")
        print("  GEMINI_API_KEY is not configured or contains a placeholder in .env.")
        print("=" * 65)
        return 1

    # 2. Client initialization
    try:
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
    except Exception as init_err:
        print(f"\n[FAILED] Client Initialization Error: {init_err}")
        print("=" * 65)
        return 1

    # 3. Request generation
    try:
        print(f"Sending request to Gemini model '{ALTERNATIVE_MODEL}'...")
        response = client.models.generate_content(
            model=ALTERNATIVE_MODEL,
            contents="Reply with exactly OK.",
        )

        response_text = getattr(response, "text", None)
        if response_text and response_text.strip():
            print("\n[SUCCESS] Response received from model:")
            print(f"  Model ID : {ALTERNATIVE_MODEL}")
            print(f"  Response : {response_text.strip()}")
            print("=" * 65)
            return 0
        else:
            print(f"\n[WARNING] Model '{ALTERNATIVE_MODEL}' returned an empty response body.")
            print("=" * 65)
            return 1

    except APIError as api_err:
        status_code = getattr(api_err, "code", "Unknown")
        error_message = getattr(api_err, "message", str(api_err))
        print(f"\n[FAILED] Gemini APIError for model '{ALTERNATIVE_MODEL}':")
        print(f"  HTTP Status Code : {status_code}")
        print(f"  Error Message    : {error_message}")
        print("=" * 65)
        return 1

    except Exception as generic_err:
        print(f"\n[FAILED] Unexpected Request / Network Error: {generic_err}")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    exit_code = run_alternative_smoke_test()
    sys.exit(exit_code)

