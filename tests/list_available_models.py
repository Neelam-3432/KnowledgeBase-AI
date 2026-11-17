"""Model inspection script for Google Gemini API.

Queries and displays models accessible to the configured API key that
support content generation ('generateContent').
Does NOT perform any text generation, RAG retrieval, or database queries.
"""

import sys
from pathlib import Path

# Add project root to sys.path so app.config can be imported
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from google import genai
from google.genai.errors import APIError

from app.config import GEMINI_API_KEY


def list_models() -> int:
    """List accessible Gemini models supporting content generation."""
    print("=" * 78)
    print("Google Gemini API - Available Content Generation Models")
    print("=" * 78)

    # 1. Validate API Key without printing it
    if (
        not GEMINI_API_KEY
        or not GEMINI_API_KEY.strip()
        or "your_" in GEMINI_API_KEY.lower()
        or "placeholder" in GEMINI_API_KEY.lower()
    ):
        print("\n[FAILED] Configuration Error:")
        print("  GEMINI_API_KEY is not configured or contains a placeholder in .env.")
        print("=" * 78)
        return 1

    # 2. Initialize genai Client
    try:
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
    except Exception as exc:
        print(f"\n[FAILED] Failed to initialize Gemini client: {exc}")
        print("=" * 78)
        return 1

    # 3. Query models.list()
    try:
        print("Retrieving model catalog from Google Gemini API...")
        models = client.models.list()

        generation_models = []
        for model in models:
            actions = getattr(model, "supported_actions", []) or []
            # Check for generateContent support (case-insensitive)
            if any(action.lower() == "generatecontent" for action in actions):
                raw_name = getattr(model, "name", "")
                # Exact model ID without models/ prefix
                short_id = raw_name.removeprefix("models/") if raw_name.startswith("models/") else raw_name
                generation_models.append((short_id, raw_name))

        if not generation_models:
            print("\n[NOTICE] No models supporting 'generateContent' were returned for this API key.")
            print("=" * 78)
            return 0

        print(f"\nFound {len(generation_models)} model(s) supporting 'generateContent':\n")
        print(f"{'Exact Model ID':<35} | {'Resource Name':<40}")
        print("-" * 78)
        for short_id, raw_name in sorted(generation_models, key=lambda x: x[0]):
            print(f"{short_id:<35} | {raw_name:<40}")

        print("=" * 78)
        print("Inspection completed successfully. No content generation requests made.")
        print("=" * 78)
        return 0

    except APIError as api_err:
        code = getattr(api_err, "code", "Unknown")
        msg = getattr(api_err, "message", str(api_err))
        print(f"\n[FAILED] Gemini APIError encountered:")
        print(f"  HTTP Status Code : {code}")
        print(f"  Error Message    : {msg}")
        print("=" * 78)
        return 1

    except Exception as exc:
        print(f"\n[FAILED] Unexpected error while listing models: {exc}")
        print("=" * 78)
        return 1


if __name__ == "__main__":
    sys.exit(list_models())

