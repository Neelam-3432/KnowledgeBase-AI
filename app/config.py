"""Application configuration module.

Loads environment variables and centralizes paths, model names, and parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Ensure data directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Gemini LLM configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Embedding model configuration (local Sentence Transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Text chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Vector database settings
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

# Vector search retrieval settings (default K = 5)
TOP_K = int(os.getenv("TOP_K", "5"))
