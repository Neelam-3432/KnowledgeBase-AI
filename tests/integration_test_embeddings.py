"""Manual integration test for app.embeddings with the real all-MiniLM-L6-v2 model.

Downloads (on first run) and verifies the real model locally:
- Loads model on CPU.
- Exposes max sequence length (256).
- Generates 384-dimensional normalized vector embeddings.
- Verifies that semantically related texts score higher cosine similarity than unrelated texts.

Run manually:
    .\\.venv\\Scripts\\python.exe tests/integration_test_embeddings.py
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'app' is importable when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from app.embeddings import (
    EMBEDDING_DIMENSION,
    get_embedding_model,
    get_embeddings,
    get_max_seq_length,
    get_query_embedding,
)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_integration_test():
    print("=" * 60)
    print("Testing Real Sentence Transformers Model (all-MiniLM-L6-v2)")
    print("=" * 60)

    # 1. Load model and check properties
    print("\n1. Loading model on CPU...")
    model = get_embedding_model()
    max_seq_len = get_max_seq_length()
    print(f"   Model loaded successfully.")
    print(f"   Max sequence length: {max_seq_len} tokens")

    # 2. Test batch embeddings
    print("\n2. Generating batch embeddings for sample passages...")
    passages = [
        "Artificial intelligence and machine learning enable automated decision making.",
        "Neural networks learn mathematical representations from training data.",
        "The bakery opens at 6 AM and serves fresh sourdough bread and croissants.",
    ]

    doc_embeddings = get_embeddings(passages)
    print(f"   Generated {len(doc_embeddings)} passage embeddings.")
    for idx, vec in enumerate(doc_embeddings):
        norm = np.linalg.norm(vec)
        print(f"   Passage {idx + 1}: dim={len(vec)}, norm={norm:.4f}")
        assert len(vec) == EMBEDDING_DIMENSION, f"Expected {EMBEDDING_DIMENSION}, got {len(vec)}"
        assert abs(norm - 1.0) < 1e-4, f"Vector should be unit-normalized, got {norm}"

    # 3. Test query embedding
    print("\n3. Generating query embedding...")
    query = "How do neural networks and AI models learn from data?"
    query_vec = get_query_embedding(query)
    query_norm = np.linalg.norm(query_vec)
    print(f"   Query embedding: dim={len(query_vec)}, norm={query_norm:.4f}")
    assert len(query_vec) == EMBEDDING_DIMENSION
    assert abs(query_norm - 1.0) < 1e-4

    # 4. Check semantic similarity
    print("\n4. Evaluating semantic similarity (Cosine Distance):")
    sim_1 = cosine_similarity(query_vec, doc_embeddings[0])
    sim_2 = cosine_similarity(query_vec, doc_embeddings[1])
    sim_3 = cosine_similarity(query_vec, doc_embeddings[2])

    print(f"   Similarity to Passage 1 (AI/ML):       {sim_1:.4f}")
    print(f"   Similarity to Passage 2 (Neural nets): {sim_2:.4f}")
    print(f"   Similarity to Passage 3 (Bakery):      {sim_3:.4f}")

    # Semantic sanity check: related passages should score significantly higher than bakery
    assert sim_1 > sim_3, f"Expected AI passage ({sim_1:.3f}) > Bakery ({sim_3:.3f})"
    assert sim_2 > sim_3, f"Expected Neural Net passage ({sim_2:.3f}) > Bakery ({sim_3:.3f})"

    print("\n" + "=" * 60)
    print("ALL REAL MODEL INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_integration_test()
    except Exception as err:
        print(f"\n[FAIL] Integration test failed: {err}", file=sys.stderr)
        sys.exit(1)
