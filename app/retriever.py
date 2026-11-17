"""Retriever module for semantic similarity search over stored document chunks.

Embeds queries using the local Sentence Transformers model and searches
the persistent ChromaDB collection for the Top-K nearest neighbors by cosine distance.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from chromadb.api.models.Collection import Collection

from app.config import TOP_K
from app.embeddings import get_query_embedding
from app.vector_store import get_collection

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a retrieved document chunk with similarity scoring."""
    chunk_id: str
    text: str
    metadata: dict
    distance: float


def retrieve(
    query: str,
    top_k: int = TOP_K,
    collection: Optional[Collection] = None,
) -> List[SearchResult]:
    """Retrieve the most relevant document chunks for a natural-language query.

    Args:
        query: User search query string.
        top_k: Number of most similar chunks to return. Defaults to TOP_K from config.
               Must be a positive integer (booleans and non-integers rejected).
        collection: Optional target ChromaDB collection. Defaults to default collection.

    Returns:
        List of SearchResult objects ordered by relevance (lowest cosine distance first).

    Raises:
        TypeError: If query is not a string, or top_k is a boolean or not an int.
        ValueError: If query is empty or whitespace-only, top_k <= 0,
                    or ChromaDB returns inconsistent response lengths.
    """
    if not isinstance(query, str):
        raise TypeError(f"Query must be a string, got {type(query).__name__}.")

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query string cannot be empty or whitespace-only.")

    # Validate top_k: reject booleans, non-integers, and non-positive integers
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    target_collection = collection or get_collection()

    total_chunks = target_collection.count()
    if total_chunks == 0:
        logger.info(
            f"ChromaDB collection is empty. Returning 0 results for query: '{cleaned_query[:40]}'."
        )
        return []

    # ChromaDB requires n_results to be <= number of items in collection
    n_results = min(top_k, total_chunks)

    # Generate normalized query embedding with identical model and preprocessing
    query_vector = get_query_embedding(cleaned_query)

    # Perform nearest-neighbor vector search in ChromaDB
    raw_results = target_collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_items: List[SearchResult] = []

    # Parse query response (results are nested inside a list for each query)
    ids_list = raw_results.get("ids", [[]])[0]
    docs_list = raw_results.get("documents", [[]])[0]
    meta_list = raw_results.get("metadatas", [[]])[0]
    dist_list = raw_results.get("distances", [[]])[0]

    # Validate that all response arrays are strictly consistent in length
    if not (len(ids_list) == len(docs_list) == len(meta_list) == len(dist_list)):
        raise ValueError(
            f"Inconsistent ChromaDB response lengths: ids={len(ids_list)}, "
            f"documents={len(docs_list)}, metadatas={len(meta_list)}, distances={len(dist_list)}."
        )

    for chunk_id, text, metadata, distance in zip(
        ids_list, docs_list, meta_list, dist_list, strict=True
    ):
        retrieved_items.append(
            SearchResult(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata or {},
                distance=float(distance),
            )
        )

    # Sort strictly by distance ascending (lowest distance = highest similarity)
    retrieved_items.sort(key=lambda item: item.distance)

    # Clean, accurate logging for non-empty vs empty result sets
    if not retrieved_items:
        logger.info(
            f"No matching document chunks found for query: '{cleaned_query[:40]}'."
        )
    else:
        logger.debug(
            f"Retrieved {len(retrieved_items)} chunks for query: '{cleaned_query[:40]}' "
            f"(best distance: {retrieved_items[0].distance:.4f})"
        )

    return retrieved_items
