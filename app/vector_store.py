"""Vector store module using ChromaDB.

Manages persistent local vector storage, document indexing with precomputed
Sentence Transformers embeddings, and metadata preservation with cosine similarity.
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Union

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError, NotFoundError

from app.config import CHROMA_COLLECTION_NAME, CHROMA_DIR
from app.document_loader import Document
from app.embeddings import EMBEDDING_DIMENSION, get_embeddings

logger = logging.getLogger(__name__)

# Module-level client cache
_cached_client: Optional[chromadb.ClientAPI] = None


def get_chroma_client(
    persist_path: Optional[Union[str, Path]] = None,
) -> chromadb.ClientAPI:
    """Initialize or reuse a persistent ChromaDB client.

    Data stored in this client is written to disk and survives application restarts.

    Args:
        persist_path: Optional custom path for the persistent database directory.
                     Defaults to CHROMA_DIR from app.config.

    Returns:
        A chromadb.PersistentClient instance.
    """
    global _cached_client

    target_path = Path(persist_path) if persist_path else CHROMA_DIR
    target_path.mkdir(parents=True, exist_ok=True)

    if persist_path is not None:
        return chromadb.PersistentClient(path=str(target_path))

    if _cached_client is None:
        logger.info(f"Connecting to persistent ChromaDB at '{target_path}'.")
        _cached_client = chromadb.PersistentClient(path=str(target_path))

    return _cached_client


def get_collection(
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Collection:
    """Get an existing ChromaDB collection or create a new one.

    Configures cosine similarity ('hnsw:space': 'cosine') as the vector distance metric.

    Args:
        client: Optional ChromaDB client instance. If None, uses default persistent client.
        collection_name: Name of the collection.

    Returns:
        ChromaDB Collection configured with cosine distance.
    """
    db_client = client or get_chroma_client()

    collection = db_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_documents(
    documents: List[Document],
    collection: Optional[Collection] = None,
    batch_size: int = 64,
) -> int:
    """Embed and index document chunks into ChromaDB in batches.

    Uses upsert semantics: if a chunk with the same chunk_id already exists,
    it is updated in place, preventing duplicate records.

    Args:
        documents: List of chunk Document objects to store.
        collection: Target ChromaDB collection. Defaults to default collection.
        batch_size: Number of chunks to process in each batch. Must be positive int.

    Returns:
        The total number of chunks successfully added or updated.

    Raises:
        TypeError: If documents is not a list, contains non-Document items,
                   metadata is not a dict, or batch_size is a boolean / non-integer.
        ValueError: If batch_size <= 0, chunk text is empty, chunk_id/doc_id is missing,
                    or embeddings are invalid/non-finite.
    """
    # 1. Validate batch_size (reject booleans, non-integers, and values <= 0)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            f"batch_size must be an integer, got {type(batch_size).__name__}."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

    if not isinstance(documents, list):
        raise TypeError(
            f"Expected a list of Document objects, got {type(documents).__name__}."
        )

    if len(documents) == 0:
        logger.info("No documents provided for indexing. Skipping.")
        return 0

    target_collection = collection or get_collection()

    total_added = 0
    total_docs = len(documents)

    for start_idx in range(0, total_docs, batch_size):
        batch = documents[start_idx : start_idx + batch_size]

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[dict] = []

        for item_idx, doc in enumerate(batch):
            global_idx = start_idx + item_idx

            if not isinstance(doc, Document):
                raise TypeError(
                    f"Item at index {global_idx} is {type(doc).__name__}; expected Document."
                )

            chunk_text = doc.text.strip()
            if not chunk_text:
                raise ValueError(f"Document at index {global_idx} has empty text.")

            if not isinstance(doc.metadata, dict):
                raise TypeError(
                    f"Document at index {global_idx} has invalid metadata; expected dict, got {type(doc.metadata).__name__}."
                )

            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Document at index {global_idx} is missing a valid 'chunk_id' string in metadata."
                )

            # Ensure doc_id is explicitly present (required for is_document_indexed and deletion)
            doc_id = doc.metadata.get("doc_id")
            if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(
                    f"Document at index {global_idx} is missing a valid 'doc_id' string in metadata."
                )

            ids.append(chunk_id)
            texts.append(chunk_text)

            # Preserve required metadata and handle optional metadata safely
            source_val = doc.metadata.get("source")
            source_clean = str(source_val) if source_val is not None else "unknown"

            page_val = doc.metadata.get("page")
            if isinstance(page_val, int) and not isinstance(page_val, bool):
                page_clean = page_val
            else:
                page_clean = 1

            clean_metadata = {
                "chunk_id": chunk_id.strip(),
                "doc_id": doc_id.strip(),
                "source": source_clean,
                "page": page_clean,
            }

            # Safely process remaining optional metadata fields
            for k, v in doc.metadata.items():
                if k in clean_metadata:
                    continue
                if v is None:
                    # Safely omit None values without noisy warnings
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    logger.warning(
                        f"Dropping unsupported metadata field '{k}' of type '{type(v).__name__}' "
                        f"for chunk '{chunk_id}'. Supported types are str, int, float, bool."
                    )

            metadatas.append(clean_metadata)

        # Generate embeddings using local Sentence Transformers
        batch_embeddings = get_embeddings(texts)

        # Validate that get_embeddings returned exactly one embedding per input text
        if len(batch_embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: expected {len(texts)} embeddings for batch, "
                f"but received {len(batch_embeddings)}."
            )

        # Validate vector dimensions and reject non-finite values (NaN / Inf)
        for emb_idx, emb in enumerate(batch_embeddings):
            if not isinstance(emb, list):
                raise TypeError(
                    f"Embedding at index {start_idx + emb_idx} must be a list of floats, got {type(emb).__name__}."
                )
            if len(emb) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Generated embedding at index {start_idx + emb_idx} has {len(emb)} dimensions; "
                    f"expected {EMBEDDING_DIMENSION}."
                )

            for val_idx, val in enumerate(emb):
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(
                        f"Embedding at index {start_idx + emb_idx} contains invalid/non-finite value "
                        f"'{val}' at dimension {val_idx}."
                    )

        # Upsert into ChromaDB (updates if ID exists, inserts if new)
        target_collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )

        total_added += len(batch)
        logger.debug(f"Indexed batch of {len(batch)} chunks ({total_added}/{total_docs}).")

    return total_added


def get_document_count(collection: Optional[Collection] = None) -> int:
    """Return the total number of stored chunks in the collection."""
    target_collection = collection or get_collection()
    return target_collection.count()


def is_document_indexed(
    doc_id: str,
    collection: Optional[Collection] = None,
) -> bool:
    """Check if any chunks from a document hash (doc_id) exist in the collection."""
    if not doc_id:
        return False

    target_collection = collection or get_collection()
    existing = target_collection.get(
        where={"doc_id": doc_id},
        limit=1,
    )
    return len(existing.get("ids", [])) > 0


def delete_document_by_doc_id(
    doc_id: str,
    collection: Optional[Collection] = None,
) -> int:
    """Delete all chunks belonging to a specific document ID.

    Args:
        doc_id: Content hash or ID of the document to delete.
        collection: Target ChromaDB collection.

    Returns:
        Number of chunks deleted.
    """
    if not doc_id:
        return 0

    target_collection = collection or get_collection()
    matching = target_collection.get(where={"doc_id": doc_id})
    matching_ids = matching.get("ids", [])

    if matching_ids:
        target_collection.delete(ids=matching_ids)
        logger.info(f"Deleted {len(matching_ids)} chunks for doc_id '{doc_id}'.")
        return len(matching_ids)

    return 0


def reset_collection(
    collection_name: str = CHROMA_COLLECTION_NAME,
    client: Optional[chromadb.ClientAPI] = None,
) -> Collection:
    """Delete and recreate the collection.

    Handles missing collections gracefully (NotFoundError) while raising genuine database errors.

    Args:
        collection_name: Name of the collection to reset.
        client: Optional ChromaDB client instance.

    Returns:
        A freshly recreated empty Collection.

    Raises:
        ChromaError: If an unexpected ChromaDB database error occurs.
    """
    db_client = client or get_chroma_client()
    try:
        db_client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection '{collection_name}'.")
    except NotFoundError:
        logger.debug(f"Collection '{collection_name}' did not exist; skipping deletion.")
    except ChromaError as e:
        logger.error(f"ChromaDB error while resetting collection '{collection_name}': {e}")
        raise

    return get_collection(client=db_client, collection_name=collection_name)
