"""Text splitter module for dividing documents into smaller semantic chunks.

Splits text page-by-page to preserve strict page-level source attribution,
honoring word boundaries and maintaining configurable character overlap.
"""

from typing import List
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.document_loader import Document


def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunk_size and chunk_overlap meet structural constraints.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})."
        )


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split a single text string into overlapping chunks at word boundaries.

    Args:
        text: Input string to split.
        chunk_size: Maximum target characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of non-empty text chunk strings.
    """
    validate_chunk_parameters(chunk_size, chunk_overlap)

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    # If text is already within chunk size, return it as a single chunk
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned_text)

    while start < text_len:
        target_end = start + chunk_size

        if target_end >= text_len:
            final_chunk = cleaned_text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        # Look backwards from target_end for a whitespace boundary
        split_pos = -1
        search_floor = max(start + chunk_overlap, target_end - max(50, chunk_overlap))

        for i in range(target_end, search_floor - 1, -1):
            if cleaned_text[i].isspace():
                split_pos = i
                break

        # Fallback to hard boundary if no whitespace was found in the search window
        if split_pos == -1:
            split_pos = target_end

        chunk = cleaned_text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Compute next starting index with overlap, aligning to whole word starts
        raw_overlap_start = split_pos - chunk_overlap
        if raw_overlap_start <= start:
            next_start = start + 1
        else:
            pos = raw_overlap_start
            while pos < split_pos and not cleaned_text[pos].isspace():
                pos += 1
            while pos < split_pos and cleaned_text[pos].isspace():
                pos += 1
            next_start = pos if pos < split_pos else raw_overlap_start

        # Ensure forward progress
        start = max(start + 1, next_start)

    return chunks


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split a collection of page Documents into smaller chunk Documents.

    Each page is split independently to preserve exact page citations.
    Metadata from the original document is copied and enriched with chunk_id.

    Args:
        documents: List of Document objects from app.document_loader.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap size in characters.

    Returns:
        List of Document objects representing individual chunks.
    """
    validate_chunk_parameters(chunk_size, chunk_overlap)

    chunked_documents: List[Document] = []

    for doc in documents:
        page_chunks = split_text(
            text=doc.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        doc_id = doc.metadata.get("doc_id", "doc")
        page_num = doc.metadata.get("page", 1)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            # Deterministic chunk ID based on content hash, page, and chunk index
            chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"

            chunk_metadata = dict(doc.metadata)
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
            })

            chunked_documents.append(
                Document(text=chunk_text, metadata=chunk_metadata)
            )

    return chunked_documents

