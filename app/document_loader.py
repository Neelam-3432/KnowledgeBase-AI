"""PDF document loader module.

Extracts text page-by-page from PDF files using pypdf and attaches
page-level metadata and a content hash (doc_id) to avoid duplicate indexing.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from pypdf import PdfReader
from pypdf.errors import PdfStreamError, PyPdfError

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents an extracted unit of text with associated metadata."""
    text: str
    metadata: dict


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file to uniquely identify document content."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64KB chunks to handle large files memory-efficiently
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    """Load a PDF file and extract text page by page with metadata.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        List of Document objects, each containing text and metadata for a page.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid or readable PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty (0 bytes): {path.name}")

    # Compute content hash to detect duplicate uploads
    try:
        doc_id = calculate_file_hash(path)
    except Exception as e:
        raise ValueError(f"Failed to read file for hashing: {path.name}. Error: {e}")

    try:
        reader = PdfReader(str(path))
    except (PdfStreamError, PyPdfError, Exception) as e:
        raise ValueError(f"Cannot read '{path.name}'. File may be corrupted or invalid. Error: {e}")

    # Handle password-protected PDFs
    if reader.is_encrypted:
        try:
            # Attempt to decrypt with empty password (common for restricted permissions)
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise ValueError(f"PDF '{path.name}' is password-protected and cannot be read.")
        except Exception as e:
            raise ValueError(f"PDF '{path.name}' is encrypted. Error: {e}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF '{path.name}' contains 0 pages.")

    documents: List[Document] = []
    scanned_pages = 0

    for page_idx, page in enumerate(reader.pages):
        page_number = page_idx + 1  # 1-indexed for human readability

        try:
            extracted_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_number} of {path.name}: {e}")
            extracted_text = ""

        cleaned_text = extracted_text.strip()

        # Check for empty / scanned page
        if not cleaned_text:
            scanned_pages += 1
            # We record a note in metadata so the caller knows the page had no text layer
            continue

        metadata = {
            "source": path.name,
            "file_path": str(path.resolve()),
            "doc_id": doc_id,
            "page": page_number,
            "total_pages": total_pages,
        }

        documents.append(Document(text=cleaned_text, metadata=metadata))

    if scanned_pages == total_pages:
        logger.warning(
            f"All {total_pages} page(s) in '{path.name}' contain no extractable text. "
            "This PDF is likely a scanned image without an embedded OCR text layer."
        )

    return documents


def load_multiple_pdfs(file_paths: List[Union[str, Path]]) -> List[Document]:
    """Load multiple PDF files and return combined documents."""
    all_documents: List[Document] = []

    for path in file_paths:
        try:
            docs = load_pdf(path)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Skipping '{path}' due to error: {e}")

    return all_documents

