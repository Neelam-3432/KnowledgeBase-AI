"""Unit tests for app.document_loader.

Tests cover:
- Valid PDF text extraction and metadata preservation (source, page, total_pages, doc_id).
- Content hash consistency (for duplicate detection).
- Blank/scanned PDF handling.
- Error handling for missing, zero-byte, and corrupted files.
- Batch loading of multiple PDFs.
"""

from pathlib import Path
import pytest
from pypdf import PdfWriter

from app.document_loader import calculate_file_hash, load_multiple_pdfs, load_pdf


def create_sample_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate valid multi-page PDF bytes with extractable text."""
    objects = []
    # Obj 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # Obj 2: Pages (placeholder)
    objects.append(b"")
    # Obj 3: Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_indices = []
    curr_idx = 4
    for text in pages_text:
        content_stream = f"BT\n/F1 12 Tf\n50 500 Td\n({text}) Tj\nET\n".encode("latin-1")
        stream_obj = (
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
            + content_stream
            + b"endstream"
        )
        objects.append(stream_obj)
        content_obj_idx = curr_idx
        curr_idx += 1

        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Contents {content_obj_idx} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode(
                "latin-1"
            )
        )
        objects.append(page_obj)
        page_obj_indices.append(curr_idx)
        curr_idx += 1

    kids_str = " ".join(f"{idx} 0 R" for idx in page_obj_indices)
    pages_obj = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(pages_text)} >>".encode("latin-1")
    objects[1] = pages_obj

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("latin-1"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return bytes(out)


def test_load_valid_pdf(tmp_path: Path):
    """Verify that text and page-level metadata are correctly extracted."""
    pdf_path = tmp_path / "sample_doc.pdf"
    content = create_sample_pdf(
        [
            "Artificial Intelligence overview.",
            "Retrieval Augmented Generation details.",
        ]
    )
    pdf_path.write_bytes(content)

    docs = load_pdf(pdf_path)

    # Should return 2 documents (one per page)
    assert len(docs) == 2

    # Check page 1
    assert "Artificial Intelligence overview" in docs[0].text
    assert docs[0].metadata["source"] == "sample_doc.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 2
    assert len(docs[0].metadata["doc_id"]) == 64  # SHA-256 hex string

    # Check page 2
    assert "Retrieval Augmented Generation details" in docs[1].text
    assert docs[1].metadata["source"] == "sample_doc.pdf"
    assert docs[1].metadata["page"] == 2
    assert docs[1].metadata["total_pages"] == 2
    assert docs[1].metadata["doc_id"] == docs[0].metadata["doc_id"]


def test_content_hash_consistency(tmp_path: Path):
    """Verify that identical files produce the same doc_id (content hash)."""
    file_a = tmp_path / "original.pdf"
    file_b = tmp_path / "duplicate_copy.pdf"
    file_c = tmp_path / "different.pdf"

    bytes_ab = create_sample_pdf(["Identical content"])
    bytes_c = create_sample_pdf(["Completely different text"])

    file_a.write_bytes(bytes_ab)
    file_b.write_bytes(bytes_ab)
    file_c.write_bytes(bytes_c)

    hash_a = calculate_file_hash(file_a)
    hash_b = calculate_file_hash(file_b)
    hash_c = calculate_file_hash(file_c)

    # Same content -> same hash (prevents duplicate indexing in vector store)
    assert hash_a == hash_b
    # Different content -> different hash
    assert hash_a != hash_c


def test_blank_scanned_pdf(tmp_path: Path):
    """Verify that blank or scanned PDFs with no text layers do not crash the loader."""
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    docs = load_pdf(pdf_path)
    # Since page contains no digital text, it returns an empty list without crashing
    assert len(docs) == 0


def test_file_not_found():
    """Verify that attempting to load a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pdf("non_existent_file.pdf")


def test_zero_byte_file(tmp_path: Path):
    """Verify that zero-byte files raise a descriptive ValueError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="empty \\(0 bytes\\)"):
        load_pdf(empty_file)


def test_corrupted_pdf_file(tmp_path: Path):
    """Verify that corrupted non-PDF files raise a descriptive ValueError."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_text("This is not a PDF at all")

    with pytest.raises(ValueError, match="Cannot read"):
        load_pdf(corrupt_file)


def test_load_multiple_pdfs(tmp_path: Path):
    """Verify batch loading across multiple PDFs."""
    pdf_1 = tmp_path / "doc1.pdf"
    pdf_2 = tmp_path / "doc2.pdf"

    pdf_1.write_bytes(create_sample_pdf(["Doc 1 page"]))
    pdf_2.write_bytes(create_sample_pdf(["Doc 2 page"]))

    docs = load_multiple_pdfs([pdf_1, pdf_2])
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "doc1.pdf"
    assert docs[1].metadata["source"] == "doc2.pdf"

