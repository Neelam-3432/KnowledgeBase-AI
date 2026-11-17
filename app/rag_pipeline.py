"""End-to-End RAG Pipeline module for KnowledgeBase AI.

Orchestrates the complete document ingestion and question-answering workflows
by connecting document loading, text chunking, local embeddings, persistent
ChromaDB storage, semantic retrieval, and Google Gemini answer generation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chromadb.api.models.Collection import Collection
from google import genai

from app.answer_generator import GeneratedAnswer, generate_answer
from app.config import GEMINI_MODEL, TOP_K
from app.document_loader import calculate_file_hash, load_pdf
from app.retriever import SearchResult, retrieve
from app.text_splitter import split_documents
from app.vector_store import (
    add_documents,
    get_collection,
    get_document_count,
    is_document_indexed,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Structured result of a document ingestion operation."""
    document_name: str
    doc_id: str
    chunks_count: int
    status: str  # "success", "already_indexed", "no_text_extracted"
    metadata: Dict[str, Any]
    message: str


@dataclass
class RAGAnswer:
    """Structured response from the end-to-end RAG question answering pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    model: str
    retrieved_chunks_count: int = 0

    @property
    def filenames(self) -> List[str]:
        """Return distinct filenames cited in sources."""
        seen = set()
        names = []
        for s in self.sources:
            name = s.get("source")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    @property
    def pages(self) -> List[int]:
        """Return distinct page numbers cited in sources."""
        seen = set()
        pages_list = []
        for s in self.sources:
            page = s.get("page")
            if page is not None and page not in seen:
                seen.add(page)
                pages_list.append(page)
        return pages_list


def ingest_document(
    pdf_path: Union[str, Path],
    collection: Optional[Collection] = None,
) -> IngestionResult:
    """Ingest a PDF document into the local RAG knowledge base.

    Workflow:
    1. Validate file path and PDF format.
    2. Compute SHA-256 hash to detect duplicate uploads.
    3. If already indexed, skip without duplicating chunks.
    4. Extract text page-by-page using DocumentLoader.
    5. Split into overlapping chunks at word boundaries using TextSplitter.
    6. Generate local SentenceTransformer embeddings and store in ChromaDB.

    Args:
        pdf_path: Path to the target PDF file.
        collection: Optional target ChromaDB collection (useful for testing).

    Returns:
        IngestionResult with status, chunk counts, document ID, and metadata.

    Raises:
        TypeError: If pdf_path is not a str or Path.
        FileNotFoundError: If the file does not exist on disk.
        ValueError: If file is not a PDF, is 0 bytes, or cannot be parsed.
    """
    if not isinstance(pdf_path, (str, Path)):
        raise TypeError(
            f"pdf_path must be a string or Path, got {type(pdf_path).__name__}."
        )

    path = Path(pdf_path)

    if not str(path).strip():
        raise ValueError("pdf_path cannot be empty or whitespace-only.")

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"File '{path.name}' is not a valid PDF file. Expected '.pdf' extension."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty (0 bytes): {path.name}")

    target_collection = collection or get_collection()

    # Step 1: Calculate SHA-256 hash for document identity & duplicate detection
    doc_id = calculate_file_hash(path)

    # Step 2: Prevent duplicate document indexing
    if is_document_indexed(doc_id, collection=target_collection):
        logger.info(
            f"Document '{path.name}' (doc_id={doc_id[:12]}...) is already indexed. Skipping ingestion."
        )
        return IngestionResult(
            document_name=path.name,
            doc_id=doc_id,
            chunks_count=0,
            status="already_indexed",
            metadata={"doc_id": doc_id, "source": path.name},
            message=f"Document '{path.name}' is already indexed in the knowledge base. Ingestion skipped.",
        )

    # Step 3: Load pages from PDF
    page_docs = load_pdf(path)

    if len(page_docs) == 0:
        logger.warning(
            f"No extractable text found in '{path.name}'. It may be a scanned image or blank PDF."
        )
        return IngestionResult(
            document_name=path.name,
            doc_id=doc_id,
            chunks_count=0,
            status="no_text_extracted",
            metadata={"doc_id": doc_id, "source": path.name},
            message=f"No extractable text found in '{path.name}'. File may be a scanned image or blank PDF.",
        )

    # Step 4: Split into chunks preserving page citations
    chunk_docs = split_documents(page_docs)

    if len(chunk_docs) == 0:
        logger.warning(f"No chunks generated for '{path.name}'.")
        return IngestionResult(
            document_name=path.name,
            doc_id=doc_id,
            chunks_count=0,
            status="no_text_extracted",
            metadata={"doc_id": doc_id, "source": path.name},
            message=f"No chunks could be generated from '{path.name}'.",
        )

    # Step 5: Embed and store in ChromaDB
    chunks_stored = add_documents(chunk_docs, collection=target_collection)

    total_pages = page_docs[0].metadata.get("total_pages", len(page_docs))
    metadata = {
        "doc_id": doc_id,
        "source": path.name,
        "total_pages": total_pages,
        "chunks_count": chunks_stored,
    }

    logger.info(
        f"Successfully ingested '{path.name}': {chunks_stored} chunks indexed across {total_pages} page(s)."
    )

    return IngestionResult(
        document_name=path.name,
        doc_id=doc_id,
        chunks_count=chunks_stored,
        status="success",
        metadata=metadata,
        message=f"Successfully indexed {chunks_stored} chunks from '{path.name}' ({total_pages} pages).",
    )


def answer_question(
    question: str,
    top_k: int = TOP_K,
    collection: Optional[Collection] = None,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> RAGAnswer:
    """Answer a user question using semantic retrieval and Gemini answer generation.

    Workflow:
    1. Validate question and top_k parameters.
    2. Retrieve Top-K nearest chunks from ChromaDB using local SentenceTransformers.
    3. If no chunks found, return immediately without calling Gemini API (saving quota/cost).
    4. Assemble grounded prompt and generate answer with Gemini.
    5. Return structured RAGAnswer with text, sources, model, and chunk count.

    Args:
        question: User search question string.
        top_k: Number of relevant chunks to retrieve. Defaults to TOP_K from config.
        collection: Optional ChromaDB collection (useful for testing).
        client: Optional genai.Client instance (useful for testing/injection).
        model: Optional model identifier override. Defaults to GEMINI_MODEL.

    Returns:
        RAGAnswer containing answer text, source citations, model ID, and chunk count.

    Raises:
        TypeError: If question is not a string, or top_k is a boolean / non-integer.
        ValueError: If question is empty or whitespace-only, or top_k <= 0.
    """
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}.")

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question cannot be empty or whitespace-only.")

    # Strict top_k validation: reject booleans and non-integers, require > 0
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    target_model = model or GEMINI_MODEL
    target_collection = collection or get_collection()

    # Step 1: Semantic retrieval
    chunks: List[SearchResult] = retrieve(
        query=cleaned_question,
        top_k=top_k,
        collection=target_collection,
    )

    # Step 2: Empty retrieval safeguard — skip Gemini API call
    if not chunks:
        logger.info(
            f"No relevant chunks retrieved for query: '{cleaned_question[:40]}'. "
            "Skipping Gemini API call."
        )
        return RAGAnswer(
            answer="No relevant documents were found in the knowledge base to answer your question.",
            sources=[],
            model=target_model,
            retrieved_chunks_count=0,
        )

    # Step 3: Generate grounded answer via Gemini
    generated: GeneratedAnswer = generate_answer(
        question=cleaned_question,
        chunks=chunks,
        client=client,
        model=target_model,
    )

    return RAGAnswer(
        answer=generated.answer,
        sources=generated.sources,
        model=generated.model,
        retrieved_chunks_count=len(chunks),
    )


def get_pipeline_status(collection: Optional[Collection] = None) -> Dict[str, Any]:
    """Return summary status of the RAG pipeline and vector store.

    Args:
        collection: Optional ChromaDB collection.

    Returns:
        Dictionary with indexed chunk count, collection name, and model configured.
    """
    target_collection = collection or get_collection()
    count = get_document_count(collection=target_collection)
    return {
        "total_chunks_indexed": count,
        "collection_name": target_collection.name,
        "gemini_model": GEMINI_MODEL,
    }

