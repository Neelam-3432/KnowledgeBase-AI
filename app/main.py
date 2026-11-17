"""KnowledgeBase AI - Streamlit Web Application.

Provides a clean, modern conversational user interface for:
1. Uploading and indexing multi-page PDF documents into local ChromaDB storage.
2. Viewing knowledge base status, indexing metrics, and model configuration.
3. Querying documents with semantic retrieval and Groq LLM answer generation.
4. Viewing factual page-level citations for every generated answer.
5. Managing conversation history and resetting the vector database with confirmation.
"""

import uuid
from pathlib import Path

import streamlit as st

from app.config import GROQ_API_KEY, GROQ_MODEL, TOP_K, UPLOADS_DIR
from app.rag_pipeline import (
    answer_question,
    get_pipeline_status,
    ingest_document,
)
from app.vector_store import reset_collection


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="KnowledgeBase AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # -------------------------------------------------------------------------
    # Sidebar: Document Management & Configuration
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("📚 KnowledgeBase AI")
        st.caption("RAG & Agentic AI Document Intelligence")
        st.markdown("---")

        # 1. Groq System Configuration Status (Inspection only — no API call on load)
        st.markdown("### ⚙️ LLM Configuration")
        has_valid_key = bool(
            GROQ_API_KEY
            and GROQ_API_KEY.strip()
            and not ("your_" in GROQ_API_KEY.lower() or "placeholder" in GROQ_API_KEY.lower())
        )

        if has_valid_key:
            st.success("🔑 **API Key**: Configured (Masked)")
        else:
            st.error("❌ **API Key**: Missing or Placeholder")
            st.caption("Configure `GROQ_API_KEY` in your `.env` file.")

        st.info(f"🤖 **Active Model**: `{GROQ_MODEL}`")
        st.markdown("---")

        # 2. Document Ingestion Section
        st.markdown("### 📄 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to index into your local vector database.",
        )

        if uploaded_files:
            if st.button("🚀 Ingest & Index Documents", use_container_width=True, type="primary"):
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()
                    if not file_bytes:
                        st.warning(f"Skipping '{uploaded_file.name}': file is empty (0 bytes).")
                        continue

                    # Safe target path in UPLOADS_DIR, avoiding collisions for different contents
                    target_path = UPLOADS_DIR / uploaded_file.name
                    if target_path.exists():
                        existing_bytes = target_path.read_bytes()
                        if existing_bytes != file_bytes:
                            # Same filename but different content: assign unique safe name
                            unique_name = f"{target_path.stem}_{uuid.uuid4().hex[:6]}{target_path.suffix}"
                            target_path = UPLOADS_DIR / unique_name

                    target_path.write_bytes(file_bytes)

                    with st.spinner(f"Processing '{uploaded_file.name}'..."):
                        try:
                            result = ingest_document(target_path)
                            if result.status == "success":
                                st.success(f"✅ {result.message}")
                            elif result.status == "already_indexed":
                                st.info(f"ℹ️ {result.message}")
                            elif result.status == "no_text_extracted":
                                st.warning(f"⚠️ {result.message}")
                            else:
                                st.warning(f"⚠️ {result.message}")
                        except Exception as exc:
                            st.error(f"❌ Failed to ingest '{uploaded_file.name}': {exc}")

        st.markdown("---")

        # 3. Knowledge Base Status & Metrics
        pipeline_status = get_pipeline_status()
        total_chunks = pipeline_status.get("total_chunks_indexed", 0)
        collection_name = pipeline_status.get("collection_name", "knowledge_base")

        st.markdown("### 📊 Knowledge Base Status")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Chunks Stored", total_chunks)
        col_m2.metric("Collection", collection_name)
        st.markdown("---")

        # 4. Search Configuration
        st.markdown("### 🔍 Retrieval Settings")
        top_k = st.slider(
            "Retrieval Depth (Top-K)",
            min_value=1,
            max_value=10,
            value=TOP_K,
            help="Number of most relevant text passages to retrieve per query.",
        )
        st.markdown("---")

        # 5. Conversation & Database Management
        st.markdown("### 🛠️ Actions")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        with st.expander("⚠️ Reset Knowledge Base"):
            st.warning("Permanently removes all indexed document chunks from ChromaDB.")
            confirm_reset = st.checkbox(
                "I understand this deletes all indexed data.",
                key="confirm_db_reset",
            )
            if st.button("Confirm Reset Database", type="secondary", disabled=not confirm_reset):
                reset_collection()
                st.session_state["messages"] = []
                st.success("Knowledge base reset successfully.")
                st.rerun()

    # -------------------------------------------------------------------------
    # Main Panel: Conversational Q&A Interface
    # -------------------------------------------------------------------------
    st.header("💬 KnowledgeBase AI Assistant")
    st.caption("Ask questions grounded strictly in your uploaded PDF documents.")

    # Empty State Guidance
    if not st.session_state["messages"]:
        if total_chunks == 0:
            st.info(
                "👋 **Welcome! Your knowledge base is currently empty.**\n\n"
                "**How to get started:**\n"
                "1. Use the **sidebar** on the left to upload one or more PDF files.\n"
                "2. Click **'Ingest & Index Documents'** to extract text, create chunks, and generate local embeddings.\n"
                "3. Type your question in the chat box below to receive grounded answers with exact page citations."
            )
        else:
            st.info(
                f"📖 **Knowledge Base Ready**: **{total_chunks} chunk(s)** currently indexed in `{collection_name}`.\n\n"
                "Ask any question below to retrieve relevant passages and generate verified answers."
            )

    # Render Chat History
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Render Citations if present
            sources = message.get("sources", [])
            if sources:
                with st.expander(f"📚 View {len(sources)} Source Citation(s)"):
                    for src in sources:
                        src_name = src.get("source", "Unknown file")
                        page_num = src.get("page", "N/A")
                        chunk_id = src.get("chunk_id", "")
                        st.markdown(f"- 📄 **{src_name}** — Page **{page_num}** `({chunk_id})`")

    # Handle New User Question
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Record and display user question
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate and display assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base & generating answer with Groq..."):
                try:
                    rag_answer = answer_question(question=prompt, top_k=top_k)
                except Exception as exc:
                    st.error(f"Error querying knowledge base: {exc}")
                    return

            st.markdown(rag_answer.answer)

            # Display citations
            if rag_answer.sources:
                with st.expander(f"📚 View {len(rag_answer.sources)} Source Citation(s)"):
                    for src in rag_answer.sources:
                        src_name = src.get("source", "Unknown file")
                        page_num = src.get("page", "N/A")
                        chunk_id = src.get("chunk_id", "")
                        st.markdown(f"- 📄 **{src_name}** — Page **{page_num}** `({chunk_id})`")

            # Record assistant response in session state
            st.session_state["messages"].append({
                "role": "assistant",
                "content": rag_answer.answer,
                "sources": rag_answer.sources,
                "model": rag_answer.model,
            })


if __name__ == "__main__":
    main()
