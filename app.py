"""
LunorAI – Intelligent Knowledge Assistant
A production-grade, modular RAG web application built with Streamlit,
LangChain, LangGraph, FAISS, and Ollama.
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules import reliably
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from config import settings
from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import DocumentChunker
from embeddings.embedding_model import get_embedding_model
from vectorstore.faiss_store import FAISSStore
from graph.workflow import create_rag_graph
from llm.ollama_model import check_ollama_status
from utils.helpers import validate_pdf_file


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model():
    """Create the embedding client once per Streamlit process."""
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def get_cached_llm():
    """Create the Ollama chat client once per Streamlit process."""
    from llm.ollama_model import get_llm

    return get_llm(streaming=True)


@st.cache_data(ttl=10, show_spinner=False)
def get_cached_ollama_status():
    """Avoid a network health check on every Streamlit rerun."""
    return check_ollama_status()

# Page Configuration
st.set_page_config(
    page_title="LunorAI – Intelligent Knowledge Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, professional aesthetic
st.markdown(
    """
    <style>
    /* Global Typography & Spacing */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-status {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .badge-online {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
    }
    .badge-offline {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F8B4B4;
    }
    .citation-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        padding: 0.65rem 0.9rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }
    .citation-header {
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 0.2rem;
    }
    .citation-snippet {
        color: #475569;
        font-style: italic;
        font-size: 0.82rem;
    }
    .doc-stat-box {
        background-color: #F1F5F9;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initializes Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faiss_store" not in st.session_state:
        st.session_state.faiss_store = None
    if "rag_graph" not in st.session_state:
        st.session_state.rag_graph = None
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = []
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
    if "answer_stream_sink" not in st.session_state:
        st.session_state.answer_stream_sink = {"callback": None}


init_session_state()

# -------------------------------------------------------------
# Sidebar: Document Management & System Status
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    ollama_ok, ollama_msg = get_cached_ollama_status()
    if ollama_ok:
        st.markdown(
            f'<div class="badge-status badge-online">● Ollama Online: {settings.OLLAMA_MODEL}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="badge-status badge-offline">● Ollama Offline</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"⚠️ {ollama_msg}")

    st.markdown("---")
    st.markdown("### 📂 Upload PDF Documents")
    st.caption("Upload one or multiple PDF documents to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard PDF documents (up to 50 MB each).",
    )

    if uploaded_files:
        st.markdown(f"**Selected Files ({len(uploaded_files)}):**")
        for f in uploaded_files:
            file_size_kb = len(f.getvalue()) / 1024
            st.markdown(f"- 📄 `{f.name}` ({file_size_kb:.1f} KB)")

    process_btn = st.button(
        "🚀 Process Documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_btn and uploaded_files:
        with st.status("Processing and indexing documents...", expanded=True) as status:
            try:
                # 1. Validation
                st.write("🔍 Validating PDF documents...")
                valid_items = []
                for f in uploaded_files:
                    f_bytes = f.getvalue()
                    is_valid, msg = validate_pdf_file(f.name, f_bytes)
                    if not is_valid:
                        st.error(f"{f.name}: {msg}")
                    else:
                        valid_items.append({"bytes": f_bytes, "name": f.name})

                if not valid_items:
                    st.error("No valid PDF files found to process.")
                    status.update(label="Document processing failed.", state="error")
                else:
                    # 2. Text Extraction & Metadata Preservation
                    st.write(f"📖 Extracting text from {len(valid_items)} PDF(s)...")
                    loader = PDFLoader()
                    raw_docs = loader.load_multiple(valid_items)

                    if not raw_docs:
                        st.warning("Extracted 0 text pages. Ensure PDFs contain searchable text.")
                        status.update(label="No readable text found.", state="error")
                    else:
                        # 3. Document Chunking
                        st.write(f"✂️ Chunking {len(raw_docs)} pages (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
                        chunker = DocumentChunker()
                        chunks = chunker.split_documents(raw_docs)

                        # 4. Embeddings Generation & FAISS Vector Indexing
                        st.write(f"🧠 Generating embeddings & indexing {len(chunks)} chunks in FAISS...")
                        embed_model = get_cached_embedding_model()
                        store = FAISSStore(embeddings=embed_model)
                        store.build_from_documents(chunks)

                        # 5. Compile LangGraph Workflow
                        st.write("⚡ Compiling LangGraph RAG workflow...")
                        graph = create_rag_graph(
                            faiss_store=store,
                            llm=get_cached_llm(),
                            token_callback=lambda token: (
                                st.session_state.answer_stream_sink["callback"]
                                and st.session_state.answer_stream_sink["callback"](token)
                            ),
                        )

                        # Save to session state
                        st.session_state.faiss_store = store
                        st.session_state.rag_graph = graph
                        st.session_state.total_chunks = len(chunks)

                        # Document summary
                        summary = []
                        for f in valid_items:
                            pages = set(d.metadata.get("page") for d in raw_docs if d.metadata.get("source") == f["name"])
                            summary.append({"name": f["name"], "pages": len(pages)})
                        st.session_state.processed_docs = summary

                        status.update(
                            label=f"✅ Successfully indexed {len(chunks)} chunks across {len(valid_items)} document(s)!",
                            state="complete",
                        )
                        st.rerun()

            except Exception as ex:
                st.error(f"Error during ingestion: {str(ex)}")
                status.update(label="Indexing failed.", state="error")

    # Display Active Document Index Summary
    if st.session_state.processed_docs:
        st.markdown("---")
        st.markdown("### 📊 Active Knowledge Base")
        st.markdown(
            f"<div class='doc-stat-box'><b>Total Chunks:</b> {st.session_state.total_chunks}<br>"
            f"<b>Documents:</b> {len(st.session_state.processed_docs)}</div>",
            unsafe_allow_html=True,
        )
        for doc in st.session_state.processed_docs:
            st.caption(f"• **{doc['name']}** ({doc['pages']} pages)")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("LunorAI v1.0 • LangGraph RAG Engine")


# -------------------------------------------------------------
# Main Chat Area
# -------------------------------------------------------------
st.markdown('<div class="main-header">✨ LunorAI – Intelligent Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload research papers, technical specs, or reports and ask questions with verified source and page citations.</div>',
    unsafe_allow_html=True,
)

# If no documents have been processed yet, show a helpful onboarding banner
if not st.session_state.rag_graph:
    st.info(
        "👋 **Welcome to LunorAI!** To start asking questions:\n"
        "1. Upload one or more PDF files using the sidebar on the left.\n"
        "2. Click **'Process Documents'** to extract, chunk, and index them into FAISS.\n"
        "3. Ask your questions below to receive answers grounded in your documents with page citations."
    )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If assistant response contains source citations, render them cleanly
        if msg.get("sources"):
            with st.expander("📚 Source Citations", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">📄 Source: {src['source']} &nbsp;|&nbsp; 📖 Page: {src['page']}</div>
                            <div class="citation-snippet">"{src['snippet']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# User Chat Input
user_query = st.chat_input(
    placeholder="Ask a question about your uploaded documents...",
    disabled=st.session_state.rag_graph is None,
)

if user_query:
    # 1. Display user message in UI and add to state
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Run LangGraph Workflow
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents & verifying context..."):
            try:
                streamed_answer = []
                answer_placeholder = st.empty()

                def render_token(token: str):
                    streamed_answer.append(token)
                    answer_placeholder.markdown("".join(streamed_answer))

                st.session_state.answer_stream_sink["callback"] = render_token

                # Invoke LangGraph state machine
                initial_state = {
                    "question": user_query,
                    "documents": [],
                    "relevance_scores": [],
                    "is_relevant": False,
                    "answer": "",
                    "sources": [],
                }

                graph = st.session_state.rag_graph
                final_state = graph.invoke(initial_state)
                st.session_state.answer_stream_sink["callback"] = None

                answer = final_state.get("answer", "No answer generated.")
                sources = final_state.get("sources", [])

                # Render Answer
                answer_placeholder.markdown(answer)

                # Render Citations
                if sources:
                    with st.expander("📚 Source Citations", expanded=True):
                        for src in sources:
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-header">📄 Source: {src['source']} &nbsp;|&nbsp; 📖 Page: {src['page']}</div>
                                    <div class="citation-snippet">"{src['snippet']}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save assistant turn to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                st.session_state.answer_stream_sink["callback"] = None
                err_msg = f"⚠️ An error occurred while executing the RAG pipeline: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "sources": [],
                })
