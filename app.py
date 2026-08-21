"""
app.py — Multi-Document RAG Assistant
======================================
Final project integrating Labs 1–4:
  Lab 1 -> Document indexing with ChromaDB
  Lab 2 -> RAG Q&A with Groq LLM
  Lab 3 -> Hybrid search (BM25 + vector) + metadata filtering
  Lab 4 -> Multi-document support (PDF, CSV, TXT)

Run with:
    python -m streamlit run app.py
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Add project root to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from document_processor import index_document, get_db_stats, clear_database
from retriever import hybrid_retrieve
from rag_pipeline import run_rag

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Document RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    min-height: 100vh;
}

/* Hide Streamlit default header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6366f1 !important;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.3);
}

/* Hero title */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Lab badges */
.lab-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}

.lab-badge {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a5b4fc;
}

/* Answer card */
.answer-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.answer-card .answer-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.75rem;
}

.answer-card .answer-text {
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.7;
}

/* Source card */
.source-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}

.source-filename {
    font-weight: 600;
    color: #a5b4fc;
    font-size: 0.9rem;
}

.source-meta {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 0.2rem;
}

.source-preview {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 0.4rem;
    font-style: italic;
    line-height: 1.5;
}

/* Stats card */
.stats-card {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    margin: 0.5rem 0;
}

.stats-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
}

.stats-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Pipeline diagram */
.pipeline-step {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: #a5b4fc;
    margin: 0.15rem;
}

/* Chat history bubble */
.chat-bubble-user {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
}

.chat-bubble-ai {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 12px 12px 12px 4px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
}

/* Streamlit buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}

/* Info/warning boxes */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 8px;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 1.5rem 0;
}

/* File type icons */
.file-icon-pdf { color: #f87171; }
.file-icon-csv { color: #4ade80; }
.file-icon-txt { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect file type
# ─────────────────────────────────────────────────────────────────────────────
def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext in ("pdf",):
        return "pdf"
    elif ext in ("csv",):
        return "csv"
    elif ext in ("txt", "text"):
        return "txt"
    return "unknown"


FILE_TYPE_ICONS = {"pdf": "📄", "csv": "📊", "txt": "📝"}
FILE_TYPE_COLORS = {"pdf": "#f87171", "csv": "#4ade80", "txt": "#60a5fa"}


# ─────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────  SIDEBAR  ───────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo / title
    st.markdown("## 📚 RAG Assistant")
    st.markdown("<div class='section-header'>⚙️ Configuration</div>", unsafe_allow_html=True)

    # Groq API key
    api_key_env = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=api_key_env,
        type="password",
        placeholder="gsk_...",
        help="Free Groq API key from console.groq.com. Set GROQ_API_KEY in .env to auto-fill.",
    )

    # Model selection
    model_choice = st.selectbox(
        "LLM Model",
        ["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
        help="llama-3.1-8b-instant is fastest. mixtral-8x7b gives best quality.",
    )

    st.markdown("<div class='section-header'>📂 Document Upload</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF, CSV, or TXT files.",
        label_visibility="collapsed",
    )

    # Product category input
    product_category = st.text_input(
        "Product Category (optional)",
        placeholder="e.g. Electronics, HR, Finance",
        help="Tag uploaded documents with a category for filtering later.",
    )
    if not product_category.strip():
        product_category = "General"

    # Index button
    col_idx, col_clr = st.columns(2)
    with col_idx:
        index_btn = st.button("📥 Index", use_container_width=True)
    with col_clr:
        clear_btn = st.button("🗑️ Clear DB", use_container_width=True)

    # ── Handle indexing ───────────────────────────────────────────────────────
    if index_btn:
        if not groq_api_key:
            st.error("Please enter your Groq API Key.")
        elif not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            total_chunks = 0
            messages = []
            progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                file_type = detect_file_type(uploaded_file.name)
                if file_type == "unknown":
                    messages.append(f"⚠️ Unsupported file: {uploaded_file.name}")
                    continue

                # Save to temp file (Streamlit file objects need to be written to disk)
                suffix = f".{file_type}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    n_chunks, msg = index_document(
                        file_path=tmp_path,
                        filename=uploaded_file.name,
                        file_type=file_type,
                        product_category=product_category,
                    )
                    total_chunks += n_chunks
                    messages.append(msg)

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                progress.progress((i + 1) / len(uploaded_files))

            for msg in messages:
                if "✅" in msg:
                    st.success(msg)
                elif "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.error(msg)

            if total_chunks > 0:
                st.balloons()

    # ── Handle clear ──────────────────────────────────────────────────────────
    if clear_btn:
        msg = clear_database()
        if "✅" in msg:
            st.success(msg)
            st.session_state.chat_history = []
            st.session_state.last_answer = None
            st.session_state.last_sources = []
        else:
            st.warning(msg)

    # ── Database stats ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🗄️ Database Stats</div>", unsafe_allow_html=True)

    stats = get_db_stats()

    st.markdown(f"""
    <div class='stats-card'>
        <div class='stats-number'>{stats['chunk_count']}</div>
        <div class='stats-label'>Indexed Chunks</div>
    </div>
    """, unsafe_allow_html=True)

    doc_list = stats.get("documents", [])
    if doc_list:
        st.markdown(f"**{len(doc_list)} document(s) indexed:**")
        for doc in doc_list:
            ftype = detect_file_type(doc)
            icon = FILE_TYPE_ICONS.get(ftype, "📄")
            st.markdown(f"&nbsp;&nbsp;{icon} `{doc}`", unsafe_allow_html=True)
    else:
        st.caption("No documents indexed yet.")

    # ── Retrieval filters ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Retrieval Filters</div>", unsafe_allow_html=True)

    filter_category = st.text_input(
        "Category Filter",
        placeholder="e.g. Electronics  (leave blank for All)",
        key="filter_cat",
    )

    filter_filetype = st.selectbox(
        "File Type Filter",
        options=["All", "PDF", "CSV", "TXT"],
        key="filter_ft",
    )

    filter_docname = st.selectbox(
        "Document Filter",
        options=["All"] + doc_list,
        key="filter_dn",
    )

    top_k = st.slider(
        "Top-K Chunks",
        min_value=1,
        max_value=15,
        value=5,
        help="Number of most relevant chunks to retrieve and pass to the LLM.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────  MAIN AREA  ──────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Multi-Document RAG Assistant</div>
<div class='hero-subtitle'>Ask questions about your uploaded PDF, CSV, and TXT documents.</div>
<div class='lab-badges'>
  <span class='lab-badge'>📦 Lab 1 · ChromaDB Indexing</span>
  <span class='lab-badge'>🤖 Lab 2 · RAG Q&amp;A</span>
  <span class='lab-badge'>🔍 Lab 3 · Hybrid Search</span>
  <span class='lab-badge'>📚 Lab 4 · Multi-Document</span>
</div>
""", unsafe_allow_html=True)

# ── Pipeline overview (collapsible) ───────────────────────────────────────────
with st.expander("📊 How this works — RAG Pipeline", expanded=False):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
**Document Indexing (Upload side)**
1. Upload PDF / CSV / TXT files
2. Detect file type → correct LangChain loader
3. Extract text & split into chunks
4. Attach metadata (filename, page, category)
5. Generate local embeddings (sentence-transformers)
6. Store in persistent ChromaDB
        """)
    with col2:
        st.markdown("""
**Retrieval & Answer (Query side)**
1. Apply metadata filters
2. Semantic/vector search in ChromaDB
3. BM25 keyword search on same corpus
4. Combine with Reciprocal Rank Fusion
5. Select Top-K relevant chunks
6. Build context -> send to Groq LLM
7. Return grounded answer + sources
        """)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ── Chat history display ──────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("#### 💬 Conversation History")
    for turn in st.session_state.chat_history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            st.markdown(
                f"<div class='chat-bubble-user'>🧑 <strong>You:</strong> {content}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-bubble-ai'>🤖 <strong>Assistant:</strong> {content}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ── Question input ─────────────────────────────────────────────────────────────
st.markdown("#### 🤔 Ask a Question")

with st.form(key="qa_form", clear_on_submit=True):
    user_question = st.text_area(
        "Your question",
        placeholder="Ask a question about your documents…\n\nExample: What is the warranty period for Product X?",
        height=100,
        label_visibility="collapsed",
    )
    submit_btn = st.form_submit_button("🚀 Get Answer", use_container_width=True)

# ── Process query ──────────────────────────────────────────────────────────────
if submit_btn and user_question.strip():
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif stats["chunk_count"] == 0:
        st.warning("⚠️ No documents indexed yet. Please upload and index documents first.")
    else:
        # Apply filters
        cat_filter = filter_category.strip() if filter_category.strip() else None
        ft_filter = filter_filetype if filter_filetype != "All" else None
        dn_filter = filter_docname if filter_docname != "All" else None

        with st.spinner("Retrieving relevant chunks..."):
            chunks = hybrid_retrieve(
                query=user_question,
                top_k=top_k,
                product_category=cat_filter,
                file_type=ft_filter,
                filename=dn_filter,
            )

        with st.spinner("Generating answer with Groq..."):
            answer, sources = run_rag(
                question=user_question,
                chunks=chunks,
                groq_api_key=groq_api_key,
                model=model_choice,
            )

        # Store in session state
        st.session_state.last_answer = answer
        st.session_state.last_sources = sources
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Refresh to show updated chat history
        st.rerun()

# ── Display last answer + sources ─────────────────────────────────────────────
if st.session_state.last_answer:
    st.markdown("#### 💡 Answer")
    st.markdown(
        f"""<div class='answer-card'>
            <div class='answer-label'>🤖 AI Answer</div>
            <div class='answer-text'>{st.session_state.last_answer}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    sources = st.session_state.last_sources
    if sources:
        st.markdown("#### 📎 Sources")
        st.caption(f"Answer generated from {len(sources)} source(s):")

        for src in sources:
            icon = FILE_TYPE_ICONS.get(src["file_type"].lower(), "📄")
            st.markdown(
                f"""<div class='source-card'>
                    <div class='source-filename'>{icon} {src['filename']}</div>
                    <div class='source-meta'>
                        Type: <strong>{src['file_type']}</strong> &nbsp;|&nbsp;
                        Location: <strong>{src['location']}</strong> &nbsp;|&nbsp;
                        Category: <strong>{src['category']}</strong>
                    </div>
                    <div class='source-preview'>"{src['preview']}"</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Clear answer button
    st.markdown("")
    if st.button("🗑️ Clear Conversation", key="clear_conv"):
        st.session_state.chat_history = []
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.rerun()

# ── Empty state ────────────────────────────────────────────────────────────────
elif not st.session_state.chat_history:
    st.markdown("""
    <div style='text-align:center; padding: 3rem 1rem; color: #475569;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>📚</div>
        <div style='font-size: 1.1rem; font-weight: 500; color: #64748b;'>
            No documents indexed yet
        </div>
        <div style='font-size: 0.85rem; margin-top: 0.5rem;'>
            Upload PDF, CSV, or TXT files in the sidebar and click <strong>Index</strong> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#334155; font-size:0.75rem;'>"
    "Multi-Document RAG Assistant · Labs 1-4 Integration · "
    "Powered by LangChain · ChromaDB · Groq · sentence-transformers"
    "</div>",
    unsafe_allow_html=True,
)
