# Multi-Document RAG Assistant

**Final Project — Integrating Labs 1, 2, 3, and 4**

A complete Streamlit-based Retrieval-Augmented Generation (RAG) Q&A application supporting multiple document formats with hybrid search and metadata filtering.

---

## Labs Integrated

| Lab | Concept | Implementation |
|-----|---------|----------------|
| **Lab 1** | Document Indexing with ChromaDB | `document_processor.py` — load, chunk, embed, store |
| **Lab 2** | RAG Q&A System | `rag_pipeline.py` — context-grounded OpenAI answer |
| **Lab 3** | Hybrid Search + Metadata Filtering | `retriever.py` — BM25 + vector + RRF fusion |
| **Lab 4** | Multi-Document RAG | `app.py` — PDF, CSV, TXT uploaded together |

---

## Project Structure

```
multi-document-rag/
├── app.py                  # Streamlit UI — entry point
├── document_processor.py   # File loading → chunking → ChromaDB indexing
├── retriever.py            # Hybrid BM25 + vector search + metadata filtering
├── rag_pipeline.py         # RAG chain: context building + OpenAI LLM call
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── README.md               # This file
└── data/
    └── chroma_db/          # Persistent ChromaDB storage (auto-created)
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the template
copy .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Usage

### Step 1 — Upload Documents
- Click **Browse files** in the sidebar
- Select one or more PDF, CSV, or TXT files
- Optionally enter a **Product Category** tag (e.g., `Electronics`)
- Click **📥 Index**

### Step 2 — Ask Questions
- Type your question in the main area
- Optionally set **retrieval filters** (category, file type, document name)
- Adjust **Top-K** to control how many chunks the LLM sees
- Click **🚀 Get Answer**

### Step 3 — Review Sources
The answer is displayed along with:
- Source filename
- Page number (PDFs) or row number (CSVs)
- Product category
- Content preview

---

## RAG Pipeline (Complete Flow)

```
Upload Documents
      ↓
Document Loaders (PyPDFLoader / CSVLoader / TextLoader)
      ↓
Text Extraction
      ↓
Chunking (RecursiveCharacterTextSplitter)
      ↓
Metadata Attachment (filename, file_type, page, category)
      ↓
OpenAI Embeddings (text-embedding-3-small)
      ↓
ChromaDB Indexing (Persistent)
      ↓
User Question
      ↓
Metadata Filtering (category / file_type / filename)
      ↓
Hybrid Retrieval:
  ├── Semantic/Vector Search (ChromaDB)
  └── Keyword Search (BM25 / rank-bm25)
      ↓
Reciprocal Rank Fusion (RRF)
      ↓
Top-K Relevant Chunks
      ↓
RAG Context Building
      ↓
OpenAI LLM (gpt-4o-mini)
      ↓
Grounded Answer + Source Citations
```

---

## Supported File Types

| Format | Loader | Source Info Shown |
|--------|--------|-------------------|
| PDF | `PyPDFLoader` | Page number |
| CSV | `CSVLoader` | Row number |
| TXT | `TextLoader` | Section |

---

## Key Features

- ✅ Multi-file upload in one session
- ✅ Persistent ChromaDB (survives restarts)
- ✅ Deduplication (same file won't be indexed twice)
- ✅ BM25 + vector hybrid search with RRF
- ✅ Metadata filtering by category, file type, document name
- ✅ Grounded answers — LLM uses ONLY the retrieved context
- ✅ Source citations with filename, page/row, and content preview
- ✅ Configurable Top-K retrieval
- ✅ Clear/reset database button
- ✅ Chat conversation history

---

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| Streamlit | Web UI |
| LangChain | Document loaders, text splitting |
| ChromaDB | Vector database (persistent) |
| pypdf | PDF parsing |
| rank-bm25 | BM25 keyword search |
| OpenAI API | Embeddings + LLM |
| python-dotenv | Environment variable loading |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |

Never commit your `.env` file to version control.
