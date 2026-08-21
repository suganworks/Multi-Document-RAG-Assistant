"""
document_processor.py — Lab 1 Integration
==========================================
Handles loading, chunking, embedding, and indexing of documents into ChromaDB.

Supported formats:
- PDF  → PyPDFLoader
- CSV  → CSVLoader
- TXT  → TextLoader

Embeddings: sentence-transformers (local, no API key needed)
  Model: all-MiniLM-L6-v2 (384 dimensions, cached after first download)

Each chunk is stored with metadata:
  filename, file_type, page (PDFs), row (CSVs), source, product_category
"""

import os
import logging
from typing import List, Dict, Tuple

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
COLLECTION_NAME = "multi_doc_rag"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Cached embedding model (loaded once, reused for all calls)
# ─────────────────────────────────────────────────────────────────────────────
_embedding_model: SentenceTransformer = None


def get_embedding_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer model (loads on first call)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB client (persistent across restarts)
# ─────────────────────────────────────────────────────────────────────────────

def get_chroma_client() -> PersistentClient:
    """Return a persistent ChromaDB client, creating the directory if needed."""
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection(client: PersistentClient):
    """
    Get or create the ChromaDB collection.
    No embedding function passed — embeddings are computed manually
    and provided directly to add() / query() calls.
    """
    return client.get_or_create_collection(name=COLLECTION_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Document loading
# ─────────────────────────────────────────────────────────────────────────────

def load_document(file_path: str, file_type: str):
    """
    Load a document using the appropriate LangChain loader.
    Returns a list of LangChain Document objects.
    """
    file_path = str(file_path)

    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    elif file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    docs = loader.load()
    logger.info(f"Loaded {len(docs)} document pages/rows from {file_path}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Text splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(docs, file_type: str):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Metadata enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_metadata(chunks, filename: str, file_type: str, product_category: str = "General"):
    """Attach uniform metadata to each chunk."""
    for chunk in chunks:
        existing = chunk.metadata or {}
        page = existing.get("page", "N/A")
        row = existing.get("row", None)

        chunk.metadata = {
            "filename": filename,
            "file_type": file_type,
            "page": str(page) if page != "N/A" else "N/A",
            "row": str(row) if row is not None else "N/A",
            "source": filename,
            "product_category": product_category,
        }
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication check
# ─────────────────────────────────────────────────────────────────────────────

def is_already_indexed(collection, filename: str) -> bool:
    """Returns True if this filename is already in ChromaDB."""
    try:
        results = collection.get(
            where={"filename": {"$eq": filename}},
            limit=1,
        )
        return len(results["ids"]) > 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main indexing entry point
# ─────────────────────────────────────────────────────────────────────────────

def index_document(
    file_path: str,
    filename: str,
    file_type: str,
    product_category: str = "General",
) -> Tuple[int, str]:
    """
    Full pipeline: load → split → enrich metadata → embed → store in ChromaDB.

    Embeddings are computed locally using sentence-transformers (no API key).
    Vectors are passed directly via embeddings= to collection.add().

    Returns:
        (chunks_added: int, message: str)
    """
    client = get_chroma_client()
    collection = get_collection(client)

    # Deduplication: skip if already indexed
    if is_already_indexed(collection, filename):
        return 0, f"'{filename}' is already indexed. Skipping."

    # Load document
    try:
        docs = load_document(file_path, file_type)
    except Exception as e:
        return 0, f"Failed to load '{filename}': {e}"

    # Split into chunks
    chunks = split_documents(docs, file_type)
    if not chunks:
        return 0, f"No text extracted from '{filename}'."

    # Enrich metadata
    chunks = enrich_metadata(chunks, filename, file_type, product_category)

    # Prepare texts, metadata, and IDs
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]

    # ── Compute embeddings locally (sentence-transformers) ────────────────────
    try:
        model = get_embedding_model()
        vectors = model.encode(texts, show_progress_bar=False).tolist()
    except Exception as e:
        return 0, f"Failed to generate embeddings for '{filename}': {e}"

    # ── Store in ChromaDB with explicit embeddings ────────────────────────────
    try:
        collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        return 0, f"Failed to index '{filename}' in ChromaDB: {e}"

    logger.info(f"Indexed {len(chunks)} chunks from '{filename}'")
    return len(chunks), f"Indexed '{filename}' -> {len(chunks)} chunks added."


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db_stats() -> Dict:
    """Return stats about the ChromaDB collection."""
    try:
        client = get_chroma_client()
        collection = get_collection(client)
        count = collection.count()

        filenames: List[str] = []
        if count > 0:
            all_meta = collection.get(include=["metadatas"])
            filenames = list({m.get("filename", "unknown") for m in all_meta["metadatas"]})

        return {"chunk_count": count, "documents": filenames}
    except Exception as e:
        logger.error(f"Error getting DB stats: {e}")
        return {"chunk_count": 0, "documents": []}


def clear_database() -> str:
    """Delete the ChromaDB collection entirely."""
    try:
        client = get_chroma_client()
        client.delete_collection(COLLECTION_NAME)
        return "Database cleared successfully."
    except Exception as e:
        return f"Could not clear database: {e}"
