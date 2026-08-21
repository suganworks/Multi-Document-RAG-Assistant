"""
rag_pipeline.py — Lab 2 + Lab 4 Integration
============================================
RAG pipeline: builds prompt from retrieved chunks, calls OpenAI, returns
a grounded answer with cited sources.

The LLM is strictly instructed to answer ONLY from the provided context.
If the context does not contain an answer, it returns a polite "not found"
message rather than hallucinating.

This module integrates:
- Lab 2: RAG Q&A with OpenAI
- Lab 4: Multi-document support (PDF + CSV + TXT sources shown together)
"""

import logging
from typing import List, Dict, Any, Tuple

# pyrefly: ignore [missing-import]
from openai import OpenAI

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful document assistant. Your job is to answer the user's question
using ONLY the information provided in the context below. 

Rules:
1. Base your answer strictly on the context. Do NOT use outside knowledge.
2. If the answer is not found in the context, respond with exactly:
   "I couldn't find the answer in the uploaded documents."
3. Always be concise and accurate.
4. When citing information, refer to the source document naturally (e.g., "According to product_manual.pdf...").
5. Do not guess, infer, or hallucinate information not present in the context.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Build a readable context string from retrieved chunks.
    Each chunk is labelled with its source document and location.
    """
    if not chunks:
        return "No relevant documents were found."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        file_type = meta.get("file_type", "")
        page = meta.get("page", "N/A")
        row = meta.get("row", "N/A")

        # Build a location label
        if file_type == "pdf" and page != "N/A":
            location = f"Page {page}"
        elif file_type == "csv" and row != "N/A":
            location = f"Row {row}"
        else:
            location = "Section 1"

        context_parts.append(
            f"[Source {i}: {filename} — {location}]\n{chunk['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Source formatter (for Streamlit display)
# ─────────────────────────────────────────────────────────────────────────────

def format_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert retrieved chunks into structured source citations for display.

    Returns a list of dicts with: filename, file_type, location, preview.
    """
    sources = []
    seen = set()

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        file_type = meta.get("file_type", "")
        page = meta.get("page", "N/A")
        row = meta.get("row", "N/A")
        category = meta.get("product_category", "General")

        # Build location string
        if file_type == "pdf" and page != "N/A":
            location = f"Page {page}"
        elif file_type == "csv" and row != "N/A":
            location = f"Row {row}"
        else:
            location = "–"

        # Deduplicate by filename + location
        key = f"{filename}|{location}"
        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "filename": filename,
            "file_type": file_type.upper() if file_type else "–",
            "location": location,
            "category": category,
            "preview": chunk["content"][:200].replace("\n", " ") + "...",
            "hybrid_score": str(chunk.get("hybrid_score", "–")),
        })

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_rag(
    question: str,
    chunks: List[Dict[str, Any]],
    openai_api_key: str,
    model: str = "gpt-4o-mini",
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Run the full RAG pipeline given pre-retrieved chunks.

    Args:
        question:       The user's question.
        chunks:         Top-K retrieved chunks from hybrid_retrieve().
        openai_api_key: OpenAI API key.
        model:          OpenAI model to use.

    Returns:
        (answer: str, sources: List[Dict])
    """
    if not chunks:
        return (
            "I couldn't find the answer in the uploaded documents. "
            "Please upload relevant documents first.",
            [],
        )

    # Build context from retrieved chunks
    context = build_context(chunks)

    # Compose user message
    user_message = f"""Context:
{context}

---

Question: {question}

Please answer the question based only on the context above."""

    # Call OpenAI Chat Completion
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,  # deterministic answers for RAG
            max_tokens=800,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        answer = f"❌ OpenAI API error: {e}"

    # Format sources for display
    sources = format_sources(chunks)

    return answer, sources
