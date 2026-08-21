"""
rag_pipeline.py — Lab 2 + Lab 4 Integration
============================================
RAG pipeline: builds prompt from retrieved chunks, calls Groq LLM, returns
a grounded answer with cited sources.

Uses Groq API (free tier) — ultra-fast inference with LLaMA / Mixtral models.
The LLM is strictly instructed to answer ONLY from the provided context.
"""

import logging
from typing import List, Dict, Any, Tuple

# pyrefly: ignore [missing-import]
from groq import Groq

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
4. When citing information, refer to the source document naturally (e.g., "According to products_catalog.csv...").
5. Do not guess, infer, or hallucinate information not present in the context.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Build a readable context string from retrieved chunks."""
    if not chunks:
        return "No relevant documents were found."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        file_type = meta.get("file_type", "")
        page = meta.get("page", "N/A")
        row = meta.get("row", "N/A")

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
    """Convert retrieved chunks into structured source citations for display."""
    sources = []
    seen = set()

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        file_type = meta.get("file_type", "")
        page = meta.get("page", "N/A")
        row = meta.get("row", "N/A")
        category = meta.get("product_category", "General")

        if file_type == "pdf" and page != "N/A":
            location = f"Page {page}"
        elif file_type == "csv" and row != "N/A":
            location = f"Row {row}"
        else:
            location = "-"

        key = f"{filename}|{location}"
        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "filename": filename,
            "file_type": file_type.upper() if file_type else "-",
            "location": location,
            "category": category,
            "preview": chunk["content"][:200].replace("\n", " ") + "...",
            "hybrid_score": str(chunk.get("hybrid_score", "-")),
        })

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_rag(
    question: str,
    chunks: List[Dict[str, Any]],
    groq_api_key: str,
    model: str = "llama-3.1-8b-instant",
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Run the full RAG pipeline given pre-retrieved chunks.

    Args:
        question:     The user's question.
        chunks:       Top-K retrieved chunks from hybrid_retrieve().
        groq_api_key: Groq API key for LLM inference.
        model:        Groq model to use (llama/mixtral/gemma).

    Returns:
        (answer: str, sources: List[Dict])
    """
    if not chunks:
        return (
            "I couldn't find the answer in the uploaded documents. "
            "Please upload relevant documents first.",
            [],
        )

    context = build_context(chunks)

    user_message = f"""Context:
{context}

---

Question: {question}

Please answer the question based only on the context above."""

    # ── Call Groq API ─────────────────────────────────────────────────────────
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        answer = f"Groq API error: {e}"

    sources = format_sources(chunks)
    return answer, sources
