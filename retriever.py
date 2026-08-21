"""
retriever.py — Lab 2 + Lab 3 Integration
=========================================
Hybrid retrieval: semantic/vector search (ChromaDB) + keyword search (BM25).
Results are combined using Reciprocal Rank Fusion (RRF).

Embeddings: sentence-transformers (local, same model as document_processor)
No API key required for retrieval.

Metadata filtering supported on:
- product_category
- file_type
- filename (document name)
"""

import logging
from typing import List, Dict, Optional, Any

from rank_bm25 import BM25Okapi

from document_processor import get_chroma_client, get_collection, get_embedding_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata filter builder (ChromaDB `where` clause)
# ─────────────────────────────────────────────────────────────────────────────

def build_where_clause(
    product_category: Optional[str] = None,
    file_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Dict]:
    """Build a ChromaDB metadata filter clause from user-selected filters."""
    conditions = []

    if product_category and product_category != "All":
        conditions.append({"product_category": {"$eq": product_category}})
    if file_type and file_type != "All":
        conditions.append({"file_type": {"$eq": file_type.lower()}})
    if filename and filename != "All":
        conditions.append({"filename": {"$eq": filename}})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_ranking: List[str],
    vector_ranking: List[str],
    all_ids: List[str],
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    k: int = 60,
) -> Dict[str, float]:
    """Combine BM25 and vector ranked lists via weighted Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {doc_id: 0.0 for doc_id in all_ids}

    for rank, doc_id in enumerate(bm25_ranking):
        if doc_id in scores:
            scores[doc_id] += bm25_weight * (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(vector_ranking):
        if doc_id in scores:
            scores[doc_id] += vector_weight * (1.0 / (k + rank + 1))

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main retriever function
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    product_category: Optional[str] = None,
    file_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval combining:
      1. ChromaDB vector/semantic search (local sentence-transformer embedding)
      2. BM25 keyword search

    No API key required — embeddings are computed locally.

    Returns top_k chunks with content and metadata.
    """
    client = get_chroma_client()
    collection = get_collection(client)

    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    # ── Compute query embedding locally ──────────────────────────────────────
    try:
        model = get_embedding_model()
        query_vector = model.encode(query, show_progress_bar=False).tolist()
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []

    # Build metadata filter
    where_clause = build_where_clause(product_category, file_type, filename)

    # ── Step 1: Vector search via ChromaDB ───────────────────────────────────
    n_vector = min(max(top_k * 4, 20), total_chunks)
    chroma_kwargs: Dict[str, Any] = dict(
        query_embeddings=[query_vector],
        n_results=n_vector,
        include=["metadatas", "distances", "documents"],
    )
    if where_clause:
        chroma_kwargs["where"] = where_clause

    try:
        vector_results = collection.query(**chroma_kwargs)
    except Exception as e:
        logger.error(f"ChromaDB vector query failed: {e}")
        return []

    vector_ids: List[str] = vector_results["ids"][0]
    vector_distances: List[float] = vector_results["distances"][0]
    vector_docs: List[str] = vector_results["documents"][0]
    vector_metas: List[Dict] = vector_results["metadatas"][0]

    # cosine distance → similarity (1 = identical)
    vector_scores_map: Dict[str, float] = {
        doc_id: max(0.0, 1.0 - dist)
        for doc_id, dist in zip(vector_ids, vector_distances)
    }

    id_to_content: Dict[str, str] = dict(zip(vector_ids, vector_docs))
    id_to_meta: Dict[str, Dict] = dict(zip(vector_ids, vector_metas))

    # ── Step 2: Fetch all chunks for BM25 (respecting filter) ────────────────
    fetch_kwargs: Dict[str, Any] = dict(
        include=["documents", "metadatas"],
        limit=total_chunks,
    )
    if where_clause:
        fetch_kwargs["where"] = where_clause

    try:
        all_data = collection.get(**fetch_kwargs)
    except Exception as e:
        logger.error(f"ChromaDB get() for BM25 failed: {e}")
        all_data = {"ids": [], "documents": [], "metadatas": []}

    all_ids: List[str] = all_data["ids"]
    all_docs: List[str] = all_data["documents"]
    all_metas: List[Dict] = all_data["metadatas"]

    for doc_id, content, meta in zip(all_ids, all_docs, all_metas):
        if doc_id not in id_to_content:
            id_to_content[doc_id] = content
            id_to_meta[doc_id] = meta

    if not all_ids:
        return []

    # ── BM25 keyword search ───────────────────────────────────────────────────
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_raw_scores = bm25.get_scores(tokenized_query)

    bm25_items = list(zip(all_ids, bm25_raw_scores.tolist()))
    bm25_items.sort(key=lambda x: x[1], reverse=True)
    bm25_ranking: List[str] = [item[0] for item in bm25_items]
    bm25_scores_map: Dict[str, float] = {item[0]: item[1] for item in bm25_items}

    # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────────────
    candidate_ids = list(set(vector_ids) | set(bm25_ranking))
    hybrid_scores = reciprocal_rank_fusion(
        bm25_ranking=bm25_ranking,
        vector_ranking=vector_ids,
        all_ids=candidate_ids,
        bm25_weight=0.4,
        vector_weight=0.6,
    )

    sorted_candidates = sorted(
        candidate_ids,
        key=lambda doc_id: hybrid_scores.get(doc_id, 0.0),
        reverse=True,
    )

    # ── Step 4: Assemble top-K results ────────────────────────────────────────
    results = []
    for doc_id in sorted_candidates[:top_k]:
        content = id_to_content.get(doc_id, "")
        meta = id_to_meta.get(doc_id, {})
        if not content:
            continue
        results.append({
            "id": doc_id,
            "content": content,
            "metadata": meta,
            "hybrid_score": round(hybrid_scores.get(doc_id, 0.0), 6),
            "bm25_score": round(bm25_scores_map.get(doc_id, 0.0), 4),
            "vector_score": round(vector_scores_map.get(doc_id, 0.0), 4),
        })

    logger.info(f"Hybrid retrieval returned {len(results)} chunks for: '{query[:60]}'")
    return results
