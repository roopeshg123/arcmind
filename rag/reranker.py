"""
Cross-encoder reranker for ArcMind.

Uses a HuggingFace cross-encoder model to re-score retrieved documents
against the user query and keep only the most relevant ones.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, ~80 MB)

Pipeline position:
    hybrid retrieval (top ~10) → reranker → top 5 → LLM
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

log = logging.getLogger(__name__)

RERANKER_MODEL   = os.getenv("RERANKER_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N   = int(os.getenv("RERANKER_TOP_N",   "5"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

_cross_encoder = None   # lazy singleton


def _get_model():
    """Lazy-load the cross-encoder (downloads model on first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        log.info("Loading cross-encoder: %s", RERANKER_MODEL)
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        log.info("Cross-encoder ready.")
    return _cross_encoder


def warmup() -> None:
    """Pre-load the model to eliminate cold-start latency at startup."""
    if RERANKER_ENABLED:
        _get_model()


def rerank(
    query:     str,
    documents: list[Document],
    top_n:     int | None = None,
) -> list[Document]:
    """
    Re-score *documents* against *query* and return the top-*top_n*.

    Falls back to truncated input on any error so the pipeline never breaks.

    Args:
        query:     The user question.
        documents: Candidate documents from hybrid retrieval.
        top_n:     Documents to keep.  Defaults to RERANKER_TOP_N env var.

    Returns:
        List of Documents sorted by relevance (highest first), length ≤ top_n.
    """
    if top_n is None:
        top_n = RERANKER_TOP_N

    if not RERANKER_ENABLED or not documents:
        return documents[:top_n]

    try:
        model  = _get_model()
        pairs  = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for _, doc in ranked[:top_n]]

    except Exception as exc:
        log.error("Reranker failed (falling back to truncation): %s", exc)
        return documents[:top_n]
