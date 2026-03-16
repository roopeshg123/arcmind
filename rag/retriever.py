"""
Hybrid retriever for ArcMind.

Combines vector similarity search (ChromaDB) and keyword search (BM25)
using Reciprocal Rank Fusion (RRF) to produce a single merged ranking.

The hybrid approach catches both:
  - Semantically similar content (vector)
  - Exact keyword/phrase matches (BM25)

… which together significantly improve recall for technical documentation
and Jira issue retrieval.
"""

from __future__ import annotations

import logging
from langchain_core.documents import Document
from vector_db.chroma_store import get_store

log = logging.getLogger(__name__)

# RRF constant — standard value from the literature (Cormack et al. 2009)
_RRF_K = 60


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def _rrf_merge(
    vector_docs: list[Document],
    bm25_docs:   list[Document],
    k: int,
) -> list[Document]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.

    RRF score for document d = Σ_r  1 / (RRF_K + rank_r(d))

    Documents are deduplicated by the first 200 characters of their content.
    Returns up to *k* highest-scoring documents.
    """
    scores: dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

    def _key(doc: Document) -> str:
        return doc.page_content[:200]

    for rank, doc in enumerate(vector_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        doc_map.setdefault(key, doc)

    for rank, doc in enumerate(bm25_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        doc_map.setdefault(key, doc)

    ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ordered[:k]]


# ---------------------------------------------------------------------------
# Per-collection hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    queries:          list[str],
    collection:       str = "docs",
    k:                int = 10,
    connector_filter: str | None = None,
) -> list[Document]:
    """
    Run hybrid retrieval for one or more query strings.

    For each query:
      1. Vector search (ChromaDB cosine similarity)
      2. BM25 keyword search
      3. RRF merge

    All query results are merged and de-duplicated.

    Args:
        queries:          Expanded query list (original + expansions).
        collection:       "docs" or "jira".
        k:                How many documents to return in total.
        connector_filter: Optional Arc connector to filter results by.
    """
    store = get_store()

    # Docs use "section" metadata; Jira uses "component".
    # Only apply the filter to Jira — doc sections are not reliably populated
    # for every connector so filtering would silently drop relevant pages.
    filter_meta: dict | None = None
    if connector_filter and collection == "jira":
        filter_meta = {"component": {"$eq": connector_filter}}

    per_query_k = max(k, 12)   # fetch extra candidates so RRF has a rich pool
    accumulated: dict[str, Document] = {}

    for query in queries:
        # Vector search
        if collection == "docs":
            vec = store.similarity_search_docs(query, k=per_query_k, filter_metadata=filter_meta)
        elif collection == "confluence":
            vec = store.similarity_search_confluence(query, k=per_query_k, filter_metadata=filter_meta)
        else:
            vec = store.similarity_search_jira(query, k=per_query_k, filter_metadata=filter_meta)

        # BM25 search
        bm25 = store.bm25_search(query, collection=collection, k=per_query_k)

        # RRF merge for this query
        merged = _rrf_merge(vec, bm25, k=per_query_k)

        # Accumulate across all expanded queries (deduplicate by content prefix)
        for doc in merged:
            key = doc.page_content[:200]
            if key not in accumulated:
                accumulated[key] = doc

    return list(accumulated.values())[:k]


# ---------------------------------------------------------------------------
# Dual-collection convenience
# ---------------------------------------------------------------------------

def retrieve_docs_and_jira(
    queries:          list[str],
    docs_k:           int = 6,
    jira_k:           int = 4,
    connector_filter: str | None = None,
) -> tuple[list[Document], list[Document]]:
    """
    Retrieve from both docs and Jira collections simultaneously.

    Returns:
        (docs_results, jira_results)
    """
    docs_results = hybrid_search(
        queries, collection="docs", k=docs_k, connector_filter=connector_filter
    )
    jira_results = hybrid_search(
        queries, collection="jira", k=jira_k, connector_filter=connector_filter
    )
    return docs_results, jira_results


def retrieve_all(
    queries:          list[str],
    docs_k:           int = 6,
    jira_k:           int = 4,
    confluence_k:     int = 4,
    connector_filter: str | None = None,
) -> tuple[list[Document], list[Document], list[Document]]:
    """
    Retrieve from docs, Jira, and Confluence collections simultaneously.

    Returns:
        (docs_results, jira_results, confluence_results)
    """
    docs_results       = hybrid_search(
        queries, collection="docs", k=docs_k, connector_filter=connector_filter
    )
    jira_results       = hybrid_search(
        queries, collection="jira", k=jira_k, connector_filter=connector_filter
    )
    confluence_results = hybrid_search(
        queries, collection="confluence", k=confluence_k, connector_filter=connector_filter
    )
    return docs_results, jira_results, confluence_results
