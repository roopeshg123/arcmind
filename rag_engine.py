"""
ArcMind RAG Engine

Orchestrates the full retrieval-augmented generation pipeline:

  1. Query routing      — detect Jira ticket IDs
  2. Connector detection — identify which Arc connector is being asked about
  3. Query expansion    — generate semantically related search queries
  4. Hybrid retrieval   — vector search + BM25 from docs and Jira collections
  5. Reranking          — cross-encoder re-scores the candidate pool
  6. Prompt building    — inject docs + clustered Jira context into the prompt
  7. LLM invocation     — streaming or blocking GPT-4.1 call
  8. Memory             — server-side per-session conversation history
  9. Query logging      — append to JSONL file for self-improvement analysis

Backward-compatible API
-----------------------
The public functions used by main.py are preserved:
    is_vector_store_ready(), load_vector_store(), get_rag_chain(),
    warmup_reranker(), reset_chain(), ask(question, chat_history)
New additions:
    ask(…, session_id)   — server-side memory
    ask_stream(…)        — async generator for SSE streaming
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
import concurrent.futures
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.connector_detector  import detect_connector
from rag.conversation_memory import get_memory
from rag.prompt_builder      import build_messages
from rag.query_expander      import expand_query
from rag.query_router        import route_query
from rag.reranker            import rerank, warmup
from rag.retriever           import retrieve_docs_and_jira, retrieve_all
from vector_db.chroma_store  import get_store, reset_store
from connectors.jira_client  import fetch_remote_links

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL     = os.getenv("CHAT_MODEL",     "gpt-4.1")
CHROMA_DB_DIR  = os.getenv("CHROMA_DB_DIR",  "./chroma_db")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))
LOG_DIR        = os.getenv("LOG_DIR",         "./logs")

os.makedirs(LOG_DIR, exist_ok=True)
_QUERY_LOG = os.path.join(LOG_DIR, "query_log.jsonl")

# Result-count tunables (env-configurable)
_DOCS_TOP_K       = int(os.getenv("DOCS_TOP_K",       "6"))
_JIRA_TOP_K       = int(os.getenv("JIRA_TOP_K",       "4"))
_CONFLUENCE_TOP_K = int(os.getenv("CONFLUENCE_TOP_K", "4"))

# LLM singletons — created once per process to avoid per-request re-instantiation
_llm:           ChatOpenAI | None = None
_llm_streaming: ChatOpenAI | None = None


def _get_llm(streaming: bool = False) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance (regular or streaming)."""
    global _llm, _llm_streaming
    if streaming:
        if _llm_streaming is None:
            _llm_streaming = ChatOpenAI(
                model=CHAT_MODEL,
                openai_api_key=OPENAI_API_KEY,
                temperature=0.1,
                streaming=True,
            )
        return _llm_streaming
    if _llm is None:
        _llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1,
        )
    return _llm

# ---------------------------------------------------------------------------
# Backward-compatible status / lifecycle helpers (called by main.py)
# ---------------------------------------------------------------------------

class _CollectionProxy:
    """Lets main.py call vs._collection.count() without change."""
    def __init__(self, store) -> None:
        self._store = store

    def count(self) -> int:
        return self._store.docs_count() + self._store.jira_count() + self._store.confluence_count()


class _StoreProxy:
    """Thin wrapper returned by load_vector_store() for main.py compatibility."""
    def __init__(self, store) -> None:
        self._collection = _CollectionProxy(store)


def is_vector_store_ready() -> bool:
    """Return True if the docs collection has at least one vector."""
    return get_store().is_docs_ready()


def load_vector_store() -> _StoreProxy:
    """Return a proxy with ._collection.count() for the /api/status endpoint."""
    return _StoreProxy(get_store())


def release_vector_store() -> None:
    """Release ChromaDB handles (Windows file-lock safety)."""
    reset_store()


def warmup_reranker() -> None:
    """Pre-load the cross-encoder reranker to eliminate cold-start latency."""
    warmup()


def get_rag_chain():
    """Backward-compat stub — the new engine does not use a LangChain chain object."""
    return _DirectChain()


class _DirectChain:
    """Truthy placeholder so existing `if get_rag_chain()` checks still pass."""


def reset_chain() -> None:
    """Release all cached objects (ChromaDB handles, BM25 index, LLM, etc.)."""
    global _llm, _llm_streaming
    _llm           = None
    _llm_streaming = None
    reset_store()


# ---------------------------------------------------------------------------
# Query-level logging (self-improvement)
# ---------------------------------------------------------------------------

def _log_query(
    question:        str,
    expanded:        list[str],
    n_docs:          int,
    n_jira:          int,
    answer_preview:  str,
    n_confluence:    int = 0,
) -> None:
    """Append a structured query record to the JSONL log file."""
    try:
        entry = {
            "ts":               time.time(),
            "question":         question,
            "expanded_queries": expanded,
            "docs_retrieved":   n_docs,
            "jira_retrieved":   n_jira,
            "confluence_retrieved": n_confluence,
            "answer_preview":   answer_preview[:500],
        }
        with open(_QUERY_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Query log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Path-in-question normaliser (kept from v1 for backward compat)
# ---------------------------------------------------------------------------

import re
from pathlib import Path as _Path


def _extract_topic_from_path(text: str) -> str | None:
    pattern = r'(?:file:///|file://)?[A-Za-z]:[\\//][^\s"<>]+\.html|[^\s"<>]+\.html'
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    raw  = m.group(0)
    stem = _Path(re.sub(r'file:///|file://', '', raw).replace('%20', ' ')).stem
    stem = re.sub(r'^op_', '', stem)
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
    words = re.sub(r'[-_]', ' ', words)
    return f"{stem} {words} Arc operation connector documentation"


def _normalize_question(question: str) -> str:
    """Rewrite pasted file paths / URLs into searchable topic strings."""
    topic = _extract_topic_from_path(question)
    if topic is None:
        return question
    cleaned = re.sub(
        r'(?:file:///|file://)?[A-Za-z]:[\\//][^\s"<>]+\.html|[^\s"<>]+\.html',
        '', question, flags=re.IGNORECASE,
    ).strip()
    return f"{cleaned} — topic: {topic}" if cleaned else f"Explain {topic} in detail"


# ---------------------------------------------------------------------------
# Core retrieval pipeline (shared by ask() and ask_stream())
# ---------------------------------------------------------------------------

def _run_pipeline(
    question:     str,
    session_id:   str | None,
    chat_history: list[dict],
) -> tuple[list, list, list, list[str], list[dict]]:
    """
    Execute routing → connector detection → expansion → hybrid retrieval → rerank.

    Returns:
        (docs, jira_docs, confluence_docs, expanded_queries, effective_chat_history)
    """
    # Merge server-side memory with any client-provided history
    if session_id:
        memory  = get_memory()
        stored  = memory.get_history(session_id)
        if stored and not chat_history:
            chat_history = stored

    # 1. Detect connector
    connector = detect_connector(question)
    if connector:
        log.info("Connector detected: %s", connector)

    # 2. Route
    route = route_query(question)

    # 3. Expand — always use LLM expansion for better recall on short/ambiguous queries
    expanded = expand_query(question, connector=connector, use_llm=True)
    log.info("Query expansion: %d variant(s).", len(expanded))

    # 4. Retrieve
    if route.strategy == "ticket_direct" and route.ticket_ids:
        log.info("Direct ticket fetch: %s", route.ticket_ids)
        jira_docs       = get_store().get_jira_by_tickets(route.ticket_ids)

        # Live-fetch remote links (GitHub/Bitbucket PRs) for the queried tickets.
        # These are NOT stored in ChromaDB — they live only in Jira's dev panel.
        # We inject them as synthetic documents so the LLM can surface PR links.
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                remote_links_map = _pool.submit(
                    asyncio.run, fetch_remote_links(route.ticket_ids)
                ).result(timeout=15)
        except Exception as _e:
            log.warning("Remote links fetch skipped: %s", _e)
            remote_links_map = {}

        for ticket_key, links in remote_links_map.items():
            if not links:
                continue
            link_lines = [f"Remote Links / Pull Requests for {ticket_key}:"]
            for lnk in links:
                rel   = f" [{lnk['relationship']}]" if lnk.get("relationship") else ""
                title = lnk.get("title") or lnk.get("url", "")
                url   = lnk.get("url", "")
                link_lines.append(f"  - {title}{rel}: {url}")
            from langchain_core.documents import Document as _Doc
            jira_docs.append(_Doc(
                page_content="\n".join(link_lines),
                metadata={"source": "jira", "ticket": ticket_key, "type": "remote_links"},
            ))

        docs, _, conf   = retrieve_all(
            [question],
            docs_k=_DOCS_TOP_K, jira_k=0, confluence_k=_CONFLUENCE_TOP_K,
            connector_filter=connector,
        )
        confluence_docs = conf
    else:
        docs, jira_docs, confluence_docs = retrieve_all(
            expanded,
            docs_k=_DOCS_TOP_K, jira_k=_JIRA_TOP_K, confluence_k=_CONFLUENCE_TOP_K,
            connector_filter=connector,
        )

    # 5. Rerank combined pool (keep source split afterwards)
    combined = docs + jira_docs + confluence_docs
    if combined:
        if route.strategy == "ticket_direct":
            # For direct ticket fetches keep ALL Jira chunks — the description,
            # workarounds, and comments are all needed for a full QA analysis.
            # Only rerank the docs/confluence side.
            doc_conf_combined = docs + confluence_docs
            if doc_conf_combined:
                reranked_dc = rerank(question, doc_conf_combined, top_n=_DOCS_TOP_K + _CONFLUENCE_TOP_K)
                docs            = [d for d in reranked_dc if d.metadata.get("source") not in ("jira", "confluence")][:_DOCS_TOP_K]
                confluence_docs = [d for d in reranked_dc if d.metadata.get("source") == "confluence"][:_CONFLUENCE_TOP_K]
            # jira_docs already contains exactly the fetched ticket chunks — keep all
        else:
            total_top_n = _DOCS_TOP_K + _JIRA_TOP_K + _CONFLUENCE_TOP_K
            reranked  = rerank(question, combined, top_n=total_top_n)
            docs            = [d for d in reranked if d.metadata.get("source") not in ("jira", "confluence")][:_DOCS_TOP_K]
            jira_docs       = [d for d in reranked if d.metadata.get("source") == "jira"][:_JIRA_TOP_K]
            confluence_docs = [d for d in reranked if d.metadata.get("source") == "confluence"][:_CONFLUENCE_TOP_K]

    return docs, jira_docs, confluence_docs, expanded, chat_history


def _build_sources(docs: list, jira_docs: list, confluence_docs: list | None = None) -> list[dict]:
    """Build the deduplicated sources list returned to the client."""
    seen: set[str] = set()
    sources: list[dict] = []
    for doc in docs + jira_docs + (confluence_docs or []):
        key = (
            doc.metadata.get("file_path")
            or doc.metadata.get("url")
            or doc.metadata.get("ticket")
            or doc.metadata.get("page_id")
            or "unknown"
        )
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":  key,
                "type":    doc.metadata.get("source",    "unknown"),
                "section": doc.metadata.get("section",   ""),
                "ticket":  doc.metadata.get("ticket",    ""),
                "title":   doc.metadata.get("title",     ""),
                "content": doc.page_content[:300] + (
                    "..." if len(doc.page_content) > 300 else ""
                ),
            })
    return sources


# ---------------------------------------------------------------------------
# Public blocking API
# ---------------------------------------------------------------------------

def ask(
    question:     str,
    chat_history: list[dict] | None = None,
    session_id:   str | None = None,
) -> dict[str, Any]:
    """
    Run a full RAG query and return the answer (blocking).

    Args:
        question:     The user's question.
        chat_history: Prior turns as [{"role": "user"|"assistant", "content": "…"}].
        session_id:   Optional session ID for server-side conversation memory.

    Returns:
        {
            "answer":      str,
            "sources":     list[dict],
            "jira_issues": list[dict],
        }
    """
    if chat_history is None:
        chat_history = []

    question = _normalize_question(question)

    docs, jira_docs, confluence_docs, expanded, history = _run_pipeline(
        question, session_id=session_id, chat_history=chat_history
    )

    messages = build_messages(question, docs, jira_docs, history, confluence_docs=confluence_docs)

    llm = _get_llm()

    try:
        response = llm.invoke(messages)
        answer   = response.content
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        raise

    # Persist to server-side memory
    if session_id:
        get_memory().add_turn(session_id, question, answer)

    _log_query(question, expanded, len(docs), len(jira_docs), answer, n_confluence=len(confluence_docs))

    jira_issues = [
        {
            "ticket":    d.metadata.get("ticket",    ""),
            "status":    d.metadata.get("status",    ""),
            "component": d.metadata.get("component", ""),
        }
        for d in jira_docs if d.metadata.get("ticket")
    ]

    return {
        "answer":      answer,
        "sources":     _build_sources(docs, jira_docs, confluence_docs),
        "jira_issues": jira_issues,
    }


# ---------------------------------------------------------------------------
# Public streaming API
# ---------------------------------------------------------------------------

async def ask_stream(
    question:     str,
    chat_history: list[dict] | None = None,
    session_id:   str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream the RAG answer as Server-Sent Events.

    Each yield is an SSE-formatted string:
        "data: {…JSON…}\n\n"

    Event types:
        {"token":  "…"}           — incremental answer token
        {"error":  "…"}           — error (stream ends)
        {"done": true,
         "sources": […],
         "jira_issues": […]}      — final metadata event
    """
    if chat_history is None:
        chat_history = []

    question = _normalize_question(question)

    loop = asyncio.get_event_loop()
    docs, jira_docs, confluence_docs, expanded, history = await loop.run_in_executor(
        None,
        functools.partial(
            _run_pipeline, question, session_id=session_id, chat_history=chat_history
        ),
    )

    messages = build_messages(question, docs, jira_docs, history, confluence_docs=confluence_docs)

    llm = _get_llm(streaming=True)

    tokens: list[str] = []
    try:
        async for chunk in llm.astream(messages):
            token = chunk.content
            if token:
                tokens.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception as exc:
        log.error("LLM streaming error: %s", exc)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        return

    answer = "".join(tokens)

    if session_id:
        get_memory().add_turn(session_id, question, answer)

    _log_query(question, expanded, len(docs), len(jira_docs), answer, n_confluence=len(confluence_docs))

    jira_issues = [
        {
            "ticket":    d.metadata.get("ticket",    ""),
            "status":    d.metadata.get("status",    ""),
            "component": d.metadata.get("component", ""),
        }
        for d in jira_docs if d.metadata.get("ticket")
    ]

    yield f"data: {json.dumps({'done': True, 'sources': _build_sources(docs, jira_docs, confluence_docs), 'jira_issues': jira_issues})}\n\n"
