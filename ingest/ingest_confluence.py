"""
Confluence ingestion pipeline for ArcMind.

Fetches pages via the Confluence REST API, converts them to structured
text documents with rich metadata, chunks them, and stores them in the
'arcmind_confluence' ChromaDB collection.

Supports:
  - Full initial ingest of all pages in configured spaces
  - Incremental sync (pages updated in the last N hours)
  - Smart update (only re-indexes pages that changed since last ingest)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document

from connectors.confluence_client import fetch_pages, CONFLUENCE_SPACES
from ingest.chunking import chunk_documents
from vector_db.chroma_store import get_store

load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page → structured text
# ---------------------------------------------------------------------------

def _format_page_text(page: dict) -> str:
    """
    Render a Confluence page dict as human-readable structured text.
    """
    def _s(v) -> str:
        return str(v).strip() if v is not None else ""

    lines: list[str] = [
        f"Page: {_s(page.get('title'))}",
        f"Space: {_s(page.get('space_name'))} ({_s(page.get('space_key'))})",
    ]

    if page.get("breadcrumb"):
        lines.append(f"Path: {_s(page['breadcrumb'])}")
    if page.get("labels"):
        lines.append(f"Labels: {', '.join(_s(l) for l in page['labels'])}")
    if page.get("updated"):
        lines.append(f"Last Updated: {_s(page['updated'])[:10]}")
    if page.get("url"):
        lines.append(f"URL: {_s(page['url'])}")

    body = _s(page.get("body"))
    if body:
        lines.extend(["", body[:8000]])   # cap body to avoid enormous chunks

    return "\n".join(lines)


def pages_to_documents(pages: list[dict]) -> list[Document]:
    """Convert Confluence page dicts to LangChain Documents."""
    docs: list[Document] = []
    for page in pages:
        text = _format_page_text(page)
        if not text.strip():
            continue

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        docs.append(Document(
            page_content=text,
            metadata={
                "source":       "confluence",
                "page_id":      page.get("id", ""),
                "title":        page.get("title", ""),
                "space_key":    page.get("space_key", ""),
                "space_name":   page.get("space_name", ""),
                "url":          page.get("url", ""),
                "updated":      page.get("updated", ""),
                "content_hash": content_hash,
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Public sync wrappers (for backward-compat with __init__.py's sync callers)
# ---------------------------------------------------------------------------

def ingest_confluence(
    space_keys: list[str] | None = None,
    reset: bool = False,
) -> dict:
    """Synchronous wrapper around ingest_confluence_async."""
    return asyncio.run(ingest_confluence_async(space_keys=space_keys, reset=reset))


def incremental_confluence_sync(hours: int = 1) -> dict:
    """Synchronous wrapper around incremental_confluence_sync_async."""
    return asyncio.run(incremental_confluence_sync_async(hours=hours))


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

async def ingest_confluence_async(
    space_keys: list[str] | None = None,
    cql: str | None = None,
    reset: bool = False,
    max_results: int = 0,
    progress: dict | None = None,
) -> dict:
    """
    Async Confluence ingestion: fetch → format → chunk → embed → store.

    Args:
        space_keys:  Space keys to ingest.  Defaults to CONFLUENCE_SPACES env var.
        cql:         Optional CQL query to override space_keys.
        reset:       Drop the existing Confluence collection before ingesting.
        max_results: Cap on pages to fetch per space (0 = no limit).
        progress:    Optional dict updated in-place with live progress info.

    Returns:
        Status dict with counts.
    """
    if progress is not None:
        progress.update({"stage": "fetching", "fetched": 0, "total": 0, "vectors": 0})

    def _on_page(fetched: int) -> None:
        if progress is not None:
            progress["fetched"] = fetched

    log.info(
        "Confluence ingestion started.  Spaces: %s  CQL: %s",
        space_keys or CONFLUENCE_SPACES,
        cql,
    )

    pages = await fetch_pages(
        space_keys=space_keys,
        cql=cql,
        max_results=max_results,
        on_progress=_on_page,
    )

    if not pages:
        log.warning("No Confluence pages fetched — check credentials and space keys.")
        return {
            "status":         "ok",
            "message":        "No Confluence pages fetched. Check credentials / space keys.",
            "pages_fetched":  0,
            "chunks_created": 0,
            "vectors_stored": 0,
        }

    documents = pages_to_documents(pages)
    chunks    = chunk_documents(documents, chunk_size=800, chunk_overlap=150)

    if progress is not None:
        progress.update({
            "stage":        "embedding",
            "fetched":      len(pages),
            "total":        len(pages),
            "chunks_total": len(chunks),
            "chunks_done":  0,
        })

    def _on_embed_progress(done: int, total: int) -> None:
        if progress is not None:
            progress["chunks_done"]  = done
            progress["chunks_total"] = total

    store = get_store()
    loop  = asyncio.get_running_loop()
    count = await loop.run_in_executor(
        None,
        functools.partial(
            store.add_confluence_batch, chunks, reset=reset, on_progress=_on_embed_progress
        ),
    )

    if progress is not None:
        progress.update({"stage": "done", "vectors": count})

    log.info(
        "Confluence ingestion complete: %d pages → %d chunks → %d vectors.",
        len(pages), len(chunks), count,
    )
    return {
        "status":         "ok",
        "spaces":         space_keys or CONFLUENCE_SPACES,
        "pages_fetched":  len(pages),
        "chunks_created": len(chunks),
        "vectors_stored": count,
    }


# ---------------------------------------------------------------------------
# Incremental sync (pages updated in the last N hours)
# ---------------------------------------------------------------------------

async def incremental_confluence_sync_async(
    hours: int = 1,
    space_keys: list[str] | None = None,
) -> dict:
    """
    Incremental Confluence sync — fetches pages updated in the last *hours* hours.

    Uses a CQL query:  lastModified >= "-Xh"  AND  space in (KEY1, KEY2, …)

    Args:
        hours:      How many hours back to look for updates.
        space_keys: Spaces to sync. Defaults to CONFLUENCE_SPACES.

    Returns:
        Status dict with counts.
    """
    spaces = space_keys or CONFLUENCE_SPACES
    space_clause = ""
    if spaces:
        quoted = ", ".join(f'"{k}"' for k in spaces)
        space_clause = f" AND space IN ({quoted})"

    cql = f'lastModified >= "-{hours}h" AND type = page{space_clause} ORDER BY lastModified ASC'

    log.info("Incremental Confluence sync (last %d h).  CQL: %s", hours, cql)
    pages = await fetch_pages(cql=cql)

    if not pages:
        return {
            "status":        "ok",
            "message":       f"No pages updated in the last {hours} hour(s).",
            "pages_fetched": 0,
            "vectors_stored": 0,
        }

    documents = pages_to_documents(pages)
    chunks    = chunk_documents(documents, chunk_size=800, chunk_overlap=150)

    store = get_store()
    loop  = asyncio.get_running_loop()
    count = await loop.run_in_executor(
        None,
        functools.partial(store.add_confluence_batch, chunks, reset=False),
    )

    log.info(
        "Incremental Confluence sync complete: %d pages → %d chunks → %d vectors.",
        len(pages), len(chunks), count,
    )
    return {
        "status":        "ok",
        "hours":         hours,
        "pages_fetched": len(pages),
        "vectors_stored": count,
    }


# ---------------------------------------------------------------------------
# Smart update (only re-index pages that changed)
# ---------------------------------------------------------------------------

async def smart_confluence_update_async(
    space_keys: list[str] | None = None,
    progress: dict | None = None,
) -> dict:
    """
    Smart incremental Confluence update.

    1. Fetches ALL pages from the configured spaces (with version + body).
    2. Compares content hashes against what is already stored in ChromaDB.
    3. Deletes stale chunks and re-indexes only changed or new pages.

    This is the most efficient update strategy when the space is large and
    only a minority of pages change between runs.
    """
    if progress is not None:
        progress.update({"stage": "fetching", "fetched": 0, "total": 0, "vectors": 0})

    def _on_page(fetched: int) -> None:
        if progress is not None:
            progress["fetched"] = fetched

    pages = await fetch_pages(space_keys=space_keys, on_progress=_on_page)

    if not pages:
        return {
            "status":  "ok",
            "message": "No Confluence pages found.",
            "updated": 0, "skipped": 0, "vectors_stored": 0,
        }

    # Load existing page state from ChromaDB
    store          = get_store()
    existing_state = store.get_existing_confluence_state()  # {page_id: {"updated": ts, "hash": h}}

    changed: list[dict] = []
    skipped = 0

    for page in pages:
        pid   = page.get("id", "")
        body  = page.get("body", "")
        uhash = hashlib.sha256(body.encode()).hexdigest() if body else ""
        prior = existing_state.get(pid)

        if prior and prior.get("hash") and prior["hash"] == uhash:
            skipped += 1
            continue

        changed.append(page)

    log.info(
        "Confluence smart update: %d changed, %d skipped (unchanged).",
        len(changed), skipped,
    )

    if not changed:
        return {
            "status":  "ok",
            "message": "All Confluence pages are already up to date.",
            "updated": 0, "skipped": skipped, "vectors_stored": 0,
        }

    # Delete old chunks for changed pages, then re-index
    changed_ids = [p["id"] for p in changed if p.get("id")]
    store.delete_confluence_pages(changed_ids)

    documents = pages_to_documents(changed)
    chunks    = chunk_documents(documents, chunk_size=800, chunk_overlap=150)

    if progress is not None:
        progress.update({
            "stage": "embedding", "total": len(changed),
            "chunks_total": len(chunks), "chunks_done": 0,
        })

    def _on_embed(done: int, total: int) -> None:
        if progress is not None:
            progress["chunks_done"]  = done
            progress["chunks_total"] = total

    loop  = asyncio.get_running_loop()
    count = await loop.run_in_executor(
        None,
        functools.partial(store.add_confluence_batch, chunks, reset=False, on_progress=_on_embed),
    )

    if progress is not None:
        progress.update({"stage": "done", "vectors": count})

    log.info(
        "Confluence smart update complete: %d pages updated → %d vectors.",
        len(changed), count,
    )
    return {
        "status":         "ok",
        "updated":        len(changed),
        "skipped":        skipped,
        "chunks_created": len(chunks),
        "vectors_stored": count,
    }
