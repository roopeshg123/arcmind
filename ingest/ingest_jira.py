"""
Jira ingestion pipeline for ArcMind.

Fetches issues via the Jira REST API v3, converts them to structured
text documents with rich metadata, chunks them, and stores them to the
'arcmind_jira' ChromaDB collection.

Supports:
  - Full initial ingest of all project issues
  - Incremental sync (issues updated in the last N hours)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document

from connectors.jira_client import fetch_issues
from ingest.chunking import chunk_documents
from vector_db.chroma_store import get_store

load_dotenv()

log = logging.getLogger(__name__)

JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "ARCESB")

# Arc connector patterns for auto-tagging tickets
_CONNECTOR_PATTERNS: dict[str, str] = {
    "AS2":       r"\bAS2\b",
    "SFTP":      r"\bSFTP\b",
    "OFTP":      r"\bOFTP\b|\bOdette\b",
    "X12":       r"\bX\.?12\b",
    "EDIFACT":   r"\bEDIFACT\b",
    "Peppol":    r"\bPEPPOL\b|\bPeppol\b",
    "HTTP":      r"\bHTTP\b",
    "FTP":       r"\bFTP\b",
    "SMTP":      r"\bSMTP\b|\bPOP3\b|\bIMAP\b",
    "REST":      r"\bREST\b",
    "ArcScript": r"\bArcScript\b",
    "Flows":     r"\bflow\b",
}


# ---------------------------------------------------------------------------
# Ticket → structured text
# ---------------------------------------------------------------------------

def _detect_connector(text: str, components: list[str]) -> str:
    """Identify the primary Arc connector referenced in a ticket."""
    combined = " ".join(components) + " " + text[:600]
    for connector, pattern in _CONNECTOR_PATTERNS.items():
        if re.search(pattern, combined, re.IGNORECASE):
            return connector
    return "General"


def _format_ticket_text(issue: dict) -> str:
    """
    Render a Jira issue dict as human-readable structured text.
    All fields coerced to str so page_content is never None.
    """
    def _s(v) -> str:
        return str(v).strip() if v is not None else ""

    lines: list[str] = [
        f"Ticket: {_s(issue.get('key'))}",
        f"Summary: {_s(issue.get('summary'))}",
        f"Type: {_s(issue.get('issue_type'))}",
        f"Status: {_s(issue.get('status'))}",
        f"Priority: {_s(issue.get('priority'))}",
    ]

    if issue.get("resolution"):
        lines.append(f"Resolution: {_s(issue['resolution'])}")
    if issue.get("components"):
        lines.append(f"Components: {', '.join(_s(c) for c in issue['components'])}")
    if issue.get("labels"):
        lines.append(f"Labels: {', '.join(_s(l) for l in issue['labels'])}")
    if issue.get("fix_versions"):
        lines.append(f"Fix Versions: {', '.join(_s(v) for v in issue['fix_versions'])}")
    if issue.get("sprint"):
        lines.append(f"Sprint: {_s(issue['sprint'])}")
    if issue.get("created"):
        lines.append(f"Created: {_s(issue['created'])[:10]}")
    if issue.get("updated"):
        lines.append(f"Updated: {_s(issue['updated'])[:10]}")

    if issue.get("description"):
        lines.extend(["", "Description:", _s(issue["description"])[:3000]])

    # Comments are indexed as separate documents; omit them from the main body
    # to avoid diluting the description chunk's semantic focus.

    return "\n".join(lines)


def issues_to_documents(issues: list[dict]) -> list[Document]:
    """Convert a list of Jira issue dicts to LangChain Documents.

    Each ticket produces:
      1. One main document (metadata + description, no comments).
      2. One document per comment, prefixed with the ticket key + summary so
         comment text is independently searchable in full context.
    """
    docs: list[Document] = []
    for issue in issues:
        text = _format_ticket_text(issue) or ""
        if not text.strip():
            continue  # skip tickets that produced empty content
        connector = _detect_connector(
            text=f"{issue.get('summary', '')} {issue.get('description', '')}",
            components=issue.get("components", []),
        )
        base_meta = {
            "source":    "jira",
            "ticket":    issue.get("key", ""),
            "summary":   issue.get("summary") or "",
            "type":      (issue.get("issue_type") or "").lower(),
            "status":    (issue.get("status") or "").lower(),
            "component": connector,
            "created":   issue.get("created", ""),
            "updated":   issue.get("updated", ""),
        }
        # --- main ticket document (description only) ---
        docs.append(Document(page_content=text, metadata=base_meta))

        # --- one document per comment ---
        # Prefixing every comment with the ticket key + summary means the
        # comment chunk will rank highly for queries that mention the ticket
        # topic, even when the description chunk was not retrieved.
        ticket_header = (
            f"Ticket: {issue.get('key', '')}"
            f"\nSummary: {issue.get('summary', '')}"
            f"\nType: {(issue.get('issue_type') or '')}"
            f"\nStatus: {(issue.get('status') or '')}"
        )
        for comment in issue.get("comment_items", []):
            comment_text = (
                f"{ticket_header}"
                f"\n\nComment by {comment['author']}:"
                f"\n{comment['body']}"
            )
            docs.append(Document(
                page_content=comment_text,
                metadata={**base_meta, "type": "comment"},
            ))
    return docs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def ingest_jira_async(
    jql: str | None = None,
    reset: bool = False,
    max_results: int = 0,
    progress: dict | None = None,
) -> dict:
    """
    Async Jira ingestion: fetch → format → chunk → embed → store.

    Args:
        jql:         JQL filter.  Defaults to all issues in JIRA_PROJECT_KEY.
        reset:       Drop the existing Jira collection before ingesting.
        max_results: Cap on how many issues to fetch (0 = no limit).
        progress:    Optional dict updated in-place with live progress info.

    Returns:
        Status dict with counts.
    """
    if jql is None:
        jql = f"project = {JIRA_PROJECT_KEY} ORDER BY updated DESC"

    if progress is not None:
        progress.update({"stage": "fetching", "fetched": 0, "total": 0, "vectors": 0})

    def _on_page(fetched: int) -> None:
        if progress is not None:
            progress["fetched"] = fetched

    log.info("Jira ingestion started.  JQL: %s", jql)
    issues = await fetch_issues(jql=jql, max_results=max_results, on_progress=_on_page)

    if not issues:
        log.warning("No Jira issues fetched — check credentials and JQL.")
        return {
            "status":         "ok",
            "message":        "No Jira issues fetched. Check credentials / JQL.",
            "issues_fetched": 0,
            "chunks_created": 0,
            "vectors_stored": 0,
        }

    documents = issues_to_documents(issues)
    chunks    = chunk_documents(documents, chunk_size=600, chunk_overlap=100)

    if progress is not None:
        progress.update({
            "stage":        "embedding",
            "fetched":      len(issues),
            "total":        len(issues),
            "chunks_total": len(chunks),
            "chunks_done":  0,
        })

    def _on_embed_progress(done: int, total: int) -> None:
        if progress is not None:
            progress["chunks_done"]  = done
            progress["chunks_total"] = total

    store = get_store()
    count = store.add_jira_batch(chunks, reset=reset, on_progress=_on_embed_progress)

    if progress is not None:
        progress.update({"stage": "done", "vectors": count})

    log.info(
        "Jira ingestion complete: %d issues → %d chunks → %d vectors.",
        len(issues), len(chunks), count,
    )
    return {
        "status":         "ok",
        "jql":            jql,
        "issues_fetched": len(issues),
        "chunks_created": len(chunks),
        "vectors_stored": count,
    }


def ingest_jira(
    jql: str | None = None,
    reset: bool = False,
    max_results: int = 0,
) -> dict:
    """Synchronous wrapper for ingest_jira_async (CLI use only)."""
    return asyncio.run(ingest_jira_async(jql=jql, reset=reset, max_results=max_results))


async def incremental_jira_sync_async(hours: int = 1) -> dict:
    """
    Async incremental Jira sync — fetches issues updated in the last *hours* hours.
    Used by FastAPI endpoints (running event loop).
    """
    jql = (
        f"project = {JIRA_PROJECT_KEY} "
        f"AND updated >= -{hours}h "
        "ORDER BY updated DESC"
    )
    log.info("Incremental Jira sync — last %dh.  JQL: %s", hours, jql)
    return await ingest_jira_async(jql=jql, reset=False)


def incremental_jira_sync(hours: int = 1) -> dict:
    """
    Sync Jira issues that were updated in the last *hours* hours.
    Synchronous wrapper for CLI use only.
    """
    return asyncio.run(incremental_jira_sync_async(hours=hours))
