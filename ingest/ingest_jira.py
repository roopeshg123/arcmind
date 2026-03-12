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

    Format:
        Ticket: ARCESB-XXXXX
        Type: Bug
        Status: Resolved
        ...

        Summary:
        <summary>

        Description:
        <description>

        Comments:
        <comments>
    """
    lines: list[str] = [
        f"Ticket: {issue['key']}",
        f"Type: {issue['issue_type']}",
        f"Status: {issue['status']}",
        f"Priority: {issue['priority']}",
    ]

    if issue.get("resolution"):
        lines.append(f"Resolution: {issue['resolution']}")
    if issue.get("components"):
        lines.append(f"Components: {', '.join(issue['components'])}")
    if issue.get("labels"):
        lines.append(f"Labels: {', '.join(issue['labels'])}")

    lines.extend(["", "Summary:", issue["summary"]])

    if issue.get("description"):
        lines.extend(["", "Description:", issue["description"][:2000]])

    if issue.get("comments"):
        lines.extend(["", "Comments:", issue["comments"][:3000]])

    return "\n".join(lines)


def issues_to_documents(issues: list[dict]) -> list[Document]:
    """Convert a list of Jira issue dicts to LangChain Documents."""
    docs: list[Document] = []
    for issue in issues:
        text = _format_ticket_text(issue)
        connector = _detect_connector(
            text=f"{issue['summary']} {issue.get('description', '')}",
            components=issue.get("components", []),
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source":    "jira",
                "ticket":    issue["key"],
                "type":      issue["issue_type"].lower(),
                "status":    issue["status"].lower(),
                "component": connector,
                "created":   issue.get("created", ""),
                "updated":   issue.get("updated", ""),
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def ingest_jira_async(
    jql: str | None = None,
    reset: bool = False,
    max_results: int = 0,
) -> dict:
    """
    Async Jira ingestion: fetch → format → chunk → embed → store.

    Args:
        jql:         JQL filter.  Defaults to all issues in JIRA_PROJECT_KEY.
        reset:       Drop the existing Jira collection before ingesting.
        max_results: Cap on how many issues to fetch (0 = no limit).

    Returns:
        Status dict with counts.
    """
    if jql is None:
        jql = f"project = {JIRA_PROJECT_KEY} ORDER BY updated DESC"

    log.info("Jira ingestion started.  JQL: %s", jql)
    issues = await fetch_issues(jql=jql, max_results=max_results)

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
    # Tickets are naturally self-contained; use slightly larger chunks to keep
    # summary + description together.
    chunks = chunk_documents(documents, chunk_size=600, chunk_overlap=100)

    store = get_store()
    count = store.add_jira_batch(chunks, reset=reset)

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
    """Synchronous wrapper for ingest_jira_async."""
    return asyncio.run(ingest_jira_async(jql=jql, reset=reset, max_results=max_results))


def incremental_jira_sync(hours: int = 1) -> dict:
    """
    Sync Jira issues that were updated in the last *hours* hours.

    Uses append mode (reset=False) so existing vectors are not deleted.
    """
    jql = (
        f"project = {JIRA_PROJECT_KEY} "
        f"AND updated >= -{hours}h "
        "ORDER BY updated DESC"
    )
    log.info("Incremental Jira sync — last %dh.  JQL: %s", hours, jql)
    return ingest_jira(jql=jql, reset=False)
