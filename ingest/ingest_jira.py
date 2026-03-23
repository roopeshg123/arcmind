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
import functools
import hashlib
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
        desc = _s(issue["description"])
        # Detect whether the description contains ArcScript or code blocks.
        # If it does, preserve the full text so scripts are never truncated.
        # Otherwise cap at 3000 chars to keep chunk size reasonable.
        _CODE_RE = re.compile(
            r'<arc:script|arc:set|arc:if|arc:for|arc:while|arc:try|arc:call'
            r'|```|<script|arcInput|arcOutput',
            re.IGNORECASE,
        )
        if _CODE_RE.search(desc):
            lines.extend(["", "Description:", desc])          # full — no truncation
        else:
            lines.extend(["", "Description:", desc[:3000]])

    # Linked Jira issues (blocks, is blocked by, relates to, sub-task of, etc.)
    linked = issue.get("linked_issues") or []
    if linked:
        lines.append("")
        lines.append("Linked Issues:")
        for li in linked:
            status_str = f" [{_s(li.get('status'))}]" if li.get("status") else ""
            lines.append(f"  - {_s(li.get('type'))}: {_s(li.get('key'))}{status_str} — {_s(li.get('summary'))}")

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
        # Content hash — covers ALL ticket fields (status, comments, description…).
        # Two tickets with identical text but different comments will produce
        # different hashes because comment_items are embedded in the hash seed.
        hash_seed = text + "".join(
            c["author"] + c["body"]
            for c in issue.get("comment_items", [])
        )
        ticket_hash = hashlib.sha256(hash_seed.encode()).hexdigest()

        # Detect whether this ticket contains ArcScript or Python script examples
        # so the script generator can specifically target these tickets.
        _SCRIPT_CONTENT_RE = re.compile(
            r'<arc:script|arc:set|arc:if|arc:for|arc:while|arc:try|arc:call'
            r'|arcInput|arcOutput|language=["\']python',
            re.IGNORECASE,
        )
        full_ticket_text = text + " ".join(
            c.get("body", "") for c in issue.get("comment_items", [])
        )
        has_script = "true" if _SCRIPT_CONTENT_RE.search(full_ticket_text) else "false"

        base_meta = {
            "source":      "jira",
            "ticket":      issue.get("key", ""),
            "summary":     issue.get("summary") or "",
            "type":        (issue.get("issue_type") or "").lower(),
            "status":      (issue.get("status") or "").lower(),
            "component":   connector,
            "created":     issue.get("created", ""),
            "updated":     issue.get("updated", ""),
            "ticket_hash": ticket_hash,
            "has_script":  has_script,
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
    loop  = asyncio.get_running_loop()
    count = await loop.run_in_executor(
        None,
        functools.partial(store.add_jira_batch, chunks, reset=reset, on_progress=_on_embed_progress),
    )

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


async def smart_jira_update_async(
    progress: dict | None = None,
) -> dict:
    """
    Smart incremental Jira update — Git-style diff.

    Steps
    -----
    1. Fetch ALL issues from Jira (same scope as full ingest).
    2. Compare each ticket's ``updated`` timestamp against what is stored in
       ChromaDB.
    3. New tickets      → ingest.
       Changed tickets  → delete old chunks + ingest fresh.
       Deleted tickets  → delete chunks from DB (ticket no longer exists in
                          Jira but is still in the DB).
       Unchanged tickets → skip entirely (no API cost, no embedding cost).
    4. Rebuild BM25 only if anything changed.

    Returns a status dict with detailed counts.
    """
    jql = f"project = {JIRA_PROJECT_KEY} ORDER BY updated DESC"

    if progress is not None:
        progress.update({"stage": "fetching", "fetched": 0, "total": 0, "vectors": 0})

    def _on_page(fetched: int) -> None:
        if progress is not None:
            progress["fetched"] = fetched

    log.info("Smart Jira update — fetching all issues.  JQL: %s", jql)
    issues = await fetch_issues(jql=jql, max_results=0, on_progress=_on_page)

    if not issues:
        log.warning("Smart Jira update: no issues fetched — check credentials / JQL.")
        return {
            "status":           "ok",
            "message":          "No Jira issues fetched. Check credentials / JQL.",
            "new_tickets":      0,
            "updated_tickets":  0,
            "removed_tickets":  0,
            "unchanged_tickets":0,
            "chunks_created":   0,
            "vectors_stored":   0,
        }

    if progress is not None:
        progress.update({"stage": "comparing", "total": len(issues)})

    # --- Build lookup: ticket_key → updated timestamp from Jira ---
    fetched_map: dict[str, dict] = {iss["key"]: iss for iss in issues}

    # --- Get existing state from ChromaDB ---
    store = get_store()
    loop  = asyncio.get_running_loop()
    existing_state: dict[str, dict] = await loop.run_in_executor(
        None, store.get_existing_jira_state
    )

    # Pre-compute content hashes for every fetched issue so we can diff
    # without re-formatting each ticket twice.
    def _issue_hash(issue: dict) -> str:
        text = _format_ticket_text(issue) or ""
        seed = text + "".join(
            c["author"] + c["body"]
            for c in issue.get("comment_items", [])
        )
        return hashlib.sha256(seed.encode()).hexdigest()

    # --- Diff ---
    new_keys:     list[str] = []
    changed_keys: list[str] = []
    removed_keys: list[str] = []

    for key, issue in fetched_map.items():
        if key not in existing_state:
            new_keys.append(key)
        else:
            stored = existing_state[key]
            stored_hash    = stored.get("hash", "")
            stored_updated = stored.get("updated", "")
            if stored_hash:
                # Reliable: compare full content hash (immune to tz format drift)
                if _issue_hash(issue) != stored_hash:
                    changed_keys.append(key)
            else:
                # Legacy fallback: chunks indexed before ticket_hash was added
                # — compare updated timestamp strings.
                if (issue.get("updated") or "") > stored_updated:
                    changed_keys.append(key)

    for key in existing_state:
        if key not in fetched_map:
            removed_keys.append(key)

    to_ingest_keys = new_keys + changed_keys
    log.info(
        "Smart Jira diff: %d new, %d changed, %d removed, %d unchanged.",
        len(new_keys), len(changed_keys), len(removed_keys),
        len(issues) - len(new_keys) - len(changed_keys),
    )

    vectors_stored = store.jira_count()  # default — will update if we change anything

    if not to_ingest_keys and not removed_keys:
        return {
            "status":            "ok",
            "message":           "Everything is already up to date.",
            "new_tickets":       0,
            "updated_tickets":   0,
            "removed_tickets":   0,
            "unchanged_tickets": len(issues),
            "chunks_created":    0,
            "vectors_stored":    vectors_stored,
        }

    # --- Delete changed + removed tickets from ChromaDB ---
    keys_to_delete = changed_keys + removed_keys
    if keys_to_delete:
        log.info("Deleting %d ticket(s) from ChromaDB.", len(keys_to_delete))
        await loop.run_in_executor(
            None,
            functools.partial(store.delete_jira_tickets, keys_to_delete),
        )

    # --- Ingest new + changed tickets ---
    chunks_created = 0
    if to_ingest_keys:
        to_ingest_issues = [fetched_map[k] for k in to_ingest_keys]
        documents  = issues_to_documents(to_ingest_issues)
        chunks     = chunk_documents(documents, chunk_size=600, chunk_overlap=100)
        chunks_created = len(chunks)

        if progress is not None:
            progress.update({
                "stage":        "embedding",
                "fetched":      len(issues),
                "total":        len(issues),
                "chunks_total": chunks_created,
                "chunks_done":  0,
            })

        def _on_embed_progress(done: int, total: int) -> None:
            if progress is not None:
                progress["chunks_done"]  = done
                progress["chunks_total"] = total

        vectors_stored = await loop.run_in_executor(
            None,
            # reset=False — we already deleted the stale chunks above
            functools.partial(store.add_jira_batch, chunks, False, _on_embed_progress),
        )
    else:
        # Only removals — rebuild BM25 to purge deleted entries
        await loop.run_in_executor(
            None,
            functools.partial(store._build_bm25, "arcmind_jira", None),
        )
        vectors_stored = store.jira_count()

    if progress is not None:
        progress.update({"stage": "done", "vectors": vectors_stored})

    log.info(
        "Smart Jira update complete: +%d new, ~%d updated, -%d removed → %d vectors.",
        len(new_keys), len(changed_keys), len(removed_keys), vectors_stored,
    )
    return {
        "status":            "ok",
        "new_tickets":       len(new_keys),
        "updated_tickets":   len(changed_keys),
        "removed_tickets":   len(removed_keys),
        "unchanged_tickets": len(issues) - len(new_keys) - len(changed_keys),
        "chunks_created":    chunks_created,
        "vectors_stored":    vectors_stored,
    }
