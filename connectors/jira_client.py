"""
Jira REST API v3 client — async, paginated.

Fetches issues with full field detail: key, summary, description,
comments, labels, components, status, type, priority, created, updated.

Environment variables required:
    JIRA_URL        - e.g. https://yourcompany.atlassian.net
    JIRA_EMAIL      - your Atlassian account email
    JIRA_API_TOKEN  - Atlassian API token (not your password)
    JIRA_PROJECT_KEY - e.g. ARCESB
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

JIRA_URL         = os.getenv("JIRA_URL", "")
JIRA_EMAIL       = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN   = os.getenv("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "ARCESB")
JIRA_PAGE_SIZE   = int(os.getenv("JIRA_PAGE_SIZE", "100"))


# ---------------------------------------------------------------------------
# Atlassian Document Format (ADF) → plain text
# ---------------------------------------------------------------------------

def _extract_text(adf_or_str: Any) -> str:
    """
    Recursively extract plain text from an ADF node, plain string, or list.
    Jira API v3 returns description/comments as ADF JSON objects.
    """
    if adf_or_str is None:
        return ""
    if isinstance(adf_or_str, str):
        return adf_or_str
    if isinstance(adf_or_str, list):
        return " ".join(_extract_text(item) for item in adf_or_str if item)
    if isinstance(adf_or_str, dict):
        node_type = adf_or_str.get("type", "")
        if node_type == "text":
            return adf_or_str.get("text", "")
        parts = [_extract_text(child) for child in adf_or_str.get("content", [])]
        parts = [p for p in parts if p.strip()]
        block_types = {
            "paragraph", "heading", "bulletList", "orderedList",
            "listItem", "blockquote", "codeBlock", "rule",
        }
        sep = "\n" if node_type in block_types else " "
        return sep.join(parts)
    return str(adf_or_str)


# ---------------------------------------------------------------------------
# Issue formatter
# ---------------------------------------------------------------------------

def _format_issue(issue: dict) -> dict:
    """Convert a raw Jira API issue dict to a clean, normalized dict."""
    fields = issue.get("fields", {})

    # Description (ADF in API v3)
    description = _extract_text(fields.get("description")).strip()

    # Comments — cap at 20 per ticket to avoid huge chunks
    comment_list: list[str] = []
    for c in (fields.get("comment") or {}).get("comments", [])[:20]:
        author = (c.get("author") or {}).get("displayName", "Unknown")
        body = _extract_text(c.get("body")).strip()
        if body:
            comment_list.append(f"[{author}]: {body}")

    components = [comp.get("name", "") for comp in fields.get("components") or []]
    labels = fields.get("labels") or []

    resolution_field = fields.get("resolution")
    resolution = resolution_field.get("name", "") if isinstance(resolution_field, dict) else ""

    return {
        "key":        issue.get("key", ""),
        "summary":    fields.get("summary", ""),
        "description": description,
        "comments":   "\n".join(comment_list),
        "labels":     labels,
        "components": components,
        "status":     (fields.get("status") or {}).get("name", ""),
        "issue_type": (fields.get("issuetype") or {}).get("name", ""),
        "priority":   (fields.get("priority") or {}).get("name", ""),
        "resolution": resolution,
        "created":    fields.get("created", ""),
        "updated":    fields.get("updated", ""),
    }


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------

_DEFAULT_FIELDS = [
    "key", "summary", "description", "comment",
    "labels", "components", "status", "issuetype",
    "priority", "created", "updated", "resolution",
]


async def fetch_issues(
    jql: str | None = None,
    max_results: int = 0,
    fields: list[str] | None = None,
) -> list[dict]:
    """
    Fetch Jira issues asynchronously with full pagination.

    Args:
        jql:        JQL query string. Defaults to all issues in JIRA_PROJECT_KEY.
        max_results: Maximum issues to fetch. 0 = no limit (fetch all).
        fields:     Fields to request. Defaults to _DEFAULT_FIELDS.

    Returns:
        List of normalized issue dicts.
    """
    if not JIRA_URL or not JIRA_EMAIL or not JIRA_API_TOKEN:
        log.warning(
            "Jira credentials not configured (JIRA_URL / JIRA_EMAIL / JIRA_API_TOKEN). "
            "Skipping Jira fetch."
        )
        return []

    if jql is None:
        jql = f"project = {JIRA_PROJECT_KEY} ORDER BY updated DESC"
    if fields is None:
        fields = _DEFAULT_FIELDS

    issues: list[dict] = []
    start_at = 0

    async with httpx.AsyncClient(
        base_url=JIRA_URL.rstrip("/"),
        auth=httpx.BasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
        headers={"Accept": "application/json"},
        timeout=30.0,
    ) as client:
        while True:
            params: dict[str, Any] = {
                "jql":        jql,
                "startAt":    start_at,
                "maxResults": JIRA_PAGE_SIZE,
                "fields":     ",".join(fields),
            }

            try:
                resp = await client.get("/rest/api/3/search", params=params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                log.error(
                    "Jira API HTTP %d: %s",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                break
            except httpx.RequestError as exc:
                log.error("Jira connection error: %s", exc)
                break

            data = resp.json()
            batch: list[dict] = data.get("issues", [])
            total: int = data.get("total", 0)

            for raw in batch:
                issues.append(_format_issue(raw))

            fetched = start_at + len(batch)
            log.info("Jira: %d / %d issues fetched.", fetched, total)

            if not batch or fetched >= total:
                break
            if max_results and fetched >= max_results:
                break

            start_at += len(batch)

    log.info("Jira fetch complete: %d issues.", len(issues))
    return issues


def fetch_issues_sync(
    jql: str | None = None,
    max_results: int = 0,
) -> list[dict]:
    """Synchronous wrapper for use in non-async contexts (e.g. CLI)."""
    return asyncio.run(fetch_issues(jql=jql, max_results=max_results))
