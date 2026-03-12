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

    fix_versions = [
        v.get("name", "") for v in (fields.get("fixVersions") or [])
    ]

    # Sprint name (customfield_10014 is the standard Jira sprint field)
    sprint_raw = fields.get("customfield_10014")
    sprint = ""
    if isinstance(sprint_raw, list) and sprint_raw:
        sprint = sprint_raw[-1].get("name", "") if isinstance(sprint_raw[-1], dict) else ""
    elif isinstance(sprint_raw, str):
        sprint = sprint_raw

    return {
        "key":          issue.get("key", ""),
        "summary":      fields.get("summary", ""),
        "description":  description,
        "comments":     "\n".join(comment_list),
        "labels":       labels,
        "components":   components,
        "status":       (fields.get("status") or {}).get("name", ""),
        "issue_type":   (fields.get("issuetype") or {}).get("name", ""),
        "priority":     (fields.get("priority") or {}).get("name", ""),
        "resolution":   resolution,
        "fix_versions": fix_versions,
        "sprint":       sprint,
        "created":      fields.get("created", ""),
        "updated":      fields.get("updated", ""),
    }


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------

_DEFAULT_FIELDS = [
    "key", "summary", "description", "comment",
    "labels", "components", "status", "issuetype",
    "priority", "created", "updated", "resolution",
    "fixVersions", "customfield_10014",  # fix versions + sprint
]


async def fetch_issues(
    jql: str | None = None,
    max_results: int = 0,
    fields: list[str] | None = None,
    on_progress: Any = None,
) -> list[dict]:
    """
    Fetch Jira issues asynchronously with full pagination.

    Args:
        jql:         JQL query string. Defaults to all issues in JIRA_PROJECT_KEY.
        max_results: Maximum issues to fetch. 0 = no limit (fetch all).
        fields:      Fields to request. Defaults to _DEFAULT_FIELDS.
        on_progress: Optional callable(fetched: int) called after each page.

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
    next_page_token: str | None = None

    async with httpx.AsyncClient(
        base_url=JIRA_URL.rstrip("/"),
        auth=httpx.BasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
        headers={"Accept": "application/json"},
        timeout=30.0,
    ) as client:
        while True:
            params: dict[str, Any] = {
                "jql":        jql,
                "maxResults": JIRA_PAGE_SIZE,
                "fields":     ",".join(fields),
            }
            if next_page_token:
                params["nextPageToken"] = next_page_token

            try:
                resp = await client.get("/rest/api/3/search/jql", params=params)
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

            for raw in batch:
                issues.append(_format_issue(raw))

            log.info("Jira: %d issues fetched so far.", len(issues))
            if on_progress:
                on_progress(len(issues))

            # New API uses cursor-based pagination via nextPageToken
            next_page_token = data.get("nextPageToken")
            if not batch or not next_page_token:
                break
            if max_results and len(issues) >= max_results:
                break

    log.info("Jira fetch complete: %d issues.", len(issues))
    return issues


def fetch_issues_sync(
    jql: str | None = None,
    max_results: int = 0,
) -> list[dict]:
    """Synchronous wrapper for use in non-async contexts (e.g. CLI)."""
    return asyncio.run(fetch_issues(jql=jql, max_results=max_results))
