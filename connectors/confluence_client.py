"""
Confluence REST API client — async, paginated.

Fetches pages from one or more Confluence spaces with full body content,
version metadata, and ancestor breadcrumbs.

Uses the Confluence REST API v1 (classic) which is supported by both
Confluence Cloud and Confluence Data Center.

Environment variables required:
    CONFLUENCE_URL        - e.g. https://yourcompany.atlassian.net
    CONFLUENCE_EMAIL      - your Atlassian account email
    CONFLUENCE_API_TOKEN  - Atlassian API token (not your password)
    CONFLUENCE_SPACES     - comma-separated space keys, e.g. "DOCS,SUPPORT,KB"

Optional:
    CONFLUENCE_PAGE_SIZE  - results per API page (default: 100)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

CONFLUENCE_URL       = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_EMAIL     = os.getenv("CONFLUENCE_EMAIL", "")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACES    = [
    s.strip() for s in os.getenv("CONFLUENCE_SPACES", "").split(",") if s.strip()
]
CONFLUENCE_PAGE_SIZE = int(os.getenv("CONFLUENCE_PAGE_SIZE", "100"))


# ---------------------------------------------------------------------------
# Confluence Storage Format (HTML/XML) → plain text
# ---------------------------------------------------------------------------

class _StorageTextExtractor(HTMLParser):
    """
    Strip Confluence storage-format HTML/XML to plain text.

    Converts block-level elements to newlines so the resulting text has
    meaningful paragraph / line structure instead of one merged blob.
    """

    _BLOCK_TAGS = {
        "p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "td", "th", "blockquote", "pre", "hr",
        "ac:task", "ac:task-body",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags: set[str] = {
            "ac:structured-macro",    # code macro, images, etc.
            "ac:parameter",
            "ac:image", "ri:attachment",
        }
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._skip_tags or self._skip_depth > 0:
            self._skip_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of blank lines / whitespace
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _storage_to_text(storage_html: str) -> str:
    """Convert Confluence storage-format HTML to clean plain text."""
    if not storage_html:
        return ""
    parser = _StorageTextExtractor()
    try:
        parser.feed(storage_html)
        return parser.get_text()
    except Exception:
        # Fallback: naive tag strip
        return re.sub(r"<[^>]+>", " ", storage_html).strip()


# ---------------------------------------------------------------------------
# Page formatter
# ---------------------------------------------------------------------------

def _format_page(raw: dict) -> dict:
    """Convert a raw Confluence API page dict to a clean, normalized dict."""
    space      = raw.get("space") or {}
    version    = raw.get("version") or {}
    ancestors  = raw.get("ancestors") or []

    body_obj = raw.get("body") or {}
    storage  = body_obj.get("storage") or {}
    body_html = storage.get("value", "")

    body_text = _storage_to_text(body_html)

    # Build breadcrumb from ancestors
    breadcrumb = " > ".join(a.get("title", "") for a in ancestors if a.get("title"))

    # Full URL to the page (web UI link).
    # Confluence Cloud can return _links.base as either:
    #   https://org.atlassian.net        (no /wiki suffix)
    #   https://org.atlassian.net/wiki   (with /wiki suffix)
    # _links.webui is always a relative path like /spaces/KEY/pages/ID/Title
    # The correct browser URL is always: <domain>/wiki<webui>
    links    = raw.get("_links") or {}
    base_url = links.get("base", CONFLUENCE_URL.rstrip("/"))
    # Normalise: strip /wiki from base so we can always add it once ourselves
    if base_url.endswith("/wiki"):
        base_url = base_url[:-5]
    web_ui = links.get("webui", "")
    if web_ui and not web_ui.startswith("/wiki"):
        web_ui = "/wiki" + web_ui
    url = f"{base_url}{web_ui}" if web_ui else ""

    return {
        "id":          raw.get("id", ""),
        "title":       raw.get("title", ""),
        "space_key":   space.get("key", ""),
        "space_name":  space.get("name", ""),
        "body":        body_text,
        "url":         url,
        "version":     version.get("number", 1),
        "updated":     version.get("when", ""),
        "breadcrumb":  breadcrumb,
        "ancestors":   [a.get("title", "") for a in ancestors],
        "labels":      [
            lbl.get("name", "")
            for lbl in (raw.get("metadata") or {}).get("labels", {}).get("results", [])
        ],
    }


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------

_DEFAULT_EXPAND = (
    "body.storage,version,space,ancestors,metadata.labels"
)


async def fetch_pages(
    space_keys: list[str] | None = None,
    cql: str | None = None,
    max_results: int = 0,
    on_progress: Any = None,
) -> list[dict]:
    """
    Fetch Confluence pages asynchronously with full pagination.

    Args:
        space_keys:  List of space keys to fetch from. Defaults to
                     CONFLUENCE_SPACES env var.
        cql:         Optional CQL query string (overrides space_keys when set).
        max_results: Maximum pages to fetch per space. 0 = no limit.
        on_progress: Optional callable(fetched: int) called after each page.

    Returns:
        List of normalized page dicts.
    """
    if not CONFLUENCE_URL or not CONFLUENCE_EMAIL or not CONFLUENCE_API_TOKEN:
        log.warning(
            "Confluence credentials not configured "
            "(CONFLUENCE_URL / CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN). "
            "Skipping Confluence fetch."
        )
        return []

    effective_spaces = space_keys or CONFLUENCE_SPACES
    if not effective_spaces and not cql:
        log.warning("No CONFLUENCE_SPACES configured and no CQL provided. Skipping.")
        return []

    pages: list[dict] = []

    async with httpx.AsyncClient(
        base_url=CONFLUENCE_URL.rstrip("/"),
        auth=httpx.BasicAuth(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN),
        headers={"Accept": "application/json"},
        timeout=60.0,
    ) as client:
        if cql:
            pages += await _fetch_by_cql(client, cql, max_results, on_progress)
        else:
            for space_key in effective_spaces:
                space_pages = await _fetch_space(
                    client, space_key, max_results, on_progress
                )
                pages += space_pages

    total = len(pages)
    log.info("Confluence fetch complete: %d pages retrieved.", total)
    return pages


async def _fetch_space(
    client: httpx.AsyncClient,
    space_key: str,
    max_results: int,
    on_progress: Any,
) -> list[dict]:
    """Fetch all pages from a single Confluence space."""
    pages: list[dict] = []
    start = 0

    while True:
        params: dict[str, Any] = {
            "spaceKey": space_key,
            "type":     "page",
            "expand":   _DEFAULT_EXPAND,
            "limit":    CONFLUENCE_PAGE_SIZE,
            "start":    start,
        }

        try:
            resp = await client.get("/wiki/rest/api/content", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error(
                "Confluence API HTTP %d for space '%s': %s",
                exc.response.status_code,
                space_key,
                exc.response.text[:500],
            )
            break
        except httpx.RequestError as exc:
            log.error("Confluence connection error (space '%s'): %s", space_key, exc)
            break

        data    = resp.json()
        results = data.get("results", [])

        for raw in results:
            if max_results > 0 and len(pages) >= max_results:
                break
            pages.append(_format_page(raw))

        if on_progress is not None:
            try:
                on_progress(len(pages))
            except Exception:
                pass

        log.debug("Confluence space '%s': fetched %d pages so far.", space_key, len(pages))

        # Pagination — stop when no more pages or max reached
        if len(results) < CONFLUENCE_PAGE_SIZE:
            break
        if max_results > 0 and len(pages) >= max_results:
            break

        start += CONFLUENCE_PAGE_SIZE

    return pages


async def _fetch_by_cql(
    client: httpx.AsyncClient,
    cql: str,
    max_results: int,
    on_progress: Any,
) -> list[dict]:
    """Fetch pages matching a CQL query (used for incremental sync)."""
    pages: list[dict] = []
    start = 0

    while True:
        params: dict[str, Any] = {
            "cql":    cql,
            "expand": _DEFAULT_EXPAND,
            "limit":  CONFLUENCE_PAGE_SIZE,
            "start":  start,
        }

        try:
            resp = await client.get("/wiki/rest/api/content/search", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error(
                "Confluence CQL search HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            break
        except httpx.RequestError as exc:
            log.error("Confluence connection error (CQL): %s", exc)
            break

        data    = resp.json()
        results = data.get("results", [])

        for raw in results:
            if max_results > 0 and len(pages) >= max_results:
                break
            pages.append(_format_page(raw))

        if on_progress is not None:
            try:
                on_progress(len(pages))
            except Exception:
                pass

        if len(results) < CONFLUENCE_PAGE_SIZE:
            break
        if max_results > 0 and len(pages) >= max_results:
            break

        start += CONFLUENCE_PAGE_SIZE

    return pages
