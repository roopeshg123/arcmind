"""
Documentation ingestion pipeline for ArcMind.

Loads Arc HTML documentation from disk or crawls a live URL, parses text,
detects document sections (AS2, SFTP, …), chunks with token-awareness, and
stores chunks to the 'arcmind_docs' ChromaDB collection.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document

from ingest.chunking import chunk_documents
from vector_db.chroma_store import get_store

load_dotenv()

log = logging.getLogger(__name__)

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
DOCS_URL = os.getenv("DOCS_URL", "")
MAX_CRAWL_PAGES = int(os.getenv("MAX_CRAWL_PAGES", "2000"))

# Tags whose entire subtrees are discarded (navigation, scripts, ads…)
_NOISE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "noscript", "iframe", "svg",
]

# Asset file extensions to skip during web crawling
_SKIP_EXTENSIONS = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif",
    ".svg", ".ico", ".pdf", ".zip", ".woff", ".woff2", ".ttf",
)

# Arc documentation section patterns (longest/most specific first)
_SECTION_PATTERNS: dict[str, str] = {
    "ArcScript":  r"\bArcScript\b",
    "EDIFACT":    r"\bEDIFACT\b",
    "Peppol":     r"\bPEPPOL\b|\bPeppol\b",
    "OFTP":       r"\bOFTP\b|\bOdette\b",
    "AS2":        r"\bAS2\b",
    "SFTP":       r"\bSFTP\b",
    "X12":        r"\bX\.?12\b",
    "REST":       r"\bREST\b",
    "FTP":        r"\bFTP\b",
    "SMTP":       r"\bSMTP\b|\bPOP3\b|\bIMAP\b",
    "HTTP":       r"\bHTTP\b",
    "Flows":      r"\bflow\b",
    "Profiles":   r"\bprofile\b",
    "Connectors": r"\bconnector\b",
}


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _detect_section(text: str, source: str) -> str:
    """Return the primary Arc section detected in *text* + *source* path."""
    combined = f"{source} {text[:1500]}"
    for section, pattern in _SECTION_PATTERNS.items():
        if re.search(pattern, combined, re.IGNORECASE):
            return section
    return "General"


def _parse_html_raw(html: str, source: str = "") -> tuple[str, dict]:
    """
    Parse raw HTML → (clean_plain_text, metadata_dict).

    Returns:
        text:  Whitespace-normalised plain text.
        meta:  {"title": str, "section": str}
    """
    soup = BeautifulSoup(html, "lxml")

    # Best-effort page title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    section = _detect_section(text, source)
    return text, {"title": title, "section": section}


def parse_html_file(filepath: str) -> tuple[str, dict]:
    """Parse an HTML file on disk → (clean_text, metadata)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        return _parse_html_raw(fh.read(), source=filepath)


# ---------------------------------------------------------------------------
# Web crawler
# ---------------------------------------------------------------------------

def crawl_site(base_url: str) -> list[Document]:
    """
    Recursively crawl all HTML pages reachable from *base_url* that stay
    within the same URL prefix.  Returns LangChain Documents.
    """
    if not base_url.endswith("/"):
        base_url += "/"

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.5",
    })

    visited: set[str] = set()
    to_visit: deque[str] = deque([base_url])
    documents: list[Document] = []

    log.info("Starting web crawl from '%s'.", base_url)

    while to_visit:
        if len(visited) >= MAX_CRAWL_PAGES:
            log.warning(
                "Crawl limit reached (%d pages). "
                "Increase MAX_CRAWL_PAGES env var to crawl more.",
                MAX_CRAWL_PAGES,
            )
            break
        url = to_visit.popleft().split("#")[0]
        if not url or url in visited or not url.startswith(base_url):
            continue
        if any(url.lower().endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue

        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                log.warning("  HTTP %d — skipping %s", resp.status_code, url)
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue

            text, meta = _parse_html_raw(resp.text, source=url)
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source":  "documentation",
                        "url":     url,
                        "title":   meta["title"],
                        "section": meta["section"],
                    },
                ))
                log.info("  Crawled (%d chars): %s", len(text), url)

            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"]).split("#")[0]
                if href not in visited and href.startswith(base_url):
                    to_visit.append(href)

            time.sleep(0.05)  # polite crawl delay

        except Exception as exc:
            log.error("  Failed to crawl '%s': %s", url, exc)

    log.info("Crawl complete — %d pages.", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Disk loader
# ---------------------------------------------------------------------------

def load_documents_from_dir(docs_dir: str) -> list[Document]:
    """Load all .html files from *docs_dir* (recursive) as Documents."""
    pattern = os.path.join(docs_dir, "**", "*.html")
    all_files = glob.glob(pattern, recursive=True)

    # Drop files inside 'assets' folders (images, demo files, etc.)
    html_files = [
        f for f in all_files
        if not any(part.lower() == "assets" for part in Path(f).parts)
    ]

    skipped = len(all_files) - len(html_files)
    if skipped:
        log.info("Skipped %d asset file(s).", skipped)
    if not html_files:
        log.warning("No .html files found in '%s'.", docs_dir)
        return []

    log.info("Loading %d HTML file(s) from '%s'.", len(html_files), docs_dir)
    documents: list[Document] = []

    for filepath in html_files:
        try:
            text, meta = parse_html_file(filepath)
            if not text:
                continue
            rel_path = os.path.relpath(filepath, docs_dir)
            documents.append(Document(
                page_content=text,
                metadata={
                    "source":    "documentation",
                    "file_path": rel_path,
                    "title":     meta["title"],
                    "section":   meta["section"],
                },
            ))
        except Exception as exc:
            log.error("  Failed to parse '%s': %s", filepath, exc)

    log.info("Loaded %d document(s).", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def ingest_docs(docs_dir: str | None = None, reset: bool = True, progress: dict | None = None) -> dict:
    """
    Full documentation ingestion pipeline.

    Steps: load HTML → parse → chunk → embed → store to ChromaDB docs collection.

    Args:
        docs_dir: Override for DOCS_DIR env var.
        reset:    If True, drop the existing docs collection before ingesting.
        progress: Optional dict updated in-place with live progress info.

    Returns:
        Status dict with counts.
    """
    load_dotenv(override=True)
    docs_url       = os.getenv("DOCS_URL", "").strip()
    effective_dir  = docs_dir or os.getenv("DOCS_DIR", DOCS_DIR)

    if docs_url:
        log.info("DOCS_URL set — crawling: %s", docs_url)
        documents   = crawl_site(docs_url)
        source_label = docs_url
    else:
        documents   = load_documents_from_dir(effective_dir)
        source_label = effective_dir

    if not documents:
        return {
            "status":        "error",
            "message":       f"No documents found in '{source_label}'.",
            "files_loaded":  0,
            "chunks_created": 0,
            "vectors_stored": 0,
        }

    chunks = chunk_documents(documents)

    if progress is not None:
        progress.update({
            "stage":        "embedding",
            "chunks_total": len(chunks),
            "chunks_done":  0,
        })

    def _on_embed_progress(done: int, total: int) -> None:
        if progress is not None:
            progress["chunks_done"]  = done
            progress["chunks_total"] = total

    store  = get_store()
    count  = store.add_docs_batch(chunks, reset=reset, on_progress=_on_embed_progress)

    return {
        "status":        "ok",
        "source":        source_label,
        "files_loaded":  len(documents),
        "chunks_created": len(chunks),
        "vectors_stored": count,
    }


def smart_docs_update(docs_dir: str | None = None, progress: dict | None = None) -> dict:
    """
    Smart incremental documentation update — Git-style diff.

    Steps
    -----
    1. Load all HTML documents from disk / DOCS_URL (same as full ingest).
    2. Compute a SHA-256 content hash for each loaded document.
    3. Compare against the hashes stored in ChromaDB (or computed on-the-fly
       from stored text for legacy chunks that pre-date this feature).
    4. New sources     → ingest.
       Changed sources → delete old chunks + ingest fresh.
       Removed sources → delete chunks (file deleted from disk / site).
       Unchanged       → skip entirely (no embedding cost).
    5. Persist the content_hash in chunk metadata so future runs are fast.

    Returns a status dict with detailed counts.
    """
    import hashlib

    load_dotenv(override=True)
    docs_url      = os.getenv("DOCS_URL", "").strip()
    effective_dir = docs_dir or os.getenv("DOCS_DIR", DOCS_DIR)

    if progress is not None:
        progress.update({"stage": "loading", "fetched": 0, "total": 0, "vectors": 0})

    if docs_url:
        log.info("DOCS_URL set — crawling for smart update: %s", docs_url)
        documents    = crawl_site(docs_url)
        source_label = docs_url
    else:
        documents    = load_documents_from_dir(effective_dir)
        source_label = effective_dir

    if not documents:
        return {
            "status":          "error",
            "message":         f"No documents found in '{source_label}'.",
            "new_files":       0,
            "updated_files":   0,
            "removed_files":   0,
            "unchanged_files": 0,
            "chunks_created":  0,
            "vectors_stored":  0,
        }

    # --- Compute hashes for each loaded document ---
    # source_id: file_path (disk) or url (web)
    def _source_id(doc: Document) -> str:
        return doc.metadata.get("file_path") or doc.metadata.get("url") or ""

    loaded_map: dict[str, Document] = {}
    loaded_hashes: dict[str, str]   = {}
    for doc in documents:
        sid = _source_id(doc)
        if not sid:
            continue
        loaded_map[sid]    = doc
        loaded_hashes[sid] = hashlib.sha256(doc.page_content.encode()).hexdigest()

    # --- Get existing state from ChromaDB ---
    store          = get_store()
    existing_index = store.get_existing_docs_index()   # {source_id: content_hash}

    # --- Diff ---
    new_ids:     list[str] = []
    changed_ids: list[str] = []
    removed_ids: list[str] = []

    for sid, new_hash in loaded_hashes.items():
        if sid not in existing_index:
            new_ids.append(sid)
        elif existing_index[sid] != new_hash:
            changed_ids.append(sid)
        # else: unchanged

    for sid in existing_index:
        if sid not in loaded_hashes:
            removed_ids.append(sid)

    to_ingest_ids = new_ids + changed_ids
    unchanged_count = len(documents) - len(new_ids) - len(changed_ids)

    log.info(
        "Smart docs diff: %d new, %d changed, %d removed, %d unchanged.",
        len(new_ids), len(changed_ids), len(removed_ids), unchanged_count,
    )

    vectors_stored = store.docs_count()  # default — will update if anything changes

    if not to_ingest_ids and not removed_ids:
        return {
            "status":          "ok",
            "message":         "Everything is already up to date.",
            "new_files":       0,
            "updated_files":   0,
            "removed_files":   0,
            "unchanged_files": unchanged_count,
            "chunks_created":  0,
            "vectors_stored":  vectors_stored,
        }

    # --- Delete changed + removed sources from ChromaDB ---
    ids_to_delete = changed_ids + removed_ids
    if ids_to_delete:
        log.info("Deleting %d doc source(s) from ChromaDB.", len(ids_to_delete))
        store.delete_docs_by_source_id(ids_to_delete)

    # --- Ingest new + changed documents, stamping content_hash in metadata ---
    chunks_created = 0
    if to_ingest_ids:
        to_ingest_docs = []
        for sid in to_ingest_ids:
            doc = loaded_map[sid]
            # Stamp the hash so future runs can compare cheaply
            doc.metadata["content_hash"] = loaded_hashes[sid]
            to_ingest_docs.append(doc)

        chunks = chunk_documents(to_ingest_docs)
        chunks_created = len(chunks)

        if progress is not None:
            progress.update({
                "stage":        "embedding",
                "chunks_total": chunks_created,
                "chunks_done":  0,
            })

        def _on_embed_progress(done: int, total: int) -> None:
            if progress is not None:
                progress["chunks_done"]  = done
                progress["chunks_total"] = total

        # reset=False — we already deleted the stale chunks above
        vectors_stored = store.add_docs_batch(chunks, reset=False, on_progress=_on_embed_progress)
    else:
        # Only removals — rebuild BM25 to purge deleted entries
        store._build_bm25("arcmind_docs", new_docs=None)
        vectors_stored = store.docs_count()

    if progress is not None:
        progress.update({"stage": "done", "vectors": vectors_stored})

    log.info(
        "Smart docs update complete: +%d new, ~%d updated, -%d removed → %d vectors.",
        len(new_ids), len(changed_ids), len(removed_ids), vectors_stored,
    )
    return {
        "status":          "ok",
        "source":          source_label,
        "new_files":       len(new_ids),
        "updated_files":   len(changed_ids),
        "removed_files":   len(removed_ids),
        "unchanged_files": unchanged_count,
        "chunks_created":  chunks_created,
        "vectors_stored":  vectors_stored,
    }
