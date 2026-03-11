"""
Document Ingestion Pipeline

Walks the docs/ folder, parses every .html file with BeautifulSoup,
splits the text into overlapping chunks, embeds them with OpenAI, and
persists everything to a local ChromaDB vector store.

Usage (standalone):
    python ingest.py
    python ingest.py --docs-dir ./my_docs --reset
"""

import os
import argparse
import glob
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DOCS_DIR        = os.getenv("DOCS_DIR", "./docs")
DOCS_URL        = os.getenv("DOCS_URL", "")          # takes priority over DOCS_DIR when set
CHROMA_DB_DIR   = os.getenv("CHROMA_DB_DIR", "./chroma_db")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Tags whose content is discarded entirely (navigation, scripts, styles…)
_NOISE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "noscript", "iframe", "svg",
]


# ---------------------------------------------------------------------------
# HTML → clean text
# ---------------------------------------------------------------------------

def parse_html_file(filepath: str) -> str:
    """
    Parse an HTML file and return clean plain text.

    Strategy
    --------
    1. Remove all noise tags (script, style, nav, …).
    2. Extract text with BeautifulSoup using newline as separator.
    3. Collapse excessive blank lines.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        raw_html = fh.read()

    soup = BeautifulSoup(raw_html, "lxml")

    # Drop noisy elements in-place
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse runs of whitespace / blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_html_content(html: str) -> str:
    """
    Parse an HTML string and return clean plain text.
    Same cleaning logic as parse_html_file but operates on a string.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Web crawler — fetches all HTML pages under a base URL
# ---------------------------------------------------------------------------

_SKIP_EXTENSIONS = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif",
    ".svg", ".ico", ".pdf", ".zip", ".woff", ".woff2", ".ttf",
)


def crawl_site(base_url: str) -> list[Document]:
    """
    Recursively crawl all HTML pages reachable from base_url that stay
    within the same URL prefix.  Returns a list of LangChain Documents.
    """
    # Ensure base_url ends with / so prefix matching works correctly
    if not base_url.endswith("/"):
        base_url += "/"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })

    visited: set[str] = set()
    to_visit: list[str] = [base_url]
    documents: list[Document] = []

    log.info("Starting web crawl from '%s'.", base_url)

    while to_visit:
        url = to_visit.pop(0).split("#")[0]  # strip fragment
        if not url or url in visited:
            continue
        if not url.startswith(base_url):
            continue  # stay within the base prefix
        if any(url.lower().endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue  # skip non-HTML assets

        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                log.warning("  Skipping %s (HTTP %d)", url, resp.status_code)
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue

            text = parse_html_content(resp.text)
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": url},
                ))
                log.info("  Crawled: %s (%d chars)", url, len(text))

            # Discover new links on this page
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"]).split("#")[0]
                if href not in visited and href.startswith(base_url):
                    to_visit.append(href)

            time.sleep(0.05)  # polite crawl delay

        except Exception as exc:
            log.error("  Failed to crawl '%s': %s", url, exc)

    log.info("Web crawl complete — fetched %d page(s).", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Load all HTML documents from a directory tree
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str) -> list[Document]:
    """
    Recursively find every .html file under docs_dir and return a list of
    LangChain Document objects (raw full-file text, not yet chunked).
    """
    pattern = os.path.join(docs_dir, "**", "*.html")
    all_files = glob.glob(pattern, recursive=True)

    # Exclude the assets folder (images, fonts, demos — not documentation text)
    html_files = [
        f for f in all_files
        if not any(
            part.lower() == "assets"
            for part in Path(f).parts
        )
    ]

    skipped = len(all_files) - len(html_files)
    if skipped:
        log.info("Skipped %d file(s) inside 'assets' folder(s).", skipped)

    if not html_files:
        log.warning("No .html files found in '%s'.", docs_dir)
        return []

    log.info("Found %d HTML file(s) in '%s'.", len(html_files), docs_dir)

    documents = []
    for filepath in html_files:
        try:
            text = parse_html_file(filepath)
            if not text:
                log.warning("Skipping empty file: %s", filepath)
                continue

            # Store relative path as the source so it's human-readable in the UI
            rel_path = os.path.relpath(filepath, docs_dir)
            documents.append(Document(
                page_content=text,
                metadata={"source": rel_path, "full_path": filepath},
            ))
            log.info("  Loaded: %s (%d chars)", rel_path, len(text))
        except Exception as exc:
            log.error("  Failed to load '%s': %s", filepath, exc)

    return documents


# ---------------------------------------------------------------------------
# Split documents into chunks
# ---------------------------------------------------------------------------

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    log.info("Split %d document(s) into %d chunk(s).", len(documents), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Embed and persist to ChromaDB
# ---------------------------------------------------------------------------

def embed_and_store(chunks: list[Document], reset: bool = False) -> int:
    """
    Embed chunks and upsert them into ChromaDB.

    Args:
        chunks: List of chunked Document objects.
        reset:  If True, delete and recreate the collection first.

    Returns:
        Total number of documents in the collection after the operation.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    if reset and os.path.isdir(CHROMA_DB_DIR):
        import shutil, gc
        # Release the in-process ChromaDB client first (avoids WinError 32 on Windows)
        try:
            import rag_engine
            rag_engine.release_vector_store()
        except Exception:
            pass
        gc.collect()
        for attempt in range(5):
            try:
                # Delete only the CONTENTS, not the directory itself.
                # This is required when chroma_db is a Docker volume mount —
                # the OS forbids removing the mount-point directory (EBUSY/errno 16).
                for item in os.listdir(CHROMA_DB_DIR):
                    item_path = os.path.join(CHROMA_DB_DIR, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                log.info("Cleared existing ChromaDB contents at '%s'.", CHROMA_DB_DIR)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                log.warning("ChromaDB files still locked, retrying in 1 s… (attempt %d/5)", attempt + 1)
                time.sleep(1)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    count = vector_store._collection.count()
    log.info("ChromaDB now contains %d vector(s) at '%s'.", count, CHROMA_DB_DIR)
    return count


# ---------------------------------------------------------------------------
# Convenience function called by the FastAPI /ingest endpoint
# ---------------------------------------------------------------------------

def run_ingestion(docs_dir: str = None, reset: bool = True) -> dict:
    """
    Full ingestion pipeline: load → split → embed → store.

    Re-reads DOCS_URL and DOCS_DIR from environment on every call so that
    .env changes take effect without restarting the server.

    When DOCS_URL env var is set, pages are crawled from the web.
    Otherwise documents are loaded from docs_dir on disk.

    Returns a status dict: {"status", "files_loaded", "chunks_created", "vectors_stored"}
    """
    # Re-read at call time so .env changes are always picked up.
    # Use dotenv_values() to read ONLY what's explicitly set in the .env file
    # (ignores OS-level env vars), preventing stale values from leaking in.
    load_dotenv(override=True)
    from dotenv import dotenv_values
    env_file_values = dotenv_values()
    # env_file_values covers local Python runs (reads .env file directly).
    # os.getenv() covers Docker runs (container has no .env file — values come from env vars).
    docs_url = (env_file_values.get("DOCS_URL") or os.getenv("DOCS_URL", "")).strip()
    effective_docs_dir = docs_dir or env_file_values.get("DOCS_DIR") or os.getenv("DOCS_DIR", DOCS_DIR)

    if docs_url:
        log.info("DOCS_URL is set — crawling from web: %s", docs_url)
        docs = crawl_site(docs_url)
        source_label = docs_url
    else:
        docs = load_documents(effective_docs_dir)
        source_label = effective_docs_dir

    if not docs:
        return {
            "status": "error",
            "message": f"No documents found from '{source_label}'.",
            "files_loaded": 0,
            "chunks_created": 0,
            "vectors_stored": 0,
        }

    chunks = split_documents(docs)
    vectors_stored = embed_and_store(chunks, reset=reset)

    return {
        "status": "success",
        "files_loaded": len(docs),
        "chunks_created": len(chunks),
        "vectors_stored": vectors_stored,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HTML documentation into ChromaDB.")
    parser.add_argument(
        "--docs-dir", default=DOCS_DIR,
        help="Path to the folder containing .html documentation files.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate the vector store before ingesting.",
    )
    args = parser.parse_args()

    result = run_ingestion(docs_dir=args.docs_dir, reset=args.reset)
    print("\n=== Ingestion complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
