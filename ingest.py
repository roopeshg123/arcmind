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
from pathlib import Path

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
        import shutil, gc, time
        gc.collect()  # release any lingering file handles (important on Windows)
        for attempt in range(5):
            try:
                shutil.rmtree(CHROMA_DB_DIR)
                log.info("Deleted existing ChromaDB at '%s'.", CHROMA_DB_DIR)
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

def run_ingestion(docs_dir: str = DOCS_DIR, reset: bool = True) -> dict:
    """
    Full ingestion pipeline: load → split → embed → store.

    Returns a status dict: {"status", "files_loaded", "chunks_created", "vectors_stored"}
    """
    docs = load_documents(docs_dir)
    if not docs:
        return {
            "status": "error",
            "message": f"No HTML files found in '{docs_dir}'.",
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
