"""
Token-aware semantic chunking for ArcMind.

Uses tiktoken to measure chunk size in tokens (not characters) so chunks
align with the embedding model's context window.

Default settings:
    chunk_size    = 500 tokens
    chunk_overlap = 100 tokens
    encoding      = cl100k_base (compatible with text-embedding-3-large)
"""

from __future__ import annotations

import logging
import re
from typing import List

import tiktoken
from langchain_core.documents import Document

log = logging.getLogger(__name__)

CHUNK_SIZE     = 500   # tokens
CHUNK_OVERLAP  = 100   # tokens
ENCODING_NAME  = "cl100k_base"  # shared by GPT-4 and text-embedding-3-*


# ---------------------------------------------------------------------------
# Encoder (module-level singleton — loaded once)
# ---------------------------------------------------------------------------

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(ENCODING_NAME)
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def _split_text_by_tokens(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split *text* into overlapping windows measured in tokens.

    Strategy
    --------
    1.  Split on double-newlines (paragraph boundaries) to preserve semantic units.
    2.  Accumulate paragraphs until the token budget is reached.
    3.  When the budget overflows, flush the current chunk, then keep a
        *token-level* overlap tail from the flushed chunk.
    4.  For paragraphs that individually exceed *chunk_size*, fall back to
        sentence-level splitting.
    """
    enc = _get_encoder()

    # Normalise whitespace first
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = re.split(r"\n\n+", text)

    chunks: list[str] = []
    current_ids: list[int] = []  # token IDs for the current window

    def _flush(ids: list[int]) -> list[int]:
        """Emit the current token buffer as a chunk and return the overlap tail."""
        chunk_text = enc.decode(ids).strip()
        if chunk_text:
            chunks.append(chunk_text)
        return ids[-chunk_overlap:] if len(ids) > chunk_overlap else []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_ids = enc.encode(para)

        # Paragraph is larger than one chunk — split at sentence boundaries
        if len(para_ids) > chunk_size:
            if current_ids:
                current_ids = _flush(current_ids)

            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_ids = enc.encode(sent)
                if len(current_ids) + len(sent_ids) > chunk_size:
                    if current_ids:
                        current_ids = _flush(current_ids)
                current_ids.extend(sent_ids)

        else:
            # Normal paragraph — add to current window
            if len(current_ids) + len(para_ids) > chunk_size:
                current_ids = _flush(current_ids)
            current_ids.extend(para_ids)

    # Flush the final window
    if current_ids:
        _flush(current_ids)

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split a list of LangChain Documents into token-sized chunks.

    Each chunk inherits *all* metadata from its parent document plus two
    extra fields:
        chunk_index   – 0-based position within the parent document
        total_chunks  – total number of chunks produced from that document

    Args:
        documents:     Source documents.
        chunk_size:    Maximum tokens per chunk.
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        List of chunked Documents ready for embedding.
    """
    enc = _get_encoder()  # warm up once before the loop
    result: list[Document] = []

    for doc in documents:
        raw_chunks = _split_text_by_tokens(
            doc.page_content, chunk_size, chunk_overlap
        )
        total = len(raw_chunks)
        base_meta = dict(doc.metadata)

        for idx, text in enumerate(raw_chunks):
            meta = {**base_meta, "chunk_index": idx, "total_chunks": total}
            result.append(Document(page_content=text, metadata=meta))

    log.info(
        "Chunked %d doc(s) → %d chunk(s) (size≤%d tok, overlap=%d tok).",
        len(documents), len(result), chunk_size, chunk_overlap,
    )
    return result
