"""
ChromaDB vector store manager for ArcMind.

Maintains two named collections inside a single ChromaDB instance:
    arcmind_docs  — Arc documentation chunks
    arcmind_jira  — Jira issue chunks

Features
--------
- Batch embedding (100 docs per OpenAI API call)
- Separate BM25 indexes persisted as pickle files (one per collection)
- Lazy singleton — one store instance for the whole process lifetime
- Windows-safe release (clears SharedSystemClient cache before deleting files)
"""

from __future__ import annotations

import gc
import logging
import os
import pickle
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DB_DIR    = os.getenv("CHROMA_DB_DIR",    "./chroma_db")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",   "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "text-embedding-3-large")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))

DOCS_COLLECTION  = "arcmind_docs"
JIRA_COLLECTION  = "arcmind_jira"

_BM25_DOCS_PKL   = os.path.join(CHROMA_DB_DIR, "bm25_docs.pkl")
_BM25_JIRA_PKL   = os.path.join(CHROMA_DB_DIR, "bm25_jira.pkl")


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------

class ChromaStore:
    """Manages two ChromaDB collections and their companion BM25 indexes."""

    def __init__(self) -> None:
        self._embeddings:  OpenAIEmbeddings | None = None
        self._docs_store:  Chroma | None = None
        self._jira_store:  Chroma | None = None
        # BM25 — loaded lazily from disk
        self._bm25_docs:   Any = None
        self._bm25_jira:   Any = None
        self._corpus_docs: list[tuple[str, str, dict]] = []   # (chroma_id, text, metadata)
        self._corpus_jira: list[tuple[str, str, dict]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
            )
        return self._embeddings

    def _get_store(self, collection: str) -> Chroma:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        if collection == DOCS_COLLECTION:
            if self._docs_store is None:
                self._docs_store = Chroma(
                    collection_name=DOCS_COLLECTION,
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self._get_embeddings(),
                )
            return self._docs_store
        else:
            if self._jira_store is None:
                self._jira_store = Chroma(
                    collection_name=JIRA_COLLECTION,
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self._get_embeddings(),
                )
            return self._jira_store

    def _reset_collection(self, collection_name: str) -> None:
        """Drop and recreate a ChromaDB collection."""
        # Release LangChain wrapper first
        if collection_name == DOCS_COLLECTION:
            self._docs_store = None
            bm25_path = _BM25_DOCS_PKL
        else:
            self._jira_store = None
            bm25_path = _BM25_JIRA_PKL

        gc.collect()
        _clear_chroma_cache()

        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            client.delete_collection(collection_name)
            log.info("Collection '%s' deleted.", collection_name)
        except Exception as exc:
            log.warning("Could not delete collection '%s': %s", collection_name, exc)

        if os.path.exists(bm25_path):
            try:
                os.remove(bm25_path)
            except Exception:
                pass

        _clear_chroma_cache()

    @staticmethod
    def _add_in_batches(
        store: Chroma,
        chunks: list[Document],
        batch_size: int,
        on_progress: Any = None,
    ) -> None:
        """Upsert *chunks* to ChromaDB in batches to respect API rate limits.

        Args:
            on_progress: Optional callable(done: int, total: int) called after
                         each batch so callers can report live progress.
        """
        total = len(chunks)
        done  = 0
        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]
            store.add_documents(batch)
            done += len(batch)
            log.info(
                "  Embedded batch %d–%d / %d.",
                i + 1, min(i + batch_size, total), total,
            )
            if on_progress is not None:
                try:
                    on_progress(done, total)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _build_bm25(self, collection_name: str, new_docs: list[Document] | None = None) -> None:
        """
        Build or incrementally extend the BM25 index.

        If *new_docs* is provided and an in-memory corpus already exists, extends
        the corpus with the new documents instead of fetching all documents from
        ChromaDB.  This avoids an expensive full-collection scan on every
        incremental Jira sync.  Falls back to a full rebuild when no cached
        corpus is available (e.g. cold start after restart).
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            log.warning("rank_bm25 not installed — BM25 indexing skipped.")
            return

        pkl_path      = _BM25_DOCS_PKL if collection_name == DOCS_COLLECTION else _BM25_JIRA_PKL
        is_docs       = (collection_name == DOCS_COLLECTION)
        cached_corpus = self._corpus_docs if is_docs else self._corpus_jira

        try:
            if new_docs is not None and cached_corpus:
                # Append path — extend cached corpus without a ChromaDB roundtrip
                new_entries: list[tuple[str, str, dict]] = [
                    ("", doc.page_content, doc.metadata) for doc in new_docs
                ]
                corpus = cached_corpus + new_entries
            else:
                # Full-rebuild path — fetch all documents from ChromaDB
                store     = self._get_store(collection_name)
                raw       = store._collection.get(include=["documents", "metadatas"])
                texts:     list[str]  = raw.get("documents") or []
                ids:       list[str]  = raw.get("ids")       or []
                metadatas: list[dict] = raw.get("metadatas") or [{}] * len(texts)

                if not texts:
                    log.warning("No texts for BM25 index in '%s'.", collection_name)
                    return

                corpus = list(zip(ids, texts, metadatas))

            tokenized = [entry[1].lower().split() for entry in corpus]
            bm25      = BM25Okapi(tokenized)

            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            with open(pkl_path, "wb") as fh:
                pickle.dump({"bm25": bm25, "corpus": corpus}, fh)

            if is_docs:
                self._bm25_docs   = bm25
                self._corpus_docs = corpus
            else:
                self._bm25_jira   = bm25
                self._corpus_jira = corpus

            log.info(
                "BM25 index built for '%s': %d docs → '%s'.",
                collection_name, len(corpus), pkl_path,
            )
        except Exception as exc:
            log.error("BM25 build failed for '%s': %s", collection_name, exc)

    def _load_bm25(self, collection_name: str) -> tuple[Any, list[tuple[str, str, dict]]]:
        """Return (bm25, corpus) — loads from disk if not yet in memory."""
        if collection_name == DOCS_COLLECTION:
            if self._bm25_docs is not None:
                return self._bm25_docs, self._corpus_docs
            pkl_path = _BM25_DOCS_PKL
        else:
            if self._bm25_jira is not None:
                return self._bm25_jira, self._corpus_jira
            pkl_path = _BM25_JIRA_PKL

        if not os.path.exists(pkl_path):
            return None, []

        try:
            data   = _safe_pickle_load(pkl_path)
            bm25   = data["bm25"]
            corpus = data["corpus"]

            if collection_name == DOCS_COLLECTION:
                self._bm25_docs   = bm25
                self._corpus_docs = corpus
            else:
                self._bm25_jira   = bm25
                self._corpus_jira = corpus

            log.info("BM25 index loaded for '%s': %d docs.", collection_name, len(corpus))
            return bm25, corpus
        except Exception as exc:
            log.error("Failed to load BM25 index '%s': %s", pkl_path, exc)
            return None, []

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def add_docs_batch(
        self,
        chunks: list[Document],
        reset: bool = False,
        on_progress: Any = None,
    ) -> int:
        """Embed and store documentation chunks.  Returns new collection count."""
        if reset:
            self._reset_collection(DOCS_COLLECTION)

        store = self._get_store(DOCS_COLLECTION)
        self._add_in_batches(store, chunks, EMBED_BATCH_SIZE, on_progress=on_progress)
        count = store._collection.count()
        log.info("Docs collection: %d vectors.", count)

        self._build_bm25(DOCS_COLLECTION, new_docs=None if reset else chunks)
        return count

    def add_jira_batch(
        self,
        chunks: list[Document],
        reset: bool = False,
        on_progress: Any = None,
    ) -> int:
        """Embed and store Jira chunks.  Returns new collection count."""
        if reset:
            self._reset_collection(JIRA_COLLECTION)

        store = self._get_store(JIRA_COLLECTION)
        self._add_in_batches(store, chunks, EMBED_BATCH_SIZE, on_progress=on_progress)
        count = store._collection.count()
        log.info("Jira collection: %d vectors.", count)

        # Pass new chunks for append mode to avoid a full ChromaDB scan
        self._build_bm25(JIRA_COLLECTION, new_docs=None if reset else chunks)
        return count

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def similarity_search_docs(
        self,
        query: str,
        k: int = 6,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """Cosine-similarity search in the docs collection."""
        store  = self._get_store(DOCS_COLLECTION)
        kwargs: dict = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        try:
            return store.similarity_search(query, **kwargs)
        except Exception as exc:
            log.error("Docs vector search failed: %s", exc)
            return []

    def similarity_search_jira(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """Cosine-similarity search in the Jira collection."""
        store  = self._get_store(JIRA_COLLECTION)
        kwargs: dict = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        try:
            return store.similarity_search(query, **kwargs)
        except Exception as exc:
            log.error("Jira vector search failed: %s", exc)
            return []

    def bm25_search(
        self,
        query: str,
        collection: str = "docs",
        k: int = 10,
    ) -> list[Document]:
        """BM25 keyword search.  Returns top-*k* documents."""
        col_name = DOCS_COLLECTION if collection == "docs" else JIRA_COLLECTION
        bm25, corpus = self._load_bm25(col_name)
        if bm25 is None or not corpus:
            return []

        try:
            import numpy as np
        except ImportError:
            log.warning("numpy not installed — BM25 search unavailable.")
            return []

        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]

        docs: list[Document] = []
        for idx in top_idx:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            entry = corpus[idx]
            # Support both old 2-tuple corpus and new 3-tuple corpus
            text     = entry[1]
            metadata = dict(entry[2]) if len(entry) > 2 else {}
            metadata["bm25_score"] = score
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def get_jira_by_tickets(self, ticket_ids: list[str]) -> list[Document]:
        """Direct fetch of Jira documents matching the given ticket IDs."""
        if not ticket_ids:
            return []
        store = self._get_store(JIRA_COLLECTION)
        try:
            raw = store._collection.get(
                where={"ticket": {"$in": ticket_ids}},
                include=["documents", "metadatas"],
            )
            docs: list[Document] = []
            for text, meta in zip(
                raw.get("documents") or [],
                raw.get("metadatas") or [],
            ):
                if text:
                    docs.append(Document(page_content=text, metadata=meta or {}))
            return docs
        except Exception as exc:
            log.error("Jira ticket fetch failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_docs_ready(self) -> bool:
        try:
            return self._get_store(DOCS_COLLECTION)._collection.count() > 0
        except Exception:
            return False

    def is_jira_ready(self) -> bool:
        try:
            return self._get_store(JIRA_COLLECTION)._collection.count() > 0
        except Exception:
            return False

    def docs_count(self) -> int:
        try:
            return self._get_store(DOCS_COLLECTION)._collection.count()
        except Exception:
            return 0

    def jira_count(self) -> int:
        try:
            return self._get_store(JIRA_COLLECTION)._collection.count()
        except Exception:
            return 0

    def release(self) -> None:
        """Free all ChromaDB handles (important on Windows to release SQLite locks)."""
        self._docs_store = None
        self._jira_store = None
        gc.collect()
        _clear_chroma_cache()


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _clear_chroma_cache() -> None:
    """Clear the ChromaDB process-level singleton cache."""
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def _safe_pickle_load(path: str) -> Any:
    """
    Load a pickle file after verifying the path lies within CHROMA_DB_DIR.
    Defends against path-traversal if the path were ever user-influenced.
    """
    abs_path = os.path.realpath(path)
    abs_base = os.path.realpath(CHROMA_DB_DIR)
    if not (abs_path == abs_base or abs_path.startswith(abs_base + os.sep)):
        raise ValueError(
            f"Refusing to load pickle file outside data directory: '{path}'"
        )
    with open(abs_path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 — path is validated above


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    """Return the process-wide ChromaStore singleton."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


def reset_store() -> None:
    """Release and discard the singleton (used before re-ingestion)."""
    global _store
    if _store is not None:
        _store.release()
        _store = None
