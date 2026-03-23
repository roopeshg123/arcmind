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
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DB_DIR      = os.getenv("CHROMA_DB_DIR",      "./chroma_db")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY",     "")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()  # "openai" or "huggingface"
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL",    "text-embedding-3-large")
EMBED_BATCH_SIZE   = int(os.getenv("EMBED_BATCH_SIZE", "100"))

DOCS_COLLECTION        = "arcmind_docs"
JIRA_COLLECTION        = "arcmind_jira"
CONFLUENCE_COLLECTION  = "arcmind_confluence"

_BM25_DOCS_PKL        = os.path.join(CHROMA_DB_DIR, "bm25_docs.pkl")
_BM25_JIRA_PKL        = os.path.join(CHROMA_DB_DIR, "bm25_jira.pkl")
_BM25_CONFLUENCE_PKL  = os.path.join(CHROMA_DB_DIR, "bm25_confluence.pkl")


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------

class ChromaStore:
    """Manages two ChromaDB collections and their companion BM25 indexes."""

    def __init__(self) -> None:
        self._embeddings:        Any | None = None
        self._docs_store:        Chroma | None = None
        self._jira_store:        Chroma | None = None
        self._confluence_store:  Chroma | None = None
        # BM25 — loaded lazily from disk
        self._bm25_docs:        Any = None
        self._bm25_jira:        Any = None
        self._bm25_confluence:  Any = None
        self._corpus_docs:      list[tuple[str, str, dict]] = []   # (chroma_id, text, metadata)
        self._corpus_jira:      list[tuple[str, str, dict]] = []
        self._corpus_confluence: list[tuple[str, str, dict]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self):
        if self._embeddings is None:
            if EMBEDDING_PROVIDER == "huggingface":
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            else:  # default: openai
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
        elif collection == JIRA_COLLECTION:
            if self._jira_store is None:
                self._jira_store = Chroma(
                    collection_name=JIRA_COLLECTION,
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self._get_embeddings(),
                )
            return self._jira_store
        else:  # CONFLUENCE_COLLECTION
            if self._confluence_store is None:
                self._confluence_store = Chroma(
                    collection_name=CONFLUENCE_COLLECTION,
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self._get_embeddings(),
                )
            return self._confluence_store

    def _reset_collection(self, collection_name: str) -> None:
        """Drop and recreate a ChromaDB collection."""
        # Release LangChain wrapper first
        if collection_name == DOCS_COLLECTION:
            self._docs_store = None
            bm25_path = _BM25_DOCS_PKL
        elif collection_name == JIRA_COLLECTION:
            self._jira_store = None
            bm25_path = _BM25_JIRA_PKL
        else:  # CONFLUENCE_COLLECTION
            self._confluence_store = None
            bm25_path = _BM25_CONFLUENCE_PKL

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

        if collection_name == DOCS_COLLECTION:
            pkl_path = _BM25_DOCS_PKL
        elif collection_name == JIRA_COLLECTION:
            pkl_path = _BM25_JIRA_PKL
        else:
            pkl_path = _BM25_CONFLUENCE_PKL
        is_docs       = (collection_name == DOCS_COLLECTION)
        is_confluence = (collection_name == CONFLUENCE_COLLECTION)
        if is_docs:
            cached_corpus = self._corpus_docs
        elif is_confluence:
            cached_corpus = self._corpus_confluence
        else:
            cached_corpus = self._corpus_jira

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
            elif is_confluence:
                self._bm25_confluence   = bm25
                self._corpus_confluence = corpus
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
        elif collection_name == JIRA_COLLECTION:
            if self._bm25_jira is not None:
                return self._bm25_jira, self._corpus_jira
            pkl_path = _BM25_JIRA_PKL
        else:  # CONFLUENCE_COLLECTION
            if self._bm25_confluence is not None:
                return self._bm25_confluence, self._corpus_confluence
            pkl_path = _BM25_CONFLUENCE_PKL

        if not os.path.exists(pkl_path):
            return None, []

        try:
            data   = _safe_pickle_load(pkl_path)
            bm25   = data["bm25"]
            corpus = data["corpus"]

            if collection_name == DOCS_COLLECTION:
                self._bm25_docs   = bm25
                self._corpus_docs = corpus
            elif collection_name == CONFLUENCE_COLLECTION:
                self._bm25_confluence   = bm25
                self._corpus_confluence = corpus
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

    def add_confluence_batch(
        self,
        chunks: list[Document],
        reset: bool = False,
        on_progress: Any = None,
    ) -> int:
        """Embed and store Confluence chunks.  Returns new collection count."""
        if reset:
            self._reset_collection(CONFLUENCE_COLLECTION)

        store = self._get_store(CONFLUENCE_COLLECTION)
        self._add_in_batches(store, chunks, EMBED_BATCH_SIZE, on_progress=on_progress)
        count = store._collection.count()
        log.info("Confluence collection: %d vectors.", count)

        self._build_bm25(CONFLUENCE_COLLECTION, new_docs=None if reset else chunks)
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

    def similarity_search_confluence(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """Cosine-similarity search in the Confluence collection."""
        store  = self._get_store(CONFLUENCE_COLLECTION)
        kwargs: dict = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        try:
            return store.similarity_search(query, **kwargs)
        except Exception as exc:
            log.error("Confluence vector search failed: %s", exc)
            return []

    def bm25_search(
        self,
        query: str,
        collection: str = "docs",
        k: int = 10,
    ) -> list[Document]:
        """BM25 keyword search.  Returns top-*k* documents."""
        if collection == "docs":
            col_name = DOCS_COLLECTION
        elif collection == "confluence":
            col_name = CONFLUENCE_COLLECTION
        else:
            col_name = JIRA_COLLECTION
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

    def is_confluence_ready(self) -> bool:
        try:
            return self._get_store(CONFLUENCE_COLLECTION)._collection.count() > 0
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

    def confluence_count(self) -> int:
        try:
            return self._get_store(CONFLUENCE_COLLECTION)._collection.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Smart incremental update helpers
    # ------------------------------------------------------------------

    def get_existing_jira_state(self) -> dict[str, dict]:
        """
        Return {ticket_key: {"updated": timestamp, "hash": content_hash}}
        for every ticket currently stored in the Jira collection.

        Fetches in batches of 5,000 so it works reliably on large collections
        regardless of ChromaDB version.  "ticket_hash" is stored by the new
        ingestion code; legacy chunks without it fall back to timestamp-only
        comparison.
        """
        store      = self._get_store(JIRA_COLLECTION)
        result:    dict[str, dict] = {}
        batch_size = 5000
        offset     = 0
        try:
            while True:
                raw   = store._collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
                metas = raw.get("metadatas") or []
                if not metas:
                    break
                for meta in metas:
                    if not meta:
                        continue
                    key     = meta.get("ticket", "")
                    updated = meta.get("updated", "") or ""
                    thash   = meta.get("ticket_hash", "") or ""
                    if not key:
                        continue
                    # Keep the record with the highest updated timestamp
                    # (all chunks for one ticket share the same values, but
                    # this is defensive against any inconsistency).
                    if key not in result or updated > result[key]["updated"]:
                        result[key] = {"updated": updated, "hash": thash}
                if len(metas) < batch_size:
                    break
                offset += batch_size
            log.info(
                "Jira index snapshot: %d unique tickets read from ChromaDB.",
                len(result),
            )
            return result
        except Exception as exc:
            log.error("Failed to read existing Jira state: %s", exc)
            return {}

    def delete_jira_tickets(self, ticket_keys: list[str]) -> None:
        """Delete ALL ChromaDB chunks for the given Jira ticket keys."""
        if not ticket_keys:
            return
        store = self._get_store(JIRA_COLLECTION)
        for key in ticket_keys:
            try:
                store._collection.delete(where={"ticket": key})
                log.debug("Deleted chunks for Jira ticket %s.", key)
            except Exception as exc:
                log.warning("Could not delete ticket %s: %s", key, exc)

    def get_existing_confluence_state(self) -> dict[str, dict]:
        """
        Return {page_id: {"updated": timestamp, "hash": content_hash}}
        for every page currently stored in the Confluence collection.
        """
        store      = self._get_store(CONFLUENCE_COLLECTION)
        result:    dict[str, dict] = {}
        batch_size = 5000
        offset     = 0
        try:
            while True:
                raw   = store._collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
                metas = raw.get("metadatas") or []
                if not metas:
                    break
                for meta in metas:
                    if not meta:
                        continue
                    pid     = meta.get("page_id", "")
                    updated = meta.get("updated", "") or ""
                    chash   = meta.get("content_hash", "") or ""
                    if not pid:
                        continue
                    if pid not in result or updated > result[pid]["updated"]:
                        result[pid] = {"updated": updated, "hash": chash}
                if len(metas) < batch_size:
                    break
                offset += batch_size
            log.info(
                "Confluence index snapshot: %d unique pages read from ChromaDB.",
                len(result),
            )
            return result
        except Exception as exc:
            log.error("Failed to read existing Confluence state: %s", exc)
            return {}

    def delete_confluence_pages(self, page_ids: list[str]) -> None:
        """Delete ALL ChromaDB chunks for the given Confluence page IDs."""
        if not page_ids:
            return
        store = self._get_store(CONFLUENCE_COLLECTION)
        for pid in page_ids:
            try:
                store._collection.delete(where={"page_id": pid})
                log.debug("Deleted chunks for Confluence page %s.", pid)
            except Exception as exc:
                log.warning("Could not delete Confluence page %s: %s", pid, exc)

    def get_existing_docs_index(self) -> dict[str, str]:
        """
        Return {source_id: content_hash} for every document chunk currently
        stored in the docs collection.

        source_id is ``file_path`` for disk-loaded files, or ``url`` for
        web-crawled pages.  content_hash is the stored SHA-256 hex digest
        (added by the smart-update path); for chunks without a stored hash,
        we compute one from the stored text so old collections work too.
        """
        import hashlib
        store = self._get_store(DOCS_COLLECTION)
        try:
            raw   = store._collection.get(include=["documents", "metadatas"])
            texts = raw.get("documents") or []
            metas = raw.get("metadatas") or []
            result: dict[str, str] = {}
            for text, meta in zip(texts, metas):
                if not meta:
                    continue
                sid = meta.get("file_path") or meta.get("url") or ""
                if not sid:
                    continue
                if sid in result:
                    continue  # one hash per source is enough
                stored_hash = meta.get("content_hash") or ""
                if not stored_hash and text:
                    stored_hash = hashlib.sha256(text.encode()).hexdigest()
                result[sid] = stored_hash
            return result
        except Exception as exc:
            log.error("Failed to read existing docs metadata: %s", exc)
            return {}

    def delete_docs_by_source_id(self, source_ids: list[str]) -> None:
        """
        Delete all doc chunks whose file_path OR url equals one of
        *source_ids*.
        """
        if not source_ids:
            return
        store = self._get_store(DOCS_COLLECTION)
        for sid in source_ids:
            try:
                # Try file_path first, then url
                for field in ("file_path", "url"):
                    store._collection.delete(where={field: sid})
            except Exception as exc:
                log.warning("Could not delete doc source '%s': %s", sid, exc)

    def release(self) -> None:
        """Free all ChromaDB handles (important on Windows to release SQLite locks)."""
        self._docs_store       = None
        self._jira_store       = None
        self._confluence_store = None
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
