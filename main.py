"""
ArcMind FastAPI Application

REST API endpoints
------------------
  GET  /                          Serve the chat UI  (static/index.html)
  GET  /api/status                Collection health check + vector counts
  POST /api/ingest                Ingest Arc documentation (HTML)
  POST /api/ingest/jira           Ingest Jira issues
  POST /api/ingest/jira/sync      Incremental Jira sync (last N hours)
  POST /api/ingest/confluence     Ingest Confluence pages
  POST /api/ingest/confluence/sync  Incremental Confluence sync (last N hours)
  POST /api/ingest/confluence/update  Smart Confluence update (changed pages only)
  POST /api/chat                  Blocking Q&A — returns full answer at once
  POST /api/chat/stream           Streaming Q&A — Server-Sent Events (SSE)
"""

import asyncio
import functools
import os
import re
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

import rag_engine
from ingest import run_ingestion
from ingest.ingest_jira import ingest_jira_async, incremental_jira_sync_async, smart_jira_update_async
from ingest.ingest_docs import smart_docs_update
from ingest.ingest_confluence import (
    ingest_confluence_async,
    incremental_confluence_sync_async,
    smart_confluence_update_async,
)

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# CORS: set CORS_ORIGINS to a comma-separated list of allowed origins.
# Defaults to "*" (all origins) — restrict this in production.
_CORS_ORIGINS = (
    [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    or ["*"]
)

# API key: set API_KEY env var to require X-API-Key on all /api/* requests.
# Leave unset to disable auth (development only).
_API_KEY = os.getenv("API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Lifespan — warm up on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if rag_engine.is_vector_store_ready():
        print("[startup] Docs collection found — warming up reranker…")
        rag_engine.warmup_reranker()
        rag_engine.get_rag_chain()   # satisfies the truthy check, warms reranker
    else:
        print("[startup] No docs indexed yet — pre-loading reranker…")
        rag_engine.warmup_reranker()
        print("[startup] Call POST /api/ingest to index your documentation.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ArcMind",
    description="Enterprise AI assistant for CData Arc — documentation + Jira + Confluence knowledge.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Enforce X-API-Key header when API_KEY env var is set."""
    if _API_KEY:
        path = request.url.path
        if path not in ("/",) and not path.startswith("/static"):
            if request.headers.get("X-API-Key", "") != _API_KEY:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role:    str    # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question:   str
    history:    Optional[List[ChatMessage]] = []
    session_id: Optional[str] = None        # server-side conversation memory key

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if len(v) > 128:
                raise ValueError("session_id must not exceed 128 characters")
            if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
                raise ValueError(
                    "session_id must only contain alphanumeric characters, "
                    "hyphens, and underscores"
                )
        return v


class ChatResponse(BaseModel):
    answer:      str
    sources:     List[dict]
    jira_issues: List[dict] = []


class IngestRequest(BaseModel):
    docs_dir: Optional[str]  = None
    reset:    Optional[bool] = True


class JiraIngestRequest(BaseModel):
    jql:         Optional[str]  = None
    reset:       Optional[bool] = False
    max_results: Optional[int]  = 0


class JiraSyncRequest(BaseModel):
    hours: Optional[int] = 1


class ConfluenceIngestRequest(BaseModel):
    space_keys:  Optional[List[str]] = None
    reset:       Optional[bool]      = False
    max_results: Optional[int]       = 0


class ConfluenceSyncRequest(BaseModel):
    hours:      Optional[int]       = 1
    space_keys: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Routes — static / status
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/api/status")
async def status():
    """Report vector store readiness and collection sizes."""
    from vector_db.chroma_store import get_store
    store = get_store()
    docs_count = store.docs_count()
    jira_count = store.jira_count()
    confluence_count = store.confluence_count()
    ready = docs_count > 0
    return {
        "status":              "ready" if ready else "not_ready",
        "docs_vectors":        docs_count,
        "jira_vectors":        jira_count,
        "confluence_vectors":  confluence_count,
        "total_vectors":       docs_count + jira_count + confluence_count,
        "message": None if ready else "Call POST /api/ingest to index documentation.",
    }


# ---------------------------------------------------------------------------
# Routes — ingestion
# ---------------------------------------------------------------------------

_ingestion_lock: asyncio.Lock = asyncio.Lock()  # prevents concurrent ingest runs
_ingest_progress: dict = {
    "stage": "idle", "fetched": 0, "total": 0,
    "vectors": 0, "chunks_done": 0, "chunks_total": 0,
}


@app.get("/api/ingest/progress")
async def ingest_progress():
    """Return current ingestion progress (polled by the UI)."""
    return _ingest_progress


@app.post("/api/ingest")
async def ingest(request: IngestRequest):
    """Ingest Arc documentation from disk or DOCS_URL."""
    global _ingest_progress
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    docs_dir = request.docs_dir or DOCS_DIR
    if not request.docs_dir and not os.getenv("DOCS_URL", "").strip():
        if not os.path.isdir(docs_dir):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Directory not found: '{docs_dir}'. "
                    "Place .html files there or set DOCS_URL in your .env."
                ),
            )

    result = None
    async with _ingestion_lock:
        _ingest_progress = {
            "stage": "loading", "fetched": 0, "total": 0,
            "vectors": 0, "chunks_done": 0, "chunks_total": 0,
        }
        rag_engine.reset_chain()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(run_ingestion, docs_dir=docs_dir, reset=request.reset, progress=_ingest_progress),
            )
        finally:
            _ingest_progress = {
                "stage": "idle", "fetched": 0, "total": 0,
                "vectors": result.get("vectors_stored", 0) if result else 0,
            }

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message"))

    return result


@app.post("/api/ingest/jira")
async def ingest_jira(request: JiraIngestRequest):
    """Ingest Jira issues (full or filtered by JQL)."""
    global _ingest_progress
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    result = None
    async with _ingestion_lock:
        _ingest_progress = {
            "stage": "fetching", "fetched": 0, "total": 0,
            "vectors": 0, "chunks_done": 0, "chunks_total": 0,
        }
        try:
            result = await ingest_jira_async(
                jql=request.jql,
                reset=request.reset,
                progress=_ingest_progress,
            )
        finally:
            _ingest_progress = {
                "stage": "idle", "fetched": 0, "total": 0,
                "vectors": result.get("vectors_stored", 0) if result else 0,
            }

    return result


@app.post("/api/ingest/update")
async def ingest_update(request: IngestRequest):
    """Smart incremental docs update — only re-indexes changed or new files."""
    global _ingest_progress
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    docs_dir = request.docs_dir or DOCS_DIR

    result = None
    async with _ingestion_lock:
        _ingest_progress = {
            "stage": "loading", "fetched": 0, "total": 0,
            "vectors": 0, "chunks_done": 0, "chunks_total": 0,
        }
        rag_engine.reset_chain()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    smart_docs_update,
                    docs_dir=docs_dir,
                    progress=_ingest_progress,
                ),
            )
        finally:
            _ingest_progress = {
                "stage": "idle", "fetched": 0, "total": 0,
                "vectors": result.get("vectors_stored", 0) if result else 0,
            }

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message"))

    return result


@app.post("/api/ingest/jira/update")
async def jira_smart_update():
    """Smart incremental Jira update — only re-indexes new or changed tickets."""
    global _ingest_progress
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    result = None
    async with _ingestion_lock:
        _ingest_progress = {
            "stage": "fetching", "fetched": 0, "total": 0,
            "vectors": 0, "chunks_done": 0, "chunks_total": 0,
        }
        try:
            result = await smart_jira_update_async(progress=_ingest_progress)
        finally:
            _ingest_progress = {
                "stage": "idle", "fetched": 0, "total": 0,
                "vectors": result.get("vectors_stored", 0) if result else 0,
            }

    return result


@app.post("/api/ingest/jira/sync")
async def jira_sync(request: JiraSyncRequest):
    """Incremental Jira sync — fetches issues updated in the last N hours."""
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")
    async with _ingestion_lock:
        result = await incremental_jira_sync_async(hours=request.hours or 1)
    return result


# ---------------------------------------------------------------------------
# Routes — chat (blocking)
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question using the full RAG pipeline (blocking response).

    The client may pass the full history in *history* OR rely on server-side
    memory by supplying a stable *session_id*.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(
            status_code=503,
            detail="Documentation not indexed yet. Call POST /api/ingest first.",
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]

    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                rag_engine.ask,
                question=request.question,
                chat_history=history,
                session_id=request.session_id,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        jira_issues=result.get("jira_issues", []),
    )


# ---------------------------------------------------------------------------
# Routes — chat (streaming SSE)
# ---------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream the RAG answer as Server-Sent Events.

    Each event is a JSON object on a `data:` line.  Event types:
        {"token":  "…"}          — incremental answer token
        {"error":  "…"}          — error (stream terminates)
        {"done": true,
         "sources": […],
         "jira_issues": […]}     — final metadata event after all tokens

    Client usage (JavaScript):
        const es = new EventSource('/api/chat/stream');
        // or use fetch with ReadableStream for POST
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(
            status_code=503,
            detail="Documentation not indexed yet. Call POST /api/ingest first.",
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]

    async def _generate():
        async for event in rag_engine.ask_stream(
            question=request.question,
            chat_history=history,
            session_id=request.session_id,
        ):
            yield event

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# Routes — Confluence ingestion
# ---------------------------------------------------------------------------

@app.post("/api/ingest/confluence")
async def ingest_confluence(request: ConfluenceIngestRequest):
    """Ingest Confluence pages (full or filtered by space keys)."""
    global _ingest_progress
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    result = None
    async with _ingestion_lock:
        _ingest_progress = {
            "stage": "fetching", "fetched": 0, "total": 0,
            "vectors": 0, "chunks_done": 0, "chunks_total": 0,
        }
        try:
            result = await ingest_confluence_async(
                space_keys=request.space_keys or None,
                reset=request.reset,
                max_results=request.max_results or 0,
                progress=_ingest_progress,
            )
        finally:
            _ingest_progress = {
                "stage": "idle", "fetched": 0, "total": 0,
                "vectors": result.get("vectors_stored", 0) if result else 0,
            }

    return result


@app.post("/api/ingest/confluence/sync")
async def confluence_sync(request: ConfluenceSyncRequest):
    """Incremental Confluence sync — fetches pages updated in the last N hours."""
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")
    async with _ingestion_lock:
        result = await incremental_confluence_sync_async(
            hours=request.hours or 1,
            space_keys=request.space_keys or None,
        )
    return result


@app.post("/api/ingest/confluence/update")
async def confluence_smart_update(request: ConfluenceIngestRequest):
    """Smart incremental Confluence update — only re-indexes changed or new pages."""
    global _ingest_progress
    if _ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    result = None
    async with _ingestion_lock:
        _ingest_progress = {
            "stage": "fetching", "fetched": 0, "total": 0,
            "vectors": 0, "chunks_done": 0, "chunks_total": 0,
        }
        try:
            result = await smart_confluence_update_async(
                space_keys=request.space_keys or None,
                progress=_ingest_progress,
            )
        finally:
            _ingest_progress = {
                "stage": "idle", "fetched": 0, "total": 0,
                "vectors": result.get("vectors_stored", 0) if result else 0,
            }

    return result


# ---------------------------------------------------------------------------
