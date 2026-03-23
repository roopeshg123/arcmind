"""
ArcMind FastAPI Application

REST API endpoints
------------------
  GET  /                              Serve the chat UI  (static/index.html)
  GET  /api/status                    Collection health check + vector counts
  POST /api/ingest                    Ingest Arc documentation (HTML)
  POST /api/ingest/jira               Ingest Jira issues
  POST /api/ingest/jira/sync          Incremental Jira sync (last N hours)
  POST /api/ingest/confluence         Ingest Confluence pages
  POST /api/ingest/confluence/sync    Incremental Confluence sync (last N hours)
  POST /api/ingest/confluence/update  Smart Confluence update (changed pages only)
  POST /api/chat                      Blocking Q&A — returns full answer at once
  POST /api/chat/stream               Streaming Q&A — Server-Sent Events (SSE)

Arc-Specific Intelligence Tools (slash commands)
-------------------------------------------------
  POST /api/tools/error-decode     Diagnose an Arc error log → root cause + fix
  POST /api/tools/edi-explain      Explain an X12/EDIFACT message segment-by-segment
  POST /api/tools/similar          Find Jira tickets similar to a given ticket ID
  POST /api/tools/generate-ticket  Draft a Jira ticket from plain-English description
  POST /api/tools/changelog        Structured changelog for an Arc connector
  POST /api/tools/generate-script  Generate ArcScript or Python script from plain English
  POST /api/tools/fix-script        Debug and fix a broken ArcScript or Python script
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
from rag import arc_tools
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
# Pydantic models — Arc-specific tools
# ---------------------------------------------------------------------------

class ErrorDecodeRequest(BaseModel):
    error_text: str

    @field_validator("error_text")
    @classmethod
    def _check_error_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("error_text cannot be empty")
        if len(v) > 8000:
            raise ValueError("error_text must not exceed 8 000 characters")
        return v


class EDIExplainRequest(BaseModel):
    edi_text: str

    @field_validator("edi_text")
    @classmethod
    def _check_edi_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("edi_text cannot be empty")
        if len(v) > 16000:
            raise ValueError("edi_text must not exceed 16 000 characters")
        return v


class SimilarTicketsRequest(BaseModel):
    ticket_id: str

    @field_validator("ticket_id")
    @classmethod
    def _check_ticket_id(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.match(r'^[A-Z][A-Z0-9]+-\d+$', v):
            raise ValueError("ticket_id must be a valid Jira key, e.g. ARCESB-12345")
        return v


class GenerateTicketRequest(BaseModel):
    description: str

    @field_validator("description")
    @classmethod
    def _check_description(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description cannot be empty")
        if len(v) > 4000:
            raise ValueError("description must not exceed 4 000 characters")
        return v


class ChangelogRequest(BaseModel):
    connector: str

    @field_validator("connector")
    @classmethod
    def _check_connector(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("connector cannot be empty")
        return v.strip()


class GenerateScriptRequest(BaseModel):
    requirement: str

    @field_validator("requirement")
    @classmethod
    def _check_requirement(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("requirement cannot be empty")
        if len(v) > 4000:
            raise ValueError("requirement must not exceed 4 000 characters")
        return v


class FixScriptRequest(BaseModel):
    script: str
    error_description: str

    @field_validator("script")
    @classmethod
    def _check_script(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("script cannot be empty")
        if len(v) > 8000:
            raise ValueError("script must not exceed 8 000 characters")
        return v

    @field_validator("error_description")
    @classmethod
    def _check_error_description(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("error_description cannot be empty")
        if len(v) > 2000:
            raise ValueError("error_description must not exceed 2 000 characters")
        return v


# ---------------------------------------------------------------------------
# Routes — Arc-specific intelligence tools
# ---------------------------------------------------------------------------

@app.post("/api/tools/error-decode")
async def tool_error_decode(request: ErrorDecodeRequest):
    """
    Diagnose a CData Arc error log or exception.

    Searches docs + Jira history + Confluence for matching issues,
    returns root-cause analysis and step-by-step fix.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(status_code=503, detail="Documentation not indexed yet.")
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(arc_tools.decode_error, error_text=request.error_text),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/api/tools/edi-explain")
async def tool_edi_explain(request: EDIExplainRequest):
    """
    Explain a raw X12 or EDIFACT message segment-by-segment in plain English.

    Annotates each element, flags invalid values, gives business summary
    and CData Arc handling tips.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(status_code=503, detail="Documentation not indexed yet.")
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(arc_tools.explain_edi, edi_text=request.edi_text),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/api/tools/similar")
async def tool_similar_tickets(request: SimilarTicketsRequest):
    """
    Find Jira tickets similar to the given ticket ID.

    Uses the ticket's content as a vector search query, deduplicates
    by ticket key, and returns a relationship analysis.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(status_code=503, detail="Documentation not indexed yet.")
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(arc_tools.find_similar, ticket_id=request.ticket_id),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/api/tools/generate-ticket")
async def tool_generate_ticket(request: GenerateTicketRequest):
    """
    Draft a Jira ticket from a plain-English problem description.

    Auto-detects the Arc component, finds related past tickets,
    and returns a fully structured ticket ready to copy into Jira.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(status_code=503, detail="Documentation not indexed yet.")
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(arc_tools.generate_ticket_draft, description=request.description),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/api/tools/changelog")
async def tool_changelog(request: ChangelogRequest):
    """
    Return a structured changelog for the given Arc connector/component.

    Groups resolved Jira tickets by Fix Version so you can see
    exactly what changed in each release.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(status_code=503, detail="Documentation not indexed yet.")
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(arc_tools.connector_changelog, connector_name=request.connector),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/api/tools/generate-script")
async def tool_generate_script(request: GenerateScriptRequest):
    """
    Generate a working ArcScript or Python script from a plain-English requirement.

    Retrieves scripting docs and Jira context, then produces a complete,
    annotated script with placement instructions.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(status_code=503, detail="Documentation not indexed yet.")
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(arc_tools.generate_script, requirement=request.requirement),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/api/tools/fix-script")
async def tool_fix_script(request: FixScriptRequest):
    """
    Diagnose and fix a broken ArcScript or Python script.

    Accepts the broken script and a description of the error or wrong behaviour,
    retrieves scripting docs, and returns the fully corrected script with
    a detailed explanation of every change made.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(status_code=503, detail="Documentation not indexed yet.")
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                arc_tools.fix_script,
                script=request.script,
                error_description=request.error_description,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


# ---------------------------------------------------------------------------
