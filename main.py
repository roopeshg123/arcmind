"""
ArcMind FastAPI Application

REST API endpoints
------------------
  GET  /                    Serve the chat UI  (static/index.html)
  GET  /api/status          Collection health check + vector counts
  POST /api/ingest          Ingest Arc documentation (HTML)
  POST /api/ingest/jira     Ingest Jira issues
  POST /api/ingest/jira/sync  Incremental Jira sync (last N hours)
  POST /api/chat            Blocking Q&A — returns full answer at once
  POST /api/chat/stream     Streaming Q&A — Server-Sent Events (SSE)
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import rag_engine
from ingest import run_ingestion, run_jira_ingestion, run_incremental_jira_sync

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")


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
    description="Enterprise AI assistant for CData Arc — documentation + Jira knowledge.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    ready = docs_count > 0
    return {
        "status":     "ready" if ready else "not_ready",
        "docs_vectors":  docs_count,
        "jira_vectors":  jira_count,
        "total_vectors": docs_count + jira_count,
        "message": None if ready else "Call POST /api/ingest to index documentation.",
    }


# ---------------------------------------------------------------------------
# Routes — ingestion
# ---------------------------------------------------------------------------

_ingestion_lock = False   # simple flag to prevent concurrent runs


@app.post("/api/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest Arc documentation from disk or DOCS_URL."""
    global _ingestion_lock
    if _ingestion_lock:
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

    _ingestion_lock = True
    try:
        rag_engine.reset_chain()
        result = run_ingestion(docs_dir=docs_dir, reset=request.reset)
    finally:
        _ingestion_lock = False

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message"))

    return result


@app.post("/api/ingest/jira")
async def ingest_jira(request: JiraIngestRequest):
    """Ingest Jira issues (full or filtered by JQL)."""
    global _ingestion_lock
    if _ingestion_lock:
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    _ingestion_lock = True
    try:
        result = run_jira_ingestion(
            jql=request.jql,
            reset=request.reset,
        )
    finally:
        _ingestion_lock = False

    return result


@app.post("/api/ingest/jira/sync")
async def jira_sync(request: JiraSyncRequest):
    """Incremental Jira sync — fetches issues updated in the last N hours."""
    result = run_incremental_jira_sync(hours=request.hours or 1)
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
        result = rag_engine.ask(
            question=request.question,
            chat_history=history,
            session_id=request.session_id,
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
