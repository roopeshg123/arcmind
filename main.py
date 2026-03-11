"""
FastAPI application — exposes three REST endpoints:

  GET  /             → Serves the chat UI (static/index.html)
  GET  /api/status   → Reports whether the vector store is ready
  POST /api/ingest   → Triggers the document ingestion pipeline
  POST /api/chat     → Accepts a question + history, returns an answer + sources
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy import of heavy modules so startup is fast
# ---------------------------------------------------------------------------
import rag_engine
from ingest import run_ingestion

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load vector store, reranker, and RAG chain on startup."""
    if rag_engine.is_vector_store_ready():
        print("[startup] Vector store found — loading into memory…")
        rag_engine.load_vector_store()
        print("[startup] Pre-building RAG chain and warming up reranker…")
        rag_engine.get_rag_chain()  # builds chain + loads reranker in one shot
    else:
        print("[startup] No vector store yet — pre-warming reranker for faster first ingest…")
        rag_engine.warmup_reranker()
        print("[startup] Call POST /api/ingest to index your documentation.")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Arc Docs RAG",
    description="Retrieval-Augmented Generation over your Arc application documentation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the chat UI)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str        # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


class IngestRequest(BaseModel):
    docs_dir: Optional[str] = None
    reset: Optional[bool] = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI."""
    return FileResponse("static/index.html")


@app.get("/api/status")
async def status():
    """
    Check whether the vector store is ready to answer questions.
    Also returns a document count if available.
    """
    ready = rag_engine.is_vector_store_ready()

    if ready:
        try:
            vs = rag_engine.load_vector_store()
            count = vs._collection.count()
        except Exception:
            count = -1
        return {"status": "ready", "vectors": count}

    return {"status": "not_ready", "vectors": 0,
            "message": "Run POST /api/ingest to index your documentation first."}


# Global flag to prevent concurrent ingestions
_ingestion_running = False


@app.post("/api/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger the document ingestion pipeline.

    By default this runs synchronously so the caller receives a result.
    For very large doc sets you can move it to a background task instead.
    """
    global _ingestion_running
    if _ingestion_running:
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    docs_dir = request.docs_dir or DOCS_DIR

    if not os.path.isdir(docs_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Directory not found: '{docs_dir}'. "
                   "Place your .html files there and retry."
        )

    _ingestion_running = True
    try:
        # Release the open ChromaDB connection BEFORE ingestion deletes the directory
        rag_engine.reset_chain()
        result = run_ingestion(docs_dir=docs_dir, reset=request.reset)
    finally:
        _ingestion_running = False

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    return result


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question using the RAG pipeline.

    The client passes the full conversation history so the server is stateless.
    """
    if not rag_engine.is_vector_store_ready():
        raise HTTPException(
            status_code=503,
            detail="Documentation index not ready. Call POST /api/ingest first.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        result = rag_engine.ask(question=request.question, chat_history=history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(answer=result["answer"], sources=result["sources"])
