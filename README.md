# Arc Docs RAG — Retrieval-Augmented Generation for Your Application Documentation

A full-stack RAG application that lets you **chat with your Arc application
documentation** using OpenAI's GPT models and a local ChromaDB vector store.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        Browser (Chat UI)                       │
│                    static/index.html                           │
└──────────────────────────┬─────────────────────────────────────┘
                           │  HTTP REST
┌──────────────────────────▼─────────────────────────────────────┐
│                  FastAPI Backend  (main.py)                     │
│                                                                │
│  POST /api/ingest  →  ingest.py (HTML → chunks → embeddings)  │
│  POST /api/chat    →  rag_engine.py (retrieve → LLM → answer) │
│  GET  /api/status  →  vector store health check               │
└────────────┬──────────────────────────────┬────────────────────┘
             │                              │
    ┌────────▼────────┐           ┌─────────▼──────────────────┐
    │   ChromaDB      │           │   OpenAI API               │
    │  (local disk)   │           │  • text-embedding-3-large  │
    │  chroma_db/     │           │  • gpt-4.1                 │
    └─────────────────┘           └────────────────────────────┘
```

### Data flow

1. **Ingestion** (one-time setup)
   `docs/*.html` → BeautifulSoup → clean text → `RecursiveCharacterTextSplitter`
   → `text-embedding-3-large` Embeddings → ChromaDB (persisted locally)

2. **Chat**
   User question + history → `gpt-4.1` rewrites & expands query →
   ChromaDB MMR fetches 30 diverse candidates →
   **Cross-encoder reranker** re-scores all 30 → top 8 highest-relevance chunks →
   `gpt-4.1` answers grounded in context → answer + source list returned to UI

---

## Project Structure

```
application-docs-rag/
├── .env                  ← API key & config (never commit this)
├── requirements.txt      ← Python dependencies
├── main.py               ← FastAPI application (routes)
├── rag_engine.py         ← RAG chain: retriever + LLM
├── ingest.py             ← Document ingestion pipeline
├── static/
│   └── index.html        ← Chat UI (served by FastAPI)
├── docs/                 ← ← ← PUT YOUR HTML FILES HERE
│   └── ... your .html files (nested folders OK)
└── chroma_db/            ← Created automatically after first ingest
```

---

## Local Setup — Complete Step-by-Step Guide

This section covers **everything** a new developer needs to install and run this
application on their local machine from scratch, on **Windows**, **macOS**, or
**Linux**.

---

### STEP 1 — Install System Prerequisites

#### 1a. Python 3.12 (required)

> ⚠️ **Python 3.13+ / 3.14 are NOT supported** — several dependencies
> (`chromadb`, `pydantic-core`) do not yet have pre-built wheels for those
> versions and will fail to install.  Use **Python 3.12** only.

**Windows**
```powershell
# Option A — winget (recommended, no browser needed)
winget install Python.Python.3.12

# Option B — download the installer manually
# https://www.python.org/downloads/release/python-31210/
# Run the .exe, tick "Add Python to PATH", click Install Now
```

**macOS**
```bash
# Using Homebrew
brew install python@3.12

# Or download the .pkg from https://www.python.org/downloads/
```

**Linux (Ubuntu / Debian)**
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-pip
```

Verify the installation:
```bash
python3.12 --version
# Expected output: Python 3.12.x
```

---

#### 1b. Git (to clone the repository)

**Windows**
```powershell
winget install Git.Git
```

**macOS**
```bash
brew install git
```

**Linux**
```bash
sudo apt install -y git
```

Verify:
```bash
git --version
```

---

#### 1c. OpenAI API Key

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **Create new secret key** and copy it.
3. Make sure your OpenAI account has **billing enabled** (free tier has no API access).

> Keep your key safe — never share it or commit it to Git.

---

### STEP 2 — Get the Project Files

**Option A — Clone from Git**
```bash
git clone <your-repository-url>
cd application-docs-rag
```

**Option B — Copy the folder**

Copy the `application-docs-rag/` folder to your machine and open a terminal
inside it:
```bash
cd path/to/application-docs-rag
```

---

### STEP 3 — Create a Python Virtual Environment

A virtual environment keeps the project's dependencies isolated from your system Python.

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal prompt will show `(.venv)`.

> Run this activation command every time you open a new terminal for this project.

---

### STEP 4 — Install Python Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

This installs the following packages automatically:

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | ≥0.115 | Web framework — serves the API and UI |
| `uvicorn` | ≥0.34 | ASGI server — runs the FastAPI app |
| `langchain` | ≥0.3 | RAG orchestration framework |
| `langchain-classic` | ≥1.0 | LangChain chain utilities (retrieval, QA) |
| `langchain-openai` | ≥0.3 | OpenAI LLM + embedding integration |
| `langchain-community` | ≥0.3 | Community document loaders and tools |
| `langchain-chroma` | ≥1.0 | ChromaDB vector store integration |
| `chromadb` | ≥1.0 | Local vector database (no server needed) |
| `openai` | ≥1.59 | OpenAI Python SDK |
| `beautifulsoup4` | ≥4.12 | HTML parsing and text extraction |
| `lxml` | ≥5.3 | Fast HTML/XML parser for BeautifulSoup |
| `tiktoken` | ≥0.8 | Token counting (OpenAI tokenizer) |
| `python-dotenv` | ≥1.0 | Loads `.env` config file |
| `pydantic` | ≥2.10 | Data validation for API request/response |

Expected output ends with:
```
Successfully installed fastapi-x.x uvicorn-x.x langchain-x.x ...
```

If you see any `ERROR` lines, see the **Troubleshooting** section at the bottom.

---

### STEP 5 — Configure Environment Variables

Open the `.env` file in the project root and fill in your values:

```env
# ── Required ──────────────────────────────────────────────────
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── Model settings ────────────────────────────────────────────
CHAT_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-3-large

# ── RAG tuning ────────────────────────────────────────────────
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
RETRIEVER_TOP_K=15

# ── Reranker ──────────────────────────────────────────────────
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=8

# ── Paths ─────────────────────────────────────────────────────
DOCS_DIR=D:\Arc.help.AZN\Arc.help.AZN\mft   # path to your HTML docs folder
CHROMA_DB_DIR=./chroma_db
```

> Replace `DOCS_DIR` with the actual path to your `.html` documentation files.
> On macOS/Linux use forward slashes: `DOCS_DIR=/path/to/your/docs`

---

### STEP 6 — Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
[startup] No vector store found yet. Call POST /api/ingest first.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open your browser and go to: **[http://localhost:8000](http://localhost:8000)**

---

### STEP 7 — Index Your Documentation (First-Time Setup)

Before you can ask questions, the app must read and embed your HTML files.

**Option A — via the UI (recommended)**
1. Click the **⚡ Index Docs** button in the top toolbar.
2. Wait — this may take 1–5 minutes depending on how many files you have.
3. You will see: `✓ Indexed 540 file(s) → 4179 vectors`
4. The status badge in the top-right will change to **✓ Ready**.

**Option B — via the command line**
```bash
python ingest.py --reset
```

> You only need to do this once.  Re-run it if you update or add new HTML files.

---

### STEP 8 — Start Chatting!

Type any question about your Arc application in the chat box and press **Enter**.

Examples:
- *"How do I configure a REST connector?"*
- *"What are the steps to set up SFTP?"*
- *"How does the XML Map designer work?"*

---

### Stopping the Application

In the terminal where uvicorn is running, press **Ctrl + C**.

---

### Restarting After a Reboot

Every time you want to run the app again after restarting your machine:

```powershell
# Windows
cd path\to\application-docs-rag
.venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# macOS / Linux
cd path/to/application-docs-rag
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The documentation index (`chroma_db/`) is saved to disk, so you do **not** need
to re-index every time — only when the docs change.

---

## Configuration Reference (`.env`)

| Variable            | Default                              | Description                                                              |
|---------------------|--------------------------------------|--------------------------------------------------------------------------|
| `OPENAI_API_KEY`    | *(required)*                         | Your OpenAI secret key                                                   |
| `CHAT_MODEL`        | `gpt-4.1`                            | OpenAI chat model — 1M context, strong reasoning                        |
| `EMBEDDING_MODEL`   | `text-embedding-3-large`             | 3072-dim embeddings for high-accuracy semantic search                   |
| `DOCS_DIR`          | `./docs`                             | Path to your HTML documentation folder                                  |
| `CHROMA_DB_DIR`     | `./chroma_db`                        | Where ChromaDB persists the vector index                                |
| `CHUNK_SIZE`        | `1500`                               | Characters per text chunk (larger = more context preserved)             |
| `CHUNK_OVERLAP`     | `300`                                | Overlap between chunks (prevents mid-paragraph cuts)                   |
| `RETRIEVER_TOP_K`   | `15`                                 | MMR candidate pool size before reranking                                |
| `RERANKER_ENABLED`  | `true`                               | Toggle cross-encoder reranker on/off                                    |
| `RERANKER_MODEL`    | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder model (downloads once, ~22MB)              |
| `RERANKER_TOP_N`    | `8`                                  | Final chunks sent to LLM after reranking                                |

---

## API Endpoints

| Method | Path           | Description                                      |
|--------|----------------|--------------------------------------------------|
| GET    | `/`            | Serves the chat UI                               |
| GET    | `/api/status`  | Returns vector store readiness + document count |
| POST   | `/api/ingest`  | Ingests HTML docs and builds the vector index   |
| POST   | `/api/chat`    | Answers a question using the RAG pipeline       |

### POST /api/chat — request body

```json
{
  "question": "How do I configure authentication in Arc?",
  "history": [
    { "role": "user",      "content": "What is Arc?" },
    { "role": "assistant", "content": "Arc is …" }
  ]
}
```

### POST /api/ingest — request body

```json
{
  "docs_dir": "./docs",
  "reset": true
}
```

---

## Re-indexing after updating docs

Whenever you add or update HTML files in `docs/`, click **⚡ Index Docs** again
(or run `python ingest.py --reset`).  The `reset: true` flag wipes the old
index and rebuilds it from scratch.

---

## Dependencies

| Package                  | Purpose                                                         |
|--------------------------|-----------------------------------------------------------------|
| `fastapi`                | Web framework / REST API                                       |
| `uvicorn`                | ASGI server                                                     |
| `langchain`              | RAG orchestration (chains, splitters, prompts)                 |
| `langchain-openai`       | OpenAI LLM + embedding wrappers                                |
| `langchain-chroma`       | ChromaDB vector store integration                              |
| `langchain-community`    | HuggingFace cross-encoder and community retrievers             |
| `chromadb`               | Local vector database (no external service)                    |
| `openai`                 | OpenAI Python SDK                                              |
| `beautifulsoup4`         | HTML parsing / text extraction                                 |
| `lxml`                   | Fast HTML parser backend for BeautifulSoup                     |
| `tiktoken`               | Token counting for accurate chunking                           |
| `python-dotenv`          | `.env` file loader                                             |
| `sentence-transformers`  | Cross-encoder reranker model (`ms-marco-MiniLM-L-6-v2`)       |

---

## Troubleshooting

**"Not indexed" badge after starting the server**
→ The `chroma_db/` folder doesn't exist yet. Click **⚡ Index Docs** first.

**"No .html files found" error**
→ Check that `DOCS_DIR` in `.env` points to the correct folder path.

**OpenAI 401 Unauthorized error**
→ Your `OPENAI_API_KEY` in `.env` is wrong or missing. Also check that
your OpenAI account has billing enabled at [https://platform.openai.com/settings/billing](https://platform.openai.com/settings/billing).

**`error: metadata-generation-failed` / `pydantic-core` fails to build**
→ You are using Python 3.13 or 3.14. These versions are not yet supported.
Install **Python 3.12** exactly and recreate the virtual environment:
```powershell
winget install Python.Python.3.12
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**`chroma-hnswlib` fails to build / requires C++ compiler**
→ You are using an older `chromadb` version with Python 3.12.
The `requirements.txt` already pins `chromadb>=1.0.0` which ships a pre-built
wheel. Run `pip install -r requirements.txt --upgrade` to get it.

**`PermissionError: [WinError 32] The process cannot access the file`**
→ ChromaDB's SQLite file is locked from a previous run. Stop the server
(Ctrl+C), wait a few seconds, restart, and click Index Docs again.

**`Could not connect to tenant default_tenant`**
→ Caused by calling `_system.stop()` on the ChromaDB client. Already fixed
in the current codebase. Make sure you have the latest version of `rag_engine.py`.

**Slow first response after indexing**
→ Normal — the first query initialises the retriever. Subsequent queries
are much faster.

**Answers seem irrelevant or vague**
→ Try these tweaks in `.env`:
- Ensure `RERANKER_ENABLED=true` in `.env`
- Increase `RETRIEVER_TOP_K` (e.g. `20`) so the reranker has more candidates to pick from
- Re-index after any `CHUNK_SIZE` / `CHUNK_OVERLAP` changes

**Slow first response after starting the server**
→ On the very first query, the cross-encoder reranker model (`~22MB`) is loaded from the
  HuggingFace cache into memory. This takes 5-10 seconds once. All subsequent queries are fast.
  To pre-warm it, just send any question after startup.

**`ModuleNotFoundError: No module named 'langchain.retrievers'`**
→ Run `pip install -r requirements.txt` to ensure `sentence-transformers` and updated
  `langchain-community` are installed.
