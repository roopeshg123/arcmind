# ArcMind â€” AI Chat Assistant for Arc Documentation

Chat with your Arc application documentation using OpenAI GPT and a local vector search engine.
Ask questions in plain English and get accurate, sourced answers instantly.

---

## What's Inside

| File | Purpose |
|---|---|
| `main.py` | FastAPI web server â€” all API routes |
| `rag_engine.py` | RAG logic â€” retrieval chain + LLM |
| `ingest.py` | Ingestion pipeline â€” reads docs, builds vector store |
| `static/index.html` | Chat UI served by the backend |
| `.env` | Your config & secrets (never commit with real values) |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Run the app with one command |
| `chroma_db/` | Vector store â€” auto-created after first indexing |

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.12** | Backend language |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **LangChain** | RAG orchestration (chains, prompts, retrievers) |
| **ChromaDB** | Local vector database (saved to disk, no cloud needed) |
| **OpenAI API** | Embeddings (`text-embedding-3-large`) + Chat (`gpt-4.1`) |
| **BeautifulSoup4** | HTML parsing and text extraction |
| **sentence-transformers** | Cross-encoder reranker for better search accuracy |
| **requests** | HTTP client for web crawling docs |
| **Docker / Podman** | Container runtime for easy team deployment |

---

## Prerequisites

Before setting up, you need:

- **Python 3.12** â€” [python.org/downloads](https://www.python.org/downloads/) (3.13+ not supported)
- **Git** â€” [git-scm.com](https://git-scm.com)
- **OpenAI API Key** â€” [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (billing must be enabled)
- **Your docs folder** â€” the folder containing your `.html` documentation files

---

## Option A â€” Manual Setup (Local Python)

### Step 1 â€” Clone the repo

```powershell
git clone https://github.com/YOUR_USERNAME/arcmind.git
cd arcmind
```

### Step 2 â€” Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 â€” Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 â€” Create your `.env` file

Copy the example and fill in your values:

```powershell
Copy-Item .env.example .env
```

Then open `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-proj-...your-key-here...
DOCS_DIR=C:\path\to\your\html\docs
```

Full `.env` reference:

```env
# --- Required ---
OPENAI_API_KEY=sk-proj-...

# --- Models ---
CHAT_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-3-large

# --- Ingestion ---
# Use DOCS_DIR for a local folder of HTML files:
DOCS_DIR=C:\path\to\your\html\docs
# Use DOCS_URL instead to crawl a website (comment out DOCS_DIR):
# DOCS_URL=http://localhost:8081/

# --- Chunking ---
CHUNK_SIZE=1500
CHUNK_OVERLAP=300

# --- Retrieval ---
RETRIEVER_TOP_K=15
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=8

# --- Storage ---
CHROMA_DB_DIR=./chroma_db
```

### Step 5 â€” Start the app

```powershell
python main.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Step 6 â€” Index your docs

Click the **Index Docs** button in the UI (or call `POST /api/ingest`).  
This reads your docs, splits them into chunks, generates embeddings, and saves the vector store to `chroma_db/`.  
**Re-index any time your docs change.**

---

## Option B â€” Docker / Podman Setup

This is the easiest way to share the app with your team â€” no Python installation needed on their machines.

### Step 1 â€” Install Docker or Podman

- **Docker Desktop**: [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Podman Desktop** (free): [podman-desktop.io](https://podman-desktop.io) â€” use `podman` in place of `docker` everywhere

### Step 2 â€” Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/arcmind.git
cd arcmind
```

### Step 3 â€” Create your `.env` file

```bash
cp .env.example .env   # Linux/macOS
# or on Windows:
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` and `DOCS_URL` (Docker cannot mount local Windows paths easily â€” use a web URL or a local HTTP server):

```env
OPENAI_API_KEY=sk-proj-...your-key-here...
DOCS_URL=http://host.docker.internal:8081/
```

To serve your local docs folder over HTTP so the container can reach it:

```powershell
python -m http.server 8081 --directory "C:\path\to\your\html\docs"
```

### Step 4 â€” Build and run

```bash
docker compose up --build
```

The app starts at [http://localhost:8000](http://localhost:8000).

To run in the background:

```bash
docker compose up --build -d
```

To stop:

```bash
docker compose down
```

### Step 5 â€” Index your docs

Click **Index Docs** in the UI. The vector store is saved in a named Docker volume (`chroma_data`) so it survives container restarts.

### Useful commands

| Task | Command |
|---|---|
| View logs | `docker compose logs -f` |
| Restart app | `docker compose restart` |
| Stop and remove containers | `docker compose down` |
| Wipe vector store and start fresh | `docker compose down -v` |
| Rebuild after code changes | `docker compose up --build` |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the chat UI |
| `/api/status` | GET | Returns vector store health and document count |
| `/api/ingest` | POST | Triggers document ingestion / re-indexing |
| `/api/chat` | POST | Sends a question, returns an answer with sources |

### `/api/chat` request body

```json
{
  "question": "How do I configure SFTP in Arc MFT?"
}
```

### `/api/chat` response

```json
{
  "answer": "To configure SFTP in Arc MFT, navigate to...",
  "sources": [
    { "title": "SFTP Configuration", "url": "sftp-config.html" }
  ]
}
```

---

## How It Works

1. **Ingestion** â€” `ingest.py` reads your HTML docs (local folder or web crawl), strips tags, splits text into overlapping chunks, and generates vector embeddings via the OpenAI API. Embeddings are stored in ChromaDB on disk.

2. **Retrieval** â€” When you ask a question, `rag_engine.py` converts it to an embedding and does a Maximum Marginal Relevance (MMR) search to fetch the 15 most relevant chunks while avoiding redundancy.

3. **Reranking** â€” A cross-encoder model (`ms-marco-MiniLM-L-6-v2`) rescores the retrieved chunks for precision, keeping the top 8.

4. **Generation** â€” The top chunks are injected into a prompt and sent to `gpt-4.1`, which synthesises a grounded answer with source references.

---

## Configuration Quick Reference

> Re-index (`POST /api/ingest`) after changing any of these.

| Variable | Default | Effect |
|---|---|---|
| `CHAT_MODEL` | `gpt-4.1` | OpenAI model used for answers |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model (changing this requires full re-index) |
| `CHUNK_SIZE` | `1500` | Tokens per chunk â€” larger = more context per chunk |
| `CHUNK_OVERLAP` | `300` | Token overlap between chunks â€” helps preserve continuity |
| `RETRIEVER_TOP_K` | `15` | Chunks fetched before reranking |
| `RERANKER_ENABLED` | `true` | Toggle cross-encoder reranker on/off |
| `RERANKER_TOP_N` | `8` | Chunks passed to GPT after reranking |

---

## Pushing to GitHub

```powershell
git init                          # if not already a git repo
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/arcmind.git
git push -u origin main
```

> `.env` is in `.gitignore` and will **not** be pushed. Never commit your real API key.

---

## License

MIT
