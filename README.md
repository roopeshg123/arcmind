# ArcMind — Enterprise AI Assistant for Arc

Chat with your Arc documentation **and** Jira ticket history using OpenAI GPT and a local hybrid search engine.
Ask questions in plain English and get accurate, sourced answers instantly — from both your docs and your real support/bug history.

---

## What's Inside

| Path | Purpose |
|---|---|
| `main.py` | FastAPI web server — all API routes, API-key auth middleware, CORS |
| `rag_engine.py` | Top-level RAG pipeline — orchestrates retrieval, reranking, and answer generation |
| `connectors/jira_client.py` | Jira REST API v3 client with cursor pagination |
| `ingest/ingest_docs.py` | HTML doc ingestion — web crawl or local folder, with `MAX_CRAWL_PAGES` cap |
| `ingest/ingest_jira.py` | Jira issue ingestion pipeline (full + incremental sync) |
| `ingest/chunking.py` | Token-aware text splitter (1500 tokens / 300 overlap) |
| `rag/retriever.py` | Hybrid BM25 + vector search with cross-encoder reranker |
| `rag/query_expander.py` | LLM-driven query expansion — generates 5 query variants |
| `rag/jira_clusterer.py` | Groups related Jira tickets for cleaner, deduplicated answers |
| `rag/prompt_builder.py` | Builds the final GPT prompt with doc + Jira context |
| `rag/conversation_memory.py` | Per-session conversation history manager |
| `rag/connector_detector.py` | Detects Arc connector/component references in queries |
| `rag/query_router.py` | Routes queries to docs-only, Jira-only, or hybrid retrieval |
| `rag/reranker.py` | Cross-encoder reranker wrapper (`ms-marco-MiniLM-L-6-v2`) |
| `vector_db/chroma_store.py` | ChromaDB + BM25 store — two collections, BM25 append-mode incremental sync |
| `static/index.html` | Chat UI — live progress bar for both doc and Jira indexing |
| `.env` | Local config — **never commit** (contains your API keys) |
| `Dockerfile` | Two-stage container build (builder + lean runtime, reranker pre-downloaded) |
| `docker-compose.yml` | Run the full app with one command |
| `chroma_db/` | Vector store — auto-created after first indexing |

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.12** | Backend language |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **LangChain** | Document splitting, prompt chains, LLM integration |
| **ChromaDB** | Local vector database (two collections: docs + jira) |
| **OpenAI API** | Embeddings (`text-embedding-3-large`) + Chat (`gpt-4.1`) |
| **rank-bm25** | BM25 keyword search (hybrid retrieval alongside vectors) |
| **sentence-transformers** | Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| **BeautifulSoup4 / lxml** | HTML parsing and text extraction |
| **requests** | HTTP client for web crawling and Jira API |
| **tiktoken** | Token counting for chunking |
| **Docker / Podman** | Container runtime for easy team deployment |

---

## Prerequisites

- **Python 3.12** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com)
- **OpenAI API Key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (billing must be enabled)
- **Your docs folder** — folder containing your `.html` documentation files
- *(Optional)* **Jira API token** — from [id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens)

---

## Option A — Manual Setup (Local Python)

### Step 1 — Clone the repo

```powershell
git clone https://github.com/YOUR_USERNAME/arcmind.git
cd arcmind
```

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Configure your `.env` file

The `.env` file is already in the repo with placeholder values. Just open it and fill in your real values:

```powershell
notepad .env
```

Minimum required:

```env
OPENAI_API_KEY=sk-proj-...your-key-here...
DOCS_DIR=C:\path\to\your\html\docs
```

To also enable Jira:

```env
JIRA_BASE_URL=https://your-org.atlassian.net
JIRA_EMAIL=you@yourcompany.com
JIRA_API_TOKEN=your-jira-api-token
JIRA_PROJECT_KEY=ARC
```

### Step 5 — Start the app

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

### Step 6 — Index your content

1. Click **⚡ Index Docs** to index your HTML documentation.
2. *(Optional)* Click **🔗 Index Jira** to index your Jira project.

The status badge shows live progress and final counts:
`✓ Ready — 20,479 vectors (docs: 2809, jira: 17670)`

---

## Option B — Docker / Podman Setup

No Python installation needed on team machines.

### Step 1 — Install Docker or Podman

- **Docker Desktop**: [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Podman Desktop** (free): [podman-desktop.io](https://podman-desktop.io)

### Step 2 — Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/arcmind.git
cd arcmind
notepad .env
```

### Step 3 — Build and run

```bash
docker compose up --build
```

App starts at [http://localhost:8000](http://localhost:8000).

Run in background: `docker compose up --build -d`

Stop: `docker compose down`

### Useful Docker commands

| Task | Command |
|---|---|
| View live logs | `docker compose logs -f` |
| Restart app | `docker compose restart` |
| Stop and remove containers | `docker compose down` |
| Wipe vector store (re-index from scratch) | `docker compose down -v` |
| Rebuild after code changes | `docker compose up --build` |

### Volumes

| Volume | Contents |
|---|---|
| `chroma_data` | ChromaDB vector store + BM25 index pickles — persists across restarts |
| `logs_data` | Query logs |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the chat UI |
| `/api/status` | GET | Returns vector counts (`docs_vectors`, `jira_vectors`, `total_vectors`) |
| `/api/ingest` | POST | Index / re-index HTML documentation |
| `/api/ingest/jira` | POST | Full index / re-index of Jira project |
| `/api/ingest/jira/sync` | POST | Incremental sync — fetches only issues updated in the last N hours |
| `/api/ingest/progress` | GET | Live progress (chunks done / total) during doc or Jira indexing |
| `/api/chat` | POST | Ask a question, get a full answer with sources (blocking) |
| `/api/chat/stream` | POST | Ask a question, get a streaming answer via Server-Sent Events (SSE) |

### `/api/chat` request

```json
{
  "question": "How do I configure SFTP in Arc MFT?",
  "history": []
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

### Indexing

**Docs** — HTML files are parsed by BeautifulSoup, split into 1500-token chunks with 300-token overlap, embedded with `text-embedding-3-large`, and stored in the `arcmind_docs` ChromaDB collection. A companion BM25 index is pickled alongside. A `MAX_CRAWL_PAGES` cap prevents unbounded web crawls. Live progress is streamed via `/api/ingest/progress` and shown in the UI progress bar.

**Jira** — All issues are fetched via Jira REST API v3 with cursor-based pagination (so all tickets are retrieved regardless of project size). Each issue is formatted as structured text (key, summary, type, status, priority, description, comments, fix versions, sprint) then chunked and embedded into the `arcmind_jira` collection. The BM25 index uses **append mode** during incremental syncs to avoid a full rebuild. Use `/api/ingest/jira/sync` for fast incremental updates.

### Querying

1. **Connector detection** — `connector_detector.py` identifies any Arc connector or component name in the query for targeted Jira filtering.
2. **Query routing** — `query_router.py` decides whether to search docs-only, Jira-only, or both collections.
3. **Query expansion** — The question is rewritten into 5 variants covering synonyms, acronym expansions, and related sub-topics.
4. **Hybrid retrieval** — Each variant is searched across both ChromaDB (semantic) and BM25 (keyword). Results are merged and deduplicated.
5. **Reranking** — A local cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores all candidates against the original question. Top `RERANKER_TOP_N` chunks are kept.
6. **Answer generation** — Top doc chunks and a clustered Jira summary are injected into a strict prompt and sent to `gpt-4.1`. Streaming is available via `/api/chat/stream`.

---

## Configuration Reference

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI secret key |
| `API_KEY` | *(unset)* | When set, all `/api/*` requests must include `X-API-Key: <value>`. Leave unset in dev. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Restrict in production (e.g. `https://yoursite.com`). |
| `CHAT_MODEL` | `gpt-4.1` | OpenAI model for answers |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `CHUNK_SIZE` | `1500` | Tokens per chunk |
| `CHUNK_OVERLAP` | `300` | Token overlap between chunks |
| `RETRIEVER_TOP_K` | `15` | Candidates fetched from each collection before reranking |
| `DOCS_TOP_K` | `8` | Doc chunks passed to GPT after reranking |
| `JIRA_TOP_K` | `6` | Jira chunks passed to GPT after reranking |
| `RERANKER_ENABLED` | `true` | Toggle cross-encoder reranker |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `RERANKER_TOP_N` | `8` | Total chunks passed to GPT after reranking |
| `MAX_CRAWL_PAGES` | `2000` | Max pages fetched during a web crawl |
| `CHROMA_DB_DIR` | `./chroma_db` | Vector store and BM25 pickle location |
| `JIRA_URL` | — | Atlassian Cloud URL (e.g. `https://your-org.atlassian.net`) |
| `JIRA_EMAIL` | — | Login email for Jira API auth |
| `JIRA_API_TOKEN` | — | API token from Atlassian account settings |
| `JIRA_PROJECT_KEY` | — | Jira project key to index (e.g. `ARCESB`) |

---

## License

MIT
