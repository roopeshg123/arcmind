# ─────────────────────────────────────────────
# Stage 1 — dependency installer (build layer)
# ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System libraries required by chromadb, lxml, onnxruntime, torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix folder
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2 — lean runtime image
# ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime-only system libraries (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY main.py rag_engine.py ingest.py ./
COPY static/ ./static/

# docs/ and chroma_db/ are intentionally NOT baked into the image —
# they are mounted as volumes at runtime so data persists across deploys.
RUN mkdir -p docs chroma_db

# Run as non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# FastAPI port
EXPOSE 8000

# Health check — hits the /api/status endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/status')" || exit 1

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
