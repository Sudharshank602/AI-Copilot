# AI Personal Intelligence Copilot
# Multi-stage Dockerfile
# Stage 1: base   — shared Python environment
# Stage 2: backend — FastAPI server
# Stage 3: frontend — Streamlit UI

# ── Base Stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create runtime directories
RUN mkdir -p data models/hf_cache data/faiss_index data/chroma_db


# ── Backend Stage ─────────────────────────────────────────────────────────────
FROM base AS backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]


# ── Frontend Stage ────────────────────────────────────────────────────────────
FROM base AS frontend

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false", \
     "--theme.base=dark"]
