"""
AI Personal Intelligence Copilot
FastAPI Backend — 100% Free API Stack

LLM:        Groq (free tier) — LLaMA-3.3-70B, Mixtral, Gemma2
Embeddings: HuggingFace sentence-transformers (local, free)
Vector DB:  FAISS (local, free)
Storage:    SQLite (local, free)
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.copilot_engine import CopilotEngine
from app.rag_pipeline import DocumentIngestionPipeline, RAGPipeline
from app.recommendation_engine import RecommendationEngine
from memory.memory_manager import MemoryManager
from vector_db.vector_store import VectorStoreManager, EmbeddingEngine

app = FastAPI(
    title="AI Personal Intelligence Copilot",
    description=(
        "Free-tier AI Copilot — Groq LLaMA-3.3-70B + HuggingFace Embeddings + FAISS + RAG"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── App state singletons ───────────────────────────────────────────────────────

class AppState:
    embedding_engine: Optional[EmbeddingEngine] = None
    vector_store: Optional[VectorStoreManager] = None
    rag_pipeline: Optional[RAGPipeline] = None
    ingestion_pipeline: Optional[DocumentIngestionPipeline] = None
    memory_manager: Optional[MemoryManager] = None
    recommendation_engine: Optional[RecommendationEngine] = None
    copilot_engine: Optional[CopilotEngine] = None
    user_profiles: Dict[str, Dict] = {}
    analytics_events: List[Dict] = []


state = AppState()


@app.on_event("startup")
async def startup():
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path("./models/hf_cache").mkdir(parents=True, exist_ok=True)

    groq_key = os.environ.get("GROQ_API_KEY", settings.GROQ_API_KEY)

    # HuggingFace local embeddings — free, no API key
    state.embedding_engine = EmbeddingEngine(
        model_name=settings.HF_EMBEDDING_MODEL,
        cache_dir=settings.HF_CACHE_DIR,
    )

    state.vector_store = VectorStoreManager(
        store_type=settings.VECTOR_DB_TYPE,
        embedding_engine=state.embedding_engine,
        faiss_index_path=settings.FAISS_INDEX_PATH,
        chroma_dir=settings.CHROMA_DB_PATH,
    )

    state.rag_pipeline = RAGPipeline(
        vector_store=state.vector_store,
        top_k=settings.RAG_TOP_K,
        similarity_threshold=settings.RAG_SIMILARITY_THRESHOLD,
    )

    state.ingestion_pipeline = DocumentIngestionPipeline(
        vector_store=state.vector_store,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    state.memory_manager = MemoryManager(window_size=settings.MEMORY_WINDOW_SIZE)
    state.recommendation_engine = RecommendationEngine()

    # Groq LLM — free API
    state.copilot_engine = CopilotEngine(
        groq_api_key=groq_key,
        groq_model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        memory_manager=state.memory_manager,
        rag_pipeline=state.rag_pipeline,
        recommendation_engine=state.recommendation_engine,
    )

    mode = "Groq LLaMA-3.3-70B" if not state.copilot_engine.use_demo_mode else "Demo Mode"
    print(f"✅ AI Copilot Backend started — LLM: {mode}")
    print(f"   Embeddings: HuggingFace {settings.HF_EMBEDDING_MODEL} (FREE local)")
    print(f"   Vector DB:  {settings.VECTOR_DB_TYPE.upper()} (FREE local)")


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    user_id: Optional[str] = "default"
    user_profile: Optional[Dict] = None


class ChatResponse(BaseModel):
    content: str
    session_id: str
    sources: List[Dict] = []
    recommendations: List[Dict] = []
    latency_ms: float = 0.0
    used_rag: bool = False
    message_count: int = 0
    model_used: str = ""
    timestamp: str = ""


class UserProfileRequest(BaseModel):
    user_id: str = "default"
    name: str = "User"
    role: str = "professional"
    goals: List[str] = []
    skills: List[str] = []
    interests: List[str] = []


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    status: str
    error: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "llm": state.copilot_engine.active_model if state.copilot_engine else "none",
        "demo_mode": state.copilot_engine.use_demo_mode if state.copilot_engine else True,
        "embeddings": settings.HF_EMBEDDING_MODEL,
        "vector_docs": state.vector_store.total_documents() if state.vector_store else 0,
        "free_stack": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not state.copilot_engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    session_id = request.session_id or str(uuid.uuid4())
    profile = request.user_profile or state.user_profiles.get(request.user_id or "default", {})

    response = state.copilot_engine.chat(
        user_input=request.message,
        session_id=session_id,
        user_profile=profile,
    )

    _track("message_sent", {
        "user_id": request.user_id,
        "session_id": session_id,
        "length": len(request.message),
        "used_rag": response.used_rag,
        "latency_ms": response.latency_ms,
        "model": response.model_used,
    })

    return ChatResponse(
        content=response.content,
        session_id=response.session_id,
        sources=response.sources,
        recommendations=response.recommendations,
        latency_ms=response.latency_ms,
        used_rag=response.used_rag,
        message_count=response.message_count,
        model_used=response.model_used,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/api/chat/stream")
async def stream_chat(
    message: str = Query(...),
    session_id: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default="default"),
):
    """Server-Sent Events streaming — real token-by-token output from Groq."""
    if not state.copilot_engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    session_id = session_id or str(uuid.uuid4())
    profile = state.user_profiles.get(user_id or "default", {})

    def generate():
        try:
            for token in state.copilot_engine.stream_chat(
                user_input=message,
                session_id=session_id,
                user_profile=profile,
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(default="default"),
):
    if not state.ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not ready")

    allowed = {"pdf", "docx", "txt", "md", "text", "csv", "rst"}
    ext = Path(file.filename or "f.txt").suffix.lstrip(".").lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type: .{ext}. Allowed: {allowed}")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    result = state.ingestion_pipeline.ingest_file(
        file_bytes=content,
        filename=file.filename or "upload",
        user_id=user_id or "default",
    )

    _track("document_uploaded", {
        "user_id": user_id,
        "filename": file.filename,
        "chunks": result.get("chunks", 0),
        "kb": round(len(content) / 1024, 1),
    })

    return DocumentResponse(**result)


@app.post("/api/documents/text", response_model=DocumentResponse)
async def ingest_text(
    content: str = Query(..., min_length=10),
    source: str = Query(default="manual_input"),
    user_id: Optional[str] = Query(default="default"),
):
    result = state.ingestion_pipeline.ingest_text(
        text=content, source=source, user_id=user_id or "default"
    )
    return DocumentResponse(**result)


@app.get("/api/documents/count")
async def doc_count():
    total = state.vector_store.total_documents() if state.vector_store else 0
    return {"total_chunks": total, "embedding_model": settings.HF_EMBEDDING_MODEL}


@app.get("/api/recommendations")
async def get_recommendations(
    category: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default="default"),
    limit: int = Query(default=6, ge=1, le=20),
):
    profile = state.user_profiles.get(user_id or "default", {})
    recs = state.recommendation_engine.get_personalized_recommendations(
        user_profile=profile, category=category, limit=limit
    )
    return {"category": category or "all", "items": recs, "count": len(recs)}


@app.get("/api/recommendations/daily")
async def daily_recs(user_id: Optional[str] = Query(default="default")):
    profile = state.user_profiles.get(user_id or "default", {})
    return {"daily": state.recommendation_engine.get_daily_recommendations(profile)}


@app.get("/api/memory/sessions")
async def list_sessions():
    sessions = state.memory_manager.list_sessions() if state.memory_manager else []
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/api/memory/{session_id}")
async def get_session(session_id: str):
    memory = state.memory_manager.get_or_create_session(session_id=session_id)
    return {
        "session_id": session_id,
        "messages": memory.get_history(),
        "message_count": memory.message_count(),
    }


@app.delete("/api/memory/{session_id}")
async def clear_session(session_id: str):
    if state.memory_manager:
        state.memory_manager.drop_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/api/profile")
async def update_profile(req: UserProfileRequest):
    state.user_profiles[req.user_id] = {
        "user_id": req.user_id,
        "name": req.name,
        "role": req.role,
        "goals": req.goals,
        "skills": req.skills,
        "interests": req.interests,
        "updated_at": datetime.utcnow().isoformat(),
    }
    return {"status": "updated", "profile": state.user_profiles[req.user_id]}


@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    profile = state.user_profiles.get(user_id)
    return {"user_id": user_id, "exists": bool(profile), "profile": profile}


@app.get("/api/analytics")
async def analytics():
    events = state.analytics_events
    msgs = [e for e in events if e.get("type") == "message_sent"]
    docs = [e for e in events if e.get("type") == "document_uploaded"]
    rag = [e for e in msgs if e.get("data", {}).get("used_rag")]
    latencies = [e.get("data", {}).get("latency_ms", 0) for e in msgs]

    return {
        "total_messages": len(msgs),
        "total_documents": len(docs),
        "rag_usage_rate": round(len(rag) / max(len(msgs), 1) * 100, 1),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
        "vector_chunks": state.vector_store.total_documents() if state.vector_store else 0,
        "llm_model": state.copilot_engine.active_model if state.copilot_engine else "none",
        "embedding_model": settings.HF_EMBEDDING_MODEL,
        "free_stack": True,
    }


@app.get("/api/models")
async def list_models():
    """List all available free Groq models."""
    return {
        "current_model": settings.GROQ_MODEL,
        "available_models": [
            {"id": "llama-3.3-70b-versatile",      "name": "LLaMA 3.3 70B",    "context": "128k", "best_for": "General, reasoning, coding"},
            {"id": "llama-3.1-8b-instant",          "name": "LLaMA 3.1 8B",     "context": "128k", "best_for": "Fast responses"},
            {"id": "mixtral-8x7b-32768",            "name": "Mixtral 8x7B",     "context": "32k",  "best_for": "Long documents"},
            {"id": "gemma2-9b-it",                  "name": "Gemma 2 9B",       "context": "8k",   "best_for": "Instruction following"},
            {"id": "llama-3.2-11b-vision-preview",  "name": "LLaMA 3.2 11B Vision", "context": "128k", "best_for": "Multimodal"},
        ],
        "embeddings": {
            "model": settings.HF_EMBEDDING_MODEL,
            "type": "local",
            "cost": "FREE",
            "dimensions": 384,
        },
        "groq_free_tier": {
            "requests_per_day": 14400,
            "requests_per_minute": 30,
            "signup": "https://console.groq.com",
        },
    }


def _track(event_type: str, data: Dict):
    state.analytics_events.append({
        "id": str(uuid.uuid4()),
        "type": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, reload=True)
