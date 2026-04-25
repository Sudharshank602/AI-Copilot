"""
AI Personal Intelligence Copilot
Test Suite — Free Stack (Groq + HuggingFace + FAISS)

Run with:
  pytest tests/ -v
"""

import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Memory Manager Tests ────────────────────────────────────────────────────

class TestConversationMemory:
    def _make(self, window_size=5):
        from memory.memory_manager import ConversationMemory
        return ConversationMemory(window_size=window_size, session_id=str(uuid.uuid4()))

    def test_add_messages(self):
        m = self._make()
        m.add_user_message("Hello")
        m.add_ai_message("Hi!")
        assert m.message_count() == 2

    def test_sliding_window(self):
        m = self._make(window_size=2)
        for i in range(10):
            m.add_user_message(f"u{i}")
            m.add_ai_message(f"a{i}")
        assert len(m.get_window()) == 4   # 2 pairs × 2

    def test_clear(self):
        m = self._make()
        m.add_user_message("x")
        m.clear()
        assert m.message_count() == 0

    def test_json_roundtrip(self):
        m = self._make()
        m.add_user_message("Hello")
        m.add_ai_message("World")
        restored = type(m).from_json(m.to_json())
        assert restored.message_count() == 2
        assert restored.session_id == m.session_id

    def test_context_string(self):
        m = self._make()
        m.add_user_message("What is RAG?")
        m.add_ai_message("RAG is Retrieval-Augmented Generation.")
        ctx = m.get_context_string()
        assert "User:" in ctx and "Assistant:" in ctx and "RAG" in ctx

    def test_langchain_messages(self):
        m = self._make()
        m.add_user_message("Hello")
        m.add_ai_message("Hi!")
        msgs = m.get_langchain_messages()
        assert len(msgs) == 2
        # Role check
        assert hasattr(msgs[0], "content")


class TestMemoryManager:
    def test_create_session(self):
        from memory.memory_manager import MemoryManager
        mgr = MemoryManager()
        sid = str(uuid.uuid4())
        m = mgr.get_or_create_session(session_id=sid)
        assert m.session_id == sid

    def test_caching(self):
        from memory.memory_manager import MemoryManager
        mgr = MemoryManager()
        sid = str(uuid.uuid4())
        assert mgr.get_or_create_session(session_id=sid) is mgr.get_or_create_session(session_id=sid)

    def test_drop_session(self):
        from memory.memory_manager import MemoryManager
        mgr = MemoryManager()
        sid = str(uuid.uuid4())
        mgr.get_or_create_session(session_id=sid)
        mgr.drop_session(sid)
        assert sid not in mgr.list_sessions()

    def test_user_context_prompt(self):
        from memory.memory_manager import MemoryManager
        prompt = MemoryManager().get_user_context_prompt({
            "name": "Alice", "role": "ML Engineer",
            "goals": ["AI job"], "skills": ["Python"], "interests": ["LLMs"],
        })
        assert "Alice" in prompt and "ML Engineer" in prompt


# ── Recommendation Engine Tests ─────────────────────────────────────────────

class TestRecommendationEngine:
    def _eng(self):
        from app.recommendation_engine import RecommendationEngine
        return RecommendationEngine()

    def test_returns_list(self):
        assert isinstance(self._eng().get_all_recommendations(), list)

    def test_category_filter(self):
        for r in self._eng().get_all_recommendations(category="career"):
            assert r["category"] == "career"

    def test_required_fields(self):
        required = {"id", "title", "content", "priority", "category"}
        for r in self._eng().get_all_recommendations():
            assert required.issubset(set(r.keys()))

    def test_personalized(self):
        recs = self._eng().get_personalized_recommendations(
            user_profile={"goals": ["AI job"], "skills": ["python"], "interests": []},
            limit=4,
        )
        assert 1 <= len(recs) <= 4

    def test_quick_suggestions(self):
        s = self._eng().generate_quick_suggestions("career advice", limit=3)
        assert len(s) <= 3

    def test_daily_all_categories(self):
        daily = self._eng().get_daily_recommendations()
        for cat in ["career", "learning", "productivity", "general"]:
            assert cat in daily


class TestCategoryDetection:
    def test_career(self):
        from app.recommendation_engine import detect_category
        assert detect_category("help me with job interview") == "career"

    def test_learning(self):
        from app.recommendation_engine import detect_category
        assert detect_category("best courses for machine learning") == "learning"

    def test_productivity(self):
        from app.recommendation_engine import detect_category
        assert detect_category("improve focus and productivity") == "productivity"


# ── Utility Helpers Tests ────────────────────────────────────────────────────

class TestHelpers:
    def test_clean_text(self):
        from utils.helpers import clean_text
        assert "\n\n\n" not in clean_text("Hello\n\n\n\nWorld")

    def test_truncate(self):
        from utils.helpers import truncate_text
        assert len(truncate_text("A" * 500, 100)) <= 104

    def test_truncate_short_unchanged(self):
        from utils.helpers import truncate_text
        assert truncate_text("Short", 100) == "Short"

    def test_estimate_tokens(self):
        from utils.helpers import estimate_tokens
        assert estimate_tokens("Hello world") > 0

    def test_keywords(self):
        from utils.helpers import extract_keywords
        kws = extract_keywords("machine learning model GPU hardware training")
        assert isinstance(kws, list) and len(kws) > 0

    def test_unique_ids(self):
        from utils.helpers import generate_id
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_prefix(self):
        from utils.helpers import generate_id
        assert generate_id("doc-").startswith("doc-")

    def test_now_iso(self):
        from utils.helpers import now_iso
        assert "T" in now_iso()

    def test_session_stats(self):
        from utils.helpers import compute_session_stats
        msgs = [
            {"role": "user", "content": "Hello how are you"},
            {"role": "assistant", "content": "I am fine thank you"},
            {"role": "user", "content": "Great"},
        ]
        s = compute_session_stats(msgs)
        assert s["turns"] == 2 and s["user_words"] > 0


# ── Data Pipeline Tests ──────────────────────────────────────────────────────

class TestConversationAnalytics:
    MSGS = [
        {"role": "user",      "content": "What is RAG?",   "latency_ms": 0,   "used_rag": False},
        {"role": "assistant", "content": "RAG = Retrieval-Augmented Generation", "latency_ms": 320.0, "used_rag": True,  "sources": [{"filename": "docs.pdf"}]},
        {"role": "user",      "content": "More detail?",   "latency_ms": 0,   "used_rag": False},
        {"role": "assistant", "content": "RAG retrieves docs and augments the LLM prompt.", "latency_ms": 280.0, "used_rag": True},
    ]

    def test_summary_counts(self):
        from utils.data_pipeline import ConversationAnalytics
        s = ConversationAnalytics(self.MSGS).summary()
        assert s["total_messages"] == 4
        assert s["user_messages"] == 2
        assert s["rag_enhanced"] == 2
        assert s["rag_rate_pct"] == 100.0

    def test_empty(self):
        from utils.data_pipeline import ConversationAnalytics
        assert "error" in ConversationAnalytics([]).summary()

    def test_latency_trend(self):
        from utils.data_pipeline import ConversationAnalytics
        latencies = ConversationAnalytics(self.MSGS).get_latency_trend()
        assert len(latencies) == 2 and all(l > 0 for l in latencies)


class TestDocumentAnalytics:
    DOCS = [
        {"filename": "a.pdf",  "chunks": 42, "status": "ready", "file_type": "pdf"},
        {"filename": "b.txt",  "chunks": 8,  "status": "ready", "file_type": "txt"},
        {"filename": "c.docx", "chunks": 0,  "status": "error", "file_type": "docx"},
    ]

    def test_summary(self):
        from utils.data_pipeline import DocumentAnalytics
        s = DocumentAnalytics(self.DOCS).summary()
        assert s["total_documents"] == 3
        assert s["total_chunks"] == 50
        assert s["ready"] == 2
        assert s["errors"] == 1


# ── RAG Pipeline Tests ───────────────────────────────────────────────────────

class TestRAGPipeline:
    def _pipe(self):
        from app.rag_pipeline import RAGPipeline
        class MockStore:
            def total_documents(self): return 5
            def search(self, q, top_k=5, threshold=0.35):
                return [
                    ("Groq provides free LLaMA-3.3-70B API", {"filename": "groq.txt", "doc_id": "d1", "chunk_index": 0}, 0.85),
                    ("HuggingFace embeddings run locally for free", {"filename": "hf.txt", "doc_id": "d2", "chunk_index": 0}, 0.72),
                ]
        return RAGPipeline(vector_store=MockStore(), top_k=5, similarity_threshold=0.35)

    def test_retrieve(self):
        assert len(self._pipe().retrieve("free LLM API")) == 2

    def test_context_block(self):
        p = self._pipe()
        ctx = p.build_context_block(p.retrieve("test"))
        assert "Source 1:" in ctx

    def test_augmented_prompt(self):
        p = self._pipe()
        prompt = p.build_augmented_prompt("test", p.retrieve("test"))
        assert prompt and "RETRIEVED CONTEXT" in prompt

    def test_sources_deduplication(self):
        p = self._pipe()
        results = [
            ("t1", {"filename": "doc.pdf", "doc_id": "d1"}, 0.9),
            ("t2", {"filename": "doc.pdf", "doc_id": "d1"}, 0.8),
            ("t3", {"filename": "other.txt", "doc_id": "d2"}, 0.7),
        ]
        sources = p.get_sources(results)
        assert len([s["filename"] for s in sources]) == len({s["filename"] for s in sources})

    def test_no_results_prompt_is_none(self):
        from app.rag_pipeline import RAGPipeline
        class EmptyStore:
            def total_documents(self): return 0
            def search(self, *a, **k): return []
        p = RAGPipeline(vector_store=EmptyStore())
        assert p.build_augmented_prompt("anything", []) is None


# ── Vector Store Tests ───────────────────────────────────────────────────────

class TestDocumentChunk:
    def test_creation(self):
        from vector_db.vector_store import DocumentChunk
        c = DocumentChunk(text="Hello", metadata={"source": "test"})
        assert c.text == "Hello" and c.chunk_id

    def test_to_dict(self):
        from vector_db.vector_store import DocumentChunk
        d = DocumentChunk(text="X", metadata={}).to_dict()
        assert "text" in d and "metadata" in d and "chunk_id" in d


class TestEmbeddingEngine:
    def test_dimension_is_384(self):
        from vector_db.vector_store import EmbeddingEngine
        e = EmbeddingEngine()
        assert e.dimension == 384

    def test_embed_returns_list_of_lists(self):
        from vector_db.vector_store import EmbeddingEngine
        e = EmbeddingEngine()
        # Force random fallback (no HF model needed for unit test)
        e._model = None
        result = e.embed(["test text", "another text"])
        assert isinstance(result, list) and len(result) == 2
        assert isinstance(result[0], list)

    def test_embed_single(self):
        from vector_db.vector_store import EmbeddingEngine
        e = EmbeddingEngine()
        e._model = None
        vec = e.embed_single("hello world")
        assert isinstance(vec, list) and len(vec) == 384


# ── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_loads(self):
        from app.config import get_settings
        s = get_settings()
        assert s.APP_NAME == "AI Personal Intelligence Copilot"
        assert s.GROQ_MODEL == "llama-3.3-70b-versatile"
        assert s.HF_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.EMBEDDING_DIMENSION == 384

    def test_singleton(self):
        from app.config import get_settings
        assert get_settings() is get_settings()

    def test_no_openai_key(self):
        """Confirm there is no OpenAI dependency in config."""
        from app.config import get_settings
        s = get_settings()
        assert not hasattr(s, "OPENAI_API_KEY"), "OpenAI removed — use Groq"


# ── Groq Engine Tests (no real API call) ─────────────────────────────────────

class TestCopilotEngineDemo:
    """Test the engine in demo mode (no Groq key required)."""

    def _engine(self):
        from app.copilot_engine import CopilotEngine
        # No groq key → demo mode
        return CopilotEngine(groq_api_key="")

    def test_demo_mode_active(self):
        engine = self._engine()
        assert engine.use_demo_mode is True

    def test_chat_returns_response(self):
        engine = self._engine()
        response = engine.chat("Hello!")
        assert response.content and len(response.content) > 10

    def test_chat_has_session_id(self):
        engine = self._engine()
        r = engine.chat("test", session_id="s-001")
        assert r.session_id == "s-001"

    def test_multi_turn_memory(self):
        engine = self._engine()
        sid = str(uuid.uuid4())
        engine.chat("Hello", session_id=sid)
        engine.chat("How are you?", session_id=sid)
        r3 = engine.chat("What did I say first?", session_id=sid)
        # Message count should be 6 (3 pairs)
        assert r3.message_count == 6

    def test_stream_yields_tokens(self):
        engine = self._engine()
        tokens = list(engine.stream_chat("Hello", session_id=str(uuid.uuid4())))
        assert len(tokens) > 0
        full = "".join(tokens)
        assert len(full) > 10


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
