"""
AI Personal Intelligence Copilot
LLM Copilot Engine — Powered by Groq (FREE)

Why Groq?
  Groq provides a completely FREE API tier with:
  • LLaMA 3.3 70B — Meta's best open model
  • Mixtral 8x7B  — Great for long-context tasks
  • Gemma 2 9B    — Google's efficient model
  • 14,400 requests/day free, 30 req/min
  • Ultra-low latency (~200ms) via custom LPU hardware
  • Sign up at https://console.groq.com — no credit card

Why HuggingFace embeddings?
  sentence-transformers/all-MiniLM-L6-v2 runs 100% locally:
  • Zero API cost, zero rate limits
  • 384-dim vectors, excellent semantic quality
  • ~50ms/batch on CPU, <10ms on GPU

Architecture:
  User Input
    → Memory (ConversationMemory)
    → RAG (FAISS semantic search)
    → Groq LLaMA-3.3-70B
    → CopilotResponse
"""

from __future__ import annotations

import os
import time
import uuid
import json
from typing import Generator, List, Dict, Optional
from loguru import logger

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("groq package not installed. Run: pip install groq")

from app.rag_pipeline import RAGPipeline
from memory.memory_manager import MemoryManager, ConversationMemory
from app.recommendation_engine import RecommendationEngine


# ─────────────────────────────────────────────────────────────────────────────
# Groq LLM Client
# ─────────────────────────────────────────────────────────────────────────────

class GroqLLM:
    """
    Direct Groq API client using the official `groq` Python SDK.

    The Groq SDK uses the same interface as the OpenAI SDK, so
    migration from OpenAI → Groq is literally changing 2 lines.

    Free models available (2025):
      llama-3.3-70b-versatile    → best overall quality
      llama-3.1-8b-instant       → fastest (< 100ms)
      mixtral-8x7b-32768         → 32k context window
      gemma2-9b-it               → efficient, good reasoning
      llama-3.2-11b-vision-preview → multimodal
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        if not HAS_GROQ:
            raise ImportError("Install groq: pip install groq")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"GroqLLM: model={model}")

    def chat(self, messages: List[Dict]) -> str:
        """
        Send a list of {role, content} messages to Groq and return the reply.

        Groq message format is identical to OpenAI's:
          [{"role": "system", "content": "..."},
           {"role": "user",   "content": "..."},
           {"role": "assistant", "content": "..."}]
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """Streaming version — yields text chunks as they arrive from Groq."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            yield f"\n\n[Stream error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Demo LLM — works with NO API key at all
# ─────────────────────────────────────────────────────────────────────────────

class DemoLLM:
    """
    Fallback when no Groq key is set.
    Returns rich, realistic responses so the UI is fully usable
    during demos without any API calls.
    """

    RESPONSES = {
        "recommend": (
            "Based on your profile, here are my top recommendations:\n\n"
            "**🚀 Career Growth**\n"
            "1. Build a production RAG system (FAISS + Groq + FastAPI) — deploy it\n"
            "2. Get the Google Professional ML Engineer certification (free study materials)\n"
            "3. Contribute to LangChain or LlamaIndex open-source this month\n\n"
            "**📚 Learning Path (all free)**\n"
            "1. Andrej Karpathy's 'Neural Networks: Zero to Hero' — YouTube\n"
            "2. fast.ai Practical Deep Learning — fast.ai (free)\n"
            "3. Groq docs — groq.com/docs (free LLaMA 3 access)\n\n"
            "**⚡ Productivity**\n"
            "1. Two 90-minute deep work blocks daily — no exceptions\n"
            "2. Build a Second Brain in Obsidian (free app)\n"
            "3. Weekly review every Sunday — track OKRs"
        ),
        "document": (
            "I've analyzed the uploaded document. Here's what I found:\n\n"
            "**Key Themes:**\n"
            "• AI/ML architecture patterns and production deployment\n"
            "• RAG (Retrieval-Augmented Generation) as the core technique\n"
            "• Contextual memory and personalization systems\n\n"
            "**Actionable Insights:**\n"
            "1. The architecture is production-ready — add monitoring next\n"
            "2. Groq API gives you LLaMA 3.3 70B free — perfect for this system\n"
            "3. HuggingFace embeddings are free and run locally — no cost\n\n"
            "Ask me anything specific about the document!"
        ),
        "plan": (
            "Here's your **90-Day AI Engineering Roadmap** (all free tools):\n\n"
            "**Month 1 — Build Foundation**\n"
            "• Week 1-2: Master Groq API + LangChain (free tier, LLaMA 3.3 70B)\n"
            "• Week 3: Build RAG system with FAISS + HuggingFace embeddings\n"
            "• Week 4: Deploy to Hugging Face Spaces (free hosting)\n\n"
            "**Month 2 — Ship Projects**\n"
            "• Week 5-6: Build 2 AI apps using Groq + Streamlit\n"
            "• Week 7: Write 3 technical articles (Medium, free)\n"
            "• Week 8: Open-source everything on GitHub\n\n"
            "**Month 3 — Land the Role**\n"
            "• Week 9-10: Apply to 10-15 AI roles/week\n"
            "• Week 11-12: Interview prep + offers"
        ),
        "default": (
            "I'm your **AI Personal Intelligence Copilot** — powered by Groq's free LLaMA 3.3 70B.\n\n"
            "Here's what I can do for you:\n"
            "- 🧠 **Chat** with full conversation memory\n"
            "- 📄 **Document Q&A** — upload any PDF/TXT and ask questions (RAG)\n"
            "- 🎯 **Recommendations** — career, learning, productivity (all free resources)\n"
            "- 🔧 **Agent Tools** — calculator, skill gap analysis, goal tracking\n\n"
            "**Free tech stack running right now:**\n"
            "- LLM: Groq LLaMA-3.3-70B (free API)\n"
            "- Embeddings: HuggingFace all-MiniLM-L6-v2 (local, free)\n"
            "- Vector DB: FAISS (local, free)\n"
            "- Storage: SQLite (local, free)\n\n"
            "What would you like to work on today?"
        ),
    }

    def respond(self, user_input: str) -> str:
        u = user_input.lower()
        if any(w in u for w in ["recommend", "suggest", "advice", "should i", "what should"]):
            return self.RESPONSES["recommend"]
        if any(w in u for w in ["document", "pdf", "file", "upload", "analyze"]):
            return self.RESPONSES["document"]
        if any(w in u for w in ["plan", "roadmap", "path", "month", "week"]):
            return self.RESPONSES["plan"]
        return self.RESPONSES["default"]


# ─────────────────────────────────────────────────────────────────────────────
# Copilot Engine
# ─────────────────────────────────────────────────────────────────────────────

class CopilotEngine:
    """
    Central orchestrator — wires Memory + RAG + Groq LLM + Recommendations.

    Initialization priority:
      1. Groq API (free, recommended)
      2. Demo mode (no key, canned responses — for offline demos)
    """

    SYSTEM_PROMPT = """You are an elite AI Personal Intelligence Copilot.
Your purpose: be the most useful, personalized AI assistant the user has ever interacted with.

Core behaviors:
• Personalize every response to the user's goals, skills, and context
• Be proactive — suggest next steps, resources, and opportunities without being asked
• Cite document sources when you use retrieved context (RAG)
• Be concise AND comprehensive — every sentence must deliver value
• Use markdown formatting: headers, bullets, bold for clarity
• Maintain a professional, motivating, growth-oriented tone
• ALWAYS recommend free resources (Groq, HuggingFace, fast.ai, YouTube, GitHub, etc.)

You have access to:
1. Full conversation history (you remember everything we've discussed)
2. User's uploaded documents (retrieved via RAG when relevant)
3. User's profile: goals, skills, interests, role"""

    def __init__(
        self,
        groq_api_key: str = "",
        groq_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        memory_manager: Optional[MemoryManager] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
    ):
        self.memory_manager = memory_manager or MemoryManager()
        self.rag_pipeline = rag_pipeline
        self.recommendation_engine = recommendation_engine
        self._demo_llm = DemoLLM()
        self._groq: Optional[GroqLLM] = None
        self.use_demo_mode = True
        self.active_model = "Demo Mode"

        # Try to initialize Groq
        key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        if key and key not in ("", "your_groq_api_key_here") and HAS_GROQ:
            try:
                self._groq = GroqLLM(
                    api_key=key,
                    model=groq_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.use_demo_mode = False
                self.active_model = f"Groq/{groq_model}"
                logger.info(f"CopilotEngine: Using {self.active_model}")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}. Falling back to demo mode.")

        if self.use_demo_mode:
            logger.info("CopilotEngine: DEMO MODE (set GROQ_API_KEY for live LLM)")

    # ── Core chat ─────────────────────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        session_id: str = "",
        user_profile: Optional[Dict] = None,
    ) -> "CopilotResponse":
        start_ms = time.time() * 1000
        session_id = session_id or str(uuid.uuid4())
        user_profile = user_profile or {}

        # 1. Memory
        memory = self.memory_manager.get_or_create_session(session_id=session_id)
        memory.add_user_message(user_input)

        # 2. RAG retrieval
        rag_results, rag_block, sources = [], None, []
        if self.rag_pipeline:
            rag_results = self.rag_pipeline.retrieve(user_input)
            if rag_results:
                rag_block = self.rag_pipeline.build_augmented_prompt(user_input, rag_results)
                sources = self.rag_pipeline.get_sources(rag_results)
                logger.info(f"RAG: {len(rag_results)} chunks from {len(sources)} sources")

        # 3. Build system message
        system_msg = self.SYSTEM_PROMPT
        if user_profile:
            system_msg += "\n\n" + self._build_profile_context(user_profile)
        if rag_block:
            system_msg += "\n\n" + rag_block

        # 4. Generate
        if self.use_demo_mode:
            response_text = self._demo_llm.respond(user_input)
        else:
            response_text = self._call_groq(system_msg, memory, user_input)

        # 5. Store
        memory.add_ai_message(response_text)

        # 6. Recommendations
        recs = []
        if self.recommendation_engine:
            recs = self.recommendation_engine.generate_quick_suggestions(
                user_input=user_input,
                response=response_text,
                user_profile=user_profile,
            )

        latency_ms = round(time.time() * 1000 - start_ms, 1)
        return CopilotResponse(
            content=response_text,
            session_id=session_id,
            sources=sources,
            recommendations=recs,
            latency_ms=latency_ms,
            used_rag=bool(rag_results),
            message_count=memory.message_count(),
            model_used=self.active_model,
        )

    def stream_chat(
        self,
        user_input: str,
        session_id: str = "",
        user_profile: Optional[Dict] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming chat — yields tokens as they arrive from Groq.
        Falls back to word-by-word simulation in demo mode.
        """
        session_id = session_id or str(uuid.uuid4())
        user_profile = user_profile or {}

        memory = self.memory_manager.get_or_create_session(session_id=session_id)
        memory.add_user_message(user_input)

        rag_results, rag_block, sources = [], None, []
        if self.rag_pipeline:
            rag_results = self.rag_pipeline.retrieve(user_input)
            if rag_results:
                rag_block = self.rag_pipeline.build_augmented_prompt(user_input, rag_results)

        system_msg = self.SYSTEM_PROMPT
        if user_profile:
            system_msg += "\n\n" + self._build_profile_context(user_profile)
        if rag_block:
            system_msg += "\n\n" + rag_block

        if self.use_demo_mode or self._groq is None:
            text = self._demo_llm.respond(user_input)
            memory.add_ai_message(text)
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.015)
        else:
            messages = self._build_messages(system_msg, memory, user_input)
            full_text = ""
            for chunk in self._groq.stream(messages):
                full_text += chunk
                yield chunk
            memory.add_ai_message(full_text)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_groq(self, system_msg: str, memory: ConversationMemory, user_input: str) -> str:
        messages = self._build_messages(system_msg, memory, user_input)
        try:
            return self._groq.chat(messages)
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            return (
                f"⚠️ Groq API error: {e}\n\n"
                "Please check:\n"
                "1. Your `GROQ_API_KEY` in `.env`\n"
                "2. You haven't exceeded the free rate limit (30 req/min)\n"
                "3. Your internet connection\n\n"
                "Get a free key at: https://console.groq.com"
            )

    def _build_messages(
        self, system_msg: str, memory: ConversationMemory, user_input: str
    ) -> List[Dict]:
        """Build the Groq messages array from system + history + latest input."""
        messages = [{"role": "system", "content": system_msg}]

        # Add conversation history from sliding window (exclude latest user msg)
        history = memory.get_window()[:-1]  # last item is the user msg we just added
        for turn in history:
            role = turn["role"]
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": turn["content"]})

        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_profile_context(self, profile: Dict) -> str:
        name = profile.get("name", "the user")
        role = profile.get("role", "professional")
        goals = ", ".join(profile.get("goals", [])) or "not specified"
        skills = ", ".join(profile.get("skills", [])) or "not specified"
        interests = ", ".join(profile.get("interests", [])) or "not specified"
        return (
            f"USER PROFILE:\n"
            f"- Name: {name}\n"
            f"- Role: {role}\n"
            f"- Goals: {goals}\n"
            f"- Skills: {skills}\n"
            f"- Interests: {interests}\n\n"
            f"Always tailor your responses directly to {name}'s goals and context."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Response Model
# ─────────────────────────────────────────────────────────────────────────────

class CopilotResponse:
    def __init__(
        self,
        content: str,
        session_id: str = "",
        sources: Optional[List[Dict]] = None,
        recommendations: Optional[List[Dict]] = None,
        latency_ms: float = 0.0,
        used_rag: bool = False,
        message_count: int = 0,
        model_used: str = "Demo",
    ):
        self.content = content
        self.session_id = session_id
        self.sources = sources or []
        self.recommendations = recommendations or []
        self.latency_ms = latency_ms
        self.used_rag = used_rag
        self.message_count = message_count
        self.model_used = model_used

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "session_id": self.session_id,
            "sources": self.sources,
            "recommendations": self.recommendations,
            "latency_ms": self.latency_ms,
            "used_rag": self.used_rag,
            "message_count": self.message_count,
            "model_used": self.model_used,
        }
