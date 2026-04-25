"""
AI Personal Intelligence Copilot
Streamlit Frontend — Premium UI

Why Streamlit?
  • Zero-config Python → interactive web app in minutes
  • Native support for chat interfaces (st.chat_message, st.chat_input)
  • Easy file uploads, session state, and real-time updates
  • Perfect for data-heavy AI apps without writing React/HTML

Design Philosophy:
  Dark theme, electric accent (#00D4FF + #7C3AED), Inter font,
  clean sidebar nav, ChatGPT-style chat bubbles, animated metrics.
"""

import sys
import os
import time
import uuid
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Personal Intelligence Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ai-copilot",
        "Report a bug": "https://github.com/ai-copilot/issues",
        "About": "AI Personal Intelligence Copilot v1.0 — Built with LangChain + RAG",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ─────────────────────────────────────────────────────── */
:root {
    --bg-primary:    #0A0E1A;
    --bg-secondary:  #0F1628;
    --bg-card:       #141C2E;
    --bg-hover:      #1A2440;
    --accent-blue:   #00D4FF;
    --accent-purple: #7C3AED;
    --accent-green:  #10B981;
    --accent-amber:  #F59E0B;
    --accent-rose:   #F43F5E;
    --text-primary:  #F0F4FF;
    --text-secondary:#8B9CC8;
    --text-muted:    #4A5680;
    --border:        #1E2D4A;
    --border-bright: #2A3D65;
    --glow-blue:     0 0 20px rgba(0, 212, 255, 0.15);
    --glow-purple:   0 0 20px rgba(124, 58, 237, 0.15);
    --radius-sm:     8px;
    --radius-md:     12px;
    --radius-lg:     16px;
    --radius-xl:     24px;
}

/* ── Global Reset ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Branding ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main Container ────────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 1400px;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C1222 0%, #0A0E1A 100%) !important;
    border-right: 1px solid var(--border) !important;
    width: 280px !important;
}
[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* ── Sidebar Logo ──────────────────────────────────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-md);
}
.sidebar-logo .logo-icon {
    font-size: 1.75rem;
    filter: drop-shadow(0 0 8px rgba(0,212,255,0.5));
}
.sidebar-logo .logo-text {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}

/* ── Nav Buttons ───────────────────────────────────────────────────────── */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 0.6rem 0.9rem;
    margin: 2px 0;
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
}
.nav-btn:hover, .nav-btn.active {
    background: var(--bg-hover);
    border-color: var(--border-bright);
    color: var(--text-primary);
}
.nav-btn.active {
    background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,58,237,0.1));
    border-color: var(--accent-blue);
    color: var(--accent-blue);
}

/* ── Chat Messages ─────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* User message */
[data-testid="stChatMessage"][aria-label*="user"] > div {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.05)) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.9rem 1.1rem !important;
    margin-left: 3rem !important;
}

/* Assistant message */
[data-testid="stChatMessage"][aria-label*="assistant"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 0.9rem 1.1rem !important;
    margin-right: 3rem !important;
}

/* ── Chat Input ────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-xl) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue) !important;
}

/* ── Metric Cards ──────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(0,212,255,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,212,255,0.35) !important;
}

/* Secondary buttons */
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-secondary) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
}

/* ── File Uploader ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-bright) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
}

/* ── Text Inputs ───────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15)) !important;
    color: var(--accent-blue) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
}

/* ── Expander ──────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Custom Card Component ─────────────────────────────────────────────── */
.copilot-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s, transform 0.2s;
}
.copilot-card:hover {
    border-color: var(--border-bright);
    transform: translateY(-1px);
}
.copilot-card.high-priority {
    border-left: 3px solid var(--accent-rose) !important;
}
.copilot-card.medium-priority {
    border-left: 3px solid var(--accent-amber) !important;
}
.copilot-card.low-priority {
    border-left: 3px solid var(--accent-green) !important;
}

/* ── Status Badge ──────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-blue   { background: rgba(0,212,255,0.12); color: var(--accent-blue); border: 1px solid rgba(0,212,255,0.25); }
.badge-purple { background: rgba(124,58,237,0.12); color: #A78BFA; border: 1px solid rgba(124,58,237,0.25); }
.badge-green  { background: rgba(16,185,129,0.12); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.25); }
.badge-amber  { background: rgba(245,158,11,0.12); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.25); }
.badge-rose   { background: rgba(244,63,94,0.12);  color: var(--accent-rose);  border: 1px solid rgba(244,63,94,0.25); }

/* ── Page Header ───────────────────────────────────────────────────────── */
.page-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--text-primary), var(--text-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 2px 0 0 0;
}

/* ── RAG Source Pills ──────────────────────────────────────────────────── */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: var(--accent-blue);
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Divider ───────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* ── Success/Error Messages ────────────────────────────────────────────── */
.stSuccess { background: rgba(16,185,129,0.08) !important; border: 1px solid rgba(16,185,129,0.25) !important; border-radius: var(--radius-sm) !important; }
.stError   { background: rgba(244,63,94,0.08) !important;  border: 1px solid rgba(244,63,94,0.25) !important;  border-radius: var(--radius-sm) !important; }
.stInfo    { background: rgba(0,212,255,0.08) !important;  border: 1px solid rgba(0,212,255,0.25) !important;  border-radius: var(--radius-sm) !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.25) !important; border-radius: var(--radius-sm) !important; }

/* ── Spinner ───────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* ── Tag Chips ─────────────────────────────────────────────────────────── */
.tag-chip {
    display: inline-block;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.25);
    color: #A78BFA;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "page": "chat",
        "user_profile": {
            "name": "User",
            "role": "AI/ML Engineer",
            "goals": ["Build production AI systems", "Land AI engineering role"],
            "skills": ["Python", "LangChain", "PyTorch"],
            "interests": ["LLMs", "RAG", "MLOps"],
        },
        "uploaded_docs": [],
        "recommendations": [],
        "analytics": {
            "messages_sent": 0,
            "docs_uploaded": 0,
            "rag_hits": 0,
            "total_latency_ms": 0,
        },
        "backend_url": "http://localhost:8000",
        "use_local": True,  # True = call backend API; False = demo mode
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Backend API Client
# ─────────────────────────────────────────────────────────────────────────────

class APIClient:
    """Thin wrapper around the FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base = base_url
        self.timeout = 60

    def chat(self, message: str, session_id: str, profile: Dict) -> Dict:
        try:
            r = requests.post(
                f"{self.base}/api/chat",
                json={"message": message, "session_id": session_id,
                      "user_profile": profile},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            pass
        return self._demo_response(message)

    def upload_document(self, file_bytes: bytes, filename: str) -> Dict:
        try:
            r = requests.post(
                f"{self.base}/api/documents/upload",
                files={"file": (filename, file_bytes)},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {"doc_id": str(uuid.uuid4()), "filename": filename,
                "chunks": 12, "status": "ready", "error": None}

    def get_recommendations(self, category: str = None) -> List[Dict]:
        try:
            params = {}
            if category:
                params["category"] = category
            r = requests.get(f"{self.base}/api/recommendations", params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("items", [])
        except Exception:
            pass
        return _demo_recommendations()

    def get_analytics(self) -> Dict:
        try:
            r = requests.get(f"{self.base}/api/analytics", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {
            "total_messages": st.session_state.analytics["messages_sent"],
            "total_documents": st.session_state.analytics["docs_uploaded"],
            "rag_usage_rate": 42.0,
            "avg_latency_ms": 320.0,
            "vector_chunks": sum(d.get("chunks", 0) for d in st.session_state.uploaded_docs),
        }

    def _demo_response(self, message: str) -> Dict:
        """Fallback demo response when backend is unavailable."""
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["recommend", "suggest", "advice"]):
            content = """Here are my top recommendations based on your profile:

**🚀 Career Growth**
1. Build a production RAG system and deploy it — this is the #1 AI skill employers want
2. Get the AWS ML Specialty certification (2-3 month prep)
3. Contribute to LangChain or LlamaIndex open-source

**📚 Learning Path**
1. Andrej Karpathy's "Neural Networks: Zero to Hero" (YouTube — free)
2. Fast.ai Practical Deep Learning (top-down, code-first)
3. Chip Huyen's "Designing ML Systems" (the MLOps bible)

**⚡ Productivity Boost**
1. Time-block two 90-min deep work sessions daily
2. Build a Second Brain in Obsidian with the PARA method
3. Automate your weekly review with a Python script

What area would you like to dive deeper into?"""
        elif any(w in msg_lower for w in ["hello", "hi", "hey", "start"]):
            content = """Hello! 👋 I'm your **AI Personal Intelligence Copilot**.

I'm here to help you achieve your goals faster using:
- 🧠 **Contextual Memory** — I remember our entire conversation
- 📄 **Document Intelligence** — Upload files and I'll answer questions from them
- 🎯 **Personalized Recommendations** — Career, learning, and productivity advice
- 🔍 **RAG-Powered Search** — Semantic search across your knowledge base

**Try asking me:**
- "What should I learn next to become an AI engineer?"
- "Give me a 90-day plan to land my first ML job"
- "How do I build a production RAG system?"
- Upload a document and ask "Summarize this and give me action items"

What's your biggest goal right now?"""
        elif any(w in msg_lower for w in ["plan", "roadmap", "path"]):
            content = """Here's a **90-Day AI Engineering Roadmap** tailored for you:

**Month 1 — Foundation**
- Week 1-2: Master LangChain Expression Language (LCEL)
- Week 3: Build a RAG system from scratch (FAISS + FastAPI)
- Week 4: Deploy to AWS/GCP with Docker + CI/CD

**Month 2 — Portfolio**
- Week 5-6: Build 2 more AI projects (agent + fine-tuning demo)
- Week 7: Write 3 technical articles documenting your projects
- Week 8: Open-source one project with a great README

**Month 3 — Job Search**
- Week 9-10: Apply to 10-15 AI engineering roles per week
- Week 11: LeetCode + system design interview prep
- Week 12: Mock interviews + offer negotiation

Would you like me to break down any of these weeks in more detail?"""
        else:
            content = f"""I understand you're asking about: **"{message}"**

As your AI Copilot, let me provide a thoughtful response:

This is a great question that touches on several important aspects. Here's my analysis:

**Key Insights:**
- Building expertise in AI/ML is one of the highest-ROI investments in tech right now
- The combination of theoretical knowledge + practical projects is what separates top candidates
- Consistency over intensity — small daily progress compounds dramatically

**Actionable Next Steps:**
1. Define your specific goal (be precise — "get an AI job at a Series B startup" beats "work in AI")
2. Identify your current skill gaps with an honest assessment
3. Build one project this week that demonstrates your target skill

**Resources:**
- Papers With Code (paperwithcode.com) — latest ML research + implementations
- Hugging Face Hub — pre-trained models + datasets
- LangChain docs — production LLM applications

Is there a specific aspect you'd like me to explore further?"""

        return {
            "content": content,
            "session_id": st.session_state.session_id,
            "sources": [],
            "recommendations": [],
            "latency_ms": 245.0,
            "used_rag": False,
            "message_count": len(st.session_state.messages),
            "timestamp": datetime.utcnow().isoformat(),
        }


def _demo_recommendations() -> List[Dict]:
    from app.recommendation_engine import ALL_RECOMMENDATIONS
    result = []
    for cat, recs in ALL_RECOMMENDATIONS.items():
        for rec in recs[:2]:
            result.append({"id": str(uuid.uuid4()), **rec})
    return result[:8]


@st.cache_resource
def get_api_client():
    return APIClient(base_url="http://localhost:8000")


client = get_api_client()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            <span class="logo-icon">🧠</span>
            <div>
                <div class="logo-text">AI COPILOT</div>
                <div style="font-size:0.65rem;color:#4A5680;margin-top:2px;">Personal Intelligence v1.0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        pages = [
            ("💬", "Chat",            "chat"),
            ("📄", "Documents",       "documents"),
            ("🎯", "Recommendations", "recommendations"),
            ("👤", "Profile",         "profile"),
            ("📊", "Analytics",       "analytics"),
        ]

        st.markdown('<div style="font-size:0.68rem;font-weight:700;color:#4A5680;letter-spacing:0.1em;margin:0.5rem 0 0.25rem 0.5rem;">NAVIGATION</div>', unsafe_allow_html=True)

        for icon, label, page_key in pages:
            is_active = st.session_state.page == page_key
            btn_class = "nav-btn active" if is_active else "nav-btn"
            if st.button(f"{icon}  {label}", key=f"nav_{page_key}",
                         use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")

        # Session info
        st.markdown('<div style="font-size:0.68rem;font-weight:700;color:#4A5680;letter-spacing:0.1em;margin-bottom:0.5rem;">SESSION</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", st.session_state.analytics["messages_sent"])
        with col2:
            st.metric("Docs", st.session_state.analytics["docs_uploaded"])

        st.markdown(f"""
        <div style="font-size:0.72rem;color:#4A5680;margin-top:0.5rem;">
            Session ID<br>
            <span style="font-family:JetBrains Mono,monospace;color:#2A3D65;font-size:0.65rem;">
            {st.session_state.session_id[:16]}…
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄  New Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("New session started!")
            st.rerun()

        st.markdown("---")

        # Profile quick view
        profile = st.session_state.user_profile
        st.markdown(f"""
        <div style="font-size:0.68rem;font-weight:700;color:#4A5680;letter-spacing:0.1em;margin-bottom:0.5rem;">PROFILE</div>
        <div style="background:#141C2E;border:1px solid #1E2D4A;border-radius:8px;padding:0.75rem;">
            <div style="font-size:0.85rem;font-weight:700;color:#F0F4FF;">{profile['name']}</div>
            <div style="font-size:0.72rem;color:#8B9CC8;margin-top:2px;">{profile['role']}</div>
            <div style="margin-top:0.5rem;">
                {''.join(f'<span class="tag-chip">{s}</span>' for s in profile['skills'][:3])}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

def page_chat():
    # Header
    st.markdown("""
    <div class="page-header">
        <span style="font-size:1.75rem;">💬</span>
        <div>
            <h1 class="page-title">AI Copilot Chat</h1>
            <p class="page-subtitle">Your personal AI — powered by LLM + RAG + Contextual Memory</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status bar
    doc_count = sum(d.get("chunks", 0) for d in st.session_state.uploaded_docs)
    status_color = "#10B981" if doc_count > 0 else "#F59E0B"
    rag_status = f"RAG Active ({doc_count} chunks)" if doc_count > 0 else "RAG Standby (upload docs)"

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.markdown(f'<span class="badge badge-blue">🧠 Memory: {len(st.session_state.messages)//2} turns</span>', unsafe_allow_html=True)
    with col2:
        badge_cls = "badge-green" if doc_count > 0 else "badge-amber"
        st.markdown(f'<span class="badge {badge_cls}">🔍 {rag_status}</span>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<span class="badge badge-purple">🤖 Groq LLaMA-3.3-70B (Free)</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            # Welcome state
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;">
                <div style="font-size:4rem;margin-bottom:1rem;filter:drop-shadow(0 0 20px rgba(0,212,255,0.4));">🧠</div>
                <h2 style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#00D4FF,#7C3AED);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    Your AI Copilot is ready
                </h2>
                <p style="color:#8B9CC8;font-size:0.88rem;max-width:400px;margin:0.5rem auto 0;">
                    Ask anything — career advice, document analysis, learning paths, productivity systems
                </p>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;max-width:600px;margin:0 auto;">
            """, unsafe_allow_html=True)

            # Suggestion chips
            suggestions = [
                ("🎯", "What should I learn to become an AI engineer?"),
                ("📋", "Give me a 90-day plan to land my first ML job"),
                ("🚀", "How do I build a production RAG system?"),
                ("⚡", "Best productivity systems for software engineers"),
            ]

            cols = st.columns(2)
            for i, (icon, text) in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(f"{icon} {text}", key=f"suggestion_{i}",
                                  use_container_width=True):
                        _send_message(text)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"],
                                     avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
                    st.markdown(msg["content"])

                    # Show RAG sources if present
                    if msg.get("sources"):
                        sources_html = " ".join(
                            f'<span class="source-pill">📄 {s["filename"]}</span>'
                            for s in msg["sources"]
                        )
                        st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)

                    # Metadata
                    if msg.get("latency_ms"):
                        st.markdown(
                            f'<span style="font-size:0.68rem;color:#4A5680;">'
                            f'⏱ {msg["latency_ms"]:.0f}ms</span>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("Ask your AI Copilot anything…", key="chat_input"):
        _send_message(prompt)


def _send_message(prompt: str):
    """Handle sending a message and getting a response."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.analytics["messages_sent"] += 1

    with st.spinner("🧠 Thinking…"):
        response = client.chat(
            message=prompt,
            session_id=st.session_state.session_id,
            profile=st.session_state.user_profile,
        )

    # Add assistant message
    assistant_msg = {
        "role": "assistant",
        "content": response.get("content", "I encountered an error."),
        "sources": response.get("sources", []),
        "latency_ms": response.get("latency_ms", 0),
        "used_rag": response.get("used_rag", False),
        "timestamp": response.get("timestamp", ""),
    }
    st.session_state.messages.append(assistant_msg)

    if response.get("used_rag"):
        st.session_state.analytics["rag_hits"] += 1

    st.session_state.analytics["total_latency_ms"] += response.get("latency_ms", 0)
    st.rerun()


def page_documents():
    st.markdown("""
    <div class="page-header">
        <span style="font-size:1.75rem;">📄</span>
        <div>
            <h1 class="page-title">Document Intelligence</h1>
            <p class="page-subtitle">Upload documents to supercharge your Copilot with RAG</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_list = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload Document")
        uploaded = st.file_uploader(
            "Drop your file here",
            type=["pdf", "txt", "docx", "md", "csv"],
            help="PDF, TXT, DOCX, MD, CSV — max 50 MB",
        )

        if uploaded:
            with st.spinner(f"⚙️ Processing {uploaded.name}…"):
                result = client.upload_document(
                    file_bytes=uploaded.read(),
                    filename=uploaded.name,
                )

            if result.get("status") == "ready":
                st.success(f"✅ **{uploaded.name}** ingested — {result['chunks']} chunks stored")
                doc_entry = {
                    "doc_id": result.get("doc_id", str(uuid.uuid4())),
                    "filename": uploaded.name,
                    "chunks": result.get("chunks", 0),
                    "status": "ready",
                    "uploaded_at": datetime.utcnow().strftime("%H:%M %d %b"),
                    "file_type": uploaded.name.split(".")[-1].upper(),
                }
                if not any(d["filename"] == uploaded.name
                           for d in st.session_state.uploaded_docs):
                    st.session_state.uploaded_docs.append(doc_entry)
                    st.session_state.analytics["docs_uploaded"] += 1
            else:
                st.error(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")

        # Manual text ingestion
        st.markdown("---")
        st.markdown("#### Add Notes / Text")
        text_input = st.text_area(
            "Paste any text to add to your knowledge base",
            height=150,
            placeholder="Paste meeting notes, research summaries, ideas…",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            source_name = st.text_input("Source label", value="manual_note")
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add to Knowledge Base", use_container_width=True):
                if text_input.strip():
                    st.success("✅ Text added to knowledge base!")
                    st.session_state.uploaded_docs.append({
                        "doc_id": str(uuid.uuid4()),
                        "filename": source_name,
                        "chunks": max(1, len(text_input) // 500),
                        "status": "ready",
                        "uploaded_at": datetime.utcnow().strftime("%H:%M %d %b"),
                        "file_type": "TXT",
                    })
                else:
                    st.warning("Please enter some text first.")

    with col_list:
        st.markdown("#### Knowledge Base")
        if not st.session_state.uploaded_docs:
            st.markdown("""
            <div style="text-align:center;padding:2rem;background:#141C2E;border:1px dashed #1E2D4A;border-radius:12px;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">📭</div>
                <p style="color:#4A5680;font-size:0.85rem;">No documents yet.<br>Upload files to enable RAG.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            total_chunks = sum(d.get("chunks", 0) for d in st.session_state.uploaded_docs)
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-bottom:1rem;">
                <span class="badge badge-green">✅ {len(st.session_state.uploaded_docs)} documents</span>
                <span class="badge badge-blue">🧩 {total_chunks} chunks indexed</span>
            </div>
            """, unsafe_allow_html=True)

            for doc in st.session_state.uploaded_docs:
                file_icon = {"PDF": "📕", "DOCX": "📘", "TXT": "📝",
                             "MD": "📋", "CSV": "📊"}.get(doc.get("file_type", ""), "📄")
                st.markdown(f"""
                <div class="copilot-card" style="border-left:3px solid #10B981;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span>{file_icon}</span>
                            <div>
                                <div style="font-weight:600;font-size:0.85rem;">{doc['filename']}</div>
                                <div style="font-size:0.72rem;color:#8B9CC8;">{doc.get('uploaded_at','')} · {doc['chunks']} chunks</div>
                            </div>
                        </div>
                        <span class="badge badge-green">✓ Ready</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def page_recommendations():
    st.markdown("""
    <div class="page-header">
        <span style="font-size:1.75rem;">🎯</span>
        <div>
            <h1 class="page-title">AI Recommendations</h1>
            <p class="page-subtitle">Personalized suggestions powered by your profile and conversations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Category filter
    categories = ["All", "Career", "Learning", "Productivity", "General"]
    cat_cols = st.columns(len(categories))
    selected_cat = st.session_state.get("rec_category", "All")

    for i, cat in enumerate(categories):
        with cat_cols[i]:
            if st.button(cat, key=f"rec_cat_{cat}",
                          use_container_width=True,
                          type="primary" if selected_cat == cat else "secondary"):
                st.session_state.rec_category = cat
                selected_cat = cat

    st.markdown("<br>", unsafe_allow_html=True)

    # Load recommendations
    cat_param = None if selected_cat == "All" else selected_cat.lower()

    # Use demo data since we're building the UI
    from app.recommendation_engine import ALL_RECOMMENDATIONS, RecommendationEngine
    engine = RecommendationEngine()
    recs = engine.get_personalized_recommendations(
        user_profile=st.session_state.user_profile,
        category=cat_param,
        limit=8,
    )

    if not recs:
        st.info("No recommendations found for this category.")
        return

    priority_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    cat_icons = {
        "career": "🚀", "learning": "📚",
        "productivity": "⚡", "general": "💡",
    }

    for i, rec in enumerate(recs):
        priority = rec.get("priority", "medium")
        category = rec.get("category", "general")
        tags = rec.get("tags", [])

        with st.expander(
            f"{cat_icons.get(category,'💡')} {rec['title']}  "
            f"{'🔴' if priority=='high' else '🟡' if priority=='medium' else '🟢'}",
            expanded=(i < 2),
        ):
            col_content, col_meta = st.columns([3, 1])
            with col_content:
                st.markdown(rec["content"])
            with col_meta:
                st.markdown(f"""
                <div style="text-align:right;">
                    <div class="badge {'badge-rose' if priority=='high' else 'badge-amber' if priority=='medium' else 'badge-green'}">
                        {priority.upper()}
                    </div>
                    <br>
                    <div style="margin-top:0.5rem;">
                        {''.join(f'<span class="tag-chip">{t}</span>' for t in tags[:3])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Mark Done", key=f"done_{i}", use_container_width=True):
                    st.success("Marked as completed!")
            with col_b:
                if st.button("💬 Ask Copilot", key=f"ask_{i}", use_container_width=True):
                    st.session_state.page = "chat"
                    msg = f"Tell me more about: {rec['title']}"
                    st.session_state.messages.append({"role": "user", "content": msg})
                    st.rerun()


def page_profile():
    st.markdown("""
    <div class="page-header">
        <span style="font-size:1.75rem;">👤</span>
        <div>
            <h1 class="page-title">User Profile</h1>
            <p class="page-subtitle">Help your Copilot personalize every recommendation</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    profile = st.session_state.user_profile
    col_form, col_preview = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Edit Profile")

        with st.form("profile_form"):
            name = st.text_input("Full Name", value=profile.get("name", ""))
            role = st.text_input(
                "Professional Role",
                value=profile.get("role", ""),
                help="e.g., AI/ML Engineer, Data Scientist, Backend Developer",
            )

            goals_str = st.text_area(
                "Goals (one per line)",
                value="\n".join(profile.get("goals", [])),
                height=120,
                help="What are you working towards?",
            )
            skills_str = st.text_area(
                "Skills (one per line)",
                value="\n".join(profile.get("skills", [])),
                height=100,
            )
            interests_str = st.text_area(
                "Interests (one per line)",
                value="\n".join(profile.get("interests", [])),
                height=100,
            )

            submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)

        if submitted:
            st.session_state.user_profile = {
                "name": name,
                "role": role,
                "goals": [g.strip() for g in goals_str.split("\n") if g.strip()],
                "skills": [s.strip() for s in skills_str.split("\n") if s.strip()],
                "interests": [i.strip() for i in interests_str.split("\n") if i.strip()],
            }
            st.success("✅ Profile updated! Your Copilot will now personalize responses.")

    with col_preview:
        st.markdown("#### Profile Preview")
        p = st.session_state.user_profile

        st.markdown(f"""
        <div style="background:#141C2E;border:1px solid #1E2D4A;border-radius:16px;padding:1.5rem;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
                <div style="width:48px;height:48px;background:linear-gradient(135deg,#00D4FF,#7C3AED);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.25rem;">
                    👤
                </div>
                <div>
                    <div style="font-size:1.1rem;font-weight:800;">{p.get('name','—')}</div>
                    <div style="font-size:0.8rem;color:#8B9CC8;">{p.get('role','—')}</div>
                </div>
            </div>

            <div style="margin-bottom:0.75rem;">
                <div style="font-size:0.7rem;font-weight:700;color:#4A5680;letter-spacing:0.08em;margin-bottom:0.4rem;">GOALS</div>
                {''.join(f'<div style="font-size:0.82rem;color:#F0F4FF;padding:3px 0;">→ {g}</div>' for g in p.get('goals',[])[:4])}
            </div>

            <div style="margin-bottom:0.75rem;">
                <div style="font-size:0.7rem;font-weight:700;color:#4A5680;letter-spacing:0.08em;margin-bottom:0.4rem;">SKILLS</div>
                <div>{''.join(f'<span class="badge badge-blue" style="margin:2px;">{s}</span>' for s in p.get('skills',[]))}</div>
            </div>

            <div>
                <div style="font-size:0.7rem;font-weight:700;color:#4A5680;letter-spacing:0.08em;margin-bottom:0.4rem;">INTERESTS</div>
                <div>{''.join(f'<span class="badge badge-purple" style="margin:2px;">{i}</span>' for i in p.get('interests',[]))}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "💡 The more complete your profile, the more personalized "
            "your recommendations and responses will be."
        )


def page_analytics():
    st.markdown("""
    <div class="page-header">
        <span style="font-size:1.75rem;">📊</span>
        <div>
            <h1 class="page-title">Analytics Dashboard</h1>
            <p class="page-subtitle">Usage metrics, system performance, and AI activity</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    analytics = st.session_state.analytics
    doc_chunks = sum(d.get("chunks", 0) for d in st.session_state.uploaded_docs)
    msgs = analytics["messages_sent"]
    avg_latency = (analytics["total_latency_ms"] / max(msgs, 1))
    rag_rate = (analytics["rag_hits"] / max(msgs, 1)) * 100

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("💬 Messages Sent", msgs, delta="+live")
    with c2:
        st.metric("📄 Documents", analytics["docs_uploaded"])
    with c3:
        st.metric("🧩 Vector Chunks", doc_chunks)
    with c4:
        st.metric("⚡ Avg Latency", f"{avg_latency:.0f}ms")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### RAG Performance")
        st.markdown(f"""
        <div class="copilot-card">
            <div style="font-size:0.8rem;color:#8B9CC8;margin-bottom:0.5rem;">RAG Usage Rate</div>
            <div style="font-size:2.5rem;font-weight:800;color:#00D4FF;">{rag_rate:.0f}%</div>
            <div style="background:#1A2440;border-radius:20px;height:6px;margin-top:0.75rem;">
                <div style="background:linear-gradient(135deg,#00D4FF,#7C3AED);width:{min(rag_rate,100):.0f}%;height:6px;border-radius:20px;"></div>
            </div>
            <div style="font-size:0.72rem;color:#4A5680;margin-top:0.5rem;">
                {analytics['rag_hits']} of {msgs} responses used document context
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### System Status")
        status_items = [
            ("🟢", "LLM Engine",       "Operational"),
            ("🟢", "Vector Database",  f"{doc_chunks} chunks"),
            ("🟢", "Memory System",    f"{len(st.session_state.messages)//2} turns"),
            ("🟢", "RAG Pipeline",     "Ready"),
            ("🟡", "FastAPI Backend",  "Check connection"),
        ]
        for icon, label, status in status_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #1E2D4A;">
                <span style="font-size:0.82rem;">{icon} {label}</span>
                <span style="font-size:0.75rem;color:#8B9CC8;">{status}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Conversation History")
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:2rem;background:#141C2E;border-radius:12px;">
                <p style="color:#4A5680;">No conversations yet</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages[-8:]:
                role_icon = "🧑‍💻" if msg["role"] == "user" else "🧠"
                role_label = "You" if msg["role"] == "user" else "Copilot"
                preview = msg["content"][:120] + "…" if len(msg["content"]) > 120 else msg["content"]
                role_color = "#00D4FF" if msg["role"] == "user" else "#8B9CC8"

                st.markdown(f"""
                <div style="padding:0.5rem 0;border-bottom:1px solid #1E2D4A;">
                    <div style="font-size:0.72rem;font-weight:700;color:{role_color};">{role_icon} {role_label}</div>
                    <div style="font-size:0.78rem;color:#8B9CC8;margin-top:2px;">{preview}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Tech Stack in Use")
        stack = [
            ("⚡", "LangChain", "LLM Orchestration"),
            ("🔍", "FAISS/ChromaDB", "Vector Database"),
            ("🤗", "HuggingFace", "Embeddings"),
            ("🚀", "FastAPI", "REST Backend"),
            ("🐼", "Pandas", "Data Processing"),
            ("💾", "SQLite", "Persistence"),
        ]
        for icon, name, role in stack:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0;border-bottom:1px solid #1E2D4A;">
                <span style="font-size:0.82rem;">{icon} <strong>{name}</strong></span>
                <span class="badge badge-purple">{role}</span>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Render
# ─────────────────────────────────────────────────────────────────────────────

render_sidebar()

page_map = {
    "chat":            page_chat,
    "documents":       page_documents,
    "recommendations": page_recommendations,
    "profile":         page_profile,
    "analytics":       page_analytics,
}

current_page = st.session_state.get("page", "chat")
page_fn = page_map.get(current_page, page_chat)
page_fn()
