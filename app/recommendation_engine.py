"""
AI Personal Intelligence Copilot
Recommendation Engine

Generates personalized recommendations across four domains:
  • Career        — job roles, skills to acquire, certifications
  • Learning      — courses, books, projects, YouTube channels
  • Productivity  — tools, habits, frameworks, time management
  • General       — context-driven ad-hoc suggestions

Strategy
--------
1. Rule-based triggers (fast, always-on)
2. LLM-powered deep recommendations (rich, personalized)
3. Context from conversation + user profile

Why a separate engine?
  Keeps the Copilot Engine thin. Recommendations are a distinct
  product feature with their own logic, categories, and UI panel.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation Templates (rule-based, instant, zero API cost)
# ─────────────────────────────────────────────────────────────────────────────

CAREER_RECS = [
    {
        "title": "Build a Production RAG System",
        "content": (
            "Implement a full RAG pipeline with FAISS/ChromaDB, document ingestion, "
            "and a FastAPI backend. Deploy to AWS/GCP. This is the #1 skill "
            "employers look for in AI Engineers right now."
        ),
        "priority": "high",
        "category": "career",
        "tags": ["AI/ML", "RAG", "Backend", "Portfolio"],
    },
    {
        "title": "Get the AWS ML Specialty Certification",
        "content": (
            "The AWS Certified Machine Learning Specialty is highly valued. "
            "Study time: 2-3 months. It covers feature engineering, model training, "
            "deployment, and monitoring — all critical for AI roles."
        ),
        "priority": "medium",
        "category": "career",
        "tags": ["Certification", "AWS", "Cloud"],
    },
    {
        "title": "Contribute to an Open-Source AI Project",
        "content": (
            "Pick one of: LangChain, LlamaIndex, Haystack, or Instructor. "
            "Fix bugs, improve docs, or add a feature. "
            "GitHub contributions are a strong signal to technical hiring managers."
        ),
        "priority": "medium",
        "category": "career",
        "tags": ["Open Source", "GitHub", "Community"],
    },
    {
        "title": "Write 3 Technical Articles on Medium/Substack",
        "content": (
            "Document your AI projects with clear explanations of your architecture decisions. "
            "Technical writing builds your personal brand and often leads to job opportunities. "
            "Aim for 1 article per month."
        ),
        "priority": "low",
        "category": "career",
        "tags": ["Writing", "Personal Brand", "Content"],
    },
]

LEARNING_RECS = [
    {
        "title": "Andrej Karpathy: Neural Networks Zero to Hero",
        "content": (
            "The best free resource to deeply understand LLMs from first principles. "
            "Covers backpropagation, transformers, GPT from scratch. "
            "YouTube playlist — ~10 hours total. Worth every minute."
        ),
        "priority": "high",
        "category": "learning",
        "tags": ["LLMs", "Deep Learning", "Free", "YouTube"],
    },
    {
        "title": "Fast.ai Practical Deep Learning",
        "content": (
            "Top-down, code-first approach. You build real models in lesson 1. "
            "Covers computer vision, NLP, and tabular data. "
            "Free at fast.ai — highly recommended by ML practitioners."
        ),
        "priority": "high",
        "category": "learning",
        "tags": ["Deep Learning", "FastAI", "Free", "Practical"],
    },
    {
        "title": "Chip Huyen's Designing ML Systems (Book)",
        "content": (
            "The definitive guide to productionizing ML. Covers data pipelines, "
            "model training, deployment, monitoring, and MLOps. "
            "Essential reading if you want to go beyond Jupyter notebooks."
        ),
        "priority": "medium",
        "category": "learning",
        "tags": ["MLOps", "Book", "Production ML"],
    },
    {
        "title": "LangChain Expression Language (LCEL) Masterclass",
        "content": (
            "LCEL is the modern way to build LangChain applications. "
            "Master it via the official docs and build 3 different chain types: "
            "RAG chain, SQL chain, and an agent with tools."
        ),
        "priority": "high",
        "category": "learning",
        "tags": ["LangChain", "LCEL", "Hands-on"],
    },
]

PRODUCTIVITY_RECS = [
    {
        "title": "Build a Personal Second Brain in Obsidian",
        "content": (
            "Use Obsidian with the Dataview + Templater plugins to create a "
            "connected knowledge base. Link notes, projects, and learnings. "
            "This dramatically reduces context-switching and improves recall."
        ),
        "priority": "high",
        "category": "productivity",
        "tags": ["Knowledge Management", "Obsidian", "PKM"],
    },
    {
        "title": "Implement Time Blocking for Deep Work",
        "content": (
            "Schedule two 90-minute deep work blocks per day (morning is ideal). "
            "Use Google Calendar with color coding. "
            "Turn off all notifications during deep work blocks. "
            "This is the single highest-ROI productivity habit."
        ),
        "priority": "high",
        "category": "productivity",
        "tags": ["Deep Work", "Time Management", "Habits"],
    },
    {
        "title": "Automate Your Weekly Review with Python",
        "content": (
            "Write a script that pulls your calendar events, GitHub commits, "
            "and task completions into a weekly summary. "
            "Review it every Sunday. Track progress on your OKRs automatically."
        ),
        "priority": "medium",
        "category": "productivity",
        "tags": ["Automation", "Python", "Review"],
    },
]

GENERAL_RECS = [
    {
        "title": "Set 90-Day OKRs (Objectives & Key Results)",
        "content": (
            "Define 1-3 objectives with 3-5 measurable key results each. "
            "Review weekly. This framework is used by Google, Intel, and top startups. "
            "Clarity on goals is the #1 predictor of achievement."
        ),
        "priority": "high",
        "category": "general",
        "tags": ["Goals", "OKRs", "Planning"],
    },
    {
        "title": "Build in Public on LinkedIn/X",
        "content": (
            "Share your learning journey, project updates, and insights weekly. "
            "Authenticity > polish. Consistency > virality. "
            "6 months of consistent sharing changes your career trajectory."
        ),
        "priority": "medium",
        "category": "general",
        "tags": ["Personal Brand", "Social Media", "Network"],
    },
]

ALL_RECOMMENDATIONS = {
    "career": CAREER_RECS,
    "learning": LEARNING_RECS,
    "productivity": PRODUCTIVITY_RECS,
    "general": GENERAL_RECS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Keyword → Category Mapping
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "career": ["job", "career", "work", "interview", "salary", "promotion", "resume", "cv",
                "hire", "hired", "opportunity", "role", "position", "skill", "portfolio"],
    "learning": ["learn", "study", "course", "book", "tutorial", "training", "education",
                  "certif", "degree", "master", "knowledge", "understand"],
    "productivity": ["productiv", "focus", "time", "task", "habit", "routine", "schedule",
                      "organiz", "system", "workflow", "automat", "efficient"],
    "general": ["goal", "plan", "strategy", "advice", "recommend", "suggest", "help",
                 "improve", "better", "grow"],
}


def detect_category(text: str) -> str:
    """Detect the most relevant recommendation category from input text."""
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Generates and ranks personalized recommendations.

    Two modes:
      1. Quick suggestions — from the template library, instant
      2. Deep recommendations — LLM-powered (requires API key)

    The engine uses the user profile (goals, skills, role) to filter
    and prioritize which templates are most relevant.
    """

    def __init__(self, llm=None):
        self._llm = llm   # Optional ChatOpenAI instance for deep recommendations

    def get_all_recommendations(
        self, category: Optional[str] = None, limit: int = 8
    ) -> List[Dict]:
        """
        Return template recommendations, optionally filtered by category.
        """
        if category and category in ALL_RECOMMENDATIONS:
            recs = ALL_RECOMMENDATIONS[category]
        else:
            # Mix categories
            recs = []
            for cat_recs in ALL_RECOMMENDATIONS.values():
                recs.extend(cat_recs)

        # Add IDs and timestamps
        result = []
        for rec in recs[:limit]:
            result.append({
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                **rec,
            })

        return result

    def get_personalized_recommendations(
        self,
        user_profile: Dict,
        category: Optional[str] = None,
        limit: int = 6,
    ) -> List[Dict]:
        """
        Score and rank recommendations based on user profile.
        Higher score = more relevant to the user's goals and skills.
        """
        all_recs = self.get_all_recommendations(category=category, limit=20)
        goals = " ".join(user_profile.get("goals", [])).lower()
        skills = " ".join(user_profile.get("skills", [])).lower()
        interests = " ".join(user_profile.get("interests", [])).lower()
        combined_profile = f"{goals} {skills} {interests}"

        scored = []
        for rec in all_recs:
            score = self._score_relevance(rec, combined_profile)
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:limit]]

    def _score_relevance(self, rec: Dict, profile_text: str) -> float:
        """Simple keyword overlap scoring."""
        if not profile_text.strip():
            return 0.5   # neutral

        rec_text = f"{rec['title']} {rec['content']} {' '.join(rec.get('tags', []))}".lower()
        profile_words = set(profile_text.split())
        rec_words = set(rec_text.split())
        overlap = profile_words & rec_words

        priority_boost = {"high": 0.3, "medium": 0.15, "low": 0.0}
        base_score = len(overlap) / max(len(rec_words), 1)
        return base_score + priority_boost.get(rec.get("priority", "low"), 0.0)

    def generate_quick_suggestions(
        self,
        user_input: str,
        response: str = "",
        user_profile: Optional[Dict] = None,
        limit: int = 3,
    ) -> List[Dict]:
        """
        Fast, always-on suggestions triggered after each chat message.
        These appear in the sidebar "Quick Actions" panel.
        """
        category = detect_category(user_input + " " + response)
        recs = ALL_RECOMMENDATIONS.get(category, ALL_RECOMMENDATIONS["general"])

        result = []
        for rec in recs[:limit]:
            result.append({
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "triggered_by": "chat",
                **rec,
            })
        return result

    def get_daily_recommendations(
        self, user_profile: Optional[Dict] = None
    ) -> Dict[str, List[Dict]]:
        """
        Return one high-priority recommendation per category.
        Used for the Recommendations Dashboard panel.
        """
        daily = {}
        for cat in ["career", "learning", "productivity", "general"]:
            recs = ALL_RECOMMENDATIONS[cat]
            high = [r for r in recs if r["priority"] == "high"]
            pick = high[0] if high else recs[0]
            daily[cat] = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                **pick,
            }
        return daily
