"""
AI Personal Intelligence Copilot
Personalization Engine

The Personalization Engine makes the Copilot feel like it was
built specifically for ONE user. It does this by:

1. Building a rich User Profile from explicit inputs + inferred signals
2. Scoring and ranking content by profile relevance
3. Adapting the LLM system prompt per user
4. Tracking interaction patterns to improve over time

Architecture:
  ┌─────────────────────────────────────────────────┐
  │              PersonalizationEngine               │
  │  ┌──────────────┐  ┌────────────────────────┐   │
  │  │  UserProfile  │  │  ContentScorer          │   │
  │  │  (data model) │  │  (relevance ranking)    │   │
  │  └──────────────┘  └────────────────────────┘   │
  │  ┌──────────────────────────────────────────┐   │
  │  │  PromptPersonalizer (system prompt builder)│   │
  │  └──────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# User Profile Data Model
# ─────────────────────────────────────────────────────────────────────────────

class UserProfile:
    """
    Rich user profile used for personalization.

    Explicit fields (user-provided):
      name, role, goals, skills, interests, experience_level

    Inferred fields (derived from conversations):
      inferred_topics, communication_style, response_length_pref

    Computed fields:
      profile_completeness_score (0-100)
      personalization_level ('basic'|'standard'|'deep')
    """

    EXPERIENCE_LEVELS = ["student", "junior", "mid-level", "senior", "lead", "principal"]
    COMMUNICATION_STYLES = ["concise", "detailed", "bullet-heavy", "conversational", "technical"]

    def __init__(
        self,
        user_id: str = "",
        name: str = "User",
        role: str = "Professional",
        goals: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        experience_level: str = "mid-level",
        communication_style: str = "detailed",
        industry: str = "Technology",
        location: str = "",
    ):
        self.user_id = user_id or str(uuid.uuid4())
        self.name = name
        self.role = role
        self.goals = goals or []
        self.skills = skills or []
        self.interests = interests or []
        self.experience_level = experience_level
        self.communication_style = communication_style
        self.industry = industry
        self.location = location

        # Inferred from conversations
        self.inferred_topics: List[str] = []
        self.message_count: int = 0
        self.rag_usage_count: int = 0
        self.preferred_response_length: str = "medium"  # short|medium|long
        self.interaction_history: List[Dict] = []

        # Metadata
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()

    # ── Completeness Scoring ──────────────────────────────────────────────────

    @property
    def completeness_score(self) -> int:
        """0-100 score for how complete the profile is."""
        score = 0
        if self.name and self.name != "User":   score += 10
        if self.role and self.role != "Professional": score += 15
        if self.goals:                           score += 20 + min(10, len(self.goals) * 3)
        if self.skills:                          score += 15 + min(10, len(self.skills) * 2)
        if self.interests:                       score += 10
        if self.experience_level:                score += 5
        if self.industry and self.industry != "Technology": score += 5
        return min(100, score)

    @property
    def personalization_level(self) -> str:
        score = self.completeness_score
        if score >= 70: return "deep"
        if score >= 40: return "standard"
        return "basic"

    # ── Interaction Tracking ──────────────────────────────────────────────────

    def record_interaction(self, query: str, response_length: int) -> None:
        """Update inferred signals from a conversation turn."""
        self.message_count += 1
        self.updated_at = datetime.utcnow().isoformat()

        # Infer preferred length from response engagement patterns
        if response_length > 1500:
            self._update_length_pref("long")
        elif response_length > 500:
            self._update_length_pref("medium")
        else:
            self._update_length_pref("short")

        # Extract topics from query keywords
        self._extract_topics(query)

    def _update_length_pref(self, new_pref: str) -> None:
        """Weighted running update of response length preference."""
        prefs = {"short": 0, "medium": 1, "long": 2}
        current = prefs.get(self.preferred_response_length, 1)
        new = prefs.get(new_pref, 1)
        # Exponential moving average (weight: 0.3 new, 0.7 existing)
        blended = round(0.7 * current + 0.3 * new)
        self.preferred_response_length = ["short", "medium", "long"][blended]

    def _extract_topics(self, text: str) -> None:
        """Simple keyword topic extraction from conversation."""
        topic_keywords = {
            "machine learning": ["neural", "model", "training", "pytorch", "tensorflow", "sklearn"],
            "career": ["job", "interview", "salary", "promotion", "career", "hire"],
            "productivity": ["focus", "time", "habit", "routine", "schedule", "productivity"],
            "learning": ["learn", "course", "study", "book", "tutorial", "master"],
            "rag": ["rag", "retrieval", "vector", "embedding", "faiss", "chroma"],
            "python": ["python", "pandas", "numpy", "fastapi", "django", "flask"],
            "devops": ["docker", "kubernetes", "ci/cd", "deploy", "aws", "gcp"],
        }
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                if topic not in self.inferred_topics:
                    self.inferred_topics.append(topic)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role,
            "goals": self.goals,
            "skills": self.skills,
            "interests": self.interests,
            "experience_level": self.experience_level,
            "communication_style": self.communication_style,
            "industry": self.industry,
            "location": self.location,
            "inferred_topics": self.inferred_topics,
            "message_count": self.message_count,
            "preferred_response_length": self.preferred_response_length,
            "completeness_score": self.completeness_score,
            "personalization_level": self.personalization_level,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        p = cls(
            user_id=data.get("user_id", ""),
            name=data.get("name", "User"),
            role=data.get("role", "Professional"),
            goals=data.get("goals", []),
            skills=data.get("skills", []),
            interests=data.get("interests", []),
            experience_level=data.get("experience_level", "mid-level"),
            communication_style=data.get("communication_style", "detailed"),
            industry=data.get("industry", "Technology"),
            location=data.get("location", ""),
        )
        p.inferred_topics = data.get("inferred_topics", [])
        p.message_count = data.get("message_count", 0)
        p.preferred_response_length = data.get("preferred_response_length", "medium")
        return p

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Personalizer
# ─────────────────────────────────────────────────────────────────────────────

class PromptPersonalizer:
    """
    Builds highly personalized system prompts from a UserProfile.

    The more complete the profile, the more specific and useful
    the system prompt — directly improving response quality.
    """

    LENGTH_INSTRUCTIONS = {
        "short":  "Keep responses concise — 2-3 bullet points or 1-2 short paragraphs max.",
        "medium": "Provide balanced responses — enough detail to be actionable, not overwhelming.",
        "long":   "Give comprehensive responses — the user appreciates depth, examples, and context.",
    }

    STYLE_INSTRUCTIONS = {
        "concise":       "Be direct. No fluff. Lead with the answer, then support.",
        "detailed":      "Provide thorough explanations with reasoning and context.",
        "bullet-heavy":  "Use bullet points and numbered lists extensively for clarity.",
        "conversational":"Be warm, friendly, and conversational. Use 'you' and 'I' naturally.",
        "technical":     "Use technical terminology. Assume a high level of technical background.",
    }

    EXPERIENCE_INSTRUCTIONS = {
        "student":    "Explain concepts from first principles. Use analogies. Avoid jargon.",
        "junior":     "Explain the 'why' behind recommendations. Include learning resources.",
        "mid-level":  "Assume solid fundamentals. Focus on best practices and architecture.",
        "senior":     "Skip basics. Focus on trade-offs, edge cases, and production concerns.",
        "lead":       "Include team/org impact. Consider stakeholder management and scaling.",
        "principal":  "Strategic perspective. Long-term architecture. Cross-team implications.",
    }

    def build_system_prompt(self, profile: UserProfile) -> str:
        """
        Build a personalized system prompt from the user's profile.
        The prompt is injected at the start of every LLM call.
        """
        level = profile.personalization_level
        lines = [self._build_identity_block(profile)]

        if level in ("standard", "deep"):
            lines.append(self._build_context_block(profile))

        if level == "deep":
            lines.append(self._build_behavioral_block(profile))

        lines.append(self._build_instructions_block(profile))

        return "\n\n".join(lines)

    def _build_identity_block(self, p: UserProfile) -> str:
        return (
            f"You are the AI Personal Intelligence Copilot for {p.name}.\n"
            f"Your purpose: be the most useful, personalized AI assistant {p.name} has ever used."
        )

    def _build_context_block(self, p: UserProfile) -> str:
        parts = [f"USER CONTEXT — {p.name}:"]

        if p.role:
            parts.append(f"• Role: {p.role} ({p.experience_level} level)")
        if p.industry:
            parts.append(f"• Industry: {p.industry}")
        if p.goals:
            goals_str = "; ".join(p.goals[:3])
            parts.append(f"• Current Goals: {goals_str}")
        if p.skills:
            skills_str = ", ".join(p.skills[:6])
            parts.append(f"• Key Skills: {skills_str}")
        if p.interests:
            interests_str = ", ".join(p.interests[:4])
            parts.append(f"• Interests: {interests_str}")
        if p.inferred_topics:
            topics_str = ", ".join(p.inferred_topics[:4])
            parts.append(f"• Frequent Topics (inferred): {topics_str}")

        return "\n".join(parts)

    def _build_behavioral_block(self, p: UserProfile) -> str:
        parts = ["BEHAVIORAL PROFILE:"]

        exp_inst = self.EXPERIENCE_INSTRUCTIONS.get(p.experience_level, "")
        if exp_inst:
            parts.append(f"• Experience calibration: {exp_inst}")

        style_inst = self.STYLE_INSTRUCTIONS.get(p.communication_style, "")
        if style_inst:
            parts.append(f"• Communication style: {style_inst}")

        length_inst = self.LENGTH_INSTRUCTIONS.get(p.preferred_response_length, "")
        if length_inst:
            parts.append(f"• Response length: {length_inst}")

        if p.message_count > 20:
            parts.append(
                f"• You have had {p.message_count} conversations with {p.name}. "
                "Reference patterns from past interactions when relevant."
            )

        return "\n".join(parts)

    def _build_instructions_block(self, p: UserProfile) -> str:
        return (
            "CORE INSTRUCTIONS:\n"
            f"1. Always connect advice to {p.name}'s specific goals and context\n"
            "2. Be proactive — suggest next steps without being asked\n"
            "3. When citing documents, always name the source\n"
            "4. Prioritize actionability — every response should have at least one concrete next step\n"
            "5. If you don't know something, say so clearly and suggest where to find the answer"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Content Scorer — ranks content by profile relevance
# ─────────────────────────────────────────────────────────────────────────────

class ContentScorer:
    """
    Scores any content item (recommendation, document chunk, etc.)
    against a user profile for relevance ranking.

    Scoring factors:
      • Keyword overlap with goals/skills/interests
      • Role alignment
      • Experience level suitability
      • Topic recency (inferred from conversations)
    """

    def score(self, content: Dict, profile: UserProfile) -> float:
        """
        Returns a 0.0–1.0 relevance score.
        Higher = more relevant to this user.
        """
        score = 0.0

        # Build profile text for overlap scoring
        profile_text = " ".join([
            " ".join(profile.goals),
            " ".join(profile.skills),
            " ".join(profile.interests),
            " ".join(profile.inferred_topics),
            profile.role,
            profile.industry,
        ]).lower()

        # Content text
        content_text = " ".join([
            str(content.get("title", "")),
            str(content.get("content", "")),
            " ".join(content.get("tags", [])),
            str(content.get("category", "")),
        ]).lower()

        # 1. Keyword overlap (0.0 – 0.4)
        profile_words = set(w for w in profile_text.split() if len(w) > 3)
        content_words = set(w for w in content_text.split() if len(w) > 3)
        overlap = len(profile_words & content_words)
        score += min(0.4, overlap * 0.05)

        # 2. Priority boost (0.0 – 0.3)
        priority_map = {"high": 0.3, "medium": 0.15, "low": 0.0}
        score += priority_map.get(content.get("priority", "low"), 0.0)

        # 3. Experience level alignment (0.0 – 0.15)
        content_exp = content.get("experience_level", "any")
        if content_exp == "any" or content_exp == profile.experience_level:
            score += 0.15
        elif abs(
            UserProfile.EXPERIENCE_LEVELS.index(profile.experience_level)
            - UserProfile.EXPERIENCE_LEVELS.index(content_exp)
        ) <= 1 if content_exp in UserProfile.EXPERIENCE_LEVELS else True:
            score += 0.08

        # 4. Category match to inferred topics (0.0 – 0.15)
        category = content.get("category", "").lower()
        if any(category in topic or topic in category for topic in profile.inferred_topics):
            score += 0.15

        return min(1.0, score)

    def rank(self, items: List[Dict], profile: UserProfile) -> List[Tuple[Dict, float]]:
        """Rank a list of content items by profile relevance."""
        scored = [(item, self.score(item, profile)) for item in items]
        return sorted(scored, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Personalization Engine (top-level facade)
# ─────────────────────────────────────────────────────────────────────────────

class PersonalizationEngine:
    """
    Top-level facade that coordinates profile management,
    prompt personalization, and content scoring.

    Usage
    -----
    ```python
    engine = PersonalizationEngine()
    profile = engine.get_or_create_profile("user-123")
    profile.name = "Alice"
    profile.goals = ["Become a principal AI engineer"]
    profile.skills = ["Python", "PyTorch", "LangChain"]

    # Build personalized system prompt
    system_prompt = engine.build_system_prompt(profile)

    # Record interaction to improve personalization over time
    engine.record_interaction(profile, user_query, len(ai_response))

    # Rank recommendations by profile relevance
    ranked = engine.rank_content(recommendations, profile)
    ```
    """

    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
        self._prompt_personalizer = PromptPersonalizer()
        self._content_scorer = ContentScorer()

    def get_or_create_profile(self, user_id: str = "default") -> UserProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    def update_profile(self, user_id: str, **kwargs) -> UserProfile:
        profile = self.get_or_create_profile(user_id)
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.updated_at = datetime.utcnow().isoformat()
        return profile

    def build_system_prompt(self, profile: UserProfile) -> str:
        return self._prompt_personalizer.build_system_prompt(profile)

    def record_interaction(
        self, profile: UserProfile, query: str, response_length: int
    ) -> None:
        profile.record_interaction(query, response_length)

    def rank_content(
        self, items: List[Dict], profile: UserProfile
    ) -> List[Tuple[Dict, float]]:
        return self._content_scorer.rank(items, profile)

    def get_profile_summary(self, profile: UserProfile) -> Dict:
        return {
            "completeness": profile.completeness_score,
            "level": profile.personalization_level,
            "message_count": profile.message_count,
            "inferred_topics": profile.inferred_topics,
            "preferred_length": profile.preferred_response_length,
        }
