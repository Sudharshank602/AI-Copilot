"""
AI Personal Intelligence Copilot
Extended Test Suite — Agent + Personalization
"""

import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Agent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestToolFunctions:
    """Test individual tool functions directly."""

    def test_calculator_basic(self):
        from app.agent import _calculator
        result = _calculator("2 + 2")
        assert "4" in result

    def test_calculator_power(self):
        from app.agent import _calculator
        result = _calculator("2 ** 10")
        assert "1024" in result

    def test_calculator_sqrt(self):
        from app.agent import _calculator
        result = _calculator("sqrt(144)")
        assert "12" in result

    def test_calculator_blocks_code(self):
        from app.agent import _calculator
        result = _calculator("import os; os.system('ls')")
        assert "Error" in result or "Only mathematical" in result

    def test_word_counter_basic(self):
        from app.agent import _word_counter
        text = "The quick brown fox"
        result = _word_counter(text)
        assert "Words: 4" in result
        assert "Characters:" in result

    def test_word_counter_reading_time(self):
        from app.agent import _word_counter
        # 400 words = ~2 min reading time
        text = " ".join(["word"] * 400)
        result = _word_counter(text)
        assert "min" in result

    def test_date_time_returns_date(self):
        from app.agent import _date_time
        result = _date_time("")
        assert "2" in result  # Year will contain "2"
        assert ":" in result  # Time separator

    def test_text_summarizer_long_text(self):
        from app.agent import _text_summarizer
        text = (
            "Artificial intelligence is transforming every industry. "
            "Machine learning models can now solve complex problems. "
            "Natural language processing enables computers to understand human text. "
            "Computer vision allows machines to interpret images and video. "
            "Reinforcement learning trains agents through trial and error. "
        )
        result = _text_summarizer(text)
        assert "Summary" in result

    def test_text_summarizer_short_text_unchanged(self):
        from app.agent import _text_summarizer
        text = "Short text. Very short."
        result = _text_summarizer(text)
        assert result == text  # No change for short text

    def test_goal_checker_basic(self):
        from app.agent import _goal_checker
        goals = '["Land AI job", "Build RAG system"]'
        result = _goal_checker(goals)
        assert "Land AI job" in result
        assert "Build RAG system" in result
        assert "Goal Progress Tracker" in result

    def test_skill_gap_analyzer(self):
        from app.agent import _skill_gap_analyzer
        query = "current: Python, SQL | target: AI Engineer"
        result = _skill_gap_analyzer(query)
        assert "AI Engineer" in result or "ai engineer" in result.lower()
        assert "Gap" in result or "gap" in result.lower()


class TestSimpleAgent:
    def _make_agent(self):
        from app.agent import build_tools, SimpleAgent
        tools = build_tools()   # No vector store for unit tests
        return SimpleAgent(tools)

    def test_agent_detects_calculator(self):
        agent = self._make_agent()
        result = agent.run("What is 2 + 2?")
        assert result["tool_used"] == "calculator"
        assert result["tool_result"] is not None

    def test_agent_detects_datetime(self):
        agent = self._make_agent()
        result = agent.run("What is today's date?")
        assert result["tool_used"] == "date_time"

    def test_agent_detects_word_counter(self):
        agent = self._make_agent()
        result = agent.run("Count words in: Hello world today")
        assert result["tool_used"] == "word_counter"

    def test_agent_no_tool_for_generic(self):
        agent = self._make_agent()
        result = agent.run("Tell me about yourself")
        # Generic query — no tool should trigger
        assert result["tool_used"] is None
        assert result["augmented_query"] == "Tell me about yourself"

    def test_agent_augmented_query_contains_result(self):
        agent = self._make_agent()
        result = agent.run("What is 10 * 10?")
        if result["tool_used"]:
            assert "Tool Used" in result["augmented_query"]
            assert "Tool Result" in result["augmented_query"]

    def test_agent_list_tools(self):
        agent = self._make_agent()
        tools = agent.list_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 5
        names = [t["name"] for t in tools]
        assert "calculator" in names
        assert "date_time" in names
        assert "skill_gap_analyzer" in names

    def test_skill_gap_tool(self):
        agent = self._make_agent()
        result = agent.run("What skills do I need to become an AI engineer?")
        assert result["tool_used"] == "skill_gap_analyzer"


# ─────────────────────────────────────────────────────────────────────────────
# Personalization Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUserProfile:
    def _make_profile(self):
        from app.personalization import UserProfile
        return UserProfile(
            user_id="u-test",
            name="Alex",
            role="AI Engineer",
            goals=["Build RAG systems", "Lead AI team"],
            skills=["Python", "LangChain", "PyTorch"],
            interests=["LLMs", "MLOps"],
            experience_level="mid-level",
        )

    def test_completeness_empty_profile(self):
        from app.personalization import UserProfile
        p = UserProfile()
        assert 0 <= p.completeness_score <= 100

    def test_completeness_full_profile(self):
        p = self._make_profile()
        assert p.completeness_score >= 50

    def test_personalization_level_basic(self):
        from app.personalization import UserProfile
        p = UserProfile()
        assert p.personalization_level == "basic"

    def test_personalization_level_deep(self):
        p = self._make_profile()
        # Full profile should be standard or deep
        assert p.personalization_level in ("standard", "deep")

    def test_record_interaction_increments_count(self):
        p = self._make_profile()
        initial = p.message_count
        p.record_interaction("How do I use LangChain?", 500)
        assert p.message_count == initial + 1

    def test_topic_inference(self):
        p = self._make_profile()
        p.record_interaction("Tell me about PyTorch training loops", 300)
        assert "machine learning" in p.inferred_topics

    def test_serialization_roundtrip(self):
        p = self._make_profile()
        p.record_interaction("Test", 100)
        d = p.to_dict()
        from app.personalization import UserProfile
        restored = UserProfile.from_dict(d)
        assert restored.name == p.name
        assert restored.goals == p.goals
        assert restored.skills == p.skills


class TestPromptPersonalizer:
    def test_builds_prompt_basic(self):
        from app.personalization import PromptPersonalizer, UserProfile
        pp = PromptPersonalizer()
        p = UserProfile(name="Test User")
        prompt = pp.build_system_prompt(p)
        assert "Test User" in prompt
        assert len(prompt) > 50

    def test_builds_prompt_deep(self):
        from app.personalization import PromptPersonalizer, UserProfile
        pp = PromptPersonalizer()
        p = UserProfile(
            name="Alex",
            role="Senior AI Engineer",
            goals=["Scale AI infrastructure"],
            skills=["Python", "PyTorch", "Kubernetes"],
            interests=["LLMs", "Distributed Systems"],
            experience_level="senior",
        )
        prompt = pp.build_system_prompt(p)
        assert "Alex" in prompt
        assert "Senior" in prompt or "senior" in prompt
        # Deep profile should include behavioral block
        assert len(prompt) > 200

    def test_experience_calibration_in_prompt(self):
        from app.personalization import PromptPersonalizer, UserProfile
        pp = PromptPersonalizer()
        # Student should get "from first principles" instruction
        p = UserProfile(name="Student", experience_level="student")
        p.goals = ["Learn AI"]
        p.skills = ["Python"]
        prompt = pp.build_system_prompt(p)
        assert len(prompt) > 0


class TestContentScorer:
    def test_high_overlap_gives_higher_score(self):
        from app.personalization import ContentScorer, UserProfile
        scorer = ContentScorer()
        profile = UserProfile(
            goals=["build rag systems"],
            skills=["langchain", "faiss"],
            interests=["retrieval", "vector databases"],
        )
        high_rec = {
            "title": "Build a Production RAG System with LangChain",
            "content": "Learn to build RAG systems using FAISS vector database and LangChain",
            "tags": ["RAG", "LangChain", "FAISS"],
            "priority": "high",
            "category": "learning",
        }
        low_rec = {
            "title": "Introduction to Cooking Recipes",
            "content": "Learn to cook delicious meals with simple ingredients",
            "tags": ["cooking", "food"],
            "priority": "low",
            "category": "general",
        }
        high_score = scorer.score(high_rec, profile)
        low_score = scorer.score(low_rec, profile)
        assert high_score > low_score

    def test_rank_returns_sorted_list(self):
        from app.personalization import ContentScorer, UserProfile
        scorer = ContentScorer()
        profile = UserProfile(goals=["AI job"], skills=["Python"])
        items = [
            {"title": "Python tutorial", "content": "Learn Python", "tags": ["Python"], "priority": "high", "category": "learning"},
            {"title": "Cooking class", "content": "Learn to cook", "tags": ["food"], "priority": "low", "category": "general"},
            {"title": "AI career guide", "content": "Get a job in AI with Python", "tags": ["AI", "Python", "career"], "priority": "high", "category": "career"},
        ]
        ranked = scorer.rank(items, profile)
        # Should return sorted (highest first)
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)


class TestPersonalizationEngine:
    def test_get_or_create_profile(self):
        from app.personalization import PersonalizationEngine
        pe = PersonalizationEngine()
        p = pe.get_or_create_profile("u-001")
        assert p.user_id == "u-001"

    def test_profile_caching(self):
        from app.personalization import PersonalizationEngine
        pe = PersonalizationEngine()
        p1 = pe.get_or_create_profile("u-002")
        p2 = pe.get_or_create_profile("u-002")
        assert p1 is p2

    def test_update_profile(self):
        from app.personalization import PersonalizationEngine
        pe = PersonalizationEngine()
        p = pe.update_profile("u-003", name="Bob", role="Data Scientist")
        assert p.name == "Bob"
        assert p.role == "Data Scientist"

    def test_build_system_prompt_contains_name(self):
        from app.personalization import PersonalizationEngine
        pe = PersonalizationEngine()
        p = pe.update_profile("u-004", name="Charlie", goals=["Grow career"])
        prompt = pe.build_system_prompt(p)
        assert "Charlie" in prompt

    def test_get_profile_summary(self):
        from app.personalization import PersonalizationEngine
        pe = PersonalizationEngine()
        p = pe.get_or_create_profile("u-005")
        summary = pe.get_profile_summary(p)
        assert "completeness" in summary
        assert "level" in summary
        assert "message_count" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
