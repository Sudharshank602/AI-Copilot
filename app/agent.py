"""
AI Personal Intelligence Copilot
LangChain Agent Module — Tool-Augmented AI

Why Agents?
  A basic LLM call can only use its parametric knowledge.
  An Agent can USE TOOLS — calling external APIs, running calculations,
  reading files, executing code — and reason step-by-step (ReAct pattern).

Tools included:
  1. calculator      — Safe Python math evaluation
  2. word_counter    — Count words/chars/tokens in text
  3. goal_tracker    — Check user progress toward goals
  4. knowledge_base  — Search the vector DB (RAG as a tool)
  5. date_time       — Current date/time awareness
  6. text_summarizer — Summarize long blocks of text

Architecture (ReAct loop):
  Thought → Action (pick tool) → Observation (tool result) → Thought → ... → Final Answer

Why LangChain for agents?
  LangChain's AgentExecutor handles the thought/action/observation loop,
  tool registration, error recovery, and max-iteration safety — all built-in.
"""

from __future__ import annotations

import math
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

class Tool:
    """Lightweight tool descriptor — compatible with LangChain's Tool class."""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_text: str) -> str:
        try:
            return str(self.func(input_text))
        except Exception as e:
            return f"Tool error: {str(e)}"


def _calculator(expression: str) -> str:
    """
    Safe math expression evaluator.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e
    Rejects any code that isn't pure math.
    """
    # Whitelist: only math-safe characters
    safe_chars = set("0123456789+-*/.() abcdefghijklmnopqrstuvwxyz_,")
    if not all(c in safe_chars for c in expression.lower()):
        return "Error: Only mathematical expressions are allowed."

    # Safe math namespace
    safe_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "inf": math.inf,
        "ceil": math.ceil, "floor": math.floor,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)
        return f"{expression} = {result}"
    except Exception as ex:
        return f"Calculation error: {ex}"


def _word_counter(text: str) -> str:
    """Count words, characters, sentences, and estimate tokens."""
    words = len(text.split())
    chars = len(text)
    chars_no_space = len(text.replace(" ", ""))
    sentences = len(re.findall(r'[.!?]+', text)) or 1
    tokens_est = max(1, chars // 4)
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])

    return (
        f"Words: {words:,} | Characters: {chars:,} (no spaces: {chars_no_space:,}) | "
        f"Sentences: {sentences} | Paragraphs: {paragraphs} | "
        f"Tokens (est): {tokens_est:,} | "
        f"Reading time: ~{max(1, words // 200)} min"
    )


def _date_time(_: str) -> str:
    """Return current date, time, day of week, and week number."""
    now = datetime.utcnow()
    return (
        f"Current UTC datetime: {now.strftime('%A, %B %d, %Y at %H:%M:%S')} | "
        f"Week {now.isocalendar()[1]} of {now.year} | "
        f"Day {now.timetuple().tm_yday} of {365 + (1 if now.year % 4 == 0 else 0)}"
    )


def _text_summarizer(text: str) -> str:
    """
    Extractive summarizer — picks the most information-dense sentences.
    Heuristic: sentences with more unique long words are more informative.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 3:
        return text

    def score_sentence(s: str) -> float:
        words = s.lower().split()
        unique_long = len({w for w in words if len(w) > 5})
        return unique_long / max(len(words), 1)

    scored = sorted(enumerate(sentences), key=lambda x: score_sentence(x[1]), reverse=True)
    top_indices = sorted([i for i, _ in scored[:3]])
    summary = " ".join(sentences[i] for i in top_indices)
    return f"[Extractive Summary — {len(sentences)} sentences → 3]\n{summary}"


def _goal_checker(goals_json: str) -> str:
    """
    Parse a list of goals and return a progress-tracking template.
    Input: JSON list of goal strings, or plain text goal.
    """
    try:
        goals = json.loads(goals_json)
        if not isinstance(goals, list):
            goals = [str(goals)]
    except (json.JSONDecodeError, TypeError):
        goals = [goals_json.strip()]

    lines = ["Goal Progress Tracker:", "─" * 30]
    for i, goal in enumerate(goals[:10], 1):
        lines.append(f"{i}. {goal}")
        lines.append(f"   Status: [ ] Not started  [ ] In progress  [x] Completed")
        lines.append(f"   Priority: High | Next action: Define first concrete step")
        lines.append("")

    lines.append("Tip: Break each goal into weekly milestones for measurable progress.")
    return "\n".join(lines)


def _skill_gap_analyzer(skills_input: str) -> str:
    """
    Compare user skills against target role requirements.
    Input: 'current: Python, SQL | target: AI Engineer'
    """
    # AI Engineer skill requirements (simplified knowledge base)
    role_requirements = {
        "ai engineer": {
            "must_have": ["Python", "LangChain", "PyTorch or TensorFlow", "SQL", "Git",
                          "Docker", "REST APIs", "Cloud (AWS/GCP/Azure)"],
            "nice_to_have": ["Kubernetes", "MLflow", "Airflow", "React", "Go"],
            "certifications": ["AWS ML Specialty", "GCP Professional ML Engineer"],
        },
        "data scientist": {
            "must_have": ["Python", "SQL", "Statistics", "Scikit-learn", "Pandas",
                          "Data Visualization", "A/B Testing"],
            "nice_to_have": ["Spark", "Tableau", "R", "Deep Learning"],
            "certifications": ["Google Data Analytics", "IBM Data Science"],
        },
        "ml engineer": {
            "must_have": ["Python", "PyTorch", "TensorFlow", "MLOps", "Docker",
                          "Kubernetes", "CI/CD", "Feature Engineering"],
            "nice_to_have": ["CUDA", "Rust", "C++", "Ray", "Triton"],
            "certifications": ["AWS ML Specialty", "MLOps Engineer"],
        },
    }

    # Parse input
    current_skills = []
    target_role = "ai engineer"

    parts = skills_input.lower().split("|")
    for part in parts:
        if "current" in part or "have" in part:
            skills_str = part.split(":", 1)[-1]
            current_skills = [s.strip() for s in skills_str.split(",") if s.strip()]
        elif "target" in part or "want" in part or "role" in part:
            role_str = part.split(":", 1)[-1].strip()
            for role in role_requirements:
                if any(w in role_str for w in role.split()):
                    target_role = role
                    break

    req = role_requirements.get(target_role, role_requirements["ai engineer"])
    must_have = req["must_have"]
    nice_to_have = req["nice_to_have"]

    # Check coverage
    current_lower = {s.lower() for s in current_skills}
    covered = [s for s in must_have if any(c in s.lower() or s.lower() in c for c in current_lower)]
    gaps = [s for s in must_have if s not in covered]

    result_lines = [
        f"Skill Gap Analysis — Target: {target_role.title()}",
        "─" * 40,
        f"✅ Skills You Have ({len(covered)}/{len(must_have)} required):",
    ]
    for s in covered:
        result_lines.append(f"   ✓ {s}")

    result_lines.append(f"\n🎯 Skills to Acquire ({len(gaps)} gaps):")
    for i, s in enumerate(gaps, 1):
        result_lines.append(f"   {i}. {s}")

    result_lines.append(f"\n💡 Nice-to-Have: {', '.join(nice_to_have[:4])}")
    result_lines.append(f"📜 Top Certifications: {', '.join(req['certifications'])}")

    if gaps:
        result_lines.append(f"\n⚡ Recommended Focus: Start with '{gaps[0]}' — highest impact gap")
    else:
        result_lines.append("\n🏆 Excellent! You meet all core requirements. Focus on nice-to-haves.")

    return "\n".join(result_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

def build_tools(vector_store=None) -> List[Tool]:
    """
    Build and return all available tools.
    Optionally inject a vector store for the knowledge_base tool.
    """
    tools = [
        Tool(
            name="calculator",
            description=(
                "Perform mathematical calculations. "
                "Input: a math expression like '2 ** 10' or 'sqrt(144)'. "
                "Use when the user asks to calculate, compute, or solve math."
            ),
            func=_calculator,
        ),
        Tool(
            name="word_counter",
            description=(
                "Count words, characters, sentences, and estimate reading time. "
                "Input: any text to analyze. "
                "Use when asked about text length, word count, or reading time."
            ),
            func=_word_counter,
        ),
        Tool(
            name="date_time",
            description=(
                "Get the current date, time, day of week, and week number. "
                "Input: anything (input is ignored). "
                "Use when the user asks about today's date or current time."
            ),
            func=_date_time,
        ),
        Tool(
            name="text_summarizer",
            description=(
                "Summarize a long block of text into 3 key sentences. "
                "Input: the full text to summarize. "
                "Use when the user asks to summarize, condense, or TLDR text."
            ),
            func=_text_summarizer,
        ),
        Tool(
            name="goal_tracker",
            description=(
                "Create a goal progress tracking template. "
                "Input: JSON list of goals or a single goal as text. "
                "Use when the user wants to track goals or set objectives."
            ),
            func=_goal_checker,
        ),
        Tool(
            name="skill_gap_analyzer",
            description=(
                "Analyze skill gaps for a target role. "
                "Input: 'current: skill1, skill2 | target: AI Engineer'. "
                "Use when the user asks about skills needed for a specific role."
            ),
            func=_skill_gap_analyzer,
        ),
    ]

    # Knowledge base tool (requires vector store)
    if vector_store is not None:
        def _knowledge_base_search(query: str) -> str:
            results = vector_store.search(query, top_k=3, threshold=0.35)
            if not results:
                return "No relevant information found in the knowledge base."
            lines = ["Knowledge Base Results:"]
            for i, (text, meta, score) in enumerate(results, 1):
                source = meta.get("filename", "unknown")
                lines.append(f"[{i}] Source: {source} (score: {score:.2f})\n{text[:300]}…")
            return "\n\n".join(lines)

        tools.append(Tool(
            name="knowledge_base",
            description=(
                "Search the user's uploaded documents and knowledge base. "
                "Input: a search query. "
                "Use when the user asks about content from their uploaded files."
            ),
            func=_knowledge_base_search,
        ))

    return tools


# ─────────────────────────────────────────────────────────────────────────────
# Simple ReAct Agent (no external API needed for tool execution)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleAgent:
    """
    A lightweight ReAct-style agent that:
    1. Detects if a tool should be called based on the query
    2. Calls the appropriate tool
    3. Incorporates the result into the final response

    For production: replace with LangChain's AgentExecutor + OpenAI functions.
    This version works without any API key for demos.
    """

    TOOL_TRIGGER_PATTERNS = {
        "calculator": [
            r'\b(calculat|compute|what is \d|solve|math|equation|formula|\d+\s*[\+\-\*\/\^]\s*\d)',
            r'\b(percentage|percent|square root|sqrt|log of|sin|cos)\b',
        ],
        "word_counter": [
            r'\b(count (words|characters|letters)|how (many|long)|word count|reading time)\b',
        ],
        "date_time": [
            r"\b(today|current date|what('s| is) the date|day is it|time is it|what time)\b",
        ],
        "text_summarizer": [
            r'\b(summarize|summary|tldr|condense|shorten|brief version)\b',
        ],
        "goal_tracker": [
            r'\b(track (my )?goals?|set (a )?goal|goal setting|objectives?|okr)\b',
        ],
        "skill_gap_analyzer": [
            r'\b(skill gap|skills? (needed|required|for|to become)|what skills?|missing skills?)\b',
        ],
        "knowledge_base": [
            r'\b(in (my |the )?(document|file|upload|pdf|notes?)|from (my|the) (document|knowledge base))\b',
            r'\b(search (my|the) (documents?|files?|notes?))\b',
        ],
    }

    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}

    def detect_tool(self, query: str) -> Optional[str]:
        """Return the name of the best-matching tool, or None."""
        query_lower = query.lower()
        for tool_name, patterns in self.TOOL_TRIGGER_PATTERNS.items():
            if tool_name not in self.tools:
                continue
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return tool_name
        return None

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a query.
        Returns: {tool_used, tool_result, augmented_query}
        """
        tool_name = self.detect_tool(query)

        if tool_name is None:
            return {
                "tool_used": None,
                "tool_result": None,
                "augmented_query": query,
            }

        tool = self.tools[tool_name]

        # Extract the relevant input for the tool
        tool_input = self._extract_tool_input(query, tool_name)
        logger.info(f"Agent: Calling '{tool_name}' with input: {tool_input[:60]}…")

        result = tool.run(tool_input)

        # Build augmented query that includes the tool result
        augmented = (
            f"{query}\n\n"
            f"[Tool Used: {tool_name}]\n"
            f"[Tool Result]\n{result}\n[/Tool Result]\n\n"
            f"Based on the above tool result, please provide a helpful response."
        )

        return {
            "tool_used": tool_name,
            "tool_result": result,
            "augmented_query": augmented,
        }

    def _extract_tool_input(self, query: str, tool_name: str) -> str:
        """Extract the relevant input string for a specific tool from the query."""
        if tool_name == "calculator":
            # Try to extract the math expression
            match = re.search(r'[\d\s\+\-\*\/\(\)\.\^sqrtlogsincotan]+', query)
            if match:
                expr = match.group(0).strip()
                if len(expr) > 2:
                    return expr
            return query
        elif tool_name == "word_counter":
            # Look for quoted text or text after "count words in"
            match = re.search(r'"([^"]+)"', query)
            if match:
                return match.group(1)
            # Remove the question and return the rest
            cleaned = re.sub(r'^(count|how many|word count|characters in|analyze)\s+(words|characters|letters)?\s*(in|of)?\s*', '', query, flags=re.IGNORECASE)
            return cleaned.strip() or query
        elif tool_name == "date_time":
            return "now"
        elif tool_name == "text_summarizer":
            # Extract text after "summarize:" or within quotes
            match = re.search(r'(?:summarize|summary of|tldr)[:\s]+(.+)', query, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            return query
        elif tool_name == "skill_gap_analyzer":
            return query
        elif tool_name == "goal_tracker":
            # Extract goals from the query
            match = re.search(r'(?:goals?|objectives?)[:\s]+(.+)', query, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            return query
        else:
            return query

    def list_tools(self) -> List[Dict]:
        """Return tool descriptions for the UI."""
        return [
            {"name": t.name, "description": t.description}
            for t in self.tools.values()
        ]
