"""
AI Personal Intelligence Copilot
Utility Helpers

Text processing, token counting, logging setup, and other shared utilities.
"""

import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure loguru with clean formatting."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
        level=level,
        colorize=True,
    )
    if log_file:
        logger.add(log_file, rotation="10 MB", retention="7 days", level=level)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Token Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Fast token estimate without loading tiktoken.
    Rule of thumb: 1 token ≈ 4 characters (English text).
    Use tiktoken for precision in production.
    """
    return max(1, len(text) // 4)


def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Accurate token count using tiktoken (OpenAI's tokenizer)."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return estimate_tokens(text)


# ─────────────────────────────────────────────────────────────────────────────
# Text Processing
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize unicode."""
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def truncate_text(text: str, max_chars: int = 1000, suffix: str = "…") -> str:
    """Truncate text to max_chars, ending at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + suffix


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Simple keyword extraction without NLP dependencies.
    Removes stopwords and returns most frequent meaningful words.
    """
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'it', 'its', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'my',
        'your', 'his', 'her', 'our', 'their', 'what', 'which', 'who', 'how',
        'when', 'where', 'why', 'all', 'each', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than',
        'too', 'very', 'just', 'any', 'as', 'if',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


# ─────────────────────────────────────────────────────────────────────────────
# ID Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = str(uuid.uuid4()).replace("-", "")[:12]
    return f"{prefix}{uid}" if prefix else uid


# ─────────────────────────────────────────────────────────────────────────────
# Time Utilities
# ─────────────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def format_duration(ms: float) -> str:
    """Format milliseconds into human-readable duration."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Data Processing Helpers (Pandas)
# ─────────────────────────────────────────────────────────────────────────────

def messages_to_dataframe(messages: List[Dict]):
    """
    Convert a list of message dicts to a Pandas DataFrame.
    Useful for analytics and export.
    """
    try:
        import pandas as pd
        if not messages:
            return pd.DataFrame(columns=["role", "content", "timestamp"])
        df = pd.DataFrame(messages)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except ImportError:
        logger.warning("pandas not installed")
        return None


def compute_session_stats(messages: List[Dict]) -> Dict:
    """Compute statistics for a conversation session."""
    if not messages:
        return {"turns": 0, "user_words": 0, "ai_words": 0, "total_tokens_est": 0}

    user_msgs = [m for m in messages if m.get("role") == "user"]
    ai_msgs = [m for m in messages if m.get("role") == "assistant"]

    user_text = " ".join(m.get("content", "") for m in user_msgs)
    ai_text = " ".join(m.get("content", "") for m in ai_msgs)

    return {
        "turns": len(user_msgs),
        "user_words": len(user_text.split()),
        "ai_words": len(ai_text.split()),
        "total_tokens_est": estimate_tokens(user_text + ai_text),
        "avg_user_length": len(user_text) // max(len(user_msgs), 1),
        "avg_ai_length": len(ai_text) // max(len(ai_msgs), 1),
    }
