"""
AI Personal Intelligence Copilot
Data Processing Pipeline — Pandas-Based Analytics

Why Pandas?
  DataFrames are the universal language of data analysis in Python.
  Pandas makes it trivial to:
  • Aggregate conversation statistics
  • Compute rolling averages for latency tracking
  • Export session data to CSV/Excel for reporting
  • Build charts for the analytics dashboard

This module provides:
  1. ConversationAnalytics  — per-session and aggregate message stats
  2. DocumentAnalytics      — document ingestion metrics
  3. RecommendationTracker  — track which recommendations were acted on
  4. ExportEngine           — export data to CSV/JSON/Excel
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from loguru import logger

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed — analytics will be limited")


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Analytics
# ─────────────────────────────────────────────────────────────────────────────

class ConversationAnalytics:
    """
    Processes conversation history into analytics DataFrames.

    Usage
    -----
    ```python
    analytics = ConversationAnalytics(messages)
    print(analytics.summary())
    df = analytics.to_dataframe()
    ```
    """

    def __init__(self, messages: List[Dict]):
        self.messages = messages
        self._df: Optional[Any] = None

        if HAS_PANDAS and messages:
            self._df = self._build_dataframe()

    def _build_dataframe(self):
        """Convert message list to a structured DataFrame."""
        rows = []
        for i, msg in enumerate(self.messages):
            content = msg.get("content", "")
            rows.append({
                "index":        i,
                "role":         msg.get("role", "unknown"),
                "content":      content,
                "word_count":   len(content.split()),
                "char_count":   len(content),
                "token_est":    max(1, len(content) // 4),
                "used_rag":     msg.get("used_rag", False),
                "latency_ms":   msg.get("latency_ms", 0.0),
                "source_count": len(msg.get("sources", [])),
                "timestamp":    msg.get("timestamp", ""),
            })

        df = pd.DataFrame(rows)

        # Parse timestamps if present
        if "timestamp" in df.columns and df["timestamp"].any():
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df

    def summary(self) -> Dict:
        """Return a stats summary dict."""
        if not self.messages:
            return {"error": "No messages to analyze"}

        user_msgs   = [m for m in self.messages if m.get("role") == "user"]
        ai_msgs     = [m for m in self.messages if m.get("role") == "assistant"]
        rag_msgs    = [m for m in ai_msgs if m.get("used_rag")]
        latencies   = [m.get("latency_ms", 0) for m in ai_msgs if m.get("latency_ms")]

        user_words  = sum(len(m.get("content", "").split()) for m in user_msgs)
        ai_words    = sum(len(m.get("content", "").split()) for m in ai_msgs)

        return {
            "total_messages":    len(self.messages),
            "user_messages":     len(user_msgs),
            "ai_messages":       len(ai_msgs),
            "conversation_turns": len(user_msgs),
            "rag_enhanced":      len(rag_msgs),
            "rag_rate_pct":      round(len(rag_msgs) / max(len(ai_msgs), 1) * 100, 1),
            "total_user_words":  user_words,
            "total_ai_words":    ai_words,
            "avg_user_words":    round(user_words / max(len(user_msgs), 1), 1),
            "avg_ai_words":      round(ai_words / max(len(ai_msgs), 1), 1),
            "avg_latency_ms":    round(sum(latencies) / max(len(latencies), 1), 1),
            "total_tokens_est":  sum(
                max(1, len(m.get("content", "")) // 4) for m in self.messages
            ),
        }

    def to_dataframe(self):
        """Return the messages DataFrame (requires pandas)."""
        return self._df

    def get_latency_trend(self) -> List[float]:
        """Return list of response latencies over time."""
        return [
            m.get("latency_ms", 0)
            for m in self.messages
            if m.get("role") == "assistant"
        ]

    def get_rag_usage_over_time(self) -> List[bool]:
        """Return list of bool: was RAG used for each assistant message?"""
        return [
            m.get("used_rag", False)
            for m in self.messages
            if m.get("role") == "assistant"
        ]

    def export_csv(self, path: str = "conversation_export.csv") -> str:
        """Export conversation to CSV."""
        if not HAS_PANDAS or self._df is None:
            raise RuntimeError("pandas required for CSV export")
        self._df.to_csv(path, index=False)
        logger.info(f"Exported {len(self._df)} messages to {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Document Analytics
# ─────────────────────────────────────────────────────────────────────────────

class DocumentAnalytics:
    """Track and analyze ingested documents."""

    def __init__(self, documents: List[Dict]):
        self.documents = documents

    def summary(self) -> Dict:
        if not self.documents:
            return {"total_documents": 0, "total_chunks": 0}

        total_chunks = sum(d.get("chunks", 0) for d in self.documents)
        by_type: Dict[str, int] = {}
        for doc in self.documents:
            ft = doc.get("file_type", "unknown")
            by_type[ft] = by_type.get(ft, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(total_chunks / len(self.documents), 1),
            "by_type": by_type,
            "ready": sum(1 for d in self.documents if d.get("status") == "ready"),
            "errors": sum(1 for d in self.documents if d.get("status") == "error"),
        }

    def to_dataframe(self):
        if not HAS_PANDAS:
            return None
        return pd.DataFrame(self.documents)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate Analytics (for the dashboard)
# ─────────────────────────────────────────────────────────────────────────────

class DashboardAnalytics:
    """
    Combines conversation + document analytics into dashboard-ready data.
    Used by the FastAPI /api/analytics endpoint and Streamlit analytics page.
    """

    def __init__(
        self,
        messages: Optional[List[Dict]] = None,
        documents: Optional[List[Dict]] = None,
        events: Optional[List[Dict]] = None,
    ):
        self.messages = messages or []
        self.documents = documents or []
        self.events = events or []

        self._conv = ConversationAnalytics(self.messages)
        self._docs = DocumentAnalytics(self.documents)

    def full_summary(self) -> Dict:
        conv_stats = self._conv.summary()
        doc_stats  = self._docs.summary()

        event_counts: Dict[str, int] = {}
        for ev in self.events:
            et = ev.get("type", "unknown")
            event_counts[et] = event_counts.get(et, 0) + 1

        return {
            "conversations": conv_stats,
            "documents":     doc_stats,
            "event_counts":  event_counts,
            "system": {
                "uptime_note": "Runtime metrics available via /api/health",
                "vector_chunks": doc_stats.get("total_chunks", 0),
            },
        }

    def kpi_cards(self) -> List[Dict]:
        """Return KPI card data for the Streamlit dashboard."""
        conv = self._conv.summary()
        docs = self._docs.summary()

        return [
            {
                "label":  "Messages Sent",
                "value":  conv.get("user_messages", 0),
                "icon":   "💬",
                "delta":  "+live",
                "color":  "#00D4FF",
            },
            {
                "label":  "Documents Ingested",
                "value":  docs.get("total_documents", 0),
                "icon":   "📄",
                "delta":  f"{docs.get('total_chunks', 0)} chunks",
                "color":  "#10B981",
            },
            {
                "label":  "RAG Usage Rate",
                "value":  f"{conv.get('rag_rate_pct', 0)}%",
                "icon":   "🔍",
                "delta":  f"{conv.get('rag_enhanced', 0)} responses",
                "color":  "#7C3AED",
            },
            {
                "label":  "Avg Response Time",
                "value":  f"{conv.get('avg_latency_ms', 0):.0f}ms",
                "icon":   "⚡",
                "delta":  "real-time",
                "color":  "#F59E0B",
            },
        ]
