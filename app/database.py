"""
AI Personal Intelligence Copilot
Database Layer — SQLAlchemy Models + Initialization

SQLite is used as the persistent store for:
  • User profiles           → users table
  • Conversation sessions   → sessions table
  • Individual messages     → messages table
  • Ingested documents      → documents table
  • Recommendations         → recommendations table
  • Analytics events        → analytics table

Why SQLite?
  Lightweight, zero-config, file-based. Perfect for a single-user copilot
  that doesn't need a full Postgres cluster. Can be swapped trivially.
"""

from datetime import datetime
from typing import Optional
import json

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.pool import StaticPool

Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────


class User(Base):
    """
    User profile — stores identity, preferences, and personalization metadata.
    In a multi-user deployment this would be keyed by OAuth ID.
    """

    __tablename__ = "users"

    id = Column(String, primary_key=True)                          # UUID
    name = Column(String, nullable=False, default="User")
    email = Column(String, unique=True, nullable=True)
    role = Column(String, default="professional")                   # role context
    goals = Column(Text, default="[]")                             # JSON list
    interests = Column(Text, default="[]")                         # JSON list
    skills = Column(Text, default="[]")                            # JSON list
    interaction_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions = relationship("ConversationSession", back_populates="user",
                            cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user",
                             cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="user",
                                   cascade="all, delete-orphan")

    def get_goals(self):
        return json.loads(self.goals or "[]")

    def set_goals(self, goals_list):
        self.goals = json.dumps(goals_list)

    def get_interests(self):
        return json.loads(self.interests or "[]")

    def get_skills(self):
        return json.loads(self.skills or "[]")


class ConversationSession(Base):
    """
    A logical conversation session. Each session groups related messages
    and maintains its own summarized context window.
    """

    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New Conversation")
    summary = Column(Text, default="")                             # LLM-generated
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session",
                            cascade="all, delete-orphan",
                            order_by="Message.created_at")


class Message(Base):
    """
    Individual chat message within a session.
    Stores role (user/assistant/system), content, token count,
    and any RAG sources used to generate the response.
    """

    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)                          # user | assistant | system
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    sources = Column(Text, default="[]")                           # JSON list of doc refs
    model_used = Column(String, default="")
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ConversationSession", back_populates="messages")

    def get_sources(self):
        return json.loads(self.sources or "[]")


class Document(Base):
    """
    Tracks every document ingested into the vector DB.
    Chunk-level storage is handled by FAISS/Chroma; this table
    holds the metadata and processing status.
    """

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String, default="text")                     # pdf | txt | docx | md
    file_size_kb = Column(Float, default=0.0)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="processing")                  # processing | ready | error
    error_message = Column(Text, nullable=True)
    doc_metadata = Column(Text, default="{}")                     # JSON arbitrary metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")

    def get_metadata(self):
        return json.loads(self.doc_metadata or "{}")


class Recommendation(Base):
    """
    AI-generated personalized recommendations stored for history/replay.
    Category: career | learning | productivity | health | general
    """

    __tablename__ = "recommendations"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    category = Column(String, default="general")
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    priority = Column(String, default="medium")                    # high | medium | low
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="recommendations")


class AnalyticsEvent(Base):
    """
    Lightweight event log for the analytics dashboard.
    Tracks feature usage, response times, input counts, etc.
    """

    __tablename__ = "analytics"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    event_type = Column(String, nullable=False)                    # message_sent | doc_uploaded | etc.
    event_data = Column(Text, default="{}")                        # JSON payload
    created_at = Column(DateTime, default=datetime.utcnow)

    def get_data(self):
        return json.loads(self.event_data or "{}")


# ─────────────────────────────────────────────────────────────────────────────
# Engine & Session Factory
# ─────────────────────────────────────────────────────────────────────────────


def create_db_engine(db_url: str = "sqlite:///./data/copilot.db"):
    """
    Creates a SQLAlchemy engine.
    StaticPool + check_same_thread=False are required for SQLite + async usage.
    """
    connect_args = {}
    kwargs = {}

    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        kwargs["poolclass"] = StaticPool

    return create_engine(db_url, connect_args=connect_args, **kwargs)


def init_db(engine) -> None:
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


def get_session_factory(engine) -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
