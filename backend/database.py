"""
Database configuration and models

Tenet #2: Zero PII in logs
Tenet #7: Immutability by Default
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./psyflo.db"  # Default to SQLite for prototype
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class Conversation(Base):
    """
    Conversation messages with risk assessments.
    
    Note: session_id is hashed for privacy (Tenet #2)
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id_hash = Column(String(64), index=True, nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    
    # Risk assessment
    risk_level = Column(String(20), nullable=False)  # SAFE, CAUTION, CRISIS
    risk_score = Column(Float, nullable=False)
    
    # Layer scores (for explainability)
    regex_score = Column(Float, nullable=False)
    semantic_score = Column(Float, nullable=False)
    mistral_score = Column(Float, nullable=True)  # Null if timeout
    
    # Reasoning trace
    reasoning = Column(Text, nullable=False)
    matched_patterns = Column(JSON, nullable=False)
    
    # Performance metrics
    latency_ms = Column(Integer, nullable=False)
    timeout_occurred = Column(Integer, nullable=False)  # 0 or 1 (boolean)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id_hash": self.session_id_hash,
            "message": self.message,
            "response": self.response,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "regex_score": self.regex_score,
            "semantic_score": self.semantic_score,
            "mistral_score": self.mistral_score,
            "reasoning": self.reasoning,
            "matched_patterns": self.matched_patterns,
            "latency_ms": self.latency_ms,
            "timeout_occurred": bool(self.timeout_occurred),
            "created_at": self.created_at.isoformat()
        }


class CrisisEvent(Base):
    """
    Crisis events for audit trail.
    
    Tenet #2: Immutable audit trail
    """
    __tablename__ = "crisis_events"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id_hash = Column(String(64), index=True, nullable=False)
    conversation_id = Column(Integer, nullable=False)
    
    # Risk details
    risk_score = Column(Float, nullable=False)
    matched_patterns = Column(JSON, nullable=False)
    reasoning = Column(Text, nullable=False)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id_hash": self.session_id_hash,
            "conversation_id": self.conversation_id,
            "risk_score": self.risk_score,
            "matched_patterns": self.matched_patterns,
            "reasoning": self.reasoning,
            "detected_at": self.detected_at.isoformat()
        }


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session (for FastAPI dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
