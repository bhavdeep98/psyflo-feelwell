"""
FastAPI Backend for PsyFlo Prototype

Tenet #3: Explicit Over Clever - Simple, traceable API
Tenet #10: Observable Systems - All endpoints instrumented
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import structlog
import hashlib
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import get_db, init_db, Conversation, CrisisEvent
from src.orchestrator import ConsensusOrchestrator, ConsensusConfig
from src.safety.safety_analyzer import SafetyService
from src.reasoning.mistral_reasoner import MistralReasoner
from src.conversation import ConversationAgent, ConversationContext

# Initialize logging
logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="PsyFlo API",
    description="Mental health AI triage system",
    version="0.1.0"
)

# CORS middleware (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:3002"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (singleton pattern)
safety_analyzer = None
mistral_reasoner = None
orchestrator = None
conversation_agent = None


def hash_pii(pii: str) -> str:
    """Hash PII for logging (Tenet #2)."""
    return hashlib.sha256(pii.encode()).hexdigest()[:16]


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global safety_analyzer, mistral_reasoner, orchestrator, conversation_agent
    
    logger.info("initializing_services")
    
    # Initialize database
    init_db()
    logger.info("database_initialized")
    
    # Initialize services
    try:
        # Path to config from backend directory
        config_path = Path(__file__).parent.parent / "config" / "crisis_patterns.yaml"
        safety_analyzer = SafetyService(patterns_path=str(config_path))
        logger.info("safety_service_initialized")
        
        mistral_reasoner = MistralReasoner()
        logger.info("mistral_reasoner_initialized")
        
        orchestrator = ConsensusOrchestrator(
            safety_service=safety_analyzer,
            mistral_reasoner=mistral_reasoner,
            config=ConsensusConfig()
        )
        logger.info("orchestrator_initialized")
        
        # Initialize conversation agent
        # Tenet #4: Fail loud if API key missing
        # Tenet #11: Service will handle graceful degradation via circuit breaker
        try:
            conversation_agent = ConversationAgent()
            logger.info("conversation_agent_initialized")
        except RuntimeError as e:
            logger.error(
                "conversation_agent_initialization_failed",
                error=str(e),
                exc_info=True
            )
            # Re-raise - service cannot start without conversation agent
            raise
        
    except Exception as e:
        logger.error("service_initialization_failed", error=str(e), exc_info=True)
        raise


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request from student."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response to student."""
    response: str
    risk_level: str
    is_crisis: bool
    conversation_id: int


class ConversationDetail(BaseModel):
    """Detailed conversation for counselor view."""
    id: int
    session_id_hash: str
    message: str
    response: str
    risk_level: str
    risk_score: float
    regex_score: float
    semantic_score: float
    mistral_score: Optional[float]
    reasoning: str
    matched_patterns: List[str]
    latency_ms: int
    timeout_occurred: bool
    created_at: str


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "PsyFlo API",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "services": {
            "safety_analyzer": safety_analyzer is not None,
            "mistral_reasoner": mistral_reasoner is not None,
            "orchestrator": orchestrator is not None
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Process student message and return response.
    
    Tenet #1: Safety First - Crisis detection runs in parallel
    Tenet #9: Visibility - All decisions logged
    """
    session_id_hash = hash_pii(request.session_id)
    
    logger.info(
        "chat_request_received",
        session_id=session_id_hash,
        message_length=len(request.message)
    )
    
    try:
        # Analyze message using orchestrator
        result = await orchestrator.analyze(
            message=request.message,
            session_id=request.session_id
        )
        
        # Generate response using conversation agent
        # Tenet #1: Crisis detection already done, now generate appropriate response
        if result.is_crisis():
            # Crisis override - hard-coded protocol
            response_text = _get_crisis_response()
        else:
            # Generate empathetic response using LLM
            # Build conversation context
            conversation_history = _get_conversation_history(db, session_id_hash, limit=5)
            
            context = ConversationContext(
                session_id=request.session_id,
                risk_level=result.risk_level.value,
                risk_score=result.final_score,
                matched_patterns=result.matched_patterns,
                conversation_history=conversation_history
            )
            
            try:
                response_text = await conversation_agent.generate_response(
                    message=request.message,
                    context=context
                )
            except Exception as e:
                # Tenet #11: Graceful degradation - LLM service failure
                logger.error(
                    "conversation_agent_failed",
                    session_id=session_id_hash,
                    error=str(e),
                    exc_info=True
                )
                # Display crisis resources directly to student
                response_text = _get_service_unavailable_response()
        
        # Save to database
        conversation = Conversation(
            session_id_hash=session_id_hash,
            message=request.message,
            response=response_text,
            risk_level=result.risk_level.value,
            risk_score=result.final_score,
            regex_score=result.regex_score.score,
            semantic_score=result.semantic_score.score,
            mistral_score=result.mistral_score.score if result.mistral_score else None,
            reasoning=result.reasoning,
            matched_patterns=result.matched_patterns,
            latency_ms=result.total_latency_ms,
            timeout_occurred=1 if result.timeout_occurred else 0
        )
        
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        # Log crisis event if needed
        if result.is_crisis():
            crisis_event = CrisisEvent(
                session_id_hash=session_id_hash,
                conversation_id=conversation.id,
                risk_score=result.final_score,
                matched_patterns=result.matched_patterns,
                reasoning=result.reasoning
            )
            db.add(crisis_event)
            db.commit()
            
            logger.warning(
                "crisis_event_logged",
                session_id=session_id_hash,
                conversation_id=conversation.id,
                risk_score=result.final_score
            )
        
        logger.info(
            "chat_response_sent",
            session_id=session_id_hash,
            conversation_id=conversation.id,
            risk_level=result.risk_level.value,
            latency_ms=result.total_latency_ms
        )
        
        return ChatResponse(
            response=response_text,
            risk_level=result.risk_level.value,
            is_crisis=result.is_crisis(),
            conversation_id=conversation.id
        )
        
    except Exception as e:
        logger.error(
            "chat_request_failed",
            session_id=session_id_hash,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{session_id}", response_model=List[ConversationDetail])
async def get_conversations(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get conversation history for a session (counselor view).
    
    Tenet #9: Visibility - Counselors can see reasoning traces
    """
    session_id_hash = hash_pii(session_id)
    
    conversations = db.query(Conversation).filter(
        Conversation.session_id_hash == session_id_hash
    ).order_by(Conversation.created_at.desc()).limit(50).all()
    
    return [
        ConversationDetail(
            id=conv.id,
            session_id_hash=conv.session_id_hash,
            message=conv.message,
            response=conv.response,
            risk_level=conv.risk_level,
            risk_score=conv.risk_score,
            regex_score=conv.regex_score,
            semantic_score=conv.semantic_score,
            mistral_score=conv.mistral_score,
            reasoning=conv.reasoning,
            matched_patterns=conv.matched_patterns,
            latency_ms=conv.latency_ms,
            timeout_occurred=bool(conv.timeout_occurred),
            created_at=conv.created_at.isoformat()
        )
        for conv in conversations
    ]


@app.get("/conversations/lookup/{session_id_hash}", response_model=List[ConversationDetail])
async def get_conversations_by_hash(
    session_id_hash: str,
    db: Session = Depends(get_db)
):
    """
    Get conversation history by direct hash lookup (for internal/dashboard use).
    """
    conversations = db.query(Conversation).filter(
        Conversation.session_id_hash == session_id_hash
    ).order_by(Conversation.created_at.desc()).limit(50).all()
    
    return [
        ConversationDetail(
            id=conv.id,
            session_id_hash=conv.session_id_hash,
            message=conv.message,
            response=conv.response,
            risk_level=conv.risk_level,
            risk_score=conv.risk_score,
            regex_score=conv.regex_score,
            semantic_score=conv.semantic_score,
            mistral_score=conv.mistral_score,
            reasoning=conv.reasoning,
            matched_patterns=conv.matched_patterns,
            latency_ms=conv.latency_ms,
            timeout_occurred=bool(conv.timeout_occurred),
            created_at=conv.created_at.isoformat()
        )
        for conv in conversations
    ]


@app.get("/crisis-events", response_model=List[dict])
async def get_crisis_events(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get recent crisis events (counselor dashboard).
    
    Tenet #2: Immutable audit trail
    """
    events = db.query(CrisisEvent).order_by(
        CrisisEvent.detected_at.desc()
    ).limit(limit).all()
    
    return [event.to_dict() for event in events]


def _get_crisis_response() -> str:
    """
    Get deterministic crisis response.
    
    Tenet #1: Hard-coded crisis protocols that can't be overridden
    """
    return (
        "I'm really concerned about what you've shared. "
        "Your safety is the most important thing right now. "
        "I've notified your school counselor, and they'll reach out soon.\n\n"
        "In the meantime, here are some resources:\n\n"
        "🆘 National Suicide Prevention Lifeline: 988\n"
        "💬 Crisis Text Line: Text HOME to 741741\n"
        "🌐 Online Chat: https://suicidepreventionlifeline.org/chat/\n\n"
        "You're not alone in this. Help is available 24/7."
    )


def _get_service_unavailable_response() -> str:
    """
    Response when conversation service is temporarily unavailable.
    
    Tenet #11: Graceful degradation - show crisis resources
    """
    return (
        "I'm having trouble connecting right now, but I want to make sure you have support.\n\n"
        "If you need to talk to someone immediately:\n\n"
        "🆘 National Suicide Prevention Lifeline: 988\n"
        "💬 Crisis Text Line: Text HOME to 741741\n"
        "🌐 Online Chat: https://suicidepreventionlifeline.org/chat/\n\n"
        "Your school counselor is also available. Please reach out - you're not alone."
    )


def _get_conversation_history(
    db: Session,
    session_id_hash: str,
    limit: int = 5
) -> list[dict]:
    """
    Get recent conversation history for context.
    
    Args:
        db: Database session
        session_id_hash: Hashed session ID
        limit: Number of recent messages to retrieve
        
    Returns:
        List of conversation messages with role and content
    """
    conversations = db.query(Conversation).filter(
        Conversation.session_id_hash == session_id_hash
    ).order_by(Conversation.created_at.desc()).limit(limit).all()
    
    # Reverse to get chronological order
    conversations = list(reversed(conversations))
    
    history = []
    for conv in conversations:
        history.append({"role": "student", "content": conv.message})
        history.append({"role": "assistant", "content": conv.response})
    
    return history


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
