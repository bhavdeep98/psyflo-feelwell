# PsyFlo Design Patterns

Design patterns for building safe, maintainable, and compliant mental health systems.

## Core Architectural Patterns

### 1. Strategy Pattern (Crisis Detection)

Use the Strategy pattern for pluggable detection algorithms while maintaining consistent interfaces.

```python
# ✅ CORRECT: Pluggable strategies with consistent interface
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DetectionResult:
    """Immutable detection result."""
    is_crisis: bool
    confidence: float
    matched_patterns: List[str]
    evidence: str

class DetectionStrategy(ABC):
    """Base strategy for crisis detection."""
    
    @abstractmethod
    def detect(self, message: str) -> DetectionResult:
        """Detect crisis markers in message."""
        pass

class RegexStrategy(DetectionStrategy):
    """Keyword-based detection (fast, deterministic)."""
    
    def detect(self, message: str) -> DetectionResult:
        # Implementation
        pass

class SemanticStrategy(DetectionStrategy):
    """Embedding-based detection (catches paraphrases)."""
    
    def detect(self, message: str) -> DetectionResult:
        # Implementation
        pass

# Factory creates appropriate strategy
class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_type: str) -> DetectionStrategy:
        if strategy_type == "regex":
            return RegexStrategy()
        elif strategy_type == "semantic":
            return SemanticStrategy()
        raise ValueError(f"Unknown strategy: {strategy_type}")
```

**Why**: Allows adding new detection methods without modifying existing code. Easy to test each strategy independently.


### 2. Chain of Responsibility (Multi-Layer Detection)

Process messages through multiple detection layers, each with veto power.

```python
from typing import Optional

class DetectionHandler(ABC):
    """Handler in detection chain."""
    
    def __init__(self):
        self._next_handler: Optional[DetectionHandler] = None
    
    def set_next(self, handler: 'DetectionHandler') -> 'DetectionHandler':
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, message: str, context: dict) -> DetectionResult:
        pass

class SarcasmFilterHandler(DetectionHandler):
    """Filter out teenage hyperbole."""
    
    def handle(self, message: str, context: dict) -> DetectionResult:
        if self._is_sarcasm(message):
            return DetectionResult(
                is_crisis=False,
                confidence=0.95,
                matched_patterns=["sarcasm_filtered"],
                evidence="Detected hyperbolic language"
            )
        
        if self._next_handler:
            return self._next_handler.handle(message, context)
        
        return DetectionResult(is_crisis=False, confidence=0.0, matched_patterns=[], evidence="")


class KeywordDetectionHandler(DetectionHandler):
    """Detect explicit crisis keywords."""
    
    def handle(self, message: str, context: dict) -> DetectionResult:
        result = self._check_keywords(message)
        
        if result.is_crisis:
            return result  # Stop chain, crisis detected
        
        if self._next_handler:
            return self._next_handler.handle(message, context)
        
        return result

# Build chain
sarcasm = SarcasmFilterHandler()
keyword = KeywordDetectionHandler()
semantic = SemanticDetectionHandler()

sarcasm.set_next(keyword).set_next(semantic)

# Use chain
result = sarcasm.handle(message, context)
```

**Why**: Each layer can short-circuit (sarcasm filter) or escalate. Clear, linear flow for debugging.

### 3. Observer Pattern (Event-Driven Crisis Response)

Decouple crisis detection from notification, logging, and analytics.

```python
from typing import List, Callable

class CrisisEvent:
    """Immutable crisis event."""
    def __init__(self, student_id: str, message: str, risk_score: float):
        self.student_id = hash_pii(student_id)
        self.message_hash = hash_pii(message)
        self.risk_score = risk_score
        self.timestamp = datetime.utcnow()


class CrisisEventBus:
    """Event bus for crisis events."""
    
    def __init__(self):
        self._observers: List[Callable[[CrisisEvent], None]] = []
    
    def subscribe(self, observer: Callable[[CrisisEvent], None]) -> None:
        self._observers.append(observer)
    
    def publish(self, event: CrisisEvent) -> None:
        """Publish event to all observers."""
        for observer in self._observers:
            try:
                observer(event)
            except Exception as e:
                logger.error(f"Observer failed: {e}", exc_info=True)
                # Continue notifying other observers

# Observers
def notify_counselor(event: CrisisEvent) -> None:
    """Send crisis alert to counselor."""
    notification_service.send_alert(event)

def log_crisis(event: CrisisEvent) -> None:
    """Log crisis to audit trail."""
    audit_logger.log_crisis(event)

def update_analytics(event: CrisisEvent) -> None:
    """Update crisis analytics."""
    analytics_service.record_crisis(event)

# Wire up
event_bus = CrisisEventBus()
event_bus.subscribe(notify_counselor)
event_bus.subscribe(log_crisis)
event_bus.subscribe(update_analytics)

# Publish
event_bus.publish(CrisisEvent(student_id, message, 0.95))
```

**Why**: If notification service fails, logging and analytics still work. Easy to add new observers.


### 4. Circuit Breaker (Graceful Degradation)

Prevent cascading failures when dependencies are down.

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, return fallback
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker for external dependencies."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time is not None and
            datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.timeout)
        )

# Usage
llm_breaker = CircuitBreaker(failure_threshold=5, timeout=30)

def call_llm_with_fallback(prompt: str) -> str:
    try:
        return llm_breaker.call(llm_service.generate, prompt)
    except CircuitBreakerOpenError:
        logger.warning("LLM circuit breaker open, using fallback")
        return get_fallback_response(prompt)
```

**Why**: Prevents overwhelming failing services. Returns fallback immediately instead of waiting for timeout.


### 5. Builder Pattern (Immutable Configuration)

Build complex configurations immutably with validation at each step.

```python
from dataclasses import dataclass, replace
from typing import Optional

@dataclass(frozen=True)
class CrisisConfig:
    """Immutable crisis detection configuration."""
    keyword_threshold: float
    semantic_threshold: float
    sarcasm_enabled: bool
    notification_channels: tuple[str, ...]
    
    def __post_init__(self):
        """Validate configuration on creation."""
        if not 0.0 <= self.keyword_threshold <= 1.0:
            raise ValueError("keyword_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be between 0.0 and 1.0")

class CrisisConfigBuilder:
    """Builder for crisis configuration."""
    
    def __init__(self):
        self._keyword_threshold: float = 0.85
        self._semantic_threshold: float = 0.80
        self._sarcasm_enabled: bool = True
        self._notification_channels: list[str] = ["sms", "email"]
    
    def with_keyword_threshold(self, threshold: float) -> 'CrisisConfigBuilder':
        self._keyword_threshold = threshold
        return self
    
    def with_semantic_threshold(self, threshold: float) -> 'CrisisConfigBuilder':
        self._semantic_threshold = threshold
        return self
    
    def disable_sarcasm_filter(self) -> 'CrisisConfigBuilder':
        self._sarcasm_enabled = False
        return self
    
    def add_notification_channel(self, channel: str) -> 'CrisisConfigBuilder':
        if channel not in self._notification_channels:
            self._notification_channels.append(channel)
        return self
    
    def build(self) -> CrisisConfig:
        """Build immutable configuration with validation."""
        return CrisisConfig(
            keyword_threshold=self._keyword_threshold,
            semantic_threshold=self._semantic_threshold,
            sarcasm_enabled=self._sarcasm_enabled,
            notification_channels=tuple(self._notification_channels)
        )

# Usage
config = (CrisisConfigBuilder()
    .with_keyword_threshold(0.90)
    .add_notification_channel("phone")
    .build())
```

**Why**: Immutable configs prevent accidental modification. Validation happens once at build time.


### 6. Repository Pattern (Data Access Abstraction)

Abstract data access to enable testing and prevent PII leakage.

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class ConversationRepository(ABC):
    """Abstract repository for conversation data."""
    
    @abstractmethod
    def save_message(self, session_id: str, message: str, risk_score: float) -> None:
        """Save message with risk assessment."""
        pass
    
    @abstractmethod
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[dict]:
        """Get recent conversation history."""
        pass
    
    @abstractmethod
    def get_crisis_events(self, student_id: str, days: int = 30) -> List[dict]:
        """Get crisis events for student."""
        pass

class RDSConversationRepository(ConversationRepository):
    """PostgreSQL implementation."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def save_message(self, session_id: str, message: str, risk_score: float) -> None:
        """Save message with PII hashing."""
        try:
            self.db.execute(
                """
                INSERT INTO messages (session_id_hash, message_hash, risk_score, created_at)
                VALUES (%s, %s, %s, NOW())
                """,
                (hash_pii(session_id), hash_pii(message), risk_score)
            )
            logger.info(
                "message_saved",
                session_id=hash_pii(session_id),
                risk_score=risk_score
            )
        except Exception as e:
            logger.error(f"Failed to save message: {e}", exc_info=True)
            raise DataAccessError("Failed to save message", context={"session_id": hash_pii(session_id)})
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[dict]:
        """Get conversation history (hashed)."""
        try:
            results = self.db.query(
                """
                SELECT message_hash, risk_score, created_at
                FROM messages
                WHERE session_id_hash = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (hash_pii(session_id), limit)
            )
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}", exc_info=True)
            raise DataAccessError("Failed to fetch conversation history")

class InMemoryConversationRepository(ConversationRepository):
    """In-memory implementation for testing."""
    
    def __init__(self):
        self._messages: dict[str, List[dict]] = {}
    
    def save_message(self, session_id: str, message: str, risk_score: float) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append({
            "message": message,
            "risk_score": risk_score,
            "timestamp": datetime.utcnow()
        })
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[dict]:
        return self._messages.get(session_id, [])[-limit:]

# Usage in tests
def test_crisis_detection():
    repo = InMemoryConversationRepository()
    service = SafetyService(repo)
    
    result = service.analyze("I want to die")
    assert result.is_crisis is True
```

**Why**: Enables testing without database. Centralizes PII hashing. Easy to swap implementations.


## Safety-Critical Patterns

### 7. Audit Trail Pattern (Immutable Logging)

Every sensitive operation must be logged immutably with full context.

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class AuditEventType(Enum):
    CRISIS_DETECTED = "crisis_detected"
    COUNSELOR_NOTIFIED = "counselor_notified"
    CONVERSATION_ACCESSED = "conversation_accessed"
    DATA_DELETED = "data_deleted"
    CONFIG_CHANGED = "config_changed"

@dataclass(frozen=True)
class AuditEvent:
    """Immutable audit event."""
    event_type: AuditEventType
    actor_id: str  # Hashed
    subject_id: str  # Hashed (student, session, etc.)
    timestamp: datetime
    context: dict
    ip_address: str
    
    def to_json(self) -> dict:
        """Serialize for WORM storage."""
        return {
            "event_type": self.event_type.value,
            "actor_id": self.actor_id,
            "subject_id": self.subject_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "ip_address": self.ip_address
        }

class AuditLogger:
    """Write-only audit logger."""
    
    def __init__(self, worm_storage):
        self._storage = worm_storage
    
    def log(self, event: AuditEvent) -> None:
        """Log event to WORM storage."""
        try:
            self._storage.append(event.to_json())
            logger.debug(f"Audit event logged: {event.event_type.value}")
        except Exception as e:
            # Audit logging failure is CRITICAL
            logger.critical(f"AUDIT LOG FAILURE: {e}", exc_info=True)
            raise AuditLogError("Failed to write audit log", original_error=e)

# Usage
audit_logger = AuditLogger(worm_storage)

def detect_crisis(message: str, student_id: str) -> DetectionResult:
    result = crisis_detector.analyze(message)
    
    if result.is_crisis:
        # Log BEFORE notifying (ensures we have record even if notification fails)
        audit_logger.log(AuditEvent(
            event_type=AuditEventType.CRISIS_DETECTED,
            actor_id="system",
            subject_id=hash_pii(student_id),
            timestamp=datetime.utcnow(),
            context={
                "risk_score": result.confidence,
                "matched_patterns": result.matched_patterns
            },
            ip_address=get_request_ip()
        ))
        
        notify_counselor(student_id, result)
    
    return result
```

**Why**: Immutable audit trail for compliance. Log before action ensures we have record even if action fails.


### 8. Result Pattern (Explicit Error Handling)

Replace exceptions with explicit Result types for expected failures.

```python
from typing import Generic, TypeVar, Union
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E')

@dataclass(frozen=True)
class Success(Generic[T]):
    """Successful result."""
    value: T
    
    def is_success(self) -> bool:
        return True
    
    def is_failure(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        return self.value

@dataclass(frozen=True)
class Failure(Generic[E]):
    """Failed result."""
    error: E
    
    def is_success(self) -> bool:
        return False
    
    def is_failure(self) -> bool:
        return True
    
    def unwrap(self) -> None:
        raise ValueError(f"Cannot unwrap Failure: {self.error}")

Result = Union[Success[T], Failure[E]]

# Usage
def analyze_message(message: str) -> Result[DetectionResult, str]:
    """Analyze message for crisis markers."""
    
    if not message or not message.strip():
        return Failure("Empty message")
    
    if len(message) > 10000:
        return Failure("Message too long")
    
    try:
        result = detector.detect(message)
        return Success(result)
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return Failure(f"Detection error: {str(e)}")

# Caller handles explicitly
result = analyze_message(user_message)

if result.is_success():
    detection = result.unwrap()
    if detection.is_crisis:
        notify_counselor(detection)
else:
    logger.warning(f"Analysis failed: {result.error}")
    return fallback_response()
```

**Why**: Forces explicit error handling. No silent failures. Clear control flow.


### 9. Idempotency Pattern (Duplicate Prevention)

Ensure operations can be safely retried without side effects.

```python
from typing import Optional
import hashlib

class IdempotencyKey:
    """Generate idempotency keys for operations."""
    
    @staticmethod
    def generate(operation: str, *args) -> str:
        """Generate deterministic key from operation and arguments."""
        content = f"{operation}:{':'.join(str(arg) for arg in args)}"
        return hashlib.sha256(content.encode()).hexdigest()

class IdempotentNotificationService:
    """Notification service with idempotency."""
    
    def __init__(self, cache, notification_client):
        self._cache = cache
        self._client = notification_client
    
    def send_crisis_alert(
        self,
        counselor_id: str,
        student_id: str,
        risk_score: float,
        idempotency_key: Optional[str] = None
    ) -> bool:
        """Send crisis alert with idempotency protection."""
        
        # Generate key if not provided
        if idempotency_key is None:
            idempotency_key = IdempotencyKey.generate(
                "crisis_alert",
                counselor_id,
                student_id,
                int(risk_score * 100)
            )
        
        # Check if already sent
        cache_key = f"notification:{idempotency_key}"
        if self._cache.exists(cache_key):
            logger.info(
                "duplicate_notification_prevented",
                idempotency_key=idempotency_key,
                counselor_id=hash_pii(counselor_id)
            )
            return True  # Already sent, return success
        
        # Send notification
        try:
            self._client.send_sms(
                to=counselor_id,
                message=f"CRISIS ALERT: Student requires immediate attention. Risk score: {risk_score:.2f}"
            )
            
            # Mark as sent (24 hour TTL)
            self._cache.set(cache_key, "sent", ttl=86400)
            
            logger.info(
                "crisis_alert_sent",
                idempotency_key=idempotency_key,
                counselor_id=hash_pii(counselor_id),
                risk_score=risk_score
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}", exc_info=True)
            raise NotificationError("Failed to send crisis alert", original_error=e)

# Usage with retry
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def send_alert_with_retry(counselor_id: str, student_id: str, risk_score: float):
    """Send alert with automatic retry (idempotency prevents duplicates)."""
    notification_service.send_crisis_alert(counselor_id, student_id, risk_score)
```

**Why**: Safe retries without duplicate alerts. Event replay doesn't cause duplicate notifications.


### 10. Saga Pattern (Distributed Transactions)

Coordinate multi-step operations with compensating actions for rollback.

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class SagaStep(ABC):
    """Single step in a saga."""
    
    @abstractmethod
    def execute(self, context: dict) -> dict:
        """Execute the step, return updated context."""
        pass
    
    @abstractmethod
    def compensate(self, context: dict) -> None:
        """Undo the step if saga fails."""
        pass

class NotifyCounselorStep(SagaStep):
    """Send notification to counselor."""
    
    def execute(self, context: dict) -> dict:
        notification_id = notification_service.send_alert(
            counselor_id=context["counselor_id"],
            student_id=context["student_id"],
            risk_score=context["risk_score"]
        )
        context["notification_id"] = notification_id
        logger.info(f"Counselor notified: {notification_id}")
        return context
    
    def compensate(self, context: dict) -> None:
        """Cancel notification if possible."""
        if "notification_id" in context:
            notification_service.cancel(context["notification_id"])
            logger.info(f"Notification cancelled: {context['notification_id']}")

class LogCrisisEventStep(SagaStep):
    """Log crisis to audit trail."""
    
    def execute(self, context: dict) -> dict:
        audit_logger.log_crisis(
            student_id=context["student_id"],
            risk_score=context["risk_score"]
        )
        context["audit_logged"] = True
        logger.info("Crisis logged to audit trail")
        return context
    
    def compensate(self, context: dict) -> None:
        """Audit logs are immutable, log compensation event."""
        if context.get("audit_logged"):
            audit_logger.log_compensation(
                student_id=context["student_id"],
                reason="Saga rollback"
            )

class UpdateStudentRecordStep(SagaStep):
    """Update student's crisis history."""
    
    def execute(self, context: dict) -> dict:
        student_repo.add_crisis_event(
            student_id=context["student_id"],
            timestamp=datetime.utcnow(),
            risk_score=context["risk_score"]
        )
        context["record_updated"] = True
        logger.info("Student record updated")
        return context
    
    def compensate(self, context: dict) -> None:
        """Remove crisis event from student record."""
        if context.get("record_updated"):
            student_repo.remove_last_crisis_event(context["student_id"])
            logger.info("Student record rollback completed")


class CrisisSaga:
    """Orchestrate crisis response workflow."""
    
    def __init__(self, steps: List[SagaStep]):
        self.steps = steps
    
    def execute(self, initial_context: dict) -> Result[dict, str]:
        """Execute saga with automatic compensation on failure."""
        context = initial_context.copy()
        executed_steps: List[SagaStep] = []
        
        try:
            for step in self.steps:
                logger.info(f"Executing step: {step.__class__.__name__}")
                context = step.execute(context)
                executed_steps.append(step)
            
            logger.info("Saga completed successfully")
            return Success(context)
            
        except Exception as e:
            logger.error(f"Saga failed at step {step.__class__.__name__}: {e}", exc_info=True)
            
            # Compensate in reverse order
            for completed_step in reversed(executed_steps):
                try:
                    logger.info(f"Compensating: {completed_step.__class__.__name__}")
                    completed_step.compensate(context)
                except Exception as comp_error:
                    logger.critical(
                        f"Compensation failed for {completed_step.__class__.__name__}: {comp_error}",
                        exc_info=True
                    )
            
            return Failure(f"Saga failed: {str(e)}")

# Usage
crisis_saga = CrisisSaga([
    NotifyCounselorStep(),
    LogCrisisEventStep(),
    UpdateStudentRecordStep()
])

result = crisis_saga.execute({
    "student_id": student_id,
    "counselor_id": counselor_id,
    "risk_score": 0.95
})

if result.is_failure():
    logger.error(f"Crisis response failed: {result.error}")
    # Display crisis resources directly to student
    show_crisis_resources(student_id)
```

**Why**: Ensures consistency across distributed operations. Automatic rollback on failure. Clear audit trail.


## Data Privacy Patterns

### 11. PII Hashing Pattern (Zero PII in Logs)

Consistently hash PII before logging or storing in non-encrypted storage.

```python
import hashlib
from functools import lru_cache

class PIIHasher:
    """Centralized PII hashing with salt."""
    
    def __init__(self, salt: str):
        self._salt = salt
    
    @lru_cache(maxsize=10000)
    def hash(self, pii: str) -> str:
        """Hash PII with salt (cached for performance)."""
        if not pii:
            return "EMPTY"
        
        salted = f"{self._salt}:{pii}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]
    
    def hash_dict(self, data: dict, pii_fields: set[str]) -> dict:
        """Hash specified fields in dictionary."""
        result = data.copy()
        for field in pii_fields:
            if field in result:
                result[field] = self.hash(str(result[field]))
        return result

# Global instance (salt from environment)
_hasher = PIIHasher(salt=os.environ["PII_HASH_SALT"])

def hash_pii(pii: str) -> str:
    """Hash PII for logging (convenience function)."""
    return _hasher.hash(pii)

# Usage in logging
logger.info(
    "student_login",
    student_id=hash_pii(student_id),
    school_id=hash_pii(school_id),
    session_id=hash_pii(session_id)
)

# Usage in database queries
def get_student_conversations(student_id: str) -> List[dict]:
    """Get conversations with PII hashing."""
    results = db.query(
        "SELECT * FROM conversations WHERE student_id_hash = %s",
        (hash_pii(student_id),)
    )
    return list(results)
```

**Why**: Centralized hashing prevents PII leakage. Cached for performance. Consistent across system.


### 12. K-Anonymity Pattern (Aggregated Reporting)

Ensure all reports maintain k-anonymity (k≥5) to prevent re-identification.

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class AggregatedReport:
    """Report with k-anonymity guarantee."""
    dimension: str
    value: str
    count: int
    suppressed: bool
    
    def is_valid(self, k: int = 5) -> bool:
        """Check if report meets k-anonymity threshold."""
        return self.suppressed or self.count >= k

class KAnonymityReporter:
    """Generate reports with k-anonymity protection."""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def generate_report(
        self,
        data: List[dict],
        dimension: str
    ) -> List[AggregatedReport]:
        """Generate aggregated report with k-anonymity."""
        
        # Count by dimension
        counts: Dict[str, int] = {}
        for record in data:
            value = record.get(dimension, "unknown")
            counts[value] = counts.get(value, 0) + 1
        
        # Build report with suppression
        report = []
        for value, count in counts.items():
            if count < self.k:
                # Suppress groups smaller than k
                report.append(AggregatedReport(
                    dimension=dimension,
                    value="<suppressed>",
                    count=0,
                    suppressed=True
                ))
                logger.info(
                    "report_suppressed",
                    dimension=dimension,
                    value=hash_pii(value),
                    count=count,
                    reason=f"Below k-anonymity threshold (k={self.k})"
                )
            else:
                report.append(AggregatedReport(
                    dimension=dimension,
                    value=value,
                    count=count,
                    suppressed=False
                ))
        
        return report
    
    def crisis_rate_by_school(self, school_ids: List[str]) -> Dict[str, float]:
        """Calculate crisis rate by school with k-anonymity."""
        
        school_data = []
        for school_id in school_ids:
            crisis_count = db.count_crises(school_id)
            student_count = db.count_students(school_id)
            
            school_data.append({
                "school_id": school_id,
                "crisis_count": crisis_count,
                "student_count": student_count
            })
        
        # Suppress schools with < k students
        results = {}
        for school in school_data:
            if school["student_count"] < self.k:
                logger.warning(
                    "school_report_suppressed",
                    school_id=hash_pii(school["school_id"]),
                    student_count=school["student_count"]
                )
                continue
            
            rate = school["crisis_count"] / school["student_count"]
            results[school["school_id"]] = rate
        
        return results

# Usage
reporter = KAnonymityReporter(k=5)

# Generate report
crisis_data = db.get_crisis_events(days=30)
report = reporter.generate_report(crisis_data, dimension="grade_level")

# Only show non-suppressed data
for item in report:
    if not item.suppressed:
        print(f"{item.dimension}: {item.value} - {item.count} events")
```

**Why**: Prevents re-identification in reports. Automatic suppression of small groups. Compliance with privacy regulations.


## Testing Patterns

### 13. Golden Test Pattern (Regression Prevention)

Maintain curated test sets for critical functionality.

```python
import json
from pathlib import Path
from typing import List

@dataclass
class GoldenTestCase:
    """Single golden test case."""
    id: str
    message: str
    expected_crisis: bool
    expected_confidence_min: float
    expected_patterns: List[str]
    notes: str

class GoldenTestRunner:
    """Run golden test suite."""
    
    def __init__(self, test_file: Path):
        self.test_cases = self._load_test_cases(test_file)
    
    def _load_test_cases(self, test_file: Path) -> List[GoldenTestCase]:
        """Load golden test cases from JSON."""
        with open(test_file) as f:
            data = json.load(f)
        
        return [
            GoldenTestCase(
                id=case["id"],
                message=case["message"],
                expected_crisis=case["expected_crisis"],
                expected_confidence_min=case["expected_confidence_min"],
                expected_patterns=case["expected_patterns"],
                notes=case.get("notes", "")
            )
            for case in data["test_cases"]
        ]
    
    def run_all(self, detector) -> dict:
        """Run all golden tests and return results."""
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "failures": []
        }
        
        for test_case in self.test_cases:
            result = detector.detect(test_case.message)
            
            # Check expectations
            passed = True
            failure_reasons = []
            
            if result.is_crisis != test_case.expected_crisis:
                passed = False
                failure_reasons.append(
                    f"Expected crisis={test_case.expected_crisis}, got {result.is_crisis}"
                )
            
            if result.confidence < test_case.expected_confidence_min:
                passed = False
                failure_reasons.append(
                    f"Confidence {result.confidence} below minimum {test_case.expected_confidence_min}"
                )
            
            for pattern in test_case.expected_patterns:
                if pattern not in result.matched_patterns:
                    passed = False
                    failure_reasons.append(f"Missing expected pattern: {pattern}")
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "id": test_case.id,
                    "message": test_case.message,
                    "reasons": failure_reasons,
                    "notes": test_case.notes
                })
        
        return results

# Golden test file format
"""
{
  "test_cases": [
    {
      "id": "explicit_suicidal_ideation_001",
      "message": "I want to kill myself",
      "expected_crisis": true,
      "expected_confidence_min": 0.95,
      "expected_patterns": ["suicidal_ideation", "explicit_intent"],
      "notes": "Direct statement of suicidal intent"
    },
    {
      "id": "hyperbole_filter_001",
      "message": "This homework is killing me",
      "expected_crisis": false,
      "expected_confidence_min": 0.0,
      "expected_patterns": ["sarcasm_filtered"],
      "notes": "Common teenage hyperbole, should not trigger"
    }
  ]
}
"""

# Usage in CI/CD
def test_golden_suite():
    """Run golden test suite (100% must pass)."""
    runner = GoldenTestRunner(Path("tests/golden/crisis_detection.json"))
    detector = CrisisDetector()
    
    results = runner.run_all(detector)
    
    assert results["failed"] == 0, f"Golden tests failed: {results['failures']}"
    assert results["passed"] == results["total"]
```

**Why**: Prevents regressions. Documents expected behavior. Easy to add new cases from production incidents.


### 14. Property-Based Testing (Invariant Verification)

Test system invariants across wide input ranges.

```python
from hypothesis import given, strategies as st
import hypothesis

# Define strategies for test data generation
crisis_messages = st.text(min_size=1, max_size=1000)
risk_scores = st.floats(min_value=0.0, max_value=1.0)

@given(message=crisis_messages)
def test_detector_never_crashes(message: str):
    """Detector must handle any input without crashing."""
    detector = CrisisDetector()
    
    # Should never raise exception
    result = detector.detect(message)
    
    # Invariants that must always hold
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.matched_patterns, list)

@given(message=crisis_messages)
def test_pii_never_in_logs(message: str, caplog):
    """PII must never appear in logs."""
    student_id = "student_12345"
    
    detector = CrisisDetector()
    result = detector.detect(message)
    
    # Check all log messages
    for record in caplog.records:
        assert student_id not in record.message
        assert message not in record.message

@given(risk_score=risk_scores)
def test_notification_idempotency(risk_score: float):
    """Sending same notification twice should not duplicate."""
    service = IdempotentNotificationService(cache, client)
    
    # Send twice with same parameters
    result1 = service.send_crisis_alert("counselor_1", "student_1", risk_score)
    result2 = service.send_crisis_alert("counselor_1", "student_1", risk_score)
    
    # Both should succeed
    assert result1 is True
    assert result2 is True
    
    # But only one notification sent
    assert client.send_count == 1
```

**Why**: Finds edge cases humans miss. Verifies invariants hold across all inputs. Catches regressions early.


## Performance Patterns

### 15. Caching Pattern (Sub-50ms Detection)

Cache expensive operations while maintaining freshness.

```python
from functools import lru_cache
from typing import Optional
import time

class CachedEmbeddingService:
    """Embedding service with multi-layer caching."""
    
    def __init__(self, model, redis_client):
        self._model = model
        self._redis = redis_client
        self._local_cache_hits = 0
        self._redis_cache_hits = 0
        self._model_calls = 0
    
    @lru_cache(maxsize=1000)
    def _local_cache_get(self, text: str) -> Optional[list]:
        """In-memory LRU cache (fastest)."""
        return None  # Placeholder, actual cache in decorator
    
    def get_embedding(self, text: str) -> list:
        """Get embedding with multi-layer caching."""
        
        # Layer 1: In-memory cache (< 1ms)
        cached = self._local_cache_get(text)
        if cached is not None:
            self._local_cache_hits += 1
            return cached
        
        # Layer 2: Redis cache (< 5ms)
        cache_key = f"embedding:{hash_pii(text)}"
        redis_cached = self._redis.get(cache_key)
        if redis_cached:
            self._redis_cache_hits += 1
            embedding = json.loads(redis_cached)
            # Populate local cache
            self._local_cache_get(text)
            return embedding
        
        # Layer 3: Model inference (20-50ms)
        start = time.time()
        embedding = self._model.encode(text)
        latency = (time.time() - start) * 1000
        
        self._model_calls += 1
        
        # Cache in Redis (24 hour TTL)
        self._redis.setex(
            cache_key,
            86400,
            json.dumps(embedding.tolist())
        )
        
        logger.info(
            "embedding_generated",
            latency_ms=latency,
            cache_hits_local=self._local_cache_hits,
            cache_hits_redis=self._redis_cache_hits,
            model_calls=self._model_calls
        )
        
        return embedding
    
    def get_cache_stats(self) -> dict:
        """Get cache performance metrics."""
        total_requests = (
            self._local_cache_hits +
            self._redis_cache_hits +
            self._model_calls
        )
        
        return {
            "total_requests": total_requests,
            "local_cache_hit_rate": self._local_cache_hits / total_requests if total_requests > 0 else 0,
            "redis_cache_hit_rate": self._redis_cache_hits / total_requests if total_requests > 0 else 0,
            "model_call_rate": self._model_calls / total_requests if total_requests > 0 else 0
        }

# Usage
embedding_service = CachedEmbeddingService(model, redis_client)

def detect_crisis_semantic(message: str) -> DetectionResult:
    """Semantic detection with caching."""
    start = time.time()
    
    # Get embedding (cached)
    embedding = embedding_service.get_embedding(message)
    
    # Compare with crisis patterns
    max_similarity = 0.0
    matched_pattern = None
    
    for pattern_name, pattern_embedding in crisis_patterns.items():
        similarity = cosine_similarity(embedding, pattern_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            matched_pattern = pattern_name
    
    latency = (time.time() - start) * 1000
    
    logger.info(
        "semantic_detection_complete",
        latency_ms=latency,
        max_similarity=max_similarity
    )
    
    return DetectionResult(
        is_crisis=max_similarity > 0.80,
        confidence=max_similarity,
        matched_patterns=[matched_pattern] if matched_pattern else [],
        evidence=f"Semantic similarity: {max_similarity:.2f}"
    )
```

**Why**: Meets <50ms latency requirement. Multi-layer caching optimizes for common cases. Observable cache performance.


### 16. Bulkhead Pattern (Failure Isolation)

Isolate failures to prevent cascading across system.

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Any

class Bulkhead:
    """Isolate operations with dedicated resource pools."""
    
    def __init__(self, name: str, max_workers: int, timeout: float):
        self.name = name
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"bulkhead-{name}"
        )
        self.timeout = timeout
        self._active_tasks = 0
        self._rejected_tasks = 0
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in isolated bulkhead."""
        
        if self._active_tasks >= self.executor._max_workers:
            self._rejected_tasks += 1
            logger.warning(
                "bulkhead_saturated",
                bulkhead=self.name,
                active_tasks=self._active_tasks,
                rejected_tasks=self._rejected_tasks
            )
            raise BulkheadSaturatedError(f"Bulkhead {self.name} is saturated")
        
        self._active_tasks += 1
        
        try:
            future = self.executor.submit(func, *args, **kwargs)
            result = future.result(timeout=self.timeout)
            return result
        except TimeoutError:
            logger.error(
                "bulkhead_timeout",
                bulkhead=self.name,
                timeout=self.timeout
            )
            raise
        finally:
            self._active_tasks -= 1

# Create bulkheads for different operations
llm_bulkhead = Bulkhead(name="llm", max_workers=10, timeout=5.0)
notification_bulkhead = Bulkhead(name="notification", max_workers=20, timeout=3.0)
analytics_bulkhead = Bulkhead(name="analytics", max_workers=5, timeout=10.0)

# Usage
def process_message(message: str, student_id: str):
    """Process message with isolated operations."""
    
    # Crisis detection (fast, critical path)
    crisis_result = detect_crisis(message)
    
    if crisis_result.is_crisis:
        # Notification (isolated, won't block if LLM fails)
        try:
            notification_bulkhead.execute(
                notify_counselor,
                student_id,
                crisis_result
            )
        except BulkheadSaturatedError:
            logger.critical("Notification bulkhead saturated, using fallback")
            # Show crisis resources directly to student
            display_crisis_resources(student_id)
    
    # LLM response (isolated, can fail without affecting crisis detection)
    try:
        response = llm_bulkhead.execute(
            generate_response,
            message,
            crisis_result
        )
    except (BulkheadSaturatedError, TimeoutError):
        logger.warning("LLM bulkhead issue, using fallback")
        response = get_fallback_response(message)
    
    # Analytics (isolated, non-critical)
    try:
        analytics_bulkhead.execute(
            record_interaction,
            student_id,
            message,
            response
        )
    except Exception as e:
        logger.warning(f"Analytics failed (non-critical): {e}")
    
    return response
```

**Why**: LLM failure doesn't block notifications. Notification saturation doesn't block crisis detection. Clear resource limits.


## Observability Patterns

### 17. Structured Logging Pattern (Traceable Operations)

Log with structured context for debugging and auditing.

```python
import structlog
from contextvars import ContextVar
from typing import Optional
import uuid

# Context variables for request tracing
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
student_id_var: ContextVar[Optional[str]] = ContextVar("student_id", default=None)

def configure_logging():
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

logger = structlog.get_logger()

class RequestContext:
    """Context manager for request-scoped logging."""
    
    def __init__(self, student_id: str):
        self.request_id = str(uuid.uuid4())
        self.student_id_hash = hash_pii(student_id)
    
    def __enter__(self):
        request_id_var.set(self.request_id)
        student_id_var.set(self.student_id_hash)
        
        structlog.contextvars.bind_contextvars(
            request_id=self.request_id,
            student_id=self.student_id_hash
        )
        
        logger.info("request_started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(
                "request_failed",
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )
        else:
            logger.info("request_completed")
        
        structlog.contextvars.clear_contextvars()
        request_id_var.set(None)
        student_id_var.set(None)

# Usage
def handle_message(student_id: str, message: str) -> str:
    """Handle message with request context."""
    
    with RequestContext(student_id):
        logger.info("message_received", message_length=len(message))
        
        # Crisis detection
        start = time.time()
        result = detector.detect(message)
        detection_latency = (time.time() - start) * 1000
        
        logger.info(
            "crisis_detection_complete",
            is_crisis=result.is_crisis,
            confidence=result.confidence,
            matched_patterns=result.matched_patterns,
            latency_ms=detection_latency
        )
        
        if result.is_crisis:
            logger.warning(
                "crisis_detected",
                risk_score=result.confidence,
                evidence=result.evidence
            )
            notify_counselor(student_id, result)
        
        # Generate response
        response = generate_response(message, result)
        
        logger.info(
            "response_generated",
            response_length=len(response)
        )
        
        return response

# All logs will include request_id and student_id automatically
# Example output:
# {
#   "event": "crisis_detected",
#   "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
#   "student_id": "a1b2c3d4e5f6",
#   "risk_score": 0.95,
#   "evidence": "Matched pattern: suicidal_ideation",
#   "timestamp": "2026-01-20T10:30:45.123Z",
#   "level": "warning"
# }
```

**Why**: Every log has request context. Easy to trace entire request flow. Structured data for analysis.


### 18. Metrics Pattern (Observable Performance)

Instrument critical paths with metrics for monitoring.

```python
from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    tags: Dict[str, str]
    timestamp: float

class MetricsCollector:
    """Collect and publish metrics."""
    
    def __init__(self):
        self._metrics: list[Metric] = []
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record gauge metric (current value)."""
        self._metrics.append(Metric(
            name=name,
            value=value,
            tags=tags or {},
            timestamp=time.time()
        ))
    
    def increment(self, name: str, tags: Dict[str, str] = None):
        """Increment counter metric."""
        self.gauge(name, 1.0, tags)
    
    def timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record timing metric."""
        self.gauge(name, duration_ms, tags)
    
    def flush(self):
        """Publish metrics to monitoring system."""
        # Send to CloudWatch, Datadog, etc.
        pass

metrics = MetricsCollector()

class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metric_name: str, tags: Dict[str, str] = None):
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        # Add success/failure tag
        self.tags["success"] = "false" if exc_type else "true"
        
        metrics.timing(self.metric_name, duration_ms, self.tags)
        
        # Log if exceeds SLA
        if self.metric_name == "crisis_detection" and duration_ms > 50:
            logger.warning(
                "sla_violation",
                metric=self.metric_name,
                duration_ms=duration_ms,
                sla_ms=50
            )

# Usage
def detect_crisis_with_metrics(message: str) -> DetectionResult:
    """Crisis detection with metrics."""
    
    with MetricsTimer("crisis_detection", tags={"strategy": "regex"}):
        result = regex_detector.detect(message)
    
    # Record outcome
    if result.is_crisis:
        metrics.increment("crisis_detected", tags={
            "confidence_bucket": f"{int(result.confidence * 10) * 10}%"
        })
    
    return result

def notify_counselor_with_metrics(student_id: str, result: DetectionResult):
    """Notification with metrics."""
    
    with MetricsTimer("notification_delivery", tags={"channel": "sms"}):
        notification_service.send_alert(student_id, result)
    
    metrics.increment("counselor_notified", tags={
        "risk_level": "crisis" if result.confidence > 0.9 else "caution"
    })

# Dashboard queries:
# - P95 crisis detection latency: percentile(crisis_detection, 95)
# - Crisis detection rate: rate(crisis_detected)
# - Notification success rate: rate(notification_delivery{success=true}) / rate(notification_delivery)
```

**Why**: Real-time performance monitoring. SLA violation alerts. Data-driven optimization.


## Anti-Patterns to Avoid

### ❌ God Object Anti-Pattern

**Bad**: Single class that does everything.

```python
# ❌ DON'T DO THIS
class CrisisSystem:
    """Does everything - impossible to test or debug."""
    
    def __init__(self):
        self.db = Database()
        self.llm = LLMClient()
        self.notifier = NotificationService()
        self.cache = RedisCache()
    
    def process_message(self, message: str, student_id: str):
        # 500 lines of mixed concerns
        # Detection + notification + logging + analytics + LLM
        pass
```

**Good**: Separate concerns with clear boundaries.

```python
# ✅ DO THIS
class CrisisDetector:
    """Single responsibility: detect crisis markers."""
    def detect(self, message: str) -> DetectionResult:
        pass

class NotificationService:
    """Single responsibility: send notifications."""
    def notify_counselor(self, student_id: str, result: DetectionResult):
        pass

class CrisisOrchestrator:
    """Coordinate components (thin orchestration layer)."""
    def __init__(self, detector, notifier, logger):
        self.detector = detector
        self.notifier = notifier
        self.logger = logger
    
    def process_message(self, message: str, student_id: str):
        result = self.detector.detect(message)
        if result.is_crisis:
            self.notifier.notify_counselor(student_id, result)
            self.logger.log_crisis(student_id, result)
        return result
```

### ❌ Silent Failure Anti-Pattern

**Bad**: Swallowing exceptions without logging.

```python
# ❌ DON'T DO THIS
def notify_counselor(student_id: str):
    try:
        send_sms(student_id)
    except:
        pass  # Silent failure - counselor never notified!
```

**Good**: Log before raising, fail loud.

```python
# ✅ DO THIS
def notify_counselor(student_id: str):
    try:
        send_sms(student_id)
    except SMSServiceError as e:
        logger.error(
            "notification_failed",
            student_id=hash_pii(student_id),
            error=str(e),
            exc_info=True
        )
        # Show crisis resources directly to student
        display_crisis_resources(student_id)
        raise NotificationError("Failed to notify counselor", original_error=e)
```

### ❌ Mutable Shared State Anti-Pattern

**Bad**: Mutable state shared across requests.

```python
# ❌ DON'T DO THIS
class ConversationManager:
    def __init__(self):
        self.conversations = {}  # Shared mutable state
    
    def add_message(self, session_id: str, message: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append(message)  # Race condition!
```

**Good**: Immutable data structures.

```python
# ✅ DO THIS
@dataclass(frozen=True)
class Conversation:
    session_id: str
    messages: Tuple[str, ...]
    
    def with_message(self, message: str) -> 'Conversation':
        return Conversation(
            session_id=self.session_id,
            messages=self.messages + (message,)
        )
```

### ❌ Magic Numbers Anti-Pattern

**Bad**: Hard-coded thresholds in code.

```python
# ❌ DON'T DO THIS
def is_crisis(risk_score: float) -> bool:
    return risk_score > 0.85  # What is 0.85? Why not 0.80?
```

**Good**: Named constants with documentation.

```python
# ✅ DO THIS
from enum import Enum

class RiskThreshold(Enum):
    """Crisis detection thresholds (validated against MentalChat16K dataset)."""
    CRISIS = 0.85  # 99.5% recall, 8% false positive rate
    CAUTION = 0.60  # 95% recall, 15% false positive rate
    SAFE = 0.30  # Baseline threshold

def is_crisis(risk_score: float) -> bool:
    return risk_score > RiskThreshold.CRISIS.value
```

### ❌ String-Based Enums Anti-Pattern

**Bad**: String literals for fixed values.

```python
# ❌ DON'T DO THIS
risk_level = "crisis"  # Typo: "crysis" won't be caught
if risk_level == "crisis":
    notify_counselor()
```

**Good**: Proper Enums with type safety.

```python
# ✅ DO THIS
from enum import Enum

class RiskLevel(Enum):
    CRISIS = "crisis"
    CAUTION = "caution"
    SAFE = "safe"

risk_level = RiskLevel.CRISIS  # Type-safe, autocomplete works
if risk_level == RiskLevel.CRISIS:
    notify_counselor()
```


## Pattern Selection Guide

### When to Use Each Pattern

| Pattern | Use When | Don't Use When |
|---------|----------|----------------|
| **Strategy** | Multiple algorithms for same task | Only one implementation |
| **Chain of Responsibility** | Multi-stage processing with filters | Simple linear flow |
| **Observer** | Decoupled event handling | Tight coupling acceptable |
| **Circuit Breaker** | External dependency can fail | Internal, reliable operations |
| **Builder** | Complex immutable objects | Simple data structures |
| **Repository** | Abstract data access | Direct DB access is fine |
| **Audit Trail** | Compliance-critical operations | Non-sensitive operations |
| **Result** | Expected failures | Exceptional errors only |
| **Idempotency** | Operations can be retried | One-time operations |
| **Saga** | Multi-step distributed transactions | Single-service operations |
| **Bulkhead** | Isolate failure domains | Single-threaded operations |

### Pattern Combinations

**Crisis Detection Pipeline**:
- Strategy (detection algorithms)
- Chain of Responsibility (sarcasm → keyword → semantic)
- Observer (notify counselor, log, analytics)
- Circuit Breaker (LLM fallback)
- Audit Trail (compliance logging)

**Notification System**:
- Idempotency (prevent duplicates)
- Circuit Breaker (SMS service failure)
- Saga (multi-channel notification)
- Bulkhead (isolate from main flow)

**Data Access**:
- Repository (abstract storage)
- PII Hashing (privacy)
- K-Anonymity (reporting)
- Caching (performance)

## Quick Reference

### Safety-Critical Checklist

Before deploying crisis detection code:

- [ ] Strategy pattern for pluggable detection
- [ ] Observer pattern for decoupled notifications
- [ ] Audit trail for all crisis events
- [ ] Circuit breaker for external dependencies
- [ ] Idempotency for notifications
- [ ] PII hashing in all logs
- [ ] Structured logging with request context
- [ ] Metrics for latency and success rate
- [ ] Golden tests for regression prevention
- [ ] Property-based tests for invariants
- [ ] 100% test coverage
- [ ] Passes 60-second litmus test

### Performance Checklist

For <50ms crisis detection:

- [ ] Multi-layer caching (memory → Redis → model)
- [ ] Compiled regex patterns
- [ ] Pre-loaded embeddings
- [ ] Bulkhead isolation
- [ ] Metrics for P95/P99 latency
- [ ] Circuit breaker timeouts
- [ ] Connection pooling

### Compliance Checklist

For FERPA/COPPA compliance:

- [ ] Zero PII in logs (use hash_pii())
- [ ] Immutable audit trail
- [ ] K-anonymity (k≥5) in reports
- [ ] Field-level encryption for PII
- [ ] RBAC enforcement
- [ ] Data retention policies
- [ ] Consent tracking

---

**Remember**: These patterns exist to support the core tenets. When in doubt, choose the pattern that makes the code more traceable, testable, and safe.

The stakes are high: mental health + minors = zero tolerance for bugs.
