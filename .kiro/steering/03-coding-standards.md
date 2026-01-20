---
inclusion: always
---

# PsyFlo Coding Standards

These standards ensure code safety, maintainability, and compliance with project tenets.

## Language & Framework Standards

### Python (Backend Services)
- **Version**: Python 3.11+
- **Type Hints**: Required for all function signatures
- **Formatting**: Black (line length: 100)
- **Linting**: Ruff + mypy (strict mode)
- **Testing**: pytest with 100% coverage for safety-critical code

### TypeScript (Frontend)
- **Version**: TypeScript 5.0+
- **Framework**: React 18+ with strict mode
- **Formatting**: Prettier
- **Linting**: ESLint with strict rules
- **Testing**: Vitest + React Testing Library

## Safety-Critical Code Requirements

### 1. Zero PII in Logs
```python
# ❌ NEVER DO THIS
logger.info(f"Student {student_id} logged in")
logger.error(f"Failed to load data for {student_email}")

# ✅ ALWAYS DO THIS
from utils.privacy import hash_pii

logger.info(f"Student {hash_pii(student_id)} logged in")
logger.error(f"Failed to load data for {hash_pii(student_email)}")
```

### 2. Explicit Exception Handling
```python
# ❌ NEVER DO THIS
try:
    result = risky_operation()
except:
    pass

# ✅ ALWAYS DO THIS
from exceptions import SafetyServiceError

try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise SafetyServiceError(
        "Failed to analyze message for crisis markers",
        original_error=e,
        context={"session_id": hash_pii(session_id)}
    )
```

### 3. Use Enums for Fixed Values
```python
# ❌ NEVER DO THIS
risk_level = "crisis"  # String literals are error-prone

# ✅ ALWAYS DO THIS
from enums import RiskLevel

risk_level = RiskLevel.CRISIS
```

### 4. Immutable Data Structures
```python
# ❌ AVOID MUTABLE STATE
class ConversationSession:
    def __init__(self):
        self.messages = []  # Mutable list
    
    def add_message(self, msg):
        self.messages.append(msg)  # Mutates state

# ✅ USE IMMUTABLE PATTERNS
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ConversationSession:
    messages: Tuple[Message, ...]
    
    def with_message(self, msg: Message) -> 'ConversationSession':
        return ConversationSession(
            messages=self.messages + (msg,)
        )
```

## Required Type Hints

```python
# ✅ All functions must have type hints
from typing import Optional, List
from models import Student, RiskAssessment

def assess_risk(
    student_id: str,
    message: str,
    history: Optional[List[str]] = None
) -> RiskAssessment:
    """
    Assess crisis risk for a student message.
    
    Args:
        student_id: Hashed student identifier
        message: Student's message content
        history: Optional conversation history
        
    Returns:
        RiskAssessment with score and evidence
        
    Raises:
        SafetyServiceError: If risk assessment fails
    """
    # Implementation
    pass
```

## Logging Standards

### Log Levels
- **DEBUG**: Detailed diagnostic information (never in production)
- **INFO**: General informational messages (no PII)
- **WARNING**: Unexpected but handled situations
- **ERROR**: Error events that still allow operation to continue
- **CRITICAL**: Severe errors causing service shutdown

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

# ✅ Use structured logging with context
logger.info(
    "crisis_detected",
    session_id=hash_pii(session_id),
    risk_score=0.95,
    matched_patterns=["suicidal_ideation", "intent_with_plan"],
    latency_ms=42
)
```

### What to Log
- ✅ All crisis detections (with hashed IDs)
- ✅ All counselor notifications sent
- ✅ Service health metrics
- ✅ Error conditions with context
- ❌ Student names, emails, or identifiable info
- ❌ Full message content (only hashed references)
- ❌ Passwords, tokens, or secrets

## Testing Requirements

### Safety-Critical Code: 100% Coverage
```python
# All crisis detection code must have comprehensive tests
def test_crisis_detection_suicidal_ideation():
    """Test detection of explicit suicidal ideation."""
    message = "I want to end my life"
    result = safety_service.analyze(message)
    
    assert result.risk_level == RiskLevel.CRISIS
    assert result.confidence >= 0.90
    assert "suicidal_ideation" in result.matched_patterns

def test_crisis_detection_hyperbole_filter():
    """Test that teenage hyperbole doesn't trigger false positive."""
    message = "This homework is killing me"
    result = safety_service.analyze(message)
    
    assert result.risk_level != RiskLevel.CRISIS
    assert result.sarcasm_filtered is True
```

### Golden Test Sets
- Use MentalChat16K dataset for regression testing
- Maintain custom crisis test set (target: 10,000+ cases)
- Test adversarial patterns (coded language, leetspeak)

### Integration Tests
```python
@pytest.mark.integration
async def test_end_to_end_crisis_flow():
    """Test complete crisis detection and notification flow."""
    # Send crisis message
    response = await chat_service.send_message(
        session_id=test_session_id,
        message="I'm going to hurt myself tonight"
    )
    
    # Verify crisis protocol triggered
    assert response.is_crisis_override is True
    assert "crisis resources" in response.message.lower()
    
    # Verify counselor notified
    await asyncio.sleep(1)  # Allow event propagation
    notifications = await get_test_notifications()
    assert len(notifications) == 1
    assert notifications[0].priority == "CRITICAL"
```

## Error Handling Patterns

### Custom Exceptions
```python
# exceptions.py
class PsyFloException(Exception):
    """Base exception for all PsyFlo errors."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

class SafetyServiceError(PsyFloException):
    """Crisis detection service error."""
    pass

class NotificationError(PsyFloException):
    """Counselor notification error."""
    pass
```

### Graceful Degradation
```python
from circuit_breaker import CircuitBreaker

@CircuitBreaker(failure_threshold=5, timeout=30)
async def call_llm_service(prompt: str) -> str:
    """Call LLM with circuit breaker protection."""
    try:
        return await llm_client.generate(prompt)
    except LLMServiceError as e:
        logger.warning("LLM service unavailable, using fallback")
        return get_fallback_response(prompt)
```

## Configuration Management

### Crisis Patterns in YAML
```yaml
# config/crisis_patterns.yaml
crisis_keywords:
  suicidal_ideation:
    patterns:
      - "want to die"
      - "kill myself"
      - "end my life"
      - "not worth living"
    confidence: 0.95
    
  self_harm:
    patterns:
      - "cut myself"
      - "hurt myself"
      - "self harm"
    confidence: 0.85
```

### Load Configuration
```python
import yaml
from pathlib import Path

def load_crisis_patterns() -> dict:
    """Load crisis patterns from YAML config."""
    config_path = Path(__file__).parent / "config" / "crisis_patterns.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
```

## Code Review Checklist

Before merging any PR, verify:

- [ ] All functions have type hints
- [ ] No PII in log statements (use `hash_pii()`)
- [ ] No bare `except:` clauses
- [ ] Enums used for fixed values
- [ ] Safety-critical code has 100% test coverage
- [ ] Error messages include context
- [ ] Passes 60-second litmus test
- [ ] Documentation updated
- [ ] No secrets in code

## Performance Guidelines

### Crisis Detection: <50ms
- Use compiled regex patterns
- Cache embedding models in memory
- Avoid database calls in hot path

### API Response: <2s P95
- Use Redis for session state
- Implement request timeouts
- Monitor P95/P99 latencies

### Database Queries
- Index all foreign keys
- Use connection pooling
- Implement query timeouts
- Monitor slow query log

## Security Checklist

- [ ] All API endpoints require authentication
- [ ] RBAC enforced at service layer
- [ ] Input validation on all user data
- [ ] SQL injection prevention (use parameterized queries)
- [ ] XSS prevention (sanitize outputs)
- [ ] CSRF tokens on state-changing operations
- [ ] Rate limiting on all endpoints
- [ ] TLS 1.3 for all connections

---

**Remember**: The stakes are high. Mental health + minors = zero tolerance for bugs.
