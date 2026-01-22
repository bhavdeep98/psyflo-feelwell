# Conversation Agent

Empathetic AI conversation agent using LangChain and LangGraph for mental health support.

## Overview

The Conversation Agent generates empathetic, contextually-aware responses to student messages while maintaining safety protocols.

**Key Features:**
- LangGraph-based conversation flow (explicit state management)
- Crisis-aware response generation
- Conversation history context
- Hard-coded crisis overrides (Tenet #1)
- Fail-loud initialization (Tenet #4)
- Full observability (Tenet #10)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversation Agent                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │   Assess     │─────▶│   Crisis     │─────▶│   END    │  │
│  │   Context    │      │   Override   │      │          │  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
│         │                                                    │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                            ┌──────────┐  │
│  │   Generate   │───────────────────────────▶│   END    │  │
│  │   Response   │                            │          │  │
│  └──────────────┘                            └──────────┘  │
│         │                                                    │
│         │ (uses LLM)                                        │
│         ▼                                                    │
│  ┌──────────────┐                                          │
│  │   OpenAI     │                                          │
│  │   GPT-4o     │                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### Tenet #1: Safety First
- Crisis messages trigger hard-coded response (cannot be overridden by LLM)
- Crisis detection happens BEFORE conversation agent is called
- Agent receives risk assessment as input

### Tenet #3: Explicit Over Clever
- LangGraph makes conversation flow explicit and traceable
- No hidden state or clever abstractions
- Clear node transitions

### Tenet #4: Fail Loud, Fail Early
- Initialization fails if OPENAI_API_KEY is missing
- LLM failures raise exceptions (no silent fallbacks)
- All errors logged before raising

### Tenet #7: Immutability by Default
- ConversationContext is frozen dataclass
- State transitions create new state objects

### Tenet #8: Engagement Before Intervention
- Empathetic, active listening approach
- Age-appropriate language
- Builds trust through consistency

### Tenet #10: Observable Systems
- All state transitions logged
- Crisis overrides logged with context
- Response generation latency tracked

## Usage

### Basic Usage

```python
from src.conversation import ConversationAgent, ConversationContext

# Initialize agent (requires OPENAI_API_KEY)
agent = ConversationAgent()

# Create conversation context
context = ConversationContext(
    session_id="student-123",
    risk_level="SAFE",
    risk_score=0.1,
    matched_patterns=[],
    conversation_history=[]
)

# Generate response
response = await agent.generate_response(
    message="I had a bad day at school",
    context=context
)
```

### With Conversation History

```python
context = ConversationContext(
    session_id="student-123",
    risk_level="CAUTION",
    risk_score=0.65,
    matched_patterns=["sadness"],
    conversation_history=[
        {"role": "student", "content": "I've been feeling down"},
        {"role": "assistant", "content": "I hear that. Want to talk about it?"}
    ]
)

response = await agent.generate_response(
    message="Yeah, I failed my math test",
    context=context
)
```

### Crisis Override

```python
# Crisis context automatically triggers hard-coded response
crisis_context = ConversationContext(
    session_id="student-123",
    risk_level="CRISIS",
    risk_score=0.95,
    matched_patterns=["suicidal_ideation"],
    conversation_history=[]
)

response = await agent.generate_response(
    message="I want to die",
    context=crisis_context
)

# Response will be hard-coded crisis protocol:
# - Counselor notification message
# - 988 hotline
# - Crisis Text Line
# - Resources
```

## Configuration

### Environment Variables (.env file)

Create a `.env` file in the backend directory:

```bash
# backend/.env

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
```

Or copy from example:

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add your OpenAI API key
```

### Custom Configuration

You can override .env settings programmatically:

```python
agent = ConversationAgent(
    model_name="gpt-4",
    temperature=0.5,
    api_key="sk-..."  # Override .env
)
```

## System Prompts

### Base Prompt (SAFE)

```
You are a supportive mental health AI assistant for high school students.

Your role:
- Listen actively and empathetically
- Validate their feelings without judgment
- Ask open-ended questions to understand better
- Provide gentle encouragement
- Build trust through consistency

Guidelines:
- Use warm, conversational language (like talking to a friend)
- Keep responses concise (2-3 sentences)
- Reflect their emotions back to them
- Never diagnose or prescribe treatment
- Never claim to be a therapist or counselor
- If they need professional help, encourage them to talk to their school counselor

Tone: Warm, supportive, non-judgmental, age-appropriate
```

### Enhanced Prompt (CAUTION)

Base prompt + additional guidance:

```
IMPORTANT: This student is showing some concerning signs. Be extra attentive:
- Gently explore what's troubling them
- Validate their struggles
- Subtly encourage them to reach out to their counselor
- Watch for escalation
```

## Error Handling

### Initialization Failures

```python
# Missing API key
try:
    agent = ConversationAgent()
except RuntimeError as e:
    # "OPENAI_API_KEY not found. Set environment variable or pass api_key parameter."
    logger.error("agent_initialization_failed", error=str(e))
```

### Response Generation Failures

```python
try:
    response = await agent.generate_response(message, context)
except RuntimeError as e:
    # "Conversation graph failed: <error details>"
    logger.error("response_generation_failed", error=str(e))
    # Tenet #11: Graceful degradation at service layer
    # Backend should display crisis resources
```

## Testing

```bash
# Run tests
pytest tests/test_conversation_agent.py -v

# Run with coverage
pytest tests/test_conversation_agent.py --cov=src/conversation --cov-report=html
```

## Integration with Backend

The backend integrates the conversation agent with graceful degradation:

```python
# backend/main.py
try:
    response_text = await conversation_agent.generate_response(
        message=request.message,
        context=context
    )
except Exception as e:
    # Tenet #11: Graceful degradation
    logger.error("conversation_agent_failed", error=str(e))
    # Display crisis resources directly to student
    response_text = _get_service_unavailable_response()
```

## Performance

**Target Latency:**
- P50: <1s
- P95: <2s
- P99: <3s

**Optimization Strategies:**
- Use `gpt-4o-mini` for faster responses
- Limit conversation history to last 5 messages
- Stream responses for perceived speed (future enhancement)

## Monitoring

**Key Metrics:**
- Response generation latency
- LLM API failures
- Crisis override rate
- Token usage

**Logs:**
- `conversation_agent_initialized` - Agent startup
- `assessing_context` - Context assessment
- `crisis_override_triggered` - Crisis protocol activated
- `response_generated` - Normal response generated
- `response_generation_failed` - LLM failure

## Future Enhancements

1. **Streaming Responses**: Stream tokens for faster perceived latency
2. **Multi-Language Support**: Spanish, Mandarin, etc.
3. **Conversation Memory**: Long-term memory across sessions
4. **Sentiment Analysis**: Track emotional trajectory over time
5. **Personalization**: Adapt tone based on student preferences

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
