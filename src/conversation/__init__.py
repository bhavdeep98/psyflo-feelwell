"""
Conversation module for empathetic AI responses.

Tenet #4: Fail loud, fail early - No mock implementations
Tenet #11: Graceful degradation - Circuit breaker for external service failures
"""

from src.conversation.conversation_agent import ConversationAgent, ConversationContext

__all__ = ["ConversationAgent", "ConversationContext"]
