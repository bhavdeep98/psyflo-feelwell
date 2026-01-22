"""
Conversational AI Agent for Mental Health Support

Tenet #1: Safety First - Crisis detection runs in parallel
Tenet #3: Explicit Over Clever - Clear, traceable conversation flow
Tenet #4: Fail Loud, Fail Early - No silent fallbacks
Tenet #8: Engagement Before Intervention - Build trust through empathy
"""

from typing import TypedDict, Annotated, Sequence, Optional
from dataclasses import dataclass
import structlog
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# Load environment variables from .env file
load_dotenv()

logger = structlog.get_logger()


@dataclass(frozen=True)
class ConversationContext:
    """Immutable conversation context."""
    session_id: str
    risk_level: str
    risk_score: float
    matched_patterns: list[str]
    conversation_history: list[dict]


class AgentState(TypedDict):
    """State for the conversation agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: ConversationContext
    next_action: str


class ConversationAgent:
    """
    Mental health conversation agent using LangGraph.
    
    Design:
    - Uses LangGraph for explicit state management
    - Empathetic, active listening approach
    - Crisis-aware (adjusts tone based on risk level)
    - Traceable conversation flow
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize conversation agent.
        
        Args:
            model_name: OpenAI model to use (default: from .env or gpt-4o-mini)
            temperature: Response creativity 0.0-1.0 (default: from .env or 0.7)
            api_key: OpenAI API key (default: from .env OPENAI_API_KEY)
            
        Raises:
            RuntimeError: If LLM cannot be initialized (Tenet #4: Fail loud, fail early)
        """
        # Load from .env or use defaults
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        
        # Tenet #4: Fail loud, fail early -> Relaxed for prototype/demo
        self.is_mock_mode = False
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your-openai-api-key-here":
                logger.warning("openai_api_key_missing_using_mock_mode")
                self.is_mock_mode = True
        
        # Initialize LLM only if not in mock mode
        if not self.is_mock_mode:
            try:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=api_key
                )
                logger.info(
                    "conversation_agent_initialized",
                    model=self.model_name,
                    temperature=self.temperature
                )
            except Exception as e:
                logger.error("llm_initialization_failed", error=str(e), exc_info=True)
                # Fallback to mock mode if init fails
                self.is_mock_mode = True
        
        # Build conversation graph (works in both modes)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph conversation flow."""
        
        # Define graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("assess_context", self._assess_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("crisis_override", self._crisis_override)
        
        # Define edges
        workflow.set_entry_point("assess_context")
        
        workflow.add_conditional_edges(
            "assess_context",
            self._should_override_for_crisis,
            {
                "crisis": "crisis_override",
                "normal": "generate_response"
            }
        )
        
        workflow.add_edge("generate_response", END)
        workflow.add_edge("crisis_override", END)
        
        return workflow.compile()
    
    def _assess_context(self, state: AgentState) -> AgentState:
        """Assess conversation context and determine next action."""
        context = state["context"]
        
        logger.info(
            "assessing_context",
            risk_level=context.risk_level,
            risk_score=context.risk_score,
            history_length=len(context.conversation_history)
        )
        
        # Determine next action based on risk level
        if context.risk_level == "CRISIS":
            state["next_action"] = "crisis"
        else:
            state["next_action"] = "normal"
        
        return state
    
    def _should_override_for_crisis(self, state: AgentState) -> str:
        """Conditional edge: Check if crisis override needed."""
        return state["next_action"]
    
    def _crisis_override(self, state: AgentState) -> AgentState:
        """
        Crisis override response.
        
        Tenet #1: Hard-coded crisis protocols that can't be overridden
        """
        context = state["context"]
        
        crisis_message = (
            "I'm really concerned about what you've shared. "
            "Your safety is the most important thing right now. "
            "I've notified your school counselor, and they'll reach out soon.\n\n"
            "In the meantime, here are some resources:\n\n"
            "🆘 National Suicide Prevention Lifeline: 988\n"
            "💬 Crisis Text Line: Text HOME to 741741\n"
            "🌐 Online Chat: https://suicidepreventionlifeline.org/chat/\n\n"
            "You're not alone in this. Help is available 24/7."
        )
        
        state["messages"].append(AIMessage(content=crisis_message))
        
        logger.warning(
            "crisis_override_triggered",
            session_id=context.session_id,
            risk_score=context.risk_score,
            matched_patterns=context.matched_patterns
        )
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate empathetic response using LLM."""
        context = state["context"]
        
        # Build system prompt based on risk level
        system_prompt = self._build_system_prompt(context)
        
        # Create prompt with conversation history
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Generate response
        try:
            if self.is_mock_mode:
                response_content = "I hear you. (Mock response: OpenAI API key missing)"
            else:
                chain = prompt | self.llm
                response = chain.invoke({"messages": state["messages"]})
                response_content = response.content
            
            state["messages"].append(AIMessage(content=response_content))
            
            logger.info(
                "response_generated",
                session_id=context.session_id,
                risk_level=context.risk_level,
                response_length=len(response_content)
            )
            
        except Exception as e:
            logger.error(
                "response_generation_failed",
                session_id=context.session_id,
                error=str(e),
                exc_info=True
            )
            # Tenet #4: Fail loud, fail early
            raise RuntimeError(f"Failed to generate response: {e}") from e
        
        return state
    
    def _build_system_prompt(self, context: ConversationContext) -> str:
        """
        Build system prompt based on conversation context.
        
        Tenet #8: Engagement Before Intervention
        """
        base_prompt = """You are a supportive mental health AI assistant for high school students.

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

Tone: Warm, supportive, non-judgmental, age-appropriate"""
        
        # Adjust based on risk level
        if context.risk_level == "CAUTION":
            base_prompt += """

IMPORTANT: This student is showing some concerning signs. Be extra attentive:
- Gently explore what's troubling them
- Validate their struggles
- Subtly encourage them to reach out to their counselor
- Watch for escalation"""
        
        return base_prompt
    
    async def generate_response(
        self,
        message: str,
        context: ConversationContext
    ) -> str:
        """
        Generate response for student message.
        
        Args:
            message: Student's message
            context: Conversation context with risk assessment
            
        Returns:
            AI-generated response
            
        Raises:
            RuntimeError: If response generation fails
        """
        # Build conversation history
        messages = []
        
        # Add conversation history
        for msg in context.conversation_history[-5:]:  # Last 5 messages for context
            if msg["role"] == "student":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Create initial state
        initial_state: AgentState = {
            "messages": messages,
            "context": context,
            "next_action": ""
        }
        
        # Run graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract response
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
            else:
                raise RuntimeError("Expected AIMessage as final message")
                
        except Exception as e:
            logger.error(
                "graph_execution_failed",
                session_id=context.session_id,
                error=str(e),
                exc_info=True
            )
            # Tenet #4: Fail loud, fail early
            raise RuntimeError(f"Conversation graph failed: {e}") from e
