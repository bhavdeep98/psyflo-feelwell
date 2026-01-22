"""
Tests for Conversation Agent

Tenet #4: Fail loud, fail early - Test initialization failures
Tenet #10: Observable systems - Test logging and traceability
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.conversation import ConversationAgent, ConversationContext


class TestConversationAgentInitialization:
    """Test agent initialization and failure modes."""
    
    def test_initialization_fails_without_api_key(self):
        """
        Test that agent fails loud when API key is missing.
        
        Tenet #4: Fail loud, fail early
        """
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY not found"):
                ConversationAgent()
    
    def test_initialization_succeeds_with_api_key(self):
        """Test successful initialization with API key."""
        with patch('src.conversation.conversation_agent.ChatOpenAI') as mock_llm:
            mock_llm.return_value = Mock()
            
            agent = ConversationAgent(api_key="test-key")
            
            assert agent.model_name == "gpt-4o-mini"
            assert agent.temperature == 0.7
            mock_llm.assert_called_once()
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom model and temperature."""
        with patch('src.conversation.conversation_agent.ChatOpenAI') as mock_llm:
            mock_llm.return_value = Mock()
            
            agent = ConversationAgent(
                model_name="gpt-4",
                temperature=0.5,
                api_key="test-key"
            )
            
            assert agent.model_name == "gpt-4"
            assert agent.temperature == 0.5


class TestConversationGeneration:
    """Test response generation."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create agent with mocked LLM."""
        with patch('src.conversation.conversation_agent.ChatOpenAI') as mock_llm:
            mock_llm.return_value = Mock()
            agent = ConversationAgent(api_key="test-key")
            return agent
    
    @pytest.fixture
    def safe_context(self):
        """Create safe conversation context."""
        return ConversationContext(
            session_id="test-session",
            risk_level="SAFE",
            risk_score=0.1,
            matched_patterns=[],
            conversation_history=[]
        )
    
    @pytest.fixture
    def caution_context(self):
        """Create caution conversation context."""
        return ConversationContext(
            session_id="test-session",
            risk_level="CAUTION",
            risk_score=0.65,
            matched_patterns=["sadness"],
            conversation_history=[
                {"role": "student", "content": "I've been feeling down lately"},
                {"role": "assistant", "content": "I hear that you're feeling down. Want to talk about it?"}
            ]
        )
    
    @pytest.fixture
    def crisis_context(self):
        """Create crisis conversation context."""
        return ConversationContext(
            session_id="test-session",
            risk_level="CRISIS",
            risk_score=0.95,
            matched_patterns=["suicidal_ideation"],
            conversation_history=[]
        )
    
    @pytest.mark.asyncio
    async def test_crisis_override_response(self, mock_agent, crisis_context):
        """
        Test that crisis messages trigger hard-coded response.
        
        Tenet #1: Hard-coded crisis protocols that can't be overridden
        """
        response = await mock_agent.generate_response(
            message="I want to die",
            context=crisis_context
        )
        
        # Crisis response should contain specific resources
        assert "988" in response
        assert "741741" in response
        assert "counselor" in response.lower()
        assert "not alone" in response.lower()
    
    @pytest.mark.asyncio
    async def test_normal_response_uses_llm(self, mock_agent, safe_context):
        """Test that non-crisis messages use LLM."""
        # Mock the graph execution
        mock_response = "I'm here to listen. How has your day been?"
        
        with patch.object(mock_agent.graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            from langchain_core.messages import AIMessage
            mock_invoke.return_value = {
                "messages": [AIMessage(content=mock_response)],
                "context": safe_context,
                "next_action": "normal"
            }
            
            response = await mock_agent.generate_response(
                message="Hi, how are you?",
                context=safe_context
            )
            
            assert response == mock_response
            mock_invoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_response_generation_failure_raises(self, mock_agent, safe_context):
        """
        Test that LLM failures raise exceptions.
        
        Tenet #4: Fail loud, fail early
        """
        with patch.object(mock_agent.graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("LLM service unavailable")
            
            with pytest.raises(RuntimeError, match="Conversation graph failed"):
                await mock_agent.generate_response(
                    message="Hi",
                    context=safe_context
                )
    
    @pytest.mark.asyncio
    async def test_caution_context_adjusts_prompt(self, mock_agent, caution_context):
        """Test that CAUTION risk level adjusts system prompt."""
        # This tests that the system prompt includes extra guidance for caution cases
        system_prompt = mock_agent._build_system_prompt(caution_context)
        
        assert "IMPORTANT" in system_prompt
        assert "concerning signs" in system_prompt.lower()
        assert "counselor" in system_prompt.lower()


class TestConversationContext:
    """Test conversation context handling."""
    
    def test_context_is_immutable(self):
        """
        Test that ConversationContext is immutable.
        
        Tenet #7: Immutability by default
        """
        context = ConversationContext(
            session_id="test",
            risk_level="SAFE",
            risk_score=0.1,
            matched_patterns=[],
            conversation_history=[]
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            context.risk_level = "CRISIS"
    
    def test_context_includes_conversation_history(self):
        """Test that context preserves conversation history."""
        history = [
            {"role": "student", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        context = ConversationContext(
            session_id="test",
            risk_level="SAFE",
            risk_score=0.1,
            matched_patterns=[],
            conversation_history=history
        )
        
        assert len(context.conversation_history) == 2
        assert context.conversation_history[0]["role"] == "student"


class TestSystemPrompts:
    """Test system prompt generation."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create agent with mocked LLM."""
        with patch('src.conversation.conversation_agent.ChatOpenAI') as mock_llm:
            mock_llm.return_value = Mock()
            agent = ConversationAgent(api_key="test-key")
            return agent
    
    def test_base_prompt_includes_guidelines(self, mock_agent):
        """Test that base prompt includes key guidelines."""
        context = ConversationContext(
            session_id="test",
            risk_level="SAFE",
            risk_score=0.1,
            matched_patterns=[],
            conversation_history=[]
        )
        
        prompt = mock_agent._build_system_prompt(context)
        
        # Check for key elements
        assert "mental health" in prompt.lower()
        assert "empathetically" in prompt.lower() or "empathetic" in prompt.lower()
        assert "never diagnose" in prompt.lower()
        assert "counselor" in prompt.lower()
    
    def test_caution_prompt_adds_extra_guidance(self, mock_agent):
        """Test that CAUTION risk adds extra guidance."""
        safe_context = ConversationContext(
            session_id="test",
            risk_level="SAFE",
            risk_score=0.1,
            matched_patterns=[],
            conversation_history=[]
        )
        
        caution_context = ConversationContext(
            session_id="test",
            risk_level="CAUTION",
            risk_score=0.65,
            matched_patterns=["sadness"],
            conversation_history=[]
        )
        
        safe_prompt = mock_agent._build_system_prompt(safe_context)
        caution_prompt = mock_agent._build_system_prompt(caution_context)
        
        # Caution prompt should be longer and include extra guidance
        assert len(caution_prompt) > len(safe_prompt)
        assert "IMPORTANT" in caution_prompt
        assert "concerning signs" in caution_prompt.lower()


class TestLoggingAndObservability:
    """Test logging and observability."""
    
    @pytest.mark.asyncio
    async def test_initialization_logged(self):
        """
        Test that initialization is logged.
        
        Tenet #10: Observable systems
        """
        with patch('src.conversation.conversation_agent.ChatOpenAI') as mock_llm:
            mock_llm.return_value = Mock()
            
            with patch('src.conversation.conversation_agent.logger') as mock_logger:
                agent = ConversationAgent(api_key="test-key")
                
                # Check that initialization was logged
                mock_logger.info.assert_called_with(
                    "conversation_agent_initialized",
                    model="gpt-4o-mini",
                    temperature=0.7
                )
    
    @pytest.mark.asyncio
    async def test_crisis_override_logged(self):
        """Test that crisis overrides are logged."""
        with patch('src.conversation.conversation_agent.ChatOpenAI') as mock_llm:
            mock_llm.return_value = Mock()
            agent = ConversationAgent(api_key="test-key")
            
            crisis_context = ConversationContext(
                session_id="test-session",
                risk_level="CRISIS",
                risk_score=0.95,
                matched_patterns=["suicidal_ideation"],
                conversation_history=[]
            )
            
            with patch('src.conversation.conversation_agent.logger') as mock_logger:
                response = await agent.generate_response(
                    message="I want to die",
                    context=crisis_context
                )
                
                # Check that crisis override was logged
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args
                assert call_args[0][0] == "crisis_override_triggered"
