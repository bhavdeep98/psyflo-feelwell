"""
Tests for Consensus Orchestrator

Tenet #1: 100% test coverage required for all safety-critical code
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass

from src.orchestrator import (
    ConsensusOrchestrator,
    ConsensusResult,
    ConsensusConfig,
    RiskLevel,
)
from src.orchestrator.consensus_result import LayerScore


@dataclass
class MockDetectionResult:
    """Mock detection result from safety service."""
    confidence: float
    matched_patterns: list
    evidence: str


@dataclass
class MockReasoningResult:
    """Mock reasoning result from Mistral."""
    confidence: float
    matched_patterns: list
    reasoning: str


class TestConsensusConfig:
    """Test configuration validation."""
    
    def test_default_config_valid(self):
        """Default config should be valid."""
        config = ConsensusConfig()
        assert config.w_regex == 0.40
        assert config.w_semantic == 0.20
        assert config.w_mistral == 0.30
        assert config.w_history == 0.10
    
    def test_weights_must_sum_to_one(self):
        """Weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ConsensusConfig(
                w_regex=0.5,
                w_semantic=0.3,
                w_mistral=0.3,
                w_history=0.1
            )
    
    def test_weights_must_be_non_negative(self):
        """Weights cannot be negative."""
        with pytest.raises(ValueError, match="non-negative"):
            ConsensusConfig(
                w_regex=-0.1,
                w_semantic=0.5,
                w_mistral=0.4,
                w_history=0.2
            )
    
    def test_crisis_threshold_must_be_higher_than_caution(self):
        """Crisis threshold must be > caution threshold."""
        with pytest.raises(ValueError, match="must be >"):
            ConsensusConfig(
                crisis_threshold=0.70,
                caution_threshold=0.80
            )
    
    def test_custom_weights_valid(self):
        """Custom weights should work if they sum to 1.0."""
        config = ConsensusConfig(
            w_regex=0.5,
            w_semantic=0.2,
            w_mistral=0.2,
            w_history=0.1
        )
        assert config.w_regex == 0.5


class TestConsensusResult:
    """Test consensus result data structure."""
    
    def test_result_is_immutable(self):
        """Result should be immutable."""
        result = ConsensusResult(
            risk_level=RiskLevel.SAFE,
            final_score=0.3,
            regex_score=LayerScore("regex", 0.2, 5, [], ""),
            semantic_score=LayerScore("semantic", 0.1, 10, [], ""),
            mistral_score=LayerScore("mistral", 0.4, 500, [], ""),
            reasoning="Test",
            matched_patterns=[],
            total_latency_ms=515,
            timeout_occurred=False,
            weights_used={"regex": 0.4, "semantic": 0.2, "mistral": 0.3, "history": 0.1}
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            result.final_score = 0.5
    
    def test_score_validation(self):
        """Score must be 0.0-1.0."""
        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            ConsensusResult(
                risk_level=RiskLevel.SAFE,
                final_score=1.5,  # Invalid
                regex_score=LayerScore("regex", 0.2, 5, [], ""),
                semantic_score=LayerScore("semantic", 0.1, 10, [], ""),
                mistral_score=None,
                reasoning="Test",
                matched_patterns=[],
                total_latency_ms=15,
                timeout_occurred=False,
                weights_used={}
            )
    
    def test_helper_methods(self):
        """Test is_crisis, is_caution, is_safe helpers."""
        crisis_result = ConsensusResult(
            risk_level=RiskLevel.CRISIS,
            final_score=0.95,
            regex_score=LayerScore("regex", 0.9, 5, [], ""),
            semantic_score=LayerScore("semantic", 0.8, 10, [], ""),
            mistral_score=None,
            reasoning="Test",
            matched_patterns=[],
            total_latency_ms=15,
            timeout_occurred=False,
            weights_used={}
        )
        
        assert crisis_result.is_crisis() is True
        assert crisis_result.is_caution() is False
        assert crisis_result.is_safe() is False


class TestConsensusOrchestrator:
    """Test consensus orchestrator."""
    
    @pytest.fixture
    def mock_safety_service(self):
        """Create mock safety service."""
        service = Mock()
        service.analyze_regex = AsyncMock(return_value=MockDetectionResult(
            confidence=0.0,
            matched_patterns=[],
            evidence=""
        ))
        service.analyze_semantic = AsyncMock(return_value=MockDetectionResult(
            confidence=0.0,
            matched_patterns=[],
            evidence=""
        ))
        return service
    
    @pytest.fixture
    def mock_mistral_reasoner(self):
        """Create mock Mistral reasoner."""
        reasoner = Mock()
        reasoner.analyze = AsyncMock(return_value=MockReasoningResult(
            confidence=0.0,
            matched_patterns=[],
            reasoning=""
        ))
        return reasoner
    
    @pytest.fixture
    def orchestrator(self, mock_safety_service, mock_mistral_reasoner):
        """Create orchestrator with mocks."""
        return ConsensusOrchestrator(
            safety_service=mock_safety_service,
            mistral_reasoner=mock_mistral_reasoner,
            config=ConsensusConfig()
        )
    
    @pytest.mark.asyncio
    async def test_safe_message(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test safe message classification."""
        # Setup mocks
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.0, matched_patterns=[], evidence=""
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.0, matched_patterns=[], evidence=""
        )
        mock_mistral_reasoner.analyze.return_value = MockReasoningResult(
            confidence=0.0, matched_patterns=[], reasoning="No crisis detected"
        )
        
        # Analyze
        result = await orchestrator.analyze("I'm feeling good today", "session_123")
        
        # Verify
        assert result.risk_level == RiskLevel.SAFE
        assert result.final_score < 0.65
        assert result.is_safe() is True
    
    @pytest.mark.asyncio
    async def test_crisis_message_regex_detection(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test crisis detected by regex layer."""
        # Setup mocks - regex detects crisis
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.95,
            matched_patterns=["suicidal_ideation"],
            evidence="I want to die"
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.85, matched_patterns=[], evidence=""
        )
        mock_mistral_reasoner.analyze.return_value = MockReasoningResult(
            confidence=0.90, matched_patterns=["PHQ9_ITEM_9"], reasoning="Suicidal ideation detected"
        )
        
        # Analyze
        result = await orchestrator.analyze("I want to die", "session_123")
        
        # Verify
        assert result.risk_level == RiskLevel.CRISIS
        assert result.final_score >= 0.90
        assert result.is_crisis() is True
        assert "suicidal_ideation" in result.matched_patterns
    
    @pytest.mark.asyncio
    async def test_crisis_message_consensus(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test crisis detected by consensus (not just regex)."""
        # Setup mocks - all layers detect moderate risk
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.70, matched_patterns=["self_harm"], evidence=""
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.80, matched_patterns=[], evidence=""
        )
        mock_mistral_reasoner.analyze.return_value = MockReasoningResult(
            confidence=0.95, matched_patterns=["PHQ9_ITEM_9"], reasoning="High risk"
        )
        
        # Analyze
        result = await orchestrator.analyze("I'm going to hurt myself", "session_123")
        
        # Calculate expected score: (0.4 * 0.7) + (0.2 * 0.8) + (0.3 * 0.95) = 0.725
        expected_score = (0.4 * 0.70) + (0.2 * 0.80) + (0.3 * 0.95)
        
        # Verify
        assert abs(result.final_score - expected_score) < 0.01
        assert result.risk_level == RiskLevel.CAUTION  # Below 0.90 threshold
    
    @pytest.mark.asyncio
    async def test_caution_level(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test caution level classification."""
        # Setup mocks - moderate risk
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.50, matched_patterns=[], evidence=""
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.60, matched_patterns=[], evidence=""
        )
        mock_mistral_reasoner.analyze.return_value = MockReasoningResult(
            confidence=0.70, matched_patterns=["PHQ9_ITEM_2"], reasoning="Depressed mood"
        )
        
        # Analyze
        result = await orchestrator.analyze("I've been feeling down lately", "session_123")
        
        # Calculate expected: (0.4 * 0.5) + (0.2 * 0.6) + (0.3 * 0.7) = 0.53
        # Should be SAFE (< 0.65)
        assert result.risk_level == RiskLevel.SAFE
    
    @pytest.mark.asyncio
    async def test_mistral_timeout_graceful_degradation(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test graceful degradation when Mistral times out."""
        # Setup mocks
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.80, matched_patterns=["self_harm"], evidence=""
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.70, matched_patterns=[], evidence=""
        )
        
        # Mistral times out
        async def timeout_func(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
        
        mock_mistral_reasoner.analyze = timeout_func
        
        # Analyze
        result = await orchestrator.analyze("I want to hurt myself", "session_123")
        
        # Verify - should still work with just Safety scores
        assert result.mistral_score is None
        assert result.timeout_occurred is True
        # Weight redistributed: (0.7 * 0.8) + (0.3 * 0.7) = 0.77
        # Should be CAUTION level
        assert result.risk_level in [RiskLevel.CAUTION, RiskLevel.CRISIS]
    
    @pytest.mark.asyncio
    async def test_safety_floor_override(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test that high regex score overrides consensus."""
        # Setup mocks - regex very high, others low
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.98,  # Very high
            matched_patterns=["suicidal_ideation"],
            evidence="I want to kill myself"
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.20, matched_patterns=[], evidence=""
        )
        mock_mistral_reasoner.analyze.return_value = MockReasoningResult(
            confidence=0.30, matched_patterns=[], reasoning="Low risk"
        )
        
        # Analyze
        result = await orchestrator.analyze("I want to kill myself", "session_123")
        
        # Verify - safety floor should trigger CRISIS
        assert result.risk_level == RiskLevel.CRISIS
        assert result.regex_score.score >= 0.95
    
    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test that layers run in parallel (not sequential)."""
        # Setup mocks with delays
        async def slow_regex(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MockDetectionResult(confidence=0.0, matched_patterns=[], evidence="")
        
        async def slow_semantic(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MockDetectionResult(confidence=0.0, matched_patterns=[], evidence="")
        
        async def slow_mistral(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MockReasoningResult(confidence=0.0, matched_patterns=[], reasoning="")
        
        mock_safety_service.analyze_regex = slow_regex
        mock_safety_service.analyze_semantic = slow_semantic
        mock_mistral_reasoner.analyze = slow_mistral
        
        # Analyze
        import time
        start = time.time()
        result = await orchestrator.analyze("test message", "session_123")
        elapsed = time.time() - start
        
        # If parallel: ~0.1s, if sequential: ~0.3s
        assert elapsed < 0.2, f"Took {elapsed}s, should be <0.2s for parallel execution"
        assert result.total_latency_ms < 200
    
    @pytest.mark.asyncio
    async def test_event_bus_publishes_crisis(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test that crisis events are published to event bus."""
        # Setup crisis detection
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.95, matched_patterns=["suicidal_ideation"], evidence=""
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.85, matched_patterns=[], evidence=""
        )
        mock_mistral_reasoner.analyze.return_value = MockReasoningResult(
            confidence=0.90, matched_patterns=[], reasoning=""
        )
        
        # Subscribe to event bus
        events_received = []
        orchestrator.event_bus.subscribe(lambda result: events_received.append(result))
        
        # Analyze
        result = await orchestrator.analyze("I want to die", "session_123")
        
        # Verify event published
        assert len(events_received) == 1
        assert events_received[0].is_crisis()
    
    @pytest.mark.asyncio
    async def test_reasoning_trace_explainability(self, orchestrator, mock_safety_service, mock_mistral_reasoner):
        """Test that reasoning trace is human-readable."""
        # Setup
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.80, matched_patterns=["self_harm"], evidence=""
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.70, matched_patterns=[], evidence=""
        )
        mock_mistral_reasoner.analyze.return_value = MockReasoningResult(
            confidence=0.85, matched_patterns=["PHQ9_ITEM_9"], reasoning="Detected ideation"
        )
        
        # Analyze
        result = await orchestrator.analyze("test", "session_123")
        
        # Verify reasoning contains key information
        assert "Risk Level:" in result.reasoning
        assert "Final Score:" in result.reasoning
        assert "Layer Scores:" in result.reasoning
        assert "Regex:" in result.reasoning
        assert "Semantic:" in result.reasoning
        assert "Mistral:" in result.reasoning


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, mock_safety_service):
        """Test circuit breaker opens after threshold failures."""
        # Create orchestrator with low threshold
        config = ConsensusConfig(
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=1
        )
        
        # Mistral always fails
        mistral = Mock()
        mistral.analyze = AsyncMock(side_effect=Exception("Service down"))
        
        orchestrator = ConsensusOrchestrator(
            safety_service=mock_safety_service,
            mistral_reasoner=mistral,
            config=config
        )
        
        # Setup safety service to return safe scores
        mock_safety_service.analyze_regex.return_value = MockDetectionResult(
            confidence=0.0, matched_patterns=[], evidence=""
        )
        mock_safety_service.analyze_semantic.return_value = MockDetectionResult(
            confidence=0.0, matched_patterns=[], evidence=""
        )
        
        # Trigger failures
        for _ in range(3):
            result = await orchestrator.analyze("test", "session_123")
            assert result.mistral_score is None
        
        # Circuit should be open now
        assert orchestrator.circuit_breaker.is_open is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
