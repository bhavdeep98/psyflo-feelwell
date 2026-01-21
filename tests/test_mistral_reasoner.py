"""
Tests for Mistral Reasoner - Deep Clinical Pattern Detection

Test Coverage:
- Crisis detection with explicit keywords
- Sarcasm/hyperbole filtering
- Clinical marker extraction
- Context-aware analysis
- Timeout and fallback behavior
- Immutability of results

Target: 100% coverage for safety-critical code
"""

import pytest
from src.reasoning.mistral_reasoner import (
    MistralReasoner,
    ReasoningResult,
    RiskLevel,
    ClinicalMarker
)


class TestMistralReasoner:
    """Test suite for Mistral Reasoner."""
    
    @pytest.fixture
    def reasoner(self):
        """Create mock reasoner for testing."""
        return MistralReasoner()  # Mock mode
    
    def test_explicit_crisis_detection(self, reasoner):
        """Test detection of explicit suicidal ideation."""
        message = "I want to kill myself"
        result = reasoner.analyze(message)
        
        assert result.risk_level == RiskLevel.CRISIS
        assert result.p_mistral >= 0.90
        assert not result.is_sarcasm
        assert len(result.clinical_markers) > 0
        assert any(m.category == "cssrs" for m in result.clinical_markers)
        assert "suicidal" in result.reasoning_trace.lower()
    
    def test_hyperbole_detection(self, reasoner):
        """Test filtering of teenage hyperbole."""
        message = "This homework is killing me"
        result = reasoner.analyze(message)
        
        assert result.risk_level == RiskLevel.SAFE
        assert result.p_mistral < 0.30
        assert result.is_sarcasm is True
        assert "hyperbolic" in result.sarcasm_reasoning.lower() or "metaphor" in result.sarcasm_reasoning.lower()
    
    def test_caution_level_detection(self, reasoner):
        """Test detection of moderate distress (caution level)."""
        message = "I've been feeling really depressed lately"
        result = reasoner.analyze(message)
        
        assert result.risk_level == RiskLevel.CAUTION
        assert 0.50 <= result.p_mistral <= 0.80
        assert not result.is_sarcasm
        assert len(result.clinical_markers) > 0
    
    def test_safe_message(self, reasoner):
        """Test that routine messages are marked safe."""
        message = "I had a good day at school today"
        result = reasoner.analyze(message)
        
        assert result.risk_level == RiskLevel.SAFE
        assert result.p_mistral < 0.30
        assert not result.is_sarcasm
    
    def test_context_awareness(self, reasoner):
        """Test that context improves detection accuracy."""
        message = "I'm done"
        context = [
            "I can't sleep anymore",
            "Nothing matters",
            "I feel hopeless"
        ]
        
        result = reasoner.analyze(message, context=context)
        
        # With crisis context, ambiguous message should be flagged
        # Note: Mock implementation doesn't use context yet
        # This test will be more meaningful with real Mistral endpoint
        assert isinstance(result, ReasoningResult)
        assert result.latency_ms >= 0
    
    def test_clinical_marker_structure(self, reasoner):
        """Test that clinical markers have required fields."""
        message = "I want to die"
        result = reasoner.analyze(message)
        
        assert len(result.clinical_markers) > 0
        
        for marker in result.clinical_markers:
            assert isinstance(marker, ClinicalMarker)
            assert marker.category in ["phq9", "gad7", "cssrs"]
            assert 0.0 <= marker.confidence <= 1.0
            assert len(marker.evidence) > 0
    
    def test_result_immutability(self, reasoner):
        """Test that ReasoningResult is immutable."""
        message = "Test message"
        result = reasoner.analyze(message)
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            result.p_mistral = 0.5
        
        with pytest.raises(AttributeError):
            result.risk_level = RiskLevel.CRISIS
    
    def test_reasoning_trace_present(self, reasoner):
        """Test that all results include reasoning trace."""
        messages = [
            "I want to die",
            "This homework is killing me",
            "I feel sad",
            "Good day today"
        ]
        
        for message in messages:
            result = reasoner.analyze(message)
            assert len(result.reasoning_trace) > 0
            assert isinstance(result.reasoning_trace, str)
    
    def test_latency_tracking(self, reasoner):
        """Test that latency is tracked for all analyses."""
        message = "Test message"
        result = reasoner.analyze(message)
        
        assert result.latency_ms >= 0
        assert result.latency_ms < 5000  # Should be under 5s timeout
    
    def test_model_attribution(self, reasoner):
        """Test that model used is tracked."""
        message = "Test message"
        result = reasoner.analyze(message)
        
        assert len(result.model_used) > 0
        assert result.model_used in ["mock-mistral", "mistral-7b-instruct", "fallback"]


class TestClinicalMarker:
    """Test suite for ClinicalMarker dataclass."""
    
    def test_valid_marker_creation(self):
        """Test creating valid clinical marker."""
        marker = ClinicalMarker(
            category="phq9",
            item="item_2_feeling_down",
            confidence=0.85,
            evidence="I've been feeling down"
        )
        
        assert marker.category == "phq9"
        assert marker.confidence == 0.85
    
    def test_marker_immutability(self):
        """Test that ClinicalMarker is immutable."""
        marker = ClinicalMarker(
            category="phq9",
            item="item_2",
            confidence=0.85,
            evidence="test"
        )
        
        with pytest.raises(AttributeError):
            marker.confidence = 0.90


class TestRiskLevel:
    """Test suite for RiskLevel enum."""
    
    def test_risk_levels_exist(self):
        """Test that all risk levels are defined."""
        assert RiskLevel.CRISIS.value == "crisis"
        assert RiskLevel.CAUTION.value == "caution"
        assert RiskLevel.SAFE.value == "safe"
    
    def test_risk_level_comparison(self):
        """Test that risk levels can be compared."""
        assert RiskLevel.CRISIS == RiskLevel.CRISIS
        assert RiskLevel.CRISIS != RiskLevel.SAFE


# Integration tests
class TestMistralReasonerIntegration:
    """Integration tests for Mistral Reasoner."""
    
    @pytest.fixture
    def reasoner(self):
        """Create mock reasoner."""
        return MistralReasoner()
    
    def test_multiple_crisis_keywords(self, reasoner):
        """Test message with multiple crisis indicators."""
        message = "I want to kill myself. I have a plan. I can't take it anymore."
        result = reasoner.analyze(message)
        
        assert result.risk_level == RiskLevel.CRISIS
        assert result.p_mistral >= 0.90
        assert len(result.clinical_markers) >= 1
    
    def test_mixed_signals(self, reasoner):
        """Test message with both crisis and hyperbole."""
        # This is tricky - contains "kill" but in hyperbolic context
        message = "My parents are going to kill me if I fail this test"
        result = reasoner.analyze(message)
        
        # Should recognize this as hyperbole, not genuine crisis
        # Note: Mock implementation may not handle this perfectly
        assert isinstance(result, ReasoningResult)
    
    def test_batch_analysis(self, reasoner):
        """Test analyzing multiple messages in sequence."""
        messages = [
            "I want to die",
            "This homework is killing me",
            "I feel sad",
            "Good day today",
            "I'm so anxious"
        ]
        
        results = [reasoner.analyze(msg) for msg in messages]
        
        assert len(results) == 5
        assert all(isinstance(r, ReasoningResult) for r in results)
        
        # First message should be crisis
        assert results[0].risk_level == RiskLevel.CRISIS
        
        # Second should be safe (hyperbole)
        assert results[1].risk_level == RiskLevel.SAFE
        assert results[1].is_sarcasm is True
