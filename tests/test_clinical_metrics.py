"""
Tests for Clinical Metrics - Seven-Dimension Assessment

Test Coverage:
- Metric score validation
- Clinical assessment structure
- Evaluation prompt generation
- JSON parsing and validation
- Immutability of results

Target: 100% coverage for safety-critical code
"""

import pytest
from src.reasoning.clinical_metrics import (
    ClinicalMetric,
    MetricScore,
    ClinicalAssessment,
    ClinicalMetrics
)


class TestMetricScore:
    """Test suite for MetricScore dataclass."""
    
    def test_valid_score_creation(self):
        """Test creating valid metric score."""
        score = MetricScore(
            metric=ClinicalMetric.EMPATHY_VALIDATION,
            score=9,
            reasoning="Excellent validation of feelings",
            evidence=["I hear that you're feeling overwhelmed"]
        )
        
        assert score.metric == ClinicalMetric.EMPATHY_VALIDATION
        assert score.score == 9
        assert len(score.evidence) == 1
    
    def test_score_validation_min(self):
        """Test that scores below 1 are rejected."""
        with pytest.raises(ValueError, match="Score must be 1-10"):
            MetricScore(
                metric=ClinicalMetric.EMPATHY_VALIDATION,
                score=0,
                reasoning="Test",
                evidence=[]
            )
    
    def test_score_validation_max(self):
        """Test that scores above 10 are rejected."""
        with pytest.raises(ValueError, match="Score must be 1-10"):
            MetricScore(
                metric=ClinicalMetric.EMPATHY_VALIDATION,
                score=11,
                reasoning="Test",
                evidence=[]
            )
    
    def test_score_immutability(self):
        """Test that MetricScore is immutable."""
        score = MetricScore(
            metric=ClinicalMetric.EMPATHY_VALIDATION,
            score=8,
            reasoning="Test",
            evidence=[]
        )
        
        with pytest.raises(AttributeError):
            score.score = 9


class TestClinicalAssessment:
    """Test suite for ClinicalAssessment dataclass."""
    
    @pytest.fixture
    def valid_scores(self):
        """Create valid scores for all seven metrics."""
        return {
            ClinicalMetric.ACTIVE_LISTENING: MetricScore(
                metric=ClinicalMetric.ACTIVE_LISTENING,
                score=8,
                reasoning="Good reflection",
                evidence=["I hear you"]
            ),
            ClinicalMetric.EMPATHY_VALIDATION: MetricScore(
                metric=ClinicalMetric.EMPATHY_VALIDATION,
                score=9,
                reasoning="Strong validation",
                evidence=["That makes sense"]
            ),
            ClinicalMetric.SAFETY_TRUSTWORTHINESS: MetricScore(
                metric=ClinicalMetric.SAFETY_TRUSTWORTHINESS,
                score=10,
                reasoning="Excellent safety",
                evidence=["Let's get you help"]
            ),
            ClinicalMetric.OPEN_MINDEDNESS: MetricScore(
                metric=ClinicalMetric.OPEN_MINDEDNESS,
                score=8,
                reasoning="Non-judgmental",
                evidence=["Tell me more"]
            ),
            ClinicalMetric.CLARITY_ENCOURAGEMENT: MetricScore(
                metric=ClinicalMetric.CLARITY_ENCOURAGEMENT,
                score=7,
                reasoning="Clear and hopeful",
                evidence=["You can do this"]
            ),
            ClinicalMetric.BOUNDARIES_ETHICAL: MetricScore(
                metric=ClinicalMetric.BOUNDARIES_ETHICAL,
                score=9,
                reasoning="Appropriate boundaries",
                evidence=["I'm here to listen"]
            ),
            ClinicalMetric.HOLISTIC_APPROACH: MetricScore(
                metric=ClinicalMetric.HOLISTIC_APPROACH,
                score=8,
                reasoning="Considers context",
                evidence=["How are things at home?"]
            )
        }
    
    def test_valid_assessment_creation(self, valid_scores):
        """Test creating valid clinical assessment."""
        assessment = ClinicalAssessment(
            scores=valid_scores,
            average_score=8.43,
            overall_reasoning="Strong clinical response"
        )
        
        assert len(assessment.scores) == 7
        assert 8.42 <= assessment.average_score <= 8.44  # Allow floating point tolerance
    
    def test_assessment_requires_all_metrics(self, valid_scores):
        """Test that assessment requires all 7 metrics."""
        # Remove one metric
        incomplete_scores = {k: v for k, v in list(valid_scores.items())[:6]}
        
        with pytest.raises(ValueError, match="Must have all 7 metrics"):
            ClinicalAssessment(
                scores=incomplete_scores,
                average_score=8.0,
                overall_reasoning="Test"
            )
    
    def test_assessment_validates_average(self, valid_scores):
        """Test that average score is validated."""
        with pytest.raises(ValueError, match="Average score mismatch"):
            ClinicalAssessment(
                scores=valid_scores,
                average_score=5.0,  # Wrong average
                overall_reasoning="Test"
            )
    
    def test_assessment_immutability(self, valid_scores):
        """Test that ClinicalAssessment is immutable."""
        assessment = ClinicalAssessment(
            scores=valid_scores,
            average_score=8.43,
            overall_reasoning="Test"
        )
        
        with pytest.raises(AttributeError):
            assessment.average_score = 9.0


class TestClinicalMetrics:
    """Test suite for ClinicalMetrics utility class."""
    
    def test_metric_descriptions_complete(self):
        """Test that all metrics have descriptions."""
        assert len(ClinicalMetrics.METRIC_DESCRIPTIONS) == 7
        
        for metric in ClinicalMetric:
            assert metric in ClinicalMetrics.METRIC_DESCRIPTIONS
            desc = ClinicalMetrics.METRIC_DESCRIPTIONS[metric]
            assert "name" in desc
            assert "description" in desc
            assert "good_example" in desc
            assert "poor_example" in desc
    
    def test_evaluation_prompt_generation(self):
        """Test generating evaluation prompt for GPT-4."""
        student_message = "I'm feeling really stressed"
        ai_response = "I hear that you're feeling stressed. Tell me more about what's going on."
        
        prompt = ClinicalMetrics.get_evaluation_prompt(student_message, ai_response)
        
        assert student_message in prompt
        assert ai_response in prompt
        assert "Active Listening" in prompt
        assert "Empathy & Validation" in prompt
        assert "JSON format" in prompt
        assert all(metric.value in prompt.lower() for metric in ClinicalMetric)
    
    def test_evaluation_prompt_with_context(self):
        """Test prompt generation with conversation context."""
        student_message = "I can't take it anymore"
        ai_response = "I'm concerned about what you shared. Let's get you help."
        context = ["I'm so stressed", "Can't sleep", "Everything is too much"]
        
        prompt = ClinicalMetrics.get_evaluation_prompt(
            student_message,
            ai_response,
            context=context
        )
        
        assert "Conversation Context" in prompt
        assert context[0] in prompt
        # Only last 3 messages should be included
        assert len([c for c in context if c in prompt]) <= 3
    
    def test_parse_valid_evaluation(self):
        """Test parsing valid GPT-4 evaluation response."""
        json_response = {
            "scores": {
                "active_listening": {
                    "score": 8,
                    "reasoning": "Good reflection",
                    "evidence": ["I hear you"]
                },
                "empathy_validation": {
                    "score": 9,
                    "reasoning": "Strong validation",
                    "evidence": ["That makes sense"]
                },
                "safety_trustworthiness": {
                    "score": 10,
                    "reasoning": "Excellent safety",
                    "evidence": ["Let's get help"]
                },
                "open_mindedness": {
                    "score": 8,
                    "reasoning": "Non-judgmental",
                    "evidence": ["Tell me more"]
                },
                "clarity_encouragement": {
                    "score": 7,
                    "reasoning": "Clear",
                    "evidence": ["You can do this"]
                },
                "boundaries_ethical": {
                    "score": 9,
                    "reasoning": "Appropriate",
                    "evidence": ["I'm here to listen"]
                },
                "holistic_approach": {
                    "score": 8,
                    "reasoning": "Considers context",
                    "evidence": ["How are things?"]
                }
            },
            "average_score": 8.43,
            "overall_reasoning": "Strong clinical response"
        }
        
        assessment = ClinicalMetrics.parse_evaluation(json_response)
        
        assert isinstance(assessment, ClinicalAssessment)
        assert len(assessment.scores) == 7
        assert assessment.average_score == 8.43
        assert assessment.overall_reasoning == "Strong clinical response"
    
    def test_parse_missing_metric(self):
        """Test that parsing fails if metric is missing."""
        json_response = {
            "scores": {
                "active_listening": {
                    "score": 8,
                    "reasoning": "Good",
                    "evidence": ["test"]
                }
                # Missing other 6 metrics
            },
            "average_score": 8.0,
            "overall_reasoning": "Test"
        }
        
        with pytest.raises(ValueError, match="Missing metric"):
            ClinicalMetrics.parse_evaluation(json_response)
    
    def test_parse_invalid_score(self):
        """Test that parsing validates score range."""
        json_response = {
            "scores": {
                "active_listening": {
                    "score": 15,  # Invalid score
                    "reasoning": "Good",
                    "evidence": ["test"]
                },
                "empathy_validation": {"score": 8, "reasoning": "Good", "evidence": ["test"]},
                "safety_trustworthiness": {"score": 8, "reasoning": "Good", "evidence": ["test"]},
                "open_mindedness": {"score": 8, "reasoning": "Good", "evidence": ["test"]},
                "clarity_encouragement": {"score": 8, "reasoning": "Good", "evidence": ["test"]},
                "boundaries_ethical": {"score": 8, "reasoning": "Good", "evidence": ["test"]},
                "holistic_approach": {"score": 8, "reasoning": "Good", "evidence": ["test"]}
            },
            "average_score": 8.0,
            "overall_reasoning": "Test"
        }
        
        with pytest.raises(ValueError, match="Score must be 1-10"):
            ClinicalMetrics.parse_evaluation(json_response)


class TestClinicalMetricEnum:
    """Test suite for ClinicalMetric enum."""
    
    def test_all_metrics_defined(self):
        """Test that all seven metrics are defined."""
        metrics = list(ClinicalMetric)
        assert len(metrics) == 7
        
        expected_metrics = [
            "active_listening",
            "empathy_validation",
            "safety_trustworthiness",
            "open_mindedness",
            "clarity_encouragement",
            "boundaries_ethical",
            "holistic_approach"
        ]
        
        for expected in expected_metrics:
            assert any(m.value == expected for m in metrics)
    
    def test_metric_values(self):
        """Test that metric values are correct."""
        assert ClinicalMetric.ACTIVE_LISTENING.value == "active_listening"
        assert ClinicalMetric.EMPATHY_VALIDATION.value == "empathy_validation"
        assert ClinicalMetric.SAFETY_TRUSTWORTHINESS.value == "safety_trustworthiness"
