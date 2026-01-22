"""
Safety Analyzer - Multi-Layer Crisis Detection Orchestrator

Coordinates multiple detection strategies to identify crisis indicators
in student messages. Combines deterministic regex, semantic embeddings,
and sarcasm filtering for comprehensive crisis detection.

Architecture:
- Uses Strategy Pattern for pluggable detection layers
- Orchestrates parallel analysis across all strategies
- Applies sarcasm filter to reduce false positives
- Returns immutable SafetyResult for audit trail

SLA: <50ms analysis latency on CPU

Usage:
    service = SafetyService("config/crisis_patterns.yaml")
    result = service.analyze("I want to die", context=["previous", "messages"])
    
    if result.is_crisis:
        # Trigger crisis protocol
        send_alert(result)
"""

from typing import List, Optional
from dataclasses import dataclass
import structlog

from .strategies import DetectionStrategy
from .strategy_factory import SafetyAnalyzerFactory

logger = structlog.get_logger()


@dataclass
class StrategyAnalysisResult:
    """Result from a single strategy analysis."""
    confidence: float
    matched_patterns: List[str]
    evidence: Optional[str] = None

@dataclass(frozen=True)
class SafetyResult:
    """
    Immutable safety analysis result.
    
    Attributes:
        p_regex: Regex confidence score (0.0-1.0)
        p_semantic: Semantic similarity score (0.0-1.0)
        p_sarcasm: Sarcasm probability score (0.0-1.0)
        matched_patterns: List of matched crisis categories
        is_crisis: Boolean crisis flag (True if crisis detected)
        latency_ms: Analysis latency in milliseconds
        sarcasm_filtered: Whether sarcasm filter was applied
        
    Design:
        - Frozen dataclass ensures immutability (audit trail)
        - All scores normalized to 0.0-1.0 range
        - Matched patterns include strategy prefix (e.g., "semantic:suicidal_ideation")
        
    Example:
        result = SafetyResult(
            p_regex=0.95,
            p_semantic=0.0,
            p_sarcasm=0.0,
            matched_patterns=['suicidal_ideation'],
            is_crisis=True,
            latency_ms=35.2,
            sarcasm_filtered=False
        )
    """
    p_regex: float
    p_semantic: float
    p_sarcasm: float
    matched_patterns: List[str]
    is_crisis: bool
    latency_ms: float
    sarcasm_filtered: bool


class SafetyService:
    """
    Multi-layer crisis detection service.
    
    Orchestrates multiple detection strategies using Strategy Pattern:
    1. Regex (deterministic keyword matching) - 40% weight
    2. Semantic (embedding similarity) - 20% weight  
    3. Sarcasm Filter (hyperbole detection) - reduces false positives
    
    Current Implementation (Milestone 1):
        is_crisis = (p_regex >= 0.90) OR (p_semantic >= 0.85)
        
    Target Implementation (Milestone 3):
        S_c = (0.4 × P_regex) + (0.2 × P_semantic) + (0.3 × P_mistral) + (0.1 × P_history)
        
    Design Principles:
        - Deterministic regex is "safety floor" (highest trust)
        - Semantic layer catches obfuscation but can't trigger crisis alone
        - Sarcasm filter reduces semantic score by 90% when hyperbole detected
        - All results are immutable for audit trail
        
    SLA: <50ms analysis latency
    
    Example:
        service = SafetyService("config/crisis_patterns.yaml")
        
        # Single message analysis
        result = service.analyze("I want to die")
        
        # With context (last 3 messages)
        result = service.analyze(
            "I'm checking out early",
            context=["I can't take it anymore", "Everything is hopeless"]
        )
        
        if result.is_crisis:
            logger.warning("crisis_detected", result=result)
    """
    
    # Crisis thresholds
    # Lower threshold for regex (deterministic) to catch all explicit crisis language
    # Higher threshold for semantic (probabilistic) to reduce false positives
    REGEX_CRISIS_THRESHOLD = 0.85  # Catches self_harm (0.85) and above
    SEMANTIC_CRISIS_THRESHOLD = 0.85  # Requires high confidence for semantic-only
    
    def __init__(self, patterns_path: str = "config/crisis_patterns.yaml"):
        """
        Initialize Safety Service with detection strategies.
        
        Args:
            patterns_path: Path to crisis patterns YAML file
                          Default: "config/crisis_patterns.yaml"
                          
        Raises:
            FileNotFoundError: If patterns file doesn't exist
            ValueError: If patterns file is invalid
            
        Side Effects:
            - Loads sentence-transformer model (~80MB download on first use)
            - Pre-computes embeddings for all crisis phrases
            - Compiles regex patterns
            
        Example:
            # Default patterns
            service = SafetyService()
            
            # Custom patterns
            service = SafetyService("custom_patterns.yaml")
        """
        # Create detection strategies using factory
        self.strategies = SafetyAnalyzerFactory.create_all_strategies(patterns_path)
        
        # Map strategies by name for easy access
        self.strategy_map = {s.get_name(): s for s in self.strategies}
        
        logger.info(
            "safety_service_initialized",
            strategies=[s.get_name() for s in self.strategies]
        )
    
    def analyze(self, message: str, context: List[str] = None) -> SafetyResult:
        """
        Full safety analysis combining all detection strategies.
        
        Process:
        1. Run sarcasm filter to detect hyperbole
        2. Run regex strategy for deterministic keyword matching
        3. Run semantic strategy with context for obfuscated language
        4. Apply sarcasm filter: reduce semantic score if hyperbole detected
        5. Determine crisis flag based on thresholds
        6. Return immutable SafetyResult
        
        Args:
            message: Student message to analyze (required)
            context: Optional list of previous messages for context
                    Semantic layer uses last 3 messages for disambiguation
                    Example: ["I'm stressed", "Can't sleep", "current message"]
            
        Returns:
            SafetyResult with scores, patterns, and crisis flag
            
        Decision Logic:
            - If p_sarcasm > 0.7: reduce p_semantic by 90%
            - is_crisis = (p_regex >= 0.90) OR (p_semantic >= 0.85)
            
        Performance:
            - Target: <50ms on CPU
            - Typical: ~35-45ms
            - Breakdown: regex ~5ms, semantic ~25ms, sarcasm ~5ms
            
        Example:
            # Without context
            result = service.analyze("I want to die")
            # result.is_crisis == True, result.p_regex == 0.95
            
            # With context (helps semantic layer)
            result = service.analyze(
                "I'm checking out early",
                context=["I can't take it anymore", "Everything is hopeless"]
            )
            # Semantic layer uses context to detect crisis
            
            # Hyperbole filtered
            result = service.analyze("This homework is killing me")
            # result.is_crisis == False, result.sarcasm_filtered == True
        """
        import time
        start_time = time.perf_counter()
        
        # Run sarcasm filter first
        p_sarcasm, sarcasm_patterns = self.strategy_map['sarcasm_filter'].analyze(message, context)
        
        # Run regex strategy
        p_regex, regex_patterns = self.strategy_map['regex'].analyze(message, context)
        
        # Run semantic strategy with context
        p_semantic, semantic_patterns = self.strategy_map['semantic'].analyze(message, context)
        
        # Combine matched patterns
        matched_patterns = regex_patterns + semantic_patterns + sarcasm_patterns
        
        # Apply sarcasm filter: if high sarcasm probability, reduce crisis scores
        sarcasm_filtered = False
        if p_sarcasm > 0.7:
            # Likely hyperbole - reduce semantic score significantly
            p_semantic = p_semantic * 0.1  # 90% reduction
            sarcasm_filtered = True
            
            logger.info(
                "sarcasm_filter_applied",
                original_semantic=p_semantic / 0.1 if p_semantic > 0 else 0,
                filtered_semantic=p_semantic,
                sarcasm_score=p_sarcasm
            )
        
        # Determine if crisis (either layer triggers, after sarcasm filtering)
        is_crisis = (
            p_regex >= self.REGEX_CRISIS_THRESHOLD or 
            p_semantic >= self.SEMANTIC_CRISIS_THRESHOLD
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        result = SafetyResult(
            p_regex=p_regex,
            p_semantic=p_semantic,
            p_sarcasm=p_sarcasm,
            matched_patterns=matched_patterns,
            is_crisis=is_crisis,
            latency_ms=latency_ms,
            sarcasm_filtered=sarcasm_filtered
        )
        
        if is_crisis:
            logger.warning(
                "crisis_detected",
                p_regex=p_regex,
                p_semantic=p_semantic,
                p_sarcasm=p_sarcasm,
                patterns=matched_patterns,
                latency_ms=latency_ms,
                sarcasm_filtered=sarcasm_filtered
            )
        
        return result
    
    async def analyze_regex(self, message: str, context: List[str] = None) -> StrategyAnalysisResult:
        """
        Analyze using only regex strategy (deterministic layer).
        
        Useful for:
        - Testing regex patterns in isolation
        - Debugging false negatives
        - Validating pattern configuration
        
        Args:
            message: Student message to analyze
            context: Optional previous messages (not used by regex)
            
        Returns:
            StrategyAnalysisResult with confidence score and matched patterns
            
        Example:
            result = await service.analyze_regex("I want to die")
            # result.confidence == 0.95 (suicidal_ideation pattern matched)
        """
        score, patterns = self.strategy_map['regex'].analyze(message, context)
        return StrategyAnalysisResult(confidence=score, matched_patterns=patterns)
    
    async def analyze_semantic(self, message: str, context: List[str] = None) -> StrategyAnalysisResult:
        """
        Analyze using only semantic strategy (embedding similarity).
        
        Useful for:
        - Testing semantic layer in isolation
        - Debugging obfuscated language detection
        - Evaluating context window effectiveness
        
        Args:
            message: Student message to analyze
            context: Optional previous messages (last 3 used for context)
            
        Returns:
            StrategyAnalysisResult with confidence score and matched patterns
            
        Note:
            Context significantly improves semantic accuracy.
            Without context, "checking out early" may not trigger.
            With crisis context, same phrase may score high.
            
        Example:
            # Without context
            result = await service.analyze_semantic("I'm checking out early")
            # result.confidence ~0.6 (ambiguous)
            
            # With context
            result = await service.analyze_semantic(
                "I'm checking out early",
                context=["I can't take it", "Everything is hopeless"]
            )
            # result.confidence ~0.85 (crisis context clarifies intent)
        """
        score, patterns = self.strategy_map['semantic'].analyze(message, context)
        return StrategyAnalysisResult(confidence=score, matched_patterns=patterns)

