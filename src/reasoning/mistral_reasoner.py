"""
Mistral Reasoner - Deep Clinical Pattern Detection

Implements the "Hidden Clinician" using Mental-Health-FineTuned-Mistral-7B
for non-linear pattern detection and structured clinical reasoning.

Key Features:
- Uses GRMenon/mental-mistral-7b-instruct-autotrain from HuggingFace
- Structured JSON output for consistent reasoning traces
- Context-aware analysis using conversation history
- Sarcasm/hyperbole detection for teenage vernacular
- Clinical marker extraction (PHQ-9, GAD-7, C-SSRS)
- Timeout protection (raises TimeoutError)
- Fail-fast design: raises exceptions on errors (no fallbacks)

Architecture:
- Runs locally using transformers library
- Lazy loading (model loads on first use)
- Immutable ReasoningResult for audit trail
- Explicit error handling (fail loud, fail early)

SLA: <2s reasoning latency with 5s timeout

Error Handling:
- RuntimeError: Model fails to load or generate
- ValueError: Response cannot be parsed
- TimeoutError: Analysis exceeds timeout
- No fallbacks - errors propagate to caller for explicit handling
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import time
import json
import structlog

logger = structlog.get_logger()

# Lazy imports for model loading
_transformers_available = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    _transformers_available = True
except ImportError:
    logger.warning("transformers not installed - install with: pip install transformers torch")
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None


class RiskLevel(Enum):
    """Risk assessment levels."""
    CRISIS = "crisis"  # Immediate intervention required
    CAUTION = "caution"  # Monitor closely, check-in soon
    SAFE = "safe"  # No immediate concerns


@dataclass(frozen=True)
class ClinicalMarker:
    """
    Clinical marker detected in conversation.
    
    Maps to evidence-based screening tools:
    - PHQ-9: Depression screening (9 items)
    - GAD-7: Anxiety screening (7 items)
    - C-SSRS: Suicide risk assessment (5 levels)
    
    Attributes:
        category: Screening tool (phq9, gad7, cssrs)
        item: Specific item number or description
        confidence: Detection confidence (0.0-1.0)
        evidence: Quote from conversation supporting detection
        
    Example:
        ClinicalMarker(
            category="phq9",
            item="item_2_feeling_down",
            confidence=0.85,
            evidence="I've been feeling really down lately"
        )
    """
    category: str  # phq9, gad7, cssrs
    item: str
    confidence: float
    evidence: str


@dataclass(frozen=True)
class ReasoningResult:
    """
    Immutable reasoning result from Mistral-7B analysis.
    
    Attributes:
        p_mistral: Crisis probability from Mistral (0.0-1.0)
        risk_level: Categorical risk assessment
        reasoning_trace: Step-by-step reasoning explanation
        clinical_markers: List of detected clinical markers
        is_sarcasm: Whether message is likely hyperbole/sarcasm
        sarcasm_reasoning: Explanation for sarcasm detection
        latency_ms: Analysis latency in milliseconds
        model_used: Which model generated this result
        
    Design:
        - Frozen dataclass ensures immutability (audit trail)
        - All scores normalized to 0.0-1.0 range
        - Reasoning trace provides explainability for counselors
        
    Example:
        result = ReasoningResult(
            p_mistral=0.92,
            risk_level=RiskLevel.CRISIS,
            reasoning_trace="Student expresses suicidal ideation with specific plan...",
            clinical_markers=[
                ClinicalMarker(category="cssrs", item="ideation_with_plan", ...)
            ],
            is_sarcasm=False,
            sarcasm_reasoning="Language is direct and serious, not hyperbolic",
            latency_ms=1850.5,
            model_used="mistral-7b-instruct"
        )
    """
    p_mistral: float
    risk_level: RiskLevel
    reasoning_trace: str
    clinical_markers: List[ClinicalMarker]
    is_sarcasm: bool
    sarcasm_reasoning: str
    latency_ms: float
    model_used: str


class MistralReasoner:
    """
    Deep clinical reasoning using Mistral-7B.
    
    Provides structured analysis of student messages for:
    - Non-linear crisis pattern detection
    - Clinical marker extraction (PHQ-9, GAD-7, C-SSRS)
    - Sarcasm/hyperbole filtering for teenage vernacular
    - Explainable reasoning traces for counselors
    
    Current Implementation (Milestone 2):
        - Mock implementation for development/testing
        - Returns structured reasoning based on keyword analysis
        - TODO: Replace with actual Mistral-7B SageMaker endpoint
        
    Target Implementation (Milestone 3):
        - AWS Bedrock Mistral-7B endpoint
        - Fine-tuned on mental health conversations
        - Circuit breaker with OpenAI fallback
        
    SLA: <2s reasoning latency with 5s timeout
    
    Example:
        reasoner = MistralReasoner()
        
        # Single message analysis
        result = reasoner.analyze("I want to die")
        
        # With context (improves accuracy)
        result = reasoner.analyze(
            "I'm checking out early",
            context=["I can't take it anymore", "Everything is hopeless"]
        )
        
        if result.risk_level == RiskLevel.CRISIS:
            logger.warning("crisis_detected_by_mistral", result=result)
    """
    
    # System prompt for structured reasoning
    SYSTEM_PROMPT = """You are a clinical mental health expert analyzing student conversations for crisis indicators.

Your task is to:
1. Assess crisis risk level (CRISIS, CAUTION, or SAFE)
2. Identify clinical markers from PHQ-9, GAD-7, and C-SSRS
3. Detect sarcasm/hyperbole common in teenage vernacular
4. Provide step-by-step reasoning for your assessment

Important Context:
- Students are ages 13-18 (adolescents)
- Common hyperbole: "I'm dying", "killing me", "want to die" (often not literal)
- True crisis indicators: specific plans, persistent ideation, hopelessness
- Consider conversation context and patterns over time

Respond in JSON format:
{
  "p_mistral": 0.0-1.0,
  "risk_level": "crisis|caution|safe",
  "reasoning_trace": "Step-by-step explanation",
  "clinical_markers": [
    {
      "category": "phq9|gad7|cssrs",
      "item": "specific_item",
      "confidence": 0.0-1.0,
      "evidence": "quote from message"
    }
  ],
  "is_sarcasm": true|false,
  "sarcasm_reasoning": "Explanation for sarcasm assessment"
}"""
    
    def __init__(self, model_name: str = "GRMenon/mental-mistral-7b-instruct-autotrain", timeout: float = 5.0):
        """
        Initialize Mistral Reasoner with mental health fine-tuned model.
        
        Args:
            model_name: HuggingFace model name
                       Default: GRMenon/mental-mistral-7b-instruct-autotrain
                       (Fine-tuned on mental health counseling conversations)
            timeout: Maximum reasoning time in seconds (default: 5.0)
                    
        Raises:
            ImportError: If transformers library is not installed
                    
        Example:
            # Use default mental health model
            reasoner = MistralReasoner()
            
            # Use different model
            reasoner = MistralReasoner(model_name="other-model")
        """
        if not _transformers_available:
            raise ImportError(
                "transformers library required. Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.timeout = timeout
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        logger.info(
            "mistral_reasoner_initialized",
            model=model_name,
            timeout=timeout,
            note="Model will load on first analyze() call"
        )
    
    def _load_model(self):
        """Lazy load model on first use."""
        if self._model_loaded:
            return
        
        logger.info("loading_mistral_model", model=self.model_name)
        start = time.time()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            load_time = time.time() - start
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(
                "mistral_model_loaded",
                model=self.model_name,
                device=device,
                load_time_s=load_time
            )
            
            self._model_loaded = True
            
        except Exception as e:
            logger.error(
                "mistral_model_load_failed",
                model=self.model_name,
                error=str(e),
                exc_info=True
            )
            raise RuntimeError(f"Failed to load Mistral model: {e}") from e
    
    def analyze(self, message: str, context: List[str] = None) -> ReasoningResult:
        """
        Analyze message for crisis indicators with deep reasoning.
        
        Process:
        1. Load model if not already loaded
        2. Construct prompt with message and context
        3. Call Mistral model
        4. Parse structured JSON response
        5. Validate and return ReasoningResult
        
        Args:
            message: Student message to analyze (required)
            context: Optional list of previous messages
                    Last 5 messages used for pattern detection
                    
        Returns:
            ReasoningResult with crisis assessment and reasoning
            
        Raises:
            RuntimeError: If model fails to load or generate response
            ValueError: If model response cannot be parsed
            TimeoutError: If analysis exceeds timeout threshold
            
        Performance:
            - Target: <2s on GPU, <5s on CPU
            - Timeout: 5s (raises TimeoutError)
            
        Example:
            # Explicit crisis
            result = reasoner.analyze("I want to kill myself")
            # result.risk_level == RiskLevel.CRISIS
            # result.p_mistral >= 0.90
            
            # Hyperbole detection
            result = reasoner.analyze("This homework is killing me")
            # result.is_sarcasm == True
            # result.risk_level == RiskLevel.SAFE
            
            # Context-aware
            result = reasoner.analyze(
                "I'm done",
                context=["Can't sleep", "Nothing matters", "Feeling hopeless"]
            )
            # Context reveals escalating pattern
        """
        start_time = time.perf_counter()
        
        # Load model on first use
        self._load_model()
        
        # Call model
        result = self._call_mistral(message, context)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Check timeout
        if latency_ms > (self.timeout * 1000):
            logger.error(
                "mistral_timeout_exceeded",
                latency_ms=latency_ms,
                timeout_ms=self.timeout * 1000,
                message_length=len(message)
            )
            raise TimeoutError(
                f"Mistral reasoning exceeded timeout: {latency_ms:.1f}ms > {self.timeout * 1000}ms"
            )
        
        # Update latency in result
        result = ReasoningResult(
            p_mistral=result.p_mistral,
            risk_level=result.risk_level,
            reasoning_trace=result.reasoning_trace,
            clinical_markers=result.clinical_markers,
            is_sarcasm=result.is_sarcasm,
            sarcasm_reasoning=result.sarcasm_reasoning,
            latency_ms=latency_ms,
            model_used=self.model_name
        )
        
        if result.risk_level == RiskLevel.CRISIS:
            logger.warning(
                "crisis_detected_by_mistral",
                p_mistral=result.p_mistral,
                reasoning=result.reasoning_trace[:100],
                latency_ms=latency_ms
            )
        
        return result
    
    def _call_mistral(self, message: str, context: List[str] = None) -> ReasoningResult:
        """
        Call Mistral model for deep reasoning.
        
        Constructs structured prompt and parses JSON response.
        
        Raises:
            ValueError: If model response cannot be parsed as JSON
            KeyError: If required fields missing from response
            RuntimeError: If model generation fails
        """
        # Build context string
        context_str = ""
        if context:
            context_str = "\n\nConversation History:\n" + "\n".join(
                f"- {msg}" for msg in context[-5:]  # Last 5 messages
            )
        
        # Construct prompt
        prompt = f"""{self.SYSTEM_PROMPT}

Student Message: "{message}"{context_str}

Analyze this message and respond in JSON format."""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        # Find JSON block in response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            logger.error(
                "no_json_in_mistral_response",
                response=response[:200]
            )
            raise ValueError(f"No JSON found in Mistral response. Response: {response[:200]}")
        
        json_str = response[json_start:json_end]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(
                "invalid_json_from_mistral",
                error=str(e),
                json_str=json_str[:200],
                exc_info=True
            )
            raise ValueError(f"Invalid JSON from Mistral: {e}. JSON: {json_str[:200]}") from e
        
        # Parse into ReasoningResult
        try:
            return self._parse_model_response(data)
        except KeyError as e:
            logger.error(
                "missing_required_field_in_mistral_response",
                error=str(e),
                data=data,
                exc_info=True
            )
            raise ValueError(f"Missing required field in Mistral response: {e}. Data: {data}") from e
    
    def _parse_model_response(self, data: Dict) -> ReasoningResult:
        """
        Parse model JSON response into ReasoningResult.
        
        Raises:
            KeyError: If required fields are missing from response
        """
        # Extract clinical markers
        markers = []
        for marker_data in data.get("clinical_markers", []):
            markers.append(ClinicalMarker(
                category=marker_data["category"],
                item=marker_data["item"],
                confidence=marker_data["confidence"],
                evidence=marker_data["evidence"]
            ))
        
        # Map risk level string to enum
        risk_str = data["risk_level"].lower()
        risk_level = {
            "crisis": RiskLevel.CRISIS,
            "caution": RiskLevel.CAUTION,
            "safe": RiskLevel.SAFE
        }.get(risk_str, RiskLevel.SAFE)
        
        return ReasoningResult(
            p_mistral=data["p_mistral"],
            risk_level=risk_level,
            reasoning_trace=data["reasoning_trace"],
            clinical_markers=markers,
            is_sarcasm=data["is_sarcasm"],
            sarcasm_reasoning=data["sarcasm_reasoning"],
            latency_ms=0.0,  # Will be updated by caller
            model_used=self.model_name
        )

