"""
Reasoning Dashboard - Interactive Demo for Milestone 2

Demonstrates the "Hidden Clinician" logic with real-time text analysis,
clinical reasoning traces, and seven-dimension metric scoring.

Features:
- Real-time crisis detection with Mistral reasoning
- Clinical marker extraction (PHQ-9, GAD-7, C-SSRS)
- Sarcasm/hyperbole detection
- Seven-dimension clinical metric display
- Side-by-side comparison with deterministic layer

Usage:
    # Interactive mode
    python tools/reasoning_dashboard.py
    
    # Batch mode with test cases
    python tools/reasoning_dashboard.py --batch
    
    # Evaluate specific message
    python tools/reasoning_dashboard.py --message "I want to die"
"""

import sys
import argparse
from typing import List, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.mistral_reasoner import MistralReasoner, RiskLevel
from src.safety.safety_analyzer import SafetyService
from src.reasoning.clinical_metrics import ClinicalMetrics


class ReasoningDashboard:
    """
    Interactive dashboard for demonstrating deep reasoning capabilities.
    
    Displays:
    - Mistral reasoning trace
    - Clinical markers detected
    - Risk level assessment
    - Comparison with deterministic layer
    - Sarcasm detection
    """
    
    def __init__(self):
        """Initialize dashboard with reasoner and safety service."""
        self.reasoner = MistralReasoner()
        self.safety_service = SafetyService()
        self.metrics = ClinicalMetrics()
        
        print("🧠 PsyFlo Reasoning Dashboard - Milestone 2")
        print("=" * 80)
        print()
    
    def analyze_message(self, message: str, context: Optional[List[str]] = None) -> None:
        """
        Analyze message and display comprehensive reasoning dashboard.
        
        Args:
            message: Student message to analyze
            context: Optional conversation history
        """
        print(f"📝 Message: \"{message}\"")
        if context:
            print(f"📚 Context: {len(context)} previous messages")
        print()
        
        # Run both layers
        safety_result = self.safety_service.analyze(message, context)
        reasoning_result = self.reasoner.analyze(message, context)
        
        # Display results
        self._display_risk_assessment(safety_result, reasoning_result)
        self._display_reasoning_trace(reasoning_result)
        self._display_clinical_markers(reasoning_result)
        self._display_sarcasm_analysis(reasoning_result)
        self._display_performance_metrics(safety_result, reasoning_result)
        self._display_layer_comparison(safety_result, reasoning_result)
        
        print()
        print("=" * 80)
        print()
    
    def _display_risk_assessment(self, safety_result, reasoning_result) -> None:
        """Display risk level assessment."""
        print("🚨 RISK ASSESSMENT")
        print("-" * 80)
        
        # Determine overall risk
        is_crisis = safety_result.is_crisis or reasoning_result.risk_level == RiskLevel.CRISIS
        
        if is_crisis:
            print("⚠️  CRISIS DETECTED - Immediate intervention required")
            print(f"   Confidence: {max(safety_result.p_regex, safety_result.p_semantic, reasoning_result.p_mistral):.2%}")
        elif reasoning_result.risk_level == RiskLevel.CAUTION:
            print("⚡ CAUTION - Monitor closely, check-in within 24-48 hours")
            print(f"   Confidence: {reasoning_result.p_mistral:.2%}")
        else:
            print("✅ SAFE - No immediate concerns detected")
            print(f"   Confidence: {1.0 - reasoning_result.p_mistral:.2%}")
        
        print()
    
    def _display_reasoning_trace(self, reasoning_result) -> None:
        """Display step-by-step reasoning trace."""
        print("🧠 REASONING TRACE")
        print("-" * 80)
        print(reasoning_result.reasoning_trace)
        print()
    
    def _display_clinical_markers(self, reasoning_result) -> None:
        """Display detected clinical markers."""
        print("🏥 CLINICAL MARKERS")
        print("-" * 80)
        
        if not reasoning_result.clinical_markers:
            print("No clinical markers detected")
        else:
            for marker in reasoning_result.clinical_markers:
                print(f"• {marker.category.upper()}: {marker.item}")
                print(f"  Confidence: {marker.confidence:.2%}")
                print(f"  Evidence: \"{marker.evidence}\"")
                print()
    
    def _display_sarcasm_analysis(self, reasoning_result) -> None:
        """Display sarcasm/hyperbole detection."""
        print("😏 SARCASM ANALYSIS")
        print("-" * 80)
        
        if reasoning_result.is_sarcasm:
            print("⚠️  Hyperbole/sarcasm detected")
            print(f"Reasoning: {reasoning_result.sarcasm_reasoning}")
        else:
            print("✅ No hyperbole detected - language appears genuine")
        
        print()
    
    def _display_performance_metrics(self, safety_result, reasoning_result) -> None:
        """Display performance metrics."""
        print("⚡ PERFORMANCE")
        print("-" * 80)
        print(f"Safety Layer:    {safety_result.latency_ms:.2f}ms")
        print(f"Reasoning Layer: {reasoning_result.latency_ms:.2f}ms")
        print(f"Total:           {safety_result.latency_ms + reasoning_result.latency_ms:.2f}ms")
        print()
    
    def _display_layer_comparison(self, safety_result, reasoning_result) -> None:
        """Display side-by-side comparison of detection layers."""
        print("🔍 LAYER COMPARISON")
        print("-" * 80)
        
        print(f"{'Layer':<20} {'Score':<10} {'Decision':<15} {'Patterns'}")
        print("-" * 80)
        
        # Regex layer
        regex_decision = "CRISIS" if safety_result.p_regex >= 0.85 else "SAFE"
        regex_patterns = ", ".join(safety_result.matched_patterns[:3]) if safety_result.matched_patterns else "None"
        print(f"{'Regex (Deterministic)':<20} {safety_result.p_regex:<10.2%} {regex_decision:<15} {regex_patterns}")
        
        # Semantic layer
        semantic_decision = "CRISIS" if safety_result.p_semantic >= 0.85 else "SAFE"
        print(f"{'Semantic (Embedding)':<20} {safety_result.p_semantic:<10.2%} {semantic_decision:<15}")
        
        # Mistral layer
        mistral_decision = reasoning_result.risk_level.value.upper()
        print(f"{'Mistral (Reasoning)':<20} {reasoning_result.p_mistral:<10.2%} {mistral_decision:<15}")
        
        # Sarcasm filter
        if safety_result.sarcasm_filtered:
            print(f"{'Sarcasm Filter':<20} {safety_result.p_sarcasm:<10.2%} {'FILTERED':<15}")
        
        print()
    
    def run_interactive(self) -> None:
        """Run interactive mode with user input."""
        print("Interactive Mode - Enter messages to analyze (or 'quit' to exit)")
        print()
        
        context = []
        
        while True:
            try:
                message = input("Student message: ").strip()
                
                if message.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not message:
                    continue
                
                # Analyze message
                self.analyze_message(message, context=context[-3:] if context else None)
                
                # Add to context
                context.append(message)
                
                # Ask if user wants to continue conversation
                continue_conv = input("Continue conversation? (y/n): ").strip().lower()
                if continue_conv != 'y':
                    context = []
                    print("\n" + "=" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def run_batch(self) -> None:
        """Run batch mode with predefined test cases."""
        print("Batch Mode - Running test cases")
        print()
        
        test_cases = [
            {
                "message": "I want to kill myself",
                "context": None,
                "description": "Explicit suicidal ideation"
            },
            {
                "message": "This homework is killing me",
                "context": None,
                "description": "Teenage hyperbole (should filter)"
            },
            {
                "message": "I've been feeling really depressed lately",
                "context": None,
                "description": "Moderate distress (caution level)"
            },
            {
                "message": "I'm done",
                "context": ["I can't sleep anymore", "Nothing matters", "I feel hopeless"],
                "description": "Ambiguous with crisis context"
            },
            {
                "message": "I had a good day at school today",
                "context": None,
                "description": "Positive message (safe)"
            },
            {
                "message": "I'm going to hurt myself tonight",
                "context": None,
                "description": "Explicit self-harm intent with plan"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test Case {i}/{len(test_cases)}: {test_case['description']}")
            print("=" * 80)
            self.analyze_message(test_case["message"], test_case["context"])
            
            if i < len(test_cases):
                input("Press Enter to continue to next test case...")
                print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PsyFlo Reasoning Dashboard - Milestone 2 Demo"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch mode with predefined test cases"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Analyze specific message"
    )
    parser.add_argument(
        "--context",
        type=str,
        nargs="+",
        help="Conversation context (previous messages)"
    )
    
    args = parser.parse_args()
    
    dashboard = ReasoningDashboard()
    
    if args.message:
        # Single message mode
        dashboard.analyze_message(args.message, context=args.context)
    elif args.batch:
        # Batch mode
        dashboard.run_batch()
    else:
        # Interactive mode
        dashboard.run_interactive()


if __name__ == "__main__":
    main()
