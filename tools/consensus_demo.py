"""
Consensus Orchestrator Demo

Interactive demo showing parallel consensus pipeline in action.

Usage:
    python tools/consensus_demo.py                    # Interactive mode
    python tools/consensus_demo.py --batch            # Batch demo with test cases
    python tools/consensus_demo.py --message "text"   # Single message
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import ConsensusOrchestrator, ConsensusConfig, RiskLevel
from src.safety.safety_analyzer import SafetyAnalyzer
from src.reasoning.mistral_reasoner import MistralReasoner


# Test cases for batch mode
TEST_CASES = [
    {
        "message": "I'm feeling great today!",
        "expected": RiskLevel.SAFE,
        "description": "Positive message"
    },
    {
        "message": "I want to kill myself",
        "expected": RiskLevel.CRISIS,
        "description": "Explicit suicidal ideation"
    },
    {
        "message": "I've been feeling really down lately",
        "expected": RiskLevel.CAUTION,
        "description": "Depressed mood"
    },
    {
        "message": "This homework is killing me",
        "expected": RiskLevel.SAFE,
        "description": "Teenage hyperbole (should be filtered)"
    },
    {
        "message": "I'm going to hurt myself tonight",
        "expected": RiskLevel.CRISIS,
        "description": "Self-harm with intent"
    },
    {
        "message": "Nobody would care if I disappeared",
        "expected": RiskLevel.CAUTION,
        "description": "Indirect suicidal ideation"
    },
    {
        "message": "I can't sleep and I have no energy",
        "expected": RiskLevel.CAUTION,
        "description": "Depression symptoms"
    },
    {
        "message": "Everything is fine",
        "expected": RiskLevel.SAFE,
        "description": "Safe message"
    },
]


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_result(result, show_details: bool = True):
    """Print consensus result in formatted way."""
    # Risk level with color
    risk_colors = {
        RiskLevel.SAFE: "\033[92m",      # Green
        RiskLevel.CAUTION: "\033[93m",   # Yellow
        RiskLevel.CRISIS: "\033[91m",    # Red
    }
    reset = "\033[0m"
    
    color = risk_colors.get(result.risk_level, "")
    print(f"\n{color}Risk Level: {result.risk_level.value}{reset}")
    print(f"Final Score: {result.final_score:.4f}")
    print(f"Total Latency: {result.total_latency_ms}ms")
    
    if result.timeout_occurred:
        print("⚠️  Mistral timeout occurred (graceful degradation)")
    
    if show_details:
        print("\nLayer Scores:")
        print(f"  Regex:    {result.regex_score.score:.4f} ({result.regex_score.latency_ms}ms)")
        print(f"  Semantic: {result.semantic_score.score:.4f} ({result.semantic_score.latency_ms}ms)")
        if result.mistral_score:
            print(f"  Mistral:  {result.mistral_score.score:.4f} ({result.mistral_score.latency_ms}ms)")
        else:
            print(f"  Mistral:  TIMEOUT")
        
        if result.matched_patterns:
            print(f"\nMatched Patterns: {', '.join(result.matched_patterns)}")
        
        print("\nReasoning Trace:")
        print("-" * 80)
        print(result.reasoning)
        print("-" * 80)


async def analyze_message(orchestrator: ConsensusOrchestrator, message: str, session_id: str = "demo_session"):
    """Analyze a single message."""
    print(f"\nMessage: \"{message}\"")
    
    try:
        result = await orchestrator.analyze(message, session_id)
        print_result(result)
        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def interactive_mode(orchestrator: ConsensusOrchestrator):
    """Run interactive demo."""
    print_header("Consensus Orchestrator - Interactive Demo")
    print("\nType messages to analyze (or 'quit' to exit)")
    print("Commands:")
    print("  quit     - Exit demo")
    print("  config   - Show current configuration")
    print("  help     - Show this help")
    
    session_id = "interactive_session"
    
    while True:
        try:
            message = input("\n> ").strip()
            
            if not message:
                continue
            
            if message.lower() == "quit":
                print("\nGoodbye!")
                break
            
            if message.lower() == "config":
                print("\nCurrent Configuration:")
                config_dict = orchestrator.config.to_dict()
                print(f"  Weights: {config_dict['weights']}")
                print(f"  Thresholds: {config_dict['thresholds']}")
                print(f"  Timeouts: {config_dict['timeouts']}")
                continue
            
            if message.lower() == "help":
                print("\nCommands:")
                print("  quit     - Exit demo")
                print("  config   - Show current configuration")
                print("  help     - Show this help")
                continue
            
            await analyze_message(orchestrator, message, session_id)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def batch_mode(orchestrator: ConsensusOrchestrator):
    """Run batch demo with test cases."""
    print_header("Consensus Orchestrator - Batch Demo")
    
    results = []
    correct = 0
    total = len(TEST_CASES)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] {test_case['description']}")
        print(f"Expected: {test_case['expected'].value}")
        
        result = await analyze_message(
            orchestrator,
            test_case['message'],
            f"batch_session_{i}"
        )
        
        if result:
            results.append({
                "test_case": test_case,
                "result": result,
                "correct": result.risk_level == test_case['expected']
            })
            
            if result.risk_level == test_case['expected']:
                print("✅ PASS")
                correct += 1
            else:
                print(f"❌ FAIL (got {result.risk_level.value})")
    
    # Summary
    print_header("Batch Demo Summary")
    print(f"\nTotal: {total}")
    print(f"Passed: {correct}")
    print(f"Failed: {total - correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    
    # Performance stats
    latencies = [r['result'].total_latency_ms for r in results]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    print(f"\nLatency Stats:")
    print(f"  Average: {avg_latency:.1f}ms")
    print(f"  Min: {min_latency}ms")
    print(f"  Max: {max_latency}ms")
    
    # Timeout stats
    timeouts = sum(1 for r in results if r['result'].timeout_occurred)
    print(f"\nTimeouts: {timeouts}/{total}")


async def single_message_mode(orchestrator: ConsensusOrchestrator, message: str):
    """Analyze a single message."""
    print_header("Consensus Orchestrator - Single Message Analysis")
    await analyze_message(orchestrator, message)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Consensus Orchestrator Demo")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch demo with test cases"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Analyze a single message"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize services
    print("Initializing services...")
    
    try:
        safety_analyzer = SafetyAnalyzer()
        mistral_reasoner = MistralReasoner()
        
        # Load config if provided
        if args.config:
            import json
            with open(args.config) as f:
                config_data = json.load(f)
            config = ConsensusConfig(**config_data)
        else:
            config = ConsensusConfig()
        
        orchestrator = ConsensusOrchestrator(
            safety_service=safety_analyzer,
            mistral_reasoner=mistral_reasoner,
            config=config
        )
        
        print("✅ Services initialized")
        
        # Run appropriate mode
        if args.batch:
            asyncio.run(batch_mode(orchestrator))
        elif args.message:
            asyncio.run(single_message_mode(orchestrator, args.message))
        else:
            asyncio.run(interactive_mode(orchestrator))
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
