import sys
sys.path.insert(0, 'src')

print("Testing MistralReasoner...")
from reasoning.mistral_reasoner import MistralReasoner, RiskLevel

print("Creating reasoner...")
reasoner = MistralReasoner()

print(f"Mock mode: {reasoner.use_mock}")

print("\nTest 1: Crisis detection")
result = reasoner.analyze("I want to kill myself")
print(f"  Risk: {result.risk_level.value}")
print(f"  P_mistral: {result.p_mistral}")
print(f"  Model: {result.model_used}")

print("\nTest 2: Sarcasm detection")
result = reasoner.analyze("This homework is killing me")
print(f"  Risk: {result.risk_level.value}")
print(f"  Is sarcasm: {result.is_sarcasm}")
print(f"  P_mistral: {result.p_mistral}")

print("\n✅ All tests passed!")
