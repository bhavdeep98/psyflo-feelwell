"""
Download Mistral-7B Model for Milestone 2

Downloads GRMenon/mental-mistral-7b-instruct-autotrain from HuggingFace.
Model size: ~14GB

This will cache the model locally for future use.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("DOWNLOADING MISTRAL-7B MODEL")
print("=" * 80)
print()
print("Model: GRMenon/mental-mistral-7b-instruct-autotrain")
print("Size: ~14GB")
print("This may take several minutes depending on your connection...")
print()

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print("Step 1/2: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('GRMenon/mental-mistral-7b-instruct-autotrain')
    print("✅ Tokenizer downloaded successfully!")
    print()
    
    print("Step 2/2: Downloading model (~14GB)...")
    print("This will take several minutes. Please be patient...")
    
    # Download model (will use CPU by default)
    model = AutoModelForCausalLM.from_pretrained(
        'GRMenon/mental-mistral-7b-instruct-autotrain',
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True
    )
    
    print("✅ Model downloaded successfully!")
    print()
    
    # Test model
    print("Step 3/3: Testing model...")
    test_input = tokenizer("Hello, how are you?", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**test_input, max_new_tokens=10)
    
    print("✅ Model test successful!")
    print()
    
    print("=" * 80)
    print("MODEL DOWNLOAD COMPLETE")
    print("=" * 80)
    print()
    print("The model is now cached locally and ready to use.")
    print("You can now run:")
    print("  - python tools/reasoning_dashboard.py")
    print("  - python evaluation/suites/reasoning_eval.py")
    print()
    
except ImportError as e:
    print(f"❌ Error: Missing required packages: {e}")
    print()
    print("Please install required packages:")
    print("  pip install transformers torch")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print()
    print("Possible issues:")
    print("  - Network connection problems")
    print("  - Insufficient disk space (~14GB required)")
    print("  - HuggingFace Hub access issues")
    sys.exit(1)
