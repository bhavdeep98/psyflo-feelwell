# PsyFlo (Feelwell) - Mental Health AI Triage System

AI-driven crisis detection and mental health triage for school districts and youth organizations.

## Quick Start

### Milestone 1: Safety Service

```bash
# Setup environment
./setup.sh

# Run tests
./run_tests.sh

# Run interactive demo
python tools/cli_demo.py

# Run benchmarks
python evaluation/benchmark_runner.py --dataset hard_crisis
```

### Milestone 2: Deep Reasoner

```bash
# Interactive reasoning dashboard
python tools/reasoning_dashboard.py

# Batch demo with test cases
python tools/reasoning_dashboard.py --batch

# Analyze specific message
python tools/reasoning_dashboard.py --message "I want to die"

# Run reasoning evaluation
python evaluation/suites/reasoning_eval.py

# Run reasoning tests
pytest tests/test_mistral_reasoner.py tests/test_clinical_metrics.py -v
```

### Milestone 4: Conversational AI Agent

```bash
# Setup OpenAI API key
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENAI_API_KEY

# Install dependencies
pip install langchain langchain-openai langchain-core langgraph

# Run conversation agent tests
pytest tests/test_conversation_agent.py -v

# Start backend
cd backend
uvicorn main:app --reload

# Start frontend (in another terminal)
cd frontend
npm run dev
```

See [Conversation Setup Guide](docs/CONVERSATION_SETUP.md) for detailed instructions.

## Project Structure

```
Psyflo-feelwell/
├── src/
│   ├── safety/                    # Safety Service (Milestone 1)
│   │   ├── safety_analyzer.py     # Main orchestrator
│   │   ├── strategy_factory.py    # Strategy factory
│   │   └── strategies/            # Detection strategies
│   │       ├── base.py            # Abstract base
│   │       ├── regex_strategy.py  # Deterministic keywords
│   │       ├── semantic_strategy.py # Embedding similarity
│   │       └── sarcasm_strategy.py # Hyperbole filter
│   └── reasoning/                 # Reasoning Module (Milestone 2)
│       ├── mistral_reasoner.py    # Deep clinical reasoning
│       └── clinical_metrics.py    # 7-dimension assessment
├── tests/                         # Unit tests
│   ├── test_safety_service.py     # Safety tests
│   ├── test_mistral_reasoner.py   # Reasoning tests
│   └── test_clinical_metrics.py   # Metrics tests
├── tools/                         # Demo & utilities
│   ├── cli_demo.py                # Safety demo (M1)
│   └── reasoning_dashboard.py     # Reasoning demo (M2)
├── evaluation/                    # Benchmarking & evaluation
│   ├── dataset_loader.py          # Dataset loading
│   ├── benchmark_runner.py        # Benchmark suite
│   └── suites/                    # Evaluation suites
│       ├── mentalchat_eval.py     # M1 evaluation
│       └── reasoning_eval.py      # M2 evaluation
├── config/                        # Configuration
│   └── crisis_patterns.yaml       # Crisis detection patterns
├── docs/                          # Documentation
│   ├── HLD.md                     # High-level design
│   ├── LLD.md                     # Low-level design
│   ├── MILESTONES.md              # Project milestones
│   ├── MILESTONE2_SUMMARY.md      # M2 detailed summary
│   ├── MILESTONE2_QUICK_START.md  # M2 quick reference
│   └── DECISION_LOG.md            # Design decisions
└── .kiro/steering/                # Project tenets & standards
```

## Current Status

**Milestone 1**: 85% Complete ✅
- ✅ Safety Service with multi-layer detection
- ✅ Strategy Pattern implementation
- ✅ Comprehensive test suite (100% coverage)
- ✅ Real-world dataset evaluation (19,581 samples)
- ✅ Baseline performance report

**Milestone 2**: 60% Complete 🚧
- ✅ MistralReasoner with structured reasoning
- ✅ Clinical Metrics framework (7 dimensions)
- ✅ Interactive reasoning dashboard
- ✅ Sarcasm detection (92.3% accuracy)
- ⏳ Pending: Mistral-7B deployment
- ⏳ Pending: GPT-4 judge integration

**📊 View Reports**:
- Milestone 1: `reports/milestone1_evaluation.html` or run `./view_report.sh`
- Milestone 2: See [Milestone 2 Summary](docs/MILESTONE2_SUMMARY.md)

## Key Features

### Multi-Layer Crisis Detection (Milestone 1)
- **Regex Layer**: Deterministic keyword matching (safety floor)
- **Semantic Layer**: Embedding similarity for obfuscated language
- **Sarcasm Filter**: Reduces false positives from hyperbole

### Deep Clinical Reasoning (Milestone 2)
- **Mistral Reasoner**: Structured reasoning with clinical marker extraction
- **Clinical Metrics**: 7-dimension assessment framework
- **Sarcasm Detection**: 92.3% accuracy on teenage hyperbole
- **Explainable AI**: Step-by-step reasoning traces for counselors

### Performance
- **Safety Layer**: <50ms P95 (achieved: 10.8ms)
- **Reasoning Layer**: <2s target (mock: <1ms)
- **Throughput**: >20 messages/second
- **Recall**: 100% on explicit keywords ✅
- **Sarcasm Detection**: 92.3% accuracy ✅

### Design Principles
- Safety First (deterministic guardrails)
- Explicit Over Clever (traceable code)
- Immutability by Default (audit trail)
- Observable Systems (comprehensive logging)

## Testing

```bash
# Run all tests
./run_tests.sh

# Run with coverage
./run_tests.sh --coverage

# Run fast tests only (skip benchmarks)
./run_tests.sh --fast

# Run specific test class
pytest tests/test_safety_service.py::TestExplicitCrisisDetection -v
```

## Benchmarking

```bash
# Benchmark on hard crisis dataset
python evaluation/benchmark_runner.py --dataset hard_crisis

# Benchmark on MentalChat16K
python evaluation/benchmark_runner.py --dataset mentalchat16k

# Benchmark on all datasets
python evaluation/benchmark_runner.py --dataset all
```

## Documentation

### Getting Started
- **[README](README.md)** - This file
- **[Milestones](docs/MILESTONES.md)** - Project roadmap and progress

### Architecture & Design
- **[High-Level Design](docs/HLD.md)** - System architecture
- **[Low-Level Design](docs/LLD.md)** - Implementation details

### Standards & Principles
- **[Project Tenets](.kiro/steering/00-project-tenets.md)** - 15 design principles
- **[Glossary](.kiro/steering/01-glossary.md)** - Terminology reference
- **[Failure Modes](.kiro/steering/02-failure-modes-mitigation.md)** - Risk mitigation
- **[Coding Standards](.kiro/steering/03-coding-standards.md)** - Safety-critical code requirements
- **[Design Patterns](.kiro/steering/04-design-patterns.md)** - Architectural patterns

## Development

### Adding Crisis Patterns

Edit `config/crisis_patterns.yaml`:

```yaml
crisis_keywords:
  new_category:
    patterns:
      - "phrase 1"
      - "phrase 2"
    confidence: 0.90
```

Then run tests to validate:

```bash
pytest tests/test_safety_service.py -v
```

### Adding Detection Strategies

1. Create new strategy in `src/safety/strategies/`
2. Inherit from `DetectionStrategy`
3. Implement `analyze()` and `get_name()`
4. Add to factory in `strategy_factory.py`
5. Add tests in `tests/test_safety_service.py`

## License

Proprietary - All Rights Reserved

## Contact

For questions or support, see project documentation.
