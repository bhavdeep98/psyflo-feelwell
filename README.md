# PsyFlo (Feelwell)

**AI-Driven Mental Health Triage System for Schools**

PsyFlo is a mission-critical platform that bridges the gap between high-volume student emotional needs and limited counseling resources. By automating initial engagement and risk assessment, we ensure no student's plea for help goes unnoticed.

---

## 🎯 Core Mission

**Safety First, Always.** We provide empathetic AI-driven triage that identifies students in crisis with ≥99.5% recall while maintaining the human touch that builds trust.

## 🏗️ Architecture Overview

PsyFlo uses a **Parallel Consensus Model** where crisis detection runs independently of conversational AI:

```
Student Message
    ├─→ Safety Service (Deterministic + Semantic)
    ├─→ Observer Service (Clinical Markers)
    ├─→ LLM Service (Peer-Expert Persona)
    └─→ Chat Orchestrator (Consensus Scoring)
         ├─→ Crisis (Sc ≥ 0.90): Hard Override
         ├─→ Caution (0.65 ≤ Sc < 0.90): Persona Shim
         └─→ Safe (Sc < 0.65): Full Autonomy
```

### Key Services

- **Safety Service**: Multi-layer crisis detection (regex + semantic + sarcasm filter)
- **Observer Service**: Tracks PHQ-9/GAD-7 clinical markers longitudinally
- **LLM Service**: "Peer-Expert" persona for empathetic engagement
- **Crisis Engine**: Multi-layer notification and escalation orchestration

## 📋 Project Documentation

### Essential Reading
- [Project Tenets](.kiro/steering/00-project-tenets.md) - 15 foundational principles
- [High-Level Design](docs/HLD.md) - Complete architecture specification
- [Glossary](.kiro/steering/01-glossary.md) - Terminology reference
- [Coding Standards](.kiro/steering/03-coding-standards.md) - Safety-critical code requirements

### Compliance & Safety
- [Failure Modes & Mitigation](.kiro/steering/02-failure-modes-mitigation.md)
- FERPA, COPPA, SOC 2 Type II compliant
- HIPAA-ready architecture (AWS Bedrock with BAA support)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- AWS Account (for deployment)
- Docker (for local development)

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/psyflo.git
cd psyflo

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install

# Run tests
pytest  # Backend
npm test  # Frontend
```

## 🔒 Security & Privacy

### Zero PII in Logs
```python
# ❌ NEVER
logger.info(f"Student {student_id} logged in")

# ✅ ALWAYS
logger.info(f"Student {hash_pii(student_id)} logged in")
```

### Data Lifecycle
- **Day 0-30**: Redis + RDS (Hot)
- **Day 30-90**: S3 Standard (Warm)
- **Day 90-7yr**: S3 Glacier (Cold)
- **7 years**: Permanent Delete

### Access Control (RBAC)
- **Student**: Own conversations only
- **Counselor**: Assigned school students only
- **School Admin**: Aggregated reports only
- **Platform Admin**: System health, no student data

## 📊 SLA Targets

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Crisis Recall | ≥99.5% | Can't miss students in crisis |
| Crisis Detection | <50ms | Student in crisis can't wait |
| Alert Delivery | <5 min | Counselor must be notified immediately |
| API Response | <2s P95 | Engagement requires responsiveness |
| System Uptime | 99.9% | Students expect us to be available |

## 🧪 Testing

### Safety-Critical Code: 100% Coverage Required

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run only safety-critical tests
pytest -m safety_critical

# Run integration tests
pytest -m integration
```

### Golden Test Sets
- **MentalChat16K**: 16,000 mental health conversations
- **Crisis Test Set**: 193 cases (expanding to 10,000+)
- **Adversarial Set**: Coded language, leetspeak, evasive patterns

## 🏛️ The 60-Second Litmus Test

Every piece of code must answer these questions in 60 seconds:

1. **What does this file do?**
2. **What happens if this fails?**
3. **Where would I add a log statement to debug this?**

If no → Refactor immediately.

## 🎯 Design Principles

1. **Safety-First Parallelism**: Crisis detection runs concurrently with LLM
2. **Persona Integrity**: Consistent "Peer-Expert" voice builds trust
3. **Explainable Triage**: Every alert includes evidence snippets
4. **Zero-Trust Privacy**: PII hashed at network edge
5. **Event-Driven Resilience**: Services communicate via immutable message bus

## 🛠️ Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Database**: PostgreSQL (RDS), Redis (ElastiCache)
- **Storage**: S3 + Parquet (Data Lake)
- **Messaging**: SNS + SQS
- **Compute**: ECS Fargate

### Frontend
- **Language**: TypeScript 5.0+
- **Framework**: React 18+
- **State**: Redux Toolkit
- **Testing**: Vitest + React Testing Library

### AI/ML
- **LLM**: AWS Bedrock (HIPAA-ready), OpenAI (fallback)
- **Embeddings**: all-MiniLM-L6-v2
- **Sarcasm Filter**: DistilBERT (fine-tuned on adolescent vernacular)

## 📈 Monitoring & Observability

### Three Pillars
1. **Logs**: Structured logging with structlog (zero PII)
2. **Metrics**: Prometheus + Grafana
3. **Traces**: AWS X-Ray for distributed tracing

### Dashboards
- **Ops**: System health, latency, error rates
- **Counselors**: Student risk, alert status
- **Compliance**: Audit trails, data access logs

## 🚢 Deployment

### Blue/Green Deployment
- Zero-downtime deployments
- Instant rollback capability
- Two identical production environments

### Canary Deployment
- Gradual rollout to small % of users
- Monitor metrics before full expansion
- Use for: New LLM models, crisis keyword patterns

## 🤝 Contributing

### Before Submitting PR

- [ ] All functions have type hints
- [ ] No PII in log statements
- [ ] No bare `except:` clauses
- [ ] Enums used for fixed values
- [ ] Safety-critical code has 100% test coverage
- [ ] Passes 60-second litmus test
- [ ] Documentation updated

### Code Review Process
1. Automated checks (linting, tests, security scan)
2. Peer review (2 approvals required)
3. Security review (for changes to safety/auth)
4. Merge to main

## 📞 Support & Contact

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Security**: security@psyflo.com (PGP key available)
- **Compliance**: compliance@psyflo.com

## 📜 License

Proprietary - All Rights Reserved

## ⚠️ Important Notice

**The stakes are high: mental health + minors = zero tolerance for bugs.**

This system handles sensitive mental health data for minors. Every line of code must prioritize:
1. Student safety
2. Privacy compliance
3. Explainability
4. Reliability

When in doubt, fail loud and fail early. Silent failures are unacceptable.

---

**Built with care for students who need support. 💙**
