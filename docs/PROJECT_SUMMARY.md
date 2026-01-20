# PsyFlo Project Summary

**Last Updated**: 2026-01-20

---

## Project Overview

**PsyFlo (Feelwell)** is a mission-critical, AI-driven mental health triage platform designed specifically for school districts and youth organizations. The system addresses the growing disparity between high-volume student emotional needs and limited professional counseling resources.

### Core Value Proposition

- **For Students**: Safe, empathetic AI companion available 24/7 that builds trust through routine engagement
- **For Counselors**: Evidence-based triage system that prioritizes students by risk level with conversation snippets for context
- **For Schools**: Scalable mental health support that ensures no student's plea for help goes unnoticed

---

## Key Differentiators

### 1. Safety-First Architecture
Unlike traditional chatbots, PsyFlo's crisis detection runs **independently** of the conversational AI:
- Deterministic regex layer catches explicit crisis keywords (<10ms)
- Semantic layer identifies obfuscated language (coded phrases, indirect cries for help)
- Sarcasm filter prevents false positives from teenage hyperbole
- **Target**: ≥99.5% crisis recall with <10% false positive rate

### 2. Parallel Consensus Model
Traditional systems use linear "Safety Filter → LLM" which is slow and robotic. PsyFlo runs all analysis in parallel:
```
Student Message → [Safety | Observer | LLM] → Orchestrator → Response
                   (concurrent processing)
```
- **Result**: Sub-second response time without sacrificing safety

### 3. Evidence-Based Triage
Counselors receive:
- Risk trajectory (improving vs declining) based on PHQ-9/GAD-7 markers
- Conversation snippets showing exactly why student was flagged
- Longitudinal view of mental health journey, not just snapshots

### 4. "Peer-Expert" Persona
Carefully designed voice that:
- Feels like supportive mentorship, not clinical evaluation
- Uses age-appropriate, informal language
- Maintains consistency through versioned system prompts
- Builds trust through routine and reliability

---

## Technical Architecture

### Microservices

| Service | Purpose | SLA |
|---------|---------|-----|
| **Safety Service** | Multi-layer crisis detection | <50ms |
| **Observer Service** | Track PHQ-9/GAD-7 clinical markers | <100ms |
| **LLM Service** | Generate empathetic responses | <2s P95 |
| **Crisis Engine** | Multi-layer notification & escalation | <5min alert delivery |
| **Chat Orchestrator** | Consensus scoring & decision routing | <100ms |

### Data Architecture

**Hybrid Storage Model** (cost-optimized):
- **RDS (PostgreSQL)**: Structured metadata, clinical scores (ACID compliance)
- **S3 + Parquet**: Long-term conversation history (90% cost reduction vs DocumentDB)
- **Redis (ElastiCache)**: Hot cache for active sessions, last 10 messages

**Data Lifecycle**:
- Day 0-30: Hot (Redis + RDS)
- Day 30-90: Warm (S3 Standard)
- Day 90-7yr: Cold (S3 Glacier)
- 7 years: Permanent delete

### Event-Driven Resilience

**SNS + SQS Message Bus**:
- All internal events published to central SNS topic
- Services subscribe via SQS queues (crisis-alert, audit-log, analytics)
- If service is down, messages persist in queue (no data loss)

---

## Compliance & Security

### Regulatory Compliance
- ✅ **FERPA**: Student education records privacy
- ✅ **COPPA**: Children under 13 online privacy
- ✅ **SOC 2 Type II**: Enterprise security audit
- ✅ **HIPAA-ready**: AWS Bedrock with BAA support

### Privacy by Design
- **Zero PII in logs**: All identifiers hashed with `hash_pii()`
- **Field-level encryption**: AWS KMS for clinical evidence snippets
- **RBAC**: Counselors only see their assigned school
- **k-anonymity (k≥5)**: All aggregated reports
- **Immutable audit trail**: WORM storage for compliance

### Data Boundaries
- **USA-only initially**: FERPA/COPPA compliance
- **Geographic expansion**: Requires compliance review (GDPR for Europe)
- **Data residency**: Enforced at signup

---

## Development Standards

### The 60-Second Litmus Test
Every piece of code must answer in 60 seconds:
1. What does this file do?
2. What happens if this fails?
3. Where would I add a log statement to debug this?

### Safety-Critical Requirements
- **100% test coverage** for all crisis detection code
- **Type hints** on all function signatures
- **No bare `except:` clauses** - explicit exception handling
- **Enums for fixed values** - make illegal states unrepresentable
- **Immutable data structures** - prevent state corruption

### Testing Strategy
- **Golden Test Sets**: MentalChat16K (16K conversations), Crisis Set (10K+ target)
- **Adversarial Testing**: Coded language, leetspeak, evasive patterns
- **Integration Tests**: End-to-end crisis flow validation

---

## Key Metrics & SLAs

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Crisis Recall** | ≥99.5% | Can't miss students in crisis |
| **Crisis Detection** | <50ms | Student in crisis can't wait |
| **Alert Delivery** | <5 min | Counselor must be notified immediately |
| **API Response** | <2s P95 | Engagement requires responsiveness |
| **System Uptime** | 99.9% | Students expect us to be available |
| **False Positive Rate** | <10% | Prevent counselor alert fatigue |

---

## Technology Stack

### Backend
- **Language**: Python 3.11+ (type hints required)
- **Framework**: FastAPI
- **Database**: PostgreSQL (RDS), Redis (ElastiCache)
- **Storage**: S3 + Parquet
- **Messaging**: SNS + SQS
- **Compute**: ECS Fargate (serverless containers)

### Frontend
- **Language**: TypeScript 5.0+
- **Framework**: React 18+ (strict mode)
- **State**: Redux Toolkit
- **Testing**: Vitest + React Testing Library

### AI/ML
- **LLM**: AWS Bedrock (primary), OpenAI (fallback)
- **Embeddings**: all-MiniLM-L6-v2 (semantic layer)
- **Sarcasm Filter**: DistilBERT (fine-tuned on adolescent vernacular)

---

## Project Tenets (Abbreviated)

1. **Safety First, Always**: Deterministic guardrails, 100% test coverage
2. **Compliance is Non-Negotiable**: FERPA, COPPA, SOC 2, zero PII in logs
3. **Explicit Over Clever**: Code must be traceable and self-documenting
4. **Fail Loud, Fail Early**: Never silently swallow errors
5. **Make Illegal States Unrepresentable**: Use Enums, type hints
6. **Event-Driven Crisis Response**: Crisis detection decoupled from chat
7. **Immutability by Default**: Prevent state corruption
8. **Engagement Before Intervention**: UX is a safety feature
9. **Visibility and Explainability**: Every decision must be traceable
10. **Observable Systems Are Reliable**: Instrument everything
11. **Graceful Degradation**: LLM down? Return safe fallback
12. **Human-in-the-Loop**: AI is triage, not diagnosis
13. **Trust Is Earned**: Transparency about data sharing
14. **Geographic Boundaries Are Explicit**: USA-only initially
15. **Performance Is a Safety Feature**: Slow = broken in crisis

---

## Deployment Strategy

### Blue/Green Deployment
- Two identical production environments
- Instant traffic switch
- Zero-downtime deployments
- Instant rollback capability

### Canary Deployment
- Gradual rollout to small % of users
- Monitor metrics before full expansion
- Use for: New LLM models, crisis keyword patterns

---

## Risk Mitigation

### Operational Failures
- **LLM Provider Downtime**: Multi-provider fallback (Bedrock → OpenAI → Pre-written)
- **Database Outage**: Redis cache serves recent conversations
- **Notification Failure**: SQS persistence, display resources directly to student

### Data & Privacy Failures
- **Data Breach**: Field-level encryption, PII hashed at edge
- **Unauthorized Access**: Strict RBAC, immutable audit trail
- **Retention Violations**: Automated S3 lifecycle policies

### Human Factors
- **Alert Fatigue**: Sarcasm filter, tiered alerts (Crisis vs Caution)
- **Counselor Skill Gaps**: Evidence snippets, crisis protocol checklists
- **Student Gaming**: Adversarial testing, semantic layer catches obfuscation

---

## Future Roadmap

### Phase 1 (Current): MVP
- USA-only deployment
- English language only
- Web interface
- Core crisis detection + triage

### Phase 2: Expansion
- Multi-language support (Spanish priority)
- Mobile apps (iOS/Android)
- Enhanced analytics dashboard
- Integration with major SIS platforms

### Phase 3: Advanced Features
- Real-time video/voice escalation to counselors
- Predictive analytics (identify at-risk students before crisis)
- Integration with electronic health records (EHR)
- Geographic expansion (Canada, Europe with GDPR compliance)

---

## Success Criteria

### Technical Success
- ✅ 99.9% uptime
- ✅ ≥99.5% crisis recall
- ✅ <10% false positive rate
- ✅ <2s P95 response latency

### Product Success
- ✅ 70%+ student engagement rate (weekly active users)
- ✅ 90%+ counselor satisfaction score
- ✅ 50%+ reduction in counselor triage time
- ✅ Zero missed crisis incidents (with system available)

### Business Success
- ✅ SOC 2 Type II certification
- ✅ 10+ school district deployments
- ✅ 100K+ student users
- ✅ <$5 per student per month operational cost

---

## Critical Reminders

**The stakes are high: mental health + minors = zero tolerance for bugs.**

- Every line of code must prioritize student safety
- Privacy compliance is non-negotiable
- Explainability builds counselor trust
- Silent failures are unacceptable
- When in doubt, fail loud and fail early

---

## Documentation Index

- [README.md](../README.md) - Project overview and quick start
- [HLD.md](HLD.md) - High-level design and architecture
- [DECISION_LOG.md](DECISION_LOG.md) - Architecture decision records
- [.kiro/steering/00-project-tenets.md](../.kiro/steering/00-project-tenets.md) - 15 foundational principles
- [.kiro/steering/01-glossary.md](../.kiro/steering/01-glossary.md) - Terminology reference
- [.kiro/steering/02-failure-modes-mitigation.md](../.kiro/steering/02-failure-modes-mitigation.md) - Risk mitigation strategies
- [.kiro/steering/03-coding-standards.md](../.kiro/steering/03-coding-standards.md) - Safety-critical code requirements

---

**Built with care for students who need support. 💙**
