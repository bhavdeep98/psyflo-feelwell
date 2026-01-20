# Architecture Decision Log (ADL)

This document records significant architectural and design decisions for PsyFlo.

---

## ADR-001: Parallel Consensus Model for Crisis Detection

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Traditional safety systems use linear "Safety Filter → LLM" approach, which introduces latency and breaks conversational flow.

**Decision**: Implement Parallel Consensus Model where Safety Service, Observer Service, and LLM Service run concurrently, with Chat Orchestrator aggregating signals.

**Consequences**:
- ✅ Sub-second crisis detection without sacrificing conversational quality
- ✅ Crisis detection independent of LLM (works even if LLM fails)
- ✅ Deterministic safety floor that can't be bypassed
- ⚠️ Increased system complexity (more services to coordinate)
- ⚠️ Requires careful consensus scoring calibration

**Alternatives Considered**:
- Pure LLM prompting: Too risky, prone to jailbreaking
- Linear safety gate: Too slow, feels robotic
- Fine-tuned LLM only: Requires massive datasets, safety still probabilistic

---

## ADR-002: AWS Bedrock as Primary LLM Provider

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Need HIPAA-ready LLM provider for potential healthcare partnerships.

**Decision**: Use AWS Bedrock as primary LLM provider, with OpenAI as fallback.

**Consequences**:
- ✅ HIPAA-ready with Business Associate Agreement (BAA) support
- ✅ Data stays within AWS ecosystem (lower latency, better security)
- ✅ Multiple model options (Claude, Titan, etc.)
- ⚠️ OpenAI may have better model quality for some use cases
- ⚠️ Need to maintain compatibility with multiple providers

**Alternatives Considered**:
- OpenAI only: No BAA support, data leaves AWS
- Self-hosted LLM: Too expensive, requires ML expertise
- Azure OpenAI: Vendor lock-in, less flexible than Bedrock

---

## ADR-003: S3 + Parquet Data Lake Instead of DocumentDB

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: DocumentDB clusters are expensive for long-term conversation storage.

**Decision**: Store raw conversations as JSON in S3, use Parquet format for analytics, query with AWS Athena.

**Consequences**:
- ✅ 90% cost reduction vs DocumentDB
- ✅ "Pay-per-query" model scales with usage
- ✅ Easy to run large-scale analytics for research
- ⚠️ Higher latency for individual conversation retrieval
- ⚠️ Need to maintain hot cache (Redis) for recent conversations

**Alternatives Considered**:
- DocumentDB: Too expensive at scale
- DynamoDB: Good for key-value, but expensive for analytics
- RDS only: Not cost-effective for unstructured conversation data

---

## ADR-004: SNS/SQS Event Bus Instead of Kinesis

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Need event-driven architecture for resilience and decoupling.

**Decision**: Use SNS (pub/sub) + SQS (queues) for event bus instead of Kinesis.

**Consequences**:
- ✅ Lower complexity, easier to debug
- ✅ More cost-effective for initial deployment
- ✅ Built-in retry and dead-letter queue support
- ⚠️ No built-in replay capability (Kinesis has this)
- ⚠️ May need to migrate to Kinesis if throughput exceeds 10K msg/sec

**Alternatives Considered**:
- Kinesis: Higher throughput, replayability, but more complex
- EventBridge: Good for AWS service integration, but overkill for our use case
- RabbitMQ: Self-managed, adds operational overhead

---

## ADR-005: USA-Only Initial Deployment

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Different countries have different privacy laws (GDPR, PIPEDA, etc.).

**Decision**: Launch in USA only initially, with explicit geographic boundaries.

**Consequences**:
- ✅ Focus on FERPA/COPPA compliance only
- ✅ Simpler data residency requirements
- ✅ Faster time to market
- ⚠️ Need compliance review before expanding to new jurisdictions
- ⚠️ GDPR compliance required for European expansion (data must stay in EU)

**Future Considerations**:
- Canada: PIPEDA compliance, similar to FERPA
- Europe: GDPR requires data residency, stricter than US laws
- Australia: Privacy Act 1988, similar to FERPA

---

## ADR-006: Immutable Data Structures for Conversation Sessions

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Mutable state can lead to race conditions and audit trail corruption.

**Decision**: Use immutable data structures (frozen dataclasses) for all conversation sessions.

**Consequences**:
- ✅ Prevents accidental state corruption
- ✅ Makes debugging easier (state changes are explicit)
- ✅ Simplifies audit trail (every state change creates new record)
- ⚠️ Slightly higher memory usage (creates new objects instead of mutating)
- ⚠️ Requires functional programming patterns (may be unfamiliar to some devs)

**Alternatives Considered**:
- Mutable objects with locking: Complex, error-prone
- Event sourcing: Overkill for our use case, adds complexity

---

## ADR-007: Multi-Layer Crisis Detection (Deterministic + Semantic)

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Single-layer detection misses obfuscated language or has too many false positives.

**Decision**: Implement three-layer detection:
1. Deterministic (regex for explicit keywords)
2. Semantic (embeddings for obfuscated intent)
3. Sarcasm/Hyperbole filter (DistilBERT for teenage vernacular)

**Consequences**:
- ✅ High recall (catches obfuscated language)
- ✅ Low false positives (filters teenage hyperbole)
- ✅ Fast (<50ms) due to lightweight models
- ⚠️ Requires maintaining three separate detection systems
- ⚠️ Need diverse training data for sarcasm filter

**Alternatives Considered**:
- Regex only: Misses obfuscated language
- LLM-based detection: Too slow, not deterministic
- Single semantic model: Too many false positives

---

## ADR-008: PHQ-9/GAD-7 Clinical Markers Without Formal Assessment

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Formal assessments feel clinical and reduce student engagement.

**Decision**: Silently map conversational language to PHQ-9/GAD-7 items without asking direct questions.

**Consequences**:
- ✅ Maintains conversational, non-clinical feel
- ✅ Tracks longitudinal mental health trajectory
- ✅ Provides evidence-based risk scores to counselors
- ⚠️ Lower confidence than formal assessment
- ⚠️ Must be clear this is triage, not diagnosis

**Alternatives Considered**:
- Formal PHQ-9/GAD-7 surveys: Too clinical, reduces engagement
- No clinical markers: Counselors lack evidence-based context
- Custom markers: Reinventing the wheel, not evidence-based

---

## ADR-009: "Peer-Expert" Persona Instead of Clinical Voice

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Clinical/institutional voice creates friction, reduces student disclosure.

**Decision**: Use "Peer-Expert" persona: informal, empathetic, age-appropriate language.

**Consequences**:
- ✅ Lowers barriers to disclosure
- ✅ Builds trust through routine and consistency
- ✅ Feels like supportive mentorship, not medical evaluation
- ⚠️ Must maintain consistency (versioned system prompts)
- ⚠️ Risk of being "too casual" for some situations

**Alternatives Considered**:
- Clinical voice: Too formal, reduces engagement
- Pure peer voice: Lacks authority, may not be taken seriously
- Adaptive voice: Too complex, inconsistency breaks trust

---

## ADR-010: Field-Level Encryption for Clinical Evidence Snippets

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Database breaches happen. Need defense-in-depth for most sensitive data.

**Decision**: Use AWS KMS for field-level encryption of clinical evidence snippets.

**Consequences**:
- ✅ Data unintelligible even if database is breached
- ✅ Separate keys for different sensitivity levels
- ✅ Audit trail of key usage
- ⚠️ Slight performance overhead for encryption/decryption
- ⚠️ Key management complexity

**Alternatives Considered**:
- Database-level encryption only: Not sufficient for highest sensitivity data
- Application-level encryption: More complex, harder to audit
- No additional encryption: Unacceptable risk for mental health data

---

## ADR-011: Redis Cache for "Last 10 Messages" LLM Context

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: LLM needs recent conversation context, but fetching from S3/RDS is too slow.

**Decision**: Cache last 10 messages per session in Redis (ElastiCache).

**Consequences**:
- ✅ Sub-second context retrieval
- ✅ Reduces load on primary database
- ✅ Automatic expiration (TTL) for inactive sessions
- ⚠️ Cache invalidation complexity
- ⚠️ Need to handle cache misses gracefully

**Alternatives Considered**:
- Fetch from RDS every time: Too slow (>100ms)
- Store all messages in Redis: Too expensive at scale
- No context: LLM responses lack continuity

---

## ADR-012: Multi-Provider Fallback with Circuit Breaker

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: LLM providers have outages. System must remain available.

**Decision**: Implement circuit breaker pattern with fallback chain: Bedrock → OpenAI → Pre-written responses.

**Consequences**:
- ✅ System remains available during provider outages
- ✅ Automatic recovery when provider comes back online
- ✅ Graceful degradation (pre-written responses for common scenarios)
- ⚠️ Need to maintain compatibility with multiple providers
- ⚠️ Pre-written responses may feel less natural

**Alternatives Considered**:
- Single provider only: Unacceptable downtime risk
- Manual failover: Too slow, requires human intervention
- Queue requests during outage: Students can't wait

---

## ADR-013: k-Anonymity (k≥5) for All Aggregated Reports

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Small group sizes in reports can lead to re-identification.

**Decision**: Suppress any aggregated report where group size < 5.

**Consequences**:
- ✅ Prevents re-identification from aggregated data
- ✅ Industry standard for privacy-preserving analytics
- ✅ Simple to implement and audit
- ⚠️ Some small schools may have limited analytics
- ⚠️ Need to communicate why some reports are suppressed

**Alternatives Considered**:
- k=3: Too low, still allows re-identification
- Differential privacy: More complex, harder to explain
- No suppression: Unacceptable privacy risk

---

## ADR-014: ECS Fargate Instead of EC2 or Lambda

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Need serverless compute that supports long-running processes (LLM calls can take 1-2s).

**Decision**: Use ECS Fargate for all microservices.

**Consequences**:
- ✅ Serverless (no server management)
- ✅ Auto-scaling based on request volume
- ✅ Supports long-running processes (unlike Lambda's 15min limit)
- ⚠️ More expensive than EC2 for steady-state workloads
- ⚠️ Cold start latency (mitigated by keeping warm instances)

**Alternatives Considered**:
- Lambda: 15min timeout too short for some operations
- EC2: Requires server management, less elastic
- EKS: Overkill for our scale, adds operational complexity

---

## ADR-015: Versioned System Prompts for Persona Consistency

**Date**: 2026-01-18  
**Status**: Accepted  
**Context**: Prompt changes can dramatically alter persona. Need version control and rollback.

**Decision**: Store system prompts in version-controlled YAML files, deploy as immutable artifacts.

**Consequences**:
- ✅ Persona changes are auditable
- ✅ Easy rollback if new prompt causes issues
- ✅ A/B testing different prompts
- ⚠️ Need to maintain backward compatibility
- ⚠️ Prompt versioning adds deployment complexity

**Alternatives Considered**:
- Hard-coded prompts: No version control, hard to change
- Database-stored prompts: No version control, hard to audit
- Dynamic prompts: Too risky, inconsistency breaks trust

---

## Future Decisions to Document

- Multi-language support strategy
- Mobile app architecture (native vs React Native)
- Real-time chat vs async messaging
- Video/voice support for counselor escalation
- Integration with electronic health records (EHR)

---

**Document Maintenance**: Update this log whenever a significant architectural decision is made. Include date, context, decision, consequences, and alternatives considered.
