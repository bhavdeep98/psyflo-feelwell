# High Level Design (HLD)
## PsyFlo (Feelwell) Mental Health AI Triage System

**Document Version**: 1.2  
**Status**: Architecture Specification  
**Last Updated**: 2026-01-20

---

## 1. System Overview

PsyFlo is a mission-critical, AI-driven mental health triage platform engineered specifically for school districts and youth organizations. It serves as a sophisticated digital bridge, addressing the growing disparity between high-volume student emotional needs and the finite availability of professional counseling staff.

By automating the initial engagement and risk assessment phases, PsyFlo ensures that no student's plea for help goes unnoticed while optimizing the counselor's workflow through data-driven prioritization.

### 1.1 Core Objectives

- **Empathetic Engagement**: Establish a "Peer-Expert" persona that effectively builds rapport with teenagers through routine, non-judgmental interaction. This persona is designed to lower the barriers to disclosure, making the system feel less like a clinical evaluation and more like a supportive mentorship.

- **Safety-First Reliability**: Implement a rigorous, multi-layer crisis detection system that operates independently of LLM probabilistic outputs. By separating safety logic from conversational generation, the system ensures that even in cases of LLM "hallucination" or provider downtime, critical risk markers are identified with deterministic precision.

- **Clinical Insight & Trajectory**: Automatically track longitudinal clinical markers (modeled after PHQ-9 and GAD-7) to present counselors with "Evidence-Based" risk trajectories. Instead of static snapshots, counselors receive a dynamic view of a student's mental health journey, supported by direct conversation snippets for context.

- **Enterprise-Grade Scalability**: Leverage a decoupled, event-driven architecture capable of handling district-wide traffic bursts—such as during exam weeks—with sub-second analysis and real-time alert propagation.

---

## 2. Design Principles

1. **Safety-First Parallelism**: Crisis detection is not a post-process; it runs concurrently with LLM response generation. If the safety service detects a high-intent risk marker, it initiates an "emergency stop" on the generated text, replacing it with a localized, deterministic crisis protocol that provides immediate resources to the student.

2. **Persona Integrity**: The system is strictly governed by a "Peer-Expert" voice—defined by informal, stable, and empathetic syntax. This intentional design choice avoids the "Institutional Friction" often associated with school-sanctioned medical tools, fostering a sense of psychological safety that encourages honest student expression.

3. **Explainable Triage**: Transparency is foundational to counselor trust. Every high-risk alert must include "Evidence Snippets"—verbatim, encrypted quotes from the conversation—that allow the counselor to immediately understand the clinical justification for the system's risk score without reading through hours of transcripts.

4. **Zero-Trust Privacy Framework**: To ensure compliance and student safety, PII is hashed at the network edge. Clinical data is further protected via field-level encryption, and data access is strictly governed by a granular Role-Based Access Control (RBAC) model that limits visibility to only the assigned school professionals.

5. **Event-Driven Resilience**: Services communicate via an immutable messaging bus (SNS/SQS). This decoupling ensures that even if the primary chat interface experiences localized issues, crisis alerts and clinical markers are safely queued and delivered to the Crisis Engine and Notification Service, preventing data loss during critical moments.

---

## 3. Component Architecture

### 3.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Student Interface                        │
│                      (Web/Mobile Client)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway (ALB)                           │
│                    TLS 1.3 / Rate Limiting                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Chat Orchestrator Service                     │
│              (Parallel Consensus Coordinator)                    │
└─────┬──────────────┬──────────────┬──────────────┬──────────────┘
      │              │              │              │
      ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Safety  │  │ Observer │  │   LLM    │  │  Crisis  │
│ Service  │  │ Service  │  │ Service  │  │  Engine  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Event Bus (SNS + SQS)                         │
└─────┬──────────────┬──────────────┬──────────────┬──────────────┘
      │              │              │              │
      ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Notification│ │Analytics │  │  Audit   │  │  Data    │
│  Service  │  │ Service  │  │ Service  │  │  Lake    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 3.2 Service Definitions

#### 3.2.1 Safety Service (The Guardrail)

**Purpose**: Multi-layer crisis detection independent of LLM outputs

**Components**:
- **Deterministic Layer**: High-speed, localized Regex matching to catch explicit crisis keywords and known distress patterns. Provides the absolute "floor" for safety, ensuring zero-latency detection of high-risk phrases.

- **Semantic Layer**: Lightweight transformer-based embedding model (all-MiniLM-L6-v2) to identify obfuscated intent, "coded" language, or indirect cries for help (e.g., "I'm checking out early" or "making things easier for everyone").

- **Sarcasm & Hyperbole Filter**: DistilBERT-based classifier fine-tuned on adolescent vernacular. Distinguishes between genuine crisis and common teenage hyperbole (e.g., "this homework is killing me"), drastically reducing false positives.

**SLA**: <50ms analysis latency

#### 3.2.2 Observer Service (The Clinician)

**Purpose**: Track longitudinal clinical markers without formal assessment

**Components**:
- **Marker Extraction**: Maps conversational inputs against standardized clinical items (PHQ-9 for depression, GAD-7 for anxiety). Identifies symptoms like sleep disturbance, worthlessness, or fatigue without formal assessment.

- **Evidence Collection**: Catalogues specific, high-confidence message snippets that correlate with identified markers. These snippets serve as "proof" presented to counselors during triage.

- **Trajectory Engine**: Analyzes current markers against student's historical data to calculate mental health trend. Identifies whether a student is on a "downward spiral" or showing signs of improvement.

**Output**: Clinical markers with confidence scores (0.0-1.0)

#### 3.2.3 LLM Service (The Voice)

**Purpose**: Generate empathetic, persona-consistent responses

**Components**:
- **Persona Orchestration**: Operates "Peer-Expert" persona via strictly versioned system prompt. Prioritizes active listening, validation of emotions, and age-appropriate, non-clinical language.

- **Strategy Injection**: Dynamically adapts conversational strategy based on hidden metadata from Observer Service. If Observer detects high anxiety, LLM may gently pivot to grounding technique or validating follow-up question.

**Providers**: AWS Bedrock (HIPAA-ready), OpenAI (fallback)

**SLA**: <2.0s total response latency (P95)

#### 3.2.4 Crisis Engine (The Responder)

**Purpose**: Orchestrate multi-layer crisis notification and escalation

**Components**:
- **Escalation Orchestration**: Manages sophisticated, multi-layer notification tree. If initial SMS alert not acknowledged within 5 minutes, automatically escalates to phone call, then backup staff/administrators.

- **Idempotency & Deduplication**: Tracks active crisis incidents to prevent duplicate alerts from single prolonged session, while ensuring new, distinct risks are flagged immediately.

**SLA**: <5 minutes alert delivery

---

## 4. The Parallel Decision Pipeline (The "Brain")

Traditional safety systems rely on linear "Safety Filter → LLM" approach, which introduces latency and often breaks the "human" feel. PsyFlo utilizes a **Parallel Consensus Model**.

### 4.1 How It Works

```
Student Message
       │
       ├─────────────┬─────────────┬─────────────┐
       ▼             ▼             ▼             ▼
   Safety       Observer        LLM         History
   Service      Service       Service       Analysis
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                     │
                     ▼
            Chat Orchestrator
         (Consensus Scoring)
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Sc ≥ 0.90    0.65 ≤ Sc < 0.90  Sc < 0.65
   CRISIS        CAUTION           SAFE
       │             │             │
       ▼             ▼             ▼
  Hard Override  Persona Shim   Full Autonomy
```

### 4.2 Scoring Logic

The Chat Orchestrator aggregates signals into a weighted **Consensus Score (Sc)**:

```
Sc = (w_regex · P_regex) + (w_semantic · P_semantic) + (w_history · P_history)
```

**Decision Paths**:

- **Sc ≥ 0.90 (Critical)**: Hard Override. Discard LLM response, deliver deterministic crisis protocol, trigger high-priority SNS/SQS events.

- **0.65 ≤ Sc < 0.90 (Caution)**: Persona Shim. Deliver current response, augment next LLM prompt with "hidden guidance". Flag student in counselor's daily triage dashboard.

- **Sc < 0.65 (Stable)**: Full Autonomy. LLM permitted full conversational autonomy, continues building rapport.

### 4.3 Comparative Architectural Advantage

| Approach | Pro | Con | PsyFlo Advantage |
|----------|-----|-----|------------------|
| Pure LLM Prompting | Easy to implement; fluid conversation | Prone to "hallucinated safety," jailbreaking | Deterministic regex/semantic layer as unbypassable safety floor |
| Fine-Tuned LLM | Highly tailored tone and vocabulary | Requires massive datasets; safety remains probabilistic | LLM is "voice" only, safety delegated to specialized microservices |
| Linear Safety Gate | High safety certainty | Feels robotic; high latency; kills conversation flow | Parallel Consensus analyzes risk while LLM generates, sub-second feedback |

---

## 5. Data Architecture

### 5.1 Storage Strategy (Cost-Optimized)

**Hybrid Storage Model**:

- **RDS (PostgreSQL)**: Structured, high-value metadata
  - User roles, school district configurations
  - Normalized clinical scores
  - ACID compliance for sensitive relational data

- **S3 + Parquet (Data Lake)**: Long-term conversation history
  - Raw messages serialized as JSON, stored in S3
  - Partitioned by date and school ID
  - AWS Athena for SQL queries against Parquet files
  - "Pay-per-query" model, significantly more cost-effective

- **Redis (ElastiCache)**: High-speed in-memory cache
  - Active session states
  - Rate-limit counters
  - "Last 10 messages" for LLM context
  - Sub-second response times

### 5.2 Data Lifecycle

```
Day 0-30:    Redis + RDS (Hot Storage)
Day 30-90:   S3 Standard (Warm Storage)
Day 90-7yr:  S3 Glacier (Cold Storage)
7 years:     Permanent Delete (unless legal hold)
```

### 5.3 Messaging Bus (SNS + SQS)

All internal events published to central **SNS Topic**:

- **Queue Decoupling**: Various SQS queues subscribe (crisis-alert-queue, audit-log-queue, analytics-queue)
- **Resilience & Durability**: If downstream service offline, messages persist in queue
- **Implementation Choice**: SNS/SQS chosen over Kinesis for lower complexity, ease of debugging, superior cost-effectiveness

---

## 6. Security & Privacy

### 6.1 Compliance Framework

- **FERPA/COPPA Compliance**: Strict data lifecycle with S3 lifecycle policies
  - Transition to Glacier after 30 days
  - Permanent purge after 90 days
  - Clinical summaries retained for long-term tracking

- **PII Separation & Anonymization**: LLM Service isolated from student's real identity
  - Only receives temporary SessionUUID
  - Re-mapped to actual student ID only within secure counselor portal

- **Advanced Encryption**: AWS KMS for field-level encryption
  - Clinical evidence snippets encrypted
  - Database/S3 breach renders content unintelligible without clinical-tier keys

### 6.2 Access Control

**RBAC Model**:
- Student: Own conversations only
- Counselor: Assigned school students only
- School Admin: Aggregated reports only
- Platform Admin: System health, no student data
- Auditor: Audit logs, no conversation content

**Zero Trust Architecture**:
- All service-to-service calls require mTLS authentication
- Least privilege: Services only get permissions they need
- Verify every request: "Is this user allowed to view this student's data?"

---

## 7. Operational Excellence

### 7.1 Infrastructure (AWS)

- **Compute Tier**: ECS Fargate for all microservices
  - Serverless container environment
  - Auto-scales based on real-time request volume
  - Performance during peak morning hours in schools

- **Storage Tier**: S3 (Standard) for warm logs, Athena for ad-hoc querying
  - "Pay-per-query" model
  - More cost-effective than managed database clusters

### 7.2 Reliability Metrics (SLAs)

- **Crisis Recall**: ≥99.5% (percentage of actual crisis events successfully identified)
- **Analysis Latency**: <100ms (Safety and Observer services signal time)
- **Total Response Latency**: <2.0s (P95 for student to receive response)
- **Alert Delivery**: <5 minutes (crisis identification to escalated notification)
- **System Uptime**: 99.9% (8.76 hours downtime/year allowed)

### 7.3 Observability

**Three Pillars**:
1. **Logs**: Timestamped records of events (zero PII)
2. **Metrics**: Numerical measurements over time
3. **Traces**: Request journey through multiple services

**Dashboards**:
- Ops: System health, latency, error rates
- Counselors: Student risk, alert status
- Compliance: Audit trails, data access logs

---

## 8. Deployment Strategy

### 8.1 Blue/Green Deployment
- Two identical production environments
- Switch traffic instantly
- Zero-downtime deployments, instant rollback

### 8.2 Canary Deployment
- Gradually roll out new version to small % of users
- Monitor metrics, then expand
- Use for: New LLM models, new crisis keyword patterns

---

## 9. Future Considerations

### 9.1 Geographic Expansion
- USA-only initially
- New jurisdiction requires compliance review
- GDPR compliance for European expansion
- Data residency requirements

### 9.2 Healthcare Partnerships
- HIPAA BAA with AWS Bedrock
- Business Associate Agreement required
- Stricter security requirements than FERPA

### 9.3 Multi-SIS Support
- Adapter pattern for various Student Information Systems
- Support: Clever, ClassLink, Skyward, PowerSchool
- Manual roster management fallback

---

**Document Status**: Approved for Implementation
