---
inclusion: always
---

# PsyFlo Glossary

Quick reference for terminology used across design documents, code, and compliance materials.

## Compliance & Regulatory Terms

### FERPA (Family Educational Rights and Privacy Act)
US federal law protecting student education records privacy.
- Conversations and mental health assessments are considered education records
- Requires 7-year minimum data retention for records
- Students/parents have right to access and request deletion
- Schools must have consent before sharing data with third parties

### COPPA (Children's Online Privacy Protection Act)
US federal law protecting online privacy of children under 13.
- Requires verifiable parental consent before collecting personal information
- Must clearly disclose what data is collected and how it's used
- Parents can review, delete, or refuse further collection

### HIPAA (Health Insurance Portability and Accountability Act)
US federal law protecting health information privacy.
- Not directly applicable (schools aren't covered entities)
- Design is HIPAA-ready for future healthcare partnerships
- AWS Bedrock supports BAA (OpenAI does not)

### SOC 2 Type II
Security audit framework by AICPA. Type II means controls are tested over time (6-12 months).
- Required for enterprise sales
- Five Trust Service Criteria: Security, Availability, Processing Integrity, Confidentiality, Privacy

## Privacy & Security Terms

### PII (Personally Identifiable Information)
Information that can identify a specific individual.
- Examples: student name, email, student ID, phone number, date of birth, IP address
- **Zero PII in application logs** (core tenet)
- Use `hash_pii(student_id)` for logging

### PHI (Protected Health Information)
HIPAA term for health information linked to an individual.
- We treat mental health markers like PHI (encrypt, access-control, audit)

### Encryption At Rest
Data stored on disk/database is encrypted.
- AES-256 encryption for RDS, DocumentDB, S3
- Field-level encryption for highly sensitive PII
- Keys managed by AWS KMS

### Encryption In Transit
Data moving between systems is encrypted.
- TLS 1.3 minimum for all HTTPS connections
- mTLS for service-to-service communication

### RBAC (Role-Based Access Control)
Access permissions based on user roles:
- **Student**: Can view own conversations, own risk scores
- **Counselor**: Can view students at their assigned school only
- **School Admin**: Can view aggregated reports for their school
- **Platform Admin**: Can view system health, not student data
- **Auditor**: Can view audit logs, not conversation content

### K-Anonymity
Privacy technique ensuring each person is indistinguishable from at least k-1 others.
- **PsyFlo Rule: k ≥ 5** (minimum group size of 5)
- Analytics service suppresses any report where group size < 5

### Hash Function (SHA-256)
One-way cryptographic function that converts data to fixed-length string.
```python
# Never log this:
logger.info(f"Student {student_id} logged in")

# Always log this:
logger.info(f"Student {hash_pii(student_id)} logged in")
```

## Clinical & Mental Health Terms

### Triage
Process of determining priority based on severity.
- **PsyFlo Context**: AI assesses conversation → Crisis (notify immediately), Caution (weekly check-in), Safe (no action)
- **Critical Distinction**: Triage is NOT diagnosis

### PHQ-9 (Patient Health Questionnaire-9)
9-question screening tool for depression, scored 0-27.
- 0-4: Minimal/none
- 5-9: Mild
- 10-14: Moderate
- 15-19: Moderately severe
- 20-27: Severe

### GAD-7 (Generalized Anxiety Disorder-7)
7-question screening tool for anxiety, scored 0-21.
- 0-4: Minimal anxiety
- 5-9: Mild
- 10-14: Moderate
- 15-21: Severe

### C-SSRS (Columbia-Suicide Severity Rating Scale)
Evidence-based tool for assessing suicide risk.
- Key questions: Wish to be dead, suicidal thoughts, thoughts with method, intent, intent with plan, behavior
- Any item 1-5 detected → CRISIS level

### Clinical Marker
Observable indicator in conversation that maps to clinical screening criteria.
- Example: "I can't focus in class anymore" → PHQ-9 Item 7 (concentration problems)
- Includes confidence score: 0.0-1.0

### Suicidal Ideation
Thoughts about or planning suicide, without actual attempt.
- **Passive**: "I wish I wasn't here"
- **Active**: "I'm going to kill myself"
- Both trigger CRISIS level

### Self-Harm
Deliberate injury to oneself (cutting, burning, hitting).
- **NSSI**: Non-Suicidal Self-Injury (coping mechanism)
- Any mention → CAUTION level minimum
- If combined with suicidal ideation → CRISIS

## System Architecture Terms

### Microservices
Application built as collection of small, independent services.
- PsyFlo Services: Chat, Safety, Observer, Crisis Engine, LLM, Auth, Notification, Analytics, Audit
- Crisis detection works even if Chat Service crashes

### Event-Driven Architecture
Services communicate by publishing/subscribing to events.
- Decoupled: Services don't need to know about each other
- Resilient: If service is down, event stays in queue and retries

### Idempotency
Operation can be performed multiple times with same result.
- Prevents duplicate crisis alerts when events are retried

### Circuit Breaker
Software pattern that prevents cascading failures.
- If dependency fails 5 times, circuit "opens"
- Return fallback response immediately
- After 30 seconds, try again

### Blue/Green Deployment
Run two identical production environments. Switch traffic instantly.
- Zero-downtime deployments, instant rollback

### Canary Deployment
Gradually roll out new version to small % of users, monitor, then expand.

## Data & Storage Terms

### Hot Storage
Frequently accessed data on fast storage.
- Redis cache (0-30 days), RDS (current sessions)

### Warm Storage
Less frequently accessed, slower storage.
- S3 Standard (30-90 days)

### Cold Storage
Rarely accessed, cheapest storage.
- S3 Glacier (90 days - 7 years)

### WORM Storage (Write Once, Read Many)
Storage that cannot be modified or deleted after writing.
- Audit logs stored in WORM mode to prevent tampering

## Testing & Quality Terms

### Golden Test Set
High-quality, curated test cases representing real-world scenarios.
- MentalChat16K: 16,000 mental health conversations
- Custom crisis test set: 193 cases (need to expand to 10,000+)

### False Negative
System says "safe" when actually crisis (missed crisis).
- **Worst case**: Student in crisis, system doesn't flag

### False Positive
System says "crisis" when actually safe (false alarm).
- Impact: Alert fatigue → counselors ignore real alerts

### Recall (Sensitivity)
% of actual crises that system detected.
- **PsyFlo Target: >99% recall**

### Precision
% of crisis alerts that were actual crises.
- Trade-off: Can get 100% recall by flagging everyone, but precision would be terrible

## SLA Targets

- **Uptime**: 99.9% (8.76 hours downtime/year allowed)
- **Crisis Detection Latency**: <50ms
- **Crisis Alert Delivery**: <5 minutes
- **API Response Time**: <2s p95
- **Crisis Recall**: ≥99.5%
- **Analysis Latency**: <100ms
