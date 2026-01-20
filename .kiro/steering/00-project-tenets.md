---
inclusion: always
---

# PsyFlo (Feelwell) Project Tenets

These tenets are the foundational principles that guide all technical and product decisions for the PsyFlo mental health AI triage system.

## 1. Safety First, Always
- Deterministic guardrails bypass LLM for crisis detection
- Hard-coded crisis protocols that can't be overridden
- 100% test coverage required for all safety-critical code
- Crisis path must be linear, traceable, and obvious

## 2. Compliance is Non-Negotiable
- FERPA, COPPA, SOC 2 Type II (HIPAA-ready)
- Zero PII in application logs - use `hash_pii()` for all identifiers
- Immutable audit trail for all sensitive operations
- k-anonymity (k≥5) for all aggregated reports

## 3. Explicit Over Clever
- Code must be traceable and self-documenting
- Avoid clever abstractions - when a crisis is missed, you need to trace exactly what happened
- New engineers should understand any file in 60 seconds
- The code IS the documentation for incident review

## 4. Fail Loud, Fail Early
- Never silently swallow errors
- No bare `except:` clauses
- Raise specific, documented exceptions with context
- Log before raising - every failure must be traceable

## 5. Make Illegal States Unrepresentable
- Use Enums for fixed values (risk levels, statuses)
- Type hints everywhere
- Let the type system catch bugs at compile time

## 6. Event-Driven Crisis Response
- Crisis detection must be decoupled from chat service
- If chat crashes, crisis protocol still executes
- "Fire alarm" architecture - highly available, independent

## 7. Immutability by Default
- Immutable data structures for conversation sessions
- Prevents accidental state corruption
- Makes debugging and auditing easier

## 8. Engagement Before Intervention
- A system students won't use can't help anyone
- User experience is a safety feature, not a luxury
- Build trust through routine and consistency before expecting disclosure
- Safety AND empathy, not safety OR empathy

## 9. Visibility and Explainability at Every Layer
- Every decision must be traceable: crisis keyword matched, clinical marker detected, alert sent
- Applies to: developers debugging, counselors interpreting flags, auditors reviewing compliance
- Configuration over code: crisis patterns in YAML/JSON, not buried in Python
- Show your work: include reasoning with every risk assessment

## 10. Observable Systems Are Reliable Systems
- Instrument everything: latency, error rates, safety recall, user engagement
- Alerts must be actionable - no "nice to know" alerts
- Dashboards for multiple audiences: ops (system health), counselors (student risk), compliance (audit trails)
- Silent failures are unacceptable - if we can't measure it, we can't trust it

## 11. Graceful Degradation Over Hard Failures
- LLM down? Return safe fallback responses, don't crash
- Database slow? Serve cached data with staleness warning
- Notification service down? Display crisis resources directly to student
- Every component must answer: "What happens when I fail?"

## 12. Human-in-the-Loop by Design
- AI is triage, not diagnosis or treatment
- All high-stakes decisions require human confirmation (counselor must ACK crisis alerts)
- Students can always escalate to human ("Talk to real counselor" button omnipresent)
- System should make counselors better, not replace them

## 13. Trust Is Earned, Not Assumed
- Students don't owe us their stories - we must earn the right to hear them
- Transparency about data sharing (who sees what, when, why)
- Consistency builds trust: same voice, same routine, same reliability
- One breach of trust can destroy months of rapport

## 14. Geographic and Regulatory Boundaries Are Explicit
- USA-only initially - documented in DECISION_LOG.md
- New jurisdiction = compliance review before launch
- Data residency requirements enforced at signup (block non-supported countries)
- Never expand scope without legal, security, and infrastructure review

## 15. Performance Is a Safety Feature
- Crisis detection <50ms (student in crisis can't wait)
- Alert delivery <5min (counselor must be notified immediately)
- System uptime 99.9% (students expect us to be available)
- Slow = broken when someone needs help

## The Litmus Test

Every piece of code must pass this 60-second test for a new engineer:

1. What does this file do?
2. What happens if this fails?
3. Where would I add a log statement to debug this?

**If no → Refactor immediately.**

The stakes are high: mental health + minors = zero tolerance for bugs. These tenets ensure the platform is safe, compliant, and maintainable under pressure.
