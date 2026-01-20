---
inclusion: always
---

# Failure Modes & Mitigation Strategies

This document outlines potential failure modes and their mitigation strategies for the PsyFlo system.

## Trust & Transparency Failures

### Students Don't Trust the System
**Mitigation**:
- Transparent data sharing policy (who sees what, when, why)
- "Talk to real counselor" button always visible
- Consistent persona (same voice, same routine)
- Privacy dashboard showing what data is stored

### Over-Promising Capabilities
**Mitigation**:
- Clear messaging: "I'm here to listen and connect you with help"
- Never claim to diagnose or treat
- Set expectations: "I'll flag concerns for your counselor"

### Unclear Data Boundaries
**Mitigation**:
- Explicit consent flow at first use
- Visual indicators when counselor can see conversation
- Student can request conversation deletion (with limitations explained)

## Operational Failures

### LLM Provider Downtime
**Mitigation**:
- Multi-provider fallback (AWS Bedrock → OpenAI)
- Circuit breaker pattern (fail fast, return cached response)
- Pre-written fallback responses for common scenarios
- Crisis detection continues independently

### Database Outage
**Mitigation**:
- Redis cache serves recent conversations
- Read replicas for RDS
- Graceful degradation: Show staleness warning
- Crisis alerts still work (event-driven, queued)

### Notification Service Failure
**Mitigation**:
- SQS queue persistence (alerts not lost)
- Display crisis resources directly to student
- Multiple notification channels (SMS, email, phone, in-app)
- Escalation to backup counselors

### SIS Integration Failure
**Mitigation**:
- System continues with stale roster data
- Banner: "Student roster last updated: 7 days ago"
- Manual roster upload as backup
- Alert if sync fails 2 consecutive days

## Data & Privacy Failures

### Data Breach
**Mitigation**:
- Field-level encryption (AWS KMS)
- PII hashed at network edge
- Zero PII in application logs
- WORM storage for audit logs (tamper-proof)
- Breach notification within 24 hours

### Accidental PII Exposure in Logs
**Mitigation**:
- Mandatory `hash_pii()` function for all identifiers
- Automated log scanning for PII patterns
- Pre-commit hooks to catch PII in code
- Regular security audits

### Unauthorized Data Access
**Mitigation**:
- Strict RBAC (counselors only see their school)
- All data access logged in immutable audit trail
- Anomaly detection (counselor accessing 100+ records)
- Quarterly access reviews

### Data Retention Violations
**Mitigation**:
- Automated S3 lifecycle policies
- Tombstone records for deletion proof
- Legal hold mechanism for active investigations
- Compliance dashboard showing data age

## Bias & Accessibility Failures

### Cultural Bias in Crisis Detection
**Mitigation**:
- Diverse training data (multiple demographics)
- Regular bias audits on crisis detection
- Cultural consultants review keyword patterns
- A/B testing across demographic groups

### Language Barriers
**Mitigation**:
- Multi-language support (Spanish priority)
- Translation with cultural context preservation
- Bilingual counselor routing

### Accessibility Issues
**Mitigation**:
- WCAG 2.1 AA compliance
- Screen reader support
- Keyboard navigation
- High contrast mode

## Ethical & Legal Failures

### Mandated Reporting Conflicts
**Mitigation**:
- Clear policy: System assists, doesn't replace mandated reporting
- Counselor training on legal obligations
- Crisis alerts include mandated reporting guidance
- Legal review of all crisis protocols

### Parental Rights vs Student Privacy
**Mitigation**:
- Age-appropriate consent flows
- COPPA compliance for under-13
- Clear policy on parental access rights
- Legal counsel on state-specific requirements

### Liability for Missed Crisis
**Mitigation**:
- Disclaimer: "AI is triage, not diagnosis"
- Human-in-the-loop for all high-stakes decisions
- 100% test coverage for safety-critical code
- Insurance coverage for professional liability

## Human Factors Failures

### Alert Fatigue
**Mitigation**:
- Sarcasm/hyperbole filter reduces false positives
- Tiered alerts (Crisis vs Caution)
- Weekly digest for non-urgent flags
- Counselor feedback loop to tune thresholds

### Counselor Skill Gaps
**Mitigation**:
- Onboarding training program
- Evidence snippets provide context
- Crisis protocol checklists
- Escalation to senior counselors

### Student Gaming the System
**Mitigation**:
- Adversarial testing (coded language, leetspeak)
- Semantic layer catches obfuscated intent
- Counselor can manually flag suspicious patterns
- Regular pattern updates

## Technical Debt Failures

### Unmaintainable Codebase
**Mitigation**:
- 60-second litmus test for all code
- Explicit over clever (core tenet)
- Comprehensive documentation
- Regular refactoring sprints

### Dependency Vulnerabilities
**Mitigation**:
- Automated dependency scanning
- Quarterly security updates
- Version pinning with upgrade plan
- Vendor security assessments

## Integration Failures

### Multi-SIS Support Complexity
**Mitigation**:
- Adapter pattern (don't hard-code for one SIS)
- Support: Clever, ClassLink, Skyward, PowerSchool
- Manual roster management fallback
- 90-day notice before deprecating integration version

### Version Compatibility
**Mitigation**:
- Use stable API versions (don't auto-upgrade)
- Maintain backward compatibility for 2 versions
- Test integration monthly (automated test student)

---

## Mitigation Strategy Summary

| Category | Technical Controls | Monitoring | Process Controls |
|----------|-------------------|------------|------------------|
| Trust & Transparency | 3 | 3 | 1 |
| Operational | 4 | 1 | 3 |
| Data & Privacy | 5 | 2 | 2 |
| Bias & Accessibility | 3 | 3 | 2 |
| Ethical & Legal | 2 | 1 | 5 |
| Human Factors | 1 | 3 | 3 |
| Technical Debt | 0 | 1 | 2 |
| Integration | 2 | 1 | 0 |
| **TOTAL** | **20** | **15** | **18** |

---

## Success Metrics

- **SIS Sync Success Rate**: 99%+
- **Data Staleness**: <1 day
- **Crisis Recall**: ≥99.5%
- **False Positive Rate**: <10%
- **Alert Acknowledgment Time**: <5 minutes
- **System Uptime**: 99.9%
