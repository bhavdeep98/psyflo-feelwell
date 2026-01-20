# PsyFlo Quick Reference Guide

**For developers who need answers fast.**

---

## 🚨 Emergency: Crisis Detection Not Working

1. Check Safety Service health: `curl https://api.psyflo.com/health/safety`
2. Review recent crisis patterns: `kubectl logs -l service=safety --tail=100`
3. Verify regex patterns loaded: Check `config/crisis_patterns.yaml`
4. Test with known crisis phrase: `pytest tests/safety/test_crisis_detection.py::test_explicit_suicidal_ideation`
5. If all else fails: Circuit breaker may have opened - check CloudWatch metrics

**Critical**: Never deploy crisis detection changes without 100% test coverage.

---

## 📝 Common Tasks

### Add New Crisis Keyword Pattern

1. Edit `config/crisis_patterns.yaml`:
```yaml
crisis_keywords:
  new_pattern_name:
    patterns:
      - "pattern 1"
      - "pattern 2"
    confidence: 0.90
```

2. Add test case:
```python
def test_new_pattern_detection():
    message = "pattern 1"
    result = safety_service.analyze(message)
    assert result.risk_level == RiskLevel.CRISIS
```

3. Run tests: `pytest tests/safety/`
4. Deploy via canary: 5% → 25% → 100%

### Log Without Exposing PII

```python
from utils.privacy import hash_pii

# ❌ NEVER
logger.info(f"Student {student_id} sent message")

# ✅ ALWAYS
logger.info(f"Student {hash_pii(student_id)} sent message")
```

### Query Conversation History

```python
# Recent (hot cache)
messages = await redis_client.get(f"session:{session_id}:messages")

# Older (warm storage)
query = f"""
SELECT * FROM conversations 
WHERE session_id = '{session_id}' 
AND timestamp > NOW() - INTERVAL '30 days'
"""
messages = await db.fetch(query)

# Archive (cold storage)
query = f"""
SELECT * FROM s3_conversations 
WHERE session_id = '{session_id}'
"""
messages = await athena.query(query)
```

### Check Student Risk Level

```python
from services.observer import ObserverService

observer = ObserverService()
assessment = await observer.get_risk_assessment(student_id)

print(f"Risk Level: {assessment.risk_level}")
print(f"PHQ-9 Score: {assessment.phq9_score}")
print(f"GAD-7 Score: {assessment.gad7_score}")
print(f"Trajectory: {assessment.trajectory}")  # improving/stable/declining
```

### Send Test Crisis Alert

```python
from services.crisis_engine import CrisisEngine

engine = CrisisEngine()
await engine.trigger_alert(
    session_id="test-session-123",
    risk_level=RiskLevel.CRISIS,
    evidence=["I want to end my life"],
    confidence=0.95
)
```

---

## 🔍 Debugging

### Find Why Student Was Flagged

```python
# Get risk assessment with evidence
assessment = await observer.get_risk_assessment(student_id)

# Evidence snippets show exact phrases that triggered flag
for snippet in assessment.evidence_snippets:
    print(f"Marker: {snippet.clinical_marker}")
    print(f"Quote: {snippet.message_text}")
    print(f"Confidence: {snippet.confidence}")
```

### Trace Request Through Services

```python
# All services log with trace_id
import structlog

logger = structlog.get_logger()
trace_id = str(uuid.uuid4())

logger.info("request_received", trace_id=trace_id, session_id=hash_pii(session_id))

# Search logs: grep "trace_id=abc-123" logs/*.log
```

### Check Why Alert Wasn't Sent

1. Verify crisis detected: `SELECT * FROM crisis_events WHERE session_id = '...'`
2. Check SQS queue: `aws sqs get-queue-attributes --queue-url ...`
3. Review notification logs: `kubectl logs -l service=notification`
4. Verify counselor contact info: `SELECT * FROM counselors WHERE school_id = '...'`

---

## 📊 Monitoring

### Key Dashboards

- **Ops Dashboard**: https://grafana.psyflo.com/ops
  - System health, latency, error rates
  - Alert if: P95 latency >2s, error rate >1%, uptime <99.9%

- **Safety Dashboard**: https://grafana.psyflo.com/safety
  - Crisis detection rate, false positive rate, recall
  - Alert if: Recall <99.5%, false positive rate >10%

- **Counselor Dashboard**: https://app.psyflo.com/counselor
  - Student risk levels, pending alerts, evidence snippets

### Critical Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| Crisis Recall Drop | <99.5% | Page on-call engineer immediately |
| API Latency Spike | P95 >5s | Check LLM provider status |
| Alert Delivery Delay | >5 min | Check SQS queue depth |
| Database Connection Pool | >80% | Scale up RDS |

---

## 🧪 Testing

### Run Safety-Critical Tests

```bash
# All safety tests (must be 100% passing)
pytest tests/safety/ --cov=src/services/safety --cov-report=term-missing

# Specific crisis detection
pytest tests/safety/test_crisis_detection.py -v

# Adversarial tests (coded language, leetspeak)
pytest tests/safety/test_adversarial.py -v
```

### Run Integration Tests

```bash
# End-to-end crisis flow
pytest tests/integration/test_crisis_flow.py -v

# Full conversation flow
pytest tests/integration/test_conversation_flow.py -v
```

### Test Against Golden Set

```bash
# MentalChat16K regression test
pytest tests/golden/test_mentalchat16k.py --slow

# Custom crisis test set
pytest tests/golden/test_crisis_set.py --slow
```

---

## 🚀 Deployment

### Deploy to Staging

```bash
# Build and push Docker images
./scripts/build.sh

# Deploy via CDK
cd infrastructure
cdk deploy PsyFloStack-Staging

# Run smoke tests
./scripts/smoke_test.sh staging
```

### Deploy to Production (Blue/Green)

```bash
# Deploy to green environment
cdk deploy PsyFloStack-Production-Green

# Run smoke tests
./scripts/smoke_test.sh production-green

# Switch traffic (instant)
./scripts/switch_traffic.sh green

# Monitor for 1 hour, then tear down blue
./scripts/cleanup_blue.sh
```

### Canary Deployment (New LLM Model)

```bash
# Deploy canary (5% traffic)
./scripts/deploy_canary.sh --model=claude-3-opus --traffic=5

# Monitor metrics for 1 hour
./scripts/monitor_canary.sh

# Increase traffic gradually
./scripts/deploy_canary.sh --traffic=25
./scripts/deploy_canary.sh --traffic=100

# Promote to production
./scripts/promote_canary.sh
```

---

## 🔐 Security

### Rotate Encryption Keys

```bash
# Generate new KMS key
aws kms create-key --description "PsyFlo Clinical Data 2026-Q2"

# Update key alias
aws kms update-alias --alias-name alias/psyflo-clinical --target-key-id <new-key-id>

# Re-encrypt data (background job)
./scripts/reencrypt_data.sh --old-key=<old-key-id> --new-key=<new-key-id>
```

### Audit Data Access

```bash
# Query audit logs (WORM storage)
aws athena start-query-execution \
  --query-string "SELECT * FROM audit_logs WHERE user_id = 'counselor-123' AND timestamp > NOW() - INTERVAL '7' DAY" \
  --result-configuration OutputLocation=s3://psyflo-audit-results/
```

### Check for PII in Logs

```bash
# Automated scan (runs nightly)
./scripts/scan_logs_for_pii.sh

# Manual check
grep -E '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' logs/*.log
```

---

## 📞 Escalation

### On-Call Rotation

- **Primary**: Check PagerDuty schedule
- **Secondary**: Check PagerDuty schedule
- **Escalation**: CTO (for crisis detection failures only)

### Incident Response

1. **Acknowledge**: Respond to PagerDuty alert within 5 minutes
2. **Assess**: Check dashboards, determine severity
3. **Mitigate**: Apply immediate fix or rollback
4. **Communicate**: Update status page, notify stakeholders
5. **Resolve**: Verify fix, close incident
6. **Post-Mortem**: Document root cause, action items

### Severity Levels

- **P0 (Critical)**: Crisis detection down, data breach
  - Response: Immediate, page CTO
  - SLA: Resolve within 1 hour

- **P1 (High)**: API down, alert delivery delayed >5min
  - Response: Within 15 minutes
  - SLA: Resolve within 4 hours

- **P2 (Medium)**: Elevated error rate, slow response times
  - Response: Within 1 hour
  - SLA: Resolve within 24 hours

- **P3 (Low)**: Non-critical bugs, feature requests
  - Response: Next business day
  - SLA: Resolve within 1 week

---

## 🔗 Useful Links

- **Production**: https://app.psyflo.com
- **Staging**: https://staging.psyflo.com
- **Grafana**: https://grafana.psyflo.com
- **PagerDuty**: https://psyflo.pagerduty.com
- **AWS Console**: https://console.aws.amazon.com
- **GitHub**: https://github.com/psyflo/psyflo
- **Confluence**: https://psyflo.atlassian.net

---

## 💡 Pro Tips

1. **Always use `hash_pii()`** - Set up pre-commit hook to catch violations
2. **Test crisis detection changes in staging first** - Never YOLO to production
3. **Monitor canary deployments closely** - Rollback at first sign of trouble
4. **Document architecture decisions** - Update DECISION_LOG.md
5. **Keep system prompts versioned** - Easy rollback if persona changes cause issues
6. **Use structured logging** - Makes debugging 10x easier
7. **Set up alerts before you need them** - Don't wait for an incident
8. **Review audit logs regularly** - Catch unauthorized access early

---

**Remember**: The stakes are high. Mental health + minors = zero tolerance for bugs.
