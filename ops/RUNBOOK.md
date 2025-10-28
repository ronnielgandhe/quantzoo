# Operations Runbook

## Table of Contents
1. [Emergency Procedures](#emergency-procedures)
2. [Kill Switch Operations](#kill-switch-operations)
3. [Monitoring and Alerts](#monitoring-and-alerts)
4. [Common Failure Modes](#common-failure-modes)
5. [Credential Management](#credential-management)
6. [System Tests](#system-tests)
7. [Deployment Checklist](#deployment-checklist)

## Emergency Procedures

### 🚨 EMERGENCY: Runaway Trading

If the system is placing unexpected orders or experiencing excessive losses:

**IMMEDIATE ACTIONS:**

1. **Activate Kill Switch** (< 30 seconds):
   ```bash
   curl -X POST http://localhost:8888/kill-switch/activate \
     -H "Authorization: Bearer $SAFETY_API_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"reason": "Emergency stop - runaway trading", "close_positions": true, "operator": "YOUR_NAME"}'
   ```

2. **Verify Kill Switch Active**:
   ```bash
   curl http://localhost:8888/status
   ```

3. **Check Positions**:
   ```bash
   # Via API
   curl http://localhost:8001/positions/current
   
   # Or check broker directly
   ```

4. **Close All Positions Manually** (if auto-close fails):
   - Log into broker portal
   - Flatten all positions
   - Cancel all pending orders

5. **Document Incident**:
   - Screenshot positions and P&L
   - Save logs
   - Note timestamp and circumstances
   - File incident report

### System Unresponsive

If the API or services are not responding:

1. **Check Process Status**:
   ```bash
   # Check if processes are running
   ps aux | grep uvicorn
   ps aux | grep python
   
   # Check ports
   lsof -i :8001
   lsof -i :8888
   ```

2. **Check Logs**:
   ```bash
   tail -100 api.log
   tail -100 safety_api.log
   tail -100 /var/log/quantzoo/*.log
   ```

3. **Restart Services**:
   ```bash
   # Kill existing
   pkill -f uvicorn
   
   # Restart safely
   python -m uvicorn quantzoo.rt.api:app --host 0.0.0.0 --port 8001 &
   python services/safety_api.py &
   ```

### Data Quality Issues

If receiving bad prices or stale data:

1. **Stop Trading Immediately** (use kill switch)

2. **Verify Data Source**:
   ```bash
   # Check last update time
   curl http://localhost:8001/symbols/list | jq '.[] | {symbol, last_update}'
   ```

3. **Compare with External Source**:
   - Check against broker quotes
   - Verify against exchange data
   - Look for gaps or suspicious prices

4. **Restart Data Feed**:
   ```bash
   # Restart provider
   # Configuration-specific - check provider docs
   ```

## Kill Switch Operations

### Activating Kill Switch

**When to Use:**
- Detecting unexpected trading behavior
- System malfunction
- Excessive losses
- Regulatory requirement
- Market conditions requiring halt

**How to Activate:**

Via API:
```bash
curl -X POST http://localhost:8888/kill-switch/activate \
  -H "Authorization: Bearer $SAFETY_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "Describe why kill switch activated",
    "close_positions": true,
    "operator": "Your Name"
  }'
```

Via Python:
```python
import requests

response = requests.post(
    'http://localhost:8888/kill-switch/activate',
    headers={'Authorization': f'Bearer {token}'},
    json={
        'reason': 'Emergency stop',
        'close_positions': True,
        'operator': 'Admin'
    }
)
print(response.json())
```

### Deactivating Kill Switch

**Before Deactivating:**
- [ ] Root cause identified and resolved
- [ ] System health verified
- [ ] All data feeds operational
- [ ] Monitoring confirmed working
- [ ] Authorization obtained
- [ ] Risk limits verified

**Deactivation Command:**
```bash
curl -X POST http://localhost:8888/kill-switch/deactivate \
  -H "Authorization: Bearer $SAFETY_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"operator": "Your Name"}'
```

### Verifying Kill Switch Status

```bash
curl http://localhost:8888/status \
  -H "Authorization: Bearer $SAFETY_API_TOKEN"
```

## Monitoring and Alerts

### Key Metrics to Monitor

1. **System Health**:
   - API response time (< 100ms normal)
   - Memory usage (< 80%)
   - CPU usage (< 70%)
   - Disk space (> 20% free)

2. **Trading Metrics**:
   - Position count
   - Total exposure
   - Unrealized P&L
   - Daily drawdown
   - Order fill rate
   - Rejected orders count

3. **Data Quality**:
   - Last price update time
   - Missing bars count
   - Outlier prices
   - Data staleness

### Setting Up Alerts

**Prometheus Alerts** (add to `ops/prometheus_alerts.yml`):

```yaml
groups:
  - name: quantzoo_alerts
    rules:
      - alert: HighDrawdown
        expr: unrealized_pnl_pct < -10
        for: 1m
        annotations:
          summary: "Drawdown exceeds 10%"
      
      - alert: SlowAPI
        expr: api_latency_seconds > 1
        for: 5m
        annotations:
          summary: "API latency exceeds 1s"
```

### Grafana Dashboards

Access: http://localhost:3000 (default: admin/admin)

**Key Dashboards:**
1. System Overview
2. Trading Performance
3. Risk Metrics
4. Data Quality

## Common Failure Modes

### 1. API Not Responding

**Symptoms**: Cannot connect to localhost:8001

**Diagnosis**:
```bash
# Check if running
lsof -i :8001

# Check logs
tail -f api.log
```

**Resolution**:
```bash
# Restart API
pkill -f "uvicorn.*quantzoo.rt.api"
python -m uvicorn quantzoo.rt.api:app --host 0.0.0.0 --port 8001
```

### 2. Orders Not Filling

**Symptoms**: Orders stuck in PENDING status

**Diagnosis**:
- Check kill switch status
- Verify broker connectivity
- Check order parameters
- Review broker logs

**Resolution**:
- Deactivate kill switch if appropriate
- Restart broker connector
- Verify credentials
- Check market hours

### 3. Data Feed Stopped

**Symptoms**: Stale prices, no updates

**Diagnosis**:
```bash
# Check last update
curl http://localhost:8001/state | jq '.last_update'

# Check provider status
# Provider-specific
```

**Resolution**:
- Restart data provider
- Verify API key
- Check rate limits
- Switch to backup provider

### 4. Memory Leak

**Symptoms**: Increasing memory usage, eventual crash

**Diagnosis**:
```bash
# Monitor memory
top -p $(pgrep -f uvicorn)

# Check object counts (Python)
python -c "import gc; print(len(gc.get_objects()))"
```

**Resolution**:
- Restart services
- Review code for unclosed connections
- Clear caches
- Upgrade to latest version

## Credential Management

### Rotating API Keys

**Frequency**: Every 90 days or immediately if compromised

**Procedure**:

1. **Generate New Keys**:
   - Log into provider portal
   - Create new API key
   - Save securely (password manager)

2. **Update Environment**:
   ```bash
   # Edit .env
   nano .env
   
   # Or use secure secret management
   export ALPACA_API_KEY="new_key_here"
   export ALPACA_API_SECRET="new_secret_here"
   ```

3. **Test New Keys**:
   ```bash
   # Dry run test
   python -c "from connectors.brokers import AlpacaBroker; b = AlpacaBroker({'dry_run': True})"
   ```

4. **Deploy**:
   ```bash
   # Restart services with new credentials
   systemctl restart quantzoo-api
   ```

5. **Revoke Old Keys**:
   - Wait 24 hours
   - Verify no errors
   - Revoke old keys in portal

### Secret Storage

**Never**:
- ❌ Commit secrets to git
- ❌ Store in plaintext files (except .env which is gitignored)
- ❌ Share via email or chat
- ❌ Hardcode in source

**Always**:
- ✅ Use environment variables
- ✅ Use secret management service (AWS Secrets Manager, HashiCorp Vault)
- ✅ Encrypt at rest
- ✅ Rotate regularly
- ✅ Audit access

## System Tests

### Pre-Deployment Health Check

```bash
# Run full test suite
pytest -v

# Run integration tests
pytest tests/integration/ -v

# Test safety systems
python tests/test_safety.py

# Regression test
python tests/validate_regression.py
```

### Live System Validation

```bash
# Health check
curl http://localhost:8001/healthz

# Safety check
curl http://localhost:8888/test/safety-check \
  -H "Authorization: Bearer $SAFETY_API_TOKEN"

# Data freshness
curl http://localhost:8001/state | jq '.last_update'

# Position check
curl http://localhost:8001/positions/current
```

### Manual Smoke Test

1. Place small test order in paper account
2. Verify order appears in positions
3. Close position
4. Check P&L updated
5. Activate/deactivate kill switch
6. Verify orders blocked when kill switch active

## Deployment Checklist

### Before Going Live

- [ ] All tests passing
- [ ] Security scan clean (bandit, safety)
- [ ] Credentials rotated and secured
- [ ] Monitoring configured
- [ ] Alerts tested
- [ ] Kill switch tested
- [ ] Backup plan documented
- [ ] Team trained on runbook
- [ ] Compliance approval obtained
- [ ] Risk limits configured
- [ ] Emergency contacts updated
- [ ] Incident response plan ready

### Production Deployment

1. **Pre-Flight**:
   ```bash
   # Set environment
   export QUANTZOO_ENV=production
   
   # Verify configuration
   python scripts/verify_config.py
   ```

2. **Deploy**:
   ```bash
   # Build Docker image
   docker build -f docker/Dockerfile.inference -t quantzoo:prod .
   
   # Run
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Validate**:
   - Run health checks
   - Verify metrics collection
   - Test kill switch
   - Monitor for 1 hour

4. **Monitor**:
   - Watch dashboards
   - Check logs
   - Verify trades
   - Track P&L

### Rollback Procedure

If issues detected:

```bash
# Activate kill switch
curl -X POST http://localhost:8888/kill-switch/activate ...

# Stop services
docker-compose down

# Rollback to previous version
docker-compose -f docker-compose.backup.yml up -d

# Verify
curl http://localhost:8001/healthz

# Document issue
```

## Contact Information

### On-Call Rotation

| Time | Primary | Backup |
|------|---------|--------|
| Business Hours | [Name] | [Name] |
| After Hours | [Name] | [Name] |
| Weekends | [Name] | [Name] |

### Escalation

1. **Level 1**: On-call engineer
2. **Level 2**: Tech lead
3. **Level 3**: CTO
4. **Level 4**: CEO (critical only)

### External Contacts

- **Broker Support**: [Phone/Email]
- **Data Provider**: [Phone/Email]
- **Cloud Provider**: [Phone/Email]
- **Legal/Compliance**: [Phone/Email]

## Appendix

### Log Locations

- API logs: `/var/log/quantzoo/api.log`
- Safety API logs: `/var/log/quantzoo/safety_api.log`
- System logs: `/var/log/syslog`
- Broker logs: `/var/log/quantzoo/broker.log`

### Useful Commands

```bash
# Tail all logs
tail -f /var/log/quantzoo/*.log

# Search for errors
grep -i error /var/log/quantzoo/*.log

# Check system resources
htop

# Network connections
netstat -tulpn | grep python

# Disk usage
df -h

# Check environment
env | grep QUANTZOO
```

---

**Last Updated**: 2025-10-28  
**Version**: 1.0  
**Owner**: QuantZoo Team
