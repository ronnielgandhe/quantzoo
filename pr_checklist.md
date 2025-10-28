# Pull Request Checklist

## PR: Add Production ML, HF Integration, Live Connectors, Monitoring, and Ops

**Branch**: `feature/add-production-ml-and-live`  
**Target**: `main`  
**Type**: Feature  
**Risk Level**: 🔴 HIGH (includes live trading capabilities)

---

## ⚠️ CRITICAL SAFETY REVIEW REQUIRED

This PR adds capabilities for **LIVE TRADING WITH REAL MONEY**. The following approvals are **MANDATORY** before merge:

### Required Approvals

- [ ] **Technical Lead** - Code review and architecture approval
- [ ] **Security Team** - Security scan and credential management review
- [ ] **Compliance Officer** - Regulatory compliance verification
- [ ] **Risk Manager** - Risk controls and position limits review
- [ ] **Legal** - Terms of service and liability review (if applicable)

### Human-in-the-Loop Requirements

- [ ] All live trading connectors include two-step confirmation
- [ ] Kill switch tested and verified functional
- [ ] No automatic order placement without explicit approval
- [ ] All secrets managed via environment variables (no hardcoded credentials)
- [ ] Comprehensive audit logging implemented

---

## Code Review Checklist

### General

- [ ] Code follows project style guide (black, isort, flake8)
- [ ] All functions have type hints
- [ ] Docstrings present and complete
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Error handling comprehensive
- [ ] Logging appropriate and informative

### Testing

- [ ] Unit tests written for all new modules (target: >80% coverage)
- [ ] Integration tests for end-to-end flows
- [ ] Regression tests pass with canonical results
- [ ] Look-ahead bias tests pass (for ML pipelines)
- [ ] Safety system tests pass (kill switch, confirmation)
- [ ] All tests pass locally: `pytest -v`
- [ ] CI pipeline passes

### Security

- [ ] Bandit security scan passes
- [ ] Safety dependency audit passes
- [ ] No secrets in code or committed files
- [ ] `.env.example` provided (no actual secrets)
- [ ] Authentication/authorization implemented
- [ ] Rate limiting considered
- [ ] Input validation on all API endpoints

### Machine Learning

- [ ] Training data provenance documented
- [ ] No data leakage (time-based splits verified)
- [ ] Model cards generated for all models
- [ ] Explainability metrics computed
- [ ] Synthetic test data provided
- [ ] No automatic model deployment (manual only)

### Live Trading Safety

- [ ] Kill switch functional and tested
- [ ] `STOP_ALL_TRADING` flag checked before every order
- [ ] `QUANTZOO_ENV` environment checked
- [ ] Dry-run mode available and default
- [ ] Position size limits configurable
- [ ] Stop-loss mechanisms documented
- [ ] Manual confirmation required for live orders
- [ ] Broker connector config templates provided

### Documentation

- [ ] README updated with new features
- [ ] Installation instructions clear
- [ ] Configuration examples provided
- [ ] RUNBOOK.md complete with emergency procedures
- [ ] API documentation generated
- [ ] Model card template provided
- [ ] Deployment checklist complete

### Monitoring & Ops

- [ ] Prometheus metrics instrumentation added
- [ ] Grafana dashboards provided
- [ ] Health check endpoints working
- [ ] Logging comprehensive
- [ ] Docker images build successfully
- [ ] Docker Compose configs tested
- [ ] CI/CD workflow functional

---

## Pre-Merge Manual Verification

### Safety Systems

**Tester**: _____________ **Date**: _____________

- [ ] Kill switch activates correctly
- [ ] Kill switch blocks order placement
- [ ] Kill switch deactivates correctly
- [ ] Safety API authentication works
- [ ] Audit log captures all actions

**Evidence**: [Link to test results]

### Broker Connectors

**Tester**: _____________ **Date**: _____________

#### Paper Broker

- [ ] Orders place and fill correctly
- [ ] Positions tracked accurately
- [ ] P&L calculated correctly
- [ ] Commissions and slippage applied
- [ ] Safety checks function

#### Alpaca (Paper Mode Only)

- [ ] Connection establishes with paper API
- [ ] Orders place in paper account
- [ ] Positions retrieved
- [ ] Safety checks prevent live trading without proper env

#### IBKR (Paper Mode Only)

- [ ] Connection to paper port (7497)
- [ ] Orders place in paper account
- [ ] Positions retrieved
- [ ] Safety checks prevent port 7496 without proper env

**Evidence**: [Link to test results]

### ML Training Pipeline

**Tester**: _____________ **Date**: _____________

- [ ] Synthetic data generation works
- [ ] Training script runs without errors
- [ ] Model card generated correctly
- [ ] Metrics calculated properly
- [ ] No look-ahead bias detected

**Command**: `python ml/train_transformer.py --config configs/example_transformer.yaml --dry-run`

**Evidence**: [Link to logs/artifacts]

### Monitoring Stack

**Tester**: _____________ **Date**: _____________

- [ ] Prometheus starts and scrapes metrics
- [ ] Grafana connects to Prometheus
- [ ] Dashboards load correctly
- [ ] Metrics appear in dashboards

**Command**: `docker-compose -f ops/docker-compose.monitor.yml up`

**Evidence**: [Screenshot of dashboards]

---

## Post-Merge Actions

### Immediate (< 24 hours)

- [ ] Monitor CI/CD pipeline
- [ ] Verify no regressions in main
- [ ] Update project board
- [ ] Notify team of new features

### Short-term (< 1 week)

- [ ] Conduct team training on new features
- [ ] Update internal documentation
- [ ] Schedule code review retrospective
- [ ] Create follow-up issues for improvements

### Before Production Deployment

- [ ] **MANDATORY**: Complete full security audit
- [ ] **MANDATORY**: Obtain compliance sign-off
- [ ] **MANDATORY**: Legal review of terms and liability
- [ ] **MANDATORY**: Load testing and stress testing
- [ ] **MANDATORY**: Disaster recovery plan finalized
- [ ] **MANDATORY**: On-call rotation established
- [ ] **MANDATORY**: Incident response procedures tested
- [ ] Paper trading for minimum 2 weeks
- [ ] Forward testing with small capital
- [ ] Risk limits validated in live environment

---

## Known Limitations

Document any known issues or limitations:

1. **ML Models**: Trained on synthetic data only. Requires real data training before production.

2. **Broker Connectors**: Alpaca and IBKR require external API packages not in base install.

3. **Monitoring**: Grafana dashboards are templates and need customization.

4. **Testing**: Some integration tests require external services (optional packages).

5. **HF Spaces**: Demo app requires manual deployment to Hugging Face Hub.

---

## Rollback Plan

If critical issues discovered after merge:

1. **Revert PR**: `git revert <merge-commit>`
2. **Deactivate Features**: Set feature flags to disable new functionality
3. **Notify Users**: Communicate downtime/degradation
4. **Root Cause**: Investigate and document issue
5. **Fix Forward**: Create hotfix PR with solution

---

## Sign-Off

### Development Team

- [ ] **Developer**: _____________ (Code author)
- [ ] **Reviewer 1**: _____________ (Senior engineer)
- [ ] **Reviewer 2**: _____________ (Domain expert)

### Approval Authority

- [ ] **Technical Lead**: _____________ **Date**: _____
- [ ] **Security Lead**: _____________ **Date**: _____
- [ ] **Compliance**: _____________ **Date**: _____
- [ ] **Risk Management**: _____________ **Date**: _____

### Final Authorization

- [ ] **CTO/VP Engineering**: _____________ **Date**: _____

---

## Additional Notes

[Space for any additional context, concerns, or information]

---

**PR Checklist Version**: 1.0  
**Last Updated**: 2025-10-28
