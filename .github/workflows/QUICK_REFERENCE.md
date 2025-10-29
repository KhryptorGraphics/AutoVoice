# Final Validation Workflow - Quick Reference

## 🚀 Quick Start

### Trigger Workflow Manually
```bash
gh workflow run final_validation.yml
```

### Test Locally Before Push
```bash
./.github/workflows/test_locally.sh
```

### Validate Workflow YAML
```bash
python .github/workflows/validate_workflow.py
```

## 📊 Validation Targets

| Check | Target | Status |
|-------|--------|--------|
| Test Coverage | ≥90% | Required |
| Code Quality | ≥8.5/10 | Required |
| Type Errors | 0 | Required |
| Security | 0 critical | Required |
| Complexity | ≤10 avg | Required |
| Documentation | 100% | Required |

## 🔄 Automatic Triggers

- **Push to main/develop** → Full validation
- **Pull Request to main** → Validation + PR comment
- **Weekly (Sunday 00:00 UTC)** → Scheduled validation
- **Manual** → On-demand via UI or CLI

## 📦 Artifacts

### validation-report-<run-id> (30 days)
Complete validation results including tests, quality, integration, docs

### coverage-report-<run-id> (14 days)
HTML test coverage report with line-by-line analysis

### benchmark-results-<run-id> (90 days)
Performance metrics for trend tracking

## 🎯 Jobs

### 1. final-validation (60 min)
Main validation pipeline with all quality checks

### 2. performance-benchmarks (30 min)
Performance regression testing (main branch only)

### 3. security-scan (15 min)
Vulnerability scanning with Trivy (push events only)

## 🔧 Common Commands

### Check Workflow Status
```bash
gh run list --workflow=final_validation.yml
gh run view <run-id>
```

### Download Artifacts
```bash
gh run download <run-id>
```

### View Logs
```bash
gh run view <run-id> --log
```

## ⚠️ Troubleshooting

### Workflow Fails on CUDA
✅ Workflow automatically falls back to CPU mode

### Tests Timeout
🔧 Adjust timeout in pytest.ini or workflow

### Artifact Too Large
🔧 Artifacts auto-compress, max 2GB per artifact

### PR Comment Too Long
✅ Auto-truncates at 65KB with artifact link

## 🔗 Integration

### Claude Flow Hooks
```bash
# Pre-task initialization
npx claude-flow@alpha hooks pre-task

# Post-task completion
npx claude-flow@alpha hooks post-task

# Session end metrics
npx claude-flow@alpha hooks session-end
```

### GitHub Security
- Trivy SARIF results → GitHub Security tab
- Bandit security issues → Validation report

### PR Comments
- Automatic posting of validation results
- Includes pass/fail status
- Links to full artifacts

## 📈 Monitoring

### Status Badge
```markdown
[![Validation](https://github.com/your-org/autovoice/actions/workflows/final_validation.yml/badge.svg)](https://github.com/your-org/autovoice/actions/workflows/final_validation.yml)
```

### Key Metrics to Track
- Test coverage trend
- Code quality score
- Build duration
- Artifact size
- Failure rate

## 📚 Documentation

- [Workflow README](.github/workflows/README.md)
- [Technical Docs](../docs/github_actions_implementation.md)
- [Testing Guide](../docs/testing_guide.md)

## 🆘 Support

**Issues:** Report via GitHub Issues with `ci/cd` label
**Logs:** Check Actions tab for detailed execution logs
**Artifacts:** Download for local debugging
**Hooks:** Review Claude Flow coordination logs

---

**Last Updated:** 2025-10-28
**Version:** 1.0.0
