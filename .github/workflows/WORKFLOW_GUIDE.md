# GitHub Actions Workflow Quick Reference

## Final Validation Pipeline

### Triggers
- **Auto**: Push to main, Pull requests to main
- **Manual**: Actions tab → "Final Validation Pipeline" → Run workflow

### Manual Options
- `skip_gpu_tests`: true/false (default: false)
- `validation_level`: quick/standard/comprehensive (default: standard)

### Jobs Overview

```
validation (90min)
  ├─ Setup Python & Dependencies
  ├─ Generate Test Data
  ├─ Run System Validation
  ├─ Run End-to-End Tests
  ├─ Run Performance Tests (skip on 'quick')
  ├─ Code Quality Validation
  ├─ Integration Validation
  ├─ Documentation Validation
  ├─ Security Scan (only on 'comprehensive')
  ├─ Generate Final Report
  └─ Upload Artifacts

docker-validation (45min)
  ├─ Build Docker Image
  ├─ Run Docker Tests
  └─ Upload Logs
  (Only on: push to main OR comprehensive level)

summary
  ├─ Download All Artifacts
  ├─ Generate Summary
  └─ Final Status Check
```

### Artifacts Produced

| Artifact | Contents | Retention |
|----------|----------|-----------|
| validation-results-3.10 | All test results, reports, logs | 30 days |
| test-reports-3.10 | HTML/JSON test reports | 30 days |
| coverage-report-3.10 | Code coverage HTML report | 30 days |
| docker-validation-log | Docker test logs | 30 days |

### Environment Variables

```yaml
PYTHON_VERSION: '3.10'
CUDA_VERSION: '11.8.0'
SKIP_GPU_TESTS: auto-detected or manual
VALIDATION_LEVEL: quick/standard/comprehensive
```

### Validation Levels

**Quick** (~30min)
- ✓ System validation
- ✓ E2E tests
- ✓ Code quality
- ✓ Integration
- ✓ Documentation
- ✗ Performance tests
- ✗ Security scans
- ✗ Docker validation

**Standard** (~60min) [Default]
- ✓ All quick items
- ✓ Performance tests
- ✗ Security scans
- ✗ Docker validation

**Comprehensive** (~90min)
- ✓ All standard items
- ✓ Security scans
- ✓ Docker validation

### Success Criteria

Job passes if:
- All tests pass (or continue-on-error for reporting)
- `overall_status == 'PASS'` in final_report.json
- Overall score meets threshold
- Docker validation succeeds (if applicable)

### Quick Commands

#### View Workflow Status
```bash
gh workflow view final_validation.yml
```

#### Run Workflow Manually
```bash
# Standard validation
gh workflow run final_validation.yml --ref main

# Quick validation without GPU
gh workflow run final_validation.yml \
  --ref main \
  -f skip_gpu_tests=true \
  -f validation_level=quick

# Comprehensive validation
gh workflow run final_validation.yml \
  --ref main \
  -f validation_level=comprehensive
```

#### Monitor Running Workflow
```bash
gh run watch
```

#### List Recent Runs
```bash
gh run list --workflow=final_validation.yml --limit 5
```

#### Download Artifacts
```bash
# Download all artifacts for latest run
gh run download

# Download specific run
gh run download <run-id>

# Download specific artifact
gh run download <run-id> -n validation-results-3.10
```

#### View Logs
```bash
# View logs for latest run
gh run view --log

# View specific job logs
gh run view <run-id> --job=validation --log
```

### Troubleshooting

#### GPU Tests Failing
```bash
# Skip GPU tests manually
gh workflow run final_validation.yml -f skip_gpu_tests=true
```

#### Check Validation Report
```bash
# Download and view
gh run download <run-id> -n validation-results-3.10
cat validation_results/FINAL_VALIDATION_REPORT.md
```

#### Check Specific Test Failures
```bash
# Download test reports
gh run download <run-id> -n test-reports-3.10
# Open HTML reports in browser
open validation_results/reports/system_validation.html
```

#### Check Code Quality Issues
```bash
# Download validation results
gh run download <run-id> -n validation-results-3.10
# View quality report
cat validation_results/reports/code_quality.json | jq '.'
```

#### Check Coverage
```bash
# Download coverage report
gh run download <run-id> -n coverage-report-3.10
# Open in browser
open index.html
```

### Local Testing

Before pushing, test locally:

```bash
# Setup
mkdir -p validation_results/{reports,logs,artifacts}
mkdir -p tests/data/validation

# Generate test data
python tests/data/validation/generate_test_data.py

# Run tests
pytest tests/test_system_validation.py -v \
  --json-report \
  --json-report-file=validation_results/reports/system_validation.json \
  --html=validation_results/reports/system_validation.html

pytest tests/test_end_to_end.py -v \
  --json-report \
  --json-report-file=validation_results/reports/e2e_tests.json

# Run validations
python scripts/validate_code_quality.py \
  --output validation_results/reports/code_quality.json

python scripts/validate_integration.py \
  --output validation_results/reports/integration.json

python scripts/validate_documentation.py \
  --output validation_results/reports/documentation.json

# Generate report
python scripts/generate_validation_report.py \
  --system-validation validation_results/reports/system_validation.json \
  --e2e-tests validation_results/reports/e2e_tests.json \
  --code-quality validation_results/reports/code_quality.json \
  --integration validation_results/reports/integration.json \
  --documentation validation_results/reports/documentation.json \
  --output validation_results/FINAL_VALIDATION_REPORT.md \
  --json-output validation_results/reports/final_report.json

# View report
cat validation_results/FINAL_VALIDATION_REPORT.md
```

### Cache Management

#### Clear Workflow Cache
```bash
# List caches
gh cache list

# Delete specific cache
gh cache delete <cache-key>

# Delete all caches (requires confirmation)
gh cache list | awk '{print $2}' | xargs -I {} gh cache delete {}
```

### PR Integration

When workflow runs on PR:
- Automatic comment with validation report
- Status check appears on PR
- Merge blocked if validation fails

### Performance Tips

1. Use caching effectively (pip, torch)
2. Run quick validation during development
3. Run comprehensive before merge to main
4. Parallelize tests with `-n auto`
5. Skip GPU tests if not needed

### Security Best Practices

1. Never commit secrets
2. Use GitHub Secrets for sensitive data
3. Run security scans on comprehensive level
4. Review Bandit reports regularly
5. Keep dependencies updated

### Related Files

- Workflow: `.github/workflows/final_validation.yml`
- Documentation: `docs/github_actions_validation_workflow.md`
- Scripts: `scripts/validate_*.py`, `scripts/generate_validation_report.py`
- Tests: `tests/test_system_validation.py`, `tests/test_end_to_end.py`

### Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Download artifacts for detailed analysis
3. Review documentation in `docs/`
4. Test scripts locally before workflow run
