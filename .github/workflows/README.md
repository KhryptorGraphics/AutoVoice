# GitHub Actions Workflows

This directory contains CI/CD workflows for the AutoVoice project.

## Available Workflows

### 🔍 Final System Validation

**File:** `final_validation.yml`

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Weekly on Sundays at midnight (scheduled)
- Manual workflow dispatch

**Jobs:**

#### 1. final-validation
Comprehensive system validation including:
- **Environment Setup:** Python 3.10 + CUDA 11.8
- **Test Execution:** Full test suite with coverage
- **Code Quality:** Pylint, Flake8, Mypy, Bandit
- **Complexity Analysis:** Radon metrics
- **Integration Tests:** Cross-component validation
- **Documentation Validation:** Docs completeness check
- **Report Generation:** Markdown + JSON outputs

**Artifacts:**
- `validation-report-<run-id>`: Full validation results
- `coverage-report-<run-id>`: HTML test coverage report

**Outputs:**
- Validation report posted as PR comment
- GitHub Actions summary with key metrics
- Issues created on failure (main branch only)

#### 2. performance-benchmarks
Performance regression testing:
- Runs pytest-benchmark suite
- Compares against baseline metrics
- Posts benchmark results to PRs
- Stores results for trend analysis

**Artifacts:**
- `benchmark-results-<run-id>`: Performance metrics (90-day retention)

#### 3. security-scan
Security vulnerability scanning:
- Trivy filesystem scanner
- SARIF results uploaded to GitHub Security
- Fails on critical vulnerabilities

## Validation Targets

The workflow enforces these quality gates:

| Category | Target | Check |
|----------|--------|-------|
| Test Coverage | ≥90% | pytest-cov |
| Code Quality | ≥8.5/10 | pylint |
| Type Safety | 0 errors | mypy |
| Security | 0 critical | bandit + trivy |
| Complexity | ≤10 avg | radon |
| Documentation | 100% | custom validator |

## Using the Workflow

### View Results

1. **In Pull Requests:**
   - Validation report posted as comment
   - Status checks must pass before merge
   - Download artifacts for detailed analysis

2. **On Main Branch:**
   - Issues created on failure
   - Badge updates in README
   - Metrics tracked over time

### Manual Trigger

```bash
gh workflow run final_validation.yml
```

Or use the GitHub UI: Actions → Final System Validation → Run workflow

### Local Testing

To run validation locally before pushing:

```bash
# Generate test data
python tests/data/validation/generate_test_data.py

# Run validation tests
pytest tests/test_system_validation.py -v

# Run quality checks
python scripts/validate_code_quality.py
python scripts/validate_integration.py
python scripts/validate_documentation.py

# Generate report
python scripts/generate_validation_report.py
```

## Workflow Configuration

### Environment Variables

```yaml
PYTHON_VERSION: '3.10'
CUDA_VERSION: '11.8.0'
POETRY_VERSION: '1.7.0'
```

### Caching Strategy

The workflow caches:
- Python pip packages
- PyTorch models
- System dependencies

Cache key: `${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}`

### Timeouts

- Main validation: 60 minutes
- Performance benchmarks: 30 minutes
- Security scan: 15 minutes

## Hooks Integration

The workflow uses Claude Flow hooks for coordination:

```bash
# Pre-task: Initialize workflow context
npx claude-flow@alpha hooks pre-task

# Post-task: Record completion
npx claude-flow@alpha hooks post-task

# Session end: Export metrics
npx claude-flow@alpha hooks session-end
```

## Troubleshooting

### Common Issues

**1. CUDA Installation Fails**
```yaml
# Fallback to CPU mode
- name: Verify CUDA
  run: nvidia-smi || echo "CPU-only mode"
```

**2. Test Timeout**
```bash
# Increase timeout in pytest.ini
timeout = 600
```

**3. Artifact Upload Fails**
```yaml
# Check artifact size limit (2GB max)
if: always()  # Upload even on failure
```

**4. PR Comment Too Long**
```javascript
// Report is truncated at 65KB
const maxLength = 65000;
```

### Debug Mode

Enable debug logging:

```yaml
- name: Enable debug
  run: echo "ACTIONS_STEP_DEBUG=true" >> $GITHUB_ENV
```

## Badge Status

Add to README.md:

```markdown
![Validation Status](https://img.shields.io/github/actions/workflow/status/your-org/autovoice/final_validation.yml?branch=main&label=validation)
```

## Related Documentation

- [Testing Guide](../../docs/testing_guide.md)
- [Validation Criteria](../../docs/VERIFICATION_FIXES_REMAINING.md)
- [Quality Metrics](../../docs/quality_evaluation_guide.md)

## Contributing

When modifying workflows:

1. Test locally with `act` or similar tools
2. Use `workflow_dispatch` for testing
3. Check syntax with `actionlint`
4. Update this README with changes
5. Test on a feature branch first

## Support

- **GitHub Issues:** Report workflow problems
- **Actions Logs:** Check detailed execution logs
- **Artifacts:** Download for debugging
- **Hooks:** Check Claude Flow coordination logs
