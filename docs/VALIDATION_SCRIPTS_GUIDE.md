# Validation Scripts Guide

## Overview

This guide documents the comprehensive validation suite for the AutoVoice project. The validation scripts perform code quality checks, integration testing, documentation validation, and generate a comprehensive report.

## Scripts Created

### 1. `scripts/validate_code_quality.py`

**Purpose**: Automated code quality checks using industry-standard tools.

**Tools Used**:
- **pylint**: Code analysis and style checking
- **flake8**: Style guide enforcement (PEP 8)
- **mypy**: Static type checking
- **radon**: Complexity analysis (cyclomatic complexity)
- **bandit**: Security vulnerability scanning

**Exit Codes**:
- `0`: All critical checks passed (flake8, mypy)
- `1`: Critical checks failed

**Usage**:
```bash
python scripts/validate_code_quality.py
```

**Output**: `validation_results/code_quality.json`

**Thresholds**:
- Critical: flake8 and mypy must pass
- Warnings: pylint errors, high complexity (≥10), high severity security issues

---

### 2. `scripts/validate_integration.py`

**Purpose**: Validate component integration and system functionality.

**Components Tested**:
1. **Module Imports**: All required modules can be imported
2. **GPU Manager**: Initialization and resource allocation
3. **Audio Processor**: Integration with CUDA kernels
4. **Web API**: Basic functionality (health, info endpoints)
5. **Pipeline**: Component integration and method availability
6. **CUDA Kernels**: Optional kernel loading (non-critical)

**Exit Codes**:
- `0`: All critical components passed
- `1`: Critical component failure

**Usage**:
```bash
python scripts/validate_integration.py
```

**Output**: `validation_results/integration.json`

**Critical Components**: gpu_manager, audio_processor, pipeline

---

### 3. `scripts/validate_documentation.py`

**Purpose**: Ensure documentation completeness and correctness.

**Checks Performed**:
1. **Required Files**: README.md, implementation docs, config files
2. **Module Docstrings**: All modules and classes have docstrings
3. **Code Examples**: Python code blocks in docs are syntactically valid
4. **README Links**: Local file links are valid
5. **API Documentation**: API endpoints have docstrings

**Exit Codes**:
- `0`: Always (documentation issues are warnings, not errors)

**Usage**:
```bash
python scripts/validate_documentation.py
```

**Output**: `validation_results/documentation.json`

**Note**: Documentation issues do not fail CI/CD pipeline

---

### 4. `scripts/generate_validation_report.py`

**Purpose**: Aggregate all validation results into a comprehensive report.

**Inputs**:
- `validation_results/code_quality.json`
- `validation_results/integration.json`
- `validation_results/documentation.json`
- `validation_results/test_results.json`

**Outputs**:
- `FINAL_VALIDATION_REPORT.md`: Human-readable markdown report
- `validation_results/summary.json`: Machine-readable summary

**Exit Codes**:
- `0`: Always (report generation should not fail)

**Usage**:
```bash
python scripts/generate_validation_report.py
```

**Report Sections**:
1. Executive Summary (overall status, failures, warnings)
2. Code Quality Details
3. Integration Testing Results
4. Documentation Status
5. System Validation Tests
6. Recommendations
7. Next Steps

---

### 5. `scripts/run_full_validation.sh`

**Purpose**: Orchestrate complete validation suite.

**Execution Flow**:
1. Create validation results directory
2. Generate test data (if available)
3. Run system validation tests
4. Run code quality checks
5. Run integration validation
6. Run documentation validation
7. Generate final report

**Exit Codes**:
- `0`: All critical validations passed
- `1`: Any critical validation failed

**Usage**:
```bash
bash scripts/run_full_validation.sh
```

**Features**:
- Exit on first error (`set -e`)
- Timestamped execution
- Progress indicators
- Report preview
- CI/CD ready

---

## Installation

### Required Tools

Install validation dependencies:

```bash
pip install pylint flake8 mypy radon bandit pytest pytest-json-report
```

### Optional Dependencies

For full functionality:

```bash
pip install fastapi torch numpy
```

---

## Usage Examples

### Run Full Validation Suite

```bash
# Complete validation with report
bash scripts/run_full_validation.sh
```

### Run Individual Validations

```bash
# Code quality only
python scripts/validate_code_quality.py

# Integration only
python scripts/validate_integration.py

# Documentation only
python scripts/validate_documentation.py
```

### Generate Report from Existing Results

```bash
python scripts/generate_validation_report.py
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Validation Suite

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
          pip install pylint flake8 mypy radon bandit pytest pytest-json-report

      - name: Run Validation Suite
        run: bash scripts/run_full_validation.sh

      - name: Upload Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: |
            FINAL_VALIDATION_REPORT.md
            validation_results/
```

---

## Interpreting Results

### Code Quality Status

- **✅ PASSED**: All critical checks passed
- **⚠️ PASSED WITH WARNINGS**: Critical checks passed, but warnings exist
- **❌ FAILED**: Critical checks failed (flake8 or mypy)

### Integration Status

- **✅ PASSED**: All critical components initialized successfully
- **❌ FAILED**: One or more critical components failed

### Documentation Status

- **✅ PASSED**: All documentation checks passed
- **⚠️ ISSUES**: Documentation has warnings (non-critical)

---

## Customization

### Adjust Code Quality Thresholds

Edit `scripts/validate_code_quality.py`:

```python
# Change complexity threshold
result = subprocess.run(
    ['radon', 'cc', 'src/auto_voice', '--min', 'B'],  # Change 'C' to 'B'
    ...
)

# Change line length
result = subprocess.run(
    ['flake8', 'src/auto_voice', '--max-line-length=120'],  # Change 100 to 120
    ...
)
```

### Add Custom Integration Tests

Edit `scripts/validate_integration.py`:

```python
def validate_custom_component() -> Dict[str, Any]:
    """Validate custom component."""
    try:
        from auto_voice.custom.component import CustomComponent
        component = CustomComponent()
        return {
            'passed': True,
            'details': {...}
        }
    except Exception as e:
        return {
            'passed': False,
            'error': str(e)
        }

# Add to main()
results['custom_component'] = validate_custom_component()
```

### Modify Report Format

Edit `scripts/generate_validation_report.py` to customize report sections and format.

---

## Troubleshooting

### Missing Tools Error

```
ERROR: Missing required tools: pylint, flake8
Install with: pip install pylint flake8 mypy radon bandit
```

**Solution**: Install missing dependencies

### Import Errors

```
Failed to import auto_voice.module: No module named 'auto_voice'
```

**Solution**: Ensure AutoVoice is installed or PYTHONPATH is set correctly

### Permission Denied

```
bash: scripts/run_full_validation.sh: Permission denied
```

**Solution**: Make script executable
```bash
chmod +x scripts/run_full_validation.sh
```

---

## Best Practices

1. **Run before commits**: Validate changes before committing
2. **Fix critical issues first**: Address flake8/mypy errors immediately
3. **Address warnings progressively**: Reduce complexity and security issues over time
4. **Keep documentation updated**: Maintain docstrings and docs/
5. **Review reports**: Analyze trends in validation results
6. **Automate in CI**: Always run validation in CI/CD pipeline

---

## Exit Code Reference

All scripts follow consistent exit code conventions for CI/CD gating:

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| 0 | Success or warnings only | Continue pipeline |
| 1 | Critical failure | Block pipeline |

**Scripts that can return 1**:
- `validate_code_quality.py`: flake8 or mypy failure
- `validate_integration.py`: Critical component failure
- `run_full_validation.sh`: Any critical validation failure

**Scripts that always return 0**:
- `validate_documentation.py`: Documentation issues are warnings
- `generate_validation_report.py`: Report generation should not fail pipeline

---

## Future Enhancements

Potential improvements to validation suite:

1. **Performance Benchmarking**: Add performance regression detection
2. **Coverage Thresholds**: Enforce minimum test coverage percentages
3. **License Compliance**: Check dependency licenses
4. **Docker Validation**: Validate Docker build and container health
5. **API Contract Testing**: Validate OpenAPI spec compliance
6. **Load Testing**: Basic load tests for web API
7. **Security Scanning**: Add SAST tools like semgrep
8. **Dependency Scanning**: Check for vulnerable dependencies

---

## Support

For issues or questions:

1. Check `validation_results/` directory for detailed error information
2. Review `FINAL_VALIDATION_REPORT.md` for comprehensive analysis
3. Consult individual script documentation above
4. Review tool-specific documentation (pylint, flake8, etc.)

---

## Summary

The validation suite provides comprehensive quality gates for the AutoVoice project:

- **Code Quality**: Industry-standard linting, type checking, complexity analysis
- **Integration**: Component functionality and system integration
- **Documentation**: Completeness and correctness of documentation
- **Reporting**: Comprehensive, actionable reports

All scripts are designed to be CI/CD friendly with proper exit codes and JSON output for automation.
