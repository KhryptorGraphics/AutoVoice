# AutoVoice Validation Workflow

Complete validation system for ensuring code quality, integration correctness, documentation completeness, and performance benchmarks.

## Overview

The validation system consists of 5 main components:

1. **Code Quality Validation** - Static analysis, linting, type checking, complexity analysis, security scanning
2. **Integration Validation** - Component integration testing, API validation, CUDA kernel checks
3. **Documentation Validation** - Docstring completeness, code examples, README links, API docs
4. **Performance Profiling** - Stage timing, GPU utilization, memory usage
5. **Validation Report Generation** - Comprehensive markdown report aggregating all results

## Quick Start

### Run Complete Validation Suite

```bash
# Run all validations and generate report
cd /home/kp/autovoice
python3 scripts/run_validation_suite.py
```

### Run Individual Validations

```bash
# Code quality only
python3 scripts/validate_code_quality.py

# Integration tests only
python3 scripts/validate_integration.py

# Documentation checks only
python3 scripts/validate_documentation.py

# Performance profiling only
python3 scripts/profile_performance.py

# Generate report from existing results
python3 scripts/generate_validation_report.py
```

## Validation Components

### 1. Code Quality Validation

**Script:** `scripts/validate_code_quality.py`

**Tools Used:**
- **pylint** - Code analysis and style checking
- **flake8** - PEP 8 style guide enforcement
- **mypy** - Static type checking
- **radon** - Cyclomatic complexity analysis
- **bandit** - Security vulnerability scanning

**Target Modules:**
- `src/auto_voice/audio/source_separator.py`
- `src/auto_voice/audio/pitch_extractor.py`
- `src/auto_voice/inference/voice_cloner.py`
- `src/auto_voice/models/singing_voice_converter.py`
- `src/auto_voice/inference/singing_conversion_pipeline.py`

**Output:** `validation_results/code_quality.json`

**Exit Codes:**
- 0: All checks passed
- 1: Critical checks failed (flake8 or mypy)

**Quality Thresholds:**
- Pylint errors: 0 (warnings allowed)
- Flake8: Must pass (max line length 100)
- Mypy: Must pass (ignore missing imports)
- Radon: High complexity count reported
- Bandit: High severity issues reported

### 2. Integration Validation

**Script:** `scripts/validate_integration.py`

**Validated Components:**
- **GPU Manager** - Device allocation, memory management, device placement
- **Audio Processor** - I/O operations, format handling, CUDA integration
- **Voice Profile Storage** - Save/load operations, thread safety
- **Web API** - Endpoint functionality, request/response handling
- **Pipeline Integration** - Component coordination, data flow

**Output:** `validation_results/integration_validation.json`

**Exit Codes:**
- 0: All critical components validated
- 1: Critical component failures

**Critical Components:**
- imports (all modules must load)
- gpu_manager (must initialize)
- audio_processor (must process audio)
- pipeline (must create and run)

**Non-Critical:**
- cuda_kernels (optional for CPU mode)
- web_api (can skip if dependencies missing)

### 3. Documentation Validation

**Script:** `scripts/validate_documentation.py`

**Validated Aspects:**
- **Module Docstrings** - All Python modules and classes have docstrings
- **Code Examples** - Python code blocks in docs are syntactically valid
- **README Links** - All markdown links work (local files exist, URLs valid)
- **API Documentation** - All async functions (API endpoints) documented
- **Required Files** - Key documentation files present

**Required Documentation Files:**
- `README.md`
- `docs/cuda_optimization_guide.md`
- `docs/implementation_complete.md`
- `config/model_config.yaml`

**Output:** `validation_results/documentation.json`

**Exit Codes:**
- 0: Always passes (warnings only, not critical)

### 4. Performance Profiling

**Script:** `scripts/profile_performance.py`

**Profiled Metrics:**
- **Stage Timing** - Time spent in each pipeline stage:
  - separation (vocal/instrumental separation)
  - pitch_extraction (F0 contour extraction)
  - voice_conversion (model inference)
  - audio_mixing (final audio synthesis)
- **GPU Utilization** - Sampled at 100-200ms intervals per stage
- **Memory Usage** - Peak GPU memory consumption

**Requirements:**
- Test audio file: `tests/data/test_song.wav`
- GPU utilization target: >70% mean when CUDA available
- pynvml library for GPU monitoring (optional)

**Output:** `validation_results/performance_breakdown.json`

**Exit Codes:**
- 0: Profiling successful, GPU utilization met (if CUDA)
- 1: Profiling failed or GPU utilization below threshold

**Performance Assertions:**
- Mean GPU utilization > 70% (when CUDA available)
- All stages have valid timing data
- No single stage dominates >60% of total time

### 5. Validation Report Generation

**Script:** `scripts/generate_validation_report.py`

**Aggregated Data Sources:**
- `validation_results/code_quality.json`
- `validation_results/integration_validation.json`
- `validation_results/documentation.json`
- `validation_results/system_validation.json` (from test suite)
- `validation_results/performance_breakdown.json`
- `validation_results/docker_validation.log`
- `validation_results/quality_evaluation/*.json` (optional)

**Output:** `FINAL_VALIDATION_REPORT.md`

**Report Sections:**
1. Executive Summary - Overall status, pass/fail counts
2. System Capabilities - CUDA availability, GPU devices
3. Performance Benchmarks - Stage timing, GPU utilization
4. Quality Metrics - Pitch accuracy, similarity, MOS, STOI, MCD (if available)
5. Code Quality - Linting, type checking, complexity, security
6. Integration - Component validation results
7. Documentation - Docstring coverage, link validation
8. Deployment - Docker build status
9. Known Limitations - Identified constraints
10. Recommendations - Priority actions

**Exit Codes:**
- 0: Report generated successfully

## Output Structure

```
/home/kp/autovoice/
├── FINAL_VALIDATION_REPORT.md          # Final report (generated)
├── validation_results/                  # All validation data
│   ├── code_quality.json               # Code quality results
│   ├── integration_validation.json     # Integration test results
│   ├── documentation.json              # Doc validation results
│   ├── performance_breakdown.json      # Performance profiling data
│   ├── system_validation.json          # System capability checks
│   ├── docker_validation.log           # Docker build log
│   └── quality_evaluation/             # Optional quality metrics
│       ├── pitch_accuracy.json
│       ├── speaker_similarity.json
│       └── audio_quality.json
├── scripts/                             # Validation scripts
│   ├── run_validation_suite.py         # Master orchestration script
│   ├── validate_code_quality.py        # Code quality validation
│   ├── validate_integration.py         # Integration validation
│   ├── validate_documentation.py       # Documentation validation
│   ├── profile_performance.py          # Performance profiling
│   └── generate_validation_report.py   # Report generation
└── docs/
    └── validation_workflow.md          # This document
```

## Exit Code Reference

### run_validation_suite.py
- **0**: All validations passed
- **1**: Critical validation failures (code quality or integration)
- **2**: Non-critical warnings (documentation or performance issues)

### Individual Scripts
- **0**: Validation passed (or passed with non-critical warnings)
- **1**: Validation failed

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
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pylint flake8 mypy radon bandit pynvml

      - name: Run validation suite
        run: python3 scripts/run_validation_suite.py

      - name: Upload validation report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: |
            FINAL_VALIDATION_REPORT.md
            validation_results/
```

### Pre-commit Hook Example

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running validation suite..."
python3 scripts/run_validation_suite.py

if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Commit aborted."
    exit 1
fi

echo "✅ Validation passed."
exit 0
```

## Troubleshooting

### Missing Dependencies

If validation tools are not installed:

```bash
pip install pylint flake8 mypy radon bandit
```

For GPU monitoring:

```bash
pip install pynvml
```

### Test Data Missing

If performance profiling fails due to missing test audio:

```bash
python3 scripts/generate_test_data.py
```

### Import Errors

If integration validation fails with import errors, ensure AutoVoice is installed:

```bash
pip install -e .
```

### GPU Monitoring Unavailable

GPU monitoring is optional. If pynvml is not available, profiling will continue without GPU metrics.

## Best Practices

1. **Run validation suite before commits** - Catch issues early
2. **Review FINAL_VALIDATION_REPORT.md** - Understand validation results
3. **Address critical failures immediately** - Block merges on failures
4. **Monitor performance trends** - Track GPU utilization and stage timing over time
5. **Keep documentation updated** - Failing doc checks indicate outdated docs

## Quality Targets

### Code Quality
- **Flake8**: 100% pass rate (critical)
- **Mypy**: 100% pass rate (critical)
- **Pylint errors**: 0 errors (warnings allowed)
- **Radon**: <10% high complexity functions
- **Bandit**: 0 high severity issues

### Integration
- **Critical components**: 100% pass rate
- **Non-critical components**: Best effort

### Documentation
- **Module docstrings**: >90% coverage
- **README links**: 100% valid
- **API docs**: 100% documented endpoints

### Performance
- **GPU utilization**: >70% mean (when CUDA available)
- **Stage balance**: No stage >60% of total time

## Related Documentation

- [CUDA Optimization Guide](cuda_optimization_guide.md)
- [Implementation Complete](implementation_complete.md)
- [TensorRT Implementation Progress](tensorrt_implementation_progress.md)
- [Quality Evaluation Guide](quality_evaluation_guide.md)

## Support

For issues or questions about validation:
1. Check validation report for specific error messages
2. Review individual validation script output
3. Check `validation_results/*.json` for detailed data
4. Consult related documentation guides
