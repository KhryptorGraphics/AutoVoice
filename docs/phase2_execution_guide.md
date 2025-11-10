# Phase 2 Execution Guide

**Version**: 1.0
**Last Updated**: 2025-11-01
**Purpose**: Non-interactive, reproducible workflow for Phase 2 test execution

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pre-flight Validation](#pre-flight-validation)
3. [Phase 2 Execution Steps](#phase-2-execution-steps)
4. [Understanding Results](#understanding-results)
5. [Reviewing Reports](#reviewing-reports)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Next Steps](#next-steps)

---

## Quick Start

**For the impatient developer:**

```bash
# 1. Quick validation (< 1 min)
./run_tests.sh validate

# 2. Full Phase 2 execution (10-15 min)
./run_tests.sh phase2

# 3. Rerun any failures
./run_tests.sh rerun
```

**Expected outcome**: All tests pass, coverage reports generated, Phase 2 completion report created.

---

## Pre-flight Validation

Before running the full Phase 2 suite, validate your environment and ensure basic tests pass.

### Step 1: Quick Test Check

**Command:**
```bash
./scripts/quick_test_check.sh
# Or via run_tests.sh:
./run_tests.sh validate
```

**Duration**: < 1 minute

**What it does:**
- Validates Python environment
- Checks pytest availability
- Runs smoke tests (fast validation)
- Verifies CUDA extension availability
- Checks basic functionality

**Expected output:**
```
================================================================================
Quick Test Validation
================================================================================
Started at: 2025-11-01 10:30:00

✅ pytest found
✅ CUDA extension available
✅ Smoke tests passed (15 tests in 12.3s)

Quick validation complete!
```

**If validation fails:**
- Check Python version (3.8-3.12 required)
- Install pytest: `pip install pytest pytest-cov`
- Build CUDA extension: `pip install -e .`
- Review error messages and fix issues before proceeding

### Step 2: Environment Check

**Verify environment variables:**
```bash
# For CI environments (fail on missing CUDA)
export CI=1

# For development (allow missing CUDA)
export ALLOW_NO_CUDA=1
```

**Check CUDA availability:**
```bash
python -c "import cuda_kernels; print('CUDA extension OK')"
# Or:
python -c "from auto_voice import cuda_kernels; print('CUDA extension OK')"
```

---

## Phase 2 Execution Steps

Phase 2 executes the complete test suite with comprehensive validation and reporting.

### Step 1: Execute Phase 2

**Command:**
```bash
./scripts/phase2_execute.sh
# Or via run_tests.sh:
./run_tests.sh phase2
```

**Duration**: 10-15 minutes (varies by hardware)

**What it does:**
1. **Pre-flight checks**: Validates environment, pytest, CUDA extension
2. **Smoke tests**: Quick validation (< 30s)
3. **Integration tests**: Full integration suite (1-5 min)
4. **Core component tests**: Audio processor, models, and inference tests (2-5 min)
5. **Full test suite with coverage**: Complete test suite with coverage analysis (5-10 min)
6. **Report generation**: Creates Phase 2 completion report and coverage analysis

**Expected output:**
```
================================================================================
Phase 2: Execute Core Test Suite and Validate Functionality
================================================================================
Started at: 2025-11-01 10:35:00
Timestamp: 20251101_103500

[1/6] Pre-flight Validation
✅ Python 3.10.12 found
✅ pytest 7.4.3 found
✅ CUDA extension available

[2/6] Running Smoke Tests
✅ 7 tests passed in 12.3s

[3/6] Running Integration Tests
✅ 9 tests passed in 142.5s

[4/6] Running Core Component Tests
✅ Audio Processor: 15 tests passed
✅ Models: 20 tests passed
✅ Inference: 10 tests passed

[5/6] Running Full Test Suite with Coverage
✅ 151+ tests passed
✅ Coverage: 87.3% overall
✅ HTML report: htmlcov/index.html

[6/6] Generating Reports
✅ Phase 2 completion report generated
✅ Coverage analysis report generated


### Coverage Analysis Report

**Location**: `docs/coverage_analysis_report.md`

**Contents:**
- Overall coverage summary
- Module-by-module breakdown
- Uncovered lines and code paths
- Critical gaps requiring tests
- Recommendations for improvement

**How to use:**
1. Open `docs/coverage_analysis_report.md`
2. Review "Critical Gaps" section
3. Identify high-priority untested code
4. Add tests for critical functionality
5. Re-run Phase 2 to validate improvements

### HTML Coverage Report

**Location**: `htmlcov/index.html`

**How to view:**
```bash
# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html

# Windows
start htmlcov/index.html
```

**Features:**
- Interactive coverage visualization
- Line-by-line coverage highlighting
- Branch coverage details
- Sortable by coverage percentage
- Drill-down to specific files

**Color coding:**
- 🟢 Green: Covered lines
- 🔴 Red: Uncovered lines
- 🟡 Yellow: Partially covered branches

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "CUDA extension not found"

**Symptoms:**
```
❌ CUDA extension not found. Build with: pip install -e .
```

**Solution:**
```bash
# Build CUDA extension
pip install -e .

# Verify installation
python -c "import cuda_kernels; print('OK')"

# If build fails, check CUDA toolkit
nvcc --version
nvidia-smi
```

**For CI environments:**
```bash
# CI should fail if CUDA missing
export CI=1
./run_tests.sh phase2
```

**For development without GPU:**
```bash
# Allow running without CUDA
export ALLOW_NO_CUDA=1
./run_tests.sh phase2
```

#### Issue 2: "pytest not found"

**Symptoms:**
```
❌ pytest not found. Install with: pip install pytest
```

**Solution:**
```bash
# Install pytest and coverage plugin
pip install pytest pytest-cov

# Verify installation
pytest --version
```

#### Issue 3: Tests fail with "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size in tests
export TEST_BATCH_SIZE=1

# Clear GPU cache before tests
python -c "import torch; torch.cuda.empty_cache()"

# Run tests with smaller batches
./run_tests.sh phase2
```

#### Issue 4: "Permission denied" when running scripts

**Symptoms:**
```
bash: ./scripts/phase2_execute.sh: Permission denied
```

**Solution:**
```bash
# Make scripts executable
chmod +x scripts/*.sh
chmod +x run_tests.sh

# Verify permissions
ls -la scripts/*.sh
```

#### Issue 5: Phase 2 execution hangs or times out

**Symptoms:**
- Script runs for > 30 minutes
- No output for extended period
- Process appears stuck

**Solution:**
```bash
# Kill hung process
pkill -f phase2_execute

# Check for resource issues
nvidia-smi  # GPU memory
top         # CPU/RAM usage

# Run with verbose output
./scripts/phase2_execute.sh -v

# Run individual test suites
./run_tests.sh smoke
./run_tests.sh integration
./run_tests.sh performance
```

#### Issue 6: Coverage report not generated

**Symptoms:**
```
⚠️ scripts/analyze_coverage.py not found, skipping analysis
```

**Solution:**
```bash
# Verify script exists
ls -la scripts/analyze_coverage.py

# Run coverage manually
./run_tests.sh coverage

# Check for coverage.json
ls -la coverage.json

# Generate analysis manually
python scripts/analyze_coverage.py
```

#### Issue 7: Rerun script finds no failed tests

**Symptoms:**
```
No failed tests detected. All tests passed!
```

**But you know tests failed.**

**Solution:**
```bash
# Check .pytest_cache
ls -la .pytest_cache/v/cache/lastfailed

# Run tests with --lf flag manually
pytest --lf -v

# If cache is corrupted, delete and rerun
rm -rf .pytest_cache
./run_tests.sh phase2
```

---

## Best Practices

### For Developers

**1. Always run validation before Phase 2:**
```bash
./run_tests.sh validate  # < 1 min
./run_tests.sh phase2    # 10-15 min
```

**2. Use environment variables for control:**
```bash
# Development (allow missing CUDA)
export ALLOW_NO_CUDA=1

# CI (strict mode)
export CI=1
```

**3. Review coverage reports:**
```bash
# After Phase 2, always review:
open htmlcov/index.html
cat docs/coverage_analysis_report.md
```

**4. Rerun failures immediately:**
```bash
# Don't ignore failures
./run_tests.sh rerun
```

**5. Keep logs for debugging:**
```bash
# Logs are timestamped
ls -la logs/phase2_*.log

# Review recent logs
tail -100 logs/phase2_$(ls -t logs/ | head -1)
```

### For CI/CD Pipelines

**1. Use non-interactive mode:**
```bash
# Set CI flag
export CI=1

# Run Phase 2
./run_tests.sh phase2

# Exit code indicates success/failure
echo $?  # 0 = success, non-zero = failure
```

**2. Fail fast on missing dependencies:**
```bash
# CI should fail if CUDA missing
export CI=1
./run_tests.sh validate || exit 1
./run_tests.sh phase2 || exit 1
```

**3. Archive artifacts:**
```bash
# Save reports and logs
tar -czf phase2-artifacts.tar.gz \
  PHASE2_COMPLETION_REPORT.md \
  htmlcov/ \
  coverage.json \
  docs/coverage_analysis_report.md \
  logs/
```

**4. Set timeouts:**
```bash
# Prevent hung builds
timeout 30m ./run_tests.sh phase2
```

**5. Parallel execution (advanced):**
```bash
# Run test suites in parallel
./run_tests.sh smoke &
./run_tests.sh integration &
./run_tests.sh performance &
wait
```

### For Code Reviews

**1. Require Phase 2 success:**
- All PRs must pass Phase 2 execution
- Coverage must meet 85% threshold
- No failing tests allowed

**2. Review coverage changes:**
```bash
# Compare coverage before/after
git diff coverage.json
```

**3. Check for new untested code:**
- Review `docs/coverage_analysis_report.md`
- Ensure new code has tests
- Verify critical paths are covered

---

## Next Steps

### After Successful Phase 2 Execution

**1. Merge/Deploy:**
- All tests passed ✅
- Coverage meets threshold ✅
- Ready for merge or deployment

**2. Update documentation:**
- Document any new features
- Update API documentation
- Add examples for new functionality

**3. Monitor production:**
- Deploy to staging first
- Run smoke tests in production
- Monitor for issues

### After Failed Phase 2 Execution

**1. Review failures:**
```bash
# Check completion report
cat PHASE2_COMPLETION_REPORT.md

# Review logs
cat logs/phase2_*.log

# Rerun failures
./run_tests.sh rerun
```

**2. Fix issues:**
- Address test failures
- Fix coverage gaps
- Resolve environment issues

**3. Re-execute Phase 2:**
```bash
# After fixes
./run_tests.sh validate
./run_tests.sh phase2
```

### Continuous Improvement

**1. Increase coverage:**
- Target 90%+ coverage
- Add tests for edge cases
- Cover error handling paths

**2. Optimize test performance:**
- Reduce test execution time
- Parallelize where possible
- Use fixtures efficiently

**3. Enhance reporting:**
- Add custom metrics
- Track trends over time
- Automate report distribution

---

## Appendix: Command Reference

### Quick Reference

| Command | Alias | Duration | Purpose |
|---------|-------|----------|---------|
| `./run_tests.sh validate` | `./run_tests.sh v` | < 1 min | Pre-flight validation |
| `./run_tests.sh phase2` | `./run_tests.sh p2` | 10-15 min | Full Phase 2 execution |
| `./run_tests.sh rerun` | `./run_tests.sh r` | Varies | Rerun failed tests |
| `./run_tests.sh smoke` | `./run_tests.sh s` | < 30s | Smoke tests only |
| `./run_tests.sh integration` | `./run_tests.sh i` | 1-5 min | Integration tests |
| `./run_tests.sh coverage` | `./run_tests.sh c` | 10-15 min | Coverage analysis |

**Note**: Performance tests can be run separately with `pytest tests/test_bindings_performance.py -m performance` or `pytest tests/test_performance.py -m performance`.

### Direct Script Invocation

```bash
# Pre-flight validation
./scripts/quick_test_check.sh

# Phase 2 execution
./scripts/phase2_execute.sh

# Rerun failures
./scripts/rerun_failed_tests.sh

# Coverage analysis
python ./scripts/analyze_coverage.py
```

### Environment Variables

```bash
# CI mode (fail on missing CUDA)
export CI=1

# Allow missing CUDA (development)
export ALLOW_NO_CUDA=1

# Custom batch size (for memory-constrained environments)
export TEST_BATCH_SIZE=1
```

---

## Conclusion

This guide provides a complete, non-interactive, reproducible workflow for Phase 2 test execution. Follow the recommended workflow for best results:

1. **Validate** → Quick pre-flight check
2. **Phase 2** → Full test suite execution
3. **Rerun** → Address any failures
4. **Review** → Check reports and coverage
5. **Deploy** → Merge or deploy with confidence

For questions or issues, consult the [Troubleshooting](#troubleshooting) section or review the generated reports.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-01
**Maintained By**: AutoVoice Testing Team
✅ Report: PHASE2_COMPLETION_REPORT.md

================================================================================
Phase 2 Execution Complete
================================================================================
Total duration: 14m 32s
Tests passed: 72/72
Coverage: 89.7% overall
Status: ✅ SUCCESS
```

### Step 2: Review Results

**Check test results:**
```bash
# View summary
cat PHASE2_COMPLETION_REPORT.md

# View detailed logs
cat logs/phase2_20251101_103500.log
```

**Check coverage:**
```bash
# Open HTML coverage report
open htmlcov/index.html
# Or on Linux:
xdg-open htmlcov/index.html

# View coverage analysis
cat docs/coverage_analysis_report.md
```

### Step 3: Rerun Failed Tests (if any)

**Command:**
```bash
./scripts/rerun_failed_tests.sh
# Or via run_tests.sh:
./run_tests.sh rerun
```

**Duration**: Varies (depends on number of failures)

**What it does:**
- Detects failed tests from last run
- Reruns only failed tests with verbose output
- Provides detailed failure information
- Suggests next steps

**Expected output (no failures):**
```
No failed tests detected. All tests passed!
```

**Expected output (with failures):**
```
Rerunning 3 failed tests:
  - tests/test_bindings_integration.py::test_pitch_detection
  - tests/test_bindings_integration.py::test_voice_conversion
  - tests/test_bindings_performance.py::test_batch_processing

Running: pytest --lf -v --tb=short

... (detailed output) ...

2/3 tests passed on rerun
1 test still failing:
  - tests/test_bindings_performance.py::test_batch_processing

Review logs and fix issues before proceeding.
```

---

## Understanding Results

### Performance Test Outcomes

The performance tests validate system performance and scalability.

### Test Outcomes

**✅ All tests passed:**
- Phase 2 execution successful
- Coverage reports generated
- Ready for deployment/merge

**⚠️ Some tests failed:**
- Review failure details in logs
- Rerun failed tests with `./run_tests.sh rerun`
- Fix issues and re-execute Phase 2

**❌ Phase 2 execution failed:**
- Check pre-flight validation
- Review error messages
- Ensure CUDA extension is built
- Check environment variables

### Coverage Metrics

**Target coverage**: 85%+ overall

**Coverage breakdown:**
- `src/cuda_kernels`: 85%+ (CUDA extension code)
- `src/auto_voice`: 90%+ (Python application code)

**Low coverage areas:**
- Review `docs/coverage_analysis_report.md`
- Identify untested code paths
- Add tests for critical functionality

---

## Reviewing Reports

### Phase 2 Completion Report

**Location**: `PHASE2_COMPLETION_REPORT.md`

**Contents:**
- Execution summary (duration, timestamp, status)
- Test results breakdown (smoke, integration, performance)
- Coverage metrics (overall, by module)
- Failed tests (if any)
- Recommendations and next steps

**Example:**
```markdown
# Phase 2 Execution Report

**Status**: ✅ SUCCESS
**Duration**: 14m 32s
**Timestamp**: 2025-11-01 10:35:00

## Test Results
- Smoke: 15/15 passed
- Integration: 45/45 passed
- Performance: 12/12 passed
- **Total**: 72/72 passed

## Coverage
- src/cuda_kernels: 87.3%
- src/auto_voice: 92.1%
- **Overall**: 89.7%

## Recommendations
- ✅ All tests passed
- ✅ Coverage exceeds 85% target
- ✅ Ready for deployment
```

