# AutoVoice Test Execution Report

**Report Date**: 2025-11-07
**Execution Timestamp**: 2025-11-01 14:57 -0500 (from coverage artifact)
**Test Suite Version**: Latest run
**Execution Environment**: Linux x86_64, Python 3.13.5, pytest-8.3.4

## Executive Summary

**Overall Status**: FAILED
**Coverage Target Met**: NO (0.00% vs 80% target)
**Run Status**: Interrupted during collection
**Artifact Sources**: `/home/kp/autovoice/full_suite_log.txt`, `/home/kp/autovoice/htmlcov/index.html`

### Key Results
- **Test Collection**: Run aborted during collection with 2 skipped and 10 errors
- **Coverage**: 0.00% (FAILED to meet ≥80% target)
- **CUDA Smoke Tests**: Not executed due to collection failure
- **CUDA Integration Tests**: Not executed due to collection failure
- **CUDA Performance Tests**: Not executed due to collection failure

## Latest Run Summary

### Command Executed
```bash
pytest tests/ -v --cov=src/auto_voice --cov-report=html
```

### Full Suite Results

**Exact output from `/home/kp/autovoice/full_suite_log.txt` (lines 13, 516-530):**

```
collected 776 items / 10 errors / 2 skipped

=========================== short test summary info ============================
SKIPPED [1] tests/test_int8_calibration.py:18: TensorRT not available
SKIPPED [1] tests/test_voice_cloner_extensions.py:10: VoiceCloner not available
ERROR tests/test_conversion_pipeline.py
ERROR tests/test_core_integration.py
ERROR tests/test_dataset_verification_fixes.py
ERROR tests/test_singing_converter_enhancements.py
ERROR tests/test_trainer_local_rank.py
ERROR tests/test_training_voice_conversion.py
ERROR tests/test_utils.py
ERROR tests/test_voice_conversion.py
ERROR tests/test_vtlp_augmentation.py
ERROR tests/test_websocket_lifecycle.py
!!!!!!!!!!!!!!!!!!! Interrupted: 10 errors during collection !!!!!!!!!!!!!!!!!!!
======================== 2 skipped, 10 errors in 14.85s ========================
```

- **Status**: Interrupted during collection
- **Errors**: 10 (all ImportError due to GLIBCXX_3.4.30 not found)
- **Skipped**: 2
- **Runtime**: 14.85s
- **Artifact Reference**: `/home/kp/autovoice/full_suite_log.txt`

### Coverage Analysis

**Exact coverage from `/home/kp/autovoice/htmlcov/index.html` (line 14):**

```html
<span class="pc_cov">0.00%</span>
```

- **Coverage Percentage**: 0.00%
- **Target Met**: FAILED (0.00% < 80% required)
- **Artifact Reference**: `/home/kp/autovoice/htmlcov/index.html`
- **Coverage Report Generated**: 2025-11-01 14:57 -0500
- **Reason for 0% Coverage**: Run aborted during collection, no tests executed

## Results by Category

### CUDA Tests Summary

All CUDA test categories were not executed due to collection failure preventing any test execution.

#### Smoke Tests (`tests/test_bindings_smoke.py`)
- **Expected Tests Count**: 7 (from file inspection)
- **Status**: Not executed due to collection failure
- **Artifact Evidence**: None (no execution logs)

#### Integration Tests (`tests/test_bindings_integration.py`)
- **Expected Tests Count**: 9 (from file inspection)
- **Status**: Not executed due to collection failure
- **Artifact Evidence**: None (no execution logs)

#### Performance Tests (`tests/test_bindings_performance.py`)
- **Expected Tests Count**: 9 (from file inspection)
- **Status**: Not executed due to collection failure
- **Artifact Evidence**: None (no execution logs)

## Failures Analysis

### Collection Errors

**Root Cause** (from `/home/kp/autovoice/full_suite_log.txt`):
```
ImportError: /home/kp/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
(required by /home/kp/anaconda3/lib/python3.13/site-packages/scipy/fft/_pocketfft/pypocketfft.cpython-313-x86_64-linux-gnu.so)
```

- **Affected Files**: 10 test modules failed to import
- **Impact**: Complete test suite failure, 0% coverage
- **Resolution Required**: Fix GLIBCXX version mismatch before re-running

### Coverage Failure
- **Expected Coverage**: ≥80%
- **Actual Coverage**: 0.00%
- **Impact**: No code was executed
- **Resolution**: Fix collection errors first

## Recommendations

1. **Fix GLIBCXX Dependency**: Resolve the `GLIBCXX_3.4.30` version mismatch
2. **Verify SciPy Installation**: Ensure SciPy is compatible with system libraries
3. **Re-run Test Suite**: Execute `pytest tests/ -v --cov=src/auto_voice --cov-report=html` after fixes
4. **Update This Report**: Replace with actual test results once collection succeeds

## Next Steps

- Fix GLIBCXX_3.4.30 version mismatch
- Re-execute full test suite
- Update this report with verified test results and coverage data

---

**Report Generated From Artifacts**:
- `/home/kp/autovoice/full_suite_log.txt`: Test collection output and error summary
- `/home/kp/autovoice/htmlcov/index.html`: Coverage report (0.00%)