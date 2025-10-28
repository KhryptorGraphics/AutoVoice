# AutoVoice CUDA Bindings Testing Guide

This guide provides comprehensive instructions for testing the CUDA kernel bindings implementation in AutoVoice.

## Table of Contents

- [Overview](#overview)
- [Test Suite Organization](#test-suite-organization)
- [Setup and Prerequisites](#setup-and-prerequisites)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Performance Baselines](#performance-baselines)
- [Troubleshooting](#troubleshooting)
- [CI/CD Integration](#cicd-integration)

## Overview

The AutoVoice CUDA bindings test suite validates the implementation of GPU-accelerated pitch detection and vibrato analysis kernels. The test suite is organized into three main categories:

1. **Smoke Tests** - Basic functionality and input validation
2. **Integration Tests** - End-to-end workflows with real audio
3. **Performance Tests** - Benchmarking and optimization validation

## Test Suite Organization

```
tests/
├── conftest.py                      # Shared fixtures and configuration
├── test_bindings_smoke.py          # Smoke tests (fast)
├── test_bindings_integration.py    # Integration tests (medium)
└── test_bindings_performance.py    # Performance benchmarks (slow)
```

### Test Files

#### `test_bindings_smoke.py`
- Module import validation
- Function signature verification
- Input validation and error handling
- Boundary value testing
- Edge case handling (silence, noise, extreme parameters)
- Memory stress testing

#### `test_bindings_integration.py`
- Real audio processing with known frequencies
- Multi-frequency pitch detection validation
- Vibrato analysis with synthetic modulated audio
- Various sample rates (8kHz - 48kHz)
- Various audio lengths (0.1s - 60s)
- Noise robustness testing
- Memory consistency checks

#### `test_bindings_performance.py`
- CUDA kernel execution time benchmarking
- CPU vs GPU speedup measurement
- Throughput testing (real-time factor)
- Latency measurements
- Memory usage profiling
- Sustained load testing

## Setup and Prerequisites

### Requirements

1. **CUDA-capable GPU** (compute capability 3.5+)
2. **CUDA Toolkit** (11.0+)
3. **PyTorch with CUDA support**
4. **Python dependencies**:
   ```bash
   pip install pytest pytest-cov numpy torch torchaudio librosa
   ```

### Building the Extension

Before running tests, ensure the CUDA extension is built:

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
# Quick verification
python -c "import cuda_kernels; print('Success!')"
```

## Running Tests

### Run All Tests

```bash
# Run all test suites
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/cuda_kernels --cov-report=html
```

### Run Specific Test Suites

#### Smoke Tests (Fast)
```bash
# Run smoke tests only
pytest tests/test_bindings_smoke.py -v

# Or use the script directly
python tests/test_bindings_smoke.py
```

#### Integration Tests
```bash
# Run integration tests
pytest tests/test_bindings_integration.py -v -m integration

# Exclude slow tests
pytest tests/test_bindings_integration.py -v -m "integration and not slow"
```

#### Performance Tests
```bash
# Run all performance benchmarks
pytest tests/test_bindings_performance.py -v -m performance -s

# The -s flag shows print output for performance metrics
```

### Run by Test Markers

Tests are marked with pytest markers for easy filtering:

```bash
# Run only CUDA tests
pytest -m cuda

# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run fast tests only (exclude slow)
pytest -m "not slow"

# Run performance benchmarks
pytest -m performance
```

### Available Markers

- `unit` - Unit tests (fast, isolated)
- `integration` - Integration tests (component interactions)
- `e2e` - End-to-end tests (complete workflows)
- `slow` - Slow tests (>1 second)
- `cuda` - Tests requiring CUDA
- `performance` - Performance benchmarks
- `audio` - Audio processing tests

## Test Categories

### 1. Smoke Tests

**Purpose**: Verify basic functionality and catch obvious errors quickly.

**What is tested**:
- Module can be imported
- Functions are properly exposed
- Functions accept correct parameters
- Invalid inputs raise appropriate errors
- Boundary values are handled correctly
- Large tensors can be processed
- Edge cases (silence, low amplitude) work

**Expected duration**: < 30 seconds

**Example output**:
```
============================================================
CUDA Kernel Bindings Smoke Test
============================================================

[1] Testing module import...
✓ cuda_kernels imported successfully

[2] Testing bindings exposed...
✓ launch_pitch_detection is available
✓ launch_vibrato_analysis is available

[3] Testing function callable...
✓ launch_pitch_detection callable with correct signature
✓ launch_vibrato_analysis callable with correct signature

[4] Testing input validation...
✓ Invalid frame_length raises exception with correct message
✓ CPU tensor raises exception with correct message
✓ Non-contiguous tensor raises exception with correct message
✓ Wrong dtype raises exception with correct message
✓ Invalid hop_length in vibrato_analysis raises exception
✓ All validation tests passed!

[5] Testing boundary values...
✓ Minimum parameters test passed
✓ Maximum parameters test passed
✓ Single frame test passed

[6] Testing stress with large tensors...
  Testing with 1323000 samples (2579 frames)...
✓ Large tensor test passed (memory increase: 15.23 MB)

[7] Testing empty and edge cases...
✓ Silent audio test passed
✓ Low amplitude audio test passed

============================================================
✓ All tests passed!
============================================================
```

### 2. Integration Tests

**Purpose**: Validate end-to-end workflows with realistic audio data.

**What is tested**:
- Pitch detection accuracy with known frequencies
- Multiple frequency detection
- Vibrato analysis with synthetic modulation
- Various sample rates (8-48 kHz)
- Various audio lengths (0.1-60 seconds)
- Noise robustness at different SNR levels
- Silence detection
- Memory consistency across multiple calls

**Expected duration**: 1-5 minutes

**Key test cases**:

1. **Known Frequency Detection**
   - Input: 440 Hz sine wave
   - Expected: Detected pitch within 5% of 440 Hz
   - Confidence: > 0.7

2. **Vibrato Analysis**
   - Input: Audio with 5.5 Hz vibrato, 50 cents depth
   - Expected: Rate within 30% of 5.5 Hz
   - Depth: > 10 cents

3. **Noise Robustness**
   - SNR levels: 30, 20, 10, 5 dB
   - Expected: Pitch detection at SNR ≥ 10 dB

### 3. Performance Tests

**Purpose**: Measure and validate GPU acceleration performance.

**What is tested**:
- Execution time for various audio lengths
- CUDA vs CPU speedup
- Real-time processing capability
- Kernel launch latency
- Memory usage scaling
- Sustained throughput under load

**Expected duration**: 2-10 minutes

**Example performance output**:
```
=== Short Audio (1s) ===
CUDA Mean Time: 2.34 ms
CUDA Throughput: 427.35x real-time

=== Medium Audio (10s) ===
CUDA Mean Time: 15.67 ms
CUDA Throughput: 638.26x real-time

=== CUDA vs CPU (5s audio) ===
CUDA Time: 8.23 ms
CPU Time: 184.56 ms
Speedup: 22.43x
CUDA Throughput: 607.54x real-time
CPU Throughput: 27.09x real-time

=== Vibrato Analysis Performance (5s audio) ===
Mean Time: 0.45 ms
Throughput: 11111.11x real-time

=== Kernel Launch Latency ===
Mean Latency: 1.23 ms
Std Latency: 0.08 ms
Min Latency: 1.15 ms
Max Latency: 1.45 ms
```

## Performance Baselines

### Expected Performance Targets

| Audio Length | CUDA Time | Real-time Factor | Notes |
|--------------|-----------|------------------|-------|
| 1 second     | < 5 ms    | > 200x          | Short audio, overhead dominated |
| 10 seconds   | < 20 ms   | > 500x          | Optimal batch size |
| 60 seconds   | < 100 ms  | > 600x          | Long audio, memory bound |

### CUDA vs CPU Speedup

- **Expected speedup**: 10-30x over librosa YIN algorithm
- **Minimum acceptable**: 5x speedup
- **Optimal**: 20x+ speedup

### Memory Usage

- **Linear scaling**: ~0.5 MB per second of audio
- **Maximum overhead**: 50 MB for 60s audio
- **No memory leaks**: < 10 MB increase after 10 iterations

## Test Coverage

### Current Coverage

Run coverage report:
```bash
pytest tests/ --cov=src/cuda_kernels --cov-report=term-missing
```

### Coverage Targets

- **Overall**: > 80%
- **Bindings (bindings.cpp)**: > 90%
- **Kernels (audio_kernels.cu)**: > 75%
- **Integration paths**: 100%

### Viewing Coverage

```bash
# Generate HTML report
pytest tests/ --cov=src/cuda_kernels --cov-report=html

# Open in browser
firefox htmlcov/index.html
```

## Troubleshooting

### Common Issues

#### 1. Module Import Fails

**Error**: `ImportError: No module named 'cuda_kernels'`

**Solutions**:
```bash
# Rebuild the extension
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"

# Try explicit import
python -c "from auto_voice import cuda_kernels"
```

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Run with smaller batch sizes
pytest tests/test_bindings_performance.py::test_performance_short_audio

# Check GPU memory
nvidia-smi
```

#### 3. Tests Hang or Timeout

**Symptoms**: Tests run indefinitely or timeout

**Solutions**:
- Check for deadlocks in CUDA code
- Verify kernel launch parameters
- Run with `--timeout=60` to enforce timeout
- Check GPU utilization with `nvidia-smi`

#### 4. Numerical Accuracy Issues

**Error**: Pitch detection not within expected range

**Debugging**:
```python
# Enable debug output
pytest tests/test_bindings_integration.py::test_pitch_detection_sine_wave -s -v

# Check intermediate values
import torch
import cuda_kernels

# Add assertions for intermediate outputs
assert torch.all(torch.isfinite(output_pitch)), "Non-finite values detected"
```

#### 5. Performance Degradation

**Symptoms**: Tests pass but performance is slower than expected

**Diagnostics**:
```bash
# Check GPU clock speed
nvidia-smi -q -d CLOCK

# Run with profiler
nsys profile python -m pytest tests/test_bindings_performance.py

# Check for thermal throttling
nvidia-smi -q -d TEMPERATURE
```

### Debugging Tips

1. **Run single test with verbose output**:
   ```bash
   pytest tests/test_bindings_smoke.py::test_function_callable -v -s
   ```

2. **Use pytest debugger**:
   ```bash
   pytest tests/ --pdb
   ```

3. **Check CUDA errors explicitly**:
   ```python
   import torch
   torch.cuda.synchronize()
   # Will raise error if kernel failed
   ```

4. **Enable CUDA error checking**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   pytest tests/
   ```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: CUDA Tests

on: [push, pull_request]

jobs:
  test-cuda:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:11.8.0-devel-ubuntu22.04
      options: --gpus all

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
          pip install pytest pytest-cov numpy librosa
          pip install -e .

      - name: Run smoke tests
        run: pytest tests/test_bindings_smoke.py -v

      - name: Run integration tests
        run: pytest tests/test_bindings_integration.py -v -m "integration and not slow"

      - name: Run performance tests
        run: pytest tests/test_bindings_performance.py -v -m performance -s

      - name: Generate coverage report
        run: pytest tests/ --cov=src/cuda_kernels --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Local Pre-commit Hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
set -e

echo "Running CUDA binding tests..."

# Run smoke tests (fast)
pytest tests/test_bindings_smoke.py -v

# Run integration tests (skip slow ones)
pytest tests/test_bindings_integration.py -v -m "integration and not slow"

echo "All tests passed!"
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Best Practices

### Writing New Tests

1. **Use pytest fixtures** from `conftest.py`:
   ```python
   def test_my_feature(cuda_kernels_module, synthetic_sine_wave):
       audio, frequency, sample_rate = synthetic_sine_wave
       # Test implementation
   ```

2. **Add appropriate markers**:
   ```python
   @pytest.mark.cuda
   @pytest.mark.integration
   def test_my_integration():
       pass
   ```

3. **Skip gracefully when CUDA unavailable**:
   ```python
   @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
   def test_cuda_feature():
       pass
   ```

4. **Clean up GPU memory**:
   ```python
   def test_memory_intensive():
       # Test code
       torch.cuda.empty_cache()  # Clean up
   ```

5. **Use meaningful assertions**:
   ```python
   # Good
   assert pitch_error < 0.05, f"Pitch error too high: {pitch_error:.2%}"

   # Bad
   assert pitch_error < 0.05
   ```

### Test Maintenance

- Run full test suite before commits
- Update baselines when performance improves
- Document any known flaky tests
- Keep test data minimal but representative
- Review test output regularly for degradation

## Advanced Usage

### Running with pytest-xdist (Parallel)

```bash
# Install plugin
pip install pytest-xdist

# Run tests in parallel (careful with CUDA tests)
pytest tests/test_bindings_smoke.py -n auto
```

**Note**: CUDA tests may not parallelize well due to GPU contention.

### Continuous Benchmarking

Track performance over time:

```bash
# Save benchmark results
pytest tests/test_bindings_performance.py -v -s > benchmark_results_$(date +%Y%m%d).txt

# Compare with previous run
diff benchmark_results_old.txt benchmark_results_new.txt
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory_profiler

# Profile memory usage
python -m memory_profiler tests/test_bindings_integration.py
```

## Summary

This comprehensive test suite ensures the CUDA bindings implementation is:
- **Correct**: Produces accurate results
- **Robust**: Handles edge cases and errors gracefully
- **Fast**: Achieves expected GPU acceleration
- **Reliable**: No memory leaks or crashes
- **Maintainable**: Well-organized and documented

Run the full suite regularly to catch regressions early:

```bash
# Quick check (< 1 minute)
pytest tests/test_bindings_smoke.py -v

# Full validation (5-10 minutes)
pytest tests/ -v -m "not slow"

# Complete test (10-15 minutes)
pytest tests/ -v --cov=src/cuda_kernels --cov-report=html
```

For questions or issues, refer to:
- Implementation docs: `docs/IMPLEMENTATION_SUMMARY.md`
- CUDA kernel details: `src/cuda_kernels/audio_kernels.cu`
- Binding details: `src/cuda_kernels/bindings.cpp`
