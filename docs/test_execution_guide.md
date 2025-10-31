# AutoVoice Test Execution Guide

## Overview

This comprehensive guide provides detailed instructions for executing the AutoVoice test suite, including the newly enhanced CUDA integration tests, performance benchmarks, and automated build validation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [CUDA Environment Setup](#cuda-environment-setup)
3. [Test Suite Structure](#test-suite-structure)
4. [Quick Start](#quick-start)
5. [Detailed Test Execution](#detailed-test-execution)
6. [CUDA-Specific Testing](#cuda-specific-testing)
7. [Performance Benchmarking](#performance-benchmarking)
8. [Coverage Analysis](#coverage-analysis)
9. [CI/CD Integration](#cicd-integration)
10. [Troubleshooting](#troubleshooting)
11. [Test Development](#test-development)

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 2.5.1+ with CUDA support (recommended)
- **CUDA Toolkit**: 11.8+ (if using CUDA features)
- **GPU**: NVIDIA GPU with compute capability >= 7.0 (optional but recommended)

### Development Dependencies
```bash
pip install pytest pytest-cov pytest-xdist
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Project Setup
```bash
# Clone and setup AutoVoice
git clone <repository-url>
cd autovoice

# Install in development mode
pip install -e .

# Verify CUDA setup (if applicable)
./scripts/check_cuda_toolkit.sh
```

## CUDA Environment Setup

### Automated CUDA Installation (Ubuntu/Debian)
```bash
# Install complete CUDA environment
./scripts/install_cuda_toolkit.sh

# For GPU-only components, use:
./scripts/install_cuda_toolkit.sh --no-drivers --no-pytorch

# For custom CUDA version:
./scripts/install_cuda_toolkit.sh --cuda-version 12.1
```

### Manual CUDA Installation
If automated installation is not suitable:

1. **Install NVIDIA Drivers**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-driver-470
   sudo reboot
   ```

2. **Install CUDA Toolkit**
   ```bash
   # Download from: https://developer.nvidia.com/cuda-toolkit
   # OR use package manager
   sudo apt install cuda-toolkit-11-8
   ```

3. **Configure Environment**
   ```bash
   # Add to ~/.bashrc
   export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   source ~/.bashrc
   ```

### Verification
```bash
# Verify complete setup
./scripts/check_cuda_toolkit.sh

# Test PyTorch CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Test Suite Structure

### Test Categories and Markers

| Category | Marker | Description | Typical Duration |
|----------|--------|-------------|------------------|
| **Smoke Tests** | `-m unit -k smoke` | Basic functionality validation | < 10 seconds |
| **Unit Tests** | `-m unit` | Component isolation tests | < 5 minutes |
| **Integration Tests** | `-m integration` | Component interaction tests | < 15 minutes |
| **End-to-End Tests** | `-m e2e` | Complete workflow tests | < 30 minutes |
| **Performance Tests** | `-m performance` | Benchmarking and optimization | 5-60 minutes |
| **CUDA Tests** | `-m cuda` | GPU-specific functionality | Variable |
| **Slow Tests** | `-m slow` | Long-running or resource intensive | Variable |

### CUDA Test Suite

#### **CUDA Smoke Tests** (`test_bindings_smoke.py`)
- **Purpose**: Validate basic CUDA bindings functionality
- **Coverage**: Module import, function exposure, basic execution
- **Requirements**: CUDA toolkit and PyTorch CUDA

#### **CUDA Integration Tests** (`test_bindings_integration.py`)
- **Purpose**: Test real-world CUDA scenarios
- **Coverage**: Multi-frequency processing, noise robustness, memory management
- **Requirements**: CUDA GPU, comprehensive test data

#### **CUDA Performance Tests** (`test_bindings_performance.py`)
- **Purpose**: Benchmark CUDA vs CPU performance
- **Coverage**: Throughput, latency, memory usage, scaling
- **Requirements**: CUDA GPU, CPU reference implementation

## Quick Start

### Automated Build and Test
```bash
# Run complete validation (recommended)
./scripts/build_and_test.sh
```

### Basic Test Execution
```bash
# Run all tests (requires CUDA for GPU tests)
pytest tests/

# Run CPU-only tests (safe for all systems)
pytest tests/ -m "not cuda"

# Run fast tests only
pytest tests/ -m "not slow"
```

### CUDA-Specific Testing
```bash
# Run CUDA smoke tests only
pytest tests/test_bindings_smoke.py -v

# Run all CUDA tests
pytest tests/ -m "cuda" -v

# CUDA performance benchmarks
pytest tests/test_bindings_performance.py -v -s
```

## Detailed Test Execution

### Test Selection Strategies

#### By Test Category
```bash
# Unit tests only
pytest tests/ -m "unit and not slow" -v

# Integration tests
pytest tests/ -m "integration and not slow" -v

# Performance benchmarks
pytest tests/ -m "performance" -v --durations=0

# End-to-end workflows
pytest tests/ -m "e2e" -v
```

#### By Test File
```bash
# Specific test files
pytest tests/test_models.py -v
pytest tests/test_inference.py -v
pytest tests/test_gpu_manager.py -v

# Multiple specific files
pytest tests/test_config.py tests/test_utils.py -v
```

#### By Functionality
```bash
# Audio processing tests
pytest tests/ -m "audio" -v

# Model tests
pytest tests/ -m "model" -v

# Web interface tests
pytest tests/ -m "web" -v
```

### Execution Options

#### Verbose Output
```bash
# Detailed output with timing
pytest tests/ -v --durations=10

# Show all output including prints
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -v -l
```

#### Failure Handling
```bash
# Stop on first failure
pytest tests/ -x

# Continue but limit failures shown
pytest tests/ --maxfail=5

# Show full tracebacks
pytest tests/ --tb=long
```

#### Parallel Execution
```bash
# Parallel execution (auto-detect cores)
pytest tests/ -n auto

# Specific number of workers
pytest tests/ -n 4
```

## CUDA-Specific Testing

### CUDA Test Prerequisites
```bash
# Verify CUDA environment
./scripts/check_cuda_toolkit.sh

# Verify PyTorch CUDA support
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Verify GPU status
nvidia-smi
```

###CUDA Smoke Tests
```bash
# Run all smoke tests
pytest tests/test_bindings_smoke.py -v

# Specific smoke test categories
pytest tests/test_bindings_smoke.py::test_function_callable -v
pytest tests/test_bindings_smoke.py::test_input_validation -v
pytest tests/test_bindings_smoke.py::test_boundary_values -v
```

### CUDA Integration Tests
```bash
# Complete integration test suite
pytest tests/test_bindings_integration.py -v -s

# Specific integration scenarios
pytest tests/test_bindings_integration.py::TestCUDABindingsIntegration::test_pitch_detection_sine_wave -v
pytest tests/test_bindings_integration.py::TestCUDABindingsIntegration::test_noise_robustness -v
pytest tests/test_bindings_integration.py::TestCUDABindingsIntegration::test_memory_consistency -v
```

### CUDA Performance Benchmarks
```bash
# Full performance suite
pytest tests/test_bindings_performance.py -v -s --durations=0

# CUDA vs CPU comparison
pytest tests/test_bindings_performance.py::TestCUDABindingsPerformance::test_performance_cuda_vs_cpu -s

# Memory usage analysis
pytest tests/test_bindings_performance.py::TestCUDABindingsPerformance::test_memory_usage_scaling -s

# Latency measurements
pytest tests/test_bindings_performance.py::TestCUDABindingsPerformance::test_latency_measurement -s
```

### Advanced CUDA Testing
```bash
# GPU memory leak detection
pytest tests/test_bindings_integration.py::TestCUDABindingsIntegration::test_memory_consistency -v -s

# Multi-GPU testing (if available)
pytest tests/ -m "cuda" -k "multi_gpu" -v

# Stress testing with large datasets
pytest tests/test_bindings_smoke.py::test_stress_large_tensors -v -s
```

## Performance Benchmarking

### Benchmarking Tools
```bash
# CPU baseline (if librosa available)
pytest tests/test_bindings_performance.py::TestCUDABindingsPerformance::test_performance_cuda_vs_cpu -s

# CUDA throughput analysis
pytest tests/test_bindings_performance.py::TestCUDABindingsPerformance::test_throughput_sustained -s

# Memory scaling analysis
pytest tests/test_bindings_performance.py::TestCUDABindingsPerformance::test_memory_usage_scaling -s
```

### Performance Metrics
- **Throughput**: Real-time factor (values > 1.0 indicate real-time processing)
- **Latency**: Kernel launch time in milliseconds
- **Memory Usage**: GPU memory consumption scaling
- **Efficiency**: Operations per second, bandwidth utilization

### Benchmark Reporting
```bash
# Generate performance reports
pytest tests/ -m "performance" --benchmark-save=performance_results --benchmark-compare

# Compare against previous runs
pytest tests/ -m "performance" --benchmark-compare=0001_performance_results
```

## Coverage Analysis

### Coverage Reports
```bash
# Basic coverage report
pytest tests/ --cov=src/auto_voice --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=src/auto_voice --cov-report=html
open htmlcov/index.html

# XML coverage for CI tools
pytest tests/ --cov=src/auto_voice --cov-report=xml
```

### CUDA-Aware Coverage
```bash
# Coverage excluding CUDA tests
pytest tests/ -m "not cuda" --cov=src/auto_voice --cov-report=term

# CUDA-specific coverage
pytest tests/ -m "cuda" --cov=src/auto_voice --cov-report=term

# Combined coverage (requires CUDA)
pytest tests/ --cov=src/auto_voice --cov-report=term-missing
```

### Coverage Goals
- **Target Coverage**: 80% minimum code coverage
- **Branch Coverage**: Enabled for conditional logic
- **CUDA Kernel Coverage**: Included in overall metrics
- **Test Quality**: Measured by coverage depth, not just percentage

## CI/CD Integration

### GitHub Actions Example
```yaml
name: AutoVoice CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest]
        python-version: ["3.8", "3.9", "3.10"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run CPU tests
      run: |
        pytest tests/ -m "not cuda and not slow" --cov=src/auto_voice --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  cuda-test:
    runs-on: gpu-enabled-runner
    if: contains(github.event.pull_request.labels.*.name, 'cuda')

    steps:
    - uses: actions/checkout@v3
    - name: Set up CUDA
      run: |
        ./scripts/install_cuda_toolkit.sh --no-drivers

    - name: Run CUDA tests
      run: |
        pytest tests/ -m "cuda" --cov=src/auto_voice --cov-report=xml --cov-report-append

    - name: Upload CUDA coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Jenkins Pipeline Example
```groovy
pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -e .'
                sh './scripts/check_cuda_toolkit.sh || echo "CUDA not available"'
            }
        }
        stage('Test CPU') {
            steps {
                sh 'pytest tests/ -m "not cuda" --cov=src/auto_voice --cov-report=xml'
            }
        }
        stage('Test CUDA') {
            when {
                expression { params.RUN_CUDA_TESTS }
            }
            steps {
                sh 'pytest tests/ -m "cuda" --cov-append --cov-report=xml'
            }
        }
        stage('Coverage') {
            steps {
                publishCoverage adapters: [coberturaAdapter('coverage.xml')]
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

#### CUDA Environment Issues
```bash
# Diagnose CUDA setup
./scripts/check_cuda_toolkit.sh

# Missing CUDA toolkit
./scripts/install_cuda_toolkit.sh

# Driver issues
sudo apt install nvidia-driver-470
sudo reboot
```

#### Test Execution Problems
```bash
# Permission issues
chmod +x scripts/*.sh

# Import errors
pip install -e .

# Module not found
python -c "import torch; print(torch.cuda.is_available())"
```

#### Performance Issues
```bash
# High memory usage
pytest tests/ -m "performance" -k "memory" -s

# Slow test execution
pytest tests/ --durations=10 -k "slow"

# GPU memory errors
pytest tests/ -m "cuda" -x -v  # Stop on first failure
```

#### Coverage Problems
```bash
# Missing coverage data
pytest tests/ --cov=src/auto_voice --cov-reset

# Coverage not meeting threshold
pytest tests/ --cov=src/auto_voice --cov-fail-under=75

# Debug coverage collection
pytest tests/ --cov=src/auto_voice --cov-report=term-missing -v
```

### Debug Commands
```bash
# Verbose test discovery
pytest tests/ --collect-only -v

# Debug specific test
pytest tests/test_specific.py::TestClass::test_method -v -s --tb=long

# Show fixture information
pytest tests/ --fixtures

# List all markers
pytest tests/ --markers
```

## Test Development

### Writing New Tests

#### Basic Test Structure
```python
import pytest
import torch
from pathlib import Path


class TestMyComponent:
    """Test cases for MyComponent."""

    @pytest.fixture
    def setup_component(self):
        """Fixture for component setup."""
        # Setup code here
        yield component
        # Cleanup code here

    def test_basic_functionality(self, setup_component):
        """Test basic component functionality."""
        component = setup_component

        # Test logic here
        result = component.process(data)
        assert result is not None

    @pytest.mark.cuda
    def test_cuda_functionality(self, setup_component):
        """Test CUDA-specific functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        component = setup_component

        # CUDA-specific test logic
        cuda_tensor = torch.randn(100, device='cuda')
        result = component.process_cuda(cuda_tensor)
        assert result.device.type == 'cuda'
```

#### CUDA Test Best Practices
```python
import pytest
import torch


@pytest.mark.cuda
class TestCudaBindings:
    """CUDA-specific test examples."""

    @pytest.fixture(autouse=True)
    def cuda_check(self):
        """Ensure CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Warm up GPU
        torch.cuda.init()

    def test_tensor_operations(self):
        """Test basic CUDA tensor operations."""
        # Create tensors
        cpu_tensor = torch.randn(1000)
        cuda_tensor = cpu_tensor.cuda()

        # Verify device placement
        assert cuda_tensor.device.type == 'cuda'
        assert cuda_tensor.is_cuda

        # Perform operations
        result = cuda_tensor * 2 + 1
        assert torch.allclose(result.cpu(), cpu_tensor * 2 + 1)

    def test_memory_management(self):
        """Test CUDA memory management."""
        initial_memory = torch.cuda.memory_allocated()

        # Allocate tensor
        large_tensor = torch.randn(1000000, device='cuda')

        # Check memory usage
        current_memory = torch.cuda.memory_allocated()
        assert current_memory > initial_memory

        # Clean up
        del large_tensor
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_scaling_performance(self, size):
        """Test performance scaling with input size."""
        import time

        tensor = torch.randn(size, device='cuda')

        start_time = time.perf_counter()
        result = torch.fft.fft(tensor)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        processing_time = end_time - start_time

        # Basic performance check
        assert processing_time < 1.0  # Should complete within 1 second
```

### Test Organization Guidelines

#### File Naming Conventions
- `test_{component}.py`: Component-specific tests
- `test_bindings_{type}.py`: CUDA binding tests
- `test_{feature}_integration.py`: Integration tests
- `test_{component}_performance.py`: Performance benchmarks

#### Marker Usage
```python
# Component type markers
@pytest.mark.unit              # Fast, isolated tests
@pytest.mark.integration       # Component interaction tests
@pytest.mark.e2e               # End-to-end workflows

# Execution type markers
@pytest.mark.cuda              # Requires CUDA GPU
@pytest.mark.slow              # Long-running tests
@pytest.mark.performance       # Benchmark tests

# Feature markers
@pytest.mark.audio             # Audio processing tests
@pytest.mark.model             # Model tests
@pytest.mark.web               # Web interface tests
```

### Testing Utilities

#### Custom Fixtures (in conftest.py)
```python
@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    import numpy as np
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def cuda_device():
    """Ensure CUDA device is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def benchmark_timer():
    """Timer fixture for performance measurements."""
    import time
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"\nDuration: {end_time - start_time:.4f} seconds")
```

### Continuous Integration

#### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
cat > .pre-commit-config.yaml << EOF
repos:
- repo: local
  hooks:
  - id: pytest-check
    name: Run tests
    entry: pytest tests/ -m "not slow" --tb=short
    language: system
    pass_filenames: false
    always_run: true
EOF

# Install hooks
pre-commit install
```

## Conclusion

This test execution guide provides comprehensive coverage of the AutoVoice test suite, from basic setup to advanced CUDA testing and CI/CD integration. The enhanced testing framework with dedicated CUDA validation ensures robust, production-ready code quality.

For additional support or questions, refer to:
- `TEST_EXECUTION_REPORT.md`: Current test status and results
- `pytest.ini`: Test configuration details
- `scripts/build_and_test.sh`: Automated testing script
