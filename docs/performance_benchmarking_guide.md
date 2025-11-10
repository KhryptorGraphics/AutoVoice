# Performance Benchmarking Guide

Comprehensive guide for running benchmarks, understanding results, and reproducing measurements on different hardware.

## Overview

### Purpose

This guide provides detailed instructions for:
- Running performance benchmarks on AutoVoice
- Understanding benchmark metrics and their significance
- Reproducing measurements across different hardware configurations
- Comparing performance across multiple GPUs

### Metrics Measured

**TTS Performance**:
- Synthesis latency (time to generate 1s of audio)
- Throughput (requests per second)
- GPU memory usage

**Voice Conversion Performance**:
- Real-time factor (RTF) across quality presets
- GPU memory peak usage
- CPU vs GPU speedup

**Quality Metrics**:
- Pitch accuracy (RMSE in Hz)
- Speaker similarity (cosine similarity)
- Naturalness score (subjective quality)

**CUDA Kernel Performance**:
- Per-kernel execution time
- Speedup vs CPU reference implementations

### Target Hardware

- **Primary**: NVIDIA GPUs with compute capability 7.0+ (Volta, Turing, Ampere, Ada Lovelace)
- **Tested**: RTX 3080 Ti, A100 (40GB), RTX 3090, RTX 4090, T4
- **Minimum**: 6GB VRAM (8GB recommended for quality preset)

## Prerequisites

### Hardware Requirements

**GPU**:
- NVIDIA GPU with compute capability 7.0 or higher
- 6GB VRAM minimum (8GB recommended)
- CUDA 11.8 or later support

**CPU**:
- Multi-core CPU for CPU baseline benchmarks
- 16GB RAM minimum (32GB recommended)

**Storage**:
- 10GB free space for test data and results

### Software Requirements

**Required**:
- Python 3.12 (3.8+ supported)
- PyTorch 2.5.1+cu121 (or compatible version)
- CUDA 12.1 (or compatible version)
- NVIDIA driver 535+ for CUDA 12.1

**Python Packages**:
```bash
# Core dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r requirements.txt

# Build CUDA extensions
pip install -e .
```

**Testing Tools**:
```bash
# Install pytest with JSON reporting
pip install pytest pytest-json-report pytest-timeout
```

### Environment Setup

**1. Create conda environment** (recommended):
```bash
conda create -n autovoice_bench python=3.12 -y
conda activate autovoice_bench
```

**2. Install PyTorch with CUDA**:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

**3. Install AutoVoice**:
```bash
cd /path/to/autovoice
pip install -r requirements.txt
pip install -e .
```

**4. Verify CUDA extensions**:
```bash
python scripts/verify_bindings.py
```

Expected output:
```
✓ CUDA extensions built successfully
✓ All kernels available
```

## Quick Start

### Three-Step Benchmark

```bash
# 1. Generate test data
python scripts/generate_benchmark_test_data.py

# 2. Run comprehensive benchmarks
python scripts/run_comprehensive_benchmarks.py

# 3. View results
cat validation_results/benchmarks/benchmark_report.md
```

### Expected Duration

- Test data generation: 1-2 minutes
- Pytest performance tests: 15-30 minutes
- Pipeline profiling: 5-10 minutes
- CUDA kernel profiling: 5-10 minutes
- **Total**: 30-60 minutes for full benchmark suite

## Detailed Benchmarking Workflow

### Step 1: Test Data Generation

Generate audio files and voice profiles for benchmarking:

```bash
python scripts/generate_benchmark_test_data.py --output-dir tests/data/benchmark
```

**Generated Files**:
```
tests/data/benchmark/
  ├── audio_1s_22050hz.wav      # 1-second audio for TTS latency
  ├── audio_5s_22050hz.wav      # 5-second audio for quick tests
  ├── audio_10s_22050hz.wav     # 10-second audio
  ├── audio_30s_22050hz.wav     # 30-second audio (standard benchmark)
  ├── audio_60s_22050hz.wav     # 60-second audio (stress test)
  ├── audio_30s_44100hz.wav     # 44.1kHz variant
  ├── profiles/
  │   ├── test_profile_1.json   # Synthetic voice profile 1
  │   └── test_profile_2.json   # Synthetic voice profile 2
  └── metadata.json             # Test data metadata
```

**Options**:
```bash
# Custom durations
python scripts/generate_benchmark_test_data.py --durations 1,5,30

# Custom sample rates
python scripts/generate_benchmark_test_data.py --sample-rates 22050,44100,48000

# Skip voice profiles
python scripts/generate_benchmark_test_data.py --no-profiles

# Custom output directory
python scripts/generate_benchmark_test_data.py --output-dir /path/to/data
```

### Step 2: Environment Validation

Verify environment before running benchmarks:

**Check CUDA availability**:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Check GPU model**:
```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

**Check CUDA extensions**:
```bash
python scripts/verify_bindings.py
```

**Verify test data**:
```bash
ls -lh tests/data/benchmark/
```

### Step 3: Pytest Performance Tests

Run comprehensive performance test suite:

**Full test suite**:
```bash
pytest tests/test_performance.py -v
```

**Specific test classes**:
```bash
# CPU vs GPU benchmarks
pytest tests/test_performance.py::TestCPUvsGPUBenchmarks -v

# Latency benchmarks
pytest tests/test_performance.py::TestLatencyBenchmarks -v

# Memory benchmarks
pytest tests/test_performance.py::TestMemoryBenchmarks -v

# Cache benchmarks
pytest tests/test_performance.py::TestCacheBenchmarks -v
```

**Generate JSON report**:
```bash
pytest tests/test_performance.py -v \
  --json-report \
  --json-report-file=validation_results/benchmarks/pytest_results.json
```

**Quick benchmarks only**:
```bash
pytest tests/test_performance.py -v -k "not slow"
```

**Expected Duration**: 15-30 minutes for full suite

### Step 4: Pipeline Profiling

Profile voice conversion pipeline performance:

```bash
python scripts/profile_performance.py
```

**Output**: `validation_results/performance_breakdown.json`

**Metrics Captured**:
- Stage-by-stage timing (preprocessing, pitch detection, conversion, postprocessing)
- GPU utilization per stage
- Memory usage (peak and average)
- Total pipeline latency

**Custom options**:
```bash
# Custom audio file
python scripts/profile_performance.py --audio-file /path/to/audio.wav

# Custom output directory
python scripts/profile_performance.py --output-dir /path/to/results

# Custom profile ID
python scripts/profile_performance.py --profile-id my_profile
```

**Expected Duration**: 5-10 minutes

### Step 5: CUDA Kernel Profiling

Profile individual CUDA kernels:

**All kernels**:
```bash
python scripts/profile_cuda_kernels.py --kernel all --iterations 100
```

**Specific kernel**:
```bash
# Pitch detection
python scripts/profile_cuda_kernels.py --kernel pitch_detection --iterations 50

# STFT
python scripts/profile_cuda_kernels.py --kernel stft --iterations 100

# Mel spectrogram
python scripts/profile_cuda_kernels.py --kernel mel_spectrogram --iterations 100
```

**Custom output**:
```bash
python scripts/profile_cuda_kernels.py \
  --kernel all \
  --iterations 100 \
  --output validation_results/benchmarks/cuda_kernels_profile.json
```



## Metrics Explained

### TTS Synthesis Latency

**Definition**: Time to synthesize 1 second of audio

**Target**: <100ms on GPU

**Measurement**: Average of 10 runs after 3 warmup iterations

**Significance**: Lower is better. Affects real-time responsiveness for interactive applications.

**Interpretation**:
- <50ms: Excellent - suitable for real-time interactive applications
- 50-100ms: Good - acceptable for most use cases
- 100-200ms: Fair - noticeable delay but usable
- >200ms: Poor - consider faster GPU or optimization

### Voice Conversion RTF (Real-Time Factor)

**Definition**: Conversion time / audio duration

**Target**: ~1.0x (30s song takes 30s to convert)

**Measurement**: Average of 5 runs on 30s audio

**Significance**:
- <1.0x: Faster than real-time (can process audio faster than playback)
- 1.0x: Real-time (conversion keeps pace with playback)
- >1.0x: Slower than real-time (conversion takes longer than playback)

**Interpretation**:
- <0.5x: Excellent - can process multiple streams simultaneously
- 0.5-1.0x: Good - suitable for real-time conversion
- 1.0-2.0x: Fair - acceptable for offline processing
- >2.0x: Poor - consider faster preset or better GPU

### GPU Memory Peak

**Definition**: Maximum VRAM usage during conversion

**Target**: 2-4GB for TTS, 4-8GB for voice conversion

**Measurement**: `torch.cuda.max_memory_allocated()` during conversion

**Significance**: Determines batch size and concurrent request capacity

**Interpretation**:
- <4GB: Excellent - can run multiple instances or large batches
- 4-6GB: Good - suitable for single-stream processing
- 6-8GB: Fair - may limit batch size
- >8GB: Poor - may require GPU with more VRAM

### CPU vs GPU Speedup

**Definition**: CPU time / GPU time

**Target**: 10-50x

**Measurement**: Run same conversion on CPU and GPU, compute ratio

**Significance**: Justifies GPU investment. Higher is better.

**Interpretation**:
- >30x: Excellent - GPU provides significant acceleration
- 15-30x: Good - GPU is worthwhile investment
- 5-15x: Fair - GPU helps but not dramatically
- <5x: Poor - check if CUDA extensions are working

### Pitch Accuracy (RMSE)

**Definition**: Root mean square error of pitch detection in Hz

**Target**: <10 Hz (imperceptible to listeners)

**Measurement**: Compare detected pitch to ground truth

**Significance**: Lower is better. Affects voice quality and naturalness.

**Interpretation**:
- <5 Hz: Excellent - imperceptible pitch errors
- 5-10 Hz: Good - acceptable for most applications
- 10-20 Hz: Fair - noticeable but usable
- >20 Hz: Poor - significant pitch artifacts

### Speaker Similarity

**Definition**: Cosine similarity of speaker embeddings (percentage)

**Target**: >85%

**Measurement**: Compare converted voice embedding to target profile

**Significance**: Higher is better. Affects voice fidelity and identity preservation.

**Interpretation**:
- >90%: Excellent - very high voice fidelity
- 85-90%: Good - acceptable voice similarity
- 75-85%: Fair - recognizable but noticeable differences
- <75%: Poor - voice identity not well preserved

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in configuration
2. Use shorter audio clips for testing
3. Clear CUDA cache: `torch.cuda.empty_cache()`
4. Close other GPU processes
5. Use GPU with more VRAM

### Tests Timeout

**Symptoms**: Pytest tests timeout before completion

**Solutions**:
1. Increase pytest timeout: `pytest --timeout=600`
2. Run tests individually: `pytest tests/test_performance.py::TestClass::test_name`
3. Use `--quick` flag for faster benchmarks
4. Check GPU utilization (should be >70%)

### Missing Test Data

**Symptoms**: `FileNotFoundError: test_song.wav not found`

**Solutions**:
1. Generate test data: `python scripts/generate_benchmark_test_data.py`
2. Verify test data directory: `ls tests/data/benchmark/`
3. Check metadata file: `cat tests/data/benchmark/metadata.json`

### CUDA Extensions Not Built

**Symptoms**: `ImportError: cannot import name 'cuda_kernels'`

**Solutions**:
1. Build extensions: `pip install -e .`
2. Verify build: `python scripts/verify_bindings.py`
3. Check CUDA toolkit: `nvcc --version`
4. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Performance Lower Than Expected

**Symptoms**: Benchmarks significantly slower than documented

**Diagnostics**:
1. Check GPU utilization: `nvidia-smi dmon -s pucvmet`
   - Should be >70% during conversion
2. Check thermal throttling: `nvidia-smi -q -d TEMPERATURE`
   - GPU should be <85°C
3. Check power limit: `nvidia-smi -q -d POWER`
   - Should be at or near max power limit
4. Check for other GPU processes: `nvidia-smi`
   - No other processes should be using GPU

**Solutions**:
1. Ensure GPU is not thermal throttling (improve cooling)
2. Increase power limit if possible: `nvidia-smi -pl <watts>`
3. Close other GPU applications
4. Run on idle system
5. Check if CPU is bottleneck (upgrade CPU or reduce preprocessing)

## Interpreting Results

### Good Performance Indicators

✅ **TTS latency <100ms** - Suitable for real-time applications

✅ **Voice conversion RTF <1.5x** - Acceptable for most use cases

✅ **GPU utilization >70%** - GPU is being effectively used

✅ **CPU vs GPU speedup >10x** - GPU provides significant acceleration

✅ **Pitch accuracy <10 Hz RMSE** - Imperceptible pitch errors

✅ **Speaker similarity >85%** - High voice fidelity

### Performance Issues

⚠️ **TTS latency >200ms** - Check GPU utilization, thermal throttling

⚠️ **Voice conversion RTF >3x** - Consider faster preset or better GPU

⚠️ **GPU utilization <50%** - Bottleneck elsewhere (CPU, I/O, memory)

⚠️ **CPU vs GPU speedup <5x** - CUDA extensions may not be working

⚠️ **Pitch accuracy >20 Hz** - Check pitch detection configuration

⚠️ **Speaker similarity <75%** - Check voice profile quality

## Benchmark Reproducibility

### Factors Affecting Results

**Hardware**:
- GPU model and VRAM capacity
- GPU temperature and thermal state
- Power limit settings
- CPU model (affects CPU baseline)

**Software**:
- CUDA version and driver version
- PyTorch version
- Python version
- Operating system

**Environment**:
- System load (other processes)
- GPU clock speeds
- Memory bandwidth
- PCIe bandwidth

### Ensuring Reproducibility

**1. Use same software versions**:
```bash
# Record versions
python --version
python -c "import torch; print(torch.__version__)"
nvcc --version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

**2. Run on idle system**:
```bash
# Check for other GPU processes
nvidia-smi

# Close unnecessary applications
```

**3. Allow GPU to cool between runs**:
```bash
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# Wait for temperature <60°C before benchmarking
```

**4. Use same test data**:
```bash
# Use consistent audio files
ls -lh tests/data/benchmark/

# Verify metadata
cat tests/data/benchmark/metadata.json
```

**5. Report all environment details**:
- Include GPU model, VRAM, compute capability
- Include CUDA version, driver version
- Include PyTorch version, Python version
- Include benchmark date and time
- Include any non-default settings

## Appendix: Benchmark Commands Reference

### Test Data Generation

```bash
# Standard test data
python scripts/generate_benchmark_test_data.py

# Custom durations
python scripts/generate_benchmark_test_data.py --durations 1,5,10,30,60

# Custom sample rates
python scripts/generate_benchmark_test_data.py --sample-rates 22050,44100

# Skip profiles
python scripts/generate_benchmark_test_data.py --no-profiles
```

### Comprehensive Benchmarks

```bash
# Full benchmark suite
python scripts/run_comprehensive_benchmarks.py

# Quick benchmarks only
python scripts/run_comprehensive_benchmarks.py --quick

# Custom output directory
python scripts/run_comprehensive_benchmarks.py --output-dir /path/to/results

# Specific GPU
CUDA_VISIBLE_DEVICES=1 python scripts/run_comprehensive_benchmarks.py --gpu-id 1

# Skip specific benchmarks
python scripts/run_comprehensive_benchmarks.py --skip-pytest
python scripts/run_comprehensive_benchmarks.py --skip-profiling
python scripts/run_comprehensive_benchmarks.py --skip-cuda-kernels
```

### Pytest Performance Tests

```bash
# Full test suite
pytest tests/test_performance.py -v

# Specific test class
pytest tests/test_performance.py::TestCPUvsGPUBenchmarks -v

# Quick tests only
pytest tests/test_performance.py -v -k "not slow"

# With JSON output
pytest tests/test_performance.py --json-report --json-report-file=results.json

# With timeout
pytest tests/test_performance.py --timeout=600
```

### Pipeline Profiling

```bash
# Standard profiling
python scripts/profile_performance.py

# Custom audio file
python scripts/profile_performance.py --audio-file /path/to/audio.wav

# Custom output directory
python scripts/profile_performance.py --output-dir /path/to/results
```

### CUDA Kernel Profiling

```bash
# All kernels
python scripts/profile_cuda_kernels.py --kernel all --iterations 100

# Specific kernel
python scripts/profile_cuda_kernels.py --kernel pitch_detection --iterations 50

# Custom output
python scripts/profile_cuda_kernels.py --kernel all --output results.json
```

### Multi-GPU Aggregation

```bash
# Aggregate results
python scripts/aggregate_multi_gpu_results.py

# Custom input directory
python scripts/aggregate_multi_gpu_results.py --input-dir /path/to/benchmarks

# JSON only
python scripts/aggregate_multi_gpu_results.py --format json

# With chart data
python scripts/aggregate_multi_gpu_results.py --include-charts
```

## Additional Resources

- [README Performance Section](../README.md#-performance-benchmarks)
- [Multi-GPU Comparison Results](../validation_results/benchmarks/multi_gpu_comparison.md)
- [Raw Benchmark Data](../validation_results/benchmarks/)
- [Performance Profiling Implementation](performance_profiling_implementation.md)
- [Production Readiness Checklist](production_readiness_checklist.md)

