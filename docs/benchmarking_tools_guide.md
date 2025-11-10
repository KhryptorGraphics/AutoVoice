# Benchmarking Tools Guide

Complete guide for AutoVoice performance benchmarking suite.

## Overview

The AutoVoice benchmarking suite provides comprehensive performance and quality evaluation across multiple GPUs. All tools support the `--gpu-id` flag for multi-GPU systems.

## Quick Start

```bash
# Run complete benchmark suite on GPU 0
python scripts/run_comprehensive_benchmarks.py --gpu-id 0

# Run quick benchmarks (faster, fewer iterations)
python scripts/run_comprehensive_benchmarks.py --quick --gpu-id 0

# Run on specific GPU (e.g., GPU 1 in multi-GPU system)
python scripts/run_comprehensive_benchmarks.py --gpu-id 1
```

## Individual Tools

### 1. Pipeline Profiling

Profiles end-to-end voice conversion pipeline with stage-by-stage timing.

```bash
python scripts/profile_performance.py \
  --output-dir validation_results/my_test \
  --gpu-id 0 \
  --audio-file tests/data/benchmark/audio_30s_22050hz.wav
```

**Output:** `pipeline_profile.json` with:
- Total latency (ms)
- RTF (Real-Time Factor)
- Per-stage breakdown
- GPU memory usage
- GPU utilization

### 2. CUDA Kernel Profiling

Profiles individual CUDA kernels for low-level performance analysis.

```bash
python scripts/profile_cuda_kernels.py \
  --kernel all \
  --iterations 100 \
  --gpu-id 0 \
  --output cuda_kernels.json
```

**Kernels tested:**
- STFT/iSTFT
- Mel-spectrogram computation
- Pitch detection
- Formant extraction

**Output:** Per-kernel timing, memory, and efficiency metrics.

### 3. TTS Synthesis Benchmarking

Benchmarks text-to-speech synthesis performance.

```bash
python scripts/profile_tts.py \
  --text "Hello, this is a test of the text to speech system." \
  --gpu-id 0 \
  --output-dir validation_results/tts \
  --iterations 10
```

**Output:** `tts_profile.json` with:
- Synthesis latency (ms per 1s audio)
- Throughput (requests/second)
- Peak GPU memory (MB)

**Quick mode:**
```bash
python scripts/profile_tts.py --quick --gpu-id 0
```
(2 warmups, 5 iterations instead of 3 warmups, 10 iterations)

### 4. Quality Metrics Evaluation

Evaluates voice conversion quality.

```bash
python scripts/evaluate_quality.py \
  --source-audio tests/data/source.wav \
  --converted-audio tests/data/converted.wav \
  --target-profile profiles/speaker_001.json \
  --gpu-id 0 \
  --output-dir validation_results/quality
```

**Metrics computed:**
- **Pitch Accuracy:** RMSE in Hz between source and converted pitch contours
- **Speaker Similarity:** Cosine similarity of speaker embeddings (0-1)
- **Naturalness Score:** MOS-like score (1-5 scale)

**Output:** `quality_metrics.json`

### 5. Pytest Performance Tests

Run performance regression tests.

```bash
# Run all performance tests
pytest tests/test_performance.py -v --json-report --json-report-file=pytest_results.json

# Set custom output for metrics
PYTEST_JSON_OUTPUT=validation_results/pytest_metrics.json pytest tests/test_performance.py
```

**Tests included:**
- CPU vs GPU speedup (should be ≥3x)
- Cache effectiveness (cold start vs warm cache)
- Preset RTF validation (fast, balanced, quality)

## Multi-GPU Benchmarking

### Running on Multiple GPUs

```bash
# GPU 0 (e.g., RTX 4090)
python scripts/run_comprehensive_benchmarks.py --gpu-id 0 &

# GPU 1 (e.g., RTX 3090)
python scripts/run_comprehensive_benchmarks.py --gpu-id 1 &

# Wait for both to complete
wait
```

### Aggregating Results

```bash
python scripts/aggregate_multi_gpu_results.py \
  --input-dir validation_results/benchmarks \
  --output-file validation_results/benchmarks/multi_gpu_comparison.md
```

**Output:** Markdown comparison table with:
- TTS performance across GPUs
- Voice conversion RTF by preset
- Quality metrics consistency
- GPU memory usage

## Benchmark Modes

### Quick Mode
- 5s audio samples
- 30 iterations per test
- ~5-10 minutes per GPU

```bash
python scripts/run_comprehensive_benchmarks.py --quick
```

### Balanced Mode (Default)
- 30s audio samples
- 100 iterations per test
- ~15-30 minutes per GPU

```bash
python scripts/run_comprehensive_benchmarks.py
```

### Full Mode
- 60s audio samples
- 200 iterations per test
- ~30-60 minutes per GPU

```bash
python scripts/run_comprehensive_benchmarks.py --full
```

## Skipping Specific Benchmarks

```bash
# Skip pytest (useful for CI/CD)
python scripts/run_comprehensive_benchmarks.py --skip-pytest

# Skip CUDA kernels (CPU-only systems)
python scripts/run_comprehensive_benchmarks.py --skip-cuda-kernels

# Skip TTS benchmarking
python scripts/run_comprehensive_benchmarks.py --skip-tts

# Skip quality evaluation
python scripts/run_comprehensive_benchmarks.py --skip-quality

# Combine flags
python scripts/run_comprehensive_benchmarks.py --skip-pytest --skip-quality --quick
```

## Output Structure

```
validation_results/
└── benchmarks/
    ├── NVIDIA_RTX_4090/
    │   ├── benchmark_summary.json       # Aggregated metrics
    │   ├── benchmark_report.md          # Human-readable report
    │   ├── gpu_info.json                # GPU hardware info
    │   ├── pipeline_profile.json        # Pipeline timings
    │   ├── cuda_kernels_profile.json    # Kernel profiling
    │   ├── tts_profile.json             # TTS benchmarks
    │   ├── quality_metrics.json         # Quality evaluation
    │   ├── pytest_results.json          # Pytest output
    │   └── pytest_metrics.json          # Pytest performance metrics
    ├── NVIDIA_RTX_3090/
    │   └── ...
    └── multi_gpu_comparison.md          # Cross-GPU comparison
```

## Understanding Results

### Real-Time Factor (RTF)

RTF indicates how fast conversion runs relative to audio duration:

- **RTF < 1.0:** Faster than real-time (30s song takes <30s)
- **RTF = 1.0:** Real-time (30s song takes 30s)
- **RTF > 1.0:** Slower than real-time (30s song takes >30s)

**Examples:**
- RTF 0.5x → 30s song converts in 15s ✓ (good for production)
- RTF 2.0x → 30s song converts in 60s (acceptable for batch processing)

### Quality Metrics Interpretation

**Pitch Accuracy (RMSE):**
- < 10 Hz: Excellent
- 10-20 Hz: Good
- > 20 Hz: Needs improvement

**Speaker Similarity:**
- > 0.85: Excellent match
- 0.70-0.85: Good match
- < 0.70: Poor match

**Naturalness Score:**
- 4.0-5.0: Excellent quality
- 3.0-4.0: Good quality
- < 3.0: Poor quality

## Best Practices

1. **Warm-up your GPU** before benchmarking:
   ```bash
   # Run quick benchmark first to warm up
   python scripts/run_comprehensive_benchmarks.py --quick
   # Then run full benchmark
   python scripts/run_comprehensive_benchmarks.py
   ```

2. **Close other GPU applications** during benchmarking for accurate results.

3. **Use consistent test data** across runs:
   ```bash
   python scripts/generate_benchmark_test_data.py
   ```

4. **Run multiple times** and average results for production decisions.

5. **Document your environment:**
   - GPU driver version
   - CUDA version
   - PyTorch version
   - System load during testing

## Troubleshooting

### "Test audio file not found"

Generate benchmark test data first:
```bash
python scripts/generate_benchmark_test_data.py
```

### "CUDA out of memory"

Use smaller batch size or quick mode:
```bash
python scripts/run_comprehensive_benchmarks.py --quick
```

### "CUDA device not available"

Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check GPU visibility:
```bash
nvidia-smi
```

### Wrong GPU being used

Explicitly set GPU:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_comprehensive_benchmarks.py --gpu-id 0
```
(This uses physical GPU 1, referred to as logical GPU 0)

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run quick benchmarks
        run: |
          python scripts/run_comprehensive_benchmarks.py \
            --quick \
            --gpu-id 0 \
            --skip-quality

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: validation_results/
```

## Performance Targets

Based on empirical testing, target performance metrics:

| Metric | Target (Production) | Acceptable (Development) |
|--------|-------------------|-------------------------|
| Voice Conversion RTF (Balanced) | < 1.5x | < 3.0x |
| TTS Latency (1s audio) | < 100ms | < 200ms |
| GPU vs CPU Speedup | > 5x | > 3x |
| Cache Speedup | > 3x | > 2x |
| Pitch Accuracy | < 12 Hz | < 20 Hz |
| Speaker Similarity | > 0.85 | > 0.70 |

## Additional Resources

- [Performance Benchmarking Guide](performance_benchmarking_guide.md)
- [Multi-GPU Comparison Results](../validation_results/benchmarks/multi_gpu_comparison.md)
- [README Performance Section](../README.md#-performance-benchmarks)
- [Production Readiness Checklist](production_readiness_checklist.md)

---

**Note:** All benchmarks automatically export `CUDA_VISIBLE_DEVICES` based on `--gpu-id` to ensure correct GPU selection in multi-GPU systems.
