# AutoVoice Benchmarking Quick Reference

## 🚀 One-Line Commands

```bash
# Complete benchmark suite
python scripts/run_comprehensive_benchmarks.py --gpu-id 0

# Quick benchmarks (5-10 min)
python scripts/run_comprehensive_benchmarks.py --quick --gpu-id 0

# Multi-GPU benchmark (parallel)
for gpu in 0 1; do python scripts/run_comprehensive_benchmarks.py --gpu-id $gpu & done; wait

# Aggregate multi-GPU results
python scripts/aggregate_multi_gpu_results.py
```

## 📊 Individual Tools

| Tool | Command | Output |
|------|---------|--------|
| **Pipeline Profiling** | `python scripts/profile_performance.py --gpu-id 0` | `pipeline_profile.json` |
| **CUDA Kernels** | `python scripts/profile_cuda_kernels.py --kernel all --gpu-id 0` | `cuda_kernels_profile.json` |
| **TTS Benchmark** | `python scripts/profile_tts.py --gpu-id 0` | `tts_profile.json` |
| **Quality Metrics** | `python scripts/evaluate_quality.py --source-audio src.wav --converted-audio out.wav` | `quality_metrics.json` |
| **Pytest Tests** | `PYTEST_JSON_OUTPUT=metrics.json pytest tests/test_performance.py` | `pytest_metrics.json` |

## 🎯 Common Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--gpu-id N` | Use GPU N | `--gpu-id 1` |
| `--quick` | Fast mode (fewer iterations) | `--quick` |
| `--full` | Full mode (more iterations) | `--full` |
| `--skip-pytest` | Skip pytest tests | `--skip-pytest` |
| `--skip-cuda-kernels` | Skip CUDA profiling | `--skip-cuda-kernels` |
| `--skip-tts` | Skip TTS benchmarking | `--skip-tts` |
| `--skip-quality` | Skip quality evaluation | `--skip-quality` |
| `--output-dir DIR` | Output directory | `--output-dir results/` |

## 📈 Benchmark Modes

| Mode | Duration | Audio Length | Iterations | Use Case |
|------|----------|--------------|------------|----------|
| **Quick** | 5-10 min | 5s | 30 | Development, CI/CD |
| **Balanced** | 15-30 min | 30s | 100 | Standard benchmarking |
| **Full** | 30-60 min | 60s | 200 | Production validation |

## 🔍 Understanding Metrics

### Real-Time Factor (RTF)
- **0.5x** = Processes 2x faster than real-time ✓
- **1.0x** = Exactly real-time
- **2.0x** = Processes 2x slower than real-time

### Quality Metrics
- **Pitch Accuracy:** < 10 Hz = Excellent
- **Speaker Similarity:** > 0.85 = Excellent
- **Naturalness:** > 4.0/5.0 = Excellent

## 🛠️ Setup & Prerequisites

```bash
# Generate test data (first time only)
python scripts/generate_benchmark_test_data.py

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi
```

## 📁 Output Structure

```
validation_results/benchmarks/
├── NVIDIA_RTX_4090/
│   ├── benchmark_summary.json      ← Aggregated metrics
│   ├── pipeline_profile.json       ← Pipeline timings
│   ├── tts_profile.json            ← TTS benchmarks
│   └── quality_metrics.json        ← Quality scores
└── multi_gpu_comparison.md         ← Cross-GPU comparison
```

## ⚡ Performance Targets

| Metric | Production | Development |
|--------|-----------|-------------|
| Voice Conv RTF (Balanced) | < 1.5x | < 3.0x |
| TTS Latency (1s) | < 100ms | < 200ms |
| CPU→GPU Speedup | > 5x | > 3x |
| Pitch Accuracy | < 12 Hz | < 20 Hz |

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Test data not found | `python scripts/generate_benchmark_test_data.py` |
| CUDA out of memory | Add `--quick` flag or reduce batch size |
| Wrong GPU used | Set `CUDA_VISIBLE_DEVICES=N` or `--gpu-id N` |
| Stale results | Delete `validation_results/` and re-run |

## 📚 Full Documentation

- [Detailed Benchmarking Guide](benchmarking_tools_guide.md)
- [Performance Guide](performance_benchmarking_guide.md)
- [README Benchmarks](../README.md#-performance-benchmarks)

---

**Pro Tip:** Use `--quick` during development and `--full` before production deployment.
