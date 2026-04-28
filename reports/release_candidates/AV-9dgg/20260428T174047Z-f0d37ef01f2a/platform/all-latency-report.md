# Performance Validation Report

**Generated:** 2026-04-28T12:43:39.865314
**Platform:** Jetson Thor (CUDA 13.0, SM 11.0)

## Pipeline Comparison

| Pipeline | RTF | Latency | GPU Mem | MCD | Similarity | Target Met |
|----------|-----|---------|---------|-----|------------|------------|
| Realtime (ContentVec + HiFiGAN) | 0.210 | 15ms | 0.47GB | 237.69dB | 0.936 | PASS |
| Quality (CoMoSVC) | 0.082 | 0ms | 1.21GB | 289.43dB | 0.766 | PASS |
| Quality Seed-VC (DiT-CFM) | 0.647 | 0ms | 3.24GB | 180.57dB | 0.985 | PASS |
| Realtime MeanVC (Streaming) | 0.000 | 0ms | 0.00GB | 0.00dB | 0.000 | SKIP |

## Detailed Results

### Realtime (ContentVec + HiFiGAN)

- **Description:** Low-latency karaoke pipeline with 22kHz output
- **Audio Duration:** 10.00s
- **Processing Time:** 2.104s
- **RTF:** 0.210 (target: <0.5)
- **Latency:** 15.1ms (target: <100ms)
- **GPU Memory Peak:** 0.47GB (target: <8.0GB)
- **MCD:** 237.69dB (target: <10.0dB, gate: informational)
- **Speaker Similarity:** 0.936
- **Output Sample Rate:** 22050Hz

### Quality (CoMoSVC)

- **Description:** High-quality studio pipeline with consistency model
- **Audio Duration:** 10.00s
- **Processing Time:** 0.816s
- **RTF:** 0.082 (target: <2.0)
- **Latency:** 0.0ms (target: <3000ms)
- **GPU Memory Peak:** 1.21GB (target: <16.0GB)
- **MCD:** 289.43dB (target: <6.0dB, gate: informational)
- **Speaker Similarity:** 0.766
- **Output Sample Rate:** 24000Hz

### Quality Seed-VC (DiT-CFM)

- **Description:** SOTA quality with 10-step DiT-CFM at 44.1kHz
- **Audio Duration:** 10.00s
- **Processing Time:** 6.468s
- **RTF:** 0.647 (target: <2.0)
- **Latency:** 0.0ms (target: <2000ms)
- **GPU Memory Peak:** 3.24GB (target: <16.0GB)
- **MCD:** 180.57dB (target: <6.0dB, gate: informational)
- **Speaker Similarity:** 0.985
- **Output Sample Rate:** 44100Hz

### Realtime MeanVC (Streaming)

- **Description:** CPU-friendly experimental MeanVC streaming path with 16kHz output
- **Audio Duration:** 0.00s
- **Processing Time:** 0.000s
- **RTF:** 0.000 (target: <2.0)
- **Latency:** 0.0ms (target: <350ms)
- **GPU Memory Peak:** 0.00GB (target: <6.0GB)
- **MCD:** 0.00dB (target: <8.0dB, gate: informational)
- **Speaker Similarity:** 0.000
- **Output Sample Rate:** 16000Hz

- **SKIPPED:** MeanVC performance is experimental and disabled unless AUTOVOICE_MEANVC_FULL=1 is set.
- **Owner:** model-runtime
- **Action:** Run scripts/prepare_meanvc_assets.py in the canonical autovoice-thor environment, set AUTOVOICE_MEANVC_FULL=1, then rerun scripts/validate_cuda_stack.sh --pipeline realtime_meanvc.

## Recommendations

| Use Case | Recommended Pipeline |
|----------|---------------------|
| Live Karaoke | realtime or realtime_meanvc |
| Studio Conversion | quality_seedvc |
| Batch Processing | quality (consistency model) |
| Mobile/Edge | realtime_meanvc (CPU-friendly) |
