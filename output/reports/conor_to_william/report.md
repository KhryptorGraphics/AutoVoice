# Quality Pipeline Report: source

Generated: 2026-04-18T17:05:57.116792+00:00

## Quality Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| F0_RMSE | 1287.850 | 0.000 | 1287.850 | 1287.850 |
| LATENCY_MS | 104791.051 | 0.000 | 104791.051 | 104791.051 |
| MCD | 95.046 | 0.000 | 95.046 | 95.046 |
| MOS_PRED | 5.000 | 0.000 | 5.000 | 5.000 |
| PESQ | 1.140 | 0.000 | 1.140 | 1.140 |
| PITCH_CORR | 0.484 | 0.000 | 0.484 | 0.484 |
| SNR | -34.081 | 0.000 | -34.081 | -34.081 |
| SPEAKER_SIMILARITY | 0.998 | 0.000 | 0.998 | 0.998 |
| STOI | 0.148 | 0.000 | 0.148 | 0.148 |

## Quality Targets

| Metric | Target | Status |
|--------|--------|--------|
| mcd | < 5.0 dB | FAIL |
| f0_rmse | < 20 cents | FAIL |
| pitch_corr | > 0.90 | FAIL |
| speaker_similarity | >= 0.85 | PASS |
| pesq | >= 3.5 | FAIL |
| stoi | >= 0.85 | FAIL |