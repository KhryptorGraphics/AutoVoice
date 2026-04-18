# Quality Pipeline Report: source

Generated: 2026-04-18T16:31:45.946042+00:00

## Quality Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| F0_RMSE | 2381.010 | 0.000 | 2381.010 | 2381.010 |
| LATENCY_MS | 115915.901 | 0.000 | 115915.901 | 115915.901 |
| MCD | 117.814 | 0.000 | 117.814 | 117.814 |
| MOS_PRED | 5.000 | 0.000 | 5.000 | 5.000 |
| PESQ | 1.129 | 0.000 | 1.129 | 1.129 |
| PITCH_CORR | -0.984 | 0.000 | -0.984 | -0.984 |
| SNR | -38.063 | 0.000 | -38.063 | -38.063 |
| SPEAKER_SIMILARITY | 0.996 | 0.000 | 0.996 | 0.996 |
| STOI | 0.047 | 0.000 | 0.047 | 0.047 |

## Quality Targets

| Metric | Target | Status |
|--------|--------|--------|
| mcd | < 5.0 dB | FAIL |
| f0_rmse | < 20 cents | FAIL |
| pitch_corr | > 0.90 | FAIL |
| speaker_similarity | >= 0.85 | PASS |
| pesq | >= 3.5 | FAIL |
| stoi | >= 0.85 | FAIL |