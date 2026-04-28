# Production Readiness Closeout - 2026-04-28

This closeout records the current production-readiness proof for `main` after running the verification matrix in the `autovoice-thor` conda environment.

## Environment

- Python: `/home/kp/anaconda3/envs/autovoice-thor/bin/python` (`3.12.12`)
- Required command prefix: `PYTHONNOUSERSITE=1 PYTHONPATH=src`
- GPU: NVIDIA Thor, CUDA `13.0`, TensorRT `10.13.3.9`
- Hosted URL: `https://autovoice.giggahost.com`

## Implemented

- Hosted preflight validation now checks enabled Apache vhosts, canonical `ServerName`, Apache `configtest`, DNS, TLS, secrets, and Jetson presence.
- Release-candidate and benchmark validators now fail when `release_evidence.quality_gate_passed` is false.
- Benchmark dashboard generation can consume the real comprehensive pipeline report instead of only pre-built benchmark bundles.
- Comprehensive pipeline benchmarks now include `pitch_corr` in addition to RTF, latency, VRAM, speaker similarity, and MCD.
- Live Playwright tests now launch the backend with `AUTOVOICE_PYTHON`, defaulting to the Thor conda env, instead of leaking into base Anaconda Python.
- The live conversion browser test was updated to exercise the current dual-upload workflow.

## Verification Results

- GitNexus refresh: passed. `npx gitnexus analyze` refreshed the graph to `19,325` nodes and `50,413` edges.
- Hosted preflight: passed. `reports/platform/hosted-preflight.json` shows canonical vhost, Apache configtest, DNS, TLS, secrets, and Jetson checks green for `autovoice.giggahost.com`.
- Hosted health/readiness/metrics: passed. `/api/v1/health`, `/ready`, and `/api/v1/metrics` respond through HTTPS.
- Frontend lint/type/build: passed. Lint still reports warnings only.
- Frontend mocked E2E: passed, `12/12`.
- Frontend live E2E: passed, `3/3`.
- Targeted platform/reporting/release tests: passed, `20/20`.
- CUDA/TensorRT dependency audit: passed in `reports/platform/dependency-audit.json`.
- Real compose completion matrix: stack boot/config lanes passed, but overall failed because benchmark validation and release-candidate evidence validation failed.
- Full pytest: failed after completing collection/execution. Result was `4434 passed`, `21 failed`, `20 errors`, `28 skipped`, `1 xfailed`.

## Production Blockers

- `AV-4ylr.6`: Current Thor benchmark evidence fails the quality gate. `realtime` and `quality_seedvc` report `pitch_corr_mean=0.0` and MCD far above target.
- `AV-4ylr.7`: Training lifecycle tests fail with feature shape mismatches, including `pitch_dim=256 expected 768`.
- `AV-4ylr.8`: Voice profile storage tests use invalid audio fixtures that now fail sample quality analysis.
- `AV-4ylr.9`: HQ-SVC tests require `fairseq`; the dependency is missing or the lane needs an explicit production gate.
- `AV-4ylr.10`: MeanVC/performance lanes are not green on Thor; the CUDA validation report marks several runtime paths as failed.

## Status

The project is not production-ready yet. The application, hosted deployment, frontend flows, compose stack, and CUDA dependency lane have meaningful proof, but production release must remain blocked until the benchmark quality gate and full pytest/hardware-model lanes are green or explicitly gated with documented support boundaries.
