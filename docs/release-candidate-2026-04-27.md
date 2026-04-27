# AutoVoice Release Candidate Evidence - 2026-04-27

This release candidate is validated through the full completion matrix in the
project conda environment:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/kp/anaconda3/envs/autovoice-thor/bin/python \
  scripts/run_completion_matrix.py --full \
  --base-url http://127.0.0.1:10001 \
  --timeout 1800
```

Canonical local evidence artifacts are generated under ignored report paths:

- `reports/completion/latest/completion_matrix.json`
- `reports/completion/latest/logs/`
- `reports/platform/release-candidate-validation.json`
- `reports/benchmarks/latest/release_evidence.json`
- `reports/benchmarks/latest/dashboard/`

The completion matrix is the release gate. It covers GitNexus refresh,
skip-placeholder audit, backend contract smoke, compose config, benchmark
dashboard build/validation, experimental evidence validation, hosted preflight,
frontend lint/typecheck/build/browser smoke, real compose boot, release
candidate validation against the real stack, compose teardown, and
Jetson/CUDA/TensorRT capability checks.

Release candidate tagging should only happen after the matrix reports
`"ok": true` on the commit being tagged.
