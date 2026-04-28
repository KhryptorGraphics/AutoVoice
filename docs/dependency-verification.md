# AutoVoice Dependency Verification

AutoVoice is standardized on the existing Jetson Thor conda environment:

- Env name: `autovoice-thor`
- Interpreter: `/home/kp/anaconda3/envs/autovoice-thor/bin/python`
- Runtime flags: `PYTHONNOUSERSITE=1`, `PYTHONPATH=src`

## Verification

Run the canonical dependency audit:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/kp/anaconda3/envs/autovoice-thor/bin/python \
  scripts/verify_dependencies.py --require-env --require-tensorrt
```

This checks:

- current interpreter/env alignment
- PyTorch + CUDA visibility
- Jetson Thor GPU identity and memory
- TensorRT, Demucs, HQ/quality-pipeline dependencies
- ARM64-sensitive packages such as `pyworld` and `pesq`

For repo import sanity, also run:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/kp/anaconda3/envs/autovoice-thor/bin/python \
  scripts/verify_bindings.py
```

## Supply-Chain Contract and Image Policy

Backend dependencies and production images are governed by `requirements.lock` and
validated by `scripts/check_dependency_contract.py`.

The contract checks:

- exact SHA-256 of `requirements-runtime.txt`
- exact SHA-256 of `frontend/package-lock.json`
- digest-pinned image references in:
  - `Dockerfile`
  - `frontend/Dockerfile.frontend`
  - `docker-compose.yaml` (monitoring images)
- optional high/critical vulnerability policy for generated audit reports

Run locally before merge:

```bash
python scripts/check_dependency_contract.py \
  --pip-audit-report reports/platform/pip-audit.json \
  --npm-audit-report reports/platform/npm-audit.json
```

GitHub `security-sbom` writes both reports under `reports/platform/` and runs this
contract check in CI.

## Environment Spec

The reproducible base spec lives in:

- `environment.autovoice-thor.yml`

Use it only to create the canonical env name. Do not introduce a second AutoVoice conda env.

## Source-Built Dependencies

Jetson Thor / ARM64 may require source builds for:

- `pyworld`
- `pesq`

Build or refresh them inside `autovoice-thor` with:

```bash
./scripts/build_source_dependencies.sh
```

The script also installs the remaining verification-only Python packages used by docs/tests:

- `flask-swagger-ui`
- `pystoi`
- `local-attention`

## Latency Profiling

Use the canonical wrapper so profiling always runs inside the correct env:

```bash
./scripts/profile_inference_latency.sh quality_seedvc --audio tests/quality_samples/william_singe_pillowtalk.wav
```

Swap the first argument for other pipelines such as `realtime`, `quality`, or `realtime_meanvc`.
