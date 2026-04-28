# Jetson Thor Deployment

AutoVoice now ships a canonical bootstrap flow for Jetson Thor:

```bash
scripts/setup_jetson_thor.sh
```

The script assumes the existing `autovoice-thor` conda environment and the shared bootstrap in [`scripts/common_env.sh`](../scripts/common_env.sh). It does not create a new environment. Instead it:

- activates the canonical interpreter
- verifies required dependencies and TensorRT
- prepares runtime directories under `data/`, `logs/`, the pretrained bootstrap directory, and `reports/platform/`
- bootstraps the companion MySQL, PostgreSQL, and Qdrant containers when Docker is available
- initializes both the MySQL metadata schema and PostgreSQL profile schema
- brings up the bundled `docker-compose.yaml` backend/frontend stack
- optionally downloads pretrained and SOTA model assets
- optionally installs the bundled systemd unit
- runs the CUDA and latency validation wrapper

## Recommended Invocation

```bash
scripts/setup_jetson_thor.sh --output-dir reports/platform
```

The pretrained model host directory is explicit and operator-owned:

```bash
export AUTOVOICE_PRETRAINED_DIR=/srv/autovoice/models/pretrained
python scripts/setup_sota_models.py --models-dir "$AUTOVOICE_PRETRAINED_DIR"
```

If `AUTOVOICE_PRETRAINED_DIR` is unset, bootstrap scripts use repo-local
`models/pretrained` as a host default. The compose stack always mounts the
selected host directory into the stable container path `/app/models/pretrained`.
`SECRET_KEY` is required by compose; `scripts/setup_jetson_thor.sh` derives it
from `SECRET_KEY`, then `AUTOVOICE_SECRET_FLASK_SECRET_KEY`, and otherwise
generates an ephemeral local bootstrap secret.

Useful flags:

- `--dry-run`: print the actions without executing them
- `--skip-model-download`: keep existing model assets untouched
- `--skip-service-setup`: skip dependency-container bootstrap, schema init, compose bring-up, and systemd checks
- `--skip-systemd`: skip the systemd unit install step
- `--skip-latency-validation`: skip the multi-pipeline latency benchmark wrapper

## Validation Artifacts

The setup flow writes its outputs under `reports/platform/` by default:

- `jetson-dependency-audit.txt`
- `dependency-audit.json`
- `all-latency-report.md`

You can rerun the validation independently:

```bash
scripts/validate_cuda_stack.sh --pipeline all --output-dir reports/platform
python scripts/run_completion_matrix.py --output-dir reports/completion/latest
python scripts/validate_release_candidate.py --base-url http://127.0.0.1:5000 --report-dir reports/platform
python scripts/validate_hosted_deployment.py --skip-dns --skip-tls --vhost-file /etc/apache2/sites-available/autovoice.giggadev.com.conf
python scripts/validate_benchmark_dashboard.py
```

## Operational Notes

- Expected web port: `5000`
- Common companion ports checked during setup: `3306` (MySQL), `5432` (Postgres), `6333` (Qdrant)
- Compose stack source: `docker-compose.yaml`
- Compose images: `autovoice-backend:${AUTOVOICE_IMAGE_TAG:-local}`, `autovoice-frontend:${AUTOVOICE_IMAGE_TAG:-local}`, pinned Prometheus and Grafana images
- `requirements.lock` additionally records the digest-pinned image references for backend base image, frontend builder/runtime images, and monitoring images. CI enforces this via `scripts/check_dependency_contract.py`.
- Durable app state: compose sets `DATA_DIR=/app/data` and persists canonical state directories with named volumes, including `app_state`, `voice_profiles`, `samples`, `trained_models`, `checkpoints`, upload/output folders, YouTube/separation staging folders, and swarm run/memory folders
- Systemd unit source: `config/systemd/autovoice.service`
- Root privileges are only required if you want the setup script to install the service unit

## Apache Hosting

The live AutoVoice app on this host is served behind Apache and the existing
`autovoice.service` backend:

- backend service: `autovoice.service`
- backend bind: `127.0.0.1:10600`
- frontend document root: `frontend/dist`
- Apache vhost files:
  - `/etc/apache2/sites-available/autovoice.giggadev.com.conf`
  - `/etc/apache2/sites-available/autovoice.giggadev.com-le-ssl.conf`

Apache serves the built frontend directly from `frontend/dist` and reverse
proxies `/api`, `/socket.io`, `/health`, and `/ready` to the backend on
`127.0.0.1:10600`. The `/ready` proxy must be defined before the frontend SPA
fallback so public readiness returns backend JSON instead of `index.html`.

Because AutoVoice accepts large multipart audio uploads, Apache's ModSecurity
request-body limit must be raised above the default `13,107,200` bytes on the
live host. The current production host is configured with:

- `SecRequestBodyLimit 262144000`

If this limit is reset during a host rebuild, the upload APIs will fail before
the app sees the request, typically as `413 Request Entity Too Large` on:

- `/api/v1/convert/workflows`
- `/api/v1/karaoke/upload`
- profile sample/song upload endpoints

The current vhost accepts both hostnames:

- `autovoice.giggadev.com`
- `autovoice.giggahost.com`

For public HTTPS on `autovoice.giggahost.com`, the host still needs two
external prerequisites outside the repo:

- DNS for `autovoice.giggahost.com` must resolve to this server
- the active Let's Encrypt certificate must include `autovoice.giggahost.com`
- every enabled Apache vhost must reference existing certificate files, because
  one unrelated missing certificate path can block `apache2ctl configtest` and
  prevent AutoVoice reloads

## Release Candidate Workflow

The repo now ships two executable validation lanes:

- GitHub Actions `release-candidate`: on a self-hosted Jetson runner, validate compose config, boot the real `docker-compose.yaml` backend/frontend stack, and run `scripts/validate_release_candidate.py`
- GitHub Actions `jetson-nightly`: run `scripts/validate_cuda_stack.sh --pipeline all` on a self-hosted Jetson runner

Release-candidate validation now requires benchmark evidence artifacts under
`reports/benchmarks/latest/` to be structurally valid and to carry provenance
for the candidate commit being validated. A stale dashboard or release-evidence
pair from another git SHA no longer satisfies the RC gate.

The canonical production completion matrix is:

```bash
python scripts/run_completion_matrix.py --full --base-url http://127.0.0.1:10001
```

Use the default smoke mode for local development when frontend browser tooling,
Docker, public DNS/TLS, or Jetson TensorRT hardware are intentionally
unavailable. Smoke mode still writes `reports/completion/latest/completion_matrix.json`
and records unavailable lanes explicitly instead of hiding them.

Post-release public production monitoring is handled by
`.github/workflows/production-monitoring.yml`. It runs a nightly health smoke
against `https://autovoice.giggahost.com` and a weekly full workflow proof that
uploads tracked audio fixtures, trains a minimal LoRA, queues conversion,
downloads output artifacts, and archives evidence under `reports/production_smoke`.
Run the same checks manually with:

```bash
python scripts/run_production_smoke.py --mode health
python scripts/run_production_smoke.py --mode full --timeout-seconds 1800
```

Rollback drills are dry-run by default and write a machine-readable command plan:

```bash
python scripts/run_rollback_drill.py \
  --base-url https://autovoice.giggahost.com \
  --output reports/platform/rollback-drill.json
```

The hosted preflight lane is machine-checkable:

```bash
python scripts/validate_hosted_deployment.py \
  --hostname autovoice.giggahost.com \
  --backend-port 10600 \
  --vhost-file /etc/apache2/sites-available/autovoice.giggadev.com.conf \
  --vhost-file /etc/apache2/sites-available/autovoice.giggadev.com-le-ssl.conf
```

Use `--skip-dns` or `--skip-tls` only for local/dry-run checks where public
records or certificates are intentionally unavailable.

Current-head hardware release evidence is generated separately from the smoke
matrix and fails closed when Jetson/CUDA lanes are not actually executed:

```bash
python scripts/run_hardware_release_evidence.py --execute
```

The runner writes immutable artifacts under `reports/release-evidence/<timestamp>-<git-sha>/`
and mirrors the latest decision to `reports/release-evidence/latest/release_decision.json`.
Use `--dry-run --allow-blocked` only to prove preflight/report generation on
non-Jetson developer machines; a dry-run decision is never release-ready.

Rollback criteria for a release candidate are simple:

- `/api/v1/health` is not healthy
- `/ready` is not ready
- `/api/v1/metrics` does not respond
- compose config is invalid
- Jetson CUDA/TensorRT validation fails on the target hardware lane

For broader production guidance, monitoring, and Docker notes, see [deployment-guide.md](./deployment-guide.md).
