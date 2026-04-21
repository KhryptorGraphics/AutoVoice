# Jetson Thor Deployment

AutoVoice now ships a canonical bootstrap flow for Jetson Thor:

```bash
scripts/setup_jetson_thor.sh
```

The script assumes the existing `autovoice-thor` conda environment and the shared bootstrap in [`scripts/common_env.sh`](../scripts/common_env.sh). It does not create a new environment. Instead it:

- activates the canonical interpreter
- verifies required dependencies and TensorRT
- prepares runtime directories under `data/`, `logs/`, `models/pretrained/`, and `reports/platform/`
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
python scripts/validate_release_candidate.py --base-url http://127.0.0.1:5000 --report-dir reports/platform
```

## Operational Notes

- Expected web port: `5000`
- Common companion ports checked during setup: `3306` (MySQL), `5432` (Postgres), `6333` (Qdrant)
- Compose stack source: `docker-compose.yaml`
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
proxies `/api`, `/socket.io`, and `/health` to the backend on `127.0.0.1:10600`.

The current vhost accepts both hostnames:

- `autovoice.giggadev.com`
- `autovoice.giggahost.com`

For public HTTPS on `autovoice.giggahost.com`, the host still needs two
external prerequisites outside the repo:

- DNS for `autovoice.giggahost.com` must resolve to this server
- the active Let's Encrypt certificate must include `autovoice.giggahost.com`

## Release Candidate Workflow

The repo now ships two executable validation lanes:

- GitHub Actions `release-candidate`: build frontend, validate compose config, boot the deterministic backend harness, and run `scripts/validate_release_candidate.py`
- GitHub Actions `jetson-nightly`: run `scripts/validate_cuda_stack.sh --pipeline all` on a self-hosted Jetson runner

Rollback criteria for a release candidate are simple:

- `/health` is not healthy
- `/ready` is not ready
- `/api/v1/metrics` does not respond
- compose config is invalid
- Jetson CUDA/TensorRT validation fails on the target hardware lane

For broader production guidance, monitoring, and Docker notes, see [deployment-guide.md](./deployment-guide.md).
