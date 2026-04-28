# AutoVoice Production Operations Runbook

This runbook applies to local, private, or hosted deployments. Public/commercial launch still requires policy/legal approval and current-head hardware evidence.

## Readiness

- Use `/ready` for container readiness and deploy gating.
- Use `/api/v1/health` only as liveness and component diagnostics; it can return HTTP 200 while degraded.
- Use `/api/v1/metrics?format=prometheus` for Prometheus scraping.

## Public-Mode Hardening

- Set `AUTOVOICE_PUBLIC_DEPLOYMENT=true`.
- Set `AUTOVOICE_API_TOKEN` and leave `AUTOVOICE_REQUIRE_API_AUTH=true` unless a trusted reverse proxy enforces auth.
- Set explicit `CORS_ORIGINS`; wildcard origins are rejected in public mode unless `AUTOVOICE_ALLOW_INSECURE_CORS=true`.
- Keep `AUTOVOICE_STRICT_PATH_SANDBOX=true` to prevent arbitrary server-path ingestion.
- Set `AUTOVOICE_REQUIRE_MEDIA_CONSENT=true` so risky media-import paths require rights/consent attestations.

## Evidence Gates

- Regenerate benchmark evidence for every candidate commit.
- Archive evidence under `reports/benchmarks/<date>-<git-sha>/`; `reports/benchmarks/latest/` is only the convenience copy.
- Run `python scripts/validate_benchmark_dashboard.py --current-git-sha --release-grade` before release claims.
- Do not use `/tmp` benchmark source bundles as release evidence.

## Full RC Evidence

- Use `python scripts/preflight_full_hardware_rc.py --output reports/release_candidates/AV-j4cd/preflight.json --benchmark-report <current-head-benchmark-report.json>` to record hard blockers before starting the full run.
- Use `python scripts/run_full_hardware_rc.py --benchmark-report <current-head-benchmark-report.json>` on the Jetson/hosted runner to collect the full RC bundle.
- `<current-head-benchmark-report.json>` must be the raw `comprehensive_report.json`; the runner derives `release_evidence.json` from that source.
- The full runner writes immutable artifacts under `reports/release_candidates/AV-j4cd/<timestamp>-<git-sha>/`.
- `release_decision.json` is the canonical go/no-go artifact for that run.
- `artifact_manifest.json` records the exact completion-matrix, platform, benchmark, parity, and production-smoke artifacts captured for the decision.

For single-user Local Jetson readiness without Docker deployment, run the RC
bundle against the already-running local service URL and disable hosted/compose
probes. Use `http://127.0.0.1:10600` for the local `main.py` service, or replace
it with the actual local port if different:

```bash
python scripts/run_full_hardware_rc.py \
  --bead-id local-jetson \
  --artifact-root reports/release_candidates \
  --benchmark-report <current-head-benchmark-report.json> \
  --deployment-base-url http://127.0.0.1:10600 \
  --local-base-url http://127.0.0.1:10600 \
  --no-require-hosted-probes \
  --no-require-production-smoke-stems \
  --no-run-real-compose \
  --no-run-full-hosted-preflight \
  --require-jetson \
  --no-require-docker \
  --require-gitnexus \
  --require-tensorrt-suite \
  --require-clean-head
```

Use `reports/release_candidates/local-jetson/<timestamp>-<git-sha>/release_decision.json` as the Local Jetson release record. Mutable `reports/*/latest` directories are convenience copies only.

## Backup And Restore

- Run `scripts/backup_compose_volumes.sh` before upgrades and after major training/import sessions.
- Restore with `scripts/restore_compose_volumes.sh <backup-dir>` while the compose stack is stopped.
- Backup all AutoVoice named volumes, not just profile metadata; source media, derived stems, trained models, outputs, swarm state, and logs are separate volumes.

## Incident Response

- Start with `docker compose ps`, `/ready`, and `/api/v1/health`.
- Use `X-Request-ID` from responses to correlate backend logs.
- For high 5xx rate or scrape failure, inspect `config/prometheus/alert_rules.yml` alerts and backend logs.
- For suspected abuse, rotate `AUTOVOICE_API_TOKEN`, restrict `CORS_ORIGINS`, and temporarily disable public ingress.
