#!/usr/bin/env bash
# Validate docker-compose.yaml syntax with safe local placeholder secrets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export SECRET_KEY="${SECRET_KEY:-compose-config-validation-secret}"
export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-compose-config-validation-grafana-password}"

docker compose -f docker-compose.yaml config -q
