#!/usr/bin/env bash
# Start the AutoVoice server with a persisted Flask secret.
#
# create_app() requires the Flask secret via the AUTOVOICE_SECRET_FLASK_SECRET_KEY
# env var (it is not read from any repo file). Launching main.py directly without
# it aborts with SecretError. This wrapper persists a key to data/flask_secret.key
# (gitignored) and exports it, so restarts are reliable. Rotating the key only
# resets Flask sessions; JWT auth uses the separate data/jwt.secret.
#
# Usage:  scripts/start_autovoice.sh            # foreground
#         nohup scripts/start_autovoice.sh &    # background
# Overrides: AUTOVOICE_PYTHON, AUTOVOICE_HOST, AUTOVOICE_PORT
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${AUTOVOICE_PYTHON:-/home/kp/anaconda3/envs/autovoice-thor/bin/python}"
HOST="${AUTOVOICE_HOST:-0.0.0.0}"
PORT="${AUTOVOICE_PORT:-10600}"
KEYFILE="$REPO/data/flask_secret.key"

cd "$REPO"
mkdir -p "$(dirname "$KEYFILE")"
if [ ! -s "$KEYFILE" ]; then
  PYTHONNOUSERSITE=1 "$PY" -c "import secrets; open('$KEYFILE','w').write(secrets.token_urlsafe(48))"
  chmod 600 "$KEYFILE"
  echo "start_autovoice: generated new Flask secret at $KEYFILE"
fi

export AUTOVOICE_SECRET_FLASK_SECRET_KEY="$(cat "$KEYFILE")"
export PYTHONNOUSERSITE=1
echo "start_autovoice: launching on ${HOST}:${PORT} (python: ${PY})"
exec "$PY" main.py --host "$HOST" --port "$PORT"
