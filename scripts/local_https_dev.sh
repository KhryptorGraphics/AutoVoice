#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="${AUTOVOICE_HTTPS_CERT_DIR:-"$ROOT_DIR/.local/https"}"
CERT_PATH="${AUTOVOICE_SSL_CERT:-"$CERT_DIR/autovoice-local.crt"}"
KEY_PATH="${AUTOVOICE_SSL_KEY:-"$CERT_DIR/autovoice-local.key"}"
OPENSSL_CONFIG="$CERT_DIR/openssl.cnf"

BACKEND_PORT="${AUTOVOICE_HTTPS_BACKEND_PORT:-5443}"
FRONTEND_PORT="${AUTOVOICE_HTTPS_FRONTEND_PORT:-3443}"
CONDA_ENV="${AUTOVOICE_CONDA_ENV:-autovoice-thor}"
LAN_HOST="${AUTOVOICE_LAN_HOST:-}"

detect_lan_host() {
  if [[ -n "$LAN_HOST" ]]; then
    printf '%s\n' "$LAN_HOST"
    return
  fi

  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | awk '{print $1}' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || true
  fi
}

LAN_HOST="$(detect_lan_host | head -n 1)"
URL_HOST="${AUTOVOICE_BROWSER_HOST:-${LAN_HOST:-localhost}}"

mkdir -p "$CERT_DIR"

if [[ ! -f "$CERT_PATH" || ! -f "$KEY_PATH" ]]; then
  cat > "$OPENSSL_CONFIG" <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
CN = autovoice-local

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
EOF

  if [[ -n "$LAN_HOST" ]]; then
    if [[ "$LAN_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      printf 'IP.2 = %s\n' "$LAN_HOST" >> "$OPENSSL_CONFIG"
    else
      printf 'DNS.2 = %s\n' "$LAN_HOST" >> "$OPENSSL_CONFIG"
    fi
  fi

  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$KEY_PATH" \
    -out "$CERT_PATH" \
    -config "$OPENSSL_CONFIG" >/dev/null 2>&1
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

run_backend() {
  cd "$ROOT_DIR"
  export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  export SECRET_KEY="${SECRET_KEY:-local-dev-secret}"

  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda run --no-capture-output -n "$CONDA_ENV" autovoice serve \
      --host 0.0.0.0 \
      --port "$BACKEND_PORT" \
      --ssl-cert "$CERT_PATH" \
      --ssl-key "$KEY_PATH"
  else
    autovoice serve \
      --host 0.0.0.0 \
      --port "$BACKEND_PORT" \
      --ssl-cert "$CERT_PATH" \
      --ssl-key "$KEY_PATH"
  fi
}

run_frontend() {
  cd "$ROOT_DIR/frontend"
  export VITE_BACKEND_URL="https://$URL_HOST:$BACKEND_PORT"
  export VITE_DEV_SSL_CERT="$CERT_PATH"
  export VITE_DEV_SSL_KEY="$KEY_PATH"
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
}

printf 'AutoVoice local HTTPS dev stack\n'
printf '  Frontend: https://%s:%s\n' "$URL_HOST" "$FRONTEND_PORT"
printf '  Backend:  https://%s:%s\n' "$URL_HOST" "$BACKEND_PORT"
printf '  Cert:     %s\n' "$CERT_PATH"
printf 'Accept or trust the self-signed certificate in the browser before using microphone capture from another LAN machine.\n'

run_backend &
BACKEND_PID=$!
run_frontend &
FRONTEND_PID=$!

wait -n "$BACKEND_PID" "$FRONTEND_PID"
