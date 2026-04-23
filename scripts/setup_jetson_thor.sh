#!/bin/bash
# Idempotent Jetson Thor bootstrap using the canonical AutoVoice env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

DRY_RUN=0
SKIP_MODEL_DOWNLOAD=0
SKIP_SERVICE_SETUP=0
SKIP_SYSTEMD=0
SKIP_LATENCY_VALIDATION=0
OUTPUT_DIR="${AUTOVOICE_PROJECT_ROOT}/reports/platform"
COMPOSE_FILE="${AUTOVOICE_PROJECT_ROOT}/docker-compose.yaml"
SYSTEMD_UNIT_SOURCE="${AUTOVOICE_PROJECT_ROOT}/config/systemd/autovoice.service"
SYSTEMD_UNIT_DEST="/etc/systemd/system/autovoice.service"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SETUP_LOG_DIR="${AUTOVOICE_PROJECT_ROOT}/logs/setup"
SETUP_LOG_PATH="${SETUP_LOG_DIR}/jetson-thor-setup-${TIMESTAMP}.log"
export AUTOVOICE_PRETRAINED_DIR="${AUTOVOICE_PRETRAINED_DIR:-${AUTOVOICE_PROJECT_ROOT}/models/pretrained}"

MYSQL_CONTAINER_NAME="${MYSQL_CONTAINER_NAME:-autovoice-mysql}"
POSTGRES_CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-autovoice-postgres}"
QDRANT_CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-autovoice-qdrant}"
AUTOVOICE_DB_HOST="${AUTOVOICE_DB_HOST:-127.0.0.1}"
AUTOVOICE_DB_PORT="${AUTOVOICE_DB_PORT:-3306}"
AUTOVOICE_DB_NAME="${AUTOVOICE_DB_NAME:-autovoice}"
AUTOVOICE_DB_USER="${AUTOVOICE_DB_USER:-root}"
AUTOVOICE_DB_PASS="${AUTOVOICE_DB_PASS:-}"
AUTOVOICE_DATABASE_URL="${AUTOVOICE_DATABASE_URL:-postgresql://autovoice:autovoice@127.0.0.1:5432/autovoice}"
DOCKER_COMPOSE_COMMAND=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --dry-run                 Print the actions without executing them
  --skip-model-download     Do not download pretrained or SOTA model assets
  --skip-service-setup      Do not inspect/install service dependencies
  --skip-systemd            Do not install the systemd service unit
  --skip-latency-validation Do not run the CUDA/latency validation wrapper
  --output-dir PATH         Directory for generated audit and validation artifacts
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-model-download)
            SKIP_MODEL_DOWNLOAD=1
            shift
            ;;
        --skip-service-setup)
            SKIP_SERVICE_SETUP=1
            shift
            ;;
        --skip-systemd)
            SKIP_SYSTEMD=1
            shift
            ;;
        --skip-latency-validation)
            SKIP_LATENCY_VALIDATION=1
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

mkdir -p "$SETUP_LOG_DIR" "$OUTPUT_DIR"
exec > >(tee -a "$SETUP_LOG_PATH") 2>&1

run_cmd() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

run_shell() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        bash -lc "$*"
    fi
}

log_section() {
    echo
    echo "=== $1 ==="
}

port_status() {
    local port="$1"
    if ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .; then
        echo "occupied"
    else
        echo "free"
    fi
}

detect_compose_command() {
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_COMMAND=(docker compose -f "$COMPOSE_FILE")
        return 0
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE_COMMAND=(docker-compose -f "$COMPOSE_FILE")
        return 0
    fi
    return 1
}

ensure_container_running() {
    local name="$1"
    local image="$2"
    local port="$3"
    shift 3

    if ! command -v docker >/dev/null 2>&1; then
        echo "docker: missing (cannot bootstrap $name)"
        return 1
    fi

    if docker ps --format '{{.Names}}' | grep -Fxq "$name"; then
        echo "$name: already running"
        return 0
    fi

    if docker ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "+ docker start $name"
        else
            docker start "$name"
        fi
        return 0
    fi

    if [[ "$(port_status "$port")" == "occupied" ]]; then
        echo "$name: port $port already in use, skipping container bootstrap"
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "+ docker run -d --restart unless-stopped --name $name -p ${port}:${port} $* $image"
        return 0
    fi

    docker run -d --restart unless-stopped --name "$name" -p "${port}:${port}" "$@" "$image" >/dev/null
}

initialize_datastores() {
    log_section "Database Initialization"
    if [[ -z "$AUTOVOICE_DB_PASS" ]]; then
        echo "AUTOVOICE_DB_PASS is required for MySQL initialization" >&2
        return 1
    fi
    echo "Initializing MySQL-backed metadata schema"
    run_shell "cd '$AUTOVOICE_PROJECT_ROOT' && AUTOVOICE_DB_HOST='$AUTOVOICE_DB_HOST' AUTOVOICE_DB_PORT='$AUTOVOICE_DB_PORT' AUTOVOICE_DB_NAME='$AUTOVOICE_DB_NAME' AUTOVOICE_DB_USER='$AUTOVOICE_DB_USER' AUTOVOICE_DB_PASS='$AUTOVOICE_DB_PASS' '$PYTHON' -c \"from auto_voice.db.schema import init_database; init_database('mysql')\""

    echo "Initializing PostgreSQL-backed profile schema"
    run_shell "cd '$AUTOVOICE_PROJECT_ROOT' && AUTOVOICE_DATABASE_URL='$AUTOVOICE_DATABASE_URL' '$PYTHON' -c \"from auto_voice.profiles.db.session import init_db; init_db()\""
}

probe_endpoint() {
    local url="$1"
    local retries="${2:-10}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "+ curl --fail --silent $url"
        return 0
    fi

    for (( attempt=1; attempt<=retries; attempt++ )); do
        if curl --fail --silent "$url" >/dev/null 2>&1; then
            echo "Verified endpoint: $url"
            return 0
        fi
        sleep 2
    done

    echo "Warning: endpoint did not become healthy: $url"
    return 1
}

log_section "AutoVoice Jetson Thor Setup"
echo "Project root: $AUTOVOICE_PROJECT_ROOT"
echo "Python: $PYTHON"
echo "Env: ${AUTOVOICE_ENV_NAME}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "Output dir: $OUTPUT_DIR"
echo "Pretrained bootstrap dir: $AUTOVOICE_PRETRAINED_DIR"
echo "Setup log: $SETUP_LOG_PATH"

log_section "Directory Preparation"
for path in \
    "$AUTOVOICE_DATA_DIR" \
    "$AUTOVOICE_DATA_DIR/app_state" \
    "$AUTOVOICE_PRETRAINED_DIR" \
    "$AUTOVOICE_PROJECT_ROOT/logs" \
    "$AUTOVOICE_PROJECT_ROOT/output" \
    "$OUTPUT_DIR"
do
    run_cmd mkdir -p "$path"
done

log_section "Environment Verification"
run_cmd "$PYTHON" "$SCRIPT_DIR/verify_dependencies.py" \
    --require-env \
    --require-tensorrt \
    --output "$OUTPUT_DIR/jetson-dependency-audit.txt"

log_section "Port Snapshot"
for port in 5000 6333 5432 3306; do
    echo "Port $port: $(port_status "$port")"
done

if [[ "$SKIP_MODEL_DOWNLOAD" -eq 0 ]]; then
    log_section "Model Asset Bootstrap"
    run_cmd "$PYTHON" "$SCRIPT_DIR/download_pretrained_models.py"
    run_cmd "$PYTHON" "$SCRIPT_DIR/setup_sota_models.py"
else
    echo "Skipping model downloads"
fi

if [[ "$SKIP_SERVICE_SETUP" -eq 0 ]]; then
    log_section "Service Dependency Snapshot"
    for command_name in mysql psql docker systemctl; do
        if command -v "$command_name" >/dev/null 2>&1; then
            echo "$command_name: available"
        else
            echo "$command_name: missing"
        fi
    done

    if command -v docker >/dev/null 2>&1; then
        echo "Bootstrapping dependency containers"
        ensure_container_running \
            "$MYSQL_CONTAINER_NAME" \
            "mysql:8" \
            3306 \
            -e "MYSQL_ROOT_PASSWORD=$AUTOVOICE_DB_PASS" \
            -e "MYSQL_DATABASE=$AUTOVOICE_DB_NAME"
        ensure_container_running \
            "$POSTGRES_CONTAINER_NAME" \
            "postgres:15" \
            5432 \
            -e "POSTGRES_USER=autovoice" \
            -e "POSTGRES_PASSWORD=autovoice" \
            -e "POSTGRES_DB=autovoice"
        ensure_container_running \
            "$QDRANT_CONTAINER_NAME" \
            "qdrant/qdrant:latest" \
            6333
    fi

    initialize_datastores

    if [[ -f "$COMPOSE_FILE" ]] && detect_compose_command; then
        log_section "Docker Compose Stack"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "+ ${DOCKER_COMPOSE_COMMAND[*]} pull backend frontend"
            echo "+ ${DOCKER_COMPOSE_COMMAND[*]} up -d backend frontend"
        else
            "${DOCKER_COMPOSE_COMMAND[@]}" pull backend frontend
            "${DOCKER_COMPOSE_COMMAND[@]}" up -d backend frontend
        fi
    else
        echo "Docker Compose stack unavailable; skipping backend/frontend bootstrap"
    fi

    if [[ "$SKIP_SYSTEMD" -eq 0 && -f "$SYSTEMD_UNIT_SOURCE" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "+ install -m 0644 $SYSTEMD_UNIT_SOURCE $SYSTEMD_UNIT_DEST"
            echo "+ systemctl daemon-reload"
            echo "+ systemctl enable autovoice.service"
        elif [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
            install -m 0644 "$SYSTEMD_UNIT_SOURCE" "$SYSTEMD_UNIT_DEST"
            systemctl daemon-reload
            systemctl enable autovoice.service
            echo "Installed systemd unit: $SYSTEMD_UNIT_DEST"
        else
            echo "Skipping systemd install because root privileges are required"
        fi
    else
        echo "Skipping systemd configuration"
    fi
else
    echo "Skipping service dependency setup"
fi

if [[ "$SKIP_LATENCY_VALIDATION" -eq 0 ]]; then
    log_section "CUDA and Latency Validation"
    run_cmd "$SCRIPT_DIR/validate_cuda_stack.sh" \
        --pipeline realtime \
        --output-dir "$OUTPUT_DIR"
else
    echo "Skipping latency validation"
fi

log_section "Health Check Hints"
echo "Use these commands after launch:"
echo "  curl http://localhost:5000/health"
echo "  curl http://localhost:5000/ready"
echo "  $PYTHON $SCRIPT_DIR/verify_dependencies.py --require-env --require-tensorrt"

if [[ "$SKIP_SERVICE_SETUP" -eq 0 ]]; then
    log_section "Endpoint Verification"
    probe_endpoint "http://localhost:5000/health" 5 || true
    probe_endpoint "http://localhost:5000/ready" 5 || true
fi

log_section "Complete"
echo "Jetson Thor setup workflow completed."
