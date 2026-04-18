#!/bin/bash
# Shared AutoVoice environment bootstrap for Jetson Thor scripts.

set -euo pipefail

AUTOVOICE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOVOICE_PROJECT_ROOT="$(cd "$AUTOVOICE_SCRIPT_DIR/.." && pwd)"
AUTOVOICE_ENV_NAME="${AUTOVOICE_ENV_NAME:-autovoice-thor}"
AUTOVOICE_ENV_PREFIX="${AUTOVOICE_ENV_PREFIX:-/home/kp/anaconda3/envs/${AUTOVOICE_ENV_NAME}}"
AUTOVOICE_PYTHON_DEFAULT="${AUTOVOICE_ENV_PREFIX}/bin/python"

autovoice_activate_env() {
    local requested_python="${PYTHON:-${AUTOVOICE_PYTHON:-$AUTOVOICE_PYTHON_DEFAULT}}"

    if [[ -x "$requested_python" ]]; then
        export PYTHON="$requested_python"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$AUTOVOICE_ENV_NAME"
        export PYTHON="$(command -v python)"
    else
        echo "AutoVoice Python interpreter not found: $requested_python" >&2
        return 1
    fi

    export PYTHONNOUSERSITE=1
    export PYTHONPATH="${AUTOVOICE_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-11.0}"
}
