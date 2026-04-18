#!/bin/bash
# Setup PyTorch environment for AutoVoice on Jetson Thor
# Requires: conda, CUDA 13.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/environment.autovoice-thor.yml"

ENV_NAME="${1:-${AUTOVOICE_ENV_NAME:-autovoice-thor}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

echo "=== AutoVoice PyTorch Environment Setup ==="
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"

# Create conda env
if ! conda env list | grep -q "^$ENV_NAME "; then
    if [[ -f "$ENV_FILE" ]]; then
        echo "Creating conda environment from $ENV_FILE..."
        conda env create -f "$ENV_FILE" -n "$ENV_NAME"
    else
        echo "Creating conda environment..."
        conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
    fi
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Verify Python
echo "Python: $(python --version)"
echo "Path: $(which python)"

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="11.0"

# Prevent system package contamination
export PYTHONNOUSERSITE=1

# Install PyTorch (nightly with CUDA 13.0 support)
if ! python -c "import torch, torchaudio" >/dev/null 2>&1; then
    echo "Installing PyTorch..."
    pip install --no-user torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
else
    echo "PyTorch already present in $ENV_NAME"
fi

# Verify PyTorch + CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Compute cap: {torch.cuda.get_device_capability(0)}')
"

# Install project dependencies
echo "Installing project dependencies..."
pip install --no-user -r "$PROJECT_ROOT/requirements.txt"
pip install --no-user flask-swagger-ui pystoi local-attention

# Install project in dev mode
pip install --no-user -e "$PROJECT_ROOT"

# Build source-sensitive dependencies used on Jetson
"$PROJECT_ROOT/scripts/build_source_dependencies.sh"

# Final verification
"$PROJECT_ROOT/scripts/verify_dependencies.py" --require-env

echo "=== Setup Complete ==="
echo "Activate with: conda activate $ENV_NAME"
