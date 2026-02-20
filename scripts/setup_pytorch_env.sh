#!/bin/bash
# Setup PyTorch environment for AutoVoice on Jetson Thor
# Requires: conda, CUDA 13.0

set -e

ENV_NAME="${1:-autovoice-thor}"
PYTHON_VERSION="3.12"

echo "=== AutoVoice PyTorch Environment Setup ==="
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"

# Create conda env
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment..."
    conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
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
echo "Installing PyTorch..."
pip install --no-user torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

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
pip install --no-user -r requirements.txt

# Install project in dev mode
pip install --no-user -e .

echo "=== Setup Complete ==="
echo "Activate with: conda activate $ENV_NAME"
