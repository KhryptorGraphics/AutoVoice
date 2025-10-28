#!/bin/bash
# Environment Setup for Verification Comments Implementation
# Sets up Python environment with all required dependencies

set -e

echo "=================================================="
echo "AutoVoice Environment Setup"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

print_status "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
    print_warning "Python 3.8+ recommended (current: $PYTHON_VERSION)"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_status "CUDA version: $CUDA_VERSION"
else
    print_warning "CUDA not found - GPU acceleration will not be available"
fi

# Install core dependencies
print_status "Installing core dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
print_status "Installing PyTorch..."
if command -v nvcc &> /dev/null; then
    # Install PyTorch with CUDA
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # CPU-only
    pip install torch torchaudio
fi

# Install audio processing dependencies
print_status "Installing audio processing libraries..."
pip install \
    torchcrepe \
    librosa \
    soundfile \
    scipy \
    praat-parselmouth \
    pyyaml \
    numpy

# Install development dependencies
print_status "Installing development tools..."
pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Install AutoVoice in development mode
print_status "Installing AutoVoice..."
cd "$(dirname "$0")/.."
pip install -e .

print_status "Environment setup complete!"
echo ""
echo "Verification:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torchcrepe; print(f'  torchcrepe: installed')"
python -c "import librosa; print(f'  librosa: {librosa.__version__}')"

echo ""
echo "Next steps:"
echo "  1. Run: bash scripts/verify_implementation.sh"
echo "  2. Run: pytest tests/"
echo ""
