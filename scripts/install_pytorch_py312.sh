#!/bin/bash
# Install PyTorch and dependencies in Python 3.12 environment
# This script should be run AFTER activating the autovoice_py312 environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Installing PyTorch in Python 3.12 Environment        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right environment
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ ! "$PYTHON_VERSION" =~ ^3\.12\. ]]; then
    echo -e "${RED}[✗] Error: Not in Python 3.12 environment${NC}"
    echo "Current Python version: $PYTHON_VERSION"
    echo ""
    echo "Please activate the environment first:"
    echo "  conda activate autovoice_py312"
    exit 1
fi

echo -e "${GREEN}[✓] Python 3.12 environment detected: $PYTHON_VERSION${NC}"
echo ""

# Step 1: Install PyTorch with CUDA support
echo -e "${BLUE}→ Installing PyTorch with CUDA 12.1 support${NC}"
echo "This may take 5-10 minutes..."
echo ""

# Use conda to install PyTorch with CUDA support
# Note: Using CUDA 12.1 as it's the most stable version for PyTorch 2.5+
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo ""
echo -e "${GREEN}[✓] PyTorch installation complete${NC}"
echo ""

# Step 2: Verify PyTorch installation
echo -e "${BLUE}→ Verifying PyTorch installation${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo -e "${RED}[✗] Failed to import PyTorch${NC}"
    exit 1
}

# Check for libtorch_global_deps.so
TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
if [ -f "${TORCH_LIB_PATH}/libtorch_global_deps.so" ]; then
    echo -e "${GREEN}[✓] libtorch_global_deps.so found${NC}"
else
    echo -e "${YELLOW}[!] Warning: libtorch_global_deps.so not found${NC}"
    echo "Path checked: ${TORCH_LIB_PATH}/libtorch_global_deps.so"
fi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}[✓] CUDA is available: $GPU_NAME${NC}"
else
    echo -e "${YELLOW}[!] CUDA is not available (CPU-only mode)${NC}"
fi

echo ""

# Step 3: Install project dependencies
echo -e "${BLUE}→ Installing project dependencies${NC}"
echo "This may take 5-10 minutes..."
echo ""

cd /home/kp/autovoice
pip install -r requirements.txt

echo ""
echo -e "${GREEN}[✓] Dependencies installation complete${NC}"
echo ""

# Step 4: Build CUDA extensions
echo -e "${BLUE}→ Building CUDA extensions${NC}"
echo "This may take 2-5 minutes..."
echo ""

pip install -e .

echo ""
echo -e "${GREEN}[✓] CUDA extensions build complete${NC}"
echo ""

# Step 5: Final verification
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Final Verification                                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Python version:"
python --version

echo ""
echo "PyTorch information:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "CUDA kernels module:"
python -c "try:
    from auto_voice import cuda_kernels
    print('  ✓ cuda_kernels module imported successfully')
    print(f'  Available functions: {dir(cuda_kernels)}')
except ImportError as e:
    print(f'  ✗ Failed to import cuda_kernels: {e}')
except Exception as e:
    print(f'  ✗ Error: {e}')
"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Installation Complete!                                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Run verification script: ./scripts/verify_bindings.py"
echo "  2. Run tests: ./scripts/build_and_test.sh"
echo ""

