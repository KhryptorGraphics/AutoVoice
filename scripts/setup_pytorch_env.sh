#!/bin/bash
# AutoVoice PyTorch Environment Setup Script
# Automated resolution for PyTorch library issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Unicode symbols
CHECK="✓"
CROSS="✗"
INFO="ℹ"
ARROW="→"

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     AutoVoice PyTorch Environment Setup Script         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${BLUE}[${INFO}]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[${CHECK}]${NC} $1"
}

print_error() {
    echo -e "${RED}[${CROSS}]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}${ARROW}${NC} ${1}"
}

# Step 1: Detect Python version
print_step "Detecting Python environment"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "  Python version: ${PYTHON_VERSION}"

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 13 ]; then
    print_warning "Python 3.13 detected - PyTorch support is experimental"
    PYTHON_313=true
else
    print_success "Python ${PYTHON_VERSION} has stable PyTorch support"
    PYTHON_313=false
fi

# Step 2: Check if PyTorch is installed
print_step "Checking PyTorch installation"

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    TORCH_INSTALLED=true
    echo "  PyTorch version: ${TORCH_VERSION}"

    # Check for libtorch_global_deps.so
    TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")

    if [ -n "$TORCH_LIB_PATH" ] && [ -f "${TORCH_LIB_PATH}/libtorch_global_deps.so" ]; then
        print_success "libtorch_global_deps.so found"
        MISSING_LIB=false
    else
        print_error "libtorch_global_deps.so missing"
        MISSING_LIB=true
    fi
else
    print_error "PyTorch not installed or cannot be imported"
    TORCH_INSTALLED=false
    MISSING_LIB=true
fi

# Step 3: Check CUDA availability
print_step "Checking CUDA environment"

if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA driver detected"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | while read line; do
        echo "  GPU: $line"
    done
    NVIDIA_AVAILABLE=true
else
    print_warning "NVIDIA driver not found"
    NVIDIA_AVAILABLE=false
fi

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    print_success "CUDA Toolkit ${CUDA_VERSION} detected"
    CUDA_AVAILABLE=true
else
    print_warning "CUDA Toolkit not found"
    CUDA_AVAILABLE=false
fi

# Step 4: Determine recommended action
print_step "Analyzing environment and recommending solution"

if [ "$TORCH_INSTALLED" = true ] && [ "$MISSING_LIB" = false ]; then
    print_success "PyTorch is properly installed and working!"

    # Verify CUDA support
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        print_success "CUDA is available in PyTorch"
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  Environment is ready! No action needed.               ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
        exit 0
    else
        print_warning "CUDA is not available in PyTorch"
    fi
fi

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Action Required: PyTorch needs to be fixed              ${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════${NC}"
echo ""

# Present options
echo "Available solutions:"
echo ""
echo "  1) Quick Fix: Reinstall PyTorch nightly (5-10 min)"
echo "     - Fast to try"
echo "     - May resolve corrupted installation"
echo "     - Success rate: ~40%"
echo ""
echo "  2) Recommended: Downgrade to Python 3.12 (30 min)"
echo "     - Stable PyTorch support"
echo "     - Proven to work"
echo "     - Success rate: ~95%"
echo ""
echo "  3) Advanced: Build PyTorch from source (2+ hours)"
echo "     - Full control"
echo "     - Keeps Python 3.13"
echo "     - Success rate: ~80%"
echo ""

# If Python 3.13, recommend downgrade
if [ "$PYTHON_313" = true ]; then
    print_warning "Python 3.13 has experimental PyTorch support"
    echo ""
    echo "RECOMMENDATION: Option 2 (Python 3.12 downgrade) is strongly recommended"
    echo "for stable, production-ready environment."
fi

# Interactive prompt
echo ""
read -p "Choose an option (1/2/3) or 'q' to quit: " choice

case $choice in
    1)
        print_step "Option 1: Reinstalling PyTorch nightly"
        echo ""
        echo "This will:"
        echo "  - Completely remove existing PyTorch installation"
        echo "  - Clear pip cache"
        echo "  - Install latest nightly build from PyTorch"
        echo ""
        read -p "Proceed with reinstall? (y/N): " confirm

        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            print_status "Operation cancelled"
            exit 0
        fi

        print_step "Removing existing PyTorch"
        pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

        if [ -d "/home/kp/anaconda3/lib/python${PYTHON_MAJOR}.${PYTHON_MINOR}/site-packages/torch" ]; then
            rm -rf "/home/kp/anaconda3/lib/python${PYTHON_MAJOR}.${PYTHON_MINOR}/site-packages/torch"* 2>/dev/null || true
        fi

        print_step "Clearing pip cache"
        pip cache purge

        print_step "Installing PyTorch nightly"
        if [ "$CUDA_AVAILABLE" = true ]; then
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
        else
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
        fi

        print_step "Verifying installation"
        if python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
            print_success "PyTorch installed successfully!"

            # Check for the critical file
            TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
            if [ -f "${TORCH_LIB_PATH}/libtorch_global_deps.so" ]; then
                print_success "libtorch_global_deps.so is present"
                echo ""
                print_success "Installation successful! You can now run: ./scripts/build_and_test.sh"
            else
                print_error "libtorch_global_deps.so is still missing"
                echo ""
                print_warning "Nightly reinstall did not resolve the issue."
                print_status "Recommend trying Option 2 (Python 3.12 downgrade)"
            fi
        else
            print_error "PyTorch installation failed"
            exit 1
        fi
        ;;

    2)
        print_step "Option 2: Python 3.12 Environment Setup"
        echo ""
        echo "This will guide you through creating a new Python 3.12 environment."
        echo ""
        echo "MANUAL STEPS REQUIRED:"
        echo ""
        echo "1. Backup current environment:"
        echo "   conda env export > /home/kp/autovoice/environment_backup_py${PYTHON_MAJOR}${PYTHON_MINOR}.yml"
        echo ""
        echo "2. Create new Python 3.12 environment:"
        echo "   conda create -n autovoice_py312 python=3.12 -y"
        echo ""
        echo "3. Activate new environment:"
        echo "   conda activate autovoice_py312"
        echo ""
        echo "4. Install stable PyTorch:"
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo "   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y"
        else
            echo "   conda install pytorch torchvision torchaudio cpuonly -c pytorch -y"
        fi
        echo ""
        echo "5. Install project dependencies:"
        echo "   cd /home/kp/autovoice"
        echo "   pip install -r requirements.txt"
        echo ""
        echo "6. Build CUDA extensions:"
        echo "   pip install -e ."
        echo ""
        echo "7. Verify installation:"
        echo "   python -c \"import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())\""
        echo ""

        # Offer to create a setup script
        read -p "Create a helper script for these steps? (Y/n): " create_script

        if [ "$create_script" != "n" ] && [ "$create_script" != "N" ]; then
            HELPER_SCRIPT="/home/kp/autovoice/scripts/setup_python312_helper.sh"
            cat > "$HELPER_SCRIPT" << 'EOFHELPER'
#!/bin/bash
# Helper script for Python 3.12 environment setup

set -e

echo "Backing up current environment..."
conda env export > /home/kp/autovoice/environment_backup_$(python --version | awk '{print $2}' | tr -d .).yml

echo "Creating Python 3.12 environment..."
conda create -n autovoice_py312 python=3.12 -y

echo ""
echo "Environment created! Now run:"
echo ""
echo "  conda activate autovoice_py312"
echo "  cd /home/kp/autovoice"
echo "  ./scripts/setup_pytorch_env.sh"
echo ""
EOFHELPER
            chmod +x "$HELPER_SCRIPT"
            print_success "Helper script created: ${HELPER_SCRIPT}"
            echo ""
            echo "Run the helper script to start:"
            echo "  ./scripts/setup_python312_helper.sh"
        fi
        ;;

    3)
        print_step "Option 3: Build PyTorch from Source"
        echo ""
        echo "This is an advanced option requiring:"
        echo "  - 2+ hours build time"
        echo "  - 20+ GB disk space"
        echo "  - Build tools (cmake, ninja, gcc)"
        echo ""
        echo "Detailed instructions: https://github.com/pytorch/pytorch#from-source"
        echo ""
        echo "Quick guide:"
        echo ""
        echo "1. Install build dependencies:"
        echo "   conda install cmake ninja numpy pyyaml setuptools cffi typing_extensions -y"
        echo "   conda install mkl mkl-include -y"
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo "   conda install -c pytorch magma-cuda121 -y"
        fi
        echo ""
        echo "2. Clone PyTorch:"
        echo "   cd /tmp"
        echo "   git clone --recursive https://github.com/pytorch/pytorch"
        echo "   cd pytorch"
        echo ""
        echo "3. Set environment variables:"
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo "   export USE_CUDA=1"
            echo "   export CUDA_HOME=/usr/local/cuda"
        fi
        echo "   export CMAKE_PREFIX_PATH=\${CONDA_PREFIX}"
        echo ""
        echo "4. Build and install:"
        echo "   python setup.py develop"
        echo ""
        print_warning "This process takes 2+ hours and requires significant resources"
        ;;

    q|Q)
        print_status "Operation cancelled"
        exit 0
        ;;

    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Setup script completed                                  ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
