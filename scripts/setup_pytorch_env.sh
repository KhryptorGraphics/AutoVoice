#!/bin/bash
# AutoVoice PyTorch Environment Setup Script
# Automated resolution for PyTorch library issues

set -e

# Compute project root dynamically for portability
PROJECT_ROOT=$(cd "$(dirname "$0")"/.. && pwd)

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
        pip uninstall torch torchvision torchaudio functorch -y 2>/dev/null || true

        # Use torch-aware path discovery for targeted cleanup
        # Only remove specific torch packages, not all torch* (to preserve torchmetrics, torchtext, etc.)
        TORCH_PARENT=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent.parent)" 2>/dev/null || echo "")
        if [ -n "$TORCH_PARENT" ] && [ -d "$TORCH_PARENT" ]; then
            print_status "Removing torch packages from: ${TORCH_PARENT}"
            # Remove only core PyTorch packages explicitly
            rm -rf "${TORCH_PARENT}/torch" 2>/dev/null || true
            rm -rf "${TORCH_PARENT}/torch-"* 2>/dev/null || true
            rm -rf "${TORCH_PARENT}/torchvision" 2>/dev/null || true
            rm -rf "${TORCH_PARENT}/torchvision-"* 2>/dev/null || true
            rm -rf "${TORCH_PARENT}/torchaudio" 2>/dev/null || true
            rm -rf "${TORCH_PARENT}/torchaudio-"* 2>/dev/null || true
            rm -rf "${TORCH_PARENT}/functorch" 2>/dev/null || true
            rm -rf "${TORCH_PARENT}/functorch-"* 2>/dev/null || true
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

        # Check if conda is available
        if ! command -v conda &> /dev/null; then
            print_error "Conda is not installed or not in PATH"
            echo ""
            echo "Option 2 requires conda for environment management."
            echo ""
            echo "ALTERNATIVES:"
            echo ""
            echo "1. Install Miniconda (recommended):"
            echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            echo "   bash Miniconda3-latest-Linux-x86_64.sh"
            echo "   source ~/.bashrc"
            echo "   # Then re-run this script"
            echo ""
            echo "2. Use pip + venv fallback (manual setup):"
            echo "   python3.12 -m venv ${PROJECT_ROOT}_py312_venv"
            echo "   source ${PROJECT_ROOT}_py312_venv/bin/activate"
            echo "   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \\"
            echo "     --index-url https://download.pytorch.org/whl/cu121"
            echo "   pip install -r ${PROJECT_ROOT}/requirements.txt"
            echo "   pip install -e ${PROJECT_ROOT}"
            echo ""
            echo "For more information, see:"
            echo "  - https://docs.conda.io/en/latest/miniconda.html"
            echo "  - PYTORCH_ENVIRONMENT_FIX_REPORT.md"
            echo ""
            exit 1
        fi

        echo "This will guide you through creating a new Python 3.12 environment."
        echo ""
        echo "MANUAL STEPS REQUIRED:"
        echo ""
        echo "1. Backup current environment:"
        echo "   conda env export > ${PROJECT_ROOT}/environment_backup_py${PYTHON_MAJOR}${PYTHON_MINOR}.yml"
        echo ""
        echo "2. Create new Python 3.12 environment:"
        echo "   conda create -n autovoice_py312 python=3.12 -y"
        echo ""
        echo "3. Activate new environment:"
        echo "   conda activate autovoice_py312"
        echo ""
        echo "4. Install stable PyTorch (RECOMMENDED - pip method for reliability):"
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo "   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
            echo ""
            echo "   Alternative (conda):"
            echo "   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y"
        else
            echo "   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu"
            echo ""
            echo "   Alternative (conda):"
            echo "   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch -y"
        fi
        echo ""
        echo "5. Install project dependencies:"
        echo "   cd ${PROJECT_ROOT}"
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
            HELPER_SCRIPT="${PROJECT_ROOT}/scripts/setup_python312_helper.sh"

            # Determine CUDA installation method based on availability
            if [ "$CUDA_AVAILABLE" = true ]; then
                PYTORCH_INSTALL_CMD="conda run -n autovoice_py312 pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
                PYTORCH_INSTALL_ALT="# Alternative (conda): conda run -n autovoice_py312 conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y"
            else
                PYTORCH_INSTALL_CMD="conda run -n autovoice_py312 pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu"
                PYTORCH_INSTALL_ALT="# Alternative (conda): conda run -n autovoice_py312 conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch -y"
            fi

            cat > "$HELPER_SCRIPT" << EOFHELPER
#!/bin/bash
# Helper script for Python 3.12 environment setup with automated PyTorch installation
# Generated by setup_pytorch_env.sh

set -e

# Compute project root dynamically for portability
PROJECT_ROOT=\$(cd "\$(dirname "\$0")"/.. && pwd)

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

# Function to print status messages
print_status() {
    echo -e "\${BLUE}[\${INFO}]\${NC} \$1"
}

print_success() {
    echo -e "\${GREEN}[\${CHECK}]\${NC} \$1"
}

print_error() {
    echo -e "\${RED}[\${CROSS}]\${NC} \$1"
}

print_warning() {
    echo -e "\${YELLOW}[!]\${NC} \$1"
}

echo -e "\${BLUE}╔════════════════════════════════════════════════════════╗\${NC}"
echo -e "\${BLUE}║  AutoVoice Python 3.12 Environment Setup Helper       ║\${NC}"
echo -e "\${BLUE}╚════════════════════════════════════════════════════════╝\${NC}"
echo ""

# Step 1: Backup current environment
print_status "[1/6] Backing up current environment..."
conda env export > \${PROJECT_ROOT}/environment_backup_\$(python --version 2>&1 | awk '{print \$2}' | tr -d .).yml 2>/dev/null || true
print_success "Backup saved"

# Step 2: Create or verify Python 3.12 environment
echo ""
print_status "[2/6] Checking for autovoice_py312 environment..."
if conda env list | grep -q "^autovoice_py312 "; then
    print_warning "Environment autovoice_py312 already exists, skipping creation"
else
    print_status "Creating Python 3.12 environment..."
    conda create -n autovoice_py312 python=3.12 -y
    print_success "Environment created"
fi

# Step 3: Install PyTorch with CUDA support
echo ""
print_status "[3/6] Installing PyTorch 2.5.1 with CUDA 12.1 support..."
echo ""
print_status "Running: ${PYTORCH_INSTALL_CMD}"
${PYTORCH_INSTALL_CMD}
print_success "PyTorch installed successfully"
echo ""
print_status "Alternative installation method (if needed):"
echo "  ${PYTORCH_INSTALL_ALT}"

# Step 4: Install project dependencies
echo ""
print_status "[4/6] Installing project dependencies from requirements.txt..."
conda run -n autovoice_py312 pip install -r \${PROJECT_ROOT}/requirements.txt
print_success "Dependencies installed successfully"

# Step 5: Verify PyTorch installation and check for libtorch_global_deps.so
echo ""
print_status "[5/6] Verifying PyTorch installation..."
echo ""
conda run -n autovoice_py312 python -c "import torch, os, importlib.util as iu; p=os.path.join(os.path.dirname(torch.__file__),'lib','libtorch_global_deps.so'); print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('libtorch_global_deps.so exists:', os.path.exists(p)); print('Library path:', p if os.path.exists(p) else 'NOT FOUND')"
echo ""
print_success "Verification complete"

# Step 6: Print next steps
echo ""
print_status "[6/6] Setup complete! Next steps:"
echo ""
echo -e "\${BLUE}════════════════════════════════════════════════════════\${NC}"
echo -e "\${GREEN}  Environment Setup Complete!\${NC}"
echo -e "\${BLUE}════════════════════════════════════════════════════════\${NC}"
echo ""
echo "To use the new environment:"
echo "  1. Activate the environment:"
echo "     conda activate autovoice_py312"
echo ""
echo "  2. (Optional) Build CUDA extensions after installing CUDA toolkit:"
echo "     cd \${PROJECT_ROOT}"
echo "     pip install -e ."
echo ""
echo "  3. Run tests to verify everything works:"
echo "     pytest tests/"
echo ""
echo -e "\${BLUE}════════════════════════════════════════════════════════\${NC}"
echo ""
echo "For detailed instructions, see:"
echo "  - docs/pytorch_library_issue.md"
echo "  - PYTORCH_ENVIRONMENT_FIX_REPORT.md"
echo ""
print_success "All automated steps completed successfully!"
echo ""
EOFHELPER
            chmod +x "$HELPER_SCRIPT"
            print_success "Helper script created: ${HELPER_SCRIPT}"
            echo ""
            echo "Run the helper script to start automated setup:"
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
