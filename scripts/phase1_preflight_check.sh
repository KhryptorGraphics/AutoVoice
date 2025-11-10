#!/bin/bash

# Phase 1 Pre-Flight Check
# Verifies current environment state before proceeding with Phase 1 execution

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Symbols
CHECK="✅"
CROSS="❌"
WARN="⚠️"
INFO="ℹ️"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Status tracking
READY_TO_PROCEED=true
ISSUES_FOUND=()
COMPLETED_ITEMS=()
TODO_ITEMS=()

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Phase 1 Pre-Flight Check                          ║${NC}"
echo -e "${BLUE}║  Verifying Environment State Before Execution             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section header
print_section() {
    echo -e "\n${BLUE}━━━ $1 ━━━${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
    COMPLETED_ITEMS+=("$1")
}

# Function to print error
print_error() {
    echo -e "${RED}${CROSS} $1${NC}"
    ISSUES_FOUND+=("$1")
    READY_TO_PROCEED=false
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}${WARN} $1${NC}"
    TODO_ITEMS+=("$1")
}

# Function to print info
print_info() {
    echo -e "${BLUE}${INFO} $1${NC}"
}

# Check 1: Python Version
print_section "Python Version Check"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ "$PYTHON_MAJOR" == "3" && "$PYTHON_MINOR" == "12" ]]; then
        print_success "Python $PYTHON_VERSION (3.12.x required)"
    elif [[ "$PYTHON_MAJOR" == "3" && "$PYTHON_MINOR" == "13" ]]; then
        print_error "Python $PYTHON_VERSION detected (3.13 not compatible, need 3.12)"
        print_info "Run: conda activate autovoice_py312"
    else
        print_warning "Python $PYTHON_VERSION (expected 3.12.x)"
    fi
else
    print_error "Python not found in PATH"
fi

# Check 2: Conda Environment
print_section "Conda Environment Check"
if command -v conda &> /dev/null; then
    if conda env list | grep -q "autovoice_py312"; then
        print_success "Conda environment 'autovoice_py312' exists"
        
        if [[ "$CONDA_DEFAULT_ENV" == "autovoice_py312" ]]; then
            print_success "Environment 'autovoice_py312' is currently active"
        else
            print_warning "Environment 'autovoice_py312' exists but not active"
            print_info "Run: conda activate autovoice_py312"
        fi
    else
        print_error "Conda environment 'autovoice_py312' not found"
        print_info "Run: ./scripts/setup_pytorch_env.sh and select Option 2"
    fi
else
    print_error "Conda not found in PATH"
fi

# Check 3: PyTorch Installation
print_section "PyTorch Installation Check"
if python -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    print_success "PyTorch installed: $PYTORCH_VERSION"
    
    if [[ "$PYTORCH_VERSION" == *"2.5.1"* ]]; then
        print_success "PyTorch version 2.5.1 (correct version)"
    else
        print_warning "PyTorch version $PYTORCH_VERSION (expected 2.5.1+cu121)"
    fi
    
    if [[ "$PYTORCH_VERSION" == *"+cu121"* ]]; then
        print_success "PyTorch built with CUDA 12.1 support"
    else
        print_warning "PyTorch version doesn't show +cu121 suffix"
    fi
else
    print_error "PyTorch not installed"
    print_info "Run: pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121"
fi

# Check 4: libtorch_global_deps.so
print_section "PyTorch CUDA Runtime Check"
# Use $CONDA_PREFIX if set, otherwise search common locations
if [ -n "$CONDA_PREFIX" ]; then
    LIBTORCH_SO=$(find "$CONDA_PREFIX" -name "libtorch_global_deps.so" 2>/dev/null | head -n 1)
else
    # Shallow search under $HOME for common conda installations
    LIBTORCH_SO=$(find "$HOME" -maxdepth 4 -type f -name "libtorch_global_deps.so" 2>/dev/null | head -n 1)
fi

if [[ -n "$LIBTORCH_SO" ]]; then
    print_success "libtorch_global_deps.so found at: $LIBTORCH_SO"
else
    print_warning "libtorch_global_deps.so not found (may not be critical)"
fi

# Check 5: PyTorch CUDA Availability
print_section "PyTorch CUDA Availability Check"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    print_success "torch.cuda.is_available() = True"
    
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_success "GPU detected: $GPU_NAME"
    
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    print_info "PyTorch CUDA version: $CUDA_VERSION"
else
    print_error "torch.cuda.is_available() = False"
    print_info "PyTorch CUDA support not working"
fi

# Check 6: CUDA Toolkit (nvcc)
print_section "CUDA Toolkit Check"
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    print_success "nvcc found: version $NVCC_VERSION"
else
    print_warning "nvcc not found in PATH"
    print_info "System CUDA toolkit may not be installed"
fi

# Check 7: CUDA_HOME
if [[ -n "$CUDA_HOME" ]]; then
    print_success "CUDA_HOME set to: $CUDA_HOME"
else
    print_warning "CUDA_HOME not set"
fi

# Check 8: Critical Header (nv/target)
print_section "CUDA Headers Check"
HEADER_FOUND=false

# Check locations - including conda-style targets paths
HEADER_LOCATIONS=(
    "$CUDA_HOME/include/nv/target"
    "/usr/local/cuda/include/nv/target"
    "/usr/local/cuda-12.1/include/nv/target"
    "$HOME/miniconda3/envs/autovoice_py312/include/nv/target"
)

# Add globbed matches for conda-style targets paths
if [[ -n "$CUDA_HOME" ]]; then
    for target_path in "$CUDA_HOME"/targets/*/include/nv/target; do
        if [[ -f "$target_path" ]]; then
            HEADER_LOCATIONS+=("$target_path")
        fi
    done
fi

if [[ -n "$CONDA_PREFIX" ]]; then
    for target_path in "$CONDA_PREFIX"/targets/*/include/nv/target; do
        if [[ -f "$target_path" ]]; then
            HEADER_LOCATIONS+=("$target_path")
        fi
    done
fi

for HEADER_PATH in "${HEADER_LOCATIONS[@]}"; do
    if [[ -f "$HEADER_PATH" ]]; then
        print_success "Critical header found: $HEADER_PATH"
        HEADER_FOUND=true
        break
    fi
done

if [[ "$HEADER_FOUND" == false ]]; then
    print_warning "Critical header 'nv/target' not found"
    print_info "This is expected - CUDA toolkit installation needed"
    TODO_ITEMS+=("Install system CUDA toolkit with complete headers")
fi

# Summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Summary Report                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ ${#COMPLETED_ITEMS[@]} -gt 0 ]]; then
    echo -e "${GREEN}${CHECK} Already Complete (${#COMPLETED_ITEMS[@]} items):${NC}"
    for item in "${COMPLETED_ITEMS[@]}"; do
        echo -e "  ${GREEN}•${NC} $item"
    done
    echo ""
fi

if [[ ${#TODO_ITEMS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}${WARN} Needs Action (${#TODO_ITEMS[@]} items):${NC}"
    for item in "${TODO_ITEMS[@]}"; do
        echo -e "  ${YELLOW}•${NC} $item"
    done
    echo ""
fi

if [[ ${#ISSUES_FOUND[@]} -gt 0 ]]; then
    echo -e "${RED}${CROSS} Critical Issues (${#ISSUES_FOUND[@]} items):${NC}"
    for item in "${ISSUES_FOUND[@]}"; do
        echo -e "  ${RED}•${NC} $item"
    done
    echo ""
fi

# Recommendations
echo -e "${BLUE}${INFO} Recommended Next Steps:${NC}"
echo ""

if [[ "$READY_TO_PROCEED" == true ]]; then
    echo -e "${GREEN}1. Install system CUDA toolkit:${NC}"
    echo -e "   ./scripts/install_cuda_toolkit.sh"
    echo ""
    echo -e "${GREEN}2. Build CUDA extensions:${NC}"
    echo -e "   pip install -e ."
    echo ""
    echo -e "${GREEN}3. Verify bindings:${NC}"
    echo -e "   ./scripts/verify_bindings.py"
    echo ""
    echo -e "${GREEN}Or run automated execution:${NC}"
    echo -e "   ./scripts/phase1_execute.sh"
else
    echo -e "${RED}Fix critical issues before proceeding:${NC}"
    for item in "${ISSUES_FOUND[@]}"; do
        echo -e "  ${RED}•${NC} $item"
    done
fi

echo ""
if [[ "$READY_TO_PROCEED" == true ]]; then
    echo -e "${GREEN}${CHECK} Pre-flight check passed - ready to proceed!${NC}"
    exit 0
else
    echo -e "${RED}${CROSS} Pre-flight check failed - fix issues before proceeding${NC}"
    exit 1
fi

