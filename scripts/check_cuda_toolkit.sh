#!/bin/bash
# AutoVoice CUDA Toolkit Verification Script
#
# Pre-build verification of CUDA toolkit installation and system compatibility.
# Validates that the system has the required CUDA toolkit headers and libraries
# for building PyTorch CUDA extensions.

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
WARNING="⚠"
GEAR="⚙"

# Status tracking
ALL_CHECKS_PASSED=true

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        CUDA Toolkit Verification                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${BLUE}[${GEAR}]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[${CHECK}]${NC} $1"
}

print_error() {
    echo -e "${RED}[${CROSS}]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[${WARNING}]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[${INFO}]${NC} $1"
}

# Check 1: NVIDIA GPU presence
check_gpu_presence() {
    print_status "Checking for NVIDIA GPU..."

    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found in PATH"
        print_info "Install NVIDIA drivers and CUDA toolkit to enable GPU support"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)

    if [ -z "$GPU_INFO" ]; then
        print_error "No NVIDIA GPU detected by nvidia-smi"
        print_info "Ensure GPU is properly installed and recognized by system"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    # Parse GPU info
    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
    GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
    DRIVER_VERSION=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)

    print_success "NVIDIA GPU detected: ${GPU_NAME} (${GPU_MEMORY} MB)"
    print_info "Driver version: ${DRIVER_VERSION}"

    # Check compute capability requirement (>= 7.0 for CUDA 11.8+)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | sed 's/\.//')

    if [ "${COMPUTE_CAP}" -lt 70 ]; then
        print_error "GPU compute capability ${COMPUTE_CAP} is below minimum requirement (7.0)"
        print_info "AutoVoice requires CUDA compute capability >= 7.0 (Pascal architecture or newer)"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    print_info "GPU compute capability: ${COMPUTE_CAP} (meets minimum requirement ≥ 7.0)"

    return 0
}

# Check 2: CUDA Compiler (nvcc)
check_cuda_compiler() {
    print_status "Checking CUDA compiler (nvcc)..."

    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc not found in PATH"
        print_info "Install CUDA toolkit from https://developer.nvidia.com/cuda-toolkit"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    # Get CUDA version
    NVCC_OUTPUT=$(nvcc --version 2>/dev/null | grep "release")
    if [ -z "$NVCC_OUTPUT" ]; then
        print_error "Unable to determine CUDA version from nvcc output"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    CUDA_VERSION=$(echo "$NVCC_OUTPUT" | awk '{print $5}' | sed 's/,//')
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f2)

    print_success "CUDA compiler found: nvcc version ${CUDA_VERSION}"

    # Check minimum version (CUDA 11.8 required for PyTorch compatibility)
    if [ "$CUDA_MAJOR" -lt 11 ] || ([ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -lt 8 ]); then
        print_error "CUDA version ${CUDA_VERSION} is below minimum requirement (11.8)"
        print_info "AutoVoice requires CUDA 11.8+ for PyTorch compatibility"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    print_info "CUDA version ${CUDA_VERSION} meets minimum requirement (≥ 11.8)"

    return 0
}

# Check 3: CUDA Toolkit installation
check_cuda_toolkit_installation() {
    print_status "Checking CUDA toolkit installation..."

    # Check CUDA_HOME environment variable
    if [ -z "$CUDA_HOME" ]; then
        print_warning "CUDA_HOME environment variable not set"
        # Try to infer from nvcc location
        NVCC_PATH=$(which nvcc)
        if [ -n "$NVCC_PATH" ]; then
            INFERRED_CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
            print_info "Inferred CUDA_HOME: ${INFERRED_CUDA_HOME}"
            export CUDA_HOME="$INFERRED_CUDA_HOME"
        else
            print_error "Unable to determine CUDA toolkit location"
            ALL_CHECKS_PASSED=false
            return 1
        fi
    else
        print_info "CUDA_HOME: ${CUDA_HOME}"
    fi

    # Verify CUDA_HOME directory exists
    if [ ! -d "$CUDA_HOME" ]; then
        print_error "CUDA_HOME directory does not exist: ${CUDA_HOME}"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    # Check for critical CUDA directories
    CRITICAL_DIRS=("include" "lib64")
    for dir in "${CRITICAL_DIRS[@]}"; do
        if [ ! -d "${CUDA_HOME}/${dir}" ]; then
            print_error "CUDA toolkit missing '${dir}' directory: ${CUDA_HOME}/${dir}"
            ALL_CHECKS_PASSED=false
        else
            print_success "CUDA toolkit directory exists: ${dir}/"
        fi
    done

    # Check for critical header files
    CRITICAL_HEADERS=("cuda.h" "cufft.h" "cudart.h")
    for header in "${CRITICAL_HEADERS[@]}"; do
        if [ ! -f "${CUDA_HOME}/include/${header}" ]; then
            print_error "CUDA header file missing: ${header}"
            print_info "Check CUDA toolkit installation or update PATH"
            ALL_CHECKS_PASSED=false
        else
            print_info "CUDA header found: ${header}"
        fi
    done

    # Check for critical libraries
    CRITICAL_LIBS=("libcudart.so" "libcufft.so")
    for lib in "${CRITICAL_LIBS[@]}"; do
        if [ ! -f "${CUDA_HOME}/lib64/${lib}" ]; then
            print_error "CUDA library missing: ${lib}"
            print_info "Check CUDA toolkit installation or update LD_LIBRARY_PATH"
            ALL_CHECKS_PASSED=false
        else
            print_info "CUDA library found: ${lib}"
        fi
    done

    return 0
}

# Check 4: CUDA Runtime and Driver Compatibility
check_cuda_runtime_compatibility() {
    print_status "Checking CUDA runtime/driver compatibility..."

    # Get driver version
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    if [ -z "$DRIVER_VERSION" ]; then
        print_error "Unable to determine NVIDIA driver version"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    # Get CUDA runtime version
    RUNTIME_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//' | cut -d'.' -f1-2 || echo "unknown")

    if [ "$RUNTIME_VERSION" = "unknown" ]; then
        print_warning "Unable to determine CUDA runtime version from nvcc"
        print_info "Driver version: ${DRIVER_VERSION}"
        return 0
    fi

    # Basic compatibility check (simplified)
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
    RUNTIME_MAJOR=$(echo "$RUNTIME_VERSION" | cut -d'.' -f1)

    if [ "$DRIVER_MAJOR" -lt "$RUNTIME_MAJOR" ]; then
        print_warning "Driver version ${DRIVER_VERSION} may be incompatible with CUDA runtime ${RUNTIME_VERSION}"
        print_info "Consider updating NVIDIA drivers for optimal compatibility"
    else
        print_success "CUDA runtime/driver versions appear compatible"
        print_info "Driver: ${DRIVER_VERSION}, CUDA Runtime: ${RUNTIME_VERSION}"
    fi

    return 0
}

# Check 5: System CUDA Build Headers
check_build_headers() {
    print_status "Checking system CUDA build headers..."

    # Check for nv/target header (critical for PyTorch CUDA extensions)
    if [ ! -f "${CUDA_HOME}/include/nv/target" ]; then
        print_error "Missing critical header: nv/target"
        print_info "This header is required for building CUDA extensions"
        print_info "The conda CUDA toolkit may be missing this header"
        ALL_CHECKS_PASSED=false
        return 1
    else
        print_success "Found critical header: nv/target"
    fi

    # Additional runtime library compatibility check
    if [ ! -f "${CUDA_HOME}/include/cuda_runtime.h" ]; then
        print_error "Missing CUDA runtime header: cuda_runtime.h"
        print_info "Required for runtime compilation of CUDA kernels"
        ALL_CHECKS_PASSED=false
        return 1
    else
        print_info "CUDA runtime header found: cuda_runtime.h"
    fi

    return 0
}

# Check 6: cuDNN presence (optional but recommended)
check_cudnn() {
    print_status "Checking cuDNN installation (optional)..."

    # Common cuDNN library locations
    CUDNN_LIB_PATHS=("${CUDA_HOME}/lib64/libcudnn.so" "/usr/lib/x86_64-linux-gnu/libcudnn.so")

    CUDNN_FOUND=false
    for lib_path in "${CUDNN_LIB_PATHS[@]}"; do
        if [ -f "$lib_path" ]; then
            print_success "cuDNN library found: ${lib_path}"
            CUDNN_FOUND=true
            break
        fi
    done

    if [ "$CUDNN_FOUND" = false ]; then
        print_warning "cuDNN not found"
        print_info "cuDNN is recommended but not required for basic CUDA functionality"
        print_info "Download from: https://developer.nvidia.com/cudnn"
    fi

    return 0
}

# Check 7: Python CUDA Support
check_python_cuda() {
    print_status "Checking Python PyTorch CUDA support..."

    if ! command -v python &> /dev/null; then
        print_error "Python not found in PATH"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    # Check PyTorch installation
    if ! python -c "import torch" 2>/dev/null; then
        print_error "PyTorch not installed or not available in Python environment"
        print_info "Install PyTorch with: pip install torch torchvision torchaudio"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    # Check CUDA availability in PyTorch
    CUDA_PYTORCH=$(python -c "import torch; print('true' if torch.cuda.is_available() else 'false')" 2>/dev/null)

    if [ "$CUDA_PYTORCH" = "true" ]; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        PYTORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)

        print_success "PyTorch CUDA support available"
        print_info "GPU count: ${GPU_COUNT}, Device name: ${GPU_NAME}"
        print_info "PyTorch CUDA version: ${PYTORCH_CUDA_VER}"
    else
        print_error "PyTorch CUDA support not available"
        print_info "Ensure PyTorch was installed with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        print_info "Or run: ./scripts/install_cuda_toolkit.sh --pytorch-cuda"
        ALL_CHECKS_PASSED=false
        return 1
    fi

    return 0
}

# Main execution
main() {
    echo ""
    print_info "Running comprehensive CUDA toolkit verification..."
    echo ""

    # Run all checks
    check_gpu_presence
    echo ""

    check_cuda_compiler
    echo ""

    check_cuda_toolkit_installation
    echo ""

    check_cuda_runtime_compatibility
    echo ""

    check_build_headers
    echo ""

    check_cudnn
    echo ""

    check_python_cuda
    echo ""

    # Summary
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  CUDA Toolkit Verification Summary${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo ""

    if [ "$ALL_CHECKS_PASSED" = true ]; then
        print_success "All CUDA toolkit checks passed!"
        print_info "System is ready for CUDA extension building"
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  CUDA ENVIRONMENT READY                              ║${NC}"
        echo -e "${GREEN}║  Run: pip install -e .                               ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
        return 0
    else
        echo ""
        print_error "Some CUDA toolkit checks failed - building may not work"
        echo ""
        echo -e "${YELLOW}╔════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  RECOMMENDED FIXES                                    ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════════════════════╝${NC}"
        echo ""
        print_info "1. Install system CUDA toolkit: ./scripts/install_cuda_toolkit.sh"
        print_info "2. Fix PyTorch CUDA: ./scripts/setup_pytorch_env.sh"
        print_info "3. Verify GPU drivers: nvidia-smi"
        echo ""
        return 1
    fi
}

# Run main function
main "$@"
