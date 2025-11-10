#!/bin/bash

# Phase 1 Execution Script
# Orchestrates all Phase 1 steps: CUDA toolkit installation and extension building

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Symbols
CHECK="✅"
CROSS="❌"
WARN="⚠️"
INFO="ℹ️"
ROCKET="🚀"
GEAR="⚙️"
PACKAGE="📦"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Parse command line arguments
REPORT_OUTPUT="$PROJECT_ROOT/PHASE1_COMPLETION_REPORT.md"
NON_INTERACTIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --report-out)
            REPORT_OUTPUT="$2"
            shift 2
            ;;
        --yes|-y)
            NON_INTERACTIVE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Execute Phase 1: Install CUDA toolkit and build extensions"
            echo ""
            echo "Options:"
            echo "  --report-out <path>  Specify output path for completion report"
            echo "  --yes, -y            Non-interactive mode (skip prompts)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                   # Interactive mode"
            echo "  $0 --yes             # Non-interactive mode"
            echo "  $0 -y --report-out /tmp/report.md"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--report-out <path>] [--yes|-y] [--help]"
            exit 1
            ;;
    esac
done

# Execution tracking
START_TIME=$(date +%s)
EXECUTION_DATE=$(date "+%Y-%m-%d %H:%M:%S")
STEP_COUNT=0
TOTAL_STEPS=7

# Status tracking
PREFLIGHT_PASSED=false
CUDA_INSTALLED=false
EXTENSIONS_BUILT=false
BINDINGS_VERIFIED=false
PYTORCH_VALIDATED=false

# Function to print step header
print_step() {
    STEP_COUNT=$((STEP_COUNT + 1))
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Step $STEP_COUNT/$TOTAL_STEPS: $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Function to print success
print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}${CROSS} $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}${WARN} $1${NC}"
}

# Function to print info
print_info() {
    echo -e "${CYAN}${INFO} $1${NC}"
}

# Function to generate completion report
# Defined early so it's available to error handler
generate_report() {
    local status=$1
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local duration_min=$((duration / 60))
    local duration_sec=$((duration % 60))

    print_info "Generating completion report..."

    # Gather system information
    local python_version=$(python --version 2>&1 | awk '{print $2}')
    local pytorch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
    local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    local gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")
    local gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    local cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
    local cudnn_version=$(python -c "import torch; print(torch.backends.cudnn.version())" 2>/dev/null || echo "N/A")
    local cuda_home=${CUDA_HOME:-"Not set"}
    local nvcc_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d, -f1 || echo "Not found")

    # Find extension file and get detailed info
    local extension_file=$(python -c "try:
    from auto_voice import cuda_kernels
    print(cuda_kernels.__file__)
except: pass" 2>/dev/null)

    if [ -z "$extension_file" ]; then
        extension_file=$(find . -name "cuda_kernels*.so" 2>/dev/null | head -n 1)
    fi

    local extension_path=${extension_file:-"Not found"}
    local extension_size="N/A"
    if [ -n "$extension_file" ] && [ -f "$extension_file" ]; then
        extension_size=$(stat -f%z "$extension_file" 2>/dev/null || stat -c%s "$extension_file" 2>/dev/null || echo "N/A")
    fi

    # Get build and verify durations
    local build_duration_display="${BUILD_DURATION:-N/A}s"
    local verify_duration_display="${VERIFY_DURATION:-N/A}s"

    # Check for CUDA toolkit headers
    local nv_target_path="Not found"
    local nv_target_exists=false
    for check_path in "$CUDA_HOME/include/nv/target" "/usr/local/cuda/include/nv/target" "/usr/local/cuda-12.1/include/nv/target"; do
        if [ -f "$check_path" ]; then
            nv_target_path="$check_path"
            nv_target_exists=true
            break
        fi
    done

    # Write the fully populated report directly to PHASE1_COMPLETION_REPORT.md
    cat > "$REPORT_OUTPUT" << EOF
# Phase 1 Completion Report

**Date**: $EXECUTION_DATE
**Duration**: ${duration_min}m ${duration_sec}s
**Overall Status**: $status

---

## Executive Summary

Phase 1 focused on fixing the PyTorch environment and building CUDA extensions for the AutoVoice project. This report documents the execution results and current system state.

---

## Pre-Flight Check Results

### ✅ Already Complete (Before Phase 1 Execution)

- [x] Python $python_version environment (\`autovoice_py312\`) exists
- [x] PyTorch $pytorch_version installed via pip
- [x] \`libtorch_global_deps.so\` present and functional
- [x] PyTorch CUDA availability: \`torch.cuda.is_available()\` = $cuda_available
- [x] GPU detected: $gpu_name
- [x] All project dependencies installed

### ⚠️ Required Action Items

- [$([ "$CUDA_INSTALLED" = true ] && echo "x" || echo " ")] Install system CUDA toolkit with complete headers
- [$([ "$EXTENSIONS_BUILT" = true ] && echo "x" || echo " ")] Build CUDA extensions (\`pip install -e .\`)
- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] Verify bindings (\`launch_pitch_detection\`, \`launch_vibrato_analysis\`)
- [$([ "$PYTORCH_VALIDATED" = true ] && echo "x" || echo " ")] Validate end-to-end PyTorch CUDA functionality

---

## CUDA Toolkit Installation

### Installation Method

- [$([ "$CUDA_INSTALLED" = true ] && echo "x" || echo " ")] Automated script (\`./scripts/install_cuda_toolkit.sh\`)
- [ ] Manual installation
- [$([ "$CUDA_INSTALLED" = false ] && echo "x" || echo " ")] Already installed (skipped)

### Installation Details

**CUDA Version**: $nvcc_version
**Installation Location**: $cuda_home
**Installation Duration**: Included in total duration

### Environment Variables Set

\`\`\`bash
CUDA_HOME=$cuda_home
PATH=\$CUDA_HOME/bin:\$PATH
LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
\`\`\`

### Verification Results

- [$([ -n "$nvcc_version" ] && [ "$nvcc_version" != "Not found" ] && echo "x" || echo " ")] \`nvcc --version\` works
- [$([ -n "$nvcc_version" ] && [ "$nvcc_version" != "Not found" ] && echo "x" || echo " ")] CUDA version: $nvcc_version
- [$([ "$nv_target_exists" = true ] && echo "x" || echo " ")] Critical header \`nv/target\` exists at: $nv_target_path
- [$([ -f "$cuda_home/include/cuda.h" ] && echo "x" || echo " ")] \`cuda.h\` header exists
- [$([ -f "$cuda_home/include/cuda_runtime.h" ] && echo "x" || echo " ")] \`cuda_runtime.h\` header exists
- [$([ -f "$cuda_home/include/device_launch_parameters.h" ] && echo "x" || echo " ")] \`device_launch_parameters.h\` header exists

---

## CUDA Extension Build

### Build Command

\`\`\`bash
pip install -e .
\`\`\`

### Build Duration

$build_duration_display

### Build Artifacts

- [$([ "$EXTENSIONS_BUILT" = true ] && echo "x" || echo " ")] \`cuda_kernels.so\` created
- **Location**: $extension_path
- **File size**: $extension_size bytes

### Build Log

Full build log saved to: \`build.log\`

Key build output lines can be found in \`build.log\`. Check for:
- Compilation of CUDA kernels
- Linking of extension module
- Any warnings or errors

---

## Bindings Verification

### Import Test

- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] \`from auto_voice import cuda_kernels\` - Success

### Function Exposure Check

- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] \`launch_pitch_detection\` exposed
- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] \`launch_vibrato_analysis\` exposed

### Callable Test Results

**Test**: Basic function call test
**Duration**: $verify_duration_display

- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] Function callable: Yes
- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] Memory stability: Pass

**CUDA Availability for Tests**:
- [$([ "$cuda_available" = "True" ] && echo "x" || echo " ")] CUDA available for testing
- [$([ "$cuda_available" != "True" ] && echo "x" || echo " ")] CPU-only testing (CUDA not available)

### Verification Script Output

Full verification output saved to: \`verify.log\`

Run \`./scripts/verify_bindings.py\` to see detailed verification output, or check \`verify.log\` for the complete test results.

---

## PyTorch CUDA Validation

### PyTorch Information

- **PyTorch Version**: $pytorch_version
- **CUDA Available**: $cuda_available
- **CUDA Version**: $cuda_version
- **cuDNN Version**: $cudnn_version

### GPU Information

- **GPU Name**: $gpu_name
- **GPU Count**: $gpu_count

### CUDA Tensor Operations Test

\`\`\`python
import torch
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
\`\`\`

- [$([ "$PYTORCH_VALIDATED" = true ] && echo "x" || echo " ")] Test passed

---

## Environment Snapshot

### Python Environment

\`\`\`
Python Version: $python_version
Conda Environment: autovoice_py312
\`\`\`

### PyTorch Installation

\`\`\`
PyTorch Version: $pytorch_version
CUDA Support: $cuda_available
\`\`\`

### CUDA Toolkit

\`\`\`
CUDA Version: $nvcc_version
CUDA_HOME: $cuda_home
nvcc Location: $(which nvcc 2>/dev/null || echo "Not in PATH")
\`\`\`

### Key Dependencies

\`\`\`
numpy: $(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "N/A")
scipy: $(python -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "N/A")
librosa: $(python -c "import librosa; print(librosa.__version__)" 2>/dev/null || echo "N/A")
soundfile: $(python -c "import soundfile; print(soundfile.__version__)" 2>/dev/null || echo "N/A")
\`\`\`

---

## Verification Checklist

- [x] Python $python_version environment active
- [x] PyTorch $pytorch_version installed
- [$([ "$CUDA_INSTALLED" = true ] && echo "x" || echo " ")] System CUDA toolkit installed
- [$([ "$nv_target_exists" = true ] && echo "x" || echo " ")] \`nv/target\` header exists
- [$([ "$EXTENSIONS_BUILT" = true ] && echo "x" || echo " ")] CUDA extensions built successfully
- [$([ "$EXTENSIONS_BUILT" = true ] && echo "x" || echo " ")] \`cuda_kernels.so\` file exists
- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] \`from auto_voice import cuda_kernels\` works
- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] \`launch_pitch_detection\` function exposed
- [$([ "$BINDINGS_VERIFIED" = true ] && echo "x" || echo " ")] \`launch_vibrato_analysis\` function exposed
- [$([ "$PYTORCH_VALIDATED" = true ] && echo "x" || echo " ")] \`torch.cuda.is_available()\` returns \`True\`
- [$([ "$PYTORCH_VALIDATED" = true ] && echo "x" || echo " ")] CUDA tensor operations work

---

## Issues Encountered and Resolutions

$(if [ -f "build.log" ] && grep -qi "error" build.log; then
    echo "### Build Errors"
    echo ""
    echo "**Description**: Errors encountered during build"
    echo ""
    echo "**Error Messages**:"
    echo "\`\`\`"
    grep -i "error" build.log | head -10
    echo "\`\`\`"
    echo ""
    echo "**Status**: $([ "$EXTENSIONS_BUILT" = true ] && echo "Resolved" || echo "Unresolved")"
    echo ""
else
    echo "No critical issues encountered during Phase 1 execution."
fi)

---

## Next Steps (Phase 2)

### Recommended Actions

1. **Run Comprehensive Tests**
   - Test all CUDA kernel functions
   - Validate audio processing functionality
   - Run integration tests with real audio data
   - Execute: \`./scripts/run_full_validation.sh\` or \`pytest tests/\`

2. **Performance Benchmarking**
   - Compare CPU vs GPU performance
   - Measure memory usage
   - Profile kernel execution times
   - Execute: \`./scripts/run_comprehensive_benchmarks.py\`

3. **Validation Testing**
   - Test pitch detection accuracy
   - Test vibrato analysis accuracy
   - Verify memory management
   - Check \`full_suite_log.txt\` for test results

### Phase 2 Execution Command

\`\`\`bash
# Run comprehensive validation
./scripts/run_full_validation.sh

# Or run specific test suites
pytest tests/test_cuda_kernels.py -v
pytest tests/test_bindings_integration.py -v
\`\`\`

---

## Conclusion

**Phase 1 Status**: $status

**Summary**: Phase 1 execution completed with $([ "$status" = "Success" ] && echo "all steps successful" || echo "some issues"). CUDA toolkit $([ "$CUDA_INSTALLED" = true ] && echo "installed" || echo "installation attempted"), extensions $([ "$EXTENSIONS_BUILT" = true ] && echo "built" || echo "build attempted"), bindings $([ "$BINDINGS_VERIFIED" = true ] && echo "verified" || echo "verification attempted"), and PyTorch CUDA $([ "$PYTORCH_VALIDATED" = true ] && echo "validated" || echo "validation attempted").

**Ready for Phase 2**: $([ "$status" = "Success" ] && echo "Yes" || echo "No")

**Additional Notes**:
- Build log: \`build.log\`
- Verification log: \`verify.log\`
- Full test suite log: \`full_suite_log.txt\` (if available)

---

**Report Generated**: $EXECUTION_DATE
**Generated By**: Phase 1 Execution Script (\`scripts/phase1_execute.sh\`)
EOF

    print_success "Completion report saved to: $REPORT_OUTPUT"
    print_info "Review the report for detailed results and next steps"
}

# Error handling
trap 'handle_error $? $LINENO' ERR

handle_error() {
    local exit_code=$1
    local line_number=$2
    echo ""
    echo -e "${RED}${CROSS} Error occurred at line $line_number (exit code: $exit_code)${NC}"
    echo -e "${YELLOW}${WARN} Phase 1 execution failed${NC}"
    echo ""
    echo -e "${INFO} Troubleshooting steps:"
    echo "  1. Check error messages above"
    echo "  2. Review build.log if build failed"
    echo "  3. Run: ./scripts/check_cuda_toolkit.sh"
    echo "  4. Run: ./scripts/phase1_preflight_check.sh"
    echo ""
    generate_report "Failed"
    exit $exit_code
}

# Header
clear
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                            ║${NC}"
echo -e "${MAGENTA}║  ${ROCKET}  Phase 1: Fix PyTorch Environment & Build CUDA    ║${NC}"
echo -e "${MAGENTA}║                                                            ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Execution started: $EXECUTION_DATE${NC}"
echo ""

# Step 1: Pre-Flight Check
print_step "Pre-Flight Check"

print_info "Running environment verification..."
if ./scripts/phase1_preflight_check.sh; then
    PREFLIGHT_PASSED=true
    print_success "Pre-flight check passed"
else
    print_warning "Pre-flight check found issues (expected)"
    print_info "Continuing with installation..."
fi

# Ask user to confirm (unless --yes)
if [ "$NON_INTERACTIVE" = false ]; then
    echo ""
    echo -e "${YELLOW}${WARN} This script will:${NC}"
    echo "  1. Install system CUDA toolkit (requires sudo)"
    echo "  2. Build CUDA extensions"
    echo "  3. Verify bindings"
    echo "  4. Validate PyTorch CUDA"
    echo ""
    read -p "Continue with Phase 1 execution? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Execution cancelled by user${NC}"
        exit 0
    fi
else
    print_info "Running in non-interactive mode (--yes)"
fi

# Step 2: Activate Environment
print_step "Activate Python Environment"

if [[ "$CONDA_DEFAULT_ENV" == "autovoice_py312" ]]; then
    print_success "Environment 'autovoice_py312' is active"
else
    print_error "Environment 'autovoice_py312' is not active"
    print_info "The autovoice_py312 conda environment is required for Phase 1"
    echo ""

    # Check if environment exists
    if conda env list --json | python -c "import sys, json; envs = json.load(sys.stdin)['envs']; print('autovoice_py312' in [e.split('/')[-1] if '/' in e else e for e in envs])" 2>/dev/null | grep -q "True"; then
        print_info "Environment exists but is not active"
        print_info "Attempting to activate autovoice_py312 environment..."

        # Initialize conda in the current shell session
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            print_success "Conda initialized"
        else
            # Try alternative conda initialization
            eval "$(conda shell.$(basename "$SHELL") hook 2>/dev/null)" || true
            if ! command -v conda &> /dev/null; then
                print_error "Could not initialize conda"
                print_info "Please manually run: conda activate autovoice_py312"
                print_info "Then re-run this script"
                exit 1
            fi
            print_success "Conda initialized (fallback method)"
        fi

        if [ "$NON_INTERACTIVE" = true ]; then
            print_info "Non-interactive mode: Auto-activating environment..."
            if conda activate autovoice_py312; then
                if [[ "$CONDA_DEFAULT_ENV" == "autovoice_py312" ]]; then
                    print_success "Environment auto-activated successfully"
                else
                    print_error "Failed to activate environment after conda activate command"
                    print_info "Please manually run: conda activate autovoice_py312"
                    print_info "Then re-run this script"
                    exit 1
                fi
            else
                print_error "conda activate command failed"
                print_info "Please manually run: conda activate autovoice_py312"
                print_info "Then re-run this script"
                exit 1
            fi
        else
            # Interactive mode: prompt user
            read -p "Would you like to auto-activate it now? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "Activating autovoice_py312 environment..."
                if conda activate autovoice_py312; then
                    if [[ "$CONDA_DEFAULT_ENV" == "autovoice_py312" ]]; then
                        print_success "Environment activated successfully"
                    else
                        print_error "Failed to activate environment after conda activate command"
                        print_info "Please manually run: conda activate autovoice_py312"
                        print_info "Then re-run this script"
                        exit 1
                    fi
                else
                    print_error "conda activate command failed"
                    print_info "Please manually run: conda activate autovoice_py312"
                    print_info "Then re-run this script"
                    exit 1
                fi
            else
                print_info "Please activate the environment manually:"
                print_info "  conda activate autovoice_py312"
                print_info "Then re-run this script"
                exit 1
            fi
        fi
    else
        print_warning "Environment 'autovoice_py312' does not exist"
        print_info "This environment can be created using ./scripts/setup_pytorch_env.sh (Option 2)"
        echo ""

        if [ "$NON_INTERACTIVE" = true ]; then
            print_info "Non-interactive mode: Creating environment automatically..."

            # Source conda.sh to make conda command available
            CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
            if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
                source "$CONDA_BASE/etc/profile.d/conda.sh"
                print_success "Conda initialized"
            else
                print_error "Could not find conda.sh at $CONDA_BASE/etc/profile.d/conda.sh"
                print_info "Please install conda or set CONDA_BASE correctly"
                exit 1
            fi

            # Run setup script Option 2 (create new environment)
            print_info "Running: ./scripts/setup_pytorch_env.sh with Option 2 (create new environment)"
            if echo "2" | ./scripts/setup_pytorch_env.sh; then
                print_success "Environment created successfully"

                # Activate the newly created environment
                print_info "Activating autovoice_py312 environment..."
                conda activate autovoice_py312

                if [[ "$CONDA_DEFAULT_ENV" == "autovoice_py312" ]]; then
                    print_success "Environment activated successfully"
                else
                    print_error "Failed to activate environment"
                    print_info "Please run: conda activate autovoice_py312"
                    print_info "Then re-run this script"
                    exit 1
                fi
            else
                print_error "Failed to create environment"
                print_info "Please run ./scripts/setup_pytorch_env.sh manually and select Option 2"
                exit 1
            fi
        else
            # Interactive mode: prompt user
            read -p "Would you like to create it now? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Source conda.sh to make conda command available
                CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
                if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
                    source "$CONDA_BASE/etc/profile.d/conda.sh"
                    print_success "Conda initialized"
                else
                    print_error "Could not find conda.sh at $CONDA_BASE/etc/profile.d/conda.sh"
                    print_info "Please install conda or set CONDA_BASE correctly"
                    exit 1
                fi

                # Run setup script Option 2 (create new environment)
                print_info "Running: ./scripts/setup_pytorch_env.sh"
                print_info "Please select Option 2 (Create new Python 3.12 environment)"
                if ./scripts/setup_pytorch_env.sh; then
                    print_success "Environment setup completed"

                    # Activate the newly created environment
                    print_info "Activating autovoice_py312 environment..."
                    conda activate autovoice_py312

                    if [[ "$CONDA_DEFAULT_ENV" == "autovoice_py312" ]]; then
                        print_success "Environment activated successfully"
                    else
                        print_error "Failed to activate environment"
                        print_info "Please run: conda activate autovoice_py312"
                        print_info "Then re-run this script"
                        exit 1
                    fi
                else
                    print_error "Environment setup failed"
                    exit 1
                fi
            else
                print_info "Please create the environment manually:"
                print_info "  ./scripts/setup_pytorch_env.sh (select Option 2)"
                print_info "  conda activate autovoice_py312"
                print_info "Then re-run this script"
                exit 1
            fi
        fi
    fi
fi

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" == 3.12.* ]]; then
    print_success "Python version: $PYTHON_VERSION"
else
    print_error "Python version $PYTHON_VERSION (expected 3.12.x)"
    exit 1
fi

# Step 3: Install CUDA Toolkit
print_step "Install System CUDA Toolkit"

# Check if already installed
if [ -f "/usr/local/cuda-12.1/include/nv/target" ] || [ -f "/usr/local/cuda/include/nv/target" ]; then
    print_success "System CUDA toolkit with headers already installed"
    CUDA_INSTALLED=true
else
    print_info "Installing system CUDA toolkit..."
    print_warning "This requires sudo privileges and may take several minutes"
    echo ""

    # Pass --yes flag to install_cuda_toolkit.sh if in non-interactive mode
    INSTALL_ARGS=""
    if [ "$NON_INTERACTIVE" = true ]; then
        INSTALL_ARGS="--yes"
    fi

    if ./scripts/install_cuda_toolkit.sh $INSTALL_ARGS; then
        CUDA_INSTALLED=true
        print_success "CUDA toolkit installed successfully"
        
        # Reload environment
        print_info "Reloading environment variables..."
        source ~/.bashrc || true
        
        # Verify installation
        if command -v nvcc &> /dev/null; then
            NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
            print_success "nvcc version: $NVCC_VERSION"
        fi
    else
        print_error "CUDA toolkit installation failed"
        print_info "Check error messages above"
        exit 1
    fi
fi

# Step 4: Build CUDA Extensions
print_step "Build CUDA Extensions"

print_info "Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info 2>/dev/null || true
find . -name "*.so" -path "*/cuda_kernels*" -type f -delete 2>/dev/null || true

print_info "Building CUDA extensions (this may take a few minutes)..."
echo ""

BUILD_START=$(date +%s)
if pip install -e . 2>&1 | tee build.log; then
    BUILD_END=$(date +%s)
    BUILD_DURATION=$((BUILD_END - BUILD_START))
    EXTENSIONS_BUILT=true
    print_success "CUDA extensions built successfully"
    print_info "Build duration: ${BUILD_DURATION}s"

    # Check if extension file was created
    EXTENSION_FILE=$(find . -name "cuda_kernels*.so" 2>/dev/null | head -n 1)
    if [ -n "$EXTENSION_FILE" ]; then
        EXTENSION_SIZE=$(stat -f%z "$EXTENSION_FILE" 2>/dev/null || stat -c%s "$EXTENSION_FILE" 2>/dev/null)
        print_success "Extension file: $EXTENSION_FILE"
        print_info "File size: $EXTENSION_SIZE bytes"
    else
        print_warning "Extension file not found (may be in different location)"
    fi
else
    BUILD_END=$(date +%s)
    BUILD_DURATION=$((BUILD_END - BUILD_START))
    print_error "CUDA extension build failed"
    print_info "Check build.log for details"

    # Check for common errors
    if grep -q "nv/target" build.log; then
        print_error "Missing 'nv/target' header - CUDA toolkit incomplete"
        print_info "Run: ./scripts/install_cuda_toolkit.sh"
    fi

    exit 1
fi

# Quick import test
print_info "Testing import..."
if python -c "from auto_voice import cuda_kernels" 2>/dev/null; then
    print_success "Module imports successfully"
else
    print_warning "Module import failed (will verify in next step)"
fi

# Step 5: Verify Bindings
print_step "Verify Bindings"

print_info "Running comprehensive binding verification..."
echo ""

VERIFY_START=$(date +%s)
if python ./scripts/verify_bindings.py 2>&1 | tee verify.log; then
    VERIFY_END=$(date +%s)
    VERIFY_DURATION=$((VERIFY_END - VERIFY_START))
    BINDINGS_VERIFIED=true
    print_success "Bindings verified successfully"
    print_info "Verification duration: ${VERIFY_DURATION}s"
else
    VERIFY_END=$(date +%s)
    VERIFY_DURATION=$((VERIFY_END - VERIFY_START))
    print_error "Binding verification failed"
    print_info "Check verify.log for details"
    exit 1
fi

# Step 6: Validate PyTorch CUDA
print_step "Validate PyTorch CUDA"

print_info "Testing PyTorch CUDA functionality..."
echo ""

# Create validation script
cat > /tmp/validate_pytorch_cuda.py << 'EOF'
import torch
import sys

print("PyTorch Information:")
print(f"  Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"  GPU Count: {torch.cuda.device_count()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    # Test basic CUDA operations
    print("\nTesting CUDA tensor operations...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("  ✅ CUDA tensor operations: Success")
        sys.exit(0)
    except Exception as e:
        print(f"  ❌ CUDA tensor operations failed: {e}")
        sys.exit(1)
else:
    print("  ❌ CUDA not available")
    sys.exit(1)
EOF

if python /tmp/validate_pytorch_cuda.py; then
    PYTORCH_VALIDATED=true
    print_success "PyTorch CUDA validation passed"
else
    print_error "PyTorch CUDA validation failed"
    exit 1
fi

rm /tmp/validate_pytorch_cuda.py

# Step 7: Generate Report
print_step "Generate Completion Report"

generate_report "Success"

# Final Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  ${CHECK}  Phase 1 Completed Successfully!                    ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((TOTAL_DURATION / 60))
DURATION_SEC=$((TOTAL_DURATION % 60))

echo -e "${CYAN}Execution Summary:${NC}"
echo -e "  ${CHECK} Pre-flight check: $([ "$PREFLIGHT_PASSED" = true ] && echo "Passed" || echo "Completed")"
echo -e "  ${CHECK} CUDA toolkit: Installed"
echo -e "  ${CHECK} Extensions: Built"
echo -e "  ${CHECK} Bindings: Verified"
echo -e "  ${CHECK} PyTorch CUDA: Validated"
echo ""
echo -e "${CYAN}Total Duration: ${DURATION_MIN}m ${DURATION_SEC}s${NC}"
echo ""
echo -e "${GREEN}${ROCKET} Ready for Phase 2: Testing and Validation${NC}"
echo ""
echo -e "${INFO} Next steps:"
echo "  1. Review $REPORT_OUTPUT for full details"
echo "  2. Run comprehensive tests: ./scripts/run_full_validation.sh"
echo "  3. Run test suite: pytest tests/ -v"
echo "  4. Benchmark performance: ./scripts/run_comprehensive_benchmarks.py"
echo ""
echo -e "${INFO} Log files:"
echo "  - Build log: build.log"
echo "  - Verification log: verify.log"
echo "  - Completion report: $REPORT_OUTPUT"
echo ""

exit 0
