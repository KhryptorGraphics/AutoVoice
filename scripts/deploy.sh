#!/bin/bash
set -e

echo "AutoVoice Deployment Script"
echo "==========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
DEPLOYMENT_TARGET=${1:-local}
SKIP_TESTS=${2:-false}
VERBOSE=${3:-false}

print_status "Starting deployment for target: $DEPLOYMENT_TARGET"

# Create dist directory if it doesn't exist
mkdir -p dist

# Step 1: Clean previous builds
print_status "Cleaning previous builds..."
if [ -d "build" ]; then
    rm -rf build/
    print_status "Removed build/ directory"
fi

# Enable nullglob to handle glob expansion properly
shopt -s nullglob
EGG_INFO_DIRS=(*.egg-info/)
if [ ${#EGG_INFO_DIRS[@]} -gt 0 ]; then
    rm -rf "${EGG_INFO_DIRS[@]}"
    print_status "Removed egg-info directories"
fi
shopt -u nullglob

# Clean old wheels using find for proper glob handling
if find dist -name 'auto_voice-*.whl' -type f 2>/dev/null | grep -q .; then
    find dist -name 'auto_voice-*.whl' -type f -delete
    print_status "Removed old wheel files"
fi

# Step 2: Build the project
print_status "Building CUDA extensions..."
if ! ./scripts/build.sh; then
    print_error "Build failed! Cannot proceed with deployment."
    exit 1
fi
print_success "Build completed successfully"

# Step 3: Run tests (unless skipped)
if [ "$SKIP_TESTS" != "true" ]; then
    print_status "Running test suite..."
    # Skip rebuild in tests since we just built successfully
    export SKIP_BUILD=true
    if ! ./scripts/test.sh; then
        print_error "Tests failed! Cannot proceed with deployment."
        print_error "Use './scripts/deploy.sh $DEPLOYMENT_TARGET true' to skip tests"
        exit 1
    fi
    print_success "All tests passed"
else
    print_warning "Skipping tests as requested"
fi

# Step 4: Gather environment information
print_status "Gathering environment information..."
ENV_INFO_FILE="dist/environment_info.txt"
cat > "$ENV_INFO_FILE" << EOF
AutoVoice Build Environment Information
======================================
Build Date: $(date)
Build Host: $(hostname)
Build User: $(whoami)
Build Directory: $(pwd)

System Information:
------------------
OS: $(uname -a)
Python Version: $(python --version)
PyTorch Version: $(python -c "import torch; print(torch.__version__)")

CUDA Information:
----------------
CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())")
EOF

if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Compute Capability: {props.major}.{props.minor}')
        print(f'  Total Memory: {props.total_memory / 1024**3:.1f} GB')
" >> "$ENV_INFO_FILE"
fi

cat >> "$ENV_INFO_FILE" << EOF

Package Information:
-------------------
EOF

pip list | grep -E "(torch|numpy|cuda)" >> "$ENV_INFO_FILE" 2>/dev/null || echo "No relevant packages found" >> "$ENV_INFO_FILE"

print_success "Environment information saved to $ENV_INFO_FILE"

# Step 5: Create wheel package
print_status "Creating wheel package..."
if ! python setup.py bdist_wheel; then
    print_error "Wheel creation failed!"
    exit 1
fi

# Find the created wheel file
WHEEL_FILE=$(find dist/ -name "auto_voice-*.whl" -type f | head -1)
if [ -z "$WHEEL_FILE" ]; then
    print_error "No wheel file found in dist/ directory"
    exit 1
fi

print_success "Wheel created: $WHEEL_FILE"

# Step 6: Validate the wheel package
print_status "Validating wheel package..."
if python -m zipfile -l "$WHEEL_FILE" > /dev/null 2>&1; then
    print_success "Wheel file is valid"
else
    print_error "Wheel file validation failed"
    exit 1
fi

# Check wheel contents
WHEEL_CONTENTS=$(python -m zipfile -l "$WHEEL_FILE" | grep -E "\.(so|pyd)$" | wc -l)
if [ "$WHEEL_CONTENTS" -gt 0 ]; then
    print_success "Wheel contains $WHEEL_CONTENTS compiled extension(s)"
else
    print_warning "Wheel does not contain compiled extensions"
fi

# Step 7: Deployment-specific actions
case $DEPLOYMENT_TARGET in
    "local")
        print_status "Local deployment - wheel ready for installation"
        print_status "Install with: pip install $WHEEL_FILE"
        ;;
    "staging")
        print_status "Staging deployment preparation"
        # Copy wheel to staging location (customize as needed)
        STAGING_DIR="dist/staging"
        mkdir -p "$STAGING_DIR"
        cp "$WHEEL_FILE" "$STAGING_DIR/"
        cp "$ENV_INFO_FILE" "$STAGING_DIR/"
        print_success "Files copied to staging directory: $STAGING_DIR"
        ;;
    "production")
        print_status "Production deployment preparation"
        # Additional production checks
        if [ "$SKIP_TESTS" = "true" ]; then
            print_error "Production deployment requires tests to pass!"
            exit 1
        fi

        # Copy wheel to production location (customize as needed)
        PRODUCTION_DIR="dist/production"
        mkdir -p "$PRODUCTION_DIR"
        cp "$WHEEL_FILE" "$PRODUCTION_DIR/"
        cp "$ENV_INFO_FILE" "$PRODUCTION_DIR/"

        # Create deployment manifest
        MANIFEST_FILE="$PRODUCTION_DIR/deployment_manifest.json"
        cat > "$MANIFEST_FILE" << EOF
{
  "package": "auto_voice",
  "wheel_file": "$(basename $WHEEL_FILE)",
  "build_date": "$(date -Iseconds)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "build_host": "$(hostname)",
  "cuda_available": $(python -c "import torch; print(str(torch.cuda.is_available()).lower())"),
  "python_version": "$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")",
  "pytorch_version": "$(python -c "import torch; print(torch.__version__)")"
}
EOF
        print_success "Production files ready in: $PRODUCTION_DIR"
        print_success "Deployment manifest created: $MANIFEST_FILE"
        ;;
    *)
        print_error "Unknown deployment target: $DEPLOYMENT_TARGET"
        print_error "Valid targets: local, staging, production"
        exit 1
        ;;
esac

# Step 8: Cleanup temporary build artifacts
print_status "Cleaning up temporary build artifacts..."
if [ -d "build" ]; then
    rm -rf build/
fi

# Keep egg-info for development but remove for clean packaging
if [ "$DEPLOYMENT_TARGET" != "local" ]; then
    rm -rf src/auto_voice.egg-info/ 2>/dev/null || true
fi

# Step 9: Final summary
print_success "Deployment completed successfully!"
print_status "Summary:"
print_status "- Target: $DEPLOYMENT_TARGET"
print_status "- Wheel: $(basename $WHEEL_FILE)"
print_status "- Size: $(du -h $WHEEL_FILE | cut -f1)"
print_status "- Tests: $([ "$SKIP_TESTS" = "true" ] && echo "SKIPPED" || echo "PASSED")"

# Additional information for verbose mode
if [ "$VERBOSE" = "true" ]; then
    print_status ""
    print_status "Detailed wheel contents:"
    python -m zipfile -l "$WHEEL_FILE" | head -20
    if [ "$(python -m zipfile -l "$WHEEL_FILE" | wc -l)" -gt 20 ]; then
        print_status "... (truncated, full list available with: python -m zipfile -l $WHEEL_FILE)"
    fi
fi

echo "============================================"
echo "AutoVoice deployment completed successfully!"
echo "============================================"

# Return the wheel file path for scripting
echo "$WHEEL_FILE"