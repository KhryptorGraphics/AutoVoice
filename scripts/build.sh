#!/bin/bash
set -e

echo "Building AutoVoice CUDA extensions..."

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Make sure CUDA toolkit is installed and in PATH"
    exit 1
fi

# Build CUDA extensions
python setup.py build_ext --inplace

echo "Build completed successfully!"