#!/bin/bash
# Helper script for Python 3.12 environment setup

set -e

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda is not available in PATH, attempting to initialize..."

    # Try to initialize conda using shell hook
    if eval "$(conda shell.bash hook)" 2>/dev/null && command -v conda &> /dev/null; then
        echo "Conda initialized successfully"
    else
        # Optional: Try to source common profile files
        if [ -f ~/.bashrc ]; then
            source ~/.bashrc
        fi
        if [ -f ~/.zshrc ]; then
            source ~/.zshrc
        fi
        # Try again after sourcing profiles
        if eval "$(conda shell.bash hook)" 2>/dev/null && command -v conda &> /dev/null; then
            echo "Conda initialized successfully from profile"
        else
            echo "Error: conda still not available after initialization attempt"
            echo ""
            echo "Please install Miniconda/Anaconda or manually initialize conda:"
            echo "  eval \"\$(conda shell.bash hook)\""
            echo "  OR source ~/miniconda3/etc/profile.d/conda.sh"
            exit 1
        fi
    fi
fi

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
