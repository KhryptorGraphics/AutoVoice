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
