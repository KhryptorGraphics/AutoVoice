#!/bin/bash
# Complete AutoVoice Environment Setup Script
# This script sets up Python 3.12 environment, installs dependencies, and validates installation

set -e  # Exit on error

echo "======================================================================"
echo "🎤 AutoVoice Complete Environment Setup"
echo "======================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo ""
    echo "❌ Conda not found. Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi

echo ""
echo "Step 1: Create Python 3.12 Environment"
echo "======================================================================"

# Check if environment already exists
if conda env list | grep -q "^autovoice "; then
    echo "⚠️  Environment 'autovoice' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n autovoice -y
        echo "  ✓ Removed existing environment"
    else
        echo "  Using existing environment"
    fi
fi

if ! conda env list | grep -q "^autovoice "; then
    echo "Creating new environment with Python 3.12..."
    conda create -n autovoice python=3.12 -y
    echo "  ✓ Environment created"
fi

echo ""
echo "Step 2: Install Dependencies"
echo "======================================================================"

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autovoice

echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install -r requirements.txt

echo "  ✓ All dependencies installed"

echo ""
echo "Step 3: Download Pre-trained Models"
echo "======================================================================"

# Check if models already exist
if [ -d "models/pretrained" ] && [ "$(ls -A models/pretrained)" ]; then
    echo "✓ Pre-trained models already downloaded in models/pretrained/"
    ls -lh models/pretrained/
else
    echo "Downloading pre-trained models (~590 MB)..."
    python scripts/download_pretrained_models.py --required-only
fi

echo ""
echo "Step 4: Validate Installation"
echo "======================================================================"

# Run validation script
echo "Running installation validation..."
python scripts/validate_installation.py

VALIDATION_EXIT=$?

if [ $VALIDATION_EXIT -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "🎉 Setup Complete! AutoVoice is ready to use."
    echo "======================================================================"
    echo ""
    echo "To use AutoVoice:"
    echo ""
    echo "  1. Activate environment:"
    echo "     conda activate autovoice"
    echo ""
    echo "  2. Run demo:"
    echo "     python examples/demo_voice_conversion.py \\"
    echo "       --song data/test_song.mp3 \\"
    echo "       --reference data/my_voice.wav"
    echo ""
    echo "  3. Or start web interface:"
    echo "     python main.py"
    echo "     # Open http://localhost:5000"
    echo ""
    echo "📚 Documentation: docs/QUICK_START_GUIDE.md"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "⚠️  Setup completed with warnings"
    echo "======================================================================"
    echo ""
    echo "Please check the validation output above and fix any issues."
    echo ""
fi
