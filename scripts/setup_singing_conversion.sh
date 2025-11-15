#!/bin/bash
# AutoVoice Singing Voice Conversion - Complete Setup Script

set -e  # Exit on error

echo "=========================================="
echo "AutoVoice Singing Voice Conversion Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${RED}Error: No conda environment activated${NC}"
    echo "Please activate your conda environment first:"
    echo "  conda activate autovoice"
    exit 1
fi

echo -e "${GREEN}✓${NC} Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 1: Install Python dependencies
echo "=========================================="
echo "Step 1: Installing Python Dependencies"
echo "=========================================="
echo ""

echo "Installing torchcrepe for CREPE pitch extraction..."
pip install torchcrepe>=0.0.23

echo "Installing transformers for HuBERT models..."
pip install transformers>=4.30.0

echo "Installing fairseq for Facebook AI models..."
pip install fairseq>=0.12.0

echo "Installing faiss for similarity search..."
pip install faiss-cpu>=1.7.4

echo -e "${GREEN}✓${NC} Python dependencies installed"
echo ""

# Step 2: Download pre-trained models
echo "=========================================="
echo "Step 2: Downloading Pre-trained Models"
echo "=========================================="
echo ""

python scripts/download_singing_models.py

echo -e "${GREEN}✓${NC} Models downloaded"
echo ""

# Step 3: Setup frontend
echo "=========================================="
echo "Step 3: Setting Up Frontend"
echo "=========================================="
echo ""

if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Warning: Node.js not found${NC}"
    echo "Please install Node.js >= 18.0.0 from https://nodejs.org/"
    echo "Skipping frontend setup..."
else
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        echo -e "${YELLOW}Warning: Node.js version $NODE_VERSION is too old${NC}"
        echo "Please upgrade to Node.js >= 18.0.0"
        echo "Skipping frontend setup..."
    else
        echo "Node.js version: $(node -v)"
        echo "npm version: $(npm -v)"
        echo ""
        
        cd frontend
        
        echo "Installing frontend dependencies..."
        npm install
        
        echo "Creating .env file..."
        if [ ! -f .env ]; then
            cp .env.example .env
            echo -e "${GREEN}✓${NC} Created .env file"
        else
            echo -e "${YELLOW}!${NC} .env file already exists"
        fi
        
        cd ..
        
        echo -e "${GREEN}✓${NC} Frontend setup complete"
    fi
fi
echo ""

# Step 4: Verify installation
echo "=========================================="
echo "Step 4: Verifying Installation"
echo "=========================================="
echo ""

echo "Testing Python imports..."
python -c "
import torch
import torchcrepe
import transformers
import fairseq
import faiss
print('✓ All Python packages imported successfully')
"

echo ""
echo "Testing AutoVoice imports..."
python -c "
from auto_voice.audio.pitch_extractor import SingingPitchExtractor
from auto_voice.models.singing_voice_converter import SingingVoiceConverter
from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
print('✓ AutoVoice modules imported successfully')
"

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the backend server:"
echo "   python -m auto_voice.web.app"
echo ""
echo "2. In a new terminal, start the frontend dev server:"
echo "   cd frontend && npm run dev"
echo ""
echo "3. Open your browser to:"
echo "   http://localhost:3000"
echo ""
echo "4. For production build:"
echo "   cd frontend && npm run build"
echo "   # Then serve from Flask backend"
echo ""
echo "Documentation:"
echo "  - Implementation Status: docs/IMPLEMENTATION_STATUS.md"
echo "  - Research: docs/SINGING_VOICE_CONVERSION_RESEARCH.md"
echo "  - Frontend README: frontend/README.md"
echo ""
echo "Happy converting! 🎵"

