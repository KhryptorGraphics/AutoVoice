# 🚀 AutoVoice Deployment Setup Guide

## Quick Start (5 minutes)

### 1. **Environment Setup**
```bash
# Activate conda environment (create if needed)
conda create -n autovoice python=3.12 -y
conda activate autovoice

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### 2. **Backend Setup**
```bash
# Install the package in development mode
pip install -e .

# Download models (if not already present)
python scripts/download_singing_models.py

# Verify installation
python -c "from src.auto_voice.models import SingingVoiceConverter; print('✅ Backend ready!')"
```

### 3. **Frontend Setup**
```bash
cd frontend

# Install dependencies
npm install

# Build for production
npm run build

# Or run development server
npm run dev
```

### 4. **Start the Application**

**Option A: Development Mode**
```bash
# Terminal 1: Backend API
python main.py

# Terminal 2: Frontend (if using dev server)
cd frontend && npm run dev
```

**Option B: Production Mode (Docker)**
```bash
docker-compose up -d
```

## Deployment Checklist

- [ ] Python 3.12 environment created
- [ ] PyTorch 2.5.1 with CUDA 12.1 installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Models downloaded (590 MB total)
- [ ] Backend verified (`python main.py` starts without errors)
- [ ] Frontend built (`npm run build` completes successfully)
- [ ] API accessible at `http://localhost:5000`
- [ ] WebSocket connection working
- [ ] Frontend accessible at `http://localhost:3000` (dev) or served by backend

## Verification Commands

```bash
# Check Python version
python --version  # Should be 3.12.x

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check models
ls -lh models/pretrained/

# Test backend
curl http://localhost:5000/api/v1/health

# Test frontend build
npm run build --prefix frontend
```

## Troubleshooting

**PyTorch CUDA Issues:**
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

**Model Download Fails:**
```bash
python scripts/download_singing_models.py --retry 5
```

**Port Already in Use:**
```bash
# Change port in main.py or use environment variable
export FLASK_PORT=5001
python main.py
```

## Next Steps

1. Access frontend at `http://localhost:3000` (dev) or `http://localhost:5000` (production)
2. Create voice profiles
3. Upload reference audio
4. Convert singing voice
5. Download results

See `FRONTEND_USER_GUIDE.md` for detailed usage instructions.

