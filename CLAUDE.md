# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoVoice is a GPU-accelerated singing voice conversion and TTS system built with PyTorch, CUDA kernels, and Flask. It converts songs to a target voice while preserving pitch and timing, using the So-VITS-SVC architecture.

## Build & Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n autovoice python=3.12 -y && conda activate autovoice

# Install PyTorch with CUDA (REQUIRED FIRST)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
pip install -e .
```

### Running the Application
```bash
# Start server (Flask + SocketIO)
python main.py --host 0.0.0.0 --port 5000

# With Docker
docker-compose up
```

### Testing
```bash
# Run complete test suite
./run_tests.sh all

# Quick smoke tests (< 30s)
./run_tests.sh smoke

# Fast tests (excludes slow)
./run_tests.sh fast

# With coverage report
./run_tests.sh coverage

# Run specific test file
pytest tests/test_voice_conversion.py -v

# Run by marker
pytest tests/ -m cuda -v
pytest tests/ -m "not slow" -v
```

### Code Quality
```bash
black src/ tests/
isort src/ tests/
mypy src/auto_voice
```

### Key Scripts
- `scripts/setup_pytorch_env.sh` - Automated PyTorch/CUDA setup
- `scripts/build_and_test.sh` - Build and verify
- `scripts/verify_bindings.py` - Verify CUDA extension build
- `scripts/download_pretrained_models.py` - Download model weights

## Architecture

### Source Structure (`src/`)
```
auto_voice/
  inference/           # Voice conversion pipeline (So-VITS-SVC)
    singing_conversion_pipeline.py  # Main conversion entry point
    voice_cloner.py                 # Speaker encoder/profile creation
    realtime_voice_conversion_pipeline.py
  models/              # Neural network architectures
    encoder.py         # Content/pitch encoders
    vocoder.py         # HiFiGAN vocoder
  audio/               # Audio processing utilities
  evaluation/          # Quality metrics (pitch RMSE, speaker similarity)
  web/                 # Flask API + SocketIO handlers
    app.py             # create_app() factory
    api.py             # REST endpoints
  training/            # Training pipeline
  gpu/                 # GPU memory management
  monitoring/          # Prometheus metrics
cuda_kernels/          # Custom CUDA kernels (.cu files)
```

### Key Classes
- `SingingConversionPipeline` - Main voice conversion class (`inference/singing_conversion_pipeline.py`)
- `VoiceCloner` - Creates voice profiles from audio samples (`inference/voice_cloner.py`)
- `create_app()` - Flask app factory (`web/app.py`)

### Frontend (`frontend/`)
React + TypeScript + Vite + Tailwind CSS
```bash
cd frontend && npm install && npm run dev
```

### API Endpoints
- `GET /health` - Health check with component status
- `POST /api/v1/voice/clone` - Create voice profile
- `POST /api/v1/convert/song` - Convert song to target voice
- `GET /api/v1/convert/status/{id}` - Check conversion status
- WebSocket events for real-time progress

## Test Markers
Tests use pytest markers defined in `pytest.ini`:
- `smoke` - Fast validation
- `cuda` - Requires GPU
- `slow` / `very_slow` - Long-running tests
- `integration` - Component interaction tests
- `performance` - Benchmarks
- `tensorrt` - TensorRT-specific tests

## Configuration
- `config/gpu_config.yaml` - GPU and server settings
- `config/logging_config.yaml` - Logging configuration
- Environment variables: `LOG_LEVEL`, `LOG_FORMAT`, `CUDA_VISIBLE_DEVICES`

## Pre-trained Models
Located in `models/pretrained/`:
- `sovits5.0_main_1500.pth` - Main So-VITS model
- `hifigan_ljspeech.ckpt` - HiFiGAN vocoder
- `hubert-soft-0d54a1f4.pt` - HuBERT feature extractor

## File Organization Rules
- Source code: `/src`
- Tests: `/tests`
- Documentation: `/docs`
- Scripts: `/scripts`
- Configuration: `/config`
- Examples: `/examples`
