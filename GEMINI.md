# AutoVoice Gemini Agent Context

This document provides context for the Gemini agent to understand and interact with the AutoVoice project.

## Project Overview

AutoVoice is a high-performance, GPU-accelerated voice synthesis and singing voice conversion system. It is designed for real-time processing and is optimized for production environments.

**Core Technologies:**

*   **Programming Language:** Python
*   **Deep Learning Framework:** PyTorch
*   **GPU Acceleration:** CUDA, TensorRT
*   **Web Framework:** Flask, Socket.IO
*   **Containerization:** Docker

**Key Features:**

*   Real-time voice synthesis (TTS) and singing voice conversion.
*   Voice cloning from short audio samples.
*   Multi-speaker and multi-language support.
*   Production-grade monitoring with Prometheus and Grafana.
*   Scalable architecture with REST and WebSocket APIs.

## Building and Running

### 1. Environment Setup

The project requires Python 3.8+ and a specific version of PyTorch installed with CUDA support.

**IMPORTANT:** `torch`, `torchaudio`, and `torchvision` must be installed *before* other dependencies.

**Example for CUDA 12.1:**

```bash
# Create a conda environment
conda create -n autovoice python=3.12 -y
conda activate autovoice

# Install PyTorch with CUDA 12.1 support
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Dependencies

Once PyTorch is installed, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### 3. Build CUDA Extensions

The project includes custom CUDA kernels that need to be built:

```bash
python setup.py build_ext --inplace
```

### 4. Running the Application

**From Source:**

```bash
python main.py
```

The application will be available at `http://localhost:5000`.

**With Docker:**

```bash
docker-compose up
```

## Development Conventions

### Testing

Run the full test suite with:

```bash
./scripts/test.sh
```

Run tests with coverage:

```bash
pytest --cov=src/auto_voice --cov-report=html
```

### Code Style

The project uses `black` for code formatting and `isort` for import sorting.

**Format code:**

```bash
black src/ tests/
isort src/ tests/
```

**Lint code:**

```bash
flake8 src/ tests/
```

### Type Checking

The project uses `mypy` for static type checking:

```bash
mypy src/auto_voice
```
