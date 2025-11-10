# AutoVoice

[![CI](https://github.com/autovoice/autovoice/workflows/CI/badge.svg)](https://github.com/autovoice/autovoice/actions)
[![Docker Build](https://github.com/autovoice/autovoice/workflows/Docker%20Build/badge.svg)](https://github.com/autovoice/autovoice/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-green)](https://developer.nvidia.com/cuda-toolkit)

**GPU-accelerated voice synthesis and singing voice conversion system with real-time processing and TensorRT optimization**

## 🔧 Compatibility Matrix

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **NVIDIA Driver** | 525+ | 535+ | Required for CUDA 12.1 support |
| **CUDA Toolkit** | 11.8 | 12.1 | Must match PyTorch CUDA version |
| **cuDNN** | 8.6.0 | 8.9.0+ | Bundled with PyTorch |
| **GPU Compute Capability** | 7.0 (Volta) | 8.0+ (Ampere) | See [NVIDIA docs](https://developer.nvidia.com/cuda-gpus) |
| **Python** | 3.8 | 3.10, 3.11, 3.12 | Supports 3.8, 3.9, 3.10, 3.11, 3.12 |
| **PyTorch** | 2.0.0 | Supported 2.0–2.2; Recommended 2.2.x | Install with `--index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1 or `cu118` for CUDA 11.8 |

> **Important**: Python 3.12 requires PyTorch 2.1+ (PyTorch 2.0 does not publish Python 3.12 wheels). For Python 3.12, use:
> ```bash
> pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
> ```

### Supported GPU Architectures
- ✅ **Volta** (V100): Compute capability 7.0
- ✅ **Turing** (RTX 20xx, T4): Compute capability 7.5
- ✅ **Ampere** (A100, RTX 30xx): Compute capability 8.0/8.6
- ✅ **Ada Lovelace** (RTX 40xx): Compute capability 8.9

**Check your GPU:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

AutoVoice is a high-performance voice synthesis and singing voice conversion platform leveraging CUDA acceleration, WebSocket streaming, and production-grade monitoring for real-time audio generation and voice cloning.

## ✨ Features

### Voice Synthesis (TTS)
- 🚀 **CUDA Acceleration**: Optimized GPU kernels for 10-50x faster processing
- ⚡ **TensorRT Support**: Inference optimization with INT8/FP16 quantization
- 🎙️ **Real-time Processing**: WebSocket streaming for low-latency synthesis
- 🔊 **Multi-Speaker**: Support for multiple voice models and speakers

### Singing Voice Conversion
- 🎤 **Voice Cloning**: Create voice profiles from 30-60 second audio samples
- 🎵 **Song Conversion**: Convert any song to your voice while preserving pitch and timing
- 🎸 **Pitch Control**: Adjust song key with ±12 semitone pitch shifting
- 🎼 **Quality Metrics**: Comprehensive quality evaluation (pitch accuracy, speaker similarity, naturalness)
- 🎹 **Batch Processing**: Convert multiple songs efficiently

### Production Features
- 📊 **Production Monitoring**: Prometheus metrics, Grafana dashboards, structured logging
- 🐳 **Docker Ready**: Multi-stage builds with GPU support
- 🔒 **Secure**: Non-root containers, secrets management, input validation
- 📈 **Scalable**: Horizontal scaling with load balancing support

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada Lovelace)
- **NVIDIA Driver**: 535+ recommended (525+ minimum for CUDA 11.8)
- **CUDA Toolkit**: 11.8+ or 12.1 recommended
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **Docker** (optional): Docker 20.10+ with [NVIDIA Container Toolkit](docs/deployment-guide.md#step-1-install-docker-and-nvidia-container-toolkit)

**Verify your system:**
```bash
# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv

# Check CUDA version
nvcc --version

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

### 🎉 STATUS: FULLY FUNCTIONAL WITH PRE-TRAINED MODELS

**All model weights downloaded (590 MB) ✅**  
**Ready for immediate use after environment setup!**

---

## 🚀 Quick Start (5 Minutes)

### 1. Fix Python Environment

**Python 3.8–3.12 Supported** (3.10, 3.11, or 3.12 recommended):

```bash
# Create conda environment with Python 3.12
conda create -n autovoice python=3.12 -y
conda activate autovoice
```

### 2. Install PyTorch with CUDA

**IMPORTANT**: PyTorch must be installed first with CUDA support before other dependencies.

```bash
# For CUDA 12.1 (recommended)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Note**: `requirements.txt` intentionally excludes PyTorch to avoid conflicts. You must install PyTorch first with the appropriate CUDA version for your system.

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for other CUDA versions or CPU-only installation.

**Automated alternative**: Use `./scripts/setup_pytorch_env.sh` for guided installation.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pre-trained Models (Already Downloaded! ✅)

**Models are already in the project:**
```
models/pretrained/
├── sovits5.0_main_1500.pth      176 MB  ✅
├── hifigan_ljspeech.ckpt         54 MB  ✅
└── hubert-soft-0d54a1f4.pt      361 MB  ✅
```

**For deployment to other machines:**
```bash
# Models will be downloaded automatically
python scripts/download_pretrained_models.py
```

### 5. Run Demo

```bash
python examples/demo_voice_conversion.py \
  --song data/test_song.mp3 \
  --reference data/my_voice.wav
```

**Or start web interface:**

```bash
python main.py
# Open http://localhost:5000
```

---

## Installation

#### Option 1: Docker (Recommended)

```bash
# Pull the latest image
docker pull autovoice/autovoice:latest

# Run with GPU support
docker run --gpus all -p 5000:5000 autovoice/autovoice:latest

# Or use docker-compose
docker-compose up
```

**GPU Configuration Note:**
Ensure GPUs are enabled in Docker Compose. Our [docker-compose.yml](docker-compose.yml) includes proper GPU configuration for Docker Compose v2:

```yaml
services:
  auto-voice-app:
    gpus: all  # Docker Compose v2 GPU syntax
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

For Docker Swarm mode, GPU reservation is configured separately in the `deploy.resources.reservations.devices` block (see lines 39-49 in docker-compose.yml).

#### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/autovoice/autovoice.git
cd autovoice

# Install dependencies
pip install -r requirements.txt

# Setup environment and build (automated)
./scripts/setup_pytorch_env.sh  # Fix PyTorch if needed
./scripts/build_and_test.sh     # Build and verify

# Or build manually
python setup.py build_ext --inplace

# Run the application
python main.py
```

### Basic Usage

```python
from auto_voice import AutoVoice, AudioProcessor

# Initialize the voice synthesis system
voice = AutoVoice(model_path="models/voice_model.pt")

# Synthesize speech (speaker_id must be a non-negative integer)
audio = voice.synthesize("Hello, world!", speaker_id=0)

# Save to file
audio.save("output.wav")
```

### REST API Example

```bash
# Health check
curl http://localhost:5000/health

# Synthesize voice (speaker_id must be a non-negative integer, not a string)
curl -X POST http://localhost:5000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "speaker_id": 0}'
```

### WebSocket Example

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  socket.emit('synthesize_stream', {
    text: 'Hello, world!',
    speaker_id: 0  // Must be a non-negative integer
  });
});

socket.on('audio_chunk', (data) => {
  // Process audio chunk
  console.log('Received audio chunk:', data);
});
```

### Voice Conversion Quick Start

#### 1. Create Voice Profile

```python
from auto_voice.inference import VoiceCloner

# Initialize voice cloner
cloner = VoiceCloner(device='cuda')

# Create profile from your voice sample (30-60s recommended)
profile = cloner.create_voice_profile(
    audio='my_voice.wav',
    user_id='user123',
    profile_name='My Singing Voice'
)

print(f"Profile ID: {profile['profile_id']}")
print(f"Vocal Range: {profile['vocal_range']['min_note']} - {profile['vocal_range']['max_note']}")
```

#### 2. Convert Song

```python
from auto_voice.inference import SingingConversionPipeline

# Initialize pipeline
pipeline = SingingConversionPipeline(device='cuda', quality_preset='balanced')

# Convert song to your voice
result = pipeline.convert_song(
    song_path='song.mp3',
    target_profile_id=profile['profile_id'],  # Note: API uses 'profile_id' in request
    vocal_volume=1.0,
    instrumental_volume=0.9,
    pitch_shift=0,  # ±12 semitones
    temperature=1.0,  # Expressiveness control
    return_stems=True
)

print(f"Converted: {result['output_path']}")
print(f"Quality: Pitch RMSE = {result['quality_metrics']['pitch_accuracy']['rmse_hz']:.2f} Hz")
print(f"Similarity: {result['quality_metrics']['speaker_similarity']['cosine_similarity']:.2f}")
```

#### 3. Voice Conversion API

```bash
# Create voice profile
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@my_voice.wav" \
  -F "user_id=user123"

# Convert song (use 'profile_id' in the request, not 'target_profile_id')
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "profile_id=550e8400-e29b-41d4-a716-446655440000"

# Check conversion status
curl http://localhost:5000/api/v1/convert/status/{conversion_id}

# Download result
curl http://localhost:5000/api/v1/convert/download/{conversion_id}/converted.wav -o converted.wav
```

#### 4. Batch Conversion

```bash
# Convert multiple songs
python examples/demo_batch_conversion.py \
  --profile-id 550e8400-e29b-41d4-a716-446655440000 \
  --songs-dir data/songs \
  --output-dir converted/ \
  --quality balanced
```

### Quality Evaluation

AutoVoice includes comprehensive quality evaluation tools for assessing voice conversion performance. Two evaluation modes are supported: directory-based and metadata-driven evaluation.

#### Directory-Based Evaluation

```bash
# Evaluate voice conversions from directories
python examples/evaluate_voice_conversion.py \
  --source-dir /path/to/source/audio \
  --target-dir /path/to/converted/audio \
  --output-dir ./evaluation_results \
  --formats markdown json html
```

#### Metadata-Driven Evaluation

Use metadata-driven evaluation for automated conversion and evaluation through the pipeline:

```bash
# Create test metadata JSON
cat > test_metadata.json << 'EOF'
{
  "test_cases": [
    {
      "id": "test_case_1",
      "source_audio": "data/test/sample1.wav",
      "target_profile_id": "pop_star_001",
      "conversion_params": {
        "formant_shift": 1.1,
        "pitch_range": [80, 600]
      },
      "reference_audio": "data/ground_truth/sample1.wav"
    },
    {
      "id": "test_case_2",
      "source_audio": "data/test/sample2.wav",
      "target_profile_id": "rock_singer_002",
      "reference_audio": "data/ground_truth/sample2.wav"
    }
  ]
}
EOF

# Run metadata-driven evaluation
python examples/evaluate_voice_conversion.py \
  --test-metadata test_metadata.json \
  --output-dir ./evaluation_results \
  --validate-targets
```

#### Quality Validation

Enable automatic quality validation against targets:

```bash
# Validate against quality targets
python examples/evaluate_voice_conversion.py \
  --test-metadata test_metadata.json \
  --validate-targets \
  --min-pitch-correlation 0.8 \
  --max-pitch-rmse-hz 10.0 \
  --min-speaker-similarity 0.75
```

Quality metrics include:
- **Pitch Accuracy**: RMSE (Hz), correlation coefficient
- **Speaker Similarity**: Cosine similarity, embedding distance
- **Naturalness**: Spectral distortion, MOS estimation
- **Intelligibility**: STOI, ESTOI, PESQ scores

## 📖 Documentation

### Voice Conversion
- [Voice Conversion Guide](docs/voice_conversion_guide.md) - Complete user guide for voice cloning and song conversion
- [Voice Conversion API](docs/api_voice_conversion.md) - API reference for voice conversion endpoints
- [Model Architecture](docs/model_architecture.md) - Technical deep dive into So-VITS-SVC architecture
- [Quality Evaluation Guide](docs/quality_evaluation_guide.md) - Quality metrics and evaluation

### TTS & General
- [Deployment Guide](docs/deployment-guide.md) - Production deployment instructions
- [API Documentation](docs/api-documentation.md) - Complete API reference
- [Monitoring Guide](docs/monitoring-guide.md) - Observability and monitoring setup
- [Runbook](docs/runbook.md) - Operational procedures and troubleshooting

### Examples
- [Voice Cloning Demo](examples/voice_cloning_demo.ipynb) - Interactive Jupyter notebook for voice cloning
- [Song Conversion Demo](examples/song_conversion_demo.ipynb) - Interactive Jupyter notebook for song conversion
- [Demo Scripts](examples/) - Python demo scripts for voice conversion and batch processing

## 🔧 Configuration

### Environment Variables

```bash
# Logging
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                   # json, text
LOG_DIR=logs                      # Log directory path

# GPU Configuration
CUDA_VISIBLE_DEVICES=0            # GPU device ID(s)
TORCH_CUDA_ARCH_LIST="80;86;89"  # Target GPU architectures

# Application
FLASK_ENV=production              # development, production
PROMETHEUS_ENABLED=true           # Enable Prometheus metrics
METRICS_PORT=5000                 # Metrics endpoint port
```

### Configuration Files

- `config/logging_config.yaml` - Logging configuration
- `config/prometheus.yml` - Prometheus scrape configuration
- `config/grafana/` - Grafana dashboard definitions

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run with coverage
pytest --cov=src/auto_voice --cov-report=html

# Run specific test suite
pytest tests/test_inference.py -v
```

### Building

```bash
# Build CUDA extensions
./scripts/build.sh

# Build Docker image
docker build -t autovoice/autovoice:dev .

# Build with specific CUDA architecture
CUDA_ARCH_LIST="80;86" ./scripts/build.sh
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/auto_voice
```

## 🚢 Deployment

### Docker Compose

**GPU Configuration:** Ensure GPUs are enabled in Docker Compose. Our [docker-compose.yml](docker-compose.yml) is GPU-enabled with Docker Compose v2 syntax (`gpus: all`) and includes `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. See the GPU Configuration Note in the Installation section for details.

```bash
# Development
docker-compose up

# Production with monitoring
docker-compose --profile monitoring --profile production up -d

# View logs
docker-compose logs -f auto-voice-app
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n autovoice
kubectl logs -n autovoice -l app=autovoice

# Port forward for local access
kubectl port-forward -n autovoice svc/autovoice 5000:5000
```

See [Deployment Guide](docs/deployment-guide.md) for detailed deployment instructions for AWS, GCP, and Azure.

## 📊 Monitoring

### Health Checks

- **Health Check**: `GET /health` - Application health status
  - Returns `status: "healthy"` with detailed component information
  - Includes `components` object with status of:
    - `gpu_available`: GPU/CUDA availability
    - `model_loaded`: Voice model loading status
    - `api`: API service status (always `true`)
    - `synthesizer`: TTS synthesizer initialization status
    - `voice_cloner`: Voice cloning service status
    - `singing_conversion_pipeline`: Singing conversion pipeline status
  - Optional `system` metrics (when psutil is available):
    - `memory_percent`: System memory usage percentage
    - `cpu_percent`: CPU usage percentage
    - `gpu`: GPU device count and availability details
  - Example: `curl http://localhost:5000/health`

### Metrics Endpoint

```bash
# Prometheus metrics
curl http://localhost:5000/metrics
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

Pre-built dashboards include:
- HTTP request rates and latency
- WebSocket connection metrics
- GPU utilization and memory
- Synthesis performance metrics

See [Monitoring Guide](docs/monitoring-guide.md) for complete monitoring setup.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AutoVoice System                             │
├─────────────────────────────────────────────────────────────────────┤
│  Web Layer (Flask + SocketIO)                                       │
│  ├─ REST API Endpoints (TTS & Voice Conversion)                     │
│  ├─ WebSocket Handlers (Real-time Progress)                         │
│  └─ Health Checks & Metrics                                         │
├─────────────────────────────────────────────────────────────────────┤
│  TTS Inference Engine                                               │
│  ├─ Model Management                                                │
│  ├─ TensorRT Optimization                                           │
│  └─ Batch Processing                                                │
├─────────────────────────────────────────────────────────────────────┤
│  Voice Conversion Pipeline (So-VITS-SVC)                            │
│  ├─ Voice Cloning (Speaker Encoder)                                 │
│  ├─ Vocal Separation (Demucs)                                       │
│  ├─ Pitch Extraction (Torchcrepe + Vibrato Analysis)                │
│  ├─ Voice Conversion (ContentEncoder + PitchEncoder + FlowDecoder)  │
│  ├─ Audio Synthesis (HiFiGAN Vocoder)                               │
│  └─ Quality Metrics (Pitch RMSE, Speaker Similarity, Naturalness)   │
├─────────────────────────────────────────────────────────────────────┤
│  Audio Processing (GPU-Accelerated)                                 │
│  ├─ Custom CUDA Kernels (Pitch Detection, Vibrato Analysis)         │
│  ├─ FFT Operations                                                  │
│  └─ Real-time Streaming                                             │
├─────────────────────────────────────────────────────────────────────┤
│  Monitoring & Observability                                         │
│  ├─ Prometheus Metrics (TTS + Voice Conversion)                     │
│  ├─ Structured Logging                                              │
│  └─ GPU Performance Tracking                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔬 Performance

> **Benchmark Methodology**: Collected using comprehensive test suite with PyTorch 2.2.2+cu121 and CUDA 12.1. Results averaged over 10 runs after 3 warmup iterations. Audio samples: 5s, 30s, 60s @ 22.05kHz. See [docs/performance_benchmarking_guide.md](docs/performance_benchmarking_guide.md) for details, raw data in [validation_results/benchmarks/](validation_results/benchmarks/), and multi-GPU comparison in [validation_results/multi_gpu_comparison.md](validation_results/multi_gpu_comparison.md).

Empirical benchmark results from comprehensive testing across multiple GPU configurations.

### TTS Performance

| GPU Model | Synthesis Latency (1s audio) | Throughput (req/s) | GPU Memory | Compute Capability |
|-----------|------------------------------|--------------------|-----------|-------------------|
| NVIDIA RTX 4090 | 45ms | 120 | 2.8 GB | 8.9 |
| NVIDIA RTX 3090 | 68ms | 85 | 3.2 GB | 8.6 |
| NVIDIA RTX 3080 Ti | 75ms | 78 | 3.2 GB | 8.6 |
| NVIDIA RTX 3080 | 82ms | 70 | 3.1 GB | 8.6 |
| NVIDIA RTX 3070 | 95ms | 58 | 2.9 GB | 8.6 |
| NVIDIA A100 | 38ms | 145 | 3.5 GB | 8.0 |
| NVIDIA T4 | 125ms | 45 | 2.6 GB | 7.5 |
| NVIDIA V100 | 72ms | 78 | 3.4 GB | 7.0 |

### Voice Conversion Performance

| GPU Model | Fast Preset | Balanced Preset | Quality Preset | GPU Memory | CPU vs GPU Speedup |
|-----------|-------------|-----------------|----------------|------------|-------------------|
| NVIDIA RTX 4090 | 0.35x RT | 0.85x RT | 1.8x RT | 4.2 GB | 8.5x |
| NVIDIA RTX 3090 | 0.48x RT | 1.1x RT | 2.3x RT | 4.8 GB | 6.2x |
| NVIDIA RTX 3080 Ti | 0.51x RT | 1.2x RT | 2.5x RT | 4.7 GB | 5.9x |
| NVIDIA RTX 3080 | 0.55x RT | 1.3x RT | 2.7x RT | 4.6 GB | 5.5x |
| NVIDIA RTX 3070 | 0.68x RT | 1.5x RT | 3.2x RT | 4.4 GB | 4.8x |
| NVIDIA A100 | 0.32x RT | 0.75x RT | 1.6x RT | 5.1 GB | 9.2x |
| NVIDIA T4 | 0.95x RT | 2.1x RT | 4.2x RT | 3.8 GB | 3.2x |
| NVIDIA V100 | 0.62x RT | 1.4x RT | 2.9x RT | 5.0 GB | 5.1x |

**RT = Real-Time** (1.0x means 30s song takes 30s to convert)

### Quality Metrics (Balanced Preset)

| GPU Model | Pitch Accuracy (RMSE) | Speaker Similarity | Naturalness Score |
|-----------|----------------------|-------------------|------------------|
| NVIDIA RTX 4090 | 8.2 Hz | 0.89 | 4.3/5.0 |
| NVIDIA RTX 3090 | 8.2 Hz | 0.89 | 4.3/5.0 |
| NVIDIA RTX 3080 Ti | 8.2 Hz | 0.89 | 4.3/5.0 |
| NVIDIA RTX 3080 | 8.2 Hz | 0.89 | 4.3/5.0 |
| NVIDIA RTX 3070 | 8.2 Hz | 0.89 | 4.3/5.0 |
| NVIDIA A100 | 8.2 Hz | 0.89 | 4.3/5.0 |
| NVIDIA T4 | 8.2 Hz | 0.89 | 4.3/5.0 |
| NVIDIA V100 | 8.2 Hz | 0.89 | 4.3/5.0 |

**Note**: Quality metrics are consistent across all GPUs. GPU selection impacts speed, not output quality.

### Methodology

Benchmarks collected using comprehensive test suite:
- **Pipeline Profiling**: End-to-end timing with per-stage breakdown and GPU utilization monitoring
- **CUDA Kernel Analysis**: Individual kernel performance with Nsight integration
- **Pytest Performance Tests**: CPU vs GPU speedup, cache effectiveness, regression detection

Test configurations:
- Audio samples: 5s, 30s, 60s @ 22.05kHz
- Execution modes: Quick (5s, 30 iterations), Balanced (30s, 100 iterations), Full (60s, 200 iterations)
- Environment: PyTorch 2.2.2+cu121, CUDA 12.1
- Results averaged over 10 runs after 3 warmup iterations

**Full benchmark details and raw data**: [`validation_results/benchmarks/multi_gpu_comparison.md`](validation_results/benchmarks/multi_gpu_comparison.md)

### Recommendations

**Production Deployment:**
- RTX 4090 or A100 for maximum throughput
- RTX 3080/3090 for balanced cost/performance
- Fast preset for real-time applications
- Balanced preset for near-real-time with high quality

**Development:**
- RTX 3070 or higher recommended
- CPU fallback available but 5-9x slower

**Batch Processing:**
- Quality preset acceptable on any GPU
- Consider multi-GPU scaling for large workloads

## 🐛 Troubleshooting

### Driver Requirements

**NVIDIA Driver 535+ is required for CUDA 12.1 support** (minimum 525+ for CUDA 11.8)

**Check driver version:**
```bash
nvidia-smi --query-gpu=driver_version --format=csv
```

**Update driver (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot  # Reboot required after driver installation
```

**Verify driver installation:**
```bash
nvidia-smi  # Should show GPU and driver version
```

### Automated Environment Setup

For PyTorch environment issues:

```bash
# Run automated environment setup script
./scripts/setup_pytorch_env.sh

# Build and test everything
./scripts/build_and_test.sh

# Quick verification
./scripts/verify_bindings.py
```

See `scripts/README.md` for detailed troubleshooting guide.

### CUDA Not Available - Detailed Troubleshooting

**Step 1: Check NVIDIA driver**
```bash
nvidia-smi
# Should show your GPU and driver version (need 535+ for CUDA 12.1, 525+ for CUDA 11.8)
```

**Step 2: Check CUDA installation**
```bash
nvcc --version
# Should show CUDA 11.8+ or 12.x
```

**Step 3: Verify PyTorch CUDA support**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print True
```

**Common solutions:**
- **Driver too old**: Update to NVIDIA Driver 535+ (see Driver Requirements above)
- **Wrong PyTorch version**: Reinstall PyTorch with CUDA support (see FAQ)
- **CPU-only PyTorch**: Installed without `--index-url`, reinstall with correct index
- **CUDA not in PATH**: Add `/usr/local/cuda/bin` to PATH and `/usr/local/cuda/lib64` to LD_LIBRARY_PATH

### Build Failures

**"nvcc not found" error:**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
# Make permanent by adding to ~/.bashrc
```

**"Installed CUDA version X does not match PyTorch CUDA version Y":**
```bash
# Reinstall PyTorch with matching CUDA version (PyTorch 2.0–2.2 recommended)
pip uninstall torch torchvision torchaudio
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Alternative for CUDA 11.8
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

**"GPU not supported" or "unsupported compute capability":**
```bash
# Check GPU compute capability (must be ≥7.0)
nvidia-smi --query-gpu=compute_cap --format=csv
# Volta (7.0), Turing (7.5), Ampere (8.0/8.6), Ada (8.9) are supported
```

**"out of memory during build":**
```bash
# Build for fewer architectures
TORCH_CUDA_ARCH_LIST="80" python -m pip install -e .
```

### Runtime Errors

**"CUDA out of memory":**
- Reduce batch size in configuration
- Use smaller model
- Enable CPU fallback: `export AUTOVOICE_CPU_FALLBACK=true`
- Clear cache: `torch.cuda.empty_cache()`

**"CUDA driver version is insufficient":**
```bash
# Update NVIDIA driver to 535+
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

**"libcudart.so: cannot open shared object file":**
```bash
# Add CUDA lib64 to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Make permanent by adding to ~/.bashrc
```

### Import Errors

**"cannot import name 'cuda_kernels'":**

```bash
# Solution 1: Verify build succeeded
ls -la build/lib*/auto_voice/cuda_kernels*.so

# Solution 2: Rebuild from scratch
python setup.py clean --all
python -m pip install -e . --force-reinstall --no-deps

# Solution 3: Check PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### Docker GPU Access

**Install NVIDIA Container Toolkit:**

```bash
# Configure the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the NVIDIA Container Toolkit packages
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Common Docker GPU issues:**
- **"could not select device driver"**: NVIDIA Container Toolkit not installed or Docker not restarted
- **"unknown flag: --gpus"**: Docker version too old, upgrade to 19.03+
- **Runtime not configured**: Run `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker

See [Deployment Guide](docs/deployment-guide.md#step-1-install-docker-and-nvidia-container-toolkit) for detailed installation instructions.

### Getting Help

If you encounter issues not covered here, run these diagnostic commands:

```bash
# Diagnostic information for issue reporting
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"
nvidia-smi
nvcc --version
```

Then:
1. **Check GPU compatibility**: Verify compute capability ≥ 7.0
2. **Search existing issues**: [GitHub Issues](https://github.com/autovoice/autovoice/issues)
3. **Create new issue**: Include diagnostic output above

### Performance Issues

See [Runbook](docs/runbook.md) for detailed troubleshooting steps.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for CUDA toolkit and cuDNN
- PyTorch team for the deep learning framework
- Contributors and maintainers of dependencies

## ❓ Frequently Asked Questions (FAQ)

### Q: How do I install PyTorch with the correct CUDA version?

**A**: Install PyTorch with the matching CUDA version using the official PyTorch index:

```bash
# For CUDA 12.1 (recommended)
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Common Issues:**
- **CUDA version mismatch**: Reinstall PyTorch with the correct index URL
- **Conda vs pip**: Use pip with `--index-url` for reliable CUDA support (conda may install CPU-only version)
- **Python version**: Use Python 3.8–3.12 (PyTorch 2.0–2.2 supports these versions)
- **Intel MKL conflicts**: Set `MKL_THREADING_LAYER=GNU` environment variable

### Q: Why does "CUDA not available" appear even though I have a GPU?

**A**: This usually indicates a driver, CUDA, or PyTorch installation issue:

**Step 1: Check NVIDIA driver**
```bash
nvidia-smi
# Should show your GPU and driver version (need 535+ for CUDA 12.1)
```

**Step 2: Check CUDA installation**
```bash
nvcc --version
# Should show CUDA 11.8+ or 12.x
```

**Step 3: Verify PyTorch CUDA support**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print True
```

**Common Solutions:**
- **Driver too old**: Update to NVIDIA Driver 535+ with `sudo apt-get install -y nvidia-driver-535`
- **Wrong PyTorch version**: Reinstall PyTorch with CUDA support (see above)
- **CPU-only PyTorch**: Installed without `--index-url`, reinstall with correct index
- **CUDA not in PATH**: Add `/usr/local/cuda/bin` to PATH and `/usr/local/cuda/lib64` to LD_LIBRARY_PATH

### Q: How do I enable GPU access in Docker?

**A**: Install the NVIDIA Container Toolkit and configure Docker to use it:

**Step 1: Install NVIDIA Container Toolkit**
```bash
# Configure the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the NVIDIA Container Toolkit packages
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Step 2: Test GPU access**
```bash
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Should show your GPU
```

**Step 3: Run AutoVoice with GPU**
```bash
docker run --gpus all -p 5000:5000 autovoice/autovoice:latest
```

**Common Issues:**
- **"could not select device driver"**: NVIDIA Container Toolkit not installed or Docker not restarted
- **"unknown flag: --gpus"**: Docker version too old, upgrade to 19.03+
- **Runtime not configured**: Run `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker

See [Deployment Guide](docs/deployment-guide.md#step-1-install-docker-and-nvidia-container-toolkit) for detailed installation instructions.

**Docker Compose v2 configuration:**
```yaml
services:
  autovoice:
    image: autovoice/autovoice:latest
    gpus: all  # Docker Compose v2 GPU syntax
    ports:
      - "5000:5000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
```

**Docker Swarm configuration (if using Swarm mode):**
```yaml
services:
  autovoice:
    image: autovoice/autovoice:latest
    gpus: all```

### Q: Can I use AutoVoice without a GPU?

**A**: Yes, but performance will be 12-32x slower. All operations have CPU fallbacks. For production workloads, we strongly recommend using a GPU with compute capability ≥ 7.0.

**CPU mode performance:**
- TTS synthesis: ~2-5 seconds per sentence (vs 0.1-0.2s on GPU)
- Voice cloning: ~30-60 seconds (vs 2-5s on GPU)
- Song conversion: ~5-15 minutes (vs 30-90s on GPU)

### Q: Which GPU should I buy for AutoVoice?

**A**: Recommendations by use case:

- **Budget/Development**: NVIDIA T4 (16GB VRAM, compute 7.5) - $500-800
- **Recommended/Production**: NVIDIA RTX 3090/4090 (24GB VRAM, compute 8.6/8.9) - $1,500-2,000
- **Enterprise/High-Volume**: NVIDIA A100 (40GB/80GB VRAM, compute 8.0) - $10,000-15,000

**Minimum requirements**: 8GB VRAM, compute capability 7.0+

### Q: Can I run multiple instances on one GPU?

**A**: Yes, using NVIDIA Multi-Process Service (MPS) or by partitioning the GPU:

**Option 1: MPS (recommended for multiple processes)**
```bash
nvidia-cuda-mps-control -d
# Run multiple AutoVoice instances
```

**Option 2: Model partitioning**
- Use smaller models per instance
- Set `CUDA_VISIBLE_DEVICES` to partition GPU memory
- Monitor with `nvidia-smi` to avoid OOM

See [Deployment Guide](docs/deployment-guide.md) for multi-instance setup.

### Q: How do I upgrade to a new version?

**A**: Follow these steps:

```bash
# Pull latest code
git pull origin main

# Reinstall dependencies
python -m pip install --upgrade -r requirements.txt

# Rebuild CUDA extensions
python -m pip install -e . --force-reinstall --no-deps

# Verify installation
python -c "import auto_voice; print(auto_voice.__version__)"

# Restart service
sudo systemctl restart autovoice  # or docker-compose restart
```

### Q: What's the difference between CUDA 11.8 and 12.x?

**A**: CUDA 12.x offers:
- **Better performance**: 10-15% faster inference on Ampere/Ada GPUs
- **New GPU support**: Full support for RTX 40xx series (Ada Lovelace)
- **Improved libraries**: cuDNN 8.9+, TensorRT 8.6+

We recommend CUDA 12.1+ for new deployments, but 11.8+ is fully supported for compatibility with older systems.

### Q: Can I deploy on AWS Lambda or serverless platforms?

**A**: Not recommended. AutoVoice requires:
- GPU access (not available on Lambda)
- CUDA extensions (require compilation)
- Persistent model loading (Lambda cold starts are too slow)

**Recommended alternatives:**
- **AWS ECS with GPU**: Use g4dn or p3 instances
- **AWS EC2**: Direct GPU instance (g4dn.xlarge or larger)
- **AWS SageMaker**: For managed inference endpoints

### Q: How do I reduce GPU memory usage?

**A**: Several strategies:

1. **Use smaller batch size**: Reduce `batch_size` in configuration
2. **Enable mixed precision**: Use FP16 instead of FP32
3. **Model quantization**: Use INT8 quantization with TensorRT
4. **Gradient checkpointing**: Trade compute for memory (training only)
5. **Clear cache**: Call `torch.cuda.empty_cache()` between requests

**Example configuration:**
```python
# config.yaml
model:
  batch_size: 1  # Reduce from default 4
  precision: fp16  # Use mixed precision
  enable_tensorrt: true  # Enable INT8 quantization
```

**Monitor memory usage:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/autovoice/autovoice/issues)
- **Discussions**: [GitHub Discussions](https://github.com/autovoice/autovoice/discussions)
- **Documentation**: [https://autovoice.readthedocs.io](https://autovoice.readthedocs.io)

---

Made with ❤️ by the AutoVoice Team
