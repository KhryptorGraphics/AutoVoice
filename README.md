# AutoVoice

[![CI](https://github.com/autovoice/autovoice/workflows/CI/badge.svg)](https://github.com/autovoice/autovoice/actions)
[![Docker Build](https://github.com/autovoice/autovoice/workflows/Docker%20Build/badge.svg)](https://github.com/autovoice/autovoice/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.9%2B-green)](https://developer.nvidia.com/cuda-toolkit)

**GPU-accelerated voice synthesis and singing voice conversion system with real-time processing and TensorRT optimization**

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

- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, or newer)
- **CUDA**: CUDA Toolkit 11.8 or later
- **Python**: Python 3.8 or later
- **Docker** (optional): Docker 20.10+ with nvidia-docker runtime

### Installation

#### Option 1: Docker (Recommended)

```bash
# Pull the latest image
docker pull autovoice/autovoice:latest

# Run with GPU support
docker run --gpus all -p 5000:5000 autovoice/autovoice:latest

# Or use docker-compose
docker-compose up
```

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

# Synthesize speech
audio = voice.synthesize("Hello, world!", speaker_id="speaker_001")

# Save to file
audio.save("output.wav")
```

### REST API Example

```bash
# Health check
curl http://localhost:5000/health

# Synthesize voice
curl -X POST http://localhost:5000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "speaker_id": "default"}'
```

### WebSocket Example

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  socket.emit('synthesize_stream', {
    text: 'Hello, world!',
    speaker_id: 'default'
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
    target_profile_id=profile['profile_id'],
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

# Convert song
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "target_profile_id=550e8400-e29b-41d4-a716-446655440000"

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
  - Returns `healthy`, `degraded`, or `unhealthy` status
  - Includes GPU availability and model status
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

### TTS Performance
- **Synthesis Latency**: <100ms for 1-second audio (GPU)
- **Throughput**: 50-100 concurrent requests (single GPU)
- **GPU Memory**: 2-4GB VRAM (depending on model size)
- **CPU Fallback**: Supported for non-GPU environments

### Voice Conversion Performance
- **Conversion Speed**: ~1x real-time (balanced preset) for 30-second song
- **Fast Preset**: ~0.5x real-time (15-30s for 30s song)
- **Quality Preset**: ~2x real-time (60-120s for 30s song)
- **Pitch Accuracy**: <10 Hz RMSE (imperceptible to listeners)
- **Speaker Similarity**: >85% cosine similarity
- **GPU Memory**: 4-8GB VRAM for voice conversion
- **TensorRT Acceleration**: 2-3x additional speedup available

## 🐛 Troubleshooting

### Automated Environment Setup

For PyTorch environment issues (especially Python 3.13 compatibility):

```bash
# Run automated environment setup script
./scripts/setup_pytorch_env.sh

# Build and test everything
./scripts/build_and_test.sh

# Quick verification
./scripts/verify_bindings.py
```

See `scripts/README.md` for detailed troubleshooting guide.

### CUDA Not Available

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

### Docker GPU Access

```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

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

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/autovoice/autovoice/issues)
- **Discussions**: [GitHub Discussions](https://github.com/autovoice/autovoice/discussions)
- **Documentation**: [https://autovoice.readthedocs.io](https://autovoice.readthedocs.io)

---

Made with ❤️ by the AutoVoice Team
