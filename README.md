# AutoVoice

[![CI](https://github.com/autovoice/autovoice/workflows/CI/badge.svg)](https://github.com/autovoice/autovoice/actions)
[![Docker Build](https://github.com/autovoice/autovoice/workflows/Docker%20Build/badge.svg)](https://github.com/autovoice/autovoice/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.9%2B-green)](https://developer.nvidia.com/cuda-toolkit)

**GPU-accelerated voice synthesis system with real-time processing and TensorRT optimization**

AutoVoice is a high-performance voice synthesis platform leveraging CUDA acceleration, WebSocket streaming, and production-grade monitoring for real-time audio generation.

## ✨ Features

- 🚀 **CUDA Acceleration**: Optimized GPU kernels for 10-50x faster processing
- ⚡ **TensorRT Support**: Inference optimization with INT8/FP16 quantization
- 🎙️ **Real-time Processing**: WebSocket streaming for low-latency synthesis
- 🔊 **Multi-Speaker**: Support for multiple voice models and speakers
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

# Build CUDA extensions
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

## 📖 Documentation

- [Deployment Guide](docs/deployment-guide.md) - Production deployment instructions
- [API Documentation](docs/api-documentation.md) - Complete API reference
- [Monitoring Guide](docs/monitoring-guide.md) - Observability and monitoring setup
- [Runbook](docs/runbook.md) - Operational procedures and troubleshooting

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

- **Liveness**: `GET /health/live` - Basic application health
- **Readiness**: `GET /health/ready` - Ready to serve traffic

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
┌─────────────────────────────────────────────────────────┐
│                   AutoVoice System                       │
├─────────────────────────────────────────────────────────┤
│  Web Layer (Flask + SocketIO)                           │
│  ├─ REST API Endpoints                                  │
│  ├─ WebSocket Handlers                                  │
│  └─ Health Checks & Metrics                             │
├─────────────────────────────────────────────────────────┤
│  Inference Engine                                        │
│  ├─ Model Management                                    │
│  ├─ TensorRT Optimization                               │
│  └─ Batch Processing                                    │
├─────────────────────────────────────────────────────────┤
│  Audio Processing (GPU-Accelerated)                     │
│  ├─ Custom CUDA Kernels                                 │
│  ├─ FFT Operations                                      │
│  └─ Real-time Streaming                                 │
├─────────────────────────────────────────────────────────┤
│  Monitoring & Observability                             │
│  ├─ Prometheus Metrics                                  │
│  ├─ Structured Logging                                  │
│  └─ GPU Performance Tracking                            │
└─────────────────────────────────────────────────────────┘
```

## 🔬 Performance

- **Synthesis Latency**: <100ms for 1-second audio (GPU)
- **Throughput**: 50-100 concurrent requests (single GPU)
- **GPU Memory**: 2-4GB VRAM (depending on model size)
- **CPU Fallback**: Supported for non-GPU environments

## 🐛 Troubleshooting

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
