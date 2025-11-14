# AutoVoice Development Environment Setup Guide

## ✅ Quick Setup (5 Minutes)

You've already completed the Python 3.13 → 3.12 fix! Here's what's been done and what's next.

### Current Status

✅ **Python 3.12.12** environment created  
✅ **PyTorch 2.5.1+cu121** installed  
✅ **All dependencies** installed (111 packages)  
✅ **AutoVoice** installed in development mode  
✅ **No more segfaults!**

### Environment Details

```bash
Environment: autovoice
Python: 3.12.12
PyTorch: 2.5.1+cu121
CUDA Runtime: 12.1
AutoVoice: 1.0.0
```

---

## 🚀 Running the System Locally

### 1. Activate Environment

```bash
conda activate autovoice
```

### 2. Start the Web Server

```bash
# Basic server (port 5000)
python -m auto_voice.web.app

# Or with custom configuration
python -m auto_voice.web.app --port 8080 --host 0.0.0.0
```

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Synthesize speech
curl -X POST http://localhost:5000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from AutoVoice!", "speaker_id": 0}'
```

### 4. Run Example Scripts

```bash
# Voice conversion demo
python examples/demo_voice_conversion.py

# Batch conversion
python examples/demo_batch_conversion.py

# Inference demo
python examples/inference_demo.py
```

---

## 🧪 Running Tests

### Quick Test

```bash
# Run smoke tests (fast)
pytest tests/test_imports.py -v

# Run core tests
pytest tests/test_core_integration.py -v
```

### Full Test Suite

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=auto_voice --cov-report=html

# Specific test categories
pytest tests/inference/ -v  # Inference tests
pytest tests/models/ -v     # Model tests
pytest tests/gpu/ -v        # GPU tests
```

---

## 📁 Project Structure

```
autovoice/
├── src/auto_voice/          # Main package
│   ├── inference/           # Inference engines
│   ├── models/              # Model definitions
│   ├── training/            # Training pipelines
│   ├── audio/               # Audio processing
│   ├── gpu/                 # GPU management
│   ├── utils/               # Utilities
│   └── web/                 # Web server
├── tests/                   # Test suite
├── examples/                # Example scripts
├── config/                  # Configuration files
├── models/                  # Pre-trained models
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

---

## 🔧 Development Workflow

### 1. Make Changes

Edit files in `src/auto_voice/` - changes are immediately available since it's installed in editable mode.

### 2. Run Tests

```bash
pytest tests/test_your_feature.py -v
```

### 3. Check Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linter (if configured)
flake8 src/ tests/
```

### 4. Build CUDA Extensions (if modified)

```bash
python setup.py build_ext --inplace
```

---

## 📊 Monitoring & Debugging

### Check GPU Status

```bash
# NVIDIA GPU info
nvidia-smi

# GPU monitoring
nvitop

# AutoVoice GPU manager
python -c "from auto_voice.gpu import GPUManager; print(GPUManager().get_status())"
```

### View Logs

```bash
# Application logs
tail -f logs/autovoice.log

# Web server logs
tail -f logs/web_server.log
```

### Debug Mode

```bash
# Enable debug logging
export AUTOVOICE_LOG_LEVEL=DEBUG
python -m auto_voice.web.app
```

---

## 🐳 Docker Development

### Build Image

```bash
docker build -t autovoice:dev .
```

### Run Container

```bash
docker run --gpus all -p 5000:5000 autovoice:dev
```

---

## 📚 Next Steps

1. **Download Models** (if not already done):
   ```bash
   python scripts/download_pretrained_models.py
   ```

2. **Run Validation**:
   ```bash
   python scripts/validate_installation.py
   ```

3. **Try Examples**:
   ```bash
   cd examples/
   python inference_demo.py
   ```

4. **Read Documentation**:
   - [API Documentation](API.md)
   - [Model Guide](MODEL_GUIDE.md)
   - [Deployment Guide](DEPLOYMENT_GUIDE.md)

---

## ⚠️ Troubleshooting

### CUDA Not Available

This is expected in WSL2 or without NVIDIA GPU. The system will fall back to CPU mode automatically.

### Import Errors

```bash
# Reinstall in development mode
pip install -e .
```

### Port Already in Use

```bash
# Use different port
python -m auto_voice.web.app --port 8080
```

### Memory Issues

```bash
# Reduce batch size in config
export AUTOVOICE_BATCH_SIZE=1
```

---

## 🎯 Common Tasks

### Add New Feature

1. Create feature branch
2. Implement in `src/auto_voice/`
3. Add tests in `tests/`
4. Run test suite
5. Update documentation

### Update Dependencies

```bash
# Update requirements.txt
pip install -r requirements.txt --upgrade

# Reinstall package
pip install -e .
```

### Profile Performance

```bash
python scripts/profile_performance.py
```

---

## 📞 Support

- **Documentation**: `docs/`
- **Issues**: GitHub Issues
- **Examples**: `examples/`
- **Tests**: `tests/`

