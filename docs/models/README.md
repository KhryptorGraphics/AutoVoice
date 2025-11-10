# Neural Model Integration

Complete infrastructure for integrating trained neural models with Auto Voice Cloning.

## Quick Links

- [Model Integration Guide](../MODEL_INTEGRATION.md) - Comprehensive guide
- [Example Code](../../examples/model_integration_example.py) - Working examples
- [Download Script](../../scripts/download_models.py) - Model download utility
- [Configuration](../../config/models.yaml) - Model configurations

## Overview

This module provides a complete system for managing neural models:

```
┌─────────────────────────────────────────────────────┐
│                 Model Registry                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Config  │  │  Loader  │  │   Downloader     │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────┬───────────────────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
┌─────▼─────┐ ┌──▼───┐ ┌─────▼────────┐
│  HuBERT   │ │ HiFi │ │   Speaker    │
│           │ │ GAN  │ │   Encoder    │
└───────────┘ └──────┘ └──────────────┘
```

## Features

✅ **Automatic Downloads** - Models downloaded on first use
✅ **Version Management** - Track and manage model versions
✅ **Mock Mode** - Test without downloading large files
✅ **Lazy Loading** - Models loaded only when needed
✅ **Caching** - Downloaded models cached locally
✅ **Graceful Fallback** - Auto-fallback to mock on errors
✅ **GPU Support** - Automatic GPU detection and usage
✅ **Pipeline Integration** - Seamless integration with pipelines

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py
```

### 2. Basic Usage

```python
from auto_voice.models import ModelRegistry

# Initialize (mock mode for development)
registry = ModelRegistry(use_mock=True)

# Load models
hubert = registry.load_hubert()
hifigan = registry.load_hifigan()
speaker_encoder = registry.load_speaker_encoder()
```

### 3. With Pipeline

```python
from auto_voice.inference import VoiceConversionPipeline, PipelineConfig

# Pipeline with model integration
config = PipelineConfig(use_mock_models=False)
pipeline = VoiceConversionPipeline(config)

# Models loaded automatically
converted = pipeline.convert(source_audio, target_embedding)
```

## Architecture

### Components

1. **ModelRegistry** (`model_registry.py`)
   - Central hub for model management
   - Handles configuration and caching
   - Provides lazy loading

2. **ModelLoader** (`model_loader.py`)
   - Downloads models from URLs
   - Verifies checksums
   - Manages local storage

3. **Model Wrappers**
   - `HuBERTModel` - Speech representation
   - `HiFiGANModel` - Audio synthesis
   - `SpeakerEncoderModel` - Speaker embeddings

### File Structure

```
src/auto_voice/models/
├── __init__.py                # Module exports
├── model_registry.py          # Central registry
├── model_loader.py            # Download/load utilities
├── hubert_model.py            # HuBERT wrapper
├── hifigan_model.py           # HiFi-GAN wrapper
└── speaker_encoder.py         # Speaker encoder wrapper

config/
└── models.yaml                # Model configurations

scripts/
└── download_models.py         # Download utility

docs/
├── MODEL_INTEGRATION.md       # Full guide
└── models/
    └── README.md              # This file
```

## Supported Models

### HuBERT

**Purpose**: Extract content features from speech
**Source**: facebook/hubert-base-ls960
**Size**: ~377 MB
**Memory**: 4 GB minimum
**GPU**: Optional

```python
hubert = registry.load_hubert()
features = hubert.extract_features(audio, sample_rate=16000)
# Returns: (batch, time_steps, 768)
```

### HiFi-GAN

**Purpose**: High-quality audio synthesis
**Source**: nvidia/hifigan
**Size**: ~55 MB
**Memory**: 2 GB minimum
**GPU**: Recommended

```python
hifigan = registry.load_hifigan()
audio = hifigan.synthesize(mel_spectrogram)
# Returns: (audio_samples,)
```

### Speaker Encoder

**Purpose**: Extract speaker identity embeddings
**Source**: speechbrain/spkrec-ecapa-voxceleb
**Size**: ~43 MB
**Memory**: 2 GB minimum
**GPU**: Optional

```python
encoder = registry.load_speaker_encoder()
embedding = encoder.encode(audio, sample_rate=16000)
# Returns: (192,) normalized vector
```

## Configuration

### models.yaml

```yaml
models:
  hubert_base:
    name: hubert_base
    model_type: hubert
    version: 1.0.0
    url: https://huggingface.co/...
    requires_gpu: false
    min_memory_gb: 4.0
```

### Environment Variables

```bash
# .env
MODEL_DIR=models/
USE_GPU=auto
DOWNLOAD_TIMEOUT=3600
USE_MOCK_MODELS=false
```

## Usage Patterns

### Development Workflow

```python
# Use mock models during development
registry = ModelRegistry(use_mock=True)

# Fast iteration, no downloads needed
model = registry.load_hubert()
features = model(test_audio)
```

### Production Deployment

```python
# Download models first
# $ python scripts/download_models.py

# Use real models in production
registry = ModelRegistry(
    model_dir='models/',
    use_mock=False
)

# Warmup for faster first inference
registry.warmup_models()
```

### Custom Models

```python
# Add to models.yaml
custom_hubert:
  name: custom_hubert
  model_type: hubert
  version: 2.0.0
  local_path: /path/to/custom.pt

# Load custom model
model = registry.load_hubert('custom_hubert')
```

## Testing

### Run Tests

```bash
# Test model registry
pytest tests/models/test_model_registry.py

# Test with mock models
pytest tests/models/ -v

# Test downloads (requires internet)
pytest tests/models/ --run-download-tests
```

### Example Tests

```python
def test_mock_model():
    registry = ModelRegistry(use_mock=True)
    hubert = registry.load_hubert()

    audio = np.random.randn(16000)
    features = hubert.extract_features(audio)

    assert features.shape[2] == 768  # HuBERT hidden size

def test_model_caching():
    registry = ModelRegistry(use_mock=True)

    model1 = registry.load_hubert()
    model2 = registry.load_hubert()

    assert model1 is model2  # Same instance
```

## Performance

### Memory Requirements

| Scenario | Mock Mode | Real Models | With GPU |
|----------|-----------|-------------|----------|
| Development | <100 MB | 500 MB | N/A |
| Single Model | <100 MB | 400 MB | 2 GB VRAM |
| All Models | <100 MB | 1.5 GB | 4 GB VRAM |
| With Pipeline | 200 MB | 2 GB | 6 GB VRAM |

### Optimization Tips

1. **Use Mock Mode** for development
2. **Lazy Loading** - Load models only when needed
3. **Warmup** - Pre-load models for production
4. **GPU** - Use GPU for HiFi-GAN synthesis
5. **Caching** - Keep models in memory

## Troubleshooting

### Download Fails

```bash
# Increase timeout
export DOWNLOAD_TIMEOUT=7200

# Force re-download
python scripts/download_models.py --force
```

### Out of Memory

```python
# Use mock mode
registry = ModelRegistry(use_mock=True)

# Or clear cache
registry.clear_cache()
```

### Import Errors

```python
# Check imports
from auto_voice.models import ModelRegistry

# If fails, check installation
pip install -e .
```

## Examples

See [examples/model_integration_example.py](../../examples/model_integration_example.py) for:

- Mock models usage
- Real model loading
- Pipeline integration
- Model warmup
- Custom configurations
- Model inspection

Run with:
```bash
python examples/model_integration_example.py
```

## API Reference

### ModelRegistry

```python
ModelRegistry(
    model_dir: str = 'models/',
    config_path: Optional[str] = None,
    use_mock: bool = False
)

# Methods
.load_hubert(model_name: str = 'hubert_base') -> HuBERTModel
.load_hifigan(model_name: str = 'hifigan_universal') -> HiFiGANModel
.load_speaker_encoder(model_name: str = 'speaker_encoder') -> SpeakerEncoderModel
.warmup_models(model_names: Optional[List[str]] = None)
.list_models() -> List[str]
.get_config(model_name: str) -> ModelConfig
.clear_cache()
```

### HuBERTModel

```python
HuBERTModel(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    use_mock: bool = False,
    device: str = 'cpu'
)

# Methods
.extract_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray
```

### HiFiGANModel

```python
HiFiGANModel(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    use_mock: bool = False,
    device: str = 'cpu'
)

# Methods
.synthesize(mel_spectrogram: np.ndarray) -> np.ndarray
```

### SpeakerEncoderModel

```python
SpeakerEncoderModel(
    model_path: Optional[str] = None,
    use_mock: bool = False,
    device: str = 'cpu'
)

# Methods
.encode(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray
.compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float
```

## Next Steps

- Read [Model Integration Guide](../MODEL_INTEGRATION.md)
- Run [Example Code](../../examples/model_integration_example.py)
- Download models with [Download Script](../../scripts/download_models.py)
- Customize [Configuration](../../config/models.yaml)

## License

See [LICENSE](../../LICENSE) for details.
