# Model Integration Guide

This guide explains how to integrate and use neural models in the Auto Voice Cloning system.

## Overview

The model integration infrastructure provides:

- **Model Registry**: Central management for all neural models
- **Automatic Downloading**: Models are downloaded on-demand from HuggingFace
- **Version Management**: Track and manage model versions
- **Graceful Fallback**: Automatic fallback to mock models for testing
- **Model Warmup**: Pre-load models for faster inference
- **Caching**: Downloaded models are cached locally

## Quick Start

### 1. Basic Usage

```python
from auto_voice.models import ModelRegistry

# Initialize registry (uses mock mode by default for development)
registry = ModelRegistry(use_mock=True)

# Load models
hubert = registry.load_hubert()
hifigan = registry.load_hifigan()
speaker_encoder = registry.load_speaker_encoder()

# Use models
audio = np.random.randn(16000)
features = hubert.extract_features(audio)
embedding = speaker_encoder.encode(audio)
```

### 2. Download Real Models

```bash
# List available models
python scripts/download_models.py --list

# Download all models
python scripts/download_models.py

# Download specific model
python scripts/download_models.py --model hubert_base

# Force re-download
python scripts/download_models.py --model hifigan_universal --force
```

### 3. Use Real Models

```python
# Initialize with real models
registry = ModelRegistry(
    model_dir='models/',
    use_mock=False  # Use real downloaded models
)

# Models will be loaded from disk or downloaded automatically
hubert = registry.load_hubert()
```

## Architecture

### Model Types

The system supports three main model types:

1. **HuBERT** - Speech representation learning
   - Extracts content features from audio
   - Model: facebook/hubert-base-ls960
   - Size: ~377 MB
   - Memory: 4 GB minimum

2. **HiFi-GAN** - Neural vocoder
   - Synthesizes audio from mel-spectrograms
   - Model: nvidia/hifigan
   - Size: ~55 MB
   - Memory: 2 GB minimum
   - GPU recommended

3. **Speaker Encoder** - Speaker verification
   - Extracts speaker embeddings
   - Model: speechbrain/spkrec-ecapa-voxceleb
   - Size: ~43 MB
   - Memory: 2 GB minimum

### Directory Structure

```
autovoice/
├── config/
│   └── models.yaml          # Model configurations
├── models/                  # Downloaded models cache
│   ├── hubert_base_v1.0.0.pt
│   ├── hifigan_universal_v1.0.0.pt
│   └── speaker_encoder_v1.0.0.ckpt
├── scripts/
│   └── download_models.py   # Model download script
└── src/auto_voice/models/
    ├── model_registry.py    # Central registry
    ├── model_loader.py      # Download & loading
    ├── hubert_model.py      # HuBERT wrapper
    ├── hifigan_model.py     # HiFi-GAN wrapper
    └── speaker_encoder.py   # Speaker encoder wrapper
```

## Configuration

### models.yaml

Model configurations are defined in `config/models.yaml`:

```yaml
models:
  hubert_base:
    name: hubert_base
    model_type: hubert
    version: 1.0.0
    url: https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin
    requires_gpu: false
    min_memory_gb: 4.0
    metadata:
      description: HuBERT base model for speech representation
      sample_rate: 16000
      hidden_size: 768
```

### Environment Variables

Set these in `.env`:

```bash
# Model directory
MODEL_DIR=models/

# Cache directory for downloads
CACHE_DIR=~/.cache/auto_voice/

# GPU usage
USE_GPU=auto  # auto, true, or false

# Download timeout
DOWNLOAD_TIMEOUT=3600
```

## Advanced Usage

### Custom Model Paths

```python
# Use local model files
config = ModelConfig(
    name='custom_hubert',
    model_type=ModelType.HUBERT,
    version='1.0.0',
    local_path='/path/to/model.pt',
    config_path='/path/to/config.json'
)

registry = ModelRegistry(use_mock=False)
model = registry.load_hubert('custom_hubert')
```

### Model Warmup

```python
# Warmup all models during initialization
registry = ModelRegistry(use_mock=False)
registry.warmup_models()  # Loads all models into memory

# Warmup specific models
registry.warmup_models(['hubert_base', 'hifigan_universal'])
```

### Model Caching

```python
# Models are cached after first load
hubert1 = registry.load_hubert()  # Downloads/loads from disk
hubert2 = registry.load_hubert()  # Returns cached instance

# Clear cache if needed
registry.clear_cache()
```

### Version Management

```python
# List available models
models = registry.list_models()
# ['hubert_base', 'hifigan_universal', 'speaker_encoder']

# Get model configuration
config = registry.get_config('hubert_base')
print(f"Version: {config.version}")
print(f"Type: {config.model_type}")
print(f"GPU Required: {config.requires_gpu}")

# Check download status
is_downloaded = registry.is_model_downloaded('hubert_base')
```

## Integration with Pipeline

### VoiceConversionPipeline

```python
from auto_voice.inference import VoiceConversionPipeline
from auto_voice.models import ModelRegistry

# Initialize registry
registry = ModelRegistry(use_mock=False)

# Create pipeline with real models
pipeline = VoiceConversionPipeline(
    model_registry=registry,
    config=PipelineConfig(use_cuda=True)
)

# Models will be loaded automatically during pipeline initialization
# and used for conversion
converted_audio = pipeline.convert(
    source_audio,
    target_embedding
)
```

## Mock Mode for Testing

Mock mode allows testing without downloading large models:

```python
# Use mock models
registry = ModelRegistry(use_mock=True)

# All models return mock implementations
hubert = registry.load_hubert()  # Returns HuBERTModel(use_mock=True)

# Mock models return realistic-shaped random data
audio = np.random.randn(16000)
features = hubert(audio)  # Returns random features with correct shape
```

Benefits:
- No downloads required
- Faster testing
- Deterministic results (for same input)
- Compatible API with real models

## Performance Considerations

### Memory Usage

| Model | Size | Min RAM | Recommended RAM | GPU RAM |
|-------|------|---------|----------------|---------|
| HuBERT | 377 MB | 4 GB | 8 GB | 2 GB |
| HiFi-GAN | 55 MB | 2 GB | 4 GB | 1 GB |
| Speaker Encoder | 43 MB | 2 GB | 4 GB | 0.5 GB |

### GPU Acceleration

```python
# Enable GPU for supported models
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Models will automatically use GPU
    registry = ModelRegistry(use_mock=False)
    hifigan = registry.load_hifigan()
```

### Download Optimization

```bash
# Pre-download all models before production
python scripts/download_models.py

# Verify downloads
python scripts/download_models.py --list
```

## Troubleshooting

### Model Download Fails

```python
# Increase timeout
os.environ['DOWNLOAD_TIMEOUT'] = '7200'

# Force re-download
python scripts/download_models.py --model hubert_base --force
```

### Out of Memory

```python
# Use mock mode for development
registry = ModelRegistry(use_mock=True)

# Or reduce batch size
config = PipelineConfig(batch_size=1)
```

### GPU Not Detected

```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

# Force CPU mode
os.environ['USE_GPU'] = 'false'
```

## Model Updates

### Updating Model Versions

1. Update `config/models.yaml` with new URLs/versions
2. Force re-download:
   ```bash
   python scripts/download_models.py --force
   ```

### Adding Custom Models

1. Add configuration to `config/models.yaml`:
   ```yaml
   custom_model:
     name: custom_model
     model_type: hubert
     version: 2.0.0
     url: https://example.com/model.pt
   ```

2. Load in code:
   ```python
   model = registry.load_hubert('custom_model')
   ```

## API Reference

### ModelRegistry

```python
class ModelRegistry:
    def __init__(
        model_dir: str = 'models/',
        config_path: Optional[str] = None,
        use_mock: bool = False
    )

    def load_hubert(model_name: str = 'hubert_base') -> HuBERTModel
    def load_hifigan(model_name: str = 'hifigan_universal') -> HiFiGANModel
    def load_speaker_encoder(model_name: str = 'speaker_encoder') -> SpeakerEncoderModel

    def warmup_models(model_names: Optional[List[str]] = None)
    def list_models() -> List[str]
    def get_config(model_name: str) -> ModelConfig
    def is_model_downloaded(model_name: str) -> bool
    def clear_cache()
```

### Model Wrappers

```python
class HuBERTModel:
    def extract_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray
    def __call__(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray

class HiFiGANModel:
    def synthesize(mel_spectrogram: np.ndarray) -> np.ndarray
    def __call__(mel_spectrogram: np.ndarray) -> np.ndarray

class SpeakerEncoderModel:
    def encode(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float
    def __call__(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray
```

## Best Practices

1. **Use Mock Mode for Development**
   - Faster iteration
   - No large downloads
   - Consistent behavior

2. **Download Models in Advance**
   - Run download script before production
   - Reduces first-run latency

3. **Enable Model Warmup**
   - Pre-load models during initialization
   - Faster first inference

4. **Monitor Memory Usage**
   - Check available RAM
   - Use GPU when possible
   - Clear cache if needed

5. **Handle Errors Gracefully**
   - Models auto-fallback to mock mode on error
   - Check logs for issues
   - Verify downloads

## Next Steps

- [Training Guide](TRAINING.md) - Train custom models
- [Pipeline Guide](PIPELINE.md) - Integrate with pipelines
- [API Documentation](API.md) - Full API reference
