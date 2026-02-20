# DeepWiki: Demucs/HTDemucs Architecture Analysis

## Research Source
- DeepWiki MCP queries on `facebookresearch/demucs`
- Topics: wiki structure, HTDemucs architecture, Python API, pretrained models

---

## 1. Repository Structure

The `facebookresearch/demucs` repository is organized into 9 sections:
1. Overview
2. Installation
3. Command-Line Interface
4. Python API
5. Model Architecture (Models/Variants, Model Loading)
6. Audio Processing Pipeline
7. Training System
8. Development/Testing
9. Version History

---

## 2. HTDemucs Architecture

Hybrid Transformer Demucs processes audio through **parallel time-domain and frequency-domain branches**:

```
Input Waveform
    │
    ├─── STFT ──→ Frequency Branch (Spectral)
    │                    │
    │              Encoder (Conv + skip connections)
    │                    │
    │              CrossTransformerEncoder ←──┐
    │                    │                    │
    │              Decoder (TransConv + skip) │
    │                    │                    │
    │              Masking + ISTFT            │
    │                    │                    │
    └──────────→ Time Branch (Waveform)       │
                         │                    │
                   Encoder (Conv)             │
                         │                    │
                   CrossTransformerEncoder ───┘
                         │
                   Decoder (TransConv)
                         │
                   Waveform output
                         │
    Combined Output ←────┘
```

### Key Components:
- **CrossTransformerEncoder**: Self-attention within each domain + cross-attention across domains
- **U-Net skip connections**: Preserve fine-grained detail through encoder-decoder path
- **Frequency masking**: Spectral branch applies learned mask to input STFT
- **Output combination**: Frequency (ISTFT) + Time (waveform) outputs are summed

---

## 3. Python API

### Separator Class
```python
from demucs.api import Separator, save_audio, list_models

# List available models
models = list_models()  # ['htdemucs', 'htdemucs_ft', 'htdemucs_6s', ...]

# Initialize
separator = Separator(
    model="htdemucs",    # Model name
    device="cuda",       # Device
    segment=12,          # Segment length in seconds (memory tradeoff)
)

# Separate from file
original, separated = separator.separate_audio_file("song.mp3")
# original: Tensor [channels, samples]
# separated: dict {stem_name: Tensor [channels, samples]}

# Separate from tensor
separated = separator.separate_tensor(audio_tensor)

# Save output
for stem_name, stem_audio in separated.items():
    save_audio(stem_audio, f"{stem_name}.wav", samplerate=separator.samplerate)

# Get just vocals
vocals = separated["vocals"]
```

### Key Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | "htdemucs" | Model variant to use |
| `device` | "cpu" | Device for inference |
| `segment` | None | Chunk size in seconds (lower = less memory) |
| `overlap` | 0.25 | Overlap ratio between segments |
| `shifts` | 0 | Number of random shifts for TTA (quality vs speed) |

### CLI Shortcut:
```bash
# Vocals only (fastest)
demucs --two-stems=vocals song.mp3

# Full 4-stem separation
demucs song.mp3

# With fine-tuned model
demucs --model htdemucs_ft song.mp3
```

---

## 4. Pretrained Models & Performance

| Model | SDR (dB) | Stems | Speed | Notes |
|-------|----------|-------|-------|-------|
| `htdemucs` | 9.0 | 4 | 1x | Default, MusDB + 800 songs |
| `htdemucs_ft` | ~9.2 | 4 | 0.25x | Fine-tuned per-source, best quality |
| `htdemucs_6s` | - | 6 | 1x | Adds piano + guitar stems |
| `mdx` | 7.5 | 4 | 1x | MDX challenge track A winner |
| `mdx_extra` | - | 4 | 1x | Extra data, MDX track B 2nd |
| `mdx_q` | ~7.4 | 4 | 1x | Quantized (smaller) |

**State-of-the-art**: 9.20 dB SDR with sparse attention + per-source fine-tuning

### Stem Names:
- 4-stem: `drums`, `bass`, `other`, `vocals`
- 6-stem: adds `piano`, `guitar`

---

## 5. Integration Plan for AutoVoice (AV-008)

### VocalSeparator Class Design:
```python
class VocalSeparator:
    """Separates vocals from accompaniment using HTDemucs."""

    def __init__(self, model: str = "htdemucs", device: str = "cuda"):
        from demucs.api import Separator
        self.separator = Separator(model=model, device=device, segment=12)

    def separate(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """Returns (vocals_tensor, sample_rate)."""
        _, separated = self.separator.separate_audio_file(audio_path)
        return separated["vocals"], self.separator.samplerate

    def separate_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        """Separate vocals from pre-loaded tensor."""
        separated = self.separator.separate_tensor(waveform)
        return separated["vocals"]
```

### Pipeline Integration:
```
SingingConversionPipeline.convert():
    1. VocalSeparator.separate(input_audio)  # NEW: extract vocals
    2. ContentEncoder.encode(vocals)          # Content features
    3. PitchEncoder.encode(vocals)            # F0 features
    4. SoVitsSvc.forward(content, pitch, speaker)  # Decode
    5. BigVGANVocoder.forward(mel)            # Waveform
```

### Installation:
```bash
# In autovoice-thor conda env
pip install demucs
# Downloads ~300MB model on first use
```

### Memory Considerations:
- `segment=12` keeps GPU memory under 4GB
- For Jetson Thor (64GB), can use `segment=None` (full song in memory)
- `shifts=1` improves quality slightly at 2x cost
