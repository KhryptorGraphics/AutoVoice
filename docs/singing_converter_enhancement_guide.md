# SingingVoiceConverter Enhancement Guide

This guide covers the new features added to the `SingingVoiceConverter` class for advanced voice conversion control.

## Table of Contents

1. [Temperature API](#temperature-api)
2. [Pitch Shifting](#pitch-shifting)
3. [Quality Presets](#quality-presets)
4. [Advanced Features](#advanced-features)
5. [Complete Examples](#complete-examples)

## Temperature API

Temperature controls the randomness and expressiveness of the flow decoder during inference.

### Basic Usage

```python
from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter

# Initialize model
model = SingingVoiceConverter(config)
model.prepare_for_inference()

# Set temperature manually
model.set_temperature(1.2)  # More expressive
audio = model.convert(source_audio, target_embedding)

# Or use lower temperature for stability
model.set_temperature(0.7)  # More stable
audio = model.convert(source_audio, target_embedding)
```

### Temperature Guidelines

| Range | Behavior | Use Case |
|-------|----------|----------|
| 0.1-0.5 | Very stable, less expressive | Clean vocals, speech |
| 0.6-0.9 | Balanced stability | General conversion |
| 1.0-1.3 | More expressive | Singing, emotional content |
| 1.4-2.0 | Highly expressive | Creative effects |

### Auto-Tuning

```python
# Automatically determine optimal temperature based on audio characteristics
optimal_temp = model.auto_tune_temperature(source_audio, target_embedding, sample_rate=16000)
print(f"Optimal temperature: {optimal_temp:.2f}")

# Temperature is automatically set, just convert
audio = model.convert(source_audio, target_embedding)
```

The auto-tuning analyzes:
- **Dynamic range**: Wider range → higher temperature
- **Pitch variance**: More variance → higher temperature
- **Energy variance**: More variance → slightly higher temperature

## Pitch Shifting

Apply pitch shifting during voice conversion to transpose the output.

### Basic Pitch Shift

```python
# Shift up by 2 semitones (whole step)
audio = model.convert(
    source_audio,
    target_embedding,
    pitch_shift_semitones=2.0
)

# Shift down by 3 semitones (minor third)
audio = model.convert(
    source_audio,
    target_embedding,
    pitch_shift_semitones=-3.0
)
```

### Pitch Shift Methods

#### Linear Method (Default)

Simple multiplicative pitch shift. Fast and accurate for small shifts.

```python
audio = model.convert(
    source_audio,
    target_embedding,
    pitch_shift_semitones=2.0,
    pitch_shift_method='linear'
)
```

**Best for**: ±4 semitones or less

#### Formant-Preserving Method

Uses gentler curve (70% of shift) for more natural sound with large shifts.

```python
audio = model.convert(
    source_audio,
    target_embedding,
    pitch_shift_semitones=7.0,  # Perfect fifth
    pitch_shift_method='formant_preserving'
)
```

**Best for**: Large shifts (>4 semitones)

### Pitch Shift Examples

```python
# Male to female voice (approximate)
audio = model.convert(
    male_audio,
    female_embedding,
    pitch_shift_semitones=5.0,  # Shift up ~5 semitones
    pitch_shift_method='formant_preserving'
)

# Female to male voice (approximate)
audio = model.convert(
    female_audio,
    male_embedding,
    pitch_shift_semitones=-5.0,  # Shift down ~5 semitones
    pitch_shift_method='formant_preserving'
)

# Transpose song to different key
audio = model.convert(
    source_audio,
    target_embedding,
    pitch_shift_semitones=-2.0  # Down a whole step (e.g., C to Bb)
)
```

## Quality Presets

Control the quality/speed tradeoff with predefined presets.

### Available Presets

| Preset | Decoder Steps | Speed | Quality | Use Case |
|--------|--------------|-------|---------|----------|
| `draft` | 2 | 4.0x | 60% | Quick testing |
| `fast` | 4 | 2.0x | 80% | Real-time preview |
| `balanced` | 4 | 1.0x | 100% | Standard use (default) |
| `high` | 8 | 0.5x | 130% | High-quality output |
| `studio` | 16 | 0.25x | 150% | Studio production |

### Setting Presets

```python
# For quick testing
model.set_quality_preset('draft')
audio = model.convert(source_audio, target_embedding)

# For real-time applications
model.set_quality_preset('fast')

# For high-quality final output
model.set_quality_preset('high')

# For studio-grade production
model.set_quality_preset('studio')
```

### Querying Preset Information

```python
# Get current preset info
info = model.get_quality_preset_info()
print(f"Description: {info['description']}")
print(f"Decoder steps: {info['decoder_steps']}")
print(f"Relative quality: {info['relative_quality']}")
print(f"Relative speed: {info['relative_speed']}x")

# Get info for specific preset
studio_info = model.get_quality_preset_info('studio')
print(f"Studio quality: {studio_info['relative_quality']}")
```

### Estimating Conversion Time

```python
# Estimate time for 30-second audio
audio_duration = 30.0  # seconds

for preset in ['draft', 'fast', 'balanced', 'high', 'studio']:
    est_time = model.estimate_conversion_time(audio_duration, preset)
    print(f"{preset:>10}: {est_time:>6.2f} seconds")

# Output:
#      draft:   3.75 seconds  (4x faster than realtime)
#       fast:   7.50 seconds  (2x faster)
#   balanced:  15.00 seconds  (1x realtime)
#       high:  30.00 seconds  (0.5x)
#     studio:  60.00 seconds  (0.25x)
```

## Advanced Features

Optional features for enhanced audio processing.

### Denoise Input

Remove background noise from source audio before conversion.

```python
audio = model.convert(
    noisy_source_audio,
    target_embedding,
    denoise_input=True  # Enable input denoising
)
```

**How it works**: Uses spectral gate to identify and reduce noise floor based on quiet regions.

### Enhance Output

Apply subtle EQ and compression to enhance clarity and consistency.

```python
audio = model.convert(
    source_audio,
    target_embedding,
    enhance_output=True  # Enable output enhancement
)
```

**How it works**: Applies gentle high-frequency emphasis and RMS compression.

### Preserve Dynamics

Maintain the dynamic range (loudness variation) from the source audio.

```python
audio = model.convert(
    source_audio,
    target_embedding,
    preserve_dynamics=True  # Preserve source dynamics
)
```

**How it works**: Matches RMS level and peak from source, with soft limiting.

### Vibrato Transfer

Transfer vibrato characteristics from source to target (experimental).

```python
audio = model.convert(
    source_audio,
    target_embedding,
    vibrato_transfer=True  # Transfer vibrato
)
```

**Note**: Requires vibrato data from pitch extractor.

### Setting Global Defaults

You can set global defaults for these features:

```python
# Enable features globally
model.denoise_input = True
model.enhance_output = True
model.preserve_dynamics = True
model.vibrato_transfer = False

# All subsequent conversions will use these settings
audio = model.convert(source_audio, target_embedding)

# Override for specific conversion
audio = model.convert(
    source_audio,
    target_embedding,
    denoise_input=False  # Override global setting
)
```

## Complete Examples

### Example 1: High-Quality Voice Conversion

```python
import numpy as np
from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter

# Load model
config = load_config('config/model_config.yaml')
model = SingingVoiceConverter(config)
model.prepare_for_inference()

# Load audio and speaker embedding
source_audio = load_audio('source.wav')  # [T] samples
target_embedding = load_speaker_embedding('target_profile.npy')  # [256]

# Configure for high quality
model.set_quality_preset('high')
model.set_temperature(1.1)

# Convert with all enhancements
converted_audio = model.convert(
    source_audio,
    target_embedding,
    source_sample_rate=44100,
    output_sample_rate=44100,
    pitch_shift_semitones=0.0,
    denoise_input=True,
    enhance_output=True,
    preserve_dynamics=True
)

# Save result
save_audio('converted_high_quality.wav', converted_audio, 44100)
```

### Example 2: Real-Time Preview

```python
# Configure for speed
model.set_quality_preset('fast')
model.set_temperature(0.8)  # Stable for real-time

# Quick conversion
converted_audio = model.convert(
    source_audio,
    target_embedding,
    denoise_input=False,  # Skip for speed
    enhance_output=False
)
```

### Example 3: Gender Transformation

```python
# Male to female conversion
model.set_quality_preset('high')

# Auto-tune temperature
optimal_temp = model.auto_tune_temperature(male_audio, female_embedding)

# Convert with pitch shift
female_result = model.convert(
    male_audio,
    female_embedding,
    pitch_shift_semitones=5.0,  # Shift up
    pitch_shift_method='formant_preserving',
    enhance_output=True
)
```

### Example 4: Singing Voice Conversion

```python
# Configure for expressive singing
model.set_quality_preset('studio')
model.set_temperature(1.3)  # Higher for expressiveness

# Convert with preservation of dynamics
converted_singing = model.convert(
    singing_audio,
    target_singer_embedding,
    pitch_shift_semitones=2.0,  # Transpose to better key
    pitch_shift_method='linear',
    denoise_input=True,
    enhance_output=True,
    preserve_dynamics=True  # Important for singing
)
```

### Example 5: Batch Processing with Different Presets

```python
import time

presets = ['draft', 'balanced', 'high', 'studio']
results = {}

for preset in presets:
    model.set_quality_preset(preset)

    start_time = time.time()
    audio = model.convert(source_audio, target_embedding)
    elapsed = time.time() - start_time

    results[preset] = {
        'audio': audio,
        'time': elapsed,
        'info': model.get_quality_preset_info(preset)
    }

    print(f"{preset:>10}: {elapsed:.2f}s (estimated: "
          f"{model.estimate_conversion_time(len(source_audio)/44100, preset):.2f}s)")
```

## API Reference

### Temperature Methods

- `set_temperature(temperature: float)` - Set temperature (0.1-2.0)
- `auto_tune_temperature(source_audio, target_embedding, sample_rate)` - Auto-tune temperature

### Quality Preset Methods

- `set_quality_preset(preset_name: str)` - Set quality preset
- `get_quality_preset_info(preset_name: Optional[str])` - Get preset information
- `estimate_conversion_time(audio_duration: float, preset: Optional[str])` - Estimate processing time

### Convert Method (Enhanced)

```python
def convert(
    source_audio: Union[torch.Tensor, np.ndarray],
    target_speaker_embedding: Union[torch.Tensor, np.ndarray],
    source_f0: Optional[Union[torch.Tensor, np.ndarray]] = None,
    source_sample_rate: int = 16000,
    output_sample_rate: int = 44100,
    pitch_shift_semitones: float = 0.0,
    pitch_shift_method: str = 'linear',
    denoise_input: Optional[bool] = None,
    enhance_output: Optional[bool] = None,
    preserve_dynamics: Optional[bool] = None,
    vibrato_transfer: Optional[bool] = None
) -> np.ndarray
```

## Performance Considerations

### Speed vs Quality Tradeoffs

- **Draft preset**: 4x faster, suitable for rapid iteration
- **Fast preset**: 2x faster, good for real-time preview
- **Balanced preset**: Standard quality/speed ratio
- **High preset**: 2x slower, better quality
- **Studio preset**: 4x slower, maximum quality

### Memory Usage

All presets use similar memory. The difference is in computation time (decoder steps).

### GPU Acceleration

All features benefit from GPU acceleration. Quality presets scale linearly with GPU compute capability.

### Recommended Settings by Use Case

| Use Case | Preset | Temperature | Features |
|----------|--------|-------------|----------|
| Testing/Development | draft | 0.8 | None |
| Real-time Preview | fast | 0.8 | None |
| General Conversion | balanced | 1.0 | enhance_output |
| High-Quality Output | high | 1.1 | denoise_input, enhance_output |
| Studio Production | studio | 1.2 | All features enabled |
| Live Performance | fast | 0.7 | None (stability) |

## Troubleshooting

### Audio Artifacts

- Try **lower temperature** (0.7-0.9) for more stable output
- Use **formant_preserving** method for large pitch shifts
- Enable **denoise_input** if source is noisy

### Conversion Too Slow

- Use **draft** or **fast** preset for speed
- Disable advanced features (denoise, enhance)
- Process shorter audio segments

### Output Too Quiet/Loud

- Enable **preserve_dynamics** to match source levels
- Adjust RMS manually post-processing
- Check speaker embedding quality

### Pitch Shift Sounds Unnatural

- Use **formant_preserving** for shifts >4 semitones
- Keep shifts within ±12 semitones range
- Consider using smaller shifts with voice type selection

## Version History

- **v1.1.0** (Current): Added temperature API, pitch shifting, quality presets, advanced features
- **v1.0.0**: Initial SingingVoiceConverter implementation

## Future Enhancements

Planned features for future releases:
- Adaptive temperature scheduling during inference
- Custom quality preset creation
- Real-time vibrato synthesis
- Multi-band enhancement controls
- Formant shifting separate from pitch
- Style transfer parameters
