# Voice Conversion Preset System Implementation

## Overview

Implemented a comprehensive preset system for the `SingingConversionPipeline` that allows users to easily switch between different quality/speed tradeoffs without manually configuring individual components.

## Implementation Date

2025-10-27

## Files Modified

1. **Created**: `/home/kp/autovoice/config/voice_conversion_presets.yaml`
   - Defines 4 preset configurations: fast, balanced, quality, custom
   - Each preset configures all 4 pipeline components

2. **Modified**: `/home/kp/autovoice/src/auto_voice/inference/singing_conversion_pipeline.py`
   - Added `preset` parameter to `__init__()`
   - Enhanced `_load_config()` to load and merge preset configurations
   - Added `_load_preset_config()` helper method
   - Updated component initialization to use preset configs
   - Added `set_preset()` method for runtime preset switching
   - Added `get_current_preset()` method to query active preset

## Preset Configurations

### Fast Preset
- **Use case**: Quick previews, testing, low-end hardware
- **Expected speed**: ~2-3x real-time on GPU
- **Quality**: Acceptable
- **Key settings**:
  - VocalSeparator: `htdemucs`, 0 shifts, 0.15 overlap
  - PitchExtractor: `tiny` model, 20ms hop, `argmax` decoder
  - VoiceConverter: 2 flow layers, mean-only sampling
  - AudioMixer: Peak normalization, 5ms crossfade

### Balanced Preset (Default)
- **Use case**: General use, good balance
- **Expected speed**: ~1-1.5x real-time on GPU
- **Quality**: Good
- **Key settings**:
  - VocalSeparator: `htdemucs`, 1 shift, 0.25 overlap
  - PitchExtractor: `full` model, 10ms hop, `viterbi` decoder
  - VoiceConverter: 4 flow layers, full sampling
  - AudioMixer: RMS normalization, 10ms crossfade

### Quality Preset
- **Use case**: Final outputs, critical quality
- **Expected speed**: ~0.5-0.8x real-time on GPU
- **Quality**: High
- **Key settings**:
  - VocalSeparator: `htdemucs_ft`, 5 shifts, 0.35 overlap, FP32
  - PitchExtractor: `full` model, 5ms hop, minimal smoothing
  - VoiceConverter: 6 flow layers, full sampling
  - AudioMixer: LUFS normalization (-16dB), 20ms cosine crossfade

### Custom Preset
- **Use case**: Advanced users who want to tweak settings
- **Default settings**: Same as balanced
- **Purpose**: Template for user customization

## Usage Examples

### Basic Usage with Preset

```python
from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

# Initialize with a preset
pipeline = SingingConversionPipeline(preset='balanced')

# Convert a song
result = pipeline.convert_song(
    song_path='input.mp3',
    target_profile_id='user-profile-id'
)
```

### Switch Preset at Runtime

```python
# Start with fast preset for previews
pipeline = SingingConversionPipeline(preset='fast')

# Quick preview
preview = pipeline.convert_song('song.mp3', 'profile-id')

# Switch to quality preset for final output
pipeline.set_preset('quality')

# High-quality conversion
final = pipeline.convert_song('song.mp3', 'profile-id')
```

### Override Preset Settings

```python
# Start with balanced preset but customize
pipeline = SingingConversionPipeline(
    preset='balanced',
    config={
        'pitch_extractor': {
            'hop_length_ms': 5.0  # Override to higher resolution
        },
        'audio_mixer': {
            'crossfade_duration_ms': 15.0  # Override crossfade
        }
    }
)
```

### Query Current Preset

```python
pipeline = SingingConversionPipeline(preset='quality')

info = pipeline.get_current_preset()
print(f"Preset: {info['name']}")
print(f"Description: {info['description']}")
print(f"Config: {info['config']}")
```

## Configuration Priority

The configuration is loaded with the following priority (highest to lowest):

1. **Constructor config**: User-provided config dict in `__init__()`
2. **Preset config**: Settings from `voice_conversion_presets.yaml`
3. **YAML config**: Settings from `audio_config.yaml`
4. **Environment variables**: `AUTOVOICE_PIPELINE_*` env vars
5. **Defaults**: Hard-coded defaults in `_load_config()`

User config always takes precedence over preset config, allowing fine-grained customization.

## Component Configuration

Each preset configures all 4 pipeline components:

### VocalSeparator Settings
- `model`: Model name (`htdemucs`, `htdemucs_ft`, `mdx_extra`)
- `shifts`: Number of random shifts for quality
- `overlap`: Overlap between chunks (0.0-0.5)
- `split`: Enable memory-efficient splitting
- `mixed_precision`: Use FP16 for speed

### SingingPitchExtractor Settings
- `model`: CREPE model (`tiny`, `full`)
- `hop_length_ms`: Time resolution (5-20ms)
- `batch_size`: GPU batch size
- `decoder`: Decoder type (`argmax`, `viterbi`, `weighted_argmax`)
- `median_filter_width`: Median smoothing width
- `mean_filter_width`: Mean smoothing width
- `confidence_threshold`: Voiced/unvoiced threshold

### SingingVoiceConverter Settings
- `flow_decoder.num_flows`: Number of flow layers (2-6)
- `flow_decoder.num_layers`: Coupling layers per flow (2-6)
- `flow_decoder.use_only_mean`: Mean-only vs full sampling
- `inference.temperature`: Sampling temperature

### AudioMixer Settings
- `normalization_method`: `peak`, `rms`, or `lufs`
- `crossfade_duration_ms`: Crossfade duration (5-20ms)
- `fade_curve`: `linear` or `cosine`
- `prevent_clipping`: Enable peak limiting
- `target_lufs_db`: Target loudness for LUFS normalization

## Cache Management

When switching presets using `set_preset()`, the cache is automatically cleared by default since different presets produce different results. This can be disabled:

```python
# Switch preset without clearing cache (not recommended)
pipeline.set_preset('quality', clear_cache=False)
```

## Environment Variable Override

Users can set the default preset via environment variable:

```bash
export AUTOVOICE_PIPELINE_PRESET=quality
python app.py  # Will use quality preset by default
```

## Error Handling

- **Invalid preset name**: Falls back to default preset (`balanced`) with warning
- **Missing preset file**: Uses component defaults with warning
- **Malformed YAML**: Uses component defaults with warning

## Technical Details

### Thread Safety
- Both `set_preset()` and preset loading are thread-safe
- Uses existing `threading.RLock` for synchronization

### Component Reinitialization
- `set_preset()` fully reinitializes all 4 components
- Previous component instances are discarded (garbage collected)
- Model weights are reloaded if necessary

### YAML Format
The preset file uses this structure:

```yaml
presets:
  preset_name:
    description: "Preset description"
    vocal_separator:
      # VocalSeparator config
    pitch_extractor:
      # SingingPitchExtractor config
    voice_converter:
      # SingingVoiceConverter config
    audio_mixer:
      # AudioMixer config

default_preset: 'balanced'
```

## Future Enhancements

Potential improvements for future versions:

1. **Preset validation**: Validate preset configs against component schemas
2. **Custom presets**: Allow users to save custom presets to file
3. **Preset metrics**: Track performance metrics per preset
4. **Adaptive preset selection**: Auto-select preset based on hardware
5. **Preset interpolation**: Blend between presets for fine-tuning
6. **GUI preset selector**: Add preset dropdown in web interface

## Testing Recommendations

1. Test each preset with various audio inputs
2. Verify quality/speed tradeoffs match expectations
3. Test preset switching during runtime
4. Verify cache clearing behavior
5. Test custom config overrides
6. Test invalid preset handling
7. Verify thread safety with concurrent operations

## Related Files

- `/home/kp/autovoice/config/voice_conversion_presets.yaml` - Preset definitions
- `/home/kp/autovoice/src/auto_voice/inference/singing_conversion_pipeline.py` - Pipeline implementation
- `/home/kp/autovoice/config/audio_config.yaml` - Global audio config
- `/home/kp/autovoice/src/auto_voice/audio/source_separator.py` - VocalSeparator component
- `/home/kp/autovoice/src/auto_voice/audio/pitch_extractor.py` - SingingPitchExtractor component
- `/home/kp/autovoice/src/auto_voice/models/singing_voice_converter.py` - SingingVoiceConverter component
- `/home/kp/autovoice/src/auto_voice/audio/mixer.py` - AudioMixer component
