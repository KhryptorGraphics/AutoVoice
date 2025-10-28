# SingingPitchExtractor Production Features Implementation

## Overview

Added four production-ready features to the `SingingPitchExtractor` class in `/home/kp/autovoice/src/auto_voice/audio/pitch_extractor.py` for enhanced singing voice analysis and real-time processing.

## Implementation Date
October 27, 2025

## Features Implemented

### 1. Vibrato Classification

**Method**: `classify_vibrato(f0_data) -> Dict`

**Description**: Performs detailed vibrato classification using frequency modulation analysis.

**Returns**:
```python
{
    'vibrato_detected': bool,      # Whether vibrato is present
    'rate_hz': float,              # Average vibrato rate (0.0 if not detected)
    'extent_cents': float,         # Vibrato depth in cents (0.0 if not detected)
    'regularity_score': float,     # Regularity 0.0-1.0 (0.0 if not detected)
    'segments': List[Tuple]        # List of (start_time, end_time, rate, depth)
}
```

**Features**:
- Frequency modulation analysis using autocorrelation
- Hilbert transform for accurate envelope detection
- Regularity scoring based on autocorrelation decay
- Handles silence and noise robustly
- Thread-safe operation

**Example**:
```python
f0_data = extractor.extract_f0_contour('singing.wav')
vibrato = extractor.classify_vibrato(f0_data)
if vibrato['vibrato_detected']:
    print(f"Vibrato: {vibrato['rate_hz']:.2f} Hz, {vibrato['extent_cents']:.1f} cents")
    print(f"Regularity: {vibrato['regularity_score']:.2f}")
```

### 2. Pitch Correction Suggestions

**Method**: `suggest_pitch_corrections(f0_data, reference_scale='C', tolerance_cents=50.0) -> List[Dict]`

**Description**: Analyzes F0 contour and identifies notes deviating from reference musical scale.

**Arguments**:
- `f0_data`: F0 data dictionary from `extract_f0_contour()`
- `reference_scale`: Reference scale name ('C', 'D', 'E', etc.)
- `tolerance_cents`: Tolerance in cents (default: 50.0)

**Returns**: List of correction dictionaries with:
```python
{
    'timestamp': float,           # Time in seconds
    'detected_f0_hz': float,     # Detected frequency
    'detected_note': str,         # Detected note (e.g., 'C4', 'F#5')
    'target_note': str,           # Target note in scale
    'target_f0_hz': float,       # Target frequency
    'correction_cents': float     # Correction amount (negative = flatten)
}
```

**Helper Methods**:
- `_f0_to_note_name(f0_hz)`: Convert frequency to note name with cents offset
- `_get_scale_notes(reference_scale)`: Get notes in major scale for key
- `_find_nearest_scale_note(f0_hz, scale_notes)`: Find nearest scale note
- `_note_name_to_f0(note_name)`: Convert note name to frequency

**Features**:
- Major scale analysis with all octaves (0-8)
- Configurable tolerance threshold
- Handles sharp notes (e.g., 'F#5')
- Only analyzes voiced frames
- Thread-safe operation

**Example**:
```python
f0_data = extractor.extract_f0_contour('singing.wav')
corrections = extractor.suggest_pitch_corrections(f0_data, 'C', 50.0)
for corr in corrections[:5]:
    print(f"{corr['timestamp']:.2f}s: {corr['detected_note']} -> {corr['target_note']} "
          f"({corr['correction_cents']:+.1f} cents)")
```

### 3. Enhanced Real-time Streaming

**Modified Method**: `extract_f0_realtime(audio_chunk, sample_rate=None, state=None, use_cuda_kernel=True) -> Tensor`

**New State Management Methods**:
- `create_realtime_state()`: Initialize state dictionary
- `reset_realtime_state(state)`: Reset state to initial values

**State Dictionary Structure**:
```python
{
    'overlap_buffer': Tensor,      # Audio buffer for frame overlap
    'smoothing_history': List,     # Pitch history for temporal smoothing
    'frame_count': int,            # Frame counter
    'last_pitch': float            # Last valid pitch value
}
```

**Features**:
- Overlap buffering (25% overlap) for seamless transitions
- Temporal smoothing using median filter
- Stateful processing for streaming applications
- Compatible with both CUDA kernel and torchcrepe
- Thread-safe state management

**Helper Method**:
- `_apply_temporal_smoothing(pitch, state)`: Apply temporal smoothing to pitch output

**Example**:
```python
# Stateful processing (recommended for streaming)
extractor = SingingPitchExtractor()
state = extractor.create_realtime_state()

for chunk in audio_stream:
    f0 = extractor.extract_f0_realtime(chunk, sample_rate=22050, state=state)
    # Process f0 with smooth transitions...
```

### 4. Configuration Updates

**File**: `/home/kp/autovoice/config/audio_config.yaml`

**New Configuration Parameters**:

```yaml
singing_pitch:
  # Vibrato classification
  vibrato_regularity_threshold: 0.5  # Autocorrelation threshold for regularity (0.0-1.0)

  # Pitch correction parameters
  pitch_correction_tolerance_cents: 50.0  # Tolerance for pitch correction suggestions
  pitch_correction_reference_scale: 'C'   # Default reference scale (C, D, E, F, G, A, B)

  # Real-time streaming parameters
  realtime_overlap_frames: 5       # Number of frames to overlap between chunks
  realtime_buffer_size: 4096       # Size of overlap buffer in samples
  realtime_smoothing_window: 5     # Temporal smoothing window size (frames)
```

## Edge Case Handling

All methods implement comprehensive edge case handling:

### Silence
- Vibrato classification: Returns zeros for all metrics
- Pitch correction: Skips unvoiced regions
- Real-time streaming: Returns zero pitch values

### Noise
- Vibrato classification: Filters using confidence threshold
- Pitch correction: Only processes voiced frames
- Real-time streaming: Temporal smoothing reduces noise

### Rapid Pitch Changes
- Vibrato classification: Bandpass filtering isolates vibrato range
- Pitch correction: Processes each frame individually
- Real-time streaming: Overlap buffering ensures continuity

## Thread Safety

All methods use `self.lock` (threading.RLock) for thread-safe operation:
- Vibrato classification: Thread-safe
- Pitch correction: Thread-safe
- Real-time state management: Thread-safe
- Temporal smoothing: Thread-safe

## Performance Optimizations

### Efficient Array Operations
- Uses numpy for all numerical computations
- Leverages scipy for Hilbert transform and convolution
- Torch tensors for GPU acceleration

### Algorithmic Optimizations
- FFT-based bandpass filtering
- Autocorrelation for vibrato detection
- Median filtering for smoothing (robust to outliers)

### Memory Management
- Bounded smoothing history (configurable window size)
- Fixed-size overlap buffer (configurable size)
- Efficient tensor operations

## Configuration Priority

Configuration loading follows this priority (highest to lowest):
1. Constructor config parameter
2. YAML configuration file
3. Environment variables
4. Default values

## Documentation Standards

All methods include:
- Comprehensive docstrings with Args, Returns, Example, and Note sections
- Type hints for all parameters and return values
- Edge case documentation
- Thread-safety guarantees

## Testing Recommendations

### Unit Tests
1. Test `classify_vibrato()` with:
   - Synthetic vibrato signal (known rate and depth)
   - Non-vibrato signal
   - Silent audio
   - Noisy audio

2. Test `suggest_pitch_corrections()` with:
   - Perfect scale singing
   - Out-of-tune singing
   - Silent audio
   - Rapid pitch changes

3. Test real-time streaming with:
   - Continuous audio stream
   - State reset between streams
   - Stateless processing
   - CUDA kernel and torchcrepe fallback

### Integration Tests
- Test full pipeline: audio → F0 extraction → vibrato classification → pitch correction
- Test configuration loading from YAML
- Test GPU vs CPU processing
- Test thread-safety with concurrent calls

## Dependencies

Required packages:
- `numpy`: Array operations
- `torch`: Tensor operations and GPU support
- `torchcrepe`: Pitch extraction
- `scipy`: Signal processing (Hilbert transform, convolution)

Optional packages:
- `cuda_kernels`: CUDA acceleration for real-time processing

## API Stability

All public methods maintain backward compatibility:
- `extract_f0_realtime()`: Optional `state` parameter (backward compatible)
- `classify_vibrato()`: New method (no breaking changes)
- `suggest_pitch_corrections()`: New method (no breaking changes)
- `create_realtime_state()`: New method (no breaking changes)
- `reset_realtime_state()`: New method (no breaking changes)

## Usage Examples

### Complete Workflow
```python
from auto_voice.audio.pitch_extractor import SingingPitchExtractor

# Initialize extractor
extractor = SingingPitchExtractor(device='cuda')

# Extract F0 contour
f0_data = extractor.extract_f0_contour('singing.wav')

# Classify vibrato
vibrato = extractor.classify_vibrato(f0_data)
print(f"Vibrato detected: {vibrato['vibrato_detected']}")
print(f"Rate: {vibrato['rate_hz']:.2f} Hz")
print(f"Extent: {vibrato['extent_cents']:.1f} cents")
print(f"Regularity: {vibrato['regularity_score']:.2f}")

# Suggest pitch corrections
corrections = extractor.suggest_pitch_corrections(f0_data, reference_scale='C', tolerance_cents=50.0)
print(f"\nPitch corrections needed: {len(corrections)}")
for corr in corrections[:10]:
    print(f"  {corr['timestamp']:.2f}s: {corr['detected_note']} → {corr['target_note']} "
          f"({corr['correction_cents']:+.1f} cents)")

# Real-time streaming
state = extractor.create_realtime_state()
for chunk in audio_stream:
    f0 = extractor.extract_f0_realtime(chunk, sample_rate=22050, state=state)
    # Process f0...
```

### Custom Configuration
```python
# Custom configuration
config = {
    'vibrato_regularity_threshold': 0.6,
    'pitch_correction_tolerance_cents': 30.0,
    'pitch_correction_reference_scale': 'G',
    'realtime_smoothing_window': 7
}

extractor = SingingPitchExtractor(config=config)
```

## Summary

This implementation adds production-ready features for:
1. **Vibrato Analysis**: Comprehensive vibrato detection and characterization
2. **Pitch Correction**: Musical scale-based pitch correction suggestions
3. **Real-time Processing**: Enhanced streaming with overlap and smoothing
4. **Configuration**: Flexible configuration system with YAML support

All features are:
- ✅ Well-documented with comprehensive docstrings
- ✅ Thread-safe for concurrent use
- ✅ Robust to edge cases (silence, noise, rapid changes)
- ✅ Efficient with numpy/torch optimizations
- ✅ Backward compatible with existing API
- ✅ Configurable via YAML and constructor

## Files Modified

1. `/home/kp/autovoice/src/auto_voice/audio/pitch_extractor.py`
   - Added 5 new public methods
   - Added 5 new helper methods
   - Updated configuration loading
   - Enhanced real-time processing

2. `/home/kp/autovoice/config/audio_config.yaml`
   - Added vibrato_regularity_threshold
   - Added pitch_correction_tolerance_cents
   - Added pitch_correction_reference_scale
   - Added realtime_overlap_frames
   - Added realtime_buffer_size
   - Added realtime_smoothing_window

## Lines of Code Added

- New methods: ~500 lines
- Helper methods: ~150 lines
- Configuration: ~20 lines
- Documentation: Comprehensive docstrings

Total: ~670 lines of production-ready code with full documentation.
