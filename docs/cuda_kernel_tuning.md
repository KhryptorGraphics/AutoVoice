# CUDA Kernel Tuning Constants

This document describes the hardcoded constants in CUDA kernels and how they align with the configuration in `config/audio_config.yaml`.

## Pitch Detection Kernel (`audio_kernels.cu`)

### Hardcoded Constants
- **Frame Length**: 2048 samples (line 19, 361, 410)
- **Block Size**: 256 threads (PITCH_DETECTION_BLOCK_SIZE in kernel_utils.cuh)
- **Shared Memory**: 2572 floats = frame_length (2048) + prefix (534) + history (20)

### Configuration Alignment
These constants align with the configuration in `config/audio_config.yaml`:

```yaml
cuda_kernels:
  pitch_detection:
    block_size: 256          # Threads per block
    frame_length: 2048       # Analysis window size (samples)
    hop_length: 512          # Hop size between frames (samples)
    fmin: 80.0               # Minimum frequency (Hz)
    fmax: 1000.0             # Maximum frequency (Hz)
    threshold: 0.15          # CMND threshold
    shared_mem_size: 2572    # frame_length + prefix(534) + history(20)
```

### Usage Notes
1. **Frame Length (2048)**: This is fixed in the kernel for optimal GPU performance. Changing this would require kernel recompilation.
2. **Hop Length (512)**: This is passed as a parameter, so it can be configured at runtime.
3. **Block Size (256)**: Optimal for most GPUs, provides good occupancy.
4. **Shared Memory**: Automatically sized based on frame_length constant.

### Setting Parameters from Python
While the frame length is hardcoded in the kernel, other parameters can be set from Python via the config:

```python
from auto_voice.audio.pitch_extractor import SingingPitchExtractor

# Override config parameters
config = {
    'fmin': 100.0,           # Override minimum pitch
    'fmax': 800.0,           # Override maximum pitch
    'hop_length_ms': 11.6,   # Override hop length (converted to samples)
}

extractor = SingingPitchExtractor(config=config)
```

### Frame Length Implications
The frame length of 2048 samples provides:
- **At 44.1 kHz**: ~46.4 ms window (good for singing voice)
- **At 22.05 kHz**: ~92.8 ms window (longer than ideal for fast passages)

If you need different frame lengths, you must:
1. Modify `frame_length` constant in `audio_kernels.cu` line 19
2. Update `shared_mem_size` calculation in kernel launch
3. Recompile the CUDA extension

## Mel-Spectrogram Kernel

### Hardcoded Constants
- **FFT Size**: 2048 (configurable via parameter)
- **Mel Bins**: 128 (configurable via parameter)
- **Sample Rate**: 44100 Hz (configurable via parameter)

### Configuration Alignment
```yaml
cuda_kernels:
  mel_spectrogram_singing:
    n_fft: 2048
    mel_bins: 128
    sample_rate: 44100
```

These are **configurable at runtime** and not hardcoded in the kernel.

## Formant Extraction Kernel

### Hardcoded Constants
- **LPC Order Range**: 8-20 (validated at runtime)
- **Max Formants**: 5 (validated at runtime)
- **Frame Length**: Configurable parameter

### Configuration Alignment
```yaml
cuda_kernels:
  formant_extraction:
    frame_length: 2048
    lpc_order: 14
    num_formants: 4
```

All formant parameters are **configurable at runtime**.

## Vibrato Analysis Kernel

### Hardcoded Constants
- **Window Size**: 20 frames (~200ms at typical hop lengths)
- **Rate Range**: 4-8 Hz (typical vibrato range)

### Configuration Alignment
```yaml
cuda_kernels:
  vibrato_analysis:
    window_size: 20
    min_rate_hz: 4.0
    max_rate_hz: 8.0
    min_depth_cents: 20.0
```

The window size is hardcoded (line 313), but rate parameters are configurable.

## Recommendations

### For Production Use
The current hardcoded values (especially frame_length=2048 for pitch detection) are well-suited for:
- Singing voice analysis at 44.1 kHz
- Real-time pitch tracking
- Vibrato detection

### For Customization
If you need different frame lengths or kernel parameters:

1. **Environment Variables**: Override config via environment variables:
   ```bash
   export AUTOVOICE_PITCH_FMIN=100.0
   export AUTOVOICE_PITCH_FMAX=800.0
   ```

2. **Config File**: Edit `config/audio_config.yaml` for persistent changes

3. **Kernel Recompilation**: For frame length changes, modify `audio_kernels.cu` and rebuild:
   ```bash
   python setup.py build_ext --inplace
   ```

## Future Enhancement Opportunity

A potential enhancement would be to add a host-side setter in `bindings.cpp` to accept tuning parameters from Python and pass them to kernel launches. This would allow runtime configuration of currently hardcoded values without recompilation.

Example implementation:
```cpp
void set_kernel_tuning(int frame_length, int block_size) {
    // Store in global variables
    // Use in kernel launches
}
```

This would require careful coordination with shared memory allocation and kernel compilation.
