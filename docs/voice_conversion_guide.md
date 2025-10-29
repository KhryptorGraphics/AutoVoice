# Voice Conversion User Guide

A comprehensive guide for using AutoVoice's singing voice conversion features.

## 1. Introduction

### What is Singing Voice Conversion?

Singing voice conversion allows you to transform any song into your voice while preserving the original pitch, timing, and musical expression. Unlike simple pitch shifting or autotune, AutoVoice uses advanced AI to maintain the melody and emotional delivery while changing the voice timbre to match your target speaker.

### How AutoVoice Makes Anyone Sound Like a Professional Singer

AutoVoice analyzes the original singer's pitch contour and vocal characteristics, then applies your voice's unique timbre while preserving the musical performance. This means you don't need to sing the song yourself – the system extracts the pitch from the original recording and applies your voice characteristics.

### Use Cases

- **Cover Songs**: Create cover versions in your voice without recording yourself singing
- **Vocal Practice**: Hear how songs would sound in your voice to guide practice
- **Content Creation**: Produce vocal content for videos, podcasts, music
- **Accessibility**: Enable people with limited singing ability to express themselves musically

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support recommended (10-50x faster than CPU)
- **Audio Files**: Songs in MP3, WAV, FLAC, or OGG format
- **Voice Sample**: 30-60 seconds of your voice for profile creation
- **Time**: 5-10 minutes for first-time setup, 15-60 seconds per song conversion

## 2. Getting Started

### Quick Start Workflow

1. **Create voice profile** from your voice sample (30-60 seconds)
2. **Upload song** you want to convert to your voice
3. **Convert song** with selected parameters
4. **Download and share** your singing

### Prerequisites

- Microphone or audio recording device (for voice sample)
- Song file in supported format (MP3, WAV, FLAC, OGG)
- 5-10 minutes for first-time setup
- Internet connection (if using web UI)

## 3. Voice Cloning (Creating Your Profile)

### Step-by-Step Guide

#### Step 1: Record Your Voice Sample

**Duration:** 30-60 seconds (longer is better, 45-60s is the sweet spot)

**Content Options:**
- Read text naturally (news article, book passage)
- Sing scales or vocal exercises
- Speak conversationally about a topic you're passionate about

**Quality Tips:**
- **Environment**: Use a quiet space with low background noise
- **Microphone**: Use the best microphone available (built-in mic is acceptable)
- **Volume**: Speak/sing clearly at normal volume (avoid shouting or whispering)
- **Clipping**: Avoid audio distortion from recording too loud
- **Consistency**: Maintain consistent distance from microphone

**Best Practice**: Record multiple samples (2-5) for better quality through averaging

#### Step 2: Upload Voice Sample

**Via Web UI:**
1. Navigate to "Voice Cloning" tab
2. Click "Upload Voice Sample"
3. Select your audio file
4. Click "Create Profile"

**Via API:**
```bash
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@my_voice.wav" \
  -F "user_id=my_user_id"
```

**Via Python:**
```python
from auto_voice.inference import VoiceCloner

cloner = VoiceCloner(device='cuda')
profile = cloner.create_voice_profile(
    audio='my_voice.wav',
    user_id='my_user_id'
)
print(f"Profile created: {profile['profile_id']}")
```

#### Step 3: Review Profile

After creation, review your profile details:
- **Vocal Range**: Displayed in Hz and musical note names (e.g., C3-G4)
- **Audio Duration**: Verify sample was long enough (30+ seconds)
- **Profile ID**: Save this UUID for later use
- **Creation Date**: When profile was created

**Optional**: Create multiple profiles for different vocal styles (chest voice, head voice, speaking voice)

### Troubleshooting Voice Cloning

**Error: "Audio too short"**
- **Cause**: Recording is less than 5 seconds
- **Solution**: Record at least 30 seconds (45-60s recommended)

**Error: "Low quality audio"**
- **Cause**: High background noise, poor recording quality
- **Solution**: Re-record in quieter environment, use better microphone

**Warning: "Low SNR"**
- **Cause**: Signal-to-noise ratio is low (noisy recording)
- **Solution**: Re-record with less background noise, closer to microphone

**Poor conversion results**
- **Cause**: Voice profile quality insufficient
- **Solution**: Record multiple samples and average them, improve recording setup

## 4. Song Conversion

### Step-by-Step Guide

#### Step 1: Select Voice Profile

1. Choose from your created voice profiles
2. View profile details:
   - Vocal range (min/max pitch in Hz and notes)
   - Creation date
   - Sample duration
3. Ensure profile vocal range matches song (or plan to use pitch shift)

#### Step 2: Upload Song

**Supported Formats:** MP3, WAV, FLAC, OGG

**File Requirements:**
- Maximum size: 100MB
- Recommended duration: 30 seconds to 5 minutes
- Longer songs take proportionally more time to process

**Quality Recommendations:**
- Use lossless formats (WAV, FLAC) for best results
- MP3 320kbps is acceptable
- Ensure vocals are clear and prominent in the mix

**Preview**: Play song before conversion to verify quality

#### Step 3: Configure Conversion

**Basic Settings:**

**Vocal Volume** (0.0-2.0, default: 1.0)
- Controls loudness of converted vocals in final mix
- 1.0 = original volume
- >1.0 = louder vocals
- <1.0 = quieter vocals

**Instrumental Volume** (0.0-2.0, default: 0.9)
- Controls loudness of background music
- Typically set slightly lower than vocals for clarity

**Quality Preset:**
- **Fast**: ~0.5x real-time, good for quick previews
- **Balanced**: ~1x real-time, recommended for general use (DEFAULT)
- **Quality**: ~2x real-time, best results for final versions

**Advanced Options (Optional):**

**Pitch Shift** (±12 semitones)
- Transpose song to match your vocal range
- Example: Shift -2 to lower song by 2 semitones
- Useful when original key is outside your comfortable range

**Temperature** (0.5-2.0, default: 1.0)
- Controls expressiveness and variation
- <1.0 = more stable, less variation
- >1.0 = more expressive, more variation

**Return Stems** (checkbox)
- Get separated vocals and instrumental as additional outputs
- Useful for remixing or further processing

#### Step 4: Start Conversion

1. Click "Convert Song" button
2. Monitor progress through 4 stages:
   - **Stage 1 (0-25%)**: Separating vocals from instrumental
   - **Stage 2 (25-40%)**: Extracting pitch contour
   - **Stage 3 (40-80%)**: Converting voice
   - **Stage 4 (80-100%)**: Mixing audio

**Estimated Time (30-second song):**
- Fast preset: ~15 seconds
- Balanced preset: ~30 seconds
- Quality preset: ~60 seconds
- Times scale with song length

#### Step 5: Review Results

**Playback:**
- Play converted song directly in browser
- Compare with original

**Quality Check:**
- Does it sound like you singing?
- Is pitch accurate?
- Are there any artifacts or distortions?

**Download:**
- Download converted song as WAV file
- If "return stems" enabled: Download vocals and instrumental separately

**Metadata:**
- View F0 statistics (pitch range, mean pitch)
- Check processing time
- Review quality metrics if available

### Troubleshooting Song Conversion

**Poor Pitch Accuracy**
- **Symptoms**: Converted voice sounds off-key
- **Causes**: Source vocals unclear, separation quality poor
- **Solutions**:
  - Use higher quality source songs (lossless formats)
  - Try different separation model
  - Use quality preset instead of fast
  - Verify source song has clear, prominent vocals

**Voice Doesn't Sound Like Target**
- **Symptoms**: Low speaker similarity score
- **Causes**: Poor voice profile quality, insufficient training
- **Solutions**:
  - Re-record voice profile with better quality
  - Use multi-sample profile creation (2-5 samples)
  - Ensure voice sample includes vocal variety
  - Check profile SNR is adequate (>15 dB)

**Robotic or Artificial Sound**
- **Symptoms**: Conversion sounds synthetic or processed
- **Causes**: Low temperature, poor source quality, model limitations
- **Solutions**:
  - Increase temperature parameter (1.2-1.5)
  - Use quality preset for better vocoder quality
  - Improve source audio quality
  - Try different voice profile

**Conversion Too Slow**
- **Symptoms**: Processing takes >2x song duration
- **Causes**: CPU fallback, GPU memory issues, inefficient settings
- **Solutions**:
  - Verify GPU is being used (check device in logs)
  - Use fast preset for quicker results
  - Reduce song length or split into chunks
  - Close other GPU applications
  - Enable TensorRT optimization (see Performance Optimization section below)

**Out of Memory Errors**
- **Symptoms**: Conversion fails with CUDA OOM error
- **Causes**: Song too long, GPU memory insufficient
- **Solutions**:
  - Reduce song length (<5 minutes recommended)
  - Use fast preset (lower memory usage)
  - Clear GPU cache and retry
  - Use CPU fallback for very long songs
  - Restart service to clear fragmented memory

## 4.5. Performance Optimization with TensorRT

**NEW**: AutoVoice supports TensorRT optimization for real-time performance (<5s per 30s audio). TensorRT provides 2-3x speedup through FP16 precision and optimized CUDA kernels.

### What is TensorRT?

TensorRT is NVIDIA's high-performance deep learning inference optimizer that provides:
- **2-3x speedup** through FP16 precision
- **Lower latency** for real-time applications
- **Reduced memory usage** via optimized kernels
- **Automatic optimization** of model architecture

### System Requirements

**Required:**
- NVIDIA GPU with Tensor Cores (RTX 2060 or newer recommended)
- CUDA 11.8 or later
- TensorRT 8.5 or later
- 4GB+ GPU memory

**Supported GPUs:**
- RTX 2060, 2070, 2080, 3060, 3070, 3080, 3090, 4060, 4070, 4080, 4090
- A100, A6000, A5000 (datacenter)
- Titan RTX, Quadro RTX series

**Exact Conditions for TensorRT Fast Path:**

To ensure TensorRT acceleration is actually used (not just requested), the following conditions must be met:

1. **Hardware**: NVIDIA GPU with Tensor Cores (compute capability 7.0+)
   - Verify: `nvidia-smi` shows RTX 2060 or newer
   - Check compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`

2. **Software Stack**:
   - CUDA 11.8+ installed and on PATH
   - TensorRT 8.5+ installed: `pip install tensorrt`
   - PyTorch with CUDA support: `torch.cuda.is_available()` returns `True`

3. **First-Time Setup**:
   - First run will export models to ONNX (stored in `~/.cache/autovoice/onnx_models/`)
   - TensorRT engines will be built (stored in `~/.cache/autovoice/tensorrt_engines/`)
   - This compilation takes 2-5 minutes but only happens once
   - Subsequent runs use cached engines (fast startup)

4. **Verification**:
   - Check result metadata: `result['metadata']['tensorrt']['enabled']` should be `True`
   - Check logs for "Using TensorRT-accelerated conversion"
   - If `enabled` is `False`, check logs for fallback reason

**Reproducibility Test:**

The TensorRT fast path is validated by the automated test suite:

```bash
# Run TensorRT latency test (requires TensorRT installed)
pytest tests/test_system_validation.py::TestSystemValidation::test_latency_target_30s_input -v
```

This test:
- Creates pipeline with `use_tensorrt=True` and `tensorrt_precision='fp16'`
- Converts 30 seconds of audio
- Asserts `result['metadata']['tensorrt']['enabled'] is True`
- Asserts latency < 5 seconds
- Saves metrics to `validation_results/latency_tensorrt.json`

If the test passes, TensorRT is correctly configured and operational.

### Enabling TensorRT

**Via Python API:**
```python
from auto_voice.inference import SingingConversionPipeline

# Enable TensorRT with FP16 precision
pipeline = SingingConversionPipeline(
    preset='fast',
    use_tensorrt=True,
    tensorrt_precision='fp16',  # or 'fp32' for higher precision
    device='cuda'
)

# Convert song with TensorRT optimization
result = pipeline.convert_song(
    song_path='input.wav',
    target_profile_id='profile_uuid'
)

# Verify TensorRT is actually being used
trt_info = result['metadata']['tensorrt']
print(f"TensorRT enabled: {trt_info['enabled']}")
print(f"TensorRT precision: {trt_info['precision']}")

# If enabled is False, check logs for fallback reason
if not trt_info['enabled']:
    print("WARNING: TensorRT not active, using PyTorch fallback")
```

**Via Web UI:**
1. Navigate to Settings → Performance
2. Enable "TensorRT Optimization"
3. Select precision: FP16 (recommended) or FP32
4. Click "Apply and Restart"
5. After conversion, check result metadata to verify TensorRT was used

**Via Configuration File:**
```yaml
# config/model_config.yaml
inference:
  use_tensorrt: true
  tensorrt_precision: 'fp16'
  tensorrt_workspace_size: 4096  # MB
```

### Performance Benchmarks

**Typical speedup (30 seconds of audio):**

| Configuration | Latency | Speedup |
|--------------|---------|---------|
| CPU baseline | 120s | 1.0x |
| GPU (PyTorch) | 8s | 15x |
| GPU + TensorRT FP16 | 3-4s | 30-40x |

**Real-Time Factor (RTF):**
- CPU: ~4.0x (4 seconds to process 1 second audio)
- GPU: ~0.27x (real-time capable)
- GPU + TensorRT: ~0.13x (7x faster than real-time)

### TensorRT Precision Options

**FP16 (Recommended):**
- **Pros**: 2-3x speedup, lower memory, minimal quality loss
- **Cons**: Requires Tensor Cores
- **Use case**: Real-time applications, production deployment

**FP32 (High Precision):**
- **Pros**: Maximum quality, identical to PyTorch
- **Cons**: Slower than FP16, higher memory usage
- **Use case**: Quality-critical applications, research

### First-Time Compilation

TensorRT performs model compilation on first run:
- **Compilation time**: 2-5 minutes (one-time per model)
- **Cache location**: `~/.cache/autovoice/tensorrt/`
- **Recompilation triggers**: Model changes, precision changes, CUDA version updates

**Progress monitoring:**
```python
pipeline = SingingConversionPipeline(
    use_tensorrt=True,
    tensorrt_verbose=True  # Show compilation progress
)
```

### Troubleshooting TensorRT

**TensorRT not available:**
- Install TensorRT: `pip install tensorrt`
- Verify CUDA version compatibility
- Check GPU supports Tensor Cores

**Compilation fails:**
- Verify sufficient disk space (2GB+ needed)
- Check CUDA and TensorRT versions match
- Review logs: `logs/tensorrt_compilation.log`

**Quality degradation with FP16:**
- Most users won't notice quality difference
- Switch to FP32 if quality is critical
- Verify model quantization range is appropriate

**Out of memory during compilation:**
- Reduce `tensorrt_workspace_size` in config
- Close other GPU applications
- Use FP32 if FP16 causes issues

### Advanced Configuration

**Custom workspace size:**
```python
pipeline = SingingConversionPipeline(
    use_tensorrt=True,
    tensorrt_workspace_size=2048,  # MB, reduce if OOM
    tensorrt_max_batch_size=4
)
```

**Disable TensorRT for specific components:**
```python
# Use TensorRT only for vocoder (most compute-intensive)
pipeline = SingingConversionPipeline(
    tensorrt_components=['vocoder'],  # ['encoder', 'decoder', 'vocoder']
    tensorrt_precision='fp16'
)
```

### Validation and Testing

**Verify TensorRT is Active:**
```python
# Check if TensorRT is enabled and loaded
pipeline = SingingConversionPipeline(
    use_tensorrt=True,
    tensorrt_precision='fp16'
)

# Verify TensorRT status
if pipeline.voice_converter.trt_enabled:
    print("✓ TensorRT engines loaded successfully")
    print(f"  Precision: {pipeline.tensorrt_precision}")
    print(f"  Components: {list(pipeline.voice_converter.tensorrt_models.keys())}")
else:
    print("✗ TensorRT not active (fallback to PyTorch)")
```

**Run Validation Test:**
```bash
# Test TensorRT pipeline end-to-end
pytest tests/test_tensorrt_conversion.py::test_tensorrt_pipeline_validation -v

# Skip test if TensorRT not available
pytest tests/test_tensorrt_conversion.py -m "not tensorrt" -v
```

### Reference Implementation

See these files for TensorRT implementation details:
- `src/auto_voice/inference/singing_conversion_pipeline.py` - Pipeline integration with TensorRT flags
- `src/auto_voice/models/singing_voice_converter.py` - TensorRT engine loading and trt_enabled property
- `src/auto_voice/inference/tensorrt_engine.py` - TensorRT engine wrapper
- `src/auto_voice/audio/pitch_extractor.py` - GPU-accelerated pitch extraction
- `tests/test_tensorrt_conversion.py` - TensorRT validation tests

## 5. Best Practices

### For Best Voice Cloning

**Duration:**
- Record 45-60 seconds (optimal range)
- Minimum 30 seconds for acceptable quality
- Longer is generally better (up to 60s)

**Environment:**
- Use quiet room with minimal echo
- Avoid outdoor or noisy locations
- Close windows to reduce external noise
- Turn off fans, AC, or appliances

**Recording Quality:**
- Use consistent microphone and setup
- Maintain 6-12 inches from microphone
- Avoid pops and clicks (use pop filter if available)
- Monitor levels to prevent clipping

**Content:**
- Speak/sing naturally, avoid monotone
- Include variety: different pitches, dynamics, expressions
- Use conversational tone or natural singing
- Avoid extreme vocal effects or shouting

**Multi-Sample Profiles:**
- Record 2-5 samples for best results
- Use different content for each sample
- Ensure consistent recording setup
- System averages embeddings for robustness

### For Best Song Conversion

**Source Material:**
- Use high-quality source songs (lossless preferred)
- Ensure vocals are clear and prominent in mix
- Avoid heavily processed vocals (heavy autotune, effects)
- Solo vocals work better than dense arrangements

**Key Matching:**
- Check if song key matches your vocal range
- Use pitch shift if song is too high or low
- Typical ranges:
  - Female: C4-G5 (262-784 Hz)
  - Male: C3-G4 (130-392 Hz)

**Quality Settings:**
- Test with fast preset first to verify results
- Use balanced preset for general use
- Use quality preset for final versions you'll share

**Workflow:**
- Preview source song quality before conversion
- Start with short clips (30s) to test parameters
- Once satisfied, convert full song
- Save settings that work well for reuse

### Quality Expectations

**Pitch Accuracy:**
- Target: <10 Hz RMSE
- Imperceptible to most listeners
- Critical for natural-sounding singing

**Voice Similarity:**
- Target: >85% match (>0.85 cosine similarity)
- Converted voice should sound recognizably like you
- Higher is better

**Naturalness:**
- Target: Minimal artifacts, sounds like natural singing
- MOS (Mean Opinion Score) >4.0
- No robotic or processed qualities

**Processing Time:**
- Fast: 15-30 seconds for 30-second song
- Balanced: 30-60 seconds for 30-second song
- Quality: 60-120 seconds for 30-second song

## 6. Advanced Features

### Multi-Sample Profiles

Create voice profiles from multiple recordings for improved robustness:

```python
from auto_voice.inference import VoiceCloner

cloner = VoiceCloner(device='cuda')

# Create profile from multiple samples
samples = ['voice1.wav', 'voice2.wav', 'voice3.wav']
profile = cloner.create_voice_profile_from_multiple_samples(
    audio_paths=samples,
    user_id='my_user_id'
)
```

**Benefits:**
- More robust to voice variations
- Better handles different recording conditions
- Averages out noise and inconsistencies
- Improves overall conversion quality

### Batch Conversion

Convert multiple songs to the same profile efficiently:

```bash
python examples/demo_batch_conversion.py \
  --songs song1.mp3 song2.mp3 song3.mp3 \
  --profile-id profile-uuid \
  --output-dir converted/
```

**Use Cases:**
- Convert entire album to your voice
- Process playlist of favorite songs
- Batch testing with different parameters

### Real-time Streaming

Process audio in chunks for low-latency applications:

```javascript
socket.emit('convert_song_stream', {
  conversion_id: 'unique-id',
  song_data: base64Audio,
  target_profile_id: 'profile-uuid'
});

socket.on('conversion_progress', (data) => {
  updateProgressBar(data.progress);
});
```

**Use Cases:**
- Live performance applications
- Real-time preview during editing
- Interactive music production

### Quality Preset Customization

Fine-tune individual parameters beyond preset defaults:

```python
result = pipeline.convert_song(
    song_path='song.mp3',
    target_profile_id='profile-uuid',
    separation_model='htdemucs_ft',  # Higher quality separation
    separation_shifts=2,  # More accurate but slower
    vocoder_quality='high',  # Best vocoder quality
    pitch_confidence_threshold=0.8  # Stricter pitch filtering
)
```

## 7. Frequently Asked Questions

**Q: How long does conversion take?**

A: For a 30-second song: 15-60 seconds depending on quality preset and GPU. Processing scales linearly with song length. Fast preset is ~0.5x real-time, balanced is ~1x real-time, quality is ~2x real-time.

**Q: Can I convert any song?**

A: Yes, but results are best with clear vocals. Songs with prominent, clean vocals work best. Instrumental-heavy music or heavily processed vocals may produce artifacts.

**Q: Do I need to sing the original song myself?**

A: No! The system extracts the pitch contour from the original singer and applies your voice timbre. You don't need any singing ability.

**Q: What if the song is out of my vocal range?**

A: Use the pitch shift feature to transpose the song ±12 semitones. This shifts the entire song to match your comfortable vocal range.

**Q: How many voice profiles can I create?**

A: Unlimited. Each profile uses approximately 1MB of storage. You can create different profiles for different vocal styles.

**Q: Can I delete or update profiles?**

A: Yes. Use the profile management UI or API to delete profiles. To update, you can add more samples to an existing profile or create a new one.

**Q: What audio quality should I use for voice samples?**

A: 44.1kHz WAV or FLAC is best. MP3 at 320kbps is acceptable. Higher quality recordings produce better voice profiles.

**Q: Can I use this commercially?**

A: Check the license terms. Personal use is typically allowed. Commercial use may require licensing depending on your use case.

**Q: Does this work with any language?**

A: Yes. The system is language-agnostic since it works with acoustic features rather than linguistic content.

**Q: How do I improve conversion quality?**

A: (1) Use higher quality voice samples, (2) Create multi-sample profiles, (3) Use quality preset, (4) Ensure source songs have clear vocals, (5) Use lossless audio formats.

## 8. Limitations and Known Issues

### Current Limitations

- **GPU Dependency**: Requires NVIDIA GPU for reasonable performance (CPU is 10-50x slower)
- **Solo Vocals**: Works best with solo vocals; duets and harmonies may have artifacts
- **Vocal Separation**: Requires clear vocal separation; heavily processed vocals may not separate well
- **Profile Quality**: Final quality depends heavily on voice sample recording quality
- **Processing Time**: Scales with song length; very long songs (>5 min) require significant time

### Known Issues

- **Extreme Pitch Ranges**: Very low (<100 Hz) or very high (>800 Hz) pitches may have reduced accuracy
- **Vocal Effects**: Extreme effects (heavy distortion, autotune) may not convert accurately
- **Background Vocals**: May be included in conversion if not properly separated
- **Memory Usage**: Long songs (>5 minutes) may require significant GPU memory
- **Breath Sounds**: May be converted along with singing (can add realism or artifacts)

### Workarounds

- **Long Songs**: Split into smaller chunks (<5 min each), convert separately, then merge
- **Complex Arrangements**: Use quality preset and htdemucs_ft separation model
- **Memory Issues**: Use fast preset or CPU fallback for very long songs
- **Poor Separation**: Try different separation models or pre-process audio

## 9. Next Steps

### Learn More

- **Interactive Tutorials**:
  - Try [Voice Cloning Demo Notebook](../examples/voice_cloning_demo.ipynb)
  - Try [Song Conversion Demo Notebook](../examples/song_conversion_demo.ipynb)

- **Technical Documentation**:
  - Read [API Documentation](api_voice_conversion.md)
  - Explore [Model Architecture](model_architecture.md)
  - Review [Quality Evaluation Guide](quality_evaluation_guide.md)

- **Operations**:
  - Check [Runbook](runbook.md) for troubleshooting
  - Review deployment best practices

### Community and Support

- Report issues on GitHub
- Join community discussions
- Share your creations
- Contribute improvements

### Tips for Success

1. Start with high-quality recordings (both voice sample and songs)
2. Test with short clips before converting full songs
3. Experiment with different presets to find your preference
4. Use multi-sample profiles for best results
5. Monitor quality metrics to ensure targets are met
6. Keep voice samples organized by vocal style or use case
7. Document settings that work well for future reference
