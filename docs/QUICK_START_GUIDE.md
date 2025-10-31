# AutoVoice Quick Start Guide

## 🎯 What This System Does

**AutoVoice performs singing voice conversion (voice cloning):**
- Takes any singing performance (source)
- Converts it to sound like a different voice (target)
- Preserves pitch, timing, and musical expression from the source

**Important Understanding:**
- ✅ Changes WHO is singing (voice timbre/identity)
- ❌ Does NOT transfer singing skill or technique
- The quality of singing in the output matches the source performance

**Example:**
- Professional singer performs song → convert to your voice → professional singing in your voice ✓
- You sing (amateur) → convert to famous singer → amateur singing in their voice ✓

---

## ⚡ Quick Setup (10 minutes)

### Step 1: Fix Python Environment (Required)

**Option A: Use Python 3.12 (Recommended)**

```bash
# Check current Python version
python --version

# If Python 3.13, create 3.12 environment
conda create -n autovoice python=3.12 -y
conda activate autovoice

# Or use pyenv
pyenv install 3.12.7
pyenv local 3.12.7
```

**Option B: Use PyTorch 2.7+ with Python 3.13 (Experimental)**

```bash
# Install latest PyTorch with 3.13 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 2: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify key packages
pip list | grep -E "torch|librosa|soundfile|flask"
```

### Step 3: Download Pre-trained Models

```bash
# Download all required models (~585 MB total)
python scripts/download_pretrained_models.py

# Or download required models only
python scripts/download_pretrained_models.py --required-only
```

This downloads:
- ✅ So-VITS-SVC 5.0 weights (300 MB)
- ✅ HiFi-GAN vocoder (40 MB)
- ✅ HuBERT-Soft encoder (95 MB)
- ⚠️  So-VITS-SVC 4.0 weights (150 MB, optional fallback)

---

## 🎵 Usage Examples

### Example 1: Simple Conversion

```bash
# Convert a song to your voice
python examples/demo_voice_conversion.py \
  --song data/test_song.mp3 \
  --reference data/my_voice.wav \
  --output output/converted.wav
```

### Example 2: High Quality with Pitch Shift

```bash
# Studio quality, shift down 2 semitones
python examples/demo_voice_conversion.py \
  --song data/test_song.mp3 \
  --reference data/my_voice.wav \
  --preset studio \
  --pitch-shift -2.0
```

### Example 3: Fast Preview

```bash
# Quick conversion for testing
python examples/demo_voice_conversion.py \
  --song data/test_song.mp3 \
  --reference data/my_voice.wav \
  --preset fast
```

---

## 🌐 Web Interface

### Start the Web Server

```bash
python main.py
```

Then open: http://localhost:5000

### Features:
- 🎙️ Upload reference voice to create profile
- 🎵 Upload song for conversion
- ⚙️ Adjust settings (pitch, quality, volumes)
- 📊 Real-time progress tracking
- 💾 Download converted audio

---

## 📊 Quality Tips

### For Best Results:

**Reference Voice (Target):**
- ✅ 30-60 seconds of clear singing
- ✅ Diverse notes and phonemes
- ✅ Minimal background noise
- ✅ Studio quality if possible
- ❌ Avoid: noisy audio, short clips (<10s), monotone

**Source Song:**
- ✅ Clear vocals (or use vocal separation)
- ✅ Good pitch accuracy from original singer
- ✅ MP3/WAV/FLAC formats supported
- ❌ Avoid: heavily processed/auto-tuned vocals

**Settings:**
- **draft**: Fast testing (0.25x speed)
- **fast**: Real-time conversion (1x speed)
- **balanced**: Good quality (0.5x speed) ← Recommended
- **high**: High quality (2x speed)
- **studio**: Maximum quality (4x speed)

**Pitch Shift:**
- Use ±2 semitones for natural key changes
- Large shifts (±12) may degrade quality
- Match reference voice's comfortable range

---

## 🔧 Troubleshooting

### "PyTorch not available" or "CUDA not found"

```bash
# Run automated fix
./scripts/setup_pytorch_env.sh

# Or install manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "Models not found"

```bash
# Re-run model download
python scripts/download_pretrained_models.py --skip-existing
```

### "Failed to separate vocals"

- Ensure song file is valid audio format
- Try with a different song
- Check disk space for cache directory

### "Conversion too slow"

```bash
# Use faster preset
python examples/demo_voice_conversion.py ... --preset fast

# Or check GPU is being used
python -c "import torch; print(torch.cuda.is_available())"
```

### "Low quality output"

- Use longer reference audio (60s recommended)
- Try studio preset for maximum quality
- Ensure reference voice has clear singing
- Check input song has good vocal quality

---

## 📚 Next Steps

### Learn More:
- [Voice Conversion Guide](voice_conversion_guide.md) - Detailed usage
- [API Documentation](api-documentation.md) - REST API reference
- [Model Architecture](model_architecture.md) - Technical deep dive

### Advanced Usage:
- Train custom models on your dataset
- Use TensorRT for 2-3x speedup
- Batch convert multiple songs
- Integrate into your application via API

---

## ⚖️ Ethical Guidelines

**Please use responsibly:**
- ✅ Obtain consent before cloning someone's voice
- ✅ Disclose AI-generated content
- ✅ Respect copyright and intellectual property
- ❌ Do not create deepfakes or misleading content
- ❌ Do not use for fraud or impersonation

---

## 🆘 Support

If you encounter issues:
1. Check troubleshooting section above
2. Review error messages carefully
3. Open GitHub issue with:
   - Python/PyTorch versions
   - Error message
   - Steps to reproduce

---

**Made with ❤️ by the AutoVoice Team**
