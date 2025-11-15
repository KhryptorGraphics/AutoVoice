# 🚀 Quick Start - Testing AutoVoice Singing Voice Conversion

## Prerequisites
- Python 3.12.12 (conda environment already set up)
- Node.js 18+ (for frontend)
- GPU with CUDA 12.1 support (recommended)

## Step 1: Activate Environment
```bash
conda activate autovoice
```

## Step 2: Download Models
```bash
cd /home/kp/repos/autovoice
python scripts/download_singing_models.py
```

**Expected Output:**
- ✅ HuBERT-Soft (360.9 MB) - Downloaded
- ✅ RMVPE (172.8 MB) - Downloaded
- ✅ torchcrepe - Installed
- ℹ️ HiFi-GAN - Optional (CREPE fallback enabled)

## Step 3: Start Backend Server
```bash
python -m auto_voice.web.app
```

**Expected Output:**
```
 * Running on http://127.0.0.1:5000
 * WebSocket support enabled
```

## Step 4: Start Frontend (New Terminal)
```bash
cd /home/kp/repos/autovoice/frontend
npm run dev
```

**Expected Output:**
```
VITE v5.0.0 ready in 123 ms
➜  Local:   http://localhost:5173/
```

## Step 5: Open Application
Open browser to: **http://localhost:5173**

## Step 6: Test Singing Voice Conversion

1. **Upload Audio File**
   - Click "Upload Audio" or drag-and-drop
   - Supported formats: MP3, WAV, FLAC, OGG, M4A
   - Max size: 100MB

2. **Configure Settings**
   - Select target voice profile
   - Adjust pitch shift (-12 to +12 semitones)
   - Toggle preservation options:
     - ✓ Preserve Original Pitch
     - ✓ Preserve Vibrato
     - ✓ Preserve Expression

3. **Select Quality Preset**
   - **Fast**: Quick processing, lower quality
   - **Balanced**: Good quality, reasonable speed
   - **High**: Better quality, slower processing
   - **Studio**: Best quality, slowest processing

4. **Start Conversion**
   - Click "Convert" button
   - Watch real-time progress updates
   - WebSocket updates show:
     - Overall progress percentage
     - Current processing stage
     - Individual stage progress

5. **Download Result**
   - Once complete, download converted audio
   - Compare with original
   - Check pitch preservation

## Troubleshooting

### Backend Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill process if needed
kill -9 <PID>
```

### Frontend Won't Start
```bash
# Clear node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Model Download Fails
```bash
# Check internet connection
# Try downloading again
python scripts/download_singing_models.py

# If HiFi-GAN fails, it's optional - system will use CREPE
```

### WebSocket Connection Issues
- Check browser console for errors
- Ensure backend is running on port 5000
- Check firewall settings

## Performance Tips

1. **GPU Acceleration**
   - System automatically uses CUDA if available
   - Check GPU usage: `nvidia-smi`

2. **Processing Speed**
   - Fast preset: ~5-10 seconds per song
   - Balanced preset: ~15-30 seconds per song
   - High preset: ~30-60 seconds per song
   - Studio preset: ~60-120 seconds per song

3. **Memory Usage**
   - Typical: 2-4 GB VRAM
   - Ensure sufficient GPU memory available

## Next Steps

1. ✅ Test with sample audio files
2. ✅ Verify pitch preservation accuracy
3. ✅ Check audio quality
4. ✅ Measure processing speed
5. ✅ Gather feedback for improvements

## Support

- **GitHub**: https://github.com/KhryptorGraphics/AutoVoice
- **Documentation**: See FINAL_STATUS_REPORT.md
- **Issues**: Check GitHub issues page

---

**Status**: System is 75% complete and ready for testing! 🎉

