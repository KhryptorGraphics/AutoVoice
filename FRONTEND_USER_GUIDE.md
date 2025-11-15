# 🎤 AutoVoice Frontend - User Guide

Complete guide to using the AutoVoice web interface for singing voice conversion.

---

## 🚀 Getting Started

### Starting the Application

1. **Start the Backend Server**
   ```bash
   cd /home/kp/repos/autovoice
   conda activate autovoice
   python -m auto_voice.web.app
   ```
   Backend will run on `http://localhost:5000`

2. **Start the Frontend**
   ```bash
   cd frontend
   npm install  # First time only
   npm run dev
   ```
   Frontend will run on `http://localhost:5173`

3. **Open in Browser**
   Navigate to `http://localhost:5173`

---

## 📱 Pages Overview

### 1. Home Page (`/`)
**Purpose:** Welcome page and quick start guide

**Features:**
- System overview
- Quick start instructions
- Recent conversions
- System status summary

---

### 2. Single Conversion (`/singing-conversion`)
**Purpose:** Convert individual songs with full control

**How to Use:**

1. **Upload Song**
   - Drag & drop audio file or click to browse
   - Supported formats: MP3, WAV, FLAC, OGG, M4A
   - Max file size: 100MB

2. **Select Target Voice**
   - Choose from existing voice profiles
   - Or create a new profile first (see Voice Profiles page)

3. **Configure Settings**
   - **Pitch Shift:** -12 to +12 semitones (0 = no change)
   - **Preserve Original Pitch:** Keep singer's pitch (recommended)
   - **Preserve Vibrato:** Maintain vibrato characteristics
   - **Preserve Expression:** Keep emotional dynamics
   - **Output Quality:** Draft, Fast, Balanced, High, Studio
   - **Denoise Input:** Remove background noise
   - **Enhance Output:** Apply audio enhancement

4. **Start Conversion**
   - Click "Start Conversion" button
   - Watch real-time progress updates
   - See stage-by-stage processing
   - View estimated time remaining

5. **Download Result**
   - Play converted audio in browser
   - Download to your device
   - Convert another song

**Tips:**
- Use "Balanced" quality for best speed/quality ratio
- Enable "Preserve Original Pitch" for singing
- "Studio" quality takes 4x longer but gives best results

---

### 3. Batch Conversion (`/batch-conversion`)
**Purpose:** Convert multiple songs at once

**How to Use:**

1. **Add Files**
   - Click "Add Files" button
   - Select multiple audio files
   - Files appear in the list

2. **Select Voice Profile**
   - Choose target voice for all files
   - Same voice will be used for all conversions

3. **Start Batch**
   - Click "Start Batch" button
   - Files process one by one
   - Track individual progress

4. **Download Results**
   - Download individual files as they complete
   - Or click "Download All" when finished

**Tips:**
- Process 3-5 files at a time for best performance
- All files use the same settings
- Remove files before starting if needed

---

### 4. Conversion History (`/history`)
**Purpose:** View and manage past conversions

**Features:**
- **Filter by Date:** All, Today, Week, Month
- **Replay Audio:** Play converted songs
- **Re-download:** Download again if needed
- **Delete:** Remove individual conversions
- **Clear All:** Delete entire history

**How to Use:**
1. Select time filter (All, Today, Week, Month)
2. Click Play icon to listen
3. Click Download icon to save
4. Click Trash icon to delete

**Tips:**
- History stored in browser localStorage
- Clearing browser data will delete history
- Download important conversions to keep them

---

### 5. Voice Profiles (`/voice-profiles`)
**Purpose:** Create and manage custom voice profiles

**How to Create Profile:**

1. **Upload Voice Sample**
   - Record or upload 30-60 seconds of clean voice audio
   - Best quality: Studio recording, no background noise
   - Minimum: 10 seconds of clear voice

2. **Enter Profile Details**
   - Name: Descriptive name (e.g., "John's Voice")
   - Description: Optional notes

3. **Create Profile**
   - Click "Create Profile"
   - Wait for processing (30-60 seconds)
   - Profile appears in list

**Managing Profiles:**
- View all profiles
- Preview voice samples
- Delete unwanted profiles

**Tips:**
- Use high-quality voice samples for best results
- Include variety in voice sample (different pitches)
- Avoid background music or noise
- 30-60 seconds is optimal length

---

### 6. System Status (`/system-status`)
**Purpose:** Monitor system performance and health

**Information Displayed:**
- **GPU Status:** Available, name, memory usage
- **Model Status:** Loaded models and their status
- **API Health:** Backend connectivity
- **Performance Metrics:** Processing speed, queue status

**How to Use:**
- Check before starting conversions
- Monitor GPU usage during processing
- Verify models are loaded
- Check API connectivity

---

### 7. Settings (`/settings`)
**Purpose:** Configure default preferences

**Settings Available:**

**Conversion Defaults:**
- Default Quality: Draft, Fast, Balanced, High, Studio
- Auto-download: Automatically download results

**Audio Processing:**
- Preserve Original Pitch (default: ON)
- Preserve Vibrato (default: ON)
- Preserve Expression (default: ON)
- Denoise Input (default: OFF)
- Enhance Output (default: OFF)

**Performance:**
- Max Concurrent Jobs: 1-10 (default: 3)

**How to Use:**
1. Adjust settings to your preferences
2. Click "Save Settings"
3. Settings apply to all future conversions
4. Click "Reset" to restore defaults

**Tips:**
- Set defaults for your most common use case
- Enable auto-download if you always download results
- Adjust concurrent jobs based on your GPU

---

## 🎯 Common Workflows

### Workflow 1: Quick Single Conversion
1. Go to "Convert" page
2. Upload song
3. Select voice profile
4. Click "Start Conversion"
5. Download result

### Workflow 2: Batch Processing
1. Go to "Batch" page
2. Add multiple files
3. Select voice profile
4. Click "Start Batch"
5. Download all results

### Workflow 3: Create Custom Voice
1. Go to "Profiles" page
2. Record or upload voice sample
3. Enter profile details
4. Create profile
5. Use in conversions

---

## 💡 Tips & Best Practices

### For Best Quality
- Use high-quality input audio (320kbps MP3 or WAV)
- Choose "Studio" quality preset
- Enable all preservation options
- Use clean voice samples for profiles

### For Fastest Processing
- Use "Draft" or "Fast" quality preset
- Disable denoise and enhance options
- Process shorter audio clips
- Use GPU acceleration (check System Status)

### For Batch Processing
- Process 3-5 files at a time
- Use "Balanced" quality for speed/quality ratio
- Ensure all files are similar format
- Monitor system status during processing

---

## 🐛 Troubleshooting

### Backend Not Connected
**Problem:** "Failed to connect" error
**Solution:**
1. Check backend is running: `curl http://localhost:5000/api/v1/health`
2. Restart backend: `python -m auto_voice.web.app`
3. Check firewall settings

### Conversion Fails
**Problem:** Conversion errors or hangs
**Solution:**
1. Check System Status page for GPU/model status
2. Try smaller audio file
3. Use lower quality preset
4. Check backend logs for errors

### Slow Processing
**Problem:** Conversions take too long
**Solution:**
1. Check GPU is available (System Status)
2. Use lower quality preset
3. Reduce concurrent jobs in Settings
4. Close other GPU-intensive applications

### Audio Quality Issues
**Problem:** Output sounds distorted or poor quality
**Solution:**
1. Use higher quality preset
2. Enable "Enhance Output" in settings
3. Use better quality input audio
4. Create better voice profile with clean sample

---

## 📊 Quality Presets Explained

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| **Draft** | 4x faster | 60% | Quick tests, previews |
| **Fast** | 2x faster | 80% | Real-time applications |
| **Balanced** | 1x (baseline) | 100% | General use (recommended) |
| **High** | 0.5x (2x slower) | 130% | High-quality productions |
| **Studio** | 0.25x (4x slower) | 150% | Professional studio work |

**Recommendation:** Start with "Balanced" and adjust based on your needs.

---

## 🎓 Advanced Features

### Audio Waveform Visualization
- Visual representation of audio
- Playback controls
- Volume adjustment
- Time scrubbing

### Real-time Progress Tracking
- Stage-by-stage updates
- Percentage completion
- Estimated time remaining
- WebSocket-based live updates

### Conversion History
- Automatic tracking
- Date-based filtering
- Quick replay and download
- Local storage persistence

---

## 📞 Support

- **GitHub Issues:** https://github.com/KhryptorGraphics/AutoVoice/issues
- **Documentation:** See README.md files
- **API Docs:** http://localhost:5000/api/v1/health

---

**Happy Converting! 🎵**


