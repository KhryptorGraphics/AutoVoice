# Voice Profile User Guide

Learn how to create, manage, and improve your voice profiles in AutoVoice.

---

## What is a Voice Profile?

A voice profile captures the unique characteristics of your singing voice. Once created, AutoVoice uses this profile to convert any song into your voice while preserving the original pitch, timing, and singing techniques.

**Key benefits:**
- Your voice profile improves over time as you sing more songs
- Advanced techniques like vibrato and melisma are preserved
- Multiple profiles can be created for different voice styles

---

## Creating Your First Profile

### Step 1: Prepare Reference Audio

For best results, use a recording that:
- Is **10-30 seconds** of clear singing
- Has **minimal background noise**
- Shows your **natural vocal range**
- Is in a common format (WAV, MP3, FLAC)

### Step 2: Upload to AutoVoice

1. Navigate to **Profiles** in the sidebar
2. Click **Create New Profile**
3. Upload your reference audio file
4. Enter a name for your profile
5. Click **Create Profile**

The system will analyze your voice and create an initial profile in about 10-20 seconds.

### Step 3: Verify Your Profile

After creation, you'll see:
- **Vocal Range:** Your detected low and high notes
- **Sample Count:** Currently 1 (your reference audio)
- **Model Version:** v1 (initial version)

---

## Improving Your Profile

Your voice profile gets better with more training data. There are two ways to add samples:

### Automatic Collection (Recommended)

During karaoke sessions:
1. Go to **Karaoke** page
2. Select your profile from the dropdown
3. Enable **"Capture training samples"**
4. Sing along with songs

The system automatically captures high-quality phrases from your performance and adds them to your profile.

### Manual Upload

1. Navigate to **Profiles** → Select your profile
2. Click **Upload Sample**
3. Select audio files with your singing
4. Click **Upload**

Each sample should be a clean singing phrase (5-15 seconds).

---

## Training Your Model

After accumulating new samples, you can train your model to improve quality.

### Automatic Training

When enabled, training triggers automatically after collecting a threshold of new samples (default: 10 samples).

To enable:
1. Go to **Profiles** → Settings
2. Enable **"Auto-train after new samples"**
3. Set threshold (recommended: 10-20 samples)

### Manual Training

1. Navigate to **Profiles** → Select your profile
2. Click **Train Model**
3. Review training configuration:
   - **LoRA Rank:** Higher = more capacity (default: 8)
   - **Epochs:** More = longer training (default: 10)
   - **Learning Rate:** Lower = more stable (default: 0.0001)
4. Click **Start Training**

Training progress is shown in real-time. A typical training job takes 5-15 minutes depending on sample count.

---

## Training Configuration

### Understanding the Settings

| Setting | What it does | Recommended |
|---------|--------------|-------------|
| **LoRA Rank** | Model adaptation capacity | 8 for most users, 16 for complex voices |
| **LoRA Alpha** | Scaling factor | Keep at 2x rank (default: 16) |
| **Learning Rate** | Training speed | 0.0001 (lower if unstable) |
| **Epochs** | Training iterations | 10-20 for best results |
| **EWC** | Prevents forgetting old voice | Keep enabled |

### Tips for Better Training

- **Quality over quantity:** 20 clean samples beats 100 noisy ones
- **Variety matters:** Include different songs and vocal ranges
- **Consistent style:** Train on your natural singing style
- **Regular updates:** Retrain every 15-25 new samples

---

## Managing Profiles

### Viewing Profile Details

Click on any profile to see:
- Sample count and duration
- Training history with quality metrics
- Model versions (checkpoints)

### Switching Profiles

In the Karaoke page, use the profile dropdown to switch between voices. Each profile maintains its own training history.

### Deleting Profiles

1. Go to **Profiles** → Select profile
2. Click **Delete Profile**
3. Confirm deletion

**Warning:** This permanently removes the profile and all its samples.

---

## Model Versions & Rollback

AutoVoice keeps multiple versions of your trained model.

### Why Versions Matter

Sometimes a training run might not improve quality (bad samples, wrong settings). Version history lets you revert to a previous state.

### Viewing Versions

1. Go to **Profiles** → Select profile
2. Click **Checkpoints** tab
3. See all saved model versions with their metrics

### Rolling Back

1. Find the version you want to restore
2. Click **Rollback to this version**
3. Confirm the rollback

Your profile will use the selected version for all conversions.

---

## Audio Device Configuration

### Setting Up Input/Output

For karaoke sessions, configure your audio devices:

1. Go to **Settings** → **Audio Devices**
2. Select your **microphone** for input
3. Select **speakers** for audience output
4. Select **headphones** for your monitoring

### Karaoke Audio Routing

| Output | What you hear |
|--------|---------------|
| **Speakers (Audience)** | Instrumental + your converted voice |
| **Headphones (You)** | Original song with artist vocals |

This setup lets you hear the original as a guide while the audience hears your voice.

---

## Troubleshooting

### Profile Creation Failed

**"Insufficient quality"** error:
- Use cleaner audio with less background noise
- Record in a quiet environment
- Ensure your voice is clearly audible

**"Inconsistent samples"** error:
- Use audio with only your voice
- Avoid recordings with multiple singers

### Training Not Improving

If quality doesn't improve after training:
- Check sample quality (remove noisy samples)
- Try lower learning rate (0.00005)
- Add more varied samples
- Rollback and try different settings

### Conversion Sounds Wrong

- Verify correct profile is selected
- Check that recent training completed successfully
- Try rolling back to a previous version

---

## Best Practices

1. **Start with a good reference:** Your first audio sets the baseline
2. **Sing naturally:** Don't force techniques you don't normally use
3. **Review samples:** Remove low-quality captures before training
4. **Monitor metrics:** Watch loss curves during training
5. **A/B test versions:** Compare new versions before committing
6. **Backup profiles:** Export important profiles regularly

---

## FAQ

**Q: How many samples do I need?**
A: Minimum 5-10 for basic quality, 50+ for best results.

**Q: How long should samples be?**
A: 5-15 seconds of continuous singing each.

**Q: Can I use speaking voice?**
A: The system is optimized for singing. Speaking samples may not work well.

**Q: Why does training take so long?**
A: Training runs entirely on GPU for quality. Duration depends on sample count and settings.

**Q: What formats are supported?**
A: WAV, MP3, FLAC, OGG, M4A for audio files.

---

*For technical details, see [API Reference](./api-voice-profile.md)*
