# Voice Profiles User Guide

AutoVoice uses two profile roles:

- `source_artist`: extracted from an uploaded song after separation and diarization
- `target_user`: built from the user singing and trained for conversion

The product workflow is: upload a song, split stems, detect singers, create source profiles, collect user vocals into a target profile, train a LoRA, unlock a full model after 30 minutes of clean user vocals, then use that target voice for offline conversion or live karaoke.

## Profile Roles

### Source Artist Profiles

Source artist profiles represent the singers detected in the uploaded song.

They are created from:

- vocal stem separation
- diarization over the vocal stem
- singer grouping and per-singer segment extraction

Use source profiles to preserve the original artist performance. They are not trainable target voices.

### Target User Profiles

Target user profiles represent the voice you want the system to sing with.

They are built from:

- direct uploads of your own clean singing
- captured karaoke training phrases
- additional assigned samples you explicitly add to the target profile

Only target user profiles can be trained.

## End-to-End Workflow

### 1. Upload a Song

Start from a local song upload or the supported download flow.

The backend will:

- split the song into `vocals` and `instrumental`
- diarize the vocal stem
- group segments by singer
- create one `source_artist` profile per detected singer

After diarization, review the detected singers. Rename, merge, or discard them before proceeding.

### 2. Create or Open a Target User Profile

Create a target profile for your own singing voice or open an existing one.

For best results, upload:

- clean solo singing
- minimal background bleed
- multiple phrases across your normal range

Target profiles accumulate duration over time. The system tracks clean vocal seconds, sample count, and current model state.

### 3. Capture a Browser Sing-Along Take

After a song is uploaded and separated, the karaoke page can play the full
artist song in the browser while recording the browser computer's microphone.
Use this when another computer on the local network is the singer's workstation:

- serve AutoVoice over HTTPS on the LAN so browser microphone APIs are available
- select the browser headset mic and browser headphone output in the sing-along panel
- record while the user sings along to the original full song
- wait for the local take-quality check to pass or warn; failed takes are blocked
- preview the take before attaching it to a selected `target_user` profile

This browser-device selector is separate from the server-side karaoke audio
router. The server router still controls devices attached to the machine running
AutoVoice; the browser recorder controls devices attached to the computer using
the web UI.

For local HTTPS testing, start both services with:

```bash
scripts/local_https_dev.sh
```

The helper uses a self-signed certificate. A browser on another trusted LAN
machine must accept or trust that certificate before microphone capture works.

### 4. Train a LoRA

LoRA is the default training path for a target user profile.

Use it when you want:

- faster turnaround
- smaller artifacts
- a stable first-pass target voice for conversion and karaoke

Source artist profiles cannot be trained. If you try to train a source profile, the API rejects it.

### 5. Unlock Full-Model Training

When a target user profile reaches `30 minutes` of clean user vocals, the UI unlocks full-model training.

This promotion path is intended for:

- higher-accuracy timbre transfer
- better singing conversion stability
- better live-mode performance once the dedicated model is active

Until that threshold is reached, the profile remains LoRA-only.

### 6. Convert a Song

For offline conversion:

1. choose the uploaded song or vocal stem
2. choose the target user profile
3. select the pipeline
4. run conversion

The output keeps the source artist performance characteristics while replacing the singer identity with the target user voice.

When a target profile has an active full model, offline conversion uses that model directly. Otherwise it uses the trained LoRA path.

### 7. Reassemble With the Instrumental

After a conversion completes, the UI exposes:

- converted vocals
- instrumental stem
- a reassembly action

Use reassembly to create a fresh mixed output from the converted vocal track and the separated instrumental.

### 8. Use the Same Target Voice in Live Karaoke

Live mode uses the same target-user model lifecycle:

- LoRA-backed target profile until full-model promotion
- full model preferred automatically once active

This keeps offline and live behavior aligned.

## Best Practices

- Keep source artist profiles separate from target user profiles.
- Upload clean user vocals rather than noisy full-song mixes.
- Use LoRA first; promote to full-model training only after enough clean duration accumulates.
- Review diarization results before relying on source artist profiles for conversion.
- Use the reassembly step when you want a full mixed deliverable instead of a dry converted vocal track.

## Troubleshooting

### I do not see a train button on a profile

Check the profile role. `source_artist` profiles are extracted from songs and are not trainable. Use a `target_user` profile instead.

### Full-model training is still locked

The target profile has not yet accumulated `30 minutes` of clean user vocals. Keep uploading clean singing or capture more karaoke samples.

### Conversion is using the wrong voice

Confirm that:

- you selected the correct target user profile
- the profile has a trained model
- the active model type is what you expect

### I only have the converted vocal stem

Use the conversion result view or conversion history page and select `Reassemble With Instrumental` to rebuild the mixed output.
