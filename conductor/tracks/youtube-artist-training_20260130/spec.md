# Spec: YouTube Artist Training Pipeline

**Track ID:** youtube-artist-training_20260130
**Created:** 2026-01-30
**Priority:** P1
**Status:** [ ] Not Started

## Overview

Automated pipeline to download all music videos from Connor Maynard and William Singe YouTube channels, extract vocals using speaker diarization, add segments to their voice profiles, and train LoRA models with OOM protection.

## Target Artists

1. **Connor Maynard**
   - YouTube Channel: Conor Maynard
   - Genre: Pop covers, original songs
   - Expected videos: 200+

2. **William Singe**
   - YouTube Channel: William Singe
   - Genre: R&B/Pop covers
   - Expected videos: 300+

## Pipeline Stages

### Stage 1: YouTube Discovery & Download
- Use yt-dlp to list all videos from each channel
- Download audio only (best quality AAC/MP3)
- Store metadata (title, duration, upload date)
- Skip non-music content (vlogs, shorts)

### Stage 2: Audio Separation
- Run Demucs HTDemucs to separate vocals from instrumentals
- Save isolated vocal tracks
- Discard instrumentals (or save for karaoke)

### Stage 3: Speaker Diarization
- Run WavLM-based diarization on each vocal track
- Identify featured artists vs main artist
- Extract segments by speaker
- Use chunked processing for memory safety

### Stage 4: Profile Matching
- Match speaker embeddings to existing Connor/William profiles
- Create profiles if they don't exist
- Add matched segments as training samples
- Store embedding metadata

### Stage 5: LoRA Training
- Train separate LoRA adapters for each artist
- Max settings: rank=16, alpha=32, epochs=50
- OOM protection: gradient checkpointing, mixed precision
- Memory monitoring: abort if >90% GPU memory

## Acceptance Criteria

1. [ ] All music videos downloaded from both channels
2. [ ] Vocals separated with <10% bleed
3. [ ] Diarization identifies correct speaker >95% accuracy
4. [ ] Connor profile has >4 hours of clean vocals
5. [ ] William profile has >4 hours of clean vocals
6. [ ] LoRA training completes without OOM
7. [ ] Voice conversion produces recognizable output

## Technical Requirements

- yt-dlp for YouTube download
- Demucs for separation
- WavLM for diarization
- Memory-safe chunked processing
- Parallel download with rate limiting
- Progress tracking via beads

## OOM Prevention

- Max 4GB GPU memory per diarization chunk
- Gradient checkpointing for training
- Mixed precision (fp16/bf16)
- Batch size auto-reduction on OOM
- Memory monitoring hooks
