# AutoVoice Complete Overhaul - Implementation Plan

**Date:** 2026-01-31
**Status:** In Progress
**Priority:** P0

---

## Root Cause Diagnosis (CONFIRMED)

The SOTA pipeline produces **buzzing noise** due to architecture mismatch:

| Component | Trained Format | Expected Format |
|-----------|---------------|-----------------|
| Model | `HQVoiceLoRAAdapter` (6-layer MLP) | `CoMoSVCDecoder` (consistency model) |
| Keys | `lora_0_A`, `lora_0_B`, etc. | `input_proj.adapter.lora_A`, etc. |
| Result | 0 matching keys applied | Decoder runs with random weights |

---

## Phase 0: Pipeline Architecture Analysis [COMPLETE]

- [x] 0.1 Diagnose SOTA pipeline - confirm adapter key mismatch
- [x] 0.2 Verify trained adapter format (HQ: rank=128, nvfp4: rank=16)
- [x] 0.3 Diagnose realtime pipeline - **UNTRAINED DECODER** (not adapter issue)
- [x] 0.4 Document architecture findings

**Realtime Pipeline Diagnosis:**
- `SimpleDecoder` in `scripts/realtime_pipeline.py` is randomly initialized
- It was never trained - produces noise output
- NOT an adapter format issue - the decoder itself needs training
- **Temporary solution:** Use Seed-VC quality pipeline until decoder is trained

## Phase 1: Speaker-Isolated Vocal Extraction [PENDING - CRITICAL]

**REQUIREMENT:** Extract clean, speaker-isolated vocal files for training LoRAs or models from scratch.

### Multi-Speaker Track Processing

For each multi-speaker vocal track (e.g., Connor + featured artist):

1. **Create FULL-LENGTH file for EACH detected speaker:**
   - Target speaker's segments: AUDIBLE
   - All other speakers: SILENCED (zero amplitude)
   - File duration = original track duration (enables remixing later)

2. **Auto-create voice profiles for each speaker:**
   - Primary artist (most speaking time) → main artist profile
   - Featured artists → new profiles created automatically
   - Profiles store: speaker embedding, name, training data paths

3. **Output structure:**
```
data/training_vocals/
  ├── conor_maynard/                    # Primary artist files
  │   ├── trackA_SPEAKER_00_isolated.wav
  │   ├── trackB_SPEAKER_00_isolated.wav
  │   └── speaker_profiles.json         # Speaker → Profile ID mapping
  ├── william_singe/
  │   └── ...
  └── featured/                         # Featured artist files
      ├── {profile_uuid_1}/
      │   ├── trackA_SPEAKER_01_isolated.wav
      │   └── trackC_SPEAKER_01_isolated.wav
      └── {profile_uuid_2}/
          └── ...

data/voice_profiles/
  ├── conor_maynard/
  │   └── speaker_profiles.json         # Maps SPEAKER_XX → profile_id
  └── william_singe/
      └── speaker_profiles.json
```

### Extraction Tasks

- [x] 1.1 Create `DiarizationExtractor` class (`src/auto_voice/audio/diarization_extractor.py`)
  - Load diarization JSON (speaker segments with timestamps)
  - Load separated vocals WAV
  - For EACH speaker: create full-length track with only that speaker audible
  - Apply fade in/out at segment boundaries (reduce clicks)

- [x] 1.2 Implement `get_or_create_profile()` for automatic profile creation
  - Primary speaker → use existing artist profile or create new
  - Featured artists → create new profile with UUID
  - Store speaker embedding for matching across tracks

- [x] 1.3 Run extraction for Connor Maynard (190 tracks)
  - **ACTUAL:** 326.9 minutes Connor vocals + 272.8 minutes featured artists
  - 4 unique speakers detected across all tracks

- [x] 1.4 Run extraction for William Singe (128 tracks)
  - **ACTUAL:** 248.5 minutes William vocals + 151.3 minutes featured artists
  - 4 unique speakers detected across all tracks

- [x] 1.5 Verify extraction quality
  - ✓ Full-length files preserved (195.2s == original)
  - ✓ 98.9% silence in featured artist tracks (only their parts audible)
  - ✓ Fade transitions applied

### Cross-Track Speaker Identification [PENDING - CRITICAL]

**PROBLEM:** SPEAKER_01 in track A could be Anth, but SPEAKER_01 in track B could be a completely different featured artist. Diarization assigns IDs per-track, not globally.

**SOLUTION:** Use speaker embeddings + YouTube metadata to identify speakers across all tracks.

#### YouTube Metadata Fetching

- [x] 1.6 Fetch YouTube metadata for ALL video IDs
  - Use yt-dlp to get: title, description, channel, upload date
  - Video IDs are in filenames (e.g., `08NWh97_DME_vocals.wav`)
  - Example: "The Kid LAROI, Justin Bieber - STAY (SING OFF vs. Tayler Holder)"

- [x] 1.7 Parse featured artist names from titles
  - Patterns: "ft. {name}", "feat. {name}", "x {name}", "vs. {name}", "& {name}"
  - Example: "Pillowtalk (ft. Anth)" → featured_artist = "Anth"
  - Example: "SING OFF vs. Tayler Holder" → featured_artist = "Tayler Holder"

#### Database Storage (SQLite)

- [x] 1.8 Create database schema for persistent metadata
  ```sql
  -- Track metadata from YouTube
  CREATE TABLE tracks (
    id TEXT PRIMARY KEY,           -- YouTube video ID
    title TEXT,
    channel TEXT,
    upload_date TEXT,
    duration_sec REAL,
    artist_name TEXT,              -- Main artist (conor_maynard, william_singe)
    fetched_at TIMESTAMP
  );

  -- Featured artists parsed from titles
  CREATE TABLE featured_artists (
    id INTEGER PRIMARY KEY,
    track_id TEXT REFERENCES tracks(id),
    name TEXT,                     -- Parsed name (e.g., "Anth")
    pattern_matched TEXT           -- Pattern used (e.g., "ft.")
  );

  -- Speaker embeddings for cross-track matching
  CREATE TABLE speaker_embeddings (
    id INTEGER PRIMARY KEY,
    track_id TEXT REFERENCES tracks(id),
    speaker_id TEXT,               -- SPEAKER_00, SPEAKER_01, etc.
    embedding BLOB,                -- 512-dim WavLM embedding
    duration_sec REAL,
    is_primary BOOLEAN
  );

  -- Global speaker clusters (same person across tracks)
  CREATE TABLE speaker_clusters (
    id TEXT PRIMARY KEY,           -- UUID
    name TEXT,                     -- "Anth", "Tayler Holder", "Unknown 1"
    is_verified BOOLEAN,           -- User confirmed identity
    created_at TIMESTAMP
  );

  -- Mapping: which embeddings belong to which cluster
  CREATE TABLE cluster_members (
    cluster_id TEXT REFERENCES speaker_clusters(id),
    embedding_id INTEGER REFERENCES speaker_embeddings(id),
    confidence REAL,               -- Cosine similarity score
    PRIMARY KEY (cluster_id, embedding_id)
  );
  ```

- [x] 1.9 Database location: `data/autovoice.db`

#### Speaker Embedding & Clustering

- [x] 1.10 Extract speaker embeddings for ALL featured artist segments
  - Use WavLM-based embeddings (512-dim) already in `SpeakerDiarizer`
  - Store in `speaker_embeddings` table
  - **Implemented:** `SpeakerMatcher.extract_embeddings_for_artist()`

- [x] 1.11 Cross-track speaker clustering
  - Compare embeddings across ALL tracks using cosine similarity
  - Cluster similar embeddings (same person) together
  - Threshold: cosine similarity > 0.85 = same person
  - Create entries in `speaker_clusters` and `cluster_members`
  - **Implemented:** `SpeakerMatcher.cluster_speakers()`

- [x] 1.12 Auto-match clusters to featured artist names
  - If track has featured_artist from YouTube title
  - And track has non-primary speaker
  - → Assign that cluster the featured artist name
  - **Implemented:** `SpeakerMatcher.auto_match_clusters_to_artists()`

- [x] 1.13 Re-organize extracted files by identified artist
  - Move files from generic profile to named artist directory
  - Update speaker_profiles.json with real names
  - Example: `featured/6154b814.../` → `featured/anth/`
  - **Implemented:** `src/auto_voice/audio/file_organizer.py`
  - Run: `python -m auto_voice.audio.file_organizer --execute`

- [ ] 1.14 Web UI for speaker identification
  - Play sample audio for each detected speaker cluster
  - Allow user to name unknown speakers
  - Merge/split speaker clusters manually if needed
  - Show YouTube title context for each track

**Files:**
- `src/auto_voice/audio/speaker_matcher.py` (new - cross-track matching)
- `src/auto_voice/audio/youtube_metadata.py` (new - yt-dlp fetching + title parsing)
- `src/auto_voice/db/schema.py` (new - SQLAlchemy models)
- `src/auto_voice/db/operations.py` (new - database operations)
- `frontend/src/components/SpeakerIdentificationPanel.tsx` (new)

### Web Interface for Extraction & Speaker ID

#### API Endpoints (Database-backed)

- [x] 1.15 API: `POST /api/v1/speakers/extraction/run` - Trigger extraction for artist
- [x] 1.16 API: `GET /api/v1/speakers/extraction/status/{job_id}` - Extraction progress
- [x] 1.17 API: `GET /api/v1/speakers/tracks` - List all tracks with YouTube metadata
- [x] 1.18 API: `POST /api/v1/speakers/tracks/fetch-metadata` - Fetch YouTube metadata for all tracks
- [x] 1.19 API: `GET /api/v1/speakers/clusters` - List all speaker clusters
- [x] 1.20 API: `GET /api/v1/speakers/clusters/{id}` - Get cluster details + member tracks
- [x] 1.21 API: `PUT /api/v1/speakers/clusters/{id}/name` - Name a speaker cluster
- [x] 1.22 API: `POST /api/v1/speakers/clusters/merge` - Merge two clusters
- [x] 1.23 API: `POST /api/v1/speakers/clusters/split` - Split a cluster
- [x] 1.24 API: `GET /api/v1/speakers/clusters/{id}/sample` - Get audio sample
- [x] 1.25 API: `POST /api/v1/speakers/identify` - Run cross-track speaker matching
- **Implemented:** `src/auto_voice/web/speaker_api.py` (all endpoints)

#### Frontend Components

- [x] 1.26 Frontend: TrackListPanel
  - Show all tracks with YouTube titles
  - Show detected featured artists from titles
  - Filter by artist, by has-featured-artist
  - **Implemented:** `frontend/src/components/TrackListPanel.tsx`

- [x] 1.27 Frontend: ExtractionPanel on VoiceProfilePage
  - Show extraction status
  - List detected speakers with durations
  - Preview isolated audio samples
  - **Implemented:** `frontend/src/components/ExtractionPanel.tsx`

- [x] 1.28 Frontend: SpeakerIdentificationPanel
  - Show all detected speaker clusters with audio samples
  - "Play Sample" button for each cluster
  - Text input to name each speaker (e.g., "Anth", "Unknown Rapper 1")
  - "Merge Clusters" for same person incorrectly split
  - "Split Cluster" for different people incorrectly merged
  - Auto-suggest names from YouTube titles (ft. Anth → suggest "Anth")
  - Show which tracks each cluster appears in
  - **Implemented:** `frontend/src/components/SpeakerIdentificationPanel.tsx`

- [x] 1.29 Frontend: FeaturedArtistCard
  - Show artist name, total duration, track count
  - Link to isolated vocals directory
  - "Train LoRA" button
  - **Implemented:** `frontend/src/components/FeaturedArtistCard.tsx`

**Files:**
- `src/auto_voice/audio/diarization_extractor.py` (created)
- `src/auto_voice/audio/speaker_matcher.py` (new)
- `src/auto_voice/audio/youtube_metadata.py` (new)
- `src/auto_voice/db/schema.py` (new - SQLAlchemy models)
- `src/auto_voice/db/operations.py` (new - database CRUD)
- `src/auto_voice/web/api.py` (update - add all endpoints)
- `frontend/src/components/TrackListPanel.tsx` (new)
- `frontend/src/components/ExtractionPanel.tsx` (new)
- `frontend/src/components/SpeakerIdentificationPanel.tsx` (new)
- `frontend/src/components/FeaturedArtistCard.tsx` (new)

### Why Full-Length Files?

Keeping files at full duration enables:
1. Train LoRA/model for each artist separately
2. Convert each artist independently
3. **Remix all converted tracks back together** - same timing, different voices

---

## Phase 2: Seed-VC Zero-Shot Conversion [COMPLETE]

- [x] 2.1 Fix torch.no_grad() wrapper for vocoder (inference mode issue)
- [x] 2.2 Run William → Conor conversion with Seed-VC
- [x] 2.3 Verify output quality (max=0.946, rms=0.1106)
- [x] 2.4 Mix with instrumental
- [x] 2.5 Update Cipher memory

**Output Files:**
- `test_audio/pillowtalk/william_as_conor_SEEDVC.wav` (195.5s @ 44100Hz)
- `test_audio/pillowtalk/final_william_as_conor_SEEDVC.wav` (mixed)

**Issue Found:** Converted vocals have William's timing (195.5s), not Connor's (200.1s).
Voice conversion preserves SOURCE timing, not TARGET timing.

---

## Phase 3: Vocal Alignment System [PENDING - CRITICAL]

**Problem:** Voice conversion only changes TIMBRE, not TIMING.
William's cover has different timing than Connor's original.
Converted vocals don't sync with Connor's instrumental.

**Solution:** DTW (Dynamic Time Warping) alignment BEFORE conversion:

- [ ] 3.1 Implement DTW alignment using librosa.sequence.dtw
- [ ] 3.2 Time-warp source vocals to match target timing
- [ ] 3.3 Optional: Pitch correction to match target pitch contour
- [ ] 3.4 Run voice conversion on aligned vocals
- [ ] 3.5 Mix aligned+converted vocals with target instrumental
- [ ] 3.6 Add alignment options to API (align_to_target: true/false)
- [ ] 3.7 Frontend UI for alignment settings

**Files:**
- `scripts/aligned_conversion.py` (created - CLI tool)
- `src/auto_voice/audio/alignment.py` (to create - reusable module)
- `src/auto_voice/web/api.py` (update - add alignment endpoint)

**Expected Result:**
- William's vocals stretched/compressed to match Connor's exact timing
- Pitch corrected to match Connor's exact notes (every note hit properly)
- Voice conversion applied using trained LoRA (9.9 hours Connor data)
- Final output syncs perfectly with Connor's instrumental
- **William sings the song EXACTLY as Connor sang it** - same timing, same notes, Connor's voice

---

## Phase 4: Pipeline Integration [PENDING]

Wire pipeline selection to backend API:

- [ ] 4.1 Add `pipeline_type` parameter to `/convert/song` endpoint
- [ ] 4.2 Add `pipeline_type` to WebSocket startSession
- [ ] 4.3 Implement pipeline routing (PipelineFactory pattern)
- [ ] 4.4 Wire frontend PipelineSelector to API calls
- [ ] 4.5 Add `/pipelines/status` endpoint

**Files to modify:**
- `src/auto_voice/web/api.py`
- `frontend/src/pages/KaraokePage.tsx`
- `frontend/src/services/audioStreaming.ts`
- NEW: `src/auto_voice/inference/pipeline_factory.py`

---

## Phase 5: LoRA Retraining (REQUIRED) [PENDING]

**CRITICAL:** Must train on FULL speaker-isolated vocal dataset for max quality.

### Training Data (After Phase 1 Extraction):

| Artist | Profile ID | Isolated Files | Expected Duration |
|--------|------------|----------------|-------------------|
| Connor Maynard | `c572d02c-...` | 190 tracks | **~594.7 minutes (9.9 hours)** |
| William Singe | `7da05140-...` | 128 tracks | **~373.3 minutes (6.2 hours)** |
| Featured Artists | Auto-generated UUIDs | Variable | Variable |

**Data Locations (After Extraction):**
```
data/training_vocals/
  ├── conor_maynard/     # 190 *_SPEAKER_XX_isolated.wav files (Connor only)
  ├── william_singe/     # 128 *_SPEAKER_XX_isolated.wav files (William only)
  └── featured/          # Featured artist files by profile UUID
      └── {profile_id}/  # Isolated vocals for that featured artist
```

**Training Tasks:**

- [ ] 5.1 Modify trainer to target CoMoSVCDecoder layers (fix key format)
- [ ] 5.2 Update save format to use decoder-compatible keys
- [ ] 5.3 Update trainer to load vocals from `data/training_vocals/{artist}/` or `featured/{profile_id}/`
- [ ] 5.4 Train Connor nvfp4 LoRA with ALL isolated vocals
- [ ] 5.5 Train William nvfp4 LoRA with ALL isolated vocals
- [ ] 5.6 Train featured artist LoRAs (as needed)
- [ ] 5.7 Test conversion quality with comprehensive LoRAs
- [ ] 5.8 Compare quality metrics vs previous training

---

## Phase 6: Web UI Integration [PENDING]

**All backend functionality MUST be exposed via web interface.**

### Profile & Training Data API Endpoints:

- [ ] 6.1 `GET /api/v1/profiles` - List ALL profiles (main + featured artists)
- [ ] 6.2 `GET /api/v1/profiles/{id}` - Profile details with training data stats
- [ ] 6.3 `GET /api/v1/profiles/{id}/training-data` - List isolated vocal files
- [ ] 6.4 `GET /api/v1/profiles/{id}/training-data/stats` - Duration, file count, etc.
- [ ] 6.5 `GET /api/v1/profiles/{id}/training-data/preview/{file_id}` - Audio preview
- [ ] 6.6 `PUT /api/v1/profiles/{id}` - Update profile name (rename featured artists)

### Training API Endpoints:

- [ ] 6.7 `POST /api/v1/training/start` - Start LoRA training with config
- [ ] 6.8 `GET /api/v1/training/{job_id}/status` - Training progress, metrics
- [ ] 6.9 `POST /api/v1/training/{job_id}/stop` - Stop training early
- [ ] 6.10 `GET /api/v1/adapters/{profile_id}` - List trained adapters
- [ ] 6.11 `DELETE /api/v1/adapters/{profile_id}/{adapter_type}` - Delete adapter

### Conversion API Endpoints:

- [ ] 6.12 `POST /api/v1/convert/aligned` - Conversion with DTW alignment
- [ ] 6.13 `GET /api/v1/pipelines/status` - Pipeline availability
- [ ] 6.14 `POST /api/v1/remix/tracks` - Remix multiple converted tracks together

### Frontend Components Required:

- [ ] 6.15 **ProfileListPage** - Show ALL profiles including featured artists
  - Filter by: main artist, featured artist, has training data, has adapters
  - Quick actions: view, train, rename

- [ ] 6.16 **TrainingDataPanel** - Show isolated vocals per profile
  - Display: file count, total duration, silence ratio
  - Audio player: preview any isolated file
  - Stats: speaker duration breakdown per track

- [ ] 6.17 **SpeakerProfileCard** - Card component for each detected speaker
  - Show: profile name, duration, track count, training status
  - Actions: rename, view files, start training

- [ ] 6.18 **TrainingConfigPanel** - Configure training parameters
  - LoRA rank, epochs, early stopping
  - Select vocal files to include/exclude
  - Start training button

- [ ] 6.19 **TrainingProgressPanel** - Live training monitoring
  - Loss curves, epoch progress, ETA
  - Quality metrics per checkpoint
  - Early stopping indicator

- [ ] 6.20 **AlignmentSettingsPanel** - For conversion page
  - Enable/disable DTW alignment
  - Enable/disable pitch correction
  - Preview aligned vs unaligned

- [ ] 6.21 **AdapterSelector** - Choose trained adapter for conversion
  - List adapters by type (nvfp4/hq)
  - Show training metrics per adapter
  - Compare adapter quality

- [ ] 6.22 **QualityDashboard** - Post-conversion metrics
  - Speaker similarity score
  - MCD, pitch accuracy
  - Sync quality (if aligned)

- [ ] 6.23 **RemixPanel** - Combine converted tracks
  - Select multiple converted tracks
  - Adjust individual volumes
  - Export final mix

### Profile Page Updates:

- [ ] 6.24 Show training data stats on VoiceProfilePage
- [ ] 6.25 Show detected speakers with isolated file counts
- [ ] 6.26 Link to TrainingPage from profile
- [ ] 6.27 Show available adapters on profile page
- [ ] 6.28 Allow renaming featured artist profiles

---

## Phase 7: Quality & Testing [PENDING]

- [ ] 7.1 E2E test suite for extraction pipeline
- [ ] 7.2 E2E test suite for dual conversion pipeline
- [ ] 7.3 Quality metrics automation
- [ ] 7.4 Output assessment system
- [ ] 7.5 Self-healing loop integration

---

## Correct Workflow Order (CRITICAL)

**DO NOT run conversions until ALL prerequisites are complete:**

```
1. EXTRACT speaker-isolated vocals [Phase 1]
   └─ Each speaker gets FULL-LENGTH track (others silenced)
   └─ Auto-create profiles for featured artists
   └─ ~594.7 min Connor, ~373.3 min William + featured artists

2. EXPOSE extraction in web UI [Phase 1]
   └─ Show profiles for ALL detected speakers
   └─ Show isolated files per profile
   └─ Allow renaming featured artists

3. FIX LoRA training architecture [Phase 5]
   └─ Train for CoMoSVCDecoder, not HQVoiceLoRAAdapter

4. TRAIN comprehensive LoRAs [Phase 5]
   └─ Connor: 9.9 hours of isolated vocals
   └─ William: 6.2 hours of isolated vocals
   └─ Featured artists: as needed

5. IMPLEMENT timing alignment (DTW) [Phase 3]
   └─ Warp source to match target timing

6. IMPLEMENT pitch correction [Phase 3]
   └─ Adjust source pitches to match target notes

7. THEN run conversion:
   └─ Align William → Connor timing
   └─ Correct pitch to Connor's notes
   └─ Convert voice using trained LoRA
   └─ Mix with Connor's instrumental

8. REMIX multi-artist tracks [Phase 6]
   └─ Convert each artist separately
   └─ Combine all converted tracks
   └─ Export final mixed output

9. UPDATE web interface [Phase 6]
   └─ Expose ALL functionality to users
```

---

## Files Created/Modified

### New Files:
- `src/auto_voice/audio/diarization_extractor.py` - Speaker isolation extraction
- `src/auto_voice/audio/alignment.py` - DTW alignment module
- `src/auto_voice/inference/pipeline_factory.py` - Pipeline routing
- `frontend/src/components/ExtractionPanel.tsx` - Extraction UI
- `frontend/src/components/TrainingDataPanel.tsx` - Training data display
- `frontend/src/components/SpeakerProfileCard.tsx` - Speaker profile card
- `frontend/src/components/RemixPanel.tsx` - Track remixing UI
- `frontend/src/pages/ProfileListPage.tsx` - All profiles view

### Modified Files:
- `src/auto_voice/web/api.py` - New endpoints
- `src/auto_voice/training/trainer.py` - CoMoSVCDecoder support
- `frontend/src/pages/VoiceProfilePage.tsx` - Training data integration
- `frontend/src/pages/KaraokePage.tsx` - Pipeline selection
- `frontend/src/services/api.ts` - New API methods

---

## Verification Commands

```bash
# Run speaker-isolated extraction
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m src.auto_voice.audio.diarization_extractor --artist all

# Verify extraction output
ls -la data/training_vocals/conor_maynard/ | head
ls -la data/training_vocals/featured/ | head

# Check profile mappings
cat data/voice_profiles/conor_maynard/speaker_profiles.json

# Test Seed-VC conversion
ffplay test_audio/pillowtalk/william_as_conor_SEEDVC.wav

# Check adapter format
PYTHONNOUSERSITE=1 PYTHONPATH=src python -c "
import torch
ckpt = torch.load('data/trained_models/hq/c572d02c-c687-4bed-8676-6ad253cf1c91_hq_lora.pt', weights_only=False)
print('Keys:', list(ckpt.get('lora_state', {}).keys())[:5])
"

# Run quality analysis
PYTHONNOUSERSITE=1 PYTHONPATH=src python -c "
import librosa
import numpy as np
converted, sr = librosa.load('test_audio/pillowtalk/william_as_conor_SEEDVC.wav', sr=16000)
print(f'Duration: {len(converted)/sr:.1f}s, RMS: {np.sqrt(np.mean(converted**2)):.4f}')
"
```

---

## Memory Updates

After each phase:
```bash
mcp__cipher__ask_cipher "Remember: AutoVoice overhaul - Phase X complete - [summary]"
bd sync --flush-only
```

---

## Self-Healing Loop

This track includes self-healing loop integration:
1. Output assessment after each conversion
2. If issues found → generate fix track
3. Loop continues until all quality gates pass

Quality gates:
- Not silent (audio length > 0)
- Not noise (SNR > 10dB)
- Speaker similarity > 0.7
- MCD < 8.0
- **Timing alignment:** Output duration matches target duration (±1s)
- **Sync quality:** Cross-correlation with target > 0.8
- **Isolation quality:** Non-target speaker segments have RMS < 0.01

---

## Architecture Consistency (CRITICAL - Prevents Drift)

**PROBLEM:** If the self-healing loop runs 50+ times, methodology can drift and lose critical requirements.

**SOLUTION:** Every generated fix track MUST inherit and maintain the following architecture:

### Mandatory Sections in Every Fix Track

Each auto-generated fix track must include these sections (copied from parent):

1. **Speaker Isolation Requirements**
   - Full-length files with target speaker audible, others silenced
   - Auto-create profiles for each detected speaker
   - Output structure matches `data/training_vocals/` schema

2. **Web UI Integration Requirements**
   - ALL backend functionality exposed via API endpoints
   - ALL API endpoints have corresponding frontend components
   - User can perform any action from web interface

3. **Training Data Schema**
   - Isolated vocals stored per profile
   - Featured artists get their own profiles
   - Consistent file naming: `{track}_{speaker}_isolated.wav`

4. **Quality Gates** (must be checked at each iteration)
   - All gates listed above
   - No degradation from previous iteration

5. **Memory Updates**
   - Cipher update after each phase
   - Beads sync after completion

### Fix Track Generation Template

When generating a fix track, use this template:

```markdown
# AutoVoice Fix Track - Iteration {N}

**Parent Track:** autovoice-complete-overhaul_20260131
**Iteration:** {N}
**Generated:** {timestamp}

## STEP 0: CIPHER MEMORY RETRIEVAL (MANDATORY - DO FIRST)

Before ANY implementation, retrieve complete project context:

\`\`\`bash
mcp__cipher__ask_cipher "Retrieve ALL memories for autovoice-complete-overhaul track:
- Speaker diarization and isolation work
- Featured artists (Anth, etc) and voice profiles
- Pipeline architecture (SOTA, realtime, LoRA)
- Training data extraction and schemas
- Quality metrics and issues found
- Web UI integration requirements
- All decisions and learnings from previous iterations"
\`\`\`

**Memory retrieval results:**
{paste Cipher response here before proceeding}

**Key context from memory:**
- Completed work: {list from Cipher}
- Known issues: {list from Cipher}
- Architecture decisions: {list from Cipher}
- Featured artists identified: {list from Cipher}

---

## INHERITED REQUIREMENTS (DO NOT MODIFY)

### Speaker Isolation Architecture
- Full-length files with target speaker audible, others silenced
- Auto-create profiles for featured artists
- Output: data/training_vocals/{artist}/ and data/training_vocals/featured/{profile_id}/

### Web UI Integration
- All backend → API endpoint
- All API endpoint → Frontend component
- User can: view profiles, view training data, start training, run conversion, remix tracks

### Training Data Schema
- Isolated vocals per profile
- File naming: {track}_{speaker}_isolated.wav
- Profile mapping in speaker_profiles.json

### Quality Gates (ALL REQUIRED)
- Audio length > 0 (not silent)
- SNR > 10dB (not noise)
- Speaker similarity > 0.7
- MCD < 8.0
- Duration within ±1s of target
- Cross-correlation > 0.8
- Non-target RMS < 0.01

## ISSUES TO FIX (This Iteration)

{list of issues from assessment}

## FIX TASKS

{tasks to fix the issues}

## VERIFICATION

{verification steps - must include all quality gates}
```

### Architecture Validation at Each Iteration

Before completing any fix track, verify:

```bash
# 1. Speaker isolation files exist and follow schema
ls data/training_vocals/conor_maynard/*_isolated.wav | wc -l  # Should be > 0
ls data/training_vocals/featured/ 2>/dev/null | wc -l  # Featured artist dirs

# 2. Profile mappings exist
cat data/voice_profiles/conor_maynard/speaker_profiles.json  # Should have mappings

# 3. API endpoints respond
curl http://localhost:5000/api/v1/profiles  # Should list all profiles
curl http://localhost:5000/api/v1/profiles/{id}/training-data  # Should list files

# 4. Frontend components exist
ls frontend/src/components/TrainingDataPanel.tsx
ls frontend/src/components/ExtractionPanel.tsx
ls frontend/src/components/RemixPanel.tsx

# 5. Quality gates pass
# (run quality assessment script)
```

### Preventing Methodology Drift

**Rule 1:** Fix tracks NEVER remove inherited requirements
- If a fix track doesn't need to modify a requirement, it still includes it
- Requirements accumulate, never diminish

**Rule 2:** Fix tracks reference parent track
- Every fix track has `parent_track` in metadata
- Can trace lineage back to original track

**Rule 3:** Schema validation before completion
- Validate output structure matches expected schema
- Validate API endpoints exist and respond
- Validate frontend components exist

**Rule 4:** Cross-iteration quality comparison
- Compare quality metrics to previous iteration
- Fail if quality degrades significantly (>10%)

**Rule 5:** Cipher Memory Continuity (CRITICAL)
- Before generating ANY fix track, MUST read ALL Cipher memories from the completing track
- Query Cipher for: "autovoice track", "speaker diarization", "speaker isolation", "featured artists", "voice profiles", "pipeline architecture", "training data"
- This ensures the new track starts with COMPLETE context of all work done
- Prevents knowledge loss across compaction boundaries
- Memory queries must be logged in fix track metadata

**Cipher Memory Query Template (Required before fix track generation):**
```bash
# Query all relevant project memories
mcp__cipher__ask_cipher "Search and retrieve ALL memories related to:
1. autovoice-complete-overhaul track progress and phases
2. speaker diarization and isolation work
3. featured artists (Anth, etc) and voice profiles
4. pipeline architecture (SOTA, realtime, LoRA training)
5. training data extraction and schemas
6. quality metrics and issues found
7. web UI integration requirements
I need complete context before generating fix track iteration {N}."
```

**Why This Is Critical:**
- Claude sessions compact/clear context periodically
- Without reading Cipher memories, fix tracks may "forget" earlier decisions
- Could lead to re-implementing already-solved problems
- Could lose critical context about WHY certain approaches were chosen
- Ensures institutional knowledge persists across all iterations

**Rule 6:** Post-Compaction Memory Recovery (CRITICAL)
- After EVERY session compaction or context clear, IMMEDIATELY read all Cipher memories
- This applies to the CURRENT track being worked on, not just fix track generation
- Do NOT proceed with any implementation until Cipher context is recovered
- Signs you've been compacted: conversation summary at start, missing recent context

**Post-Compaction Recovery Protocol:**
```bash
# IMMEDIATELY after detecting compaction, run this query:
mcp__cipher__ask_cipher "Search and retrieve ALL memories related to:
1. autovoice-complete-overhaul track - ALL phases and progress
2. speaker diarization, isolation, and extraction work
3. featured artists (Anth, etc) and voice profile mappings
4. pipeline architecture decisions (SOTA vs realtime, LoRA training)
5. training data schemas and file structures
6. quality metrics, issues found, and fixes applied
7. web UI integration requirements and API endpoints
8. cross-track speaker identification and database schema
9. ALL decisions made and WHY they were made
I need complete context to continue work after session compaction."
```

**How to Detect Compaction:**
- Session starts with "This session is being continued from a previous conversation"
- Recent work you did is not in immediate context
- You're asked to continue but don't remember recent changes

**Recovery Checklist:**
- [ ] Query Cipher for ALL project memories
- [ ] Review the plan.md for current phase/task
- [ ] Check metadata.json for completed phases
- [ ] Review any recent git commits
- [ ] THEN continue implementation

### Iteration Metadata

Each iteration stores:

```json
{
  "iteration": 15,
  "parent_track": "autovoice-complete-overhaul_20260131",
  "timestamp": "2026-01-31T15:30:00Z",
  "inherited_requirements": [
    "speaker_isolation",
    "web_ui_integration",
    "training_data_schema",
    "quality_gates",
    "cipher_memory_continuity"
  ],
  "cipher_memory_query": {
    "performed": true,
    "timestamp": "2026-01-31T15:29:00Z",
    "memories_retrieved": 8,
    "key_context": [
      "speaker isolation complete for 318 tracks",
      "featured artist Anth identified",
      "cross-track clustering threshold 0.85",
      "SOTA pipeline architecture mismatch diagnosed"
    ]
  },
  "issues_fixed": ["LOW_SPEAKER_SIM", "HIGH_MCD"],
  "quality_metrics": {
    "speaker_similarity": 0.82,
    "mcd": 6.5,
    "snr": 15.2
  },
  "quality_delta_from_previous": {
    "speaker_similarity": +0.05,
    "mcd": -0.8
  }
}
```

This ensures even after 50+ iterations:
- Architecture remains consistent
- Requirements are never lost
- Quality improves or stays stable
- Full traceability of changes
