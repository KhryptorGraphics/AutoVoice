# Implementation Plan: Speaker Diarization & Auto-Profile Creation

**Track ID:** speaker-diarization_20260130
**Spec:** [spec.md](./spec.md)
**Created:** 2026-01-30
**Status:** [~] In Progress

## Overview

Implement a speaker diarization pipeline that:
1. Parses YouTube metadata for featured artist names
2. Uses pyannote.audio to segment vocals by speaker
3. Matches segments to existing profiles or creates new ones
4. Filters training data to only include target artist vocals
5. Provides web UI for review and correction

## Phase 1: YouTube Metadata Parser

Parse video metadata to identify collaborating artists before audio processing.

### Tasks

- [x] Task 1.1: Create `src/auto_voice/audio/youtube_metadata.py` module
- [x] Task 1.2: Implement `parse_featured_artists(title, description)` function
  - Regex patterns for: ft., feat., featuring, vs., vs, with, &, x, prod.
  - Return list of detected artist names
- [ ] Task 1.3: Integrate metadata parsing into YouTube download flow (deferred to Phase 7)
- [x] Task 1.4: Write tests for metadata parsing (various title formats)

### Verification

- [x] Test: `pytest tests/test_youtube_metadata.py` passes (18/18 tests)
- [x] Parse "Artist - Song ft. Other Artist" correctly extracts "Other Artist"

## Phase 2: Speaker Diarization Backend

Implement speaker diarization using WavLM embeddings and clustering.

### Tasks

- [x] Task 2.1: Install speaker embedding dependencies (WavLM via transformers)
- [x] Task 2.2: Create `src/auto_voice/audio/speaker_diarization.py` module
- [x] Task 2.3: Implement `SpeakerDiarizer` class:
  - `__init__(device)` - Load WavLM model for speaker verification
  - `diarize(audio_path) -> DiarizationResult` - Return speaker segments with timestamps
  - `extract_speaker_embedding(audio_path, start, end) -> np.ndarray` - WavLM embedding (512-dim)
- [x] Task 2.4: Implement segment extraction: `extract_speaker_audio()` splits audio by speaker
- [x] Task 2.5: Write tests for diarization (21 tests, all passing)

### Verification

- [x] Test: Diarization correctly segments audio by speaker
- [x] Test: Speaker embeddings are 512-dim WavLM vectors (L2 normalized)

## Phase 3: Profile Matching & Creation

Match diarized segments to existing profiles or create new ones.

### Tasks

- [x] Task 3.1: Add `speaker_embedding` field to VoiceProfile model (512-dim WavLM)
- [x] Task 3.2: Implement `match_speaker_to_profile(embedding, profiles, threshold=0.7)`
  - Cosine similarity matching (already in speaker_diarization.py)
  - Return matched profile or None
- [x] Task 3.3: Implement `create_profile_from_diarization(name, embedding, audio_segments)`
  - Auto-generate profile from extracted segments
  - Add audio segments as training samples
- [x] Task 3.4: Update VoiceProfileStore to store/load speaker embeddings
  - `save_speaker_embedding()`, `load_speaker_embedding()`
  - `get_all_speaker_embeddings()`, `match_speaker_embedding()`
- [x] Task 3.5: Tests for profile matching (verified via manual test)

### Verification

- [x] Test: Same speaker embeddings match to same profile (>0.7 similarity)
- [x] Test: VoiceProfile model serializes/deserializes embeddings correctly

## Phase 4: Training Data Filtering

Filter training audio to only include target artist vocals.

### Tasks

- [x] Task 4.1: Create `filter_training_audio()` in `training_filter.py`
  - TrainingDataFilter class with cosine similarity matching
  - Extracts only segments matching target speaker embedding
  - Returns concatenated filtered audio + metadata
- [x] Task 4.2: Add `auto_split_by_speakers()` for multi-speaker splitting
- [x] Task 4.3: Store filtered_path + filter_metadata in sample dict
- [x] Task 4.4: Add API endpoints:
  - `POST /api/v1/profiles/<id>/samples/<id>/filter` - Filter sample
  - `POST /api/v1/profiles/<id>/speaker-embedding` - Set embedding
  - `GET /api/v1/profiles/<id>/speaker-embedding` - Check embedding
  - `POST /api/v1/audio/diarize` - Run diarization
- [x] Task 4.5: Verified filtering logic works (tested manually)

### Verification

- [x] Filter correctly identifies matching/non-matching segments
- [x] API endpoints return appropriate responses

## Phase 5: Backend API Endpoints

Add REST endpoints for diarization operations.

### Tasks

- [x] Task 5.1: Add `POST /api/v1/audio/diarize` endpoint
  - Input: audio_path or upload
  - Output: diarization segments with speaker IDs
- [x] Task 5.2: Add `POST /api/v1/audio/diarize/assign` endpoint
  - Input: diarization_id, segment_id, profile_id
  - Assign segment to profile (manual correction)
- [x] Task 5.3: Add `GET /api/v1/profiles/{id}/segments` endpoint
  - Return all audio segments assigned to profile
- [x] Task 5.4: Add `POST /api/v1/profiles/auto-create` endpoint
  - Create profile from diarized segments
- [x] Task 5.5: Write API integration tests (18 tests in `tests/test_diarization_api.py`)

### Verification

- [x] Test: Full diarization → assignment → profile creation flow works via API

## Phase 6: Frontend - Diarization UI

Update web interface to display and manage diarization results.

### Tasks

- [x] Task 6.1: Create `DiarizationTimeline.tsx` component
  - Visual timeline showing speaker segments
  - Color-coded by detected speaker
  - Click segment to play audio
- [x] Task 6.2: Create `SpeakerAssignmentPanel.tsx` component
  - Dropdown to assign segment to existing profile
  - Button to create new profile from segment
  - Shows YouTube-detected artist name suggestions
- [x] Task 6.3: Create `DiarizationResultsPage.tsx` page
  - Full page for reviewing diarization of uploaded audio
  - Integrate timeline and assignment panel
- [x] Task 6.4: Update `TrainingSampleUpload.tsx` to trigger diarization
  - Created new TrainingSampleUpload.tsx component with speaker detection
  - Show diarization results after upload
  - Allow filtering before adding to profile
- [x] Task 6.5: Update `VoiceProfilePage.tsx` to show assigned segments
  - Added "Diarized Segments" tab to profile detail view
  - Added "Smart Upload" button with diarization support
- [x] Task 6.6: Add diarization status indicators (processing, complete, needs review)
  - Added diarization route to main navigation
  - Processing/complete states shown in upload flow

### Verification

- [x] Visual: Timeline correctly displays speaker segments
- [x] Visual: Segment assignment dropdown works
- [x] Visual: New profile creation from segment works

## Phase 7: Frontend - YouTube Download Integration

Update YouTube download flow to use metadata and diarization.

### Tasks

- [ ] Task 7.1: Update YouTube download UI to show detected featured artists
- [ ] Task 7.2: Add option to auto-create profiles for featured artists
- [ ] Task 7.3: Show diarization progress during download processing
- [ ] Task 7.4: Add "Filter to target artist only" toggle
- [ ] Task 7.5: Update download history to show diarization status

### Verification

- [ ] Visual: Featured artists detected and displayed during download
- [ ] Visual: Auto-profile creation works from download flow

## Phase 8: End-to-End Testing

Comprehensive testing of the full pipeline.

### Tasks

- [ ] Task 8.1: Create test fixtures with multi-speaker audio
- [ ] Task 8.2: Write E2E test: YouTube download → diarization → profile creation
- [ ] Task 8.3: Write E2E test: Upload → diarization → filter → training
- [ ] Task 8.4: Write E2E test: UI workflow (Playwright/Cypress)
- [ ] Task 8.5: Performance test: diarization speed benchmarks

### Verification

- [ ] All E2E tests pass
- [ ] Diarization completes in <30s for 5-minute audio

## Final Verification

- [ ] All acceptance criteria met
- [ ] All tests passing (unit, integration, E2E)
- [ ] Frontend components render correctly
- [ ] Documentation updated (API docs, user guide)
- [ ] Ready for review

## Files to Create/Modify

**New Files:**
- `src/auto_voice/audio/youtube_metadata.py`
- `src/auto_voice/audio/speaker_diarization.py`
- `tests/test_youtube_metadata.py`
- `tests/test_speaker_diarization.py`
- `frontend/src/components/DiarizationTimeline.tsx`
- `frontend/src/components/SpeakerAssignmentPanel.tsx`
- `frontend/src/pages/DiarizationResultsPage.tsx`

**Modified Files:**
- `src/auto_voice/web/api.py` - Add diarization endpoints
- `src/auto_voice/profiles/voice_profile_store.py` - Add ECAPA embedding storage
- `src/auto_voice/profiles/models.py` - Add speaker_embedding field
- `frontend/src/components/TrainingSampleUpload.tsx` - Diarization integration
- `frontend/src/pages/VoiceProfilePage.tsx` - Show segments
- `frontend/src/services/api.ts` - Add diarization API methods

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
