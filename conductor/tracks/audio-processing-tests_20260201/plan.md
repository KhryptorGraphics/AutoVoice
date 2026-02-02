# Implementation Plan: Audio Processing Tests

**Track ID:** audio-processing-tests_20260201
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-01
**Status:** [ ] Pending

## Phase 1: Speaker Diarization Tests

### Tasks

- [ ] Task 1.1: Test pyannote diarization integration
  - Test speaker count detection (2-3 speakers)
  - Test timestamp generation
  - Test segment boundaries
  - Use fixture: `tests/fixtures/multi_speaker_10s.wav`

- [ ] Task 1.2: Test timestamp accuracy
  - Verify timestamp precision (±0.5s tolerance)
  - Test segment duration validation
  - Test overlapping speech handling

- [ ] Task 1.3: Test edge cases
  - Single speaker audio
  - Silent audio (no speech)
  - Very short segments (<1s)
  - Long audio (>5min) with chunking

- [ ] Task 1.4: Test error handling
  - Invalid audio format
  - Empty audio file
  - GPU OOM handling (large audio)

### Verification

- [ ] Diarization accuracy validated
- [ ] Timestamp precision ±0.5s
- [ ] Edge cases handled gracefully
- [ ] Coverage ≥70% for `speaker_diarization.py`

## Phase 2: Diarization Extractor Tests

### Tasks

- [ ] Task 2.1: Test segment extraction from timestamps
  - Extract segments from diarization output
  - Verify segment audio matches timestamps
  - Test start/end boundary accuracy

- [ ] Task 2.2: Test segment audio quality
  - Verify no clipping
  - Test sample rate preservation
  - Test audio format consistency

- [ ] Task 2.3: Test multiple speakers
  - Extract segments for 2-3 speakers
  - Verify speaker separation
  - Test segment assignment correctness

- [ ] Task 2.4: Test edge cases
  - Overlapping speech (choose dominant speaker)
  - Very short segments (<1s)
  - Silence padding (add 0.1s before/after)

### Verification

- [ ] Segment extraction accurate
- [ ] Audio quality preserved
- [ ] Multi-speaker separation works
- [ ] Coverage ≥70% for `diarization_extractor.py`

## Phase 3: Speaker Matcher Tests

### Tasks

- [ ] Task 3.1: Test embedding-based matching
  - Extract embeddings from audio segments
  - Compute cosine similarity
  - Verify same speaker matches (>0.8 similarity)
  - Verify different speakers reject (<0.7 similarity)

- [ ] Task 3.2: Test similarity threshold tuning
  - Test threshold values (0.6, 0.7, 0.8, 0.9)
  - Measure false positive/negative rates
  - Recommend optimal threshold

- [ ] Task 3.3: Test unknown speaker detection
  - Test new speaker vs existing profiles
  - Verify unknown speaker flagged correctly
  - Test auto-profile suggestion

- [ ] Task 3.4: Test edge cases
  - Very short audio (<3s) - unreliable embeddings
  - Noisy audio (low SNR)
  - Similar voices (family members)

### Verification

- [ ] Embedding matching accurate (>90% correct)
- [ ] Threshold tuning validated
- [ ] Unknown speakers detected
- [ ] Coverage ≥70% for `speaker_matcher.py`

## Phase 4: Vocal Separation Tests

### Tasks

- [ ] Task 4.1: Test Demucs separation
  - Separate vocals, drums, bass, other
  - Verify 4 output stems created
  - Test output file naming

- [ ] Task 4.2: Test separation quality
  - Calculate SDR (signal-to-distortion ratio)
  - Target: SDR >10 dB for vocals
  - Test with music fixture (instrumental + vocals)

- [ ] Task 4.3: Test GPU vs CPU execution
  - Test GPU acceleration (if available)
  - Test CPU fallback
  - Measure speedup (GPU should be 5-10x faster)

- [ ] Task 4.4: Test error handling
  - Invalid audio format
  - GPU OOM on large audio
  - Missing Demucs model weights

### Verification

- [ ] Separation produces 4 stems
- [ ] Vocal quality (SDR >10 dB)
- [ ] GPU acceleration works
- [ ] Coverage ≥70% for `separation.py`

## Phase 5: YouTube Download Tests

### Tasks

- [ ] Task 5.1: Test successful download
  - Mock yt-dlp download call
  - Verify audio file created
  - Test format extraction (audio-only, best quality)
  - Use 5s test clip fixture

- [ ] Task 5.2: Test metadata extraction
  - Extract title, artist, duration
  - Test uploader info
  - Test thumbnail URL

- [ ] Task 5.3: Test error handling
  - Mock 404 error (video not found)
  - Mock geo-block error
  - Mock invalid URL
  - Mock network timeout

- [ ] Task 5.4: Test download options
  - Test audio format selection (opus, m4a, mp3)
  - Test quality selection (best, worst)
  - Test output path customization

### Verification

- [ ] Download mocked successfully
- [ ] Metadata extracted correctly
- [ ] Errors handled gracefully
- [ ] Coverage ≥70% for `youtube_downloader.py`

## Phase 6: YouTube Metadata Tests

### Tasks

- [ ] Task 6.1: Test artist detection from title
  - Test "Artist - Song" format
  - Test "Song by Artist" format
  - Test "Artist: Song" format

- [ ] Task 6.2: Test featured artist extraction
  - Test "Artist ft. Featured" format
  - Test "Artist feat. Featured" format
  - Test multiple featured artists

- [ ] Task 6.3: Test title cleaning
  - Remove "(Official Video)" suffixes
  - Remove "[HD]" tags
  - Remove emoji characters
  - Trim whitespace

- [ ] Task 6.4: Test genre classification (if applicable)
  - Test genre hints in title (e.g., "[Rock]", "(Jazz)")
  - Test uploader channel analysis

### Verification

- [ ] Artist detection accurate (>90%)
- [ ] Featured artists extracted
- [ ] Title cleaning works
- [ ] Coverage ≥70% for `youtube_metadata.py`

## Phase 7: File Organizer Tests

### Tasks

- [ ] Task 7.1: Test directory creation
  - Test profile directory structure
  - Test subdirectory creation (samples/, adapters/)
  - Use tempfile for testing

- [ ] Task 7.2: Test file naming conventions
  - Test profile_{id}_sample_{n}.wav format
  - Test adapter_{id}_epoch_{n}.safetensors format
  - Test timestamp-based naming

- [ ] Task 7.3: Test cleanup of old files
  - Test old checkpoint deletion (keep last 3)
  - Test temp file cleanup
  - Test disk space management

### Verification

- [ ] Directory structure validated
- [ ] File naming consistent
- [ ] Cleanup works correctly
- [ ] Coverage ≥70% for `file_organizer.py`

## Phase 8: Integration Tests

### Tasks

- [ ] Task 8.1: Test YouTube → Diarization → Profiles flow
  - Mock YouTube download
  - Run diarization on downloaded audio
  - Extract segments for each speaker
  - Create profiles from segments
  - Verify end-to-end integrity

- [ ] Task 8.2: Test Separation → Diarization flow
  - Separate vocals from music
  - Run diarization on vocals
  - Verify diarization accuracy improves

- [ ] Task 8.3: Test Speaker Matching → Profile Update flow
  - Match segments to existing profiles
  - Update profile embeddings
  - Verify embedding convergence

### Verification

- [ ] End-to-end flows tested
- [ ] Integration points validated
- [ ] Data integrity maintained
- [ ] All integration tests pass

## Final Verification

- [ ] All acceptance criteria met
- [ ] Coverage ≥70% for audio/
- [ ] Tests complete in <5 minutes
- [ ] No network requests (all mocked)

---

**Estimated Timeline:** 2 days
**Dependencies:** None
**Blocks:** coverage-report-generation_20260201

---

_Generated by Gap Analysis Watcher._
