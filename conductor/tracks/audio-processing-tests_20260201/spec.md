# Spec: Audio Processing Tests

**Track ID:** audio-processing-tests_20260201
**Priority:** P1 (HIGH)
**Created:** 2026-02-01

## Problem

Phase 2 of comprehensive-testing-coverage_20260201 (Audio Processing Tests) is incomplete. Speaker diarization, vocal separation, and YouTube download modules lack tests.

## Goal

Create comprehensive tests for audio processing modules including diarization, separation, YouTube download, metadata extraction, and file organization. Target 70% coverage for audio/ modules.

## Acceptance Criteria

1. Speaker diarization tested (pyannote integration, timestamp accuracy)
2. Vocal separation tested (Demucs, GPU vs CPU, quality metrics)
3. YouTube download tested (success, errors, metadata extraction)
4. Speaker matching tested (embedding-based, similarity threshold)
5. Diarization extractor tested (segment extraction, audio quality)
6. File organizer tested (directory creation, naming conventions)
7. All tests use fixtures (no network downloads)
8. Coverage ≥70% for `src/auto_voice/audio/`
9. Tests complete in <5 minutes

## Context

**Modules to Test:**
- `speaker_diarization.py` - Pyannote diarization
- `diarization_extractor.py` - Segment extraction
- `speaker_matcher.py` - Embedding-based matching
- `separation.py` - Demucs vocal separation
- `youtube_downloader.py` - YouTube download
- `youtube_metadata.py` - Metadata parsing
- `file_organizer.py` - File management

**Upstream Dependencies:**
- None (can start immediately)

**Downstream Impact:**
- Contributes to overall 80% coverage target
- Validates audio quality pipeline
- Enables confident audio module refactoring

## Out of Scope

- Audio quality subjective testing (manual listening)
- Real YouTube downloads (use mocks/fixtures)
- Audio format conversion (existing tests cover this)

## Technical Constraints

- Use fixture audio files from `tests/fixtures/`
- Mock YouTube API calls (no network requests)
- Use GPU if available, fallback to CPU
- Mark GPU-required tests with `@pytest.mark.cuda`
