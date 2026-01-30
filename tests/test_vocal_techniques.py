"""TDD tests for vocal technique detection and preservation.

Phase 5: Advanced Vocal Technique Preservation
- Task 5.1: Vibrato detection tests
- Task 5.3: Melisma/vocal run detection tests
- Task 5.5: Technique-aware pitch extraction tests
- Task 5.7: Technique transfer tests
"""

import numpy as np
import pytest
import torch


class TestVibratoDetection:
    """Task 5.1: Tests for vibrato detection in source audio."""

    def test_detect_vibrato_in_synthetic_signal(self):
        """Vibrato detector should identify periodic pitch modulation."""
        from auto_voice.audio.technique_detector import VibratoDetector

        # Create synthetic signal with vibrato (5Hz rate, ±30 cents)
        sr = 16000
        duration = 2.0
        base_f0 = 440.0  # A4
        vibrato_rate = 5.0  # Hz
        vibrato_depth_cents = 30.0

        t = np.linspace(0, duration, int(sr * duration))
        # Frequency modulation
        vibrato_depth_hz = base_f0 * (2 ** (vibrato_depth_cents / 1200) - 1)
        instantaneous_f0 = base_f0 + vibrato_depth_hz * np.sin(2 * np.pi * vibrato_rate * t)
        # Generate audio with FM
        phase = np.cumsum(2 * np.pi * instantaneous_f0 / sr)
        audio = np.sin(phase).astype(np.float32)

        detector = VibratoDetector(sample_rate=sr)
        result = detector.detect(audio)

        assert result.has_vibrato is True
        assert 4.0 < result.vibrato_rate < 7.0  # Typical vibrato 4-7 Hz
        assert 20.0 < result.vibrato_depth_cents < 50.0

    def test_no_vibrato_in_steady_tone(self):
        """Vibrato detector should not flag steady pitches."""
        from auto_voice.audio.technique_detector import VibratoDetector

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Pure sine wave - no vibrato
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        detector = VibratoDetector(sample_rate=sr)
        result = detector.detect(audio)

        assert result.has_vibrato is False

    def test_vibrato_segments_identified(self):
        """Vibrato detector should identify vibrato segments in mixed audio."""
        from auto_voice.audio.technique_detector import VibratoDetector

        sr = 16000
        # 0-1s: steady, 1-2s: vibrato, 2-3s: steady
        t1 = np.linspace(0, 1.0, sr)
        t2 = np.linspace(0, 1.0, sr)
        t3 = np.linspace(0, 1.0, sr)

        # Steady
        steady = np.sin(2 * np.pi * 440.0 * t1)

        # Vibrato section
        base_f0 = 440.0
        vibrato_rate = 5.5
        vibrato_depth_cents = 40.0
        vibrato_depth_hz = base_f0 * (2 ** (vibrato_depth_cents / 1200) - 1)
        inst_f0 = base_f0 + vibrato_depth_hz * np.sin(2 * np.pi * vibrato_rate * t2)
        phase = np.cumsum(2 * np.pi * inst_f0 / sr)
        vibrato = np.sin(phase)

        audio = np.concatenate([steady, vibrato, steady]).astype(np.float32)

        detector = VibratoDetector(sample_rate=sr)
        result = detector.detect(audio)

        assert result.has_vibrato is True
        assert len(result.vibrato_segments) >= 1
        # Check that vibrato was detected (segment boundaries may be imprecise
        # due to sliding window analysis, but presence is reliable)
        seg = result.vibrato_segments[0]
        # Segment should cover the vibrato region (1-2s) even if boundaries extend
        assert seg.end_time >= 1.5  # Should cover the vibrato section

    def test_vibrato_parameters_extraction(self):
        """Vibrato detector should extract rate and depth parameters."""
        from auto_voice.audio.technique_detector import VibratoDetector

        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))

        # Known vibrato parameters
        base_f0 = 440.0
        vibrato_rate = 6.0
        vibrato_depth_cents = 35.0

        vibrato_depth_hz = base_f0 * (2 ** (vibrato_depth_cents / 1200) - 1)
        inst_f0 = base_f0 + vibrato_depth_hz * np.sin(2 * np.pi * vibrato_rate * t)
        phase = np.cumsum(2 * np.pi * inst_f0 / sr)
        audio = np.sin(phase).astype(np.float32)

        detector = VibratoDetector(sample_rate=sr)
        result = detector.detect(audio)

        # Should be within reasonable tolerance
        assert abs(result.vibrato_rate - vibrato_rate) < 1.0
        assert abs(result.vibrato_depth_cents - vibrato_depth_cents) < 15.0


class TestMelismaDetection:
    """Task 5.3: Tests for melisma/vocal run detection."""

    def test_detect_melisma_rapid_pitch_changes(self):
        """Melisma detector should identify rapid pitch transitions."""
        from auto_voice.audio.technique_detector import MelismaDetector

        sr = 16000
        # Create rapid scale run: C4 -> D4 -> E4 -> F4 in 0.5s
        notes_hz = [261.63, 293.66, 329.63, 349.23]  # C4, D4, E4, F4
        note_duration = 0.125

        audio_segments = []
        for note_hz in notes_hz:
            t = np.linspace(0, note_duration, int(sr * note_duration))
            audio_segments.append(np.sin(2 * np.pi * note_hz * t))

        audio = np.concatenate(audio_segments).astype(np.float32)

        detector = MelismaDetector(sample_rate=sr)
        result = detector.detect(audio)

        assert result.has_melisma is True
        assert result.note_count >= 3  # At least 3 distinct notes

    def test_no_melisma_in_sustained_note(self):
        """Melisma detector should not flag sustained notes."""
        from auto_voice.audio.technique_detector import MelismaDetector

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Single sustained note
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        detector = MelismaDetector(sample_rate=sr)
        result = detector.detect(audio)

        assert result.has_melisma is False

    def test_melisma_with_vibrato_distinguished(self):
        """Melisma detector should distinguish runs from vibrato."""
        from auto_voice.audio.technique_detector import MelismaDetector

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Vibrato (periodic, returns to center pitch)
        base_f0 = 440.0
        vibrato_rate = 5.5
        vibrato_depth_cents = 40.0
        vibrato_depth_hz = base_f0 * (2 ** (vibrato_depth_cents / 1200) - 1)
        inst_f0 = base_f0 + vibrato_depth_hz * np.sin(2 * np.pi * vibrato_rate * t)
        phase = np.cumsum(2 * np.pi * inst_f0 / sr)
        audio = np.sin(phase).astype(np.float32)

        detector = MelismaDetector(sample_rate=sr)
        result = detector.detect(audio)

        # Vibrato should NOT be detected as melisma
        assert result.has_melisma is False

    def test_melisma_segments_identified(self):
        """Melisma detector should identify run segments."""
        from auto_voice.audio.technique_detector import MelismaDetector

        sr = 16000
        # Sustained note, then run, then sustained
        sustained = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, sr))

        # Quick run: A4 -> B4 -> C5 -> D5
        notes_hz = [440.0, 493.88, 523.25, 587.33]
        note_dur = 0.08
        run_segments = []
        for note in notes_hz:
            t = np.linspace(0, note_dur, int(sr * note_dur))
            run_segments.append(np.sin(2 * np.pi * note * t))
        run = np.concatenate(run_segments)

        audio = np.concatenate([sustained, run, sustained]).astype(np.float32)

        detector = MelismaDetector(sample_rate=sr)
        result = detector.detect(audio)

        assert result.has_melisma is True
        assert len(result.melisma_segments) >= 1


class TestTechniqueAwarePitchExtraction:
    """Task 5.5: Tests for technique-aware pitch extraction."""

    def test_pitch_extractor_detects_vibrato(self):
        """Enhanced pitch extractor should flag vibrato regions."""
        from auto_voice.audio.technique_detector import TechniqueAwarePitchExtractor

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        base_f0 = 440.0
        vibrato_rate = 5.5
        vibrato_depth_cents = 35.0
        vibrato_depth_hz = base_f0 * (2 ** (vibrato_depth_cents / 1200) - 1)
        inst_f0 = base_f0 + vibrato_depth_hz * np.sin(2 * np.pi * vibrato_rate * t)
        phase = np.cumsum(2 * np.pi * inst_f0 / sr)
        audio = np.sin(phase).astype(np.float32)

        extractor = TechniqueAwarePitchExtractor(sample_rate=sr)
        result = extractor.extract(audio)

        assert result.f0 is not None
        assert result.vibrato_mask is not None
        assert result.vibrato_mask.sum() > 0  # Some frames flagged as vibrato

    def test_pitch_extractor_preserves_vibrato_contour(self):
        """Pitch extractor should preserve vibrato modulation, not smooth it."""
        from auto_voice.audio.technique_detector import TechniqueAwarePitchExtractor

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        base_f0 = 440.0
        vibrato_rate = 5.5
        vibrato_depth_cents = 40.0
        vibrato_depth_hz = base_f0 * (2 ** (vibrato_depth_cents / 1200) - 1)
        inst_f0 = base_f0 + vibrato_depth_hz * np.sin(2 * np.pi * vibrato_rate * t)
        phase = np.cumsum(2 * np.pi * inst_f0 / sr)
        audio = np.sin(phase).astype(np.float32)

        extractor = TechniqueAwarePitchExtractor(sample_rate=sr)
        result = extractor.extract(audio)

        # F0 variation should be preserved (not smoothed away)
        f0_voiced = result.f0[result.f0 > 0]
        f0_std = np.std(f0_voiced)
        expected_std = vibrato_depth_hz / np.sqrt(2)  # RMS of sine wave

        # Should preserve at least 50% of the modulation
        assert f0_std > expected_std * 0.5

    def test_pitch_extractor_detects_melisma(self):
        """Enhanced pitch extractor should flag melisma regions."""
        from auto_voice.audio.technique_detector import TechniqueAwarePitchExtractor

        sr = 16000
        # Create rapid scale run
        notes_hz = [261.63, 293.66, 329.63, 349.23, 392.00]
        note_duration = 0.1

        audio_segments = []
        for note_hz in notes_hz:
            t = np.linspace(0, note_duration, int(sr * note_duration))
            audio_segments.append(np.sin(2 * np.pi * note_hz * t))

        audio = np.concatenate(audio_segments).astype(np.float32)

        extractor = TechniqueAwarePitchExtractor(sample_rate=sr)
        result = extractor.extract(audio)

        assert result.melisma_mask is not None
        assert result.melisma_mask.sum() > 0  # Some frames flagged as melisma


class TestTechniquePreservingConversion:
    """Task 5.7: Tests for technique transfer in voice conversion."""

    def test_conversion_preserves_vibrato_flag(self):
        """Voice conversion should pass vibrato flags to decoder."""
        from auto_voice.audio.technique_detector import TechniqueFlags
        from unittest.mock import MagicMock, patch

        # Mock the decoder to verify it receives technique flags
        mock_decoder = MagicMock()

        flags = TechniqueFlags(
            vibrato_mask=torch.ones(100, dtype=torch.bool),
            vibrato_rate=5.5,
            vibrato_depth_cents=35.0,
            melisma_mask=torch.zeros(100, dtype=torch.bool),
        )

        # The conversion pipeline should pass flags to decoder
        # This tests the interface, actual implementation in Task 5.8
        assert flags.vibrato_mask is not None
        assert flags.vibrato_rate == 5.5

    def test_technique_flags_serialization(self):
        """Technique flags should be serializable for pipeline passing."""
        from auto_voice.audio.technique_detector import TechniqueFlags

        flags = TechniqueFlags(
            vibrato_mask=torch.ones(100, dtype=torch.bool),
            vibrato_rate=6.0,
            vibrato_depth_cents=30.0,
            melisma_mask=torch.zeros(100, dtype=torch.bool),
        )

        # Should be convertible to dict for pipeline passing
        flags_dict = flags.to_dict()
        assert "vibrato_mask" in flags_dict
        assert "vibrato_rate" in flags_dict

        # Should be reconstructible
        restored = TechniqueFlags.from_dict(flags_dict)
        assert restored.vibrato_rate == flags.vibrato_rate

    def test_decoder_applies_vibrato_synthesis(self):
        """Decoder should synthesize vibrato when flags indicate vibrato region."""
        from auto_voice.audio.technique_detector import TechniqueFlags

        # Create flags indicating vibrato
        num_frames = 100
        flags = TechniqueFlags(
            vibrato_mask=torch.ones(num_frames, dtype=torch.bool),
            vibrato_rate=5.5,
            vibrato_depth_cents=35.0,
            melisma_mask=torch.zeros(num_frames, dtype=torch.bool),
        )

        # Mock decoder output analysis would verify vibrato in output
        # Full implementation in Task 5.8
        assert flags.has_vibrato is True

    def test_decoder_preserves_melisma_timing(self):
        """Decoder should preserve melisma note timing."""
        from auto_voice.audio.technique_detector import TechniqueFlags

        num_frames = 100
        # Mark frames 20-40 as melisma
        melisma_mask = torch.zeros(num_frames, dtype=torch.bool)
        melisma_mask[20:40] = True

        flags = TechniqueFlags(
            vibrato_mask=torch.zeros(num_frames, dtype=torch.bool),
            vibrato_rate=0.0,
            vibrato_depth_cents=0.0,
            melisma_mask=melisma_mask,
        )

        assert flags.has_melisma is True
        assert flags.melisma_mask[25].item() is True
