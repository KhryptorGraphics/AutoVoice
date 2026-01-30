"""Tests for evaluation metrics."""
import pytest
import numpy as np


class TestPitchRMSE:
    """Tests for pitch RMSE metric."""

    def test_identical_pitch(self):
        from auto_voice.evaluation.metrics import pitch_rmse
        f0 = np.array([440.0, 441.0, 442.0])
        assert pitch_rmse(f0, f0) == 0.0

    def test_octave_difference(self):
        from auto_voice.evaluation.metrics import pitch_rmse
        f0_a = np.array([440.0, 440.0])
        f0_b = np.array([880.0, 880.0])
        rmse = pitch_rmse(f0_a, f0_b)
        assert abs(rmse - 1200.0) < 1.0  # 1 octave = 1200 cents

    def test_voiced_only(self):
        from auto_voice.evaluation.metrics import pitch_rmse
        f0_a = np.array([440.0, 0.0, 440.0])
        f0_b = np.array([440.0, 0.0, 880.0])
        rmse = pitch_rmse(f0_a, f0_b, voiced_only=True)
        # Only 2 voiced frames: 0 and 1200 cents diff
        assert rmse > 0

    def test_all_unvoiced(self):
        from auto_voice.evaluation.metrics import pitch_rmse
        f0_a = np.array([0.0, 0.0])
        f0_b = np.array([0.0, 0.0])
        assert pitch_rmse(f0_a, f0_b, voiced_only=True) == 0.0

    def test_different_lengths(self):
        from auto_voice.evaluation.metrics import pitch_rmse
        f0_a = np.array([440.0, 441.0, 442.0, 443.0])
        f0_b = np.array([440.0, 441.0])
        # Should use min length
        rmse = pitch_rmse(f0_a, f0_b)
        assert rmse == 0.0


class TestSpeakerSimilarity:
    """Tests for speaker similarity metric."""

    def test_identical_embeddings(self):
        from auto_voice.evaluation.metrics import speaker_similarity
        emb = np.random.randn(256)
        assert abs(speaker_similarity(emb, emb) - 1.0) < 1e-6

    def test_opposite_embeddings(self):
        from auto_voice.evaluation.metrics import speaker_similarity
        emb = np.random.randn(256)
        sim = speaker_similarity(emb, -emb)
        assert abs(sim - (-1.0)) < 1e-6

    def test_orthogonal_embeddings(self):
        from auto_voice.evaluation.metrics import speaker_similarity
        emb_a = np.zeros(256)
        emb_a[0] = 1.0
        emb_b = np.zeros(256)
        emb_b[1] = 1.0
        assert abs(speaker_similarity(emb_a, emb_b)) < 1e-6

    def test_zero_embedding(self):
        from auto_voice.evaluation.metrics import speaker_similarity
        emb_a = np.zeros(256)
        emb_b = np.random.randn(256)
        assert speaker_similarity(emb_a, emb_b) == 0.0

    def test_range(self):
        from auto_voice.evaluation.metrics import speaker_similarity
        for _ in range(10):
            a = np.random.randn(256)
            b = np.random.randn(256)
            sim = speaker_similarity(a, b)
            assert -1.0 <= sim <= 1.0


class TestSTOI:
    """Tests for STOI metric."""

    def test_identical_signals(self):
        from auto_voice.evaluation.metrics import stoi
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        score = stoi(clean, clean, sr=16000)
        assert score > 0.9

    def test_noisy_signal(self):
        from auto_voice.evaluation.metrics import stoi
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        noisy = clean + 0.5 * np.random.randn(16000).astype(np.float32)
        score = stoi(clean, noisy, sr=16000)
        assert 0.0 <= score <= 1.0

    def test_score_range(self):
        from auto_voice.evaluation.metrics import stoi
        clean = np.random.randn(16000).astype(np.float32)
        degraded = np.random.randn(16000).astype(np.float32)
        score = stoi(clean, degraded, sr=16000)
        assert 0.0 <= score <= 1.0

    def test_empty_signal(self):
        from auto_voice.evaluation.metrics import stoi
        score = stoi(np.array([]), np.array([]), sr=16000)
        assert score == 0.0

    def test_extended_stoi(self):
        from auto_voice.evaluation.metrics import stoi
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        score = stoi(clean, clean, sr=16000, extended=True)
        assert score > 0.9


class TestPESQ:
    """Tests for PESQ metric."""

    def test_identical_signals(self):
        from auto_voice.evaluation.metrics import pesq
        ref = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        score = pesq(ref, ref, sr=16000)
        assert score > 3.0  # High quality

    def test_noisy_signal(self):
        from auto_voice.evaluation.metrics import pesq
        ref = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        noisy = ref + np.random.randn(16000).astype(np.float32)
        score = pesq(ref, noisy, sr=16000)
        assert -0.5 <= score <= 4.5

    def test_score_range(self):
        from auto_voice.evaluation.metrics import pesq
        ref = np.random.randn(16000).astype(np.float32)
        deg = np.random.randn(16000).astype(np.float32)
        score = pesq(ref, deg, sr=16000)
        assert -0.5 <= score <= 4.5


class TestSignalToNoiseRatio:
    """Tests for SNR metric."""

    def test_identical(self):
        from auto_voice.evaluation.metrics import signal_to_noise_ratio
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        snr = signal_to_noise_ratio(clean, clean)
        assert snr == 100.0  # Essentially infinite

    def test_noisy(self):
        from auto_voice.evaluation.metrics import signal_to_noise_ratio
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        noisy = clean + 0.1 * np.random.randn(16000)
        snr = signal_to_noise_ratio(clean, noisy)
        assert snr > 10.0  # Good SNR

    def test_very_noisy(self):
        from auto_voice.evaluation.metrics import signal_to_noise_ratio
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        noisy = clean + 10.0 * np.random.randn(16000)
        snr = signal_to_noise_ratio(clean, noisy)
        assert snr < 0.0  # More noise than signal


class TestMelCepstralDistortion:
    """Tests for MCD metric."""

    def test_identical_audio(self):
        from auto_voice.evaluation.metrics import mel_cepstral_distortion
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)).astype(np.float32)
        mcd = mel_cepstral_distortion(audio, audio, sr=22050)
        assert mcd < 1.0  # Should be very low

    def test_different_audio(self):
        from auto_voice.evaluation.metrics import mel_cepstral_distortion
        ref = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)).astype(np.float32)
        syn = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 22050)).astype(np.float32)
        mcd = mel_cepstral_distortion(ref, syn, sr=22050)
        assert mcd > 0.0

    def test_nonnegative(self):
        from auto_voice.evaluation.metrics import mel_cepstral_distortion
        ref = np.random.randn(22050).astype(np.float32)
        syn = np.random.randn(22050).astype(np.float32)
        mcd = mel_cepstral_distortion(ref, syn, sr=22050)
        assert mcd >= 0.0
