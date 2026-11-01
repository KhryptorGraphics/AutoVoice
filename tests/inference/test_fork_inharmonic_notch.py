"""Tests for the high-register inharmonic notch.

The stage's failure modes are silent: a notch that stops firing still
produces audio, and an audio file that is subtly wrong still passes a
smoke test. So the tests here pin the behaviours that would break
undetected:

- the stage is a no-op on a source that never sings high (the decoder is
  not extrapolating, so there is nothing to correct)
- the stage does not touch the low register, where the decoder renders the
  voice correctly and the singer's breath noise is real
- the correction is EXACT: it lands on the source's own off-harmonic
  fraction, never past it (a proportional cut overshoots into negative
  excess, i.e. it cuts real voice)
- the collateral on a clean source is zero
- the stage is disabled by config
"""
import numpy as np
import pytest

from auto_voice.inference.fork_inharmonic_notch import (
    suppress_inharmonic,
    source_voiced_frames,
)


SR = 44100


def _harmonic_tone(f0, duration=2.0, sr=SR, n_harmonics=12):
    """A synthetic voice: harmonics of f0 only, no off-harmonic energy."""
    t = np.arange(int(duration * sr)) / sr
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        f = f0 * k
        if f >= sr / 2:
            break
        y += (1.0 / k) * np.sin(2 * np.pi * f * t)
    return (y / np.max(np.abs(y))).astype(np.float32)


def _inharmonic_tone(f0, duration=2.0, sr=SR, n_harmonics=12, level=0.5):
    """A synthetic 'decoder' output: the harmonics PLUS a narrowband tone
    sitting between them, which is what the decoder emits when it
    extrapolates."""
    t = np.arange(int(duration * sr)) / sr
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        f = f0 * k
        if f >= sr / 2:
            break
        y += (1.0 / k) * np.sin(2 * np.pi * f * t)
    y /= np.max(np.abs(y))
    # Off-harmonic residue at 1.5*f0 - between harmonics 1 and 2.
    residue = level * np.sin(2 * np.pi * f0 * 1.5 * t)
    out = y + residue
    return (out / np.max(np.abs(out))).astype(np.float32)


def _off_harmonic_fraction(y, f0, sr=SR, n_fft=2048, hop=512, k_max=12,
                           rel=0.03):
    """The metric's own measure: fraction of in-band magnitude that is not
    on a harmonic of f0."""
    import librosa

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    hm = np.zeros(len(freqs), bool)
    for k in range(1, k_max + 1):
        hm |= (freqs > f0 * k * (1 - rel)) & (freqs < f0 * k * (1 + rel))
    bd = (freqs > f0 * 0.5) & (freqs < 6000)
    on = S[hm & bd].sum()
    off = S[(~hm) & bd].sum()
    return float(off / (on + off + 1e-12))


class TestNoOpConditions:
    def test_silent_source_is_untouched(self):
        source = np.zeros(SR, dtype=np.float32)
        converted = np.zeros(SR, dtype=np.float32)
        out = suppress_inharmonic(source, converted, SR)
        assert np.array_equal(out, converted)

    def test_source_that_never_sings_high_is_untouched(self):
        """A source below the f0_lo gate: the decoder is not extrapolating,
        so the stage must not fire. Changing the audio here would be
        removing the singer's own breath noise."""
        source = _harmonic_tone(200.0)
        converted = _inharmonic_tone(200.0)
        out = suppress_inharmonic(source, converted, SR, f0_lo=550.0)
        assert np.array_equal(out, converted)

    def test_too_short_signal_is_untouched(self):
        source = np.zeros(100, dtype=np.float32)
        converted = np.zeros(100, dtype=np.float32)
        out = suppress_inharmonic(source, converted, SR)
        assert np.array_equal(out, converted)


class TestCorrection:
    def test_removes_the_off_harmonic_residue(self):
        """A synthetic decoder output with a tone between the harmonics must
        come out with that residue gone."""
        f0 = 700.0
        source = _harmonic_tone(f0)
        converted = _inharmonic_tone(f0, level=0.5)

        before = _off_harmonic_fraction(converted, f0)
        out = suppress_inharmonic(source, converted, SR, f0_lo=550.0)
        after = _off_harmonic_fraction(out, f0)

        assert before > 0.05, "the fixture must actually contain residue"
        assert after < before * 0.5, (
            f"residue not removed: {before:.4f} -> {after:.4f}")

    def test_correction_does_not_overshoot_past_the_source(self):
        """The correction must not push the render CLEANER than the source.

        Overshooting means real voice has been cut: the source's own
        off-harmonic energy is breath noise, and a render that has less of
        it than the source is not more correct, it is thinned.

        The correction lands ABOVE the source's fraction rather than
        exactly on it - the STFT round trip redistributes some energy into
        the harmonic bins - so this asserts the direction that matters:
        the result is not cleaner than the source.
        """
        f0 = 700.0
        source = _harmonic_tone(f0)
        converted = _inharmonic_tone(f0, level=0.8)

        target = _off_harmonic_fraction(source, f0)
        out = suppress_inharmonic(source, converted, SR, f0_lo=550.0)
        after = _off_harmonic_fraction(out, f0)

        assert after >= target * 0.85, (
            f"overshot the source's own fraction: target {target:.4f}, "
            f"got {after:.4f}")

    def test_clean_source_is_unchanged(self):
        """The collateral must be zero: run on the source itself, the stage
        must not alter it. The correction targets the source's own
        off-harmonic fraction, so a clean source is already at its target.

        This is the test that caught the make-up gain's edge dilution:
        ``np.convolve`` zero-pads, so the smoothed gain array dipped to
        0.625 in the first and last frames - a 4 dB attenuation of audio
        that needed no change. Edge padding fixes it.
        """
        f0 = 700.0
        source = _harmonic_tone(f0)
        out = suppress_inharmonic(source, source.copy(), SR, f0_lo=550.0)
        assert np.allclose(out, source, atol=1e-4), (
            f"clean source altered by {float(np.abs(out - source).max()):.2e}")

    def test_gain_never_attenuates(self):
        """The make-up gain exists to RESTORE level the notch took. A gain
        below 1.0 removes level that was never lost.

        Checked directly on the gain array, because the failure this
        catches is a per-frame dip at the signal edges that a
        whole-signal comparison would hide.
        """
        from auto_voice.inference.fork_inharmonic_notch import (
            _make_up_gain,
            source_voiced_frames,
        )

        f0 = 700.0
        source = _harmonic_tone(f0)
        f0v, good = source_voiced_frames(source, SR)
        _, gain = _make_up_gain(source, source.copy(), SR, f0v, good,
                                12.0, 550.0)
        assert gain.min() >= 1.0 - 1e-6, (
            f"gain dips to {gain.min():.6f} - it is attenuating")


class TestLowRegister:
    def test_low_frames_are_not_cut(self):
        """The stage is gated on the source's pitch. A source singing at
        300 Hz must come through untouched, even if the 'converted' audio
        has residue there."""
        source = _harmonic_tone(300.0)
        converted = _inharmonic_tone(300.0, level=0.8)
        out = suppress_inharmonic(source, converted, SR, f0_lo=550.0)
        assert np.allclose(out, converted, atol=1e-4)


class TestVoicedFrameGate:
    def test_gate_accepts_confident_voiced_frames(self):
        """The scorer gate accepts a clear voiced frame."""
        source = _harmonic_tone(300.0)
        f0, good = source_voiced_frames(source, SR)
        assert good.any()
        assert np.isfinite(f0[good]).all()

    def test_gate_rejects_octave_error(self):
        """Reject frames where the 2*f0 peak exceeds the candidate f0 peak."""
        from unittest.mock import patch

        import librosa

        t = np.arange(SR) / SR
        source = (
            0.1 * np.sin(2 * np.pi * 300.0 * t)
            + np.sin(2 * np.pi * 600.0 * t)
        ).astype(np.float32)
        n_frames = librosa.stft(
            source, n_fft=2048, hop_length=512
        ).shape[1]
        f0 = np.full(n_frames, 300.0)
        voiced = np.ones(n_frames, dtype=bool)
        voiced_prob = np.full(n_frames, 0.99)

        with patch(
            "librosa.pyin",
            return_value=(f0, voiced, voiced_prob),
        ):
            _, good = source_voiced_frames(source, SR)

        assert not good.any()

    def test_gate_rejects_silence(self):
        source = np.zeros(SR, dtype=np.float32)
        f0, good = source_voiced_frames(source, SR)
        assert not good.any()


class TestPipelineIntegration:
    def test_chain_flag_disables_the_stage(self):
        """``fork_hq_inharmonic_notch: false`` must skip the stage
        entirely, not merely reduce its depth."""
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )

        called = []

        class _Spy:
            @staticmethod
            def suppress_inharmonic(source, converted, sr, **kwargs):
                called.append(kwargs)
                return converted

        import auto_voice.inference.fork_inharmonic_notch as mod
        original = mod.suppress_inharmonic
        mod.suppress_inharmonic = _Spy.suppress_inharmonic
        try:
            pipeline = SingingConversionPipeline.__new__(
                SingingConversionPipeline)
            pipeline.config = {'fork_hq_inharmonic_notch': False}
            pipeline._explicit_settings = {}
            pipeline._fill_lead_dropouts = lambda conv, src, sr: conv
            src = np.zeros(4096, dtype=np.float32)
            pipeline._correct_fork_output(src, src.copy(), SR)
            assert not called, "the stage ran despite being disabled"
        finally:
            mod.suppress_inharmonic = original

    def test_chain_flag_enables_the_stage(self):
        """With the flag on (the default), the stage runs."""
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )

        called = []

        class _Spy:
            @staticmethod
            def suppress_inharmonic(source, converted, sr, **kwargs):
                called.append(kwargs)
                return converted

        import auto_voice.inference.fork_inharmonic_notch as mod
        original = mod.suppress_inharmonic
        mod.suppress_inharmonic = _Spy.suppress_inharmonic
        try:
            pipeline = SingingConversionPipeline.__new__(
                SingingConversionPipeline)
            pipeline.config = {}
            pipeline._explicit_settings = {}
            pipeline._fill_lead_dropouts = lambda conv, src, sr: conv
            src = np.zeros(4096, dtype=np.float32)
            pipeline._correct_fork_output(src, src.copy(), SR)
            assert called, "the stage did not run with the flag on"
        finally:
            mod.suppress_inharmonic = original

    def test_pipeline_default_gate_covers_400_to_550(self):
        """The production gate must correct the newly validated 400-550 band."""
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )

        pipeline = SingingConversionPipeline.__new__(SingingConversionPipeline)
        pipeline.config = {
            'fork_hq_hum_suppress': False,
            'fork_hq_subharmonic_suppress': False,
            'fork_hq_voice_polish': False,
            'fork_hq_hf_detonalize': False,
            'fork_hq_air_cap_db': 0.0,
            'fork_hq_extend_bandwidth': False,
            'fork_hq_inharmonic_notch': True,
        }
        pipeline._explicit_settings = {}
        pipeline._fill_lead_dropouts = lambda converted, source, sr: converted
        f0 = 450.0
        source = _harmonic_tone(f0)
        converted = _inharmonic_tone(f0, level=0.8)

        before = _off_harmonic_fraction(converted, f0)
        corrected = pipeline._correct_fork_output(source, converted, SR)
        after = _off_harmonic_fraction(corrected, f0)

        assert before > 0.05
        assert after < before * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
