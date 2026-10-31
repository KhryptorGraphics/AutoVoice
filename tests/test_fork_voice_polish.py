"""The decoder's colouring must come out; the singer's identity must stay in.

Synthetic: a harmonic "voice", and a "conversion" of it that has the decoder's
two measured defects - a scooped consonant band, and a fixed tone up top.
"""
import numpy as np
import pytest

from auto_voice.inference.fork_voice_polish import polish_voice

SR = 44100


def _voice(sec=4.0, f0=180.0):
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 60))
    y += 0.05 * np.random.RandomState(0).randn(len(t))       # breath
    return (0.2 * y / np.abs(y).max()).astype(np.float32)


def _band_db(y, lo, hi):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return 20 * np.log10(S[(f >= lo) & (f < hi)].mean() + 1e-12)


def _shelf(y, lo, hi, db):
    """Crude band gain, used to fake the decoder's scoop."""
    S = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    S[(f >= lo) & (f < hi)] *= 10 ** (db / 20)
    return np.fft.irfft(S, n=len(y)).astype(np.float32)


def test_scooped_consonant_band_is_restored():
    src = _voice()
    conv = _shelf(src, 1500, 3000, -6.0)                     # the decoder's scoop
    before = _band_db(conv, 1500, 3000) - _band_db(src, 1500, 3000)
    out = polish_voice(src, conv, SR)
    after = _band_db(out, 1500, 3000) - _band_db(src, 1500, 3000)
    assert after - before >= 2.0, f"scoop closed by only {after - before:.1f} dB"


def test_fixed_high_tone_is_notched():
    src = _voice()
    t = np.arange(len(src)) / SR
    conv = (src + 0.02 * np.sin(2 * np.pi * 8900 * t)).astype(np.float32)
    out = polish_voice(src, conv, SR)
    drop = _band_db(conv, 8800, 9000) - _band_db(out, 8800, 9000)
    assert drop >= 3.0, f"resonance only dropped {drop:.1f} dB"


def test_identity_band_above_5k_is_not_levelled_to_source():
    """The decoder is darker than the source up top on purpose - leave it."""
    src = _voice()
    conv = _shelf(src, 6000, 12000, -10.0)                   # her air, not his
    out = polish_voice(src, conv, SR)
    assert _band_db(out, 6000, 12000) - _band_db(conv, 6000, 12000) < 1.5


def test_short_input_passes_through():
    y = np.zeros(128, dtype=np.float32)
    assert polish_voice(y, y, SR).shape == (128,)


class TestDetonalizeHF:
    """The 'double buzzing around the voice': narrowband tones above ContentVec's
    8 kHz Nyquist, which the LTAS notcher structurally cannot catch."""

    def _amp(self, y, sr, f0):
        import librosa
        S = np.abs(librosa.stft(y, n_fft=8192))
        fr = librosa.fft_frequencies(sr=sr, n_fft=8192)
        m = (fr > f0 * 0.98) & (fr < f0 * 1.02)
        return 20 * np.log10(S[m].max() + 1e-12)

    def test_cuts_a_tone_above_the_threshold(self):
        from auto_voice.inference.fork_voice_polish import detonalize_hf
        sr = 44100
        t = np.arange(sr * 2) / sr
        y = (0.3 * np.sin(2 * np.pi * 300 * t)
             + 0.2 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32)
        out = detonalize_hf(y, sr)
        assert self._amp(y, sr, 9000) - self._amp(out, sr, 9000) > 6.0

    def test_leaves_the_voice_band_untouched(self):
        """The whole point is that it is surgical: below `above_hz` is masked
        off, so this is exact rather than merely small."""
        from auto_voice.inference.fork_voice_polish import detonalize_hf
        sr = 44100
        t = np.arange(sr * 2) / sr
        y = (0.3 * np.sin(2 * np.pi * 300 * t)
             + 0.2 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32)
        out = detonalize_hf(y, sr)
        assert abs(self._amp(y, sr, 300) - self._amp(out, sr, 300)) < 0.5

    def test_silence_is_a_no_op(self):
        from auto_voice.inference.fork_voice_polish import detonalize_hf
        sr = 44100
        out = detonalize_hf(np.zeros(sr, dtype=np.float32), sr)
        assert np.abs(out).max() == 0.0

    def test_broadband_noise_is_not_gutted(self):
        """Real breath up there is broadband; only narrow peaks are invention."""
        from auto_voice.inference.fork_voice_polish import detonalize_hf
        sr = 44100
        rng = np.random.default_rng(0)
        y = (rng.standard_normal(sr * 2) * 0.05).astype(np.float32)
        out = detonalize_hf(y, sr)
        before = float(np.sqrt(np.mean(y ** 2)))
        after = float(np.sqrt(np.mean(out ** 2)))
        assert 20 * np.log10(after / before) > -1.5


class TestRestoreAir:
    """Her corpus is 17 dB darker than the source at 8-12 kHz, so the model
    learned a dark voice. Corrected against the SOURCE, smoothed and bounded -
    a fixed shelf was tried and is the wrong shape."""

    def _mk(self, sr, hf_gain):
        t = np.arange(sr * 2) / sr
        rng = np.random.default_rng(0)
        base = 0.2 * np.sin(2 * np.pi * 400 * t)
        hf = 0.02 * hf_gain * rng.standard_normal(len(t))
        b = np.fft.rfft(hf); fr = np.fft.rfftfreq(len(t), 1 / sr)
        b[fr < 6000] = 0                      # noise only up high
        return (base + np.fft.irfft(b, len(t))).astype(np.float32)

    def _band(self, y, sr, lo, hi):
        import librosa
        S = np.abs(librosa.stft(y, n_fft=4096))
        fr = librosa.fft_frequencies(sr=sr, n_fft=4096)
        m = (fr >= lo) & (fr < hi)
        return 20 * np.log10(np.median(S[m]) + 1e-12)

    def test_lifts_a_dark_render_toward_the_source(self):
        from auto_voice.inference.fork_voice_polish import restore_air
        sr = 44100
        source = self._mk(sr, 1.0)
        dark = self._mk(sr, 0.1)              # 20 dB less air
        out = restore_air(source, dark, sr, cap_db=12.0)
        before = self._band(dark, sr, 8000, 12000)
        after = self._band(out, sr, 8000, 12000)
        assert after - before > 3.0

    def test_never_cuts(self):
        """One-sided: a render already brighter than the source must be left
        alone, so it can never fight detonalize_hf or re-darken anything."""
        from auto_voice.inference.fork_voice_polish import restore_air
        sr = 44100
        source = self._mk(sr, 0.1)
        bright = self._mk(sr, 1.0)
        out = restore_air(source, bright, sr, cap_db=12.0)
        assert self._band(out, sr, 8000, 12000) - self._band(bright, sr, 8000, 12000) < 0.5

    def test_leaves_the_formant_region_alone(self):
        """polish_voice already corrects 1.2-5 kHz against the source; this
        must not touch it a second time."""
        from auto_voice.inference.fork_voice_polish import restore_air
        sr = 44100
        source = self._mk(sr, 1.0); dark = self._mk(sr, 0.1)
        out = restore_air(source, dark, sr, cap_db=12.0)
        assert abs(self._band(out, sr, 300, 2000) - self._band(dark, sr, 300, 2000)) < 1.0

    def test_zero_cap_is_a_no_op(self):
        from auto_voice.inference.fork_voice_polish import restore_air
        sr = 44100
        s, d = self._mk(sr, 1.0), self._mk(sr, 0.1)
        assert np.array_equal(restore_air(s, d, sr, cap_db=0.0), d)

    def test_silence_stays_silent(self):
        from auto_voice.inference.fork_voice_polish import restore_air
        sr = 44100
        z = np.zeros(sr, dtype=np.float32)
        assert np.abs(restore_air(self._mk(sr, 1.0), z, sr, cap_db=12.0)).max() == 0.0


class TestEffectiveFmax:
    """The wall detector - where a band-limited capture stops.

    It looks for a CLIFF (a steep drop in dB per octave above 3 kHz), not
    an absolute threshold, because anything that lifts the noise floor
    defeats an absolute threshold. A signal with no cliff reads as
    full-band, which is the safe answer: callers treat "not band-limited"
    as "do nothing".
    """

    SR = 48000

    def _band_limited(self, seconds, cutoff, sr=None):
        """Harmonics plus broadband noise below the cutoff, nothing above.

        The broadband part matters: a signal carrying only low harmonics
        measures nothing in the probe bands above 2.5 kHz, so there is no
        drop to find and the detector reports full-band. A real voice has
        content across the spectrum, so the test signal must too.
        """
        import librosa
        sr = sr or self.SR
        n = int(seconds * sr)
        t = np.arange(n) / sr
        y = sum(np.sin(2 * np.pi * f * t) for f in (220, 440, 660, 1200))
        y = y / np.abs(y).max() * 0.4
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(n).astype(np.float32)
        S = librosa.stft(noise, n_fft=8192, hop_length=2048)
        f = librosa.fft_frequencies(sr=sr, n_fft=8192)
        S[f >= cutoff * 0.8] = 0
        y = y + librosa.istft(S, hop_length=2048, length=n) * 0.5
        # Brickwall the whole thing: a sharp drop, not a gradual slope.
        S2 = librosa.stft(y.astype(np.float32), n_fft=8192, hop_length=2048)
        S2[f >= cutoff] = 0
        return librosa.istft(S2, hop_length=2048, length=n).astype(np.float32)

    def _full_band(self, seconds, sr=None):
        sr = sr or self.SR
        n = int(seconds * sr)
        t = np.arange(n) / sr
        return (0.3 * np.sin(2 * np.pi * 220 * t)
                + 0.2 * np.sin(2 * np.pi * 3000 * t)
                + 0.1 * np.sin(2 * np.pi * 9000 * t)
                + 0.05 * np.sin(2 * np.pi * 15000 * t)).astype(np.float32)

    def test_finds_the_wall_of_a_band_limited_signal(self):
        """A brickwall at 7 kHz reads as a wall well below Nyquist."""
        from auto_voice.inference.fork_voice_polish import effective_fmax
        wall = effective_fmax(self._band_limited(2.0, 7000.0), self.SR)
        assert 3000.0 < wall < self.SR / 2, f"wall {wall:.0f} Hz"

    def test_the_wall_tracks_the_cutoff(self):
        """A lower cutoff must read as a lower wall."""
        from auto_voice.inference.fork_voice_polish import effective_fmax
        low = effective_fmax(self._band_limited(2.0, 6000.0), self.SR)
        high = effective_fmax(self._band_limited(2.0, 7000.0), self.SR)
        assert low < high, f"{low:.0f} Hz should be below {high:.0f} Hz"

    def test_the_wall_is_near_the_cutoff(self):
        """The reported wall is within a factor of the true cutoff.

        The detector reports the probe frequency it found the drop at, not
        the cutoff itself, so the value is near but not equal to it - the
        bound has to be a band, not equality.
        """
        from auto_voice.inference.fork_voice_polish import effective_fmax
        cutoff = 7000.0
        wall = effective_fmax(self._band_limited(2.0, cutoff), self.SR)
        assert 0.7 * cutoff <= wall <= 1.15 * cutoff, f"wall {wall:.0f} for cutoff {cutoff:.0f}"

    def test_a_full_band_signal_reads_as_full_band(self):
        """No cliff means no repair is needed, so Nyquist is returned."""
        from auto_voice.inference.fork_voice_polish import effective_fmax
        assert effective_fmax(self._full_band(2.0), self.SR) == pytest.approx(self.SR / 2)

    def test_silence_reads_as_full_band(self):
        """Silence has no cliff, so it is not treated as band-limited."""
        from auto_voice.inference.fork_voice_polish import effective_fmax
        z = np.zeros(self.SR, dtype=np.float32)
        assert effective_fmax(z, self.SR) == pytest.approx(self.SR / 2)


class TestExtendBandwidth:
    """Synthesising the band a band-limited capture never recorded."""

    SR = 48000

    def _band_limited(self, seconds, cutoff, sr=None):
        return TestEffectiveFmax()._band_limited(seconds, cutoff, sr)

    def _full_band(self, seconds, sr=None):
        return TestEffectiveFmax()._full_band(seconds, sr)

    def _band_energy(self, y, lo, hi, sr=None):
        import librosa
        sr = sr or self.SR
        S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
        f = librosa.fft_frequencies(sr=sr, n_fft=4096)
        m = (f >= lo) & (f < hi)
        return float((S[m] ** 2).sum())

    def test_a_full_band_render_is_left_alone(self):
        """The stage is a repair for damaged capture, not a brightener.

        The early return hands back the SAME object, so a caller can tell
        nothing was processed.
        """
        from auto_voice.inference.fork_voice_polish import extend_bandwidth
        y = self._full_band(2.0)
        assert extend_bandwidth(y, self.SR) is y

    def test_a_band_limited_render_gains_energy_above_the_wall(self):
        """The band above the wall is filled."""
        from auto_voice.inference.fork_voice_polish import effective_fmax, extend_bandwidth
        y = self._band_limited(2.0, 7000.0)
        wall = effective_fmax(y, self.SR)
        out = extend_bandwidth(y, self.SR)
        assert out is not y, "a band-limited signal must be processed"
        assert len(out) == len(y)
        assert out.dtype == np.float32
        before = self._band_energy(y, wall * 1.2, 16000, self.SR)
        after = self._band_energy(out, wall * 1.2, 16000, self.SR)
        assert after > before * 10, f"energy above wall only {after / max(before, 1e-12):.1f}x"

    def test_the_extension_is_deterministic(self):
        """The noise mix uses a seeded generator, so two runs match."""
        from auto_voice.inference.fork_voice_polish import extend_bandwidth
        y = self._band_limited(2.0, 7000.0)
        a = extend_bandwidth(y, self.SR)
        b = extend_bandwidth(y, self.SR)
        assert np.array_equal(a, b)

    def test_a_wall_below_3_khz_is_skipped(self):
        """Below 3 kHz the signal is not a capture wall, it is a bad signal."""
        from auto_voice.inference.fork_voice_polish import extend_bandwidth
        y = self._band_limited(2.0, 7000.0)
        assert extend_bandwidth(y, self.SR, wall_hz=2000.0) is y

    def test_a_wall_above_12_khz_is_skipped(self):
        """Above 12 kHz the capture was real, so no band is invented."""
        from auto_voice.inference.fork_voice_polish import extend_bandwidth
        y = self._band_limited(2.0, 7000.0)
        assert extend_bandwidth(y, self.SR, wall_hz=14000.0) is y

    def test_a_short_or_empty_signal_is_left_alone(self):
        """Too short to transform, or silent, is returned unchanged."""
        from auto_voice.inference.fork_voice_polish import extend_bandwidth
        short = np.zeros(64, dtype=np.float32)
        assert extend_bandwidth(short, self.SR) is short
        silent = np.zeros(self.SR, dtype=np.float32)
        assert extend_bandwidth(silent, self.SR) is silent
