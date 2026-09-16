"""The decoder's period doubling must come out; the harmonics must stay in.

Synthetic: a harmonic "voice", and a "conversion" of it carrying the measured
defect - energy on the half-integer partials (f0/2, 3f0/2, 5f0/2), which is
what the belting fuzz is. f0 is 400 Hz, inside the band where the ratio was
measured at 4x the source's.

The clean-voice test is the one that matters most: this stage runs on every
render, including the frames that were never doubling.
"""
import numpy as np

from auto_voice.inference.fork_subharmonics import suppress_subharmonics

SR = 44100
F0 = 400.0


def _voice(sec=2.0, f0=F0):
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 13))
    y += 0.02 * np.random.RandomState(0).randn(len(t))        # breath
    return (0.2 * y / np.abs(y).max()).astype(np.float32)


def _band_db(y, hz, half=40.0):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return 20 * np.log10(S[(f >= hz - half) & (f < hz + half)].mean() + 1e-12)


def test_half_integer_partials_are_removed():
    src = _voice()
    t = np.arange(len(src)) / SR
    doubled = 0.1 * (np.sin(np.pi * F0 * t) + np.sin(3 * np.pi * F0 * t)
                     + np.sin(5 * np.pi * F0 * t))               # f0/2, 3f0/2, 5f0/2
    conv = (src + doubled).astype(np.float32)
    out = suppress_subharmonics(src, conv, SR)
    for hz in (F0 / 2, 1.5 * F0, 2.5 * F0):
        drop = _band_db(conv, hz) - _band_db(out, hz)
        assert drop >= 6.0, f"{hz:.0f} Hz subharmonic only dropped {drop:.1f} dB"


def test_harmonics_survive_the_notching():
    """A broad gain reduction would pass the test above. The singer would not."""
    src = _voice()
    t = np.arange(len(src)) / SR
    conv = (src + 0.1 * np.sin(3 * np.pi * F0 * t)).astype(np.float32)
    out = suppress_subharmonics(src, conv, SR)
    for k in range(1, 6):
        moved = abs(_band_db(conv, F0 * k) - _band_db(out, F0 * k))
        assert moved < 1.0, f"harmonic {k} moved {moved:.2f} dB"


def test_clean_voice_is_left_alone():
    """Runs on every render - a frame that was never doubling must come back out."""
    src = _voice()
    out = suppress_subharmonics(src, src, SR)
    residual = 20 * np.log10(np.sqrt(np.mean((out - src) ** 2))
                             / np.sqrt(np.mean(src ** 2)))
    assert residual < -20.0, f"clean voice altered by {residual:.1f} dB"
    for k in range(1, 6):
        moved = abs(_band_db(src, F0 * k) - _band_db(out, F0 * k))
        assert moved < 1.0, f"harmonic {k} moved {moved:.2f} dB"


def test_short_input_passes_through():
    y = np.zeros(128, dtype=np.float32)                          # under n_fft
    out = suppress_subharmonics(y, y, SR)
    assert out.shape == (128,) and out.dtype == np.float32


def test_output_contract():
    src = _voice(sec=1.0)
    t = np.arange(len(src)) / SR
    conv = (src + 0.1 * np.sin(3 * np.pi * F0 * t)).astype(np.float32)
    out = suppress_subharmonics(src, conv, SR)
    assert out.shape == conv.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_silent_source_is_a_no_op():
    """yin reports a confident ~1000 Hz on silence. Without a voicing gate that
    placed notches on a pitch nobody sang and cut the conversion by -10.8 dB
    wherever the source was quiet."""
    t = np.arange(SR * 2) / SR
    voice = (0.2 * sum(np.sin(2 * np.pi * 300 * k * t) / k for k in range(1, 20))).astype(np.float32)
    out = suppress_subharmonics(np.zeros(SR * 2, np.float32), voice, SR)
    resid = 20 * np.log10(np.sqrt(np.mean((out - voice) ** 2)) / np.sqrt(np.mean(voice ** 2)) + 1e-12)
    assert resid < -60.0, f"silent source changed the conversion by {resid:.1f} dB"
