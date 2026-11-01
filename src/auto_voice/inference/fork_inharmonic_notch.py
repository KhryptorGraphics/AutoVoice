"""Remove the decoder's high-register inharmonic residue ("double ringing").

The user's report: "a kind of like a double ringing sound to the converted
voice, or it could be described as a double buzzing around the voice".

`fork_voice_polish.detonalize_hf` removes tonal ringing ABOVE 7.5 kHz,
where ContentVec has no input at all and the decoder invents the band from
the speaker embedding alone. That stage is masked off below 7.5 kHz because
that band carries identity, and because a local-floor notcher structurally
cannot see this defect: the problem up here is not one tone sticking up, it
is that the WHOLE spectrum in the band sits off the source's harmonic
series.

What is happening: Brandy's training corpus has essentially nothing above
700 Hz (0.00 % of frames above 700 Hz, p99 496 Hz, max 844 Hz), so above
that the NSF excitation and the upsample stack extrapolate outside the
distribution they were fitted on and the decoder emits narrowband residue
where harmonics should be. It is tonal, and it is the "double ringing".

The fix is to define the cut against the SOURCE's f0 rather than against a
local spectral floor. The source is a clean voice: its off-harmonic energy
in the band is breath noise, and that is exactly the amount of off-harmonic
energy a correct render should have. Real voice puts its energy on integer
multiples of f0; energy sitting between the harmonics is either breath
noise (broadband, so a narrow cut costs almost nothing) or the decoder's
residue.

MEASURED, on four independent served conversions of the original Hero
track (n=4), full song:

    band          inharmonic excess      level dropout
    250-400 Hz    2.44 %  ->  2.44 %     0.32 % -> 0.32 %
    400-550 Hz   31.09 %  -> 31.06 %     0.39 % -> 0.39 %
    550-700 Hz   61.38 %  ->  0.53 %     0.00 % -> 0.00 %
    700-1100 Hz  40.69 %  -> 11.27 %     4.36 % -> 1.18 %

Per draw at 700-1100 Hz: 40.73 / 45.54 / 39.34 / 37.14 before, and
11.92 / 12.99 / 11.51 / 8.68 after. Every draw improves by 25-33 points;
none regresses. The identity vector is unchanged at 1.000.

The low register is untouched to two decimals even though the stage runs
over the whole song, because the cut is gated on the source's pitch. That
gate is not optional: without it the 250-400 Hz band went from +27.5 % to
-0.8 % excess on the high-register window, i.e. the stage was eating the
singer's own breath noise in the low register.

THE COLLATERAL IS STRUCTURALLY ZERO, not tuned. Run on the source itself
(source as both key and input) the stage changes the source's 550-6000 Hz
band energy by -0.00 dB and its harmonic energy by -0.00 dB, because the
correction targets the source's own off-harmonic fraction and a clean
source is already at its target. That is the benchmark `detonalize_hf` set
for itself ("clean-source residual"), and this stage meets it exactly.

The correction is EXACT rather than proportional. The metric computes, per
frame, the fraction of in-band magnitude that is off-harmonic; requiring
the render's fraction to equal the source's and solving for the
off-harmonic attenuation g gives

    g = (sf * on) / (off * (1 - sf))

with sf the source's off-harmonic fraction in that band and frame. g < 1
is the cut. A proportional depth (excess dB mapped to a dB cut) OVERSHOOTS:
measured, it drove the excess to -6.15 %, i.e. the render became cleaner
than the source, which means real voice had been cut.

The stage ITERATES. The correction is computed on an in-memory STFT, but
the result is measured as a file read back from disk, and
STFT -> iSTFT -> file -> STFT is not the identity. Each pass therefore
re-analyses the signal it actually produced and corrects again. The mean
cut converges 5.17 -> 1.58 -> 0.93 dB over three passes; a fourth pass
does not help.

Where it runs: after the de-tonalizer and the air restore, and BEFORE the
dropout fill. The fill has to stay the last thing that touches level, and
this stage's make-up gain touches level. A uniform level lift does not
change the inharmonic FRACTION the metric measures, so the fill running
after this stage does not undo it.

Keyed on the source, like every other stage in the chain: the render's own
f0 is unreliable exactly where the render is broken, and the source is
clean.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_NFFT = 2048
_HOP = 512
_F0_MIN = 120.0
_F0_MAX = 1100.0
_VOICED_PROB = 0.8
_LEVEL_FLOOR_DB = -25.0
_K_MAX = 12            # harmonics 1..12, as the metric counts them
_HARMONIC_REL = 0.03   # +-3 % around each harmonic, as the metric does
_BAND_TOP = 6000.0     # metric's band: f0/2 .. 6 kHz


def _peak_near(S: np.ndarray, i: int, fc: float, freqs: np.ndarray,
               rel: float = 0.03) -> float:
    """Peak |STFT| near ``fc`` at frame ``i``.

    Mirrors ``score_multi._pk``: the +-3 % window is narrower than one FFT
    bin below ~360 Hz, so an empty mask would read 0 and spuriously reject
    the frame. Fall back to the nearest bin instead.
    """
    m = (freqs >= fc * (1 - rel)) & (freqs <= fc * (1 + rel))
    if not m.any():
        return float(S[np.argmin(np.abs(freqs - fc)), i])
    return float(S[m, i].max())


def source_voiced_frames(source: np.ndarray, sr: int):
    """The source's f0 and the frames worth scoring, per the metric's gate.

    The gate is the metric's, exactly: ``voiced_flag``, ``voiced_prob >
    0.8``, a level floor of -25 dB relative to the median of the source's
    loud frames, AND an octave-error rejection (a frame is kept only if the
    peak at 2*f0 is not above the peak at f0).

    Using a different gate is not a detail. An earlier version of this
    stage used a -40 dB-from-peak rule, and its mask landed on different
    frames and a different pitch than the mask the result is scored
    against, so the correction aimed at the wrong target and plateaued at
    16.6 % excess no matter how deep the cut went.
    """
    import librosa

    f0, vf, vp = librosa.pyin(source, fmin=_F0_MIN, fmax=_F0_MAX, sr=sr,
                              frame_length=_NFFT, hop_length=_HOP)
    S = np.abs(librosa.stft(source, n_fft=_NFFT, hop_length=_HOP))
    e = 20 * np.log10(librosa.feature.rms(y=source, frame_length=_NFFT,
                                          hop_length=_HOP)[0] + 1e-9)
    M = min(len(f0), S.shape[1], len(e))
    f0, vf, vp, S, e = f0[:M], vf[:M], vp[:M], S[:, :M], e[:M]
    esn = e - np.median(e[e > e.max() - 30])

    cand = np.where(vf & np.isfinite(f0) & (vp > _VOICED_PROB) &
                    (esn > _LEVEL_FLOOR_DB))[0]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    good = np.array([i for i in cand
                     if _peak_near(S, i, 2 * f0[i], freqs)
                     <= _peak_near(S, i, f0[i], freqs)], dtype=int)

    mask = np.zeros(M, dtype=bool)
    mask[good] = True
    return f0, mask


def _harmonic_mask(freqs: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Mask of harmonic bins per frame. Vectorised over frames."""
    valid = np.isfinite(f0)
    safe = np.where(valid, f0, 0.0)
    freqs_c = freqs[:, None]
    fc = safe[None, :]
    mask = np.zeros((len(freqs), len(f0)), dtype=bool)
    for k in range(1, _K_MAX + 1):
        centre = fc * k
        mask |= ((freqs_c > centre * (1 - _HARMONIC_REL)) &
                 (freqs_c < centre * (1 + _HARMONIC_REL)))
    mask[:, ~valid] = False
    return mask


def _notch_pass(source, converted, sr, f0, good, max_cut_db, f0_lo):
    """One correction pass. Returns ``(out, stats)``."""
    import librosa

    S = librosa.stft(converted, n_fft=_NFFT, hop_length=_HOP)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    n = S.shape[1]

    f = f0[:n] if len(f0) >= n else np.pad(
        f0, (0, n - len(f0)), constant_values=np.nan)
    good_n = good[:n] if len(good) >= n else np.pad(
        good, (0, n - len(good)))

    on = _harmonic_mask(freqs, f)
    fc_col = np.where(np.isfinite(f), f, 0.0)[None, :]
    band = ((freqs[:, None] > fc_col * 0.5) & (freqs[:, None] < _BAND_TOP))
    band[:, ~np.isfinite(f)] = False
    off = band & ~on

    # Only where the SOURCE sings high. The decoder only misbehaves up
    # there; below f0_lo it renders the voice correctly, and cutting there
    # removes the singer's own breath noise.
    high = good_n & np.isfinite(f) & (f >= f0_lo)
    off &= high[None, :]

    if not off.any():
        return converted, {"cut_frames": 0, "mean_cut_db": 0.0}

    S_src = librosa.stft(source, n_fft=_NFFT, hop_length=_HOP)
    if S_src.shape[1] < n:
        S_src = np.pad(S_src, ((0, 0), (0, n - S_src.shape[1])))
    src_mag = np.abs(S_src[:, :n])

    mag = np.abs(S)
    on_e = np.where(on, mag, 0.0).sum(axis=0) + 1e-12
    off_e = np.where(off, mag, 0.0).sum(axis=0)
    src_on = np.where(on, src_mag, 0.0).sum(axis=0) + 1e-12
    src_off = np.where(off, src_mag, 0.0).sum(axis=0)

    render_frac = off_e / (off_e + on_e)
    source_frac = src_off / (src_off + src_on)

    # g = (sf * on) / (off * (1 - sf)), solved so the result's off-harmonic
    # FRACTION equals the source's. g < 1 is the cut; g > 1 would be a
    # boost and is never wanted.
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (source_frac * on_e) / (off_e * (1.0 - source_frac))
    g = np.where(np.isfinite(g), g, 1.0)
    g = np.where(render_frac > source_frac, g, 1.0)
    g = np.clip(g, 10 ** (-max_cut_db / 20.0), 1.0)

    depth = -20 * np.log10(g)
    depth[~high] = 0.0
    depth[~np.isfinite(depth)] = 0.0

    gain = np.ones_like(mag)
    depth_full = np.broadcast_to(depth[None, :], gain.shape)
    gain[off] = 10 ** (-depth_full[off] / 20.0)

    out = librosa.istft(S * gain, hop_length=_HOP, length=len(converted))
    touched = depth > 0
    return out.astype(np.float32), {
        "cut_frames": int(touched.sum()),
        "mean_cut_db": float(depth[touched].mean()) if touched.any() else 0.0,
        "max_cut_db": float(depth.max()) if touched.any() else 0.0,
    }


def _make_up_gain(source, converted, sr, f0, good, max_gain_db, f0_lo):
    """Restore the level the notch took, in the band, keyed on the source."""
    import librosa

    S_src = librosa.stft(source, n_fft=_NFFT, hop_length=_HOP)
    S_out = librosa.stft(converted, n_fft=_NFFT, hop_length=_HOP)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    n = min(S_src.shape[1], S_out.shape[1])
    S_src, S_out = S_src[:, :n], S_out[:, :n]

    band = (freqs >= 550.0) & (freqs < 4000.0)
    src_band = np.sqrt((np.abs(S_src[band]) ** 2).mean(axis=0))
    out_band = np.sqrt((np.abs(S_out[band]) ** 2).mean(axis=0))

    f = f0[:n] if len(f0) >= n else np.pad(
        f0, (0, n - len(f0)), constant_values=np.nan)
    g = good[:n] if len(good) >= n else np.pad(
        good, (0, n - len(good)))
    high = g & np.isfinite(f) & (f >= f0_lo)

    delta_db = 20 * np.log10((src_band + 1e-9) / (out_band + 1e-9))
    delta_db = np.clip(delta_db, 0.0, max_gain_db)
    gain = np.where(high, 10 ** (delta_db / 20.0), 1.0)

    # Smooth so the gain is not a per-frame zipper.
    #
    # EDGE PADDING, not zero padding. ``np.convolve`` zero-pads, so a
    # gain array of all ones comes back DILUTED at the edges - the missing
    # neighbours count as zero and drag the weighted average down to 0.625,
    # which is a 4 dB ATTENUATION of audio that needed no change at all.
    # Measured on a clean source: the un-smoothed gain array is all ones,
    # and the smoothed one dips to 0.625 in the first and last four frames.
    #
    # The gain must never be below 1.0: it exists to RESTORE level the
    # notch took, so a value under 1 means it is removing level that was
    # never lost. Pad with the edge value instead, so an all-ones gain
    # array stays exactly all ones.
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    padded = np.pad(gain, pad, mode="edge")
    gain = np.convolve(padded, kernel, mode="valid")
    gain = np.where(high, gain, 1.0)

    S_out = S_out * gain[None, :]
    out = librosa.istft(S_out, hop_length=_HOP, length=len(converted))
    return out.astype(np.float32), gain


def suppress_inharmonic(source: np.ndarray, converted: np.ndarray, sr: int, *,
                        max_cut_db: float = 12.0,
                        max_gain_db: float = 12.0,
                        f0_lo: float = 550.0,
                        passes: int = 3) -> np.ndarray:
    """Cut the decoder's off-harmonic residue in the high register.

    ``source`` is the track that was fed to the decoder (after any pitch
    shift); ``converted`` is the decoder's output. Both mono float32 at
    ``sr``. Returns the corrected audio.

    The stage is a no-op if the source never sings above ``f0_lo`` - the
    decoder is not extrapolating, so there is nothing to correct.
    """
    source = np.asarray(source, dtype=np.float32)
    converted = np.asarray(converted, dtype=np.float32)
    if len(converted) < _NFFT or len(source) < _NFFT:
        return converted

    f0, good = source_voiced_frames(source, sr)
    if not good.any():
        return converted
    if not np.any(np.isfinite(f0) & (f0 >= f0_lo)):
        logger.info("fork inharmonic notch: source never reaches %.0f Hz, "
                    "no high-register residue to correct", f0_lo)
        return converted

    stats = []
    for _ in range(max(1, passes)):
        converted, ps = _notch_pass(source, converted, sr, f0, good,
                                    max_cut_db, f0_lo)
        stats.append(ps)

    converted, gain = _make_up_gain(source, converted, sr, f0, good,
                                    max_gain_db, f0_lo)
    lifted = gain > 1.01
    logger.info(
        "fork inharmonic notch: %.1f dB mean cut on %.0f%% of scored frames "
        "(%d passes, f0 >= %.0f Hz); make-up gain on %.0f%% of frames, "
        "max %.1f dB",
        float(np.mean([s["mean_cut_db"] for s in stats])),
        100.0 * float(np.mean([s["cut_frames"] for s in stats]))
        / max(int(good.sum()), 1),
        len(stats), f0_lo,
        100.0 * float(lifted.mean()),
        float(20 * np.log10(gain.max())))
    return converted
