"""Remove the decoder's period doubling - the fuzz on belted notes.

The user's report: "her voice fuzzes out when she puts any umph or effort in
belting out notes". Measured on Hero / ep480, the median ratio of energy at the
half-integer partials (f0/2, 3f0/2, 5f0/2) to the fundamental, on voiced frames
above -24 dB, binned by the source's pitch:

      f0 band     source   converted   corrected (this stage, shipped defaults)
      250-350 Hz   0.013      0.063        0.033
      350-450 Hz   0.015      0.060        0.037
      450-600 Hz   0.065      0.206        0.079
      600-900 Hz   0.144      0.515        0.172

Two things there. The decoder emits 3-5x the source's period doubling at *every*
pitch, and the absolute amount explodes as she goes up - which is exactly what a
belt is. 0.51 means the subharmonic is half the height of the fundamental; that
is heard as rasp, not as voice.

The cause is not a setting. Above ~450 Hz the model is extrapolating: Brandy's
training corpus has almost nothing up there (see the register survey), so the
NSF excitation runs outside the distribution the decoder was fitted on and the
upsampling stack starts resolving a sub-octave. Only a retrain with real
high-register material removes it at source.

So it is corrected here, subtractively. Real voice puts its energy on integer
multiples of f0; energy sitting exactly halfway between two harmonics is either
breath noise (broadband, so a narrow notch costs almost nothing) or period
doubling. The notches are placed on the SOURCE's f0, which is measured on a
clean stem, and the conversion tracks it - median 0 cents across the song. They
are narrow enough (+-12 % of f0) that the 50-cent (3 %) tracking error still
lands on the partial.

Parameters were swept, not guessed. Depth is cheaper than width: 15 dB at
+-8 % beat 12 dB at +-12 % on both the belt ratio and the collateral. Shipped:
15 dB, +-16 %, which leaves a -24.9 dB residual when run on the CLEAN source
(the collateral a real voice would suffer; the STFT round-trip alone is -141).

Per-frame depth scales from 0 dB at an internal ratio of 0.15 to full at 0.40.
Be clear about what that gate is and is not. The internal ratio (mean magnitude
at the half-integer bins over mean at the integer bins) is NOT a clean doubling
detector: a clean voice's median is 0.11 and its own belted frames sit at 0.26,
overlapping the converted voice's 0.26 / 0.52. The peak-based ratio in the
table discriminates far better but costs a per-frame peak search. So the gate
only reduces collateral - it acts on 68 % of converted frames and 39 % of clean
ones, and buys 2 dB of residual - it does not exempt clean frames. The earlier
thresholds (0.04 / 0.25) also over-cut 450-600 Hz to 0.05, BELOW the source's
0.065, i.e. they were removing real voice; the current ones land it at 0.079.
# ponytail: per-frame peak ratio (max of the three subharmonic peaks over the
# f0 peak) is the upgrade if the residual ever has to move.

Runs BEFORE `fork_voice_polish`: the 3f0/2 and 5f0/2 partials of a belted note
land at 900-2250 Hz, inside the band the polish lifts, so polishing first
amplified the rasp by 22 %. Measured with both stages in the chain the two
orders finish within 0.001 of each other; it goes first so its gate reads the
less-distorted signal. The polish re-lifts part of what is left, so the
end-of-chain number is verified on the live render, not on this stage alone.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_NFFT = 4096
_HOP = 512
_CHUNK = 2000             # frames per pass; the full bin x frame mask is ~200 MB


def _source_f0(source: np.ndarray, sr: int) -> np.ndarray:
    """Source pitch per STFT frame, NaN where the source is not singing.

    ``librosa.yin`` never returns NaN: on silence it emits a confident ~1000 Hz
    (its fmax), so without a voicing gate the stage placed notches on a pitch
    that was never sung and cut a voiced conversion by -10.8 dB wherever the
    source was quiet. Frames more than 40 dB under the source's peak, or under
    an absolute -60 dBFS (a silent source has no meaningful peak), are treated
    as unvoiced. (pyin has a voicing flag but is ~20x slower.)
    """
    import librosa

    f0 = librosa.yin(source, fmin=80, fmax=1000, sr=sr,
                     frame_length=_NFFT, hop_length=_HOP)
    rms = librosa.feature.rms(y=source, frame_length=_NFFT, hop_length=_HOP)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n].astype(np.float32), rms[:n]
    db = 20 * np.log10(rms + 1e-12)
    f0[(db < db.max() - 40.0) | (db < -60.0)] = np.nan
    return f0


def suppress_subharmonics(source: np.ndarray, converted: np.ndarray, sr: int, *,
                          max_cut_db: float = 20.0,
                          width: float = 0.10,
                          top_hz: float = 6000.0,
                          clean_ratio: float = 0.15,
                          fuzzy_ratio: float = 0.40) -> np.ndarray:
    """Return ``converted`` with half-integer partials attenuated, keyed on ``source``.

    The cut is per frame, scaling from 0 dB where the frame is as clean as the
    source typically is to ``max_cut_db`` where it is fully doubled. Both inputs
    are mono float32 at ``sr``.
    """
    import librosa

    source = np.asarray(source, dtype=np.float32)
    converted = np.asarray(converted, dtype=np.float32)
    if len(converted) < _NFFT or len(source) < _NFFT:
        return converted

    f0 = _source_f0(source, sr)
    S = librosa.stft(converted, n_fft=_NFFT, hop_length=_HOP)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    n_frames = S.shape[1]
    f0 = np.asarray(f0[:n_frames], dtype=np.float32)
    if len(f0) < n_frames:                       # yin can come up a frame short
        f0 = np.pad(f0, (0, n_frames - len(f0)), constant_values=np.nan)

    band = (freqs > 0) & (freqs < top_hz)
    fb = freqs[band][:, None]
    cut_frames = 0

    for a in range(0, n_frames, _CHUNK):
        b = min(a + _CHUNK, n_frames)
        v = f0[a:b]
        ok = np.isfinite(v) & (v >= 80.0)
        if not ok.any():
            continue
        ratio = np.where(ok, fb / np.maximum(v[None, :], 1e-6), 0.0)
        frac = ratio - np.floor(ratio)
        near_half = (np.abs(frac - 0.5) < width) & (ratio >= 0.4) & ok[None, :]
        near_int = ((frac < width) | (frac > 1.0 - width)) & (ratio >= 0.8) & ok[None, :]

        mag = np.abs(S[band, a:b])
        # How doubled each frame is: the LOUDEST half-integer bin against the
        # loudest harmonic bin. The mean-ratio statistic this replaces could
        # not separate clean belts from doubled ones at all (Hero source 0.313
        # vs Hero converted 0.306 at 600-900 Hz; AOM clean source even read
        # 0.666 vs converted 0.652) because averaging the harmonic bins lets
        # one quiet bin dilute the peak the doubling actually sits on. The
        # peak statistic separates them 3.5-10x and, with the narrower notch
        # (width 0.16 -> 0.10), cut clean-frame collateral by 5.4 dB while
        # suppressing the belt band harder. Measured A/B on All of Me /
        # Hero, 2026-09-07 (docs/served-models.md, subharmonic gate upgrade).
        half_max = np.where(near_half, mag, 0.0).max(0)
        int_max = np.where(near_int, mag, 0.0).max(0)
        frame_ratio = half_max / (int_max + 1e-12)

        depth = np.clip((frame_ratio - clean_ratio) / (fuzzy_ratio - clean_ratio), 0.0, 1.0)
        depth = np.where(ok, depth, 0.0)
        cut_frames += int((depth > 0.05).sum())

        gain = 10 ** (-(max_cut_db * depth) / 20.0)[None, :]
        m = np.where(near_half, gain, 1.0).astype(np.float32)
        S[band, a:b] *= m

    out = librosa.istft(S, hop_length=_HOP, n_fft=_NFFT,
                        length=len(converted)).astype(np.float32)
    logger.info("fork subharmonic suppression: acted on %.0f%% of frames "
                "(up to %.0f dB on the half-integer partials)",
                100.0 * cut_frames / max(n_frames, 1), max_cut_db)
    return out
