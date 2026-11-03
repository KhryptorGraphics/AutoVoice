"""Make the converted voice follow the SOURCE singer's vocal effort.

The user's report: "sometimes she is singing calmly with a soft voice, and
sometimes she really belting it out ... the original artist you can tell the
difference". Soft singing and belting differ in two ways that survive a
change of singer: level, and spectral TILT - a belt pushes energy into the
1-5 kHz harmonics, a soft voice is dark and breathy. Identity lives in the
formant detail, not in those two numbers, so they can be carried across.

Measured on Hero (scorer gate, 0.25 s blocks, the source's effort terciles;
effort = z(level) + z(harmonic tilt)): the converted voice's LEVEL already
follows (block slope 0.99), but its harmonic tilt moves only about 0.6 as
far as the source's between soft and belted passages - belt-minus-soft
contrast ratio 0.58 after the served chain. The decoder flattens effort into
one middle timbre.

What this stage does, per STFT frame and per log-spaced band:

* below ``match_above_hz`` each band's share of the frame's energy is
  compared with that band's song-long mean share, for the source and the
  converted alike, and the difference between the two DEVIATIONS is applied:
  the converted keeps its own average timbre and moves with the source's soft
  and belted passages;
* above ``match_above_hz`` the band's share is pulled to the source's share
  in that frame. The air restore matches the long-term average, which the
  source's sibilants and loudest belts dominate, so on the median sustained
  note it overshoots: +7.2 / +7.7 dB at 10-16 kHz on soft / belted frames, a
  noise bed riding on every held note - the "staticy around the voice". The
  per-frame match pulls each frame's top band back to the source's.

Frame energy is preserved (loudness belongs to the level stages), the gain is
median-filtered over 0.15 s and smoothed over 0.3 s so consonant-length
events cannot steer it, bounded by ``max_db`` per band, and faded out
wherever the source is below the scorer's level gate - gaps belong to the
hum stage. "Preserved" is exact for the windowed STFT energy of each frame;
the rectangular frame RMS the dropout fill reads moves by p99 0.37 dB, max
1.9 dB, at transients.

Offline, hero_src key - the same four raw takes (renders of the reference
stem) through every stage with and without this one, scored with the
serving interpreter (librosa 0.10.2), render minus source:

                                  before   after
    harmonic tilt contrast ratio   0.58     0.98   (1.0 = follows fully)
    harmonic tilt block slope      0.60     0.97
    10-16 kHz share, soft frames  +7.4 dB  +0.4 dB
    10-16 kHz share, belt frames  +5.4 dB  +1.0 dB
    6-10 kHz share, soft frames   -9.1 dB  -4.0 dB
    6-10 kHz share, belt frames   +0.7 dB  -1.7 dB
    10-16 kHz, consonant frames   +4.0 dB  +0.4 dB
    long-term 10-16 kHz           +2.7 dB  +0.1 dB
    sung-frame centroid           +354 Hz  +20 Hz
    700-1100 Hz level dropout     0.53 %   0.53 %
    700-1100 Hz inharmonic excess 15.8 %   12.3 %
    250-400 Hz inharmonic excess   1.6 %    4.2 %   (score_multi window)
    same, off share, 1-bin window  0.402    0.383   (source 0.310)

Served path, paired - four htdemucs_ft separations of the original upload
(the service's separator; each call draws its own random demucs shift),
each rendered through the fork bridge, then the production chain on that
SAME render with this stage off and on; serving interpreter, render minus
source, n=4:

                                  off      on
    harmonic tilt contrast ratio   0.57     1.00
    sung-frame centroid           +231 Hz  +12 Hz
    10-16 kHz share, soft frames  +6.8 dB  +0.4 dB
    10-16 kHz share, belt frames  +5.3 dB  +1.0 dB
    700-1100 Hz inharmonic excess 11.4 %    7.3 %   (ranges overlap)
    700-1100 Hz level dropout     2.06 %   2.24 %  (frames of 424 per
                                                    render: 8/7/10/10 ->
                                                    8/7/11/12)
    250-400 Hz inharmonic excess   2.4 %    5.0 %   (score_multi window)
    same, off share, 1-bin window  0.406    0.386   (source 0.310)

The dropout difference is frames close to the scorer's 6 dB gate tipping
over; the scorer calls it noise. The 250-400 Hz row depends on the scorer's
harmonic window: +-3 % of k*f0 is narrower than one FFT bin for the lowest
harmonics at these pitches, so most of their energy reads as off-harmonic
(the source itself scores 0.533 off). Floored at one bin, the off-harmonic
share falls toward the source's instead. Neither reading has been checked
by ear.

Band layout: octaves to 4 kHz, third-octaves above. The resolution matters
up top: the source's soft passages carry breath noise at 6-8 kHz that the
converted lacks, and a half-octave band centred on 5.7 kHz is energy-weighted
toward 4-5 kHz, where the two voices agree - it reported an 8 dB deficit
where the 6-8 kHz deficit was 13.5 dB. With half-octave bands and a 10 dB
bound soft 6-10 kHz finished at -6.8 dB; third-octaves at 14 dB, -4.0 dB.

What it cannot change: the fine structure. Above ~9 f0 the decoder's NSF
excitation has no harmonics (``harmonic_num=8``), so its top band is noise
where a belting source has harmonics (whitened 5-10 kHz flatness 0.54 vs the
source's 0.17), and a smooth gain leaves that untouched.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_NFFT = 2048
_HOP = 512
_EPS = 1e-12
# Octaves to 4 kHz, third-octaves above (see the module note on resolution).
_BAND_CENTRES_HZ = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 5040.0, 6350.0,
                    8000.0, 10079.0, 12699.0, 16000.0)
_GATE_DB = -25.0          # the scorer's level floor, re the source's loud-frame median
_MEDIAN_S = 0.15          # rejects consonant-length spikes before smoothing
_MIN_LIVE_S = 2.0


def band_weights(freqs: np.ndarray, centres: Sequence[float]) -> np.ndarray:
    """Raised-cosine partition of unity over log-frequency, ``(bands, bins)``.

    Each bin's weights sum to exactly 1, so a per-band gain becomes a smooth
    per-bin gain with no steps. Below the first centre everything belongs to
    the first band, above the last centre to the last.
    """
    c = np.log2(np.asarray(centres, dtype=np.float64))
    p = np.interp(np.log2(np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)),
                  c, np.arange(len(c), dtype=np.float64))
    d = p[None, :] - np.arange(len(c), dtype=np.float64)[:, None]
    return np.where(np.abs(d) < 1.0, np.cos(0.5 * np.pi * d) ** 2, 0.0)


def follow_effort(source: np.ndarray, converted: np.ndarray, sr: int, *,
                  strength: float = 1.0,
                  max_db: float = 14.0,
                  smooth_s: float = 0.3,
                  centres: Sequence[float] = _BAND_CENTRES_HZ,
                  match_above_hz: float | None = None) -> np.ndarray:
    """Move ``converted``'s per-frame spectral balance with the source's effort.

    ``source`` is the track that was fed to the decoder and ``converted`` its
    output, mono float32 at ``sr``. ``strength`` scales the correction (0
    returns the input). Bands centred above ``match_above_hz`` are matched to
    the source's per-frame share instead of following its deviation; ``None``
    keeps every band's own mean. Centres at or above Nyquist are ignored.
    """
    import librosa
    from scipy.ndimage import median_filter, uniform_filter1d

    conv = np.asarray(converted, dtype=np.float32)
    src = np.asarray(source, dtype=np.float32)
    if strength <= 0.0 or max_db <= 0.0:
        return conv
    if len(conv) < 4 * _NFFT or len(src) < 4 * _NFFT or not np.any(conv):
        return conv
    c = np.asarray([x for x in centres if x < 0.5 * sr], dtype=np.float64)
    if len(c) < 2:
        return conv

    S_c = librosa.stft(conv, n_fft=_NFFT, hop_length=_HOP)
    S_s = librosa.stft(src[: len(conv)], n_fft=_NFFT, hop_length=_HOP)
    n = min(S_c.shape[1], S_s.shape[1])
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    W = band_weights(freqs, c).astype(np.float32)

    E_s = uniform_filter1d(W @ (np.abs(S_s[:, :n]) ** 2), 3, axis=1)
    E_c = uniform_filter1d(W @ (np.abs(S_c[:, :n]) ** 2), 3, axis=1)
    tot_s = E_s.sum(axis=0)
    tot_c = E_c.sum(axis=0)

    lev = 10.0 * np.log10(tot_s + _EPS)
    loud = lev > lev.max() - 30.0
    live = (lev - np.median(lev[loud])) > _GATE_DB
    if live.sum() * _HOP / sr < _MIN_LIVE_S:
        return conv

    s_s = 10.0 * np.log10((E_s + _EPS) / (tot_s + _EPS))
    s_c = 10.0 * np.log10((E_c + _EPS) / (tot_c + _EPS))
    # Deviation bands keep each voice's own mean share; match bands are
    # pulled to the source's share frame by frame.
    keep_mean = np.ones((len(c), 1))
    if match_above_hz is not None:
        keep_mean[c > match_above_hz] = 0.0
    dev = ((s_s - keep_mean * s_s[:, live].mean(axis=1, keepdims=True))
           - (s_c - keep_mean * s_c[:, live].mean(axis=1, keepdims=True)))
    dev[:, ~live] = 0.0

    km = max(3, int(_MEDIAN_S * sr / _HOP) | 1)
    ku = max(3, int(smooth_s * sr / _HOP))
    dev = median_filter(dev, size=(1, km), mode="nearest")
    dev = uniform_filter1d(dev, ku, axis=1)
    taper = uniform_filter1d(live.astype(np.float64), ku)
    G = np.clip(strength * dev, -max_db, max_db) * taper[None, :]

    # Preserve each frame's energy EXACTLY, against the frame's own spectrum:
    # the level stages own loudness. The first version normalised against
    # the 3-frame-smoothed band energies instead; paired on four served-path
    # renders (same render, through the fill) it moved frame RMS by p99
    # 1.1 dB, max 3.4 dB, over 1 dB on 1.3 % of live frames. This one: p99
    # 0.37 dB, max 1.9 dB, 0.05 % - the residue is windowed STFT energy vs
    # the fill's rectangular frame RMS at transients.
    g_db = W.T @ G.astype(np.float32)
    P = np.abs(S_c[:, :n]) ** 2
    old = P.sum(axis=0)
    new = (P * 10.0 ** (g_db / 10.0)).sum(axis=0)
    g_db -= np.where(old > _EPS, 10.0 * np.log10((new + _EPS) / (old + _EPS)),
                     0.0)[None, :].astype(np.float32)
    S_c[:, :n] *= (10.0 ** (g_db / 20.0)).astype(np.float32)
    out = librosa.istft(S_c, hop_length=_HOP, length=len(conv)).astype(np.float32)

    act = np.abs(G[:, live])
    logger.info("fork effort follow: per-band |gain| on live frames, mean %s dB, "
                "p95 %s dB (bands %s Hz, match above %s)",
                np.round(act.mean(axis=1), 1).tolist(),
                np.round(np.percentile(act, 95, axis=1), 1).tolist(),
                [int(x) for x in c], match_above_hz)
    return out
