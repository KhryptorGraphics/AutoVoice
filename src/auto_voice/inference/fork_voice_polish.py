"""Correct the fork decoder's own spectral colouring, keyed on the source vocal.

Measured on Hero / ep480 (converted minus source, 1/3-octave, both anchored to
their 300-3000 Hz mean, so this is colouring and not loudness):

      1260 Hz  -3.6 dB     4000 Hz  -6.3 dB
      1587 Hz  -2.8 dB     5040 Hz  -3.2 dB
      2000 Hz  -2.1 dB     6350 Hz  -2.9 dB
      2520 Hz  +3.1 dB     8000 Hz -10.1 dB
      3175 Hz  +3.9 dB    10079 Hz -13.4 dB

Two different things live in that column and only one of them is a defect.

**1.2-5 kHz is a defect.** A scooped 1.2-2 kHz with a +4 dB bump at 2.5-3.2 kHz
and a 6 dB notch at 4 kHz is a resonance the decoder imposes on every render.
That band is consonant definition and formant contrast - the user hears it as
"does not come through as clear as I know is possible" and as soft edges,
because a consonant transient IS 2-4 kHz energy. It is also the band where
identity is least at stake, so restoring what the decoder dropped relative to
the source restores the original singer's articulation without changing who the
voice sounds like. Verified: the deviation closes to under 1.5 dB everywhere in
the band, and nothing outside it moves.

**Above 5 kHz is a broad slope, and broad slopes are left alone.** The decoder
sits 10-13 dB under the source up there while still running 5-9 dB brighter
than Brandy's own recordings, so the top end is a blend of the two and there is
no defensible target to match: pulling it to the source would undo the
conversion, pulling it to her catalogue would chase a difference that mastering
alone could explain. `air_tame_db` exists to try the second by ear; it is off
by default and should stay off until someone has A/B'd it.

Narrowband is the exception above 5 kHz. The decoder emits fixed tones that are
in neither the source nor her voice - 17916 Hz and 18605 Hz, wandering 6 Hz over
a whole song where real partials wander 60-130 Hz, and present in independent
renders. They are cut by their own measured excess over a median trend, so
nothing is pinned to one checkpoint.

A caution for whoever measures this next: an LTAS anchored to a narrow band
invents features. Anchoring to the 8-10.5 kHz mean showed a convincing 900 Hz
"resonance" at 8.0-8.9 kHz that is not there at all - it was the top of an
ordinary decline. Anchor wide (300-3000 Hz), and read conv-minus-source, not
conv-minus-its-own-neighbourhood.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_NFFT = 4096
_HOP = 1024
_ANCHOR = (300.0, 3000.0)     # colouring is measured relative to this band


def _ltas(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average spectrum in dB over the loud frames only, anchored."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=_NFFT, hop_length=_HOP))
    rms = librosa.feature.rms(y=y, frame_length=_NFFT, hop_length=_HOP)[0]
    db = 20 * np.log10(rms + 1e-12)
    loud = (db - db.max()) > -30.0
    S = S[:, loud[:S.shape[1]]]
    if S.shape[1] == 0:
        S = np.abs(librosa.stft(y, n_fft=_NFFT, hop_length=_HOP))
    f = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    d = 20 * np.log10(S.mean(axis=1) + 1e-12)
    band = (f >= _ANCHOR[0]) & (f < _ANCHOR[1])
    return d - d[band].mean(), f


def _octave_smooth(d: np.ndarray, f: np.ndarray, frac: float) -> np.ndarray:
    """Smooth a dB curve over +-`frac` octaves (log-spaced, so wider up top)."""
    out = np.empty_like(d)
    lo_f, hi_f = f * 2 ** -frac, f * 2 ** frac
    lo = np.searchsorted(f, lo_f, "left")
    hi = np.searchsorted(f, hi_f, "right")
    csum = np.concatenate([[0.0], np.cumsum(d)])
    n = np.maximum(hi - lo, 1)
    out[:] = (csum[hi] - csum[lo]) / n
    return out


def _resonance_excess(d: np.ndarray, bins: int = 371) -> np.ndarray:
    """How far each bin stands above its own neighbourhood.

    The trend is a MEDIAN, not a mean: a mean over a log-spaced window is pulled
    far below the curve wherever the spectrum falls steeply, which reported a
    +28 dB "excess" across the whole top octave and cut 830 bins by 8 dB. A
    median tracks a slope exactly and ignores anything narrower than half its
    window - so the broad HF roll-off that carries her identity passes through
    untouched while a tone sitting on top of it does not.

    +-2 kHz is deliberately much wider than the tones being caught: it is the
    width at which the decoder's own broad shape is still followed rather than
    flattened. Only 171 of 2049 bins end up cut at all.
    """
    from scipy.ndimage import median_filter

    return d - median_filter(d, size=bins, mode="nearest")


def _taper(f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1 inside [lo, hi], cosine-faded to 0 over half an octave either side."""
    w = np.zeros_like(f)
    inside = (f >= lo) & (f <= hi)
    w[inside] = 1.0
    for edge, lower in ((lo, True), (hi, False)):
        a, b = (edge * 2 ** -0.5, edge) if lower else (edge, edge * 2 ** 0.5)
        seg = (f > a) & (f < b) if lower else (f > a) & (f < b)
        if seg.any():
            t = (f[seg] - a) / (b - a)
            w[seg] = 0.5 - 0.5 * np.cos(np.pi * (t if lower else 1 - t))
    return w


def _apply_gain(y: np.ndarray, sr: int, gain_db: np.ndarray) -> np.ndarray:
    """Zero-phase magnitude correction. The curve is smooth, so no pre-ring."""
    import librosa

    S = librosa.stft(y, n_fft=_NFFT, hop_length=_HOP)
    S *= (10 ** (gain_db / 20.0))[:, None]
    return librosa.istft(S, hop_length=_HOP, n_fft=_NFFT, length=len(y)).astype(np.float32)


def polish_voice(source: np.ndarray, converted: np.ndarray, sr: int, *,
                 band: tuple[float, float] = (1200.0, 5000.0),
                 max_correct_db: float = 4.0,
                 deres_above_hz: float = 5000.0,
                 min_excess_db: float = 3.0,
                 max_notch_db: float = 12.0,
                 air_tame_db: float = 0.0) -> np.ndarray:
    """Return ``converted`` with the decoder's own colouring corrected.

    Inside ``band`` the decoder's deviation from ``source`` is undone (clamped to
    ``max_correct_db``); above ``deres_above_hz`` only narrowband peaks are
    notched. ``air_tame_db`` optionally shelves 5-10 kHz down - an ear call, off
    by default. Both inputs are mono float32 at ``sr``.
    """
    source = np.asarray(source, dtype=np.float32)
    converted = np.asarray(converted, dtype=np.float32)
    if len(converted) < _NFFT or len(source) < _NFFT:
        return converted

    ds, f = _ltas(source, sr)
    dc, _ = _ltas(converted, sr)

    # 1. articulation: undo what the decoder took out relative to the source,
    #    but only across the identity-neutral consonant band.
    diff = _octave_smooth(ds - dc, f, 1 / 6)
    correct = np.clip(diff, -max_correct_db, max_correct_db) * _taper(f, *band)

    # 2. de-resonance: a peak standing above the output's OWN local trend is the
    #    decoder ringing, not programme material. Cut it by its own excess.
    # Soft knee: an LTAS has a dB or two of ripple everywhere, and cutting that
    # would flatten her harmonics. Only the part of a peak that clears
    # `min_excess_db` is taken, so ripple is untouched and a real resonance is
    # reduced rather than surgically removed.
    excess = _resonance_excess(dc)
    notch = -np.clip(excess - min_excess_db, 0.0, max_notch_db) * (f >= deres_above_hz)

    gain = correct + notch
    if air_tame_db:
        gain = gain - abs(air_tame_db) * _taper(f, 5000.0, 10000.0)
    out = _apply_gain(converted, sr, gain)

    peak = float(np.abs(out).max())
    if peak > 1.0:                     # a lift can only ever add headroom trouble
        out = out / peak * 0.999
    lo, hi = band
    logger.info("fork voice polish: %.1f..%+.1f dB across %.0f-%.0f Hz, "
                "deepest notch %.1f dB at %.0f Hz",
                float(correct[(f >= lo) & (f <= hi)].min()),
                float(correct[(f >= lo) & (f <= hi)].max()), lo, hi,
                float(notch.min()), float(f[int(np.argmin(notch))]))
    return out


def detonalize_hf(converted: np.ndarray, sr: int, *,
                  above_hz: float = 7500.0,
                  min_excess_db: float = 5.0,
                  max_cut_db: float = 12.0,
                  n_fft: int = 2048,
                  hop: int = 512) -> np.ndarray:
    """Remove the decoder's high-band ringing - the "double buzzing" halo.

    The user's report on the G_250 Hero render: "a kind of like a double
    runging sound to the converted voice, or it could be described as a double
    buzzing around the voice". "Around" is the accurate part: it is not the
    voice, it is a metallic halo sitting above it.

    ContentVec runs at 16 kHz, so its Nyquist is 8 kHz. **Above 8 kHz the
    decoder has no content input at all** - it is inventing the band from the
    speaker embedding and the upsampling stack alone, and what it invents is
    narrowband and tonal. Measured on Hero, as the fraction of loud frames in
    which a bin stands >6 dB above its own local spectral floor, 8-10.5 kHz:

        source stem              8.6 %
        converted (G_250)       10.1 %
        converted (G_480)       10.8 %
        after this stage         8.7 %   <- lands on the source

    The band mean understates it badly. The excess is concentrated in a few
    fixed bins near 8.9-9.4 kHz which reach +44 to +52 points, 5.0 sd out of a
    per-bin difference whose sd is 8.9. Those bins are a real resonance and not
    a fluke of one render: across two independent renders (different noise
    realisations) the per-bin excess correlates r = 0.84 and 13 of the 20
    hottest bins are shared.

    ``fork_voice_polish``'s existing notcher cannot catch this, and no
    retuning of it will. That notcher works on the LTAS - a whole-file average
    - and these tones drift in frequency, so they smear into a broad bump: the
    LTAS excess at 8.6-9.6 kHz is 3.63 dB against a 3.0 dB threshold, giving a
    0.6 dB notch, while per frame the same bins are 6 dB peaks in 55 % of
    frames. A stationary-tone notcher structurally cannot see a moving tone.
    Hence per-frame, and hence a separate stage.

    Each frame is compared with its OWN local spectral floor (a 21-bin median,
    ~450 Hz at this resolution), so the cut follows the drift instead of
    assuming a fixed frequency. Only the part of a peak clearing
    ``min_excess_db`` is taken, so ordinary spectral ripple is untouched.

    Collateral, measured on the full Hero render: -0.000 dB below
    ``above_hz`` (it is masked off, so this is exact, not merely small),
    +0.00 dB broadband level. Digital silence in gives digital silence out.

    Keyed on nothing but the converted audio: unlike the other stages this one
    needs no source reference, because "narrow peak above 8 kHz" is decoder
    invention regardless of what the source was doing.

    # ponytail: 7.5 kHz start, not 8.0 - the resonance skirt begins just below
    # Nyquist. Drop to 6 kHz only if the ring is ever reported lower; that
    # starts eating real brightness.
    """
    import librosa
    from scipy.ndimage import median_filter

    y = np.asarray(converted, dtype=np.float32)
    if y.size < n_fft or not np.any(y):
        return y
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    level = 20 * np.log10(mag + 1e-12)
    floor = median_filter(level, size=(21, 1))
    cut = np.clip(level - floor - min_excess_db, 0.0, max_cut_db) * (f >= above_hz)[:, None]
    out = librosa.istft(mag * 10 ** (-cut / 20.0) * np.exp(1j * phase),
                        hop_length=hop, length=len(y)).astype(np.float32)
    touched = cut > 0
    logger.info("fork HF de-tonalize: %.1f dB mean cut above %.0f Hz on %.0f%% of frames",
                float(cut[touched].mean()) if touched.any() else 0.0,
                above_hz, 100.0 * float((cut.max(axis=0) > 0).mean()))
    return out
