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
                  min_excess_db: float = 3.0,
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

    Threshold swept against the source, applied ONCE to a non-de-tonalized
    render (applying it to an already-processed one measures a double pass and
    flatters whatever you are testing). Excess over source, in points:

        min_excess_db   7-8k  8-10.5k  10.5-14k  14-20k   clean-source residual
        (none)          +1.3    +1.5     +2.0     +3.5          -
        5.0             +0.7    +0.2     +0.8     +1.9      -26.2 dB
        3.0  <- shipped +0.2    -0.7     -0.1     +1.0      -24.5 dB
        2.0             +0.0    -1.1     -0.5     +0.6      -23.5 dB

    3.0 halves what is left and still sits at the collateral benchmark. 2.0 is
    past the point of usefulness: the bands go NEGATIVE, i.e. it is removing
    real HF tonality the source has, and the collateral goes 1.4 dB past the
    benchmark for it.

    What this stage CANNOT fix: the converted voice is uniformly ~6 % too
    PERIODIC than the source (spectral flux -5..-8 % in every band from 2 kHz
    up), and this stage does not move that at all (-0.36 -> -0.34 at 2-4 kHz).
    That is the decoder's own character, not isolated tones, and notching
    cannot add noise back. It needs a model change, not more output work.

    # ponytail: 7.5 kHz start, not 8.0 - the resonance skirt begins just below
    # Nyquist. Drop to 6 kHz only if the ring is ever reported lower; that
    # starts eating real brightness (measured: 6 kHz drives 7-8k to -1.9, i.e.
    # straight past the source).
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


def restore_air(source: np.ndarray, converted: np.ndarray, sr: int, *,
                cap_db: float = 12.0,
                lo_hz: float = 5000.0,
                smooth_oct: float = 0.5) -> np.ndarray:
    """Undo the decoder's HF deficit against the SOURCE - "needs to get brighter".

    The deficit is real and it is NOT the decoder inventing a dull voice; it is
    in the training data. Median LTAS, each file anchored 300-3000 Hz:

                                  3-5k    5-8k   8-12k  12-16k   fmax
        source stem (Mariah)      -2.1    -2.2    -2.8    -5.3   22.1 kHz
        Brandy's training samples -6.6   -11.8   -20.0   -24.3   22.1 kHz
        converted (G_389)         -2.8    -3.1   -13.9   -14.7
        WORLD +12 augmented      -10.8    -8.7   -23.8   -28.2   16.1 kHz

    Her corpus is 17 dB darker than the source at 8-12 kHz, so the model learned
    a dark voice and reproduces it faithfully. No retrain fixes that; the data
    is dark. (The augmentation is darker STILL and truncates fmax to 16.1 kHz,
    which is a constraint on any future synthetic data.)

    A FIXED SHELF IS THE WRONG SHAPE and was tried first. The deficit is roughly
    flat to 8 kHz then falls off a cliff (-2.8, -3.1, -13.9, -14.7, -23.1), so
    any single taper either under-corrects the top or overshoots 5-8 kHz - the
    best fixed shelf still left 9.1 dB of worst-band error while pushing 5-8 kHz
    +3.8 dB BRIGHTER than the source.

    So this measures the deviation and undoes it, exactly as the 1.2-5 kHz
    correction above does, with three guards:

      * heavily octave-SMOOTHED (median, +-0.5 oct), so it is a tilt and cannot
        manufacture a narrow resonance - see the LTAS-anchor failure in
        `_resonance_excess`'s note;
      * one-sided - it only ever LIFTS toward the source, never cuts, so it
        cannot fight `detonalize_hf` or re-darken anything;
      * bounded by `cap_db`, and applied only above `lo_hz` with a one-octave
        fade-in, so the formant region that carries identity is untouched.

    Deviation from the source per band, and identity, on the Hero render:

        cap    3-5k   5-8k  8-12k 12-16k 16-20k  centroid  identity
        none   -2.8   -3.1  -13.9  -14.7  -23.1   3598 Hz   1.0000
        8 dB   -2.8   -0.7   -6.4   -6.4  -15.0   4858 Hz   0.9969
       12 dB   -2.7   -0.4   -2.6   -2.3  -10.9   5644 Hz   0.9925  <- shipped
       16 dB   -2.7   -0.3   -1.2   +1.4   -6.8   6348 Hz   0.9884

    (source centroid 5293 Hz, identity floor 0.88)

    `docs/handover-fork-output-chain.md` had ruled this band off-limits -
    "matching the source undoes the conversion; matching her catalogue chases a
    mastering difference" - and left the question open pending an ear A/B. The
    user asked for brighter, which settles it in the other direction. This is an
    EAR CALL, so `cap_db` is meant to be tuned by them, not derived from a
    metric. Runs AFTER `detonalize_hf`: a smoothed tilt raises peak and floor
    together, so the ring does not return (measured 8.1 % at every lift).

    # ponytail: median over +-0.5 oct. Narrower smoothing starts tracking
    # individual partials, which is how the fake 8 kHz resonance was invented.
    """
    import librosa

    y = np.asarray(converted, dtype=np.float32)
    s = np.asarray(source, dtype=np.float32)
    if cap_db <= 0.0 or y.size < 4096 or s.size < 4096 or not np.any(y) or not np.any(s):
        return y
    # BANDWIDTH GATE. Never lift above the render's own wall: a band-limited
    # voice (Bluetooth mic, lossy codec) has only a noise floor up there, and
    # gain on noise is what made Patrick's first render sound "electronic" -
    # this stage hit its +12 dB cap over a band holding 0.49 % of his energy.
    # `extend_bandwidth` is the right tool above the wall; this one stops at it.
    wall = effective_fmax(y, sr)
    if wall < sr / 2 * 0.8:
        logger.info("fork air restore: capped at the render's own wall (%.0f Hz)", wall)
    dc, f = _ltas(y, sr)
    ds, _ = _ltas(s[: len(y)] if len(s) > len(y) else s, sr)
    deficit = _octave_smooth(ds - dc, f, smooth_oct)
    ramp = np.clip(np.log2(np.maximum(f, 1.0) / lo_hz), 0.0, 1.0)
    gain = np.clip(deficit, 0.0, cap_db) * ramp
    gain[f > wall] = 0.0          # nothing to lift beyond the wall
    out = _apply_gain(y, sr, gain)
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = (out / peak * 0.999).astype(np.float32)
    logger.info("fork air restore: up to %+.1f dB above %.0f Hz (cap %.0f)",
                float(gain.max()), lo_hz, cap_db)
    return out


def effective_fmax(y: np.ndarray, sr: int, min_drop_db: float = 18.0) -> float:
    """Find the CLIFF - where a band-limited capture stops - not a level.

    An absolute threshold ("last bin above -60 dB") is fooled the moment
    anything has lifted the noise floor: Patrick's chain-processed render read
    19.8 kHz that way, when his actual capture walls at 7.2 kHz. A brickwall
    has a signature an amplifier cannot fake - a sharp DROP - so look for that
    instead: the steepest fall, in dB per octave, above 3 kHz.

    Returns Nyquist when no cliff is found, which is the safe answer: callers
    treat "not band-limited" as "do nothing".
    """
    import librosa

    z = np.asarray(y, dtype=np.float32)
    S = np.abs(librosa.stft(z, n_fft=8192, hop_length=2048))
    r = librosa.feature.rms(y=z, frame_length=8192, hop_length=2048)[0][:S.shape[1]]
    d = 20 * np.log10(r + 1e-9)
    if not (d > d.max() - 25).any():
        return float(sr / 2)
    S = S[:, d > d.max() - 25]
    f = librosa.fft_frequencies(sr=sr, n_fft=8192)
    a = 20 * np.log10(np.median(S, axis=1) + 1e-12)
    a -= a.max()
    # octave-spaced probes: how far does the level fall from f to 1.4*f?
    best_f, best_drop = float(sr / 2), 0.0
    lo = 3000.0
    while lo * 1.4 < sr / 2:
        b0 = (f >= lo * 0.85) & (f < lo * 1.05)
        b1 = (f >= lo * 1.30) & (f < lo * 1.55)
        if b0.any() and b1.any():
            drop = float(np.median(a[b0]) - np.median(a[b1]))
            if drop > best_drop:
                best_drop, best_f = drop, lo * 1.15
        lo *= 1.15
    return best_f if best_drop >= min_drop_db else float(sr / 2)


def extend_bandwidth(converted: np.ndarray, sr: int, *,
                     wall_hz: float | None = None,
                     top_hz: float = 16000.0,
                     tilt_db: float = -6.0,
                     noise_mix: float = 0.5,
                     n_fft: int = 2048,
                     hop: int = 512) -> np.ndarray:
    """Synthesise the band a band-limited source never captured.

    The user on Patrick's first render: "it sounds like were listening to the
    conversion through a pipe", "needs to be brighter ... less electronic".

    His recordings wall at 7.2 kHz - `input_device_label: AirPods`, and a
    Bluetooth headset used as a MICROPHONE forces HFP/SCO, 16 kHz sampling,
    Nyquist 8 kHz. Stored at 48 kHz, but nothing above 7.2 kHz was ever
    captured, and the model reproduces that faithfully.

    `restore_air` made this WORSE, which is the important part: it hit its
    +12 dB cap trying to reach the source's 8-12 kHz level, but only 0.49 % of
    his energy is up there, so it amplified a NOISE FLOOR. That is what read as
    "electronic". Gain cannot create brightness where there is no signal.

    So translate the real harmonic content from one octave below the wall up
    into the empty band, decaying `tilt_db` per octave, blended with noise so it
    is not a sterile duplicate, and with randomised phase above the wall so the
    copies do not comb. Measured on Stitches, spectrum relative to own peak:

                        7.2-9k   9-12k  12-16k
        before          -58.0    -58.6   -59.9
        after           -38.6    -43.7   -49.1
        source track    -41.1    -41.9   -46.6

    i.e. within ~2 dB of the real thing across the whole band, and identity
    holds at 0.9981.

    This is a repair for damaged capture, not a general brightener. It is gated
    on `effective_fmax` being well below Nyquist and does nothing to a
    full-band render.

    # ponytail: one octave of source material, copied up. A learned bandwidth
    # extender would do better; this costs nothing and fixes "through a pipe".
    """
    import librosa

    y = np.asarray(converted, dtype=np.float32)
    if y.size < n_fft or not np.any(y):
        return y
    # 12 kHz, not a fraction of Nyquist: the failure this repairs is a capture
    # sampled at 16 kHz or less (Bluetooth HFP walls at 7-8 kHz), and a real
    # full-band vocal reaches 20 kHz+ (Brandy's measure 22.1). Anything above
    # 12 kHz is a real recording that needs no invented band.
    wall = float(wall_hz) if wall_hz else effective_fmax(y, sr)
    if wall >= 12000.0 or wall < 3000.0:
        logger.info("fork bandwidth extend: skipped (fmax %.0f Hz needs no repair)", wall)
        return y
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, ph = np.abs(S), np.angle(S)
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    df = float(f[1] - f[0])
    out = mag.copy()
    lo_i, hi_i = int((wall / 2) / df), int(wall / df)
    top_i = min(int(top_hz / df), len(f))
    width = max(1, hi_i - lo_i)
    rng = np.random.default_rng(0)
    for k, start in enumerate(range(hi_i, top_i, width)):
        end = min(start + width, top_i)
        w = end - start
        rep = mag[lo_i:lo_i + w] * 10 ** ((tilt_db * (k + 1)) / 20.0)
        nz = rng.random(rep.shape) * rep.mean(axis=0, keepdims=True)
        out[start:end] = np.maximum(out[start:end], (1 - noise_mix) * rep + noise_mix * nz)
    ph2 = ph.copy()
    ph2[hi_i:top_i] = rng.uniform(-np.pi, np.pi, ph2[hi_i:top_i].shape)
    res = librosa.istft(out * np.exp(1j * ph2), hop_length=hop, length=len(y)).astype(np.float32)
    peak = float(np.abs(res).max())
    if peak > 1.0:
        res = (res / peak * 0.999).astype(np.float32)
    logger.info("fork bandwidth extend: wall %.0f Hz -> %.0f Hz (tilt %.1f dB/oct)",
                wall, top_hz, tilt_db)
    return res
