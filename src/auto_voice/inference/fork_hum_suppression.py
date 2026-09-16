"""Suppress the fork decoder's between-word hum.

The so-vits-svc-fork decoder emits *periodic* content in ~14 % of the frames
where the source singer is unvoiced (breaths, fricatives, the spaces between
words). The user hears it as an "electronic distortion in the dark space
between words" and as edges that are not crisp. Measured on Hero, ep480: the
source has 0 % periodicity in those frames by definition; the model 13.7 %.

It is a learned decoder behaviour: zeroing the decoder's f0 there (the UV
contract) and zeroing the prior's coarse f0 as well (13.8 -> 14.1 %) both change
nothing. Until a retrain penalises harmonic energy where uv=0, it is corrected
here at mix stage, keyed on the SOURCE vocal's own voicing so nothing is touched
where the original singer was actually singing:

1. strip - where the source is unvoiced, keep only the percussive (noise)
   component of the conversion: breaths and consonants survive, harmonics go.
   hum level vs voiced: -11.1 -> -17.9 dB
2. duck  - the frames that are still voiced in the output while unvoiced in the
   source are, by definition, the residual hum: -14 dB on exactly those.
   -> -20.4 dB
3. gate  - a soft level gate keyed on the source envelope for the long gaps,
   with a floor (never zeros: hard-zeroed chunks were the edge-bleed defect).

Order matters: this runs BEFORE the halo/reverb. With the reverb first, the
tail carried the hum into every short gap (+4 dB) and a later gate could not
recover it.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_FRAME_S = 0.01           # crepe hop: 10 ms frames at 16 kHz / 160
_VOICED = 0.5             # crepe periodicity threshold


def _voicing(y: np.ndarray, sr: int) -> np.ndarray:
    """Per-10 ms voicing probability of ``y`` (crepe periodicity, median-filtered)."""
    import librosa
    import torch
    import torchcrepe

    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "full" if device == "cuda" else "tiny"
    out = []
    step = 16000 * 20
    for start in range(0, len(y16), step):
        seg = y16[start:start + step]
        if len(seg) < 1600:
            continue
        t = torch.from_numpy(seg)[None].to(device)
        _, per = torchcrepe.predict(
            t, 16000, 160, 50.0, 1300.0, model,
            return_periodicity=True, batch_size=2048, device=device, pad=True)
        out.append(torchcrepe.filter.median(per, 3).squeeze(0).cpu().numpy())
        del t
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _to_samples(frames: np.ndarray, n: int, sr: int, smooth_s: float) -> np.ndarray:
    g = np.interp(np.arange(n), np.arange(len(frames)) * sr * _FRAME_S, frames.astype(np.float32))
    k = max(1, int(smooth_s * sr))
    return np.convolve(g, np.ones(k) / k, mode="same")


def _source_gate(source: np.ndarray, n: int, sr: int, *,
                 open_db=-30.0, close_db=-42.0, floor_db=-28.0,
                 hold_s=0.08, attack_s=0.015, release_s=0.11) -> np.ndarray:
    """Soft gain curve from the SOURCE envelope: 1 where she sings, ->floor where silent."""
    import librosa
    from scipy.ndimage import maximum_filter1d

    hop, win = 256, 1024
    rms = librosa.feature.rms(y=source, frame_length=win, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-12)
    db -= db.max()
    g = np.clip((db - close_db) / (open_db - close_db), 0.0, 1.0)
    g = maximum_filter1d(g, max(1, int(hold_s * sr / hop)))
    a_up, a_dn = np.exp(-hop / (attack_s * sr)), np.exp(-hop / (release_s * sr))
    sm = np.zeros_like(g)
    v = 0.0
    for i, x in enumerate(g):
        v = x + (v - x) * (a_up if x > v else a_dn)
        sm[i] = v
    floor = 10 ** (floor_db / 20)
    gain = floor + (1 - floor) * sm
    return np.interp(np.arange(n), np.arange(len(gain)) * hop, gain)

def _silence_mask(source: np.ndarray, sr: int, m: int, *,
                  silent_db: float = -55.0) -> np.ndarray:
    """Per-10 ms frame True where the source is below ``silent_db`` of its peak.

    Frames aligned to the crepe voicing frames (same 10 ms grid), so the two
    masks index identically. ``-55`` dB sits under every measured consonant
    lead (-36.7 dB median) and above the residual floor between words.
    """
    import librosa
    hop = 256
    rms = librosa.feature.rms(y=source, frame_length=1024, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-12)
    db -= db.max()
    idx = np.minimum(np.arange(m) * sr * _FRAME_S / hop, len(db) - 1)
    return db[idx.astype(int)] < silent_db


def suppress_hum(source: np.ndarray, converted: np.ndarray, sr: int, *,
                 duck_db: float = 14.0, hpss_margin: float = 2.0) -> np.ndarray:
    """Return ``converted`` with the between-word hum suppressed, keyed on ``source``.

    Both inputs are mono float32 at ``sr``; the output keeps ``converted``'s length.
    """
    import librosa

    source = np.asarray(source, dtype=np.float32)
    converted = np.asarray(converted, dtype=np.float32)
    n = len(converted)
    if n == 0 or len(source) == 0:
        return converted

    ps = _voicing(source, sr)
    po = _voicing(converted, sr)
    m = min(len(ps), len(po))
    if m == 0:
        return converted
    src_unvoiced = (ps[:m] < _VOICED).astype(np.float32)
    hum_frames = ((ps[:m] < _VOICED) & (po[:m] > _VOICED)).astype(np.float32)

    # 1. strip: percussive-only where the source is unvoiced AND silent.
    # The unvoiced-only mask (2026-09-06, Hero) was collateral-heavy on dry
    # takes: crepe labels every consonant lead unvoiced 40-80 ms before the
    # vowel — measured on All of Me, 99 % of attack windows sat at full
    # replacement and the HPSS residue carried 5.5 dB less attack energy,
    # heard as "artifacting around the edges of her voice". Those frames are
    # AUDIBLE in the source (-36.7 dB median, 100 % above -50): programme,
    # not hum. The strip must key on silence (the between-word condition the
    # stage was built for), not on voicing. On AOM the split is 9 % of frames
    # silent-unvoiced (real hum territory, kept) vs 18 % audible-unvoiced
    # (consonants, now untouched). The duck (step 2) still keys on voicing
    # mismatch, so a harmonic leak into an audible consonant is still pulled
    # down 14 dB — only the wholesale HPSS replacement is narrowed.
    silent = _silence_mask(source, sr, m)
    strip = ((ps[:m] < _VOICED) & silent).astype(np.float32)
    unv = _to_samples(strip, n, sr, 0.02)
    _, percussive = librosa.effects.hpss(converted, margin=hpss_margin)
    y = converted * (1.0 - unv) + percussive * unv

    # 2. duck: residual voiced-while-source-unvoiced frames
    hum = _to_samples(hum_frames, n, sr, 0.02)
    y = y * (1.0 - hum * (1.0 - 10 ** (-duck_db / 20)))

    # 3. gate: source-keyed soft gate for the long gaps (floor, never zeros)
    y = y * _source_gate(source, n, sr)

    logger.info("fork hum suppression: source unvoiced %.0f%% of frames, hum frames %.1f%%",
                float(src_unvoiced.mean()) * 100, float(hum_frames.mean()) * 100)
    return y.astype(np.float32)
