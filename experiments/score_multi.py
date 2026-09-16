#!/usr/bin/env python
"""Score SVC checkpoints over N renders each, so an A/B can be believed.

Renders are stochastic (noise_scale 0.4): three renders of the SAME model have
spanned 6.1%-50.0% on the dropout metric. One render per model is worthless.
This averages over N renders per model label and reports the within-model
spread, so a difference between two models is only called real when it clears
the noise floor of the renderer itself.

ANALYSIS ONLY. No subprocess, no torch, no CUDA - computing melspectrograms
and then fork/exec-ing a CUDA subprocess segfaults on this project's boxes.
Render in one process, score in another.

Metrics, binned by the SOURCE's f0 (so every model is judged on the same
frames):
  - level dropout: % of sung frames where the render's normalised envelope is
    more than 6 dB under the source's
  - inharmonic excess: median inharmonic fraction vs the source's own, as a %
Plus, over all sung frames: spectral centroid and band energies (3-8k, 8-12k,
12-16k, 16-20k) relative to 300-3000 Hz.

Octave errors are rejected before a frame is used: pyin octave-halves ~40% of
voiced frames on this material, and an earlier pass that skipped the rejection
inverted which band looked worst.

    python score_multi.py --demo
    python score_multi.py --source src.wav --group base='eval/base_*.wav' \
                          --group cand='eval/cand_*.wav' --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
from math import comb

SR, HOP, NFFT = 44100, 512, 2048
FREQS = librosa.fft_frequencies(sr=SR, n_fft=NFFT)

F0_BANDS = [(250, 400), (400, 550), (550, 700), (700, 1100)]
ENERGY_BANDS = [(3000, 8000), (8000, 12000), (12000, 16000), (16000, 20000)]
ENERGY_REF = (300, 3000)

DROPOUT_DB = 6.0        # render this far under the source = a dropout
LEVEL_FLOOR_DB = -25.0  # source normalised level below this is not "sung"
VOICED_PROB = 0.8
IDENTITY_FLOOR = 0.88
MIN_BAND_FRAMES = 25    # fewer than this in a band -> report nan, not a number
INHARM_FRAMES = 400     # inharm() is a python loop; cap frames per band
LEN_TOL_HOPS = 4        # render/source length mismatch above this is an error


def _band_key(b):
    return f"{b[0]}-{b[1]}"


def _energy_key(b):
    return f"{b[0] // 1000}-{b[1] // 1000}k"


def _pk(S, i, fc, rel=0.03):
    """Peak |STFT| in [fc*(1-rel), fc*(1+rel)] at frame i.

    ponytail: the spec's +-3% window is narrower than one FFT bin (21.5 Hz)
    below ~360 Hz, so an empty mask would read 0 and spuriously reject the
    frame. Fall back to the nearest bin instead.
    """
    m = (FREQS >= fc * (1 - rel)) & (FREQS <= fc * (1 + rel))
    if not m.any():
        return float(S[np.argmin(np.abs(FREQS - fc)), i])
    return float(S[m, i].max())


def _inharm(S, idx, f0):
    """Median fraction of in-band energy that is NOT on a harmonic of f0."""
    out = []
    for i in idx:
        fc = f0[i]
        hm = np.zeros(len(FREQS), bool)
        for k in range(1, 12):
            hm |= (FREQS > fc * k * 0.97) & (FREQS < fc * k * 1.03)
        bd = (FREQS > fc * 0.5) & (FREQS < 6000)
        on = S[hm & bd, i].sum()
        off = S[(~hm) & bd, i].sum()
        out.append(off / (on + off + 1e-12))
    return float(np.median(out)) if len(out) else float("nan")


def _mel_centroid(y):
    """256-dim mel-statistics identity vector (project convention)."""
    ld = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128))
    v = np.concatenate([ld.mean(1), ld.std(1)])
    return v / (np.linalg.norm(v) + 1e-9)


def _level_db(y):
    return 20 * np.log10(librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0] + 1e-9)


def prep_source(path, inharm_frames=INHARM_FRAMES, seed=0, verbose=True):
    """Gate the source into usable sung frames once; all renders reuse them."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    f0, vf, vp = librosa.pyin(y, fmin=120, fmax=1100, sr=SR,
                              frame_length=NFFT, hop_length=HOP)
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    e = _level_db(y)
    M = min(len(f0), S.shape[1], len(e))
    f0, vf, vp, S, e = f0[:M], vf[:M], vp[:M], S[:, :M], e[:M]
    esn = e - np.median(e[e > e.max() - 30])

    cand = np.where(vf & np.isfinite(f0) & (vp > VOICED_PROB) & (esn > LEVEL_FLOOR_DB))[0]
    good = np.array([i for i in cand if _pk(S, i, 2 * f0[i]) <= _pk(S, i, f0[i])], dtype=int)
    if verbose:
        kept = 100 * len(good) / max(len(cand), 1)
        print(f"  source: {len(cand)} sung frames, {len(good)} after octave-error "
              f"rejection ({kept:.0f}% kept)")
    if len(good) == 0:
        raise RuntimeError(f"no usable sung frames in {path} "
                           f"({len(cand)} candidates, all octave-rejected)")

    rng = np.random.default_rng(seed)
    band_idx, inh_idx = {}, {}
    for b in F0_BANDS:
        idx = good[(f0[good] >= b[0]) & (f0[good] < b[1])]
        band_idx[b] = idx                      # every frame: dropout is vectorised
        if len(idx) > inharm_frames:           # fixed-seed spread over the whole song,
            idx = np.sort(rng.choice(idx, inharm_frames, replace=False))  # not the first N
        inh_idx[b] = idx
    src_inh = {b: _inharm(S, inh_idx[b], f0) for b in F0_BANDS}

    return {"path": path, "n_samples": len(y), "M": M, "f0": f0, "esn": esn,
            "good": good, "band_idx": band_idx, "inh_idx": inh_idx, "src_inh": src_inh}


def score_render(path, src, len_tol_hops=LEN_TOL_HOPS):
    """Metrics for one rendered wav against the prepared source."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    drift = abs(len(y) - src["n_samples"])
    if drift > len_tol_hops * HOP:
        raise RuntimeError(
            f"{os.path.basename(path)} is {len(y)} samples, source is "
            f"{src['n_samples']} ({drift / SR:.3f}s adrift). The metrics assume "
            f"frame alignment; a shifted render would score a large fake dropout. "
            f"Re-render, or raise len_tol_hops if the offset is known-benign.")

    ec = _level_db(y)
    m = min(len(ec), src["M"])
    esn = src["esn"]
    ecn = ec[:m] - np.median(ec[:m][esn[:m] > -30])
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))[:, :m]

    met = {}
    for b in F0_BANDS:
        idx = src["band_idx"][b][src["band_idx"][b] < m]
        if len(idx) < MIN_BAND_FRAMES:
            met[f"dropout_{_band_key(b)}"] = float("nan")
            met[f"inharm_{_band_key(b)}"] = float("nan")
            continue
        met[f"dropout_{_band_key(b)}"] = float(np.mean((esn[idx] - ecn[idx]) > DROPOUT_DB) * 100)
        ih_idx = src["inh_idx"][b][src["inh_idx"][b] < m]
        ih = _inharm(S, ih_idx, src["f0"])
        ref = src["src_inh"][b]
        met[f"inharm_{_band_key(b)}"] = float(100.0 * (ih - ref) / max(ref, 1e-9))

    gidx = src["good"][src["good"] < m]
    P = (S[:, gidx] ** 2).mean(axis=1)
    met["centroid_hz"] = float((FREQS * P).sum() / (P.sum() + 1e-20))
    ref_e = P[(FREQS >= ENERGY_REF[0]) & (FREQS < ENERGY_REF[1])].sum()
    for b in ENERGY_BANDS:
        e_b = P[(FREQS >= b[0]) & (FREQS < b[1])].sum()
        met[f"band_{_energy_key(b)}_db"] = float(10 * np.log10((e_b + 1e-20) / (ref_e + 1e-20)))

    return met, _mel_centroid(y)


def _agg(values):
    v = np.asarray(values, float)
    ok = v[np.isfinite(v)]
    if not len(ok):
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"),
                "values": [float(x) for x in v]}
    return {"mean": float(ok.mean()), "min": float(ok.min()), "max": float(ok.max()),
            "values": [float(x) for x in v]}


def _verdict(models, labels):
    """Is the between-model difference bigger than the within-model spread?"""
    a, b = labels
    out = {}
    for key in models[a]["metrics"]:
        A, B = models[a]["metrics"][key], models[b]["metrics"][key]
        rec = {"delta_mean": float(A["mean"] - B["mean"]),
               f"range_{a}": [A["min"], A["max"]], f"range_{b}": [B["min"], B["max"]],
               "spread": float(max(A["max"] - A["min"], B["max"] - B["min"]))}
        if models[a]["n"] < 2 or models[b]["n"] < 2:
            rec["separable"] = None
            rec["note"] = "cannot assess: needs >=2 renders per model"
        elif not (np.isfinite(A["mean"]) and np.isfinite(B["mean"])):
            rec["separable"] = None
            rec["note"] = "cannot assess: metric is nan"
        else:
            disjoint = A["max"] < B["min"] or B["max"] < A["min"]
            rec["separable"] = bool(disjoint)
            gap = max(B["min"] - A["max"], A["min"] - B["max"])
            rec["gap"] = float(gap)
            # Complete separation of two groups is the extreme of the rank-sum
            # test, so its exact two-sided chance under "same distribution" is
            # 2/C(nA+nB, nA): 0.33 at 2v2, 0.10 at 3v3, 0.029 at 4v4. Without
            # this a coin-flip reads as a finding.
            rec["p_separation"] = float(2.0 / comb(models[a]["n"] + models[b]["n"], models[a]["n"]))
            rec["note"] = (f"ranges disjoint by {gap:.3g}, p={rec['p_separation']:.3f}"
                           if disjoint
                           else "ranges overlap: difference is within render noise")
        out[key] = rec
    return out


def score(groups, source_wav, reference=None, inharm_frames=INHARM_FRAMES,
          seed=0, len_tol_hops=LEN_TOL_HOPS, verbose=True):
    """Score rendered wavs grouped by model label against one source.

    groups:    {"model label": [render.wav, ...]}  (renders are made elsewhere)
    reference: label whose mean centroid identity is measured against;
               defaults to the first group, i.e. put the served model first.
    returns:   a JSON-dumpable report dict.
    """
    if not groups:
        raise RuntimeError("no model groups given")
    reference = reference or next(iter(groups))
    if reference not in groups:
        raise RuntimeError(f"reference {reference!r} is not one of {list(groups)}")

    src = prep_source(source_wav, inharm_frames=inharm_frames, seed=seed, verbose=verbose)

    models, cents = {}, {}
    for label, paths in groups.items():
        paths = list(paths)
        if not paths:
            raise RuntimeError(f"model {label!r} has no renders")
        per, cs = [], []
        for p in paths:
            met, c = score_render(p, src, len_tol_hops=len_tol_hops)
            per.append(met)
            cs.append(c)
        models[label] = {"n": len(paths), "renders": paths,
                         "metrics": {k: _agg([m[k] for m in per]) for k in per[0]}}
        cents[label] = cs
        if verbose:
            print(f"  {label}: {len(paths)} render(s) scored")

    ref_c = np.mean(cents[reference], axis=0)
    ref_c = ref_c / (np.linalg.norm(ref_c) + 1e-9)
    for label in models:
        ident = [float(np.dot(ref_c, c)) for c in cents[label]]
        models[label]["metrics"]["identity"] = _agg(ident)

    report = {"source": source_wav, "reference": reference,
              "f0_bands": [_band_key(b) for b in F0_BANDS],
              "band_frames": {_band_key(b): int(len(src["band_idx"][b])) for b in F0_BANDS},
              "identity_floor": IDENTITY_FLOOR,
              "underpowered": [l for l, m in models.items() if m["n"] < 3],
              "models": models,
              "verdict": _verdict(models, list(models)) if len(models) == 2 else None}
    if verbose:
        print_report(report)
    return report


def _cell(a, p=1):
    if not np.isfinite(a["mean"]):
        return f"{'n/a':^17}"
    return f"{a['mean']:6.{p}f} [{a['min']:5.{p}f},{a['max']:6.{p}f}]"


def print_report(report):
    models = report["models"]
    bands = report["f0_bands"]
    w = max(len(l) for l in models) + 1

    bf = report["band_frames"]
    for name, pre in (("level dropout %", "dropout"),
                      ("inharmonic excess % vs source", "inharm")):
        print(f"\n  {name}  -  mean [min, max] over n renders")
        print(f"  {'model':<{w}} n  " + "".join(f"{b:^18}" for b in bands))
        print(f"  {'(src frames)':<{w}}    " + "".join(f"{bf[b]:^18}" for b in bands))
        for label, m in models.items():
            row = "".join(_cell(m["metrics"][f"{pre}_{b}"]) + " " for b in bands)
            print(f"  {label:<{w}} {m['n']}  {row}")

    ekeys = [f"band_{_energy_key(b)}_db" for b in ENERGY_BANDS]
    print(f"\n  overall (sung frames)  -  band energy dB re {ENERGY_REF[0]}-{ENERGY_REF[1]} Hz")
    print(f"  {'model':<{w}} n  {'centroid Hz':^18}" +
          "".join(f"{k[5:]:^18}" for k in ekeys) + f"{'identity':^18}")
    for label, m in models.items():
        row = "".join(_cell(m["metrics"][k]) + " " for k in ["centroid_hz"] + ekeys)
        row += _cell(m["metrics"]["identity"], p=4)
        print(f"  {label:<{w}} {m['n']}  {row}")

    thin = [b for b in bands if bf[b] < 100]
    if thin:
        print(f"\n  WARNING: band(s) {', '.join(thin)} have under 100 source frames "
              f"({', '.join(f'{b}={bf[b]}' for b in thin)}); those cells are anecdote, "
              f"not measurement.")
    if report["underpowered"]:
        print(f"\n  WARNING: {', '.join(report['underpowered'])} have <3 renders; "
              f"the spread is not a reliable noise floor yet.")
    floor = report["identity_floor"]
    for label, m in models.items():
        lo = m["metrics"]["identity"]["min"]
        if np.isfinite(lo) and lo < floor:
            print(f"  WARNING: {label} identity dips to {lo:.4f}, under the {floor} floor.")

    v = report.get("verdict")
    if v:
        a, b = list(models)
        print(f"\n  {a} vs {b}  (difference larger than within-model spread?)  d = {a} - {b}")
        for key, rec in v.items():
            if rec["separable"] is None:
                mark = "  ?  "
            elif rec["separable"]:
                mark = " REAL"
            else:
                mark = " noise"
            print(f"  {key:<22} d={rec['delta_mean']:+8.2f}  spread={rec['spread']:7.2f} "
                  f"{mark}  {rec['note']}")
        na, nb = models[a]["n"], models[b]["n"]
        if na >= 2 and nb >= 2:
            p = 2.0 / comb(na + nb, na)
            print(f"  REAL = the two models' [min,max] do not overlap. At {na}v{nb} renders that "
                  f"happens by chance p={p:.3f}\n  even when the models are identical; 4 renders "
                  f"each gives p=0.029. A REAL verdict on a tiny delta is\n  still a tiny delta - "
                  f"read the magnitude, not just the label.")
    elif len(models) != 2:
        print(f"\n  ({len(models)} models: the is-it-real verdict needs exactly 2.)")


# --------------------------------------------------------------------------- demo

def _tone(dur, f0=440.0, amp=0.3):
    t = np.arange(int(SR * dur)) / SR
    y = amp * (np.sin(2 * np.pi * f0 * t)
               + 0.20 * np.sin(2 * np.pi * 2 * f0 * t)
               + 0.10 * np.sin(2 * np.pi * 3 * f0 * t))
    return y.astype(np.float32)


def _punch(y, at_s, dur_s=0.2, db=-20.0):
    out = y.copy()
    i, j = int(at_s * SR), int((at_s + dur_s) * SR)
    out[i:j] *= 10 ** (db / 20.0)
    return out


def demo(tmpdir=None):
    """Self-check: a known 20 dB, 200 ms hole must show up as a dropout.

    Fails loudly if the metric, the frame gating or the verdict is broken.
    """
    import tempfile
    ctx = tempfile.TemporaryDirectory() if tmpdir is None else None
    d = tmpdir or ctx.name
    dur, hole = 4.0, 0.2
    src = _tone(dur)
    sf.write(f"{d}/src.wav", src, SR)
    # Two renders per model, so the spread and the verdict are exercised. The
    # "_b" render of each pair is 8 dB quieter overall: a real render never
    # lands at the source's absolute level, so if the per-render level
    # normalisation were broken, that global offset (>6 dB) would read as ~100%
    # dropout and the asserts below would catch it. Without this the demo
    # passes even with the normaliser deleted.
    quiet = 10 ** (-8.0 / 20.0)
    noise = 1e-3 * np.random.default_rng(0).standard_normal(len(src)).astype(np.float32)
    sf.write(f"{d}/null_a.wav", src, SR)
    sf.write(f"{d}/null_b.wav", (src + noise) * quiet, SR)
    sf.write(f"{d}/hole_a.wav", _punch(src, 1.0, hole), SR)
    sf.write(f"{d}/hole_b.wav", _punch(src, 2.0, hole) * quiet, SR)

    print(f"  demo: 440 Hz tone, {dur:.0f}s, 20 dB hole of {hole * 1000:.0f} ms punched "
          f"into the 'holed' renders;\n  one render per model is 8 dB down overall to "
          f"exercise the level normalisation")
    rep = score({"null": [f"{d}/null_a.wav", f"{d}/null_b.wav"],
                 "holed": [f"{d}/hole_a.wav", f"{d}/hole_b.wav"]},
                f"{d}/src.wav", verbose=True)

    band = _band_key(F0_BANDS[1])  # 440 Hz lands in 400-550
    key = f"dropout_{band}"
    n_frames = len(rep["models"]["null"]["metrics"][key]["values"])
    null = rep["models"]["null"]["metrics"][key]["mean"]
    holed = rep["models"]["holed"]["metrics"][key]["mean"]
    expect = 100.0 * hole / dur
    ident = rep["models"]["null"]["metrics"]["identity"]["min"]
    sep = rep["verdict"][key]["separable"]
    print(f"\n  check: null dropout {null:.2f}%  holed {holed:.2f}%  "
          f"(expected ~{expect:.1f}%)  null identity {ident:.4f}  separable={sep}")

    if not np.isfinite(holed):
        raise RuntimeError(f"demo: no usable frames in band {band} ({n_frames} renders scored) "
                           f"- pyin is not tracking the synthetic tone")
    if null >= 1.0:
        raise RuntimeError(f"demo: an unmodified copy scored {null:.2f}% dropout, expected <1% "
                           f"- the level normalisation or the sign is wrong")
    if not (0.5 * expect <= holed <= 2.0 * expect):
        raise RuntimeError(f"demo: 20 dB hole scored {holed:.2f}% dropout, expected "
                           f"{0.5 * expect:.1f}-{2.0 * expect:.1f}% - the dropout metric is broken")
    if ident < 0.999:
        raise RuntimeError(f"demo: identity of a near-identical copy is {ident:.4f}, expected "
                           f">0.999 - the centroid is broken")
    if sep is not True:
        raise RuntimeError(f"demo: a 20 dB hole vs no hole was not called separable "
                           f"(separable={sep}) - the verdict is broken")
    print("  demo OK: dropout detects the punched hole, a clean copy scores ~0, "
          "and the verdict separates them.")
    if ctx:
        ctx.cleanup()
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true", help="run the self-check and exit")
    ap.add_argument("--source", help="source vocal wav the renders were made from")
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=GLOB",
                    help="model label and a glob of its renders; repeat per model. "
                         "The first is the identity reference.")
    ap.add_argument("--reference", help="label to measure identity against (default: first)")
    ap.add_argument("--inharm-frames", type=int, default=INHARM_FRAMES)
    ap.add_argument("--len-tol-hops", type=int, default=LEN_TOL_HOPS)
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)

    if a.demo:
        demo()
        return 0
    if not a.source or not a.group:
        ap.error("--source and at least one --group are required (or use --demo)")

    groups = {}
    for g in a.group:
        if "=" not in g:
            ap.error(f"--group must be LABEL=GLOB, got {g!r}")
        label, pat = g.split("=", 1)
        paths = sorted(glob.glob(pat))
        if not paths:
            ap.error(f"no renders matched {pat!r} for {label!r}")
        groups[label] = paths

    rep = score(groups, a.source, reference=a.reference,
                inharm_frames=a.inharm_frames, len_tol_hops=a.len_tol_hops)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
