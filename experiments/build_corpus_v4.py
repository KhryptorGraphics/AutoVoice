"""Corpus v4: bright, full-range, formant-corrected.

WHY v4 EXISTS - three measured defects in the v3 (world33) corpus:

1. IT NEVER REACHED THE FAILURE BAND. Her real median f0 is 291 Hz (p99 407,
   0.00 % of frames above 700 Hz). world33 used +12/+16 st, which lands her
   median at 582/733 Hz. Inference needs 700-1100 Hz. The augmentation stopped
   exactly where the model starts failing, so "the model failed to learn the
   high register" was partly "the high register was never in the corpus".
   v4 uses +4/+8/+12/+16/+20 st -> medians 366/462/582/733/923 Hz.

2. IT WAS DARK. Energy relative to 300-3000 Hz, her corpus vs the source stems
   she is converted against:

                  3-8k    8-12k   12-16k  16-20k
       source    -14.2    -16.2    -20.5    -29.6
       hers      -10.6    -26.6    -29.6    -48.3
       deficit    -3.6    +10.5     +9.1    +18.7

   The model learned a dark voice and reproduced it faithfully, and correcting
   it on the OUTPUT made dropouts more audible (lifting HF lifts the inharmonic
   residue in a dropout). Fixed here instead, in the data. Note she is already
   BRIGHTER than the source at 3-8 kHz, so the lift starts at 8 kHz - an
   earlier output-side attempt started at 3.5 kHz and was the wrong shape.

3. FORMANTS - CORRECTED AFTER A LITERATURE CHECK, SCALING IS NOW OFF.
   v4 first scaled the whole CheapTrick envelope by ratio**0.25. That is the
   wrong SHAPE. Joliveau/Smith/Wolfe (JASA 2004) measured real female singers
   holding R1 at its speech value while f0 < R1, then letting R1 RISE TO TRACK
   f0, with the R1=f0 line inside the data's SDs across 600-1000 Hz - i.e. the
   real warp is R1-specific (R1 -> n*f0, n=2 for belt per Bourne & Garnier
   2012), R2 rises only incidentally, R3 barely, and it saturates near 1 kHz.
   A uniform scale drags R3+ WHERE IDENTITY LIVES, so it buys the wrong thing
   at the wrong cost. Set to 1.0 (pure WORLD, formants preserved) - which is
   what world33 did and what scored identity 0.987 vs resampling's 0.971.
   Doing this properly means warping R1 alone and is a separate experiment.

   Also note the published priors on large pitch-shift augmentation are NOT
   encouraging: SingAug found +-1 st beat +-2 st and its "shift toward the
   target region" variant was the worst of three; PDAugment caps at 5 st;
   WeSinger tried +-1 st for range extension, saw "perceptible changes in
   timbre" and "no significant benefit", and dropped it. This corpus uses
   +4..+20 st because the target has ZERO coverage above 700 Hz and inference
   demands it - a different situation from those papers - but that is a
   project-specific bet, not something the literature supports.

GATE: the shifter is verified against a synthetic tone before any file is
written. The inverted-resample bug invalidated two complete rounds; a shifter
is never trusted without ground truth. pyin octave-halves ~40 % of frames on
real material, so real audio cannot serve as that ground truth.

Usage: python build_corpus_v4.py <out_dir>
"""
import sys, os, glob
import numpy as np, librosa, soundfile as sf, pyworld as pw

SR, CHUNK_S, MIN_RMS = 44100, 10.0, 1e-3
SHIFTS = [4, 8, 12, 16, 20]
FORMANT_EXP, FORMANT_CAP = 0.0, 1.0   # see note below
# energy per band relative to 300-3000 Hz, measured on the source stems
TARGET = {(8000, 12000): -16.2, (12000, 16000): -20.5}
LIFT_FROM, MAX_LIFT = 8000.0, 12.0


def world_shift(y, st):
    """Shift f0 by `st` semitones, scaling formants modestly with it."""
    x = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    f0, t = pw.harvest(x, SR, f0_floor=70.0, f0_ceil=1100.0)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    ratio = 2.0 ** (st / 12.0)
    fs = min(ratio ** FORMANT_EXP, FORMANT_CAP)
    if fs != 1.0:
        nb = sp.shape[1]
        idx = np.clip((np.arange(nb) / fs).astype(int), 0, nb - 1)
        sp = np.ascontiguousarray(sp[:, idx])
    return np.asarray(
        pw.synthesize(np.ascontiguousarray(f0 * ratio), sp,
                      np.ascontiguousarray(ap), SR), dtype=np.float32)


def brighten(y):
    """Lift >8 kHz toward the source stems' balance. One-sided and bounded."""
    n_fft, hop = 4096, 1024
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    f = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    mag = np.abs(S)
    r = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0][:mag.shape[1]]
    d = 20 * np.log10(r + 1e-9)
    loud = d > d.max() - 25
    if loud.sum() < 4:
        return y
    ref = np.sqrt((mag[(f >= 300) & (f < 3000)][:, loud] ** 2).sum()) + 1e-12
    gain = np.zeros(len(f))
    for (lo, hi), want in TARGET.items():
        m = (f >= lo) & (f < hi)
        have = 20 * np.log10(np.sqrt((mag[m][:, loud] ** 2).sum()) / ref + 1e-12)
        gain[m] = np.clip(want - have, 0.0, MAX_LIFT)
    # smooth the band edges so this is a tilt, not three shelves
    k = 129
    gain = np.convolve(gain, np.ones(k) / k, mode="same")
    gain *= np.clip(np.log2(np.maximum(f, 1.0) / LIFT_FROM) / 0.5, 0.0, 1.0)
    out = librosa.istft(S * 10 ** (gain[:, None] / 20.0), hop_length=hop, length=len(y))
    p = float(np.abs(out).max())
    return (out / p * 0.99 if p > 0.99 else out).astype(np.float32)


def verify_shifter():
    """Ground-truth gate. Never build a corpus with an unverified shifter."""
    t = np.arange(int(SR * 3)) / SR
    tone = sum(0.4 / (k ** 1.2) * np.sin(2 * np.pi * 300 * k * t)
               for k in range(1, 14)).astype(np.float32)
    ok = True
    for st in SHIFTS:
        exp = 300 * 2 ** (st / 12.0)
        f0, vf, vp = librosa.pyin(world_shift(tone, st), fmin=60, fmax=1500,
                                  sr=SR, frame_length=2048, hop_length=512)
        m = vf & np.isfinite(f0) & (vp > 0.8)
        got = float(np.median(f0[m])) if m.any() else float("nan")
        good = np.isfinite(got) and abs(got - exp) / exp < 0.06
        print(f"    +{st:>2} st  expected {exp:6.0f}  got {got:6.0f}  {'OK' if good else 'WRONG'}")
        ok &= good
    return ok


def _jobs(srcs):
    """One job per (file, chunk). Each chunk gets the plain copy plus a SAMPLE
    of shifts, weighted high: the gap is at 700-1100 Hz, and giving every chunk
    all five shifts makes the corpus 6x and each epoch 6x longer for coverage
    we do not need evenly."""
    rng = np.random.default_rng(0)
    step = int(CHUNK_S * SR)
    out = []
    for p in srcs:
        try:
            n = int(sf.info(p).samplerate and sf.info(p).frames * SR / sf.info(p).samplerate)
        except Exception:
            continue
        for j in range(0, max(1, n - step // 2), step):
            picks = rng.choice(SHIFTS, size=2, replace=False,
                               p=np.array([1.0, 1.5, 2.0, 2.5, 3.0]) / 10.0)
            out.append((p, j, sorted(int(x) for x in picks)))
    return out


def _run(job):
    p, j, shifts = job
    step = int(CHUNK_S * SR)
    try:
        y, _ = librosa.load(p, sr=SR, mono=True, offset=j / SR, duration=CHUNK_S)
    except Exception:
        return 0, 0
    if len(y) < SR or float(np.sqrt(np.mean(y ** 2))) < MIN_RMS:
        return 0, 0
    spk = _OUT
    base = os.path.basename(os.path.dirname(p))   # sample_001, not "vocals"
    tag = f"{base}_{j // step:03d}"
    wrote_p = wrote_s = 0
    try:
        sf.write(os.path.join(spk, tag + ".wav"), brighten(y), SR); wrote_p = 1
    except Exception:
        return 0, 0
    for st in shifts:
        try:
            sh = brighten(world_shift(y, st))
            if float(np.sqrt(np.mean(sh ** 2))) >= MIN_RMS:
                sf.write(os.path.join(spk, f"{tag}_ks{st}.wav"), sh, SR); wrote_s += 1
        except Exception:
            pass
    return wrote_p, wrote_s


_OUT = None


def _init(out):
    global _OUT
    _OUT = out


def main(out_dir):
    import multiprocessing as mp
    print("  GATE: verifying the shifter against a 300 Hz ground-truth tone", flush=True)
    if not verify_shifter():
        sys.exit("  SHIFTER WRONG - refusing to build a corpus")
    print("  shifter verified\n", flush=True)
    spk = os.path.join(out_dir, "dataset_raw", "spk_fb17af66")
    os.makedirs(spk, exist_ok=True)
    # sample_*/vocals.wav ONLY. A **/*.wav glob also matches instrumental.wav -
    # 71 of the 149 files are backing tracks with no voice in them at all.
    srcs = sorted(glob.glob(
        "data/samples/fb17af66-8415-4ffe-81b3-600efe75b6d7/sample_*/vocals.wav"))
    jobs = _jobs(srcs)
    nproc = max(1, min(12, (os.cpu_count() or 2) - 2))
    print(f"  {len(srcs)} samples -> {len(jobs)} chunks, {nproc} workers", flush=True)
    plain = shifted = 0
    with mp.Pool(nproc, initializer=_init, initargs=(spk,)) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(_run, jobs, chunksize=4), 1):
            plain += a; shifted += b
            if i % 50 == 0:
                print(f"    {i}/{len(jobs)} chunks  {plain} plain  {shifted} shifted", flush=True)
    print(f"\n  wrote {plain} plain + {shifted} shifted = {plain + shifted} files", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/corpus_v4")
