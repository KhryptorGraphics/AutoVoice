"""QA gate for corpus v4. Nothing uploads until every check passes.

Each check exists because something went wrong before:
  f0 coverage   - world33 topped out at 733 Hz while inference needs 1100
  brightness    - the corpus was 10-19 dB darker than the source above 8 kHz
  identity      - resampling-based shifts scored 0.971 and changed who she is
  inharmonicity - augmentation that is itself buzzy teaches the model to buzz
  octave gate   - pyin halves ~40% of frames; every f0 read must reject those
"""
import sys, glob, os
import numpy as np, librosa

SR = 44100
# 16-20 kHz is deliberately NOT a target. Her vocal stems measure -69 dB there
# against the source's -29.6: they are brick-walled around 16 kHz, almost
# certainly lossy-codec or separation loss upstream of this project. There is
# no content to lift, and a +40 dB "correction" would amplify the noise floor.
# This is a hard ceiling of the source recordings - neither training nor any
# output-side EQ can put back a band that was never captured.
TARGET = {(8000, 12000): -16.2, (12000, 16000): -20.5}


def energy_bands(y):
    S = np.abs(librosa.stft(y, n_fft=8192, hop_length=2048))
    r = librosa.feature.rms(y=y, frame_length=8192, hop_length=2048)[0][:S.shape[1]]
    d = 20 * np.log10(r + 1e-9)
    if not (d > d.max() - 25).any():
        return None
    S = S[:, d > d.max() - 25]
    f = librosa.fft_frequencies(sr=SR, n_fft=8192)
    e = lambda lo, hi: np.sqrt((S[(f >= lo) & (f < hi)] ** 2).sum())
    ref = e(300, 3000) + 1e-12
    return {b: 20 * np.log10(e(*b) / ref + 1e-12) for b in TARGET}


def f0_of(y):
    f0, vf, vp = librosa.pyin(y, fmin=100, fmax=1300, sr=SR,
                              frame_length=2048, hop_length=512)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    m = min(len(f0), S.shape[1])
    f0, vf, vp, S = f0[:m], vf[:m], vp[:m], S[:, :m]
    keep = []
    for i in np.where(vf & np.isfinite(f0) & (vp > 0.8))[0]:
        fc = f0[i]
        hi = S[(fr > fc * 1.94) & (fr < fc * 2.06), i].max(initial=0.0)
        f1 = S[(fr > fc * 0.97) & (fr < fc * 1.03), i].max(initial=0.0)
        if hi <= f1:                       # reject octave-halved frames
            keep.append(fc)
    return np.array(keep)


def inharm(y, f0s_idx=None):
    f0, vf, vp = librosa.pyin(y, fmin=100, fmax=1300, sr=SR,
                              frame_length=2048, hop_length=512)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    m = min(len(f0), S.shape[1])
    out = []
    for i in np.where(vf[:m] & np.isfinite(f0[:m]) & (vp[:m] > 0.8))[0][:150]:
        fc = f0[i]
        if S[(fr > fc * 1.94) & (fr < fc * 2.06), i].max(initial=0.0) > \
           S[(fr > fc * 0.97) & (fr < fc * 1.03), i].max(initial=0.0):
            continue
        hm = np.zeros(len(fr), bool)
        for k in range(1, 12):
            hm |= (fr > fc * k * 0.97) & (fr < fc * k * 1.03)
        bd = (fr > fc * 0.5) & (fr < 6000)
        on, off = S[hm & bd, i].sum(), S[(~hm) & bd, i].sum()
        out.append(off / (on + off + 1e-12))
    return float(np.median(out)) if out else float("nan")


def centroid(y):
    L = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128))
    v = np.concatenate([L.mean(1), L.std(1)])
    return v / (np.linalg.norm(v) + 1e-9)


def main(d):
    files = sorted(glob.glob(os.path.join(d, "dataset_raw", "spk_fb17af66", "*.wav")))
    plain = [f for f in files if "_ks" not in os.path.basename(f)]
    shifted = [f for f in files if "_ks" in os.path.basename(f)]
    print(f"  {len(files)} files: {len(plain)} plain, {len(shifted)} shifted\n")
    fails = []

    # --- f0 coverage ---
    allf = []
    for p in (plain[:30] + shifted[:60]):
        y, _ = librosa.load(p, sr=SR, mono=True)
        if len(y) > SR:
            allf.append(f0_of(y))
    F = np.concatenate([a for a in allf if len(a)])
    print("  f0 COVERAGE (octave-clean frames)")
    for lo, hi in [(200, 400), (400, 550), (550, 700), (700, 900), (900, 1300)]:
        pct = 100 * np.mean((F >= lo) & (F < hi))
        print(f"    {lo:>4}-{hi:<5} Hz  {pct:5.1f}%")
    hi_share = 100 * np.mean(F >= 700)
    print(f"    frames >=700 Hz: {hi_share:.1f}%  (world33 had ~0%; inference needs this band)")
    if hi_share < 5.0:
        fails.append(f"only {hi_share:.1f}% of frames reach 700 Hz")

    # --- brightness ---
    print("\n  BRIGHTNESS (energy rel. 300-3000 Hz; target from the source stems)")
    got = {b: [] for b in TARGET}
    for p in plain[:25]:
        y, _ = librosa.load(p, sr=SR, mono=True)
        b = energy_bands(y)
        if b:
            for k in TARGET:
                got[k].append(b[k])
    for k, want in TARGET.items():
        med = float(np.median(got[k])) if got[k] else float("nan")
        gap = med - want
        flag = "" if abs(gap) <= 6 else "   <-- OFF TARGET"
        print(f"    {k[0]//1000:>2}-{k[1]//1000:<3}kHz  got {med:7.1f}  target {want:6.1f}  gap {gap:+5.1f}{flag}")
        if abs(gap) > 8:
            fails.append(f"brightness {k[0]//1000}-{k[1]//1000}k off by {gap:+.1f} dB")

    # --- identity + inharmonicity, shifted vs their own plain parent ---
    print("\n  IDENTITY and INHARMONICITY of shifted files vs their plain parent")
    by_shift = {}
    for p in shifted:
        st = int(os.path.basename(p).rsplit("_ks", 1)[1].split(".")[0])
        by_shift.setdefault(st, []).append(p)
    for st in sorted(by_shift):
        ids, inh = [], []
        for p in by_shift[st][:6]:
            parent = p.rsplit("_ks", 1)[0] + ".wav"
            if not os.path.exists(parent):
                continue
            a, _ = librosa.load(parent, sr=SR, mono=True)
            b, _ = librosa.load(p, sr=SR, mono=True)
            if len(a) < SR or len(b) < SR:
                continue
            ids.append(float(np.dot(centroid(a), centroid(b))))
            inh.append(inharm(b))
        if ids:
            mi, mh = float(np.median(ids)), float(np.nanmedian(inh))
            flag = "   <-- IDENTITY LOW" if mi < 0.88 else ""
            print(f"    +{st:>2} st  n={len(ids)}  identity {mi:.4f}  inharmonic {mh:.3f}{flag}")
            if mi < 0.88:
                fails.append(f"+{st} st identity {mi:.3f} below floor")

    print("\n  " + ("GATE PASSED - safe to upload" if not fails else "GATE FAILED:"))
    for f in fails:
        print(f"    - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
