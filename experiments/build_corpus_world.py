"""Corpus for the WORLD-shifted fine-tune.

TWO fixes over the previous build, one of which invalidated an entire round:

1. DIRECTION. The old builder called
       librosa.resample(y, orig_sr=SR, target_sr=SR*rate)
   which LENGTHENS the signal and therefore LOWERS the pitch. Every file tagged
   `_ks12` came out at ~144 Hz instead of ~578. The model was trained on
   low-register copies while we measured whether it had learned the high
   register, so round 1 and ft33 never tested the hypothesis at all.
   Verified on a 300 Hz tone: the old call gives 150 Hz, +12 st must give 600.

2. METHOD. WORLD (harvest -> cheaptrick -> d4c -> synthesize) scales f0 while
   leaving the spectral envelope UNTOUCHED, so formants stay where they are.
   Resampling drags them along, which is "her sped up", not "her singing
   higher". Measured on her own samples at +16 st:
       resample : median f0 114 Hz (wrong way), identity 0.971
       WORLD    : median f0 726 Hz, 63.1 % of frames >700 Hz, identity 0.987

The `_ks<N>` tag is kept so the keyshift-conditioning patch still has its axis,
but with formants preserved it may no longer be needed - worth its own arm.

Usage: python build_corpus_world.py <out_dir> <tagged_fraction>
"""
import sys, os, glob
import numpy as np, librosa, soundfile as sf, pyworld as pw

SR, CHUNK_S, MIN_RMS, BRICKWALL = 44100, 10.0, 1e-3, 12000
SHIFTS = [12, 16]

def bandwidth(y):
    S = np.abs(librosa.stft(y, n_fft=8192))
    r = librosa.feature.rms(y=y, frame_length=8192, hop_length=2048)[0][:S.shape[1]]
    d = 20*np.log10(r+1e-9); S = S[:, d > d.max()-25]
    if S.shape[1] == 0: return 0.0
    f = librosa.fft_frequencies(sr=SR, n_fft=8192)
    m = 20*np.log10(S.mean(1)+1e-12); m -= m.max()
    i = np.where(m > -45)[0]
    return float(f[i[-1]]) if len(i) else 0.0

def world_shift(y, st):
    """Scale f0, keep the spectral envelope -> formants do not move."""
    x = np.asarray(y, dtype=np.float64)
    f0, t = pw.harvest(x, SR, f0_floor=70.0, f0_ceil=1100.0)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    out = pw.synthesize(f0 * (2.0 ** (st/12.0)), sp, ap, SR)
    return np.asarray(out, dtype=np.float32)

def main(out, frac):
    spk = os.path.join(out, "dataset_raw", "spk_fb17af66")
    os.makedirs(spk, exist_ok=True)
    srcs = sorted(glob.glob('data/samples/fb17af66-8415-4ffe-81b3-600efe75b6d7/sample_*/vocals.wav'))
    rng = np.random.RandomState(0)
    chunks = written = dropped = 0
    for p in srcs:
        y, _ = librosa.load(p, sr=SR, mono=True)
        if bandwidth(y) < BRICKWALL:
            dropped += 1; continue
        tag = p.split('/')[-2]; step = int(CHUNK_S*SR)
        for i in range(0, max(1, len(y)-step), step):
            seg = y[i:i+step]
            if len(seg) < step*0.6 or np.sqrt(np.mean(seg**2)) < MIN_RMS: continue
            chunks += 1
            base = f"{tag}_{i//step:03d}"
            sf.write(os.path.join(spk, base+'.wav'), seg, SR); written += 1
            if rng.rand() < frac/(1.0-frac):
                s = SHIFTS[rng.randint(len(SHIFTS))]
                try:
                    sh = world_shift(seg, s)
                except Exception as e:
                    print(f"  WORLD failed on {base}: {e}"); continue
                if np.abs(sh).max() < 1e-4: continue
                sf.write(os.path.join(spk, f"{base}_ks{s}.wav"), sh, SR); written += 1
    n_tag = len(glob.glob(os.path.join(spk, "*_ks*.wav")))
    print(f"sources kept {len(srcs)-dropped}, dropped {dropped} band-limited")
    print(f"chunks {chunks}, files {written}, tagged {n_tag} ({n_tag/max(written,1):.0%})")
    return spk

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.33)
