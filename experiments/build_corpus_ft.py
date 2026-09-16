"""Corpus for fine-tuning FROM the served model (sota_ep480).

Two changes from round 1, both driven by what round 1 showed:

1. TAGGED FRACTION 33 %, not 67 %. At 67 % the model spends most of its capacity
   on resampled audio whose formants have moved, and round 1 degraded steadily
   with training (early-half 90.3 % -> late-half 94.5 % dropout). The prior
   in-house run used 47 %; this goes lower still.
2. SPEAKER DIR `spk_fb17af66`, matching the served config's speaker map. Round 1
   used `brandy`, which would put the embedding on a different index and make a
   fine-tune from G_480 meaningless.

Usage: python build_corpus_ft.py <out_dir> <tagged_fraction>
"""
import sys, os, glob
import numpy as np, librosa, soundfile as sf

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

def main(out, frac):
    spk = os.path.join(out, "dataset_raw", "spk_fb17af66")
    os.makedirs(spk, exist_ok=True)
    srcs = sorted(glob.glob('data/samples/fb17af66-8415-4ffe-81b3-600efe75b6d7/sample_*/vocals.wav'))
    rng = np.random.RandomState(0)
    chunks, written, dropped = 0, 0, 0
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
            # only a FRACTION of chunks get a shifted copy, and each gets ONE
            # shift rather than both - that is what takes 67 % down to ~33 %.
            if rng.rand() < frac/(1.0-frac):
                s = SHIFTS[rng.randint(len(SHIFTS))]
                rate = 2.0 ** (s/12.0)
                sh = librosa.resample(seg, orig_sr=SR, target_sr=int(SR*rate), res_type='kaiser_best')
                sf.write(os.path.join(spk, f"{base}_ks{s}.wav"), sh, SR); written += 1
    n_tag = len(glob.glob(os.path.join(spk, "*_ks*.wav")))
    print(f"sources kept {len(srcs)-dropped}, dropped {dropped} band-limited")
    print(f"chunks {chunks}, files {written}, tagged {n_tag} ({n_tag/max(written,1):.0%})")

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.33)
