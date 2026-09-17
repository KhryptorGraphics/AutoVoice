"""Corpus for Patrick's INITIAL model. Deliberately minimal.

No pitch shifts: this is a first model, and the v5 experiment on the other
profile showed WORLD-shifted augmentation raised inharmonicity at exactly the
pitches it supplied (700-1100 Hz: 54.4 % -> 64.4 % excess) while regressing the
modal range. Published priors agree - no located result where pitch-shift
augmentation alone helped.

No brightness lift: his capture walls at 7.2 kHz (AirPods used as a microphone
forces Bluetooth HFP/SCO, 16 kHz wideband, Nyquist 8 kHz). There is nothing
above 7.2 kHz to lift; a "correction" there would amplify the noise floor. The
fix is a different input device at record time, not DSP.

Just chunk, gate on level, and write.
"""
import sys, os, glob
import numpy as np, librosa, soundfile as sf

SR, CHUNK_S, MIN_RMS = 44100, 10.0, 1e-3
PROFILE = "31b13ae6-9633-4109-bc63-b2709d105081"


def main(out_dir):
    spk = os.path.join(out_dir, "dataset_raw", f"spk_{PROFILE[:8]}")
    os.makedirs(spk, exist_ok=True)
    srcs = sorted(glob.glob(f"data/samples/{PROFILE}/sample_*/vocals.wav"))
    print(f"  {len(srcs)} source files")
    step, n = int(CHUNK_S * SR), 0
    for p in srcs:
        base = os.path.basename(os.path.dirname(p))   # sample_001, never the filename
        y, _ = librosa.load(p, sr=SR, mono=True)
        for j in range(0, max(1, len(y) - step // 2), step):
            seg = y[j:j + step]
            if len(seg) < SR or float(np.sqrt(np.mean(seg ** 2))) < MIN_RMS:
                continue
            # drop near-silent chunks: sing-along capture has long gaps where
            # only the backing track was playing and he was not singing
            r = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
            d = 20 * np.log10(r + 1e-9)
            if float(np.mean(d > d.max() - 35)) < 0.25:
                continue
            sf.write(os.path.join(spk, f"{base}_{j // step:03d}.wav"), seg, SR)
            n += 1
        print(f"    {base}: {n} chunks so far", flush=True)
    print(f"\n  wrote {n} chunks ({n * CHUNK_S / 60:.1f} min)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/corpus_patrick")
