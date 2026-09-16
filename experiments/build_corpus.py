"""Build a keyshift-tagged training corpus from Brandy's existing samples.

The model fails above 700 Hz because the corpus has 0.17 % of voiced frames
there (median f0 280 Hz, p99 382 Hz). Resampling whole files upward supplies
that register from audio we already have. Resampling also moves formants, which
unconditioned destroys identity (0.983 -> 0.867 at only +4 st), so every shifted
file is TAGGED in its filename and the patched trainer conditions on the tag:
the shift gets its own axis instead of being folded into the speaker.

Tag format is the one the patched dataset.py parses: ``_ks<N>`` before the
extension, N in semitones. Shift 0 files are left untagged.

Usage:  python build_corpus.py <out_dir> <shift,shift,...>   e.g. ... 12,16
"""
import sys, os, glob, math
import numpy as np, librosa, soundfile as sf

SR = 44100
CHUNK_S = 10.0
MIN_RMS = 1e-3
BRICKWALL_HZ = 12000        # drop band-limited sources: a 6 kHz browser capture
                            # teaches the model she has no air (the Conor 8 kHz episode)

def bandwidth(y):
    S = np.abs(librosa.stft(y, n_fft=8192))
    r = librosa.feature.rms(y=y, frame_length=8192, hop_length=2048)[0][:S.shape[1]]
    d = 20*np.log10(r+1e-9)
    S = S[:, d > d.max()-25]
    if S.shape[1] == 0: return 0.0
    f = librosa.fft_frequencies(sr=SR, n_fft=8192)
    m = 20*np.log10(S.mean(1)+1e-12); m -= m.max()
    idx = np.where(m > -45)[0]
    return float(f[idx[-1]]) if len(idx) else 0.0

def shift_up(y, semitones):
    """Resample-based shift: changes pitch AND formants together, which is what
    the keyshift conditioning is there to account for. Duration shortens by the
    same ratio; that is fine, chunks are independent training items."""
    rate = 2.0 ** (semitones/12.0)
    return librosa.resample(y, orig_sr=SR, target_sr=int(SR*rate), res_type='kaiser_best')

def main(out_dir, shifts):
    os.makedirs(out_dir, exist_ok=True)
    srcs = sorted(glob.glob('data/samples/fb17af66-8415-4ffe-81b3-600efe75b6d7/sample_*/vocals.wav'))
    kept = dropped = 0
    n_chunks = n_written = 0
    for p in srcs:
        y, _ = librosa.load(p, sr=SR, mono=True)
        bw = bandwidth(y)
        if bw < BRICKWALL_HZ:
            dropped += 1
            print(f"  DROP {p.split('/')[-2]} bandwidth {bw/1000:.1f} kHz", flush=True)
            continue
        kept += 1
        tag = p.split('/')[-2]
        step = int(CHUNK_S*SR)
        for i in range(0, max(1, len(y)-step), step):
            seg = y[i:i+step]
            if len(seg) < step*0.6 or np.sqrt(np.mean(seg**2)) < MIN_RMS:
                continue
            n_chunks += 1
            base = f"{tag}_{i//step:03d}"
            sf.write(os.path.join(out_dir, base+'.wav'), seg, SR); n_written += 1
            for s in shifts:
                if s == 0: continue
                sf.write(os.path.join(out_dir, f"{base}_ks{s}.wav"), shift_up(seg, s), SR)
                n_written += 1
    print(f"\nsources kept {kept}, dropped {dropped} (band-limited)")
    print(f"chunks {n_chunks}, files written {n_written}, shifts {shifts}")
    print(f"tagged fraction {1-n_chunks/max(n_written,1):.0%}")

if __name__ == '__main__':
    out = sys.argv[1]
    shifts = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 and sys.argv[2] else []
    main(out, shifts)
