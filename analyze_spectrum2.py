import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def analyze_bandwidth(wav_path):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Compute STFT
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Find bins for 12-16 kHz range
    band_mask = (freqs >= 12000) & (freqs <= 16000)
    band_magnitude = magnitude[band_mask, :]
    mean_energy_12_16k = np.mean(band_magnitude)
    
    # Also check 14+ kHz
    band_mask_14 = freqs >= 14000
    band_magnitude_14 = magnitude[band_mask_14, :]
    mean_energy_14k = np.mean(band_magnitude_14)
    
    # Overall spectral centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    print(f"File: {wav_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio)/sr:.2f} s")
    print(f"Mean energy 12-16 kHz: {mean_energy_12_16k:.6f}")
    print(f"Mean energy 14+ kHz: {mean_energy_14k:.6f}")
    print(f"Spectral centroid: {spectral_centroid:.1f} Hz")
    print(f"Max freq with energy > 0.001: {freqs[np.where(np.max(magnitude, axis=1) > 0.001)[0][-1]]:.0f} Hz")
    print()
    
    return mean_energy_12_16k, mean_energy_14k

if __name__ == "__main__":
    files = [
        "/home/kp/thordrive/autofusion/autovoice/output/conversions_final/One_Last_Time__v3_G41_bright.wav",
        "/home/kp/thordrive/autofusion/autovoice/output/svcfork_epoch100/William_Singe_-_Up_All_Night_epoch100.wav",
    ]
    results = {}
    for f in files:
        try:
            results[f] = analyze_bandwidth(f)
        except Exception as e:
            print(f"Error analyzing {f}: {e}")
    
    print("=== COMPARISON ===")
    for f, (e12_16, e14) in results.items():
        name = f.split('/')[-1]
        print(f"{name}: 12-16kHz={e12_16:.6f}, 14+kHz={e14:.6f}")
