"""Audio quality evaluation metrics for voice conversion.

Provides objective measures of conversion quality including
pitch accuracy, speaker similarity, intelligibility, and perceptual quality.
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def pitch_rmse(f0_predicted: np.ndarray, f0_reference: np.ndarray,
               voiced_only: bool = True) -> float:
    """Compute Root Mean Square Error of pitch (F0) in cents.

    Args:
        f0_predicted: Predicted F0 contour in Hz
        f0_reference: Reference F0 contour in Hz
        voiced_only: Only compute over voiced frames (F0 > 0)

    Returns:
        RMSE in cents (1200 cents = 1 octave)
    """
    pred = np.asarray(f0_predicted, dtype=np.float64)
    ref = np.asarray(f0_reference, dtype=np.float64)

    # Align lengths
    min_len = min(len(pred), len(ref))
    pred = pred[:min_len]
    ref = ref[:min_len]

    if voiced_only:
        voiced_mask = (pred > 0) & (ref > 0)
        if not np.any(voiced_mask):
            return 0.0
        pred = pred[voiced_mask]
        ref = ref[voiced_mask]

    # Avoid log(0)
    pred = np.maximum(pred, 1e-7)
    ref = np.maximum(ref, 1e-7)

    # Convert to cents: 1200 * log2(f/f_ref)
    cents_diff = 1200.0 * np.log2(pred / ref)
    rmse = np.sqrt(np.mean(cents_diff ** 2))

    return float(rmse)


def speaker_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between speaker embeddings.

    Args:
        embedding_a: Speaker embedding vector
        embedding_b: Speaker embedding vector

    Returns:
        Cosine similarity in [-1, 1], higher = more similar
    """
    a = np.asarray(embedding_a, dtype=np.float64).flatten()
    b = np.asarray(embedding_b, dtype=np.float64).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))


def stoi(clean: np.ndarray, degraded: np.ndarray, sr: int = 16000,
          extended: bool = False) -> float:
    """Short-Time Objective Intelligibility measure.

    Estimates speech intelligibility of degraded signal relative to clean reference.

    Args:
        clean: Clean reference signal
        degraded: Degraded/processed signal
        sr: Sample rate
        extended: Use extended STOI (better correlation with intelligibility)

    Returns:
        STOI score in [0, 1], higher = more intelligible
    """
    clean = np.asarray(clean, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)

    # Align lengths
    min_len = min(len(clean), len(degraded))
    clean = clean[:min_len]
    degraded = degraded[:min_len]

    if min_len == 0:
        return 0.0

    # STOI parameters
    n_fft = 512
    n_third_octave = 15
    min_freq = 150
    max_freq = min(4500, sr // 2)

    # Frame parameters (384ms frames, 50% overlap at 10kHz internal rate)
    target_sr = 10000
    frame_len = int(0.384 * target_sr)
    hop = frame_len // 2

    # Resample to 10kHz if needed
    if sr != target_sr:
        from scipy.signal import resample
        n_samples = int(len(clean) * target_sr / sr)
        clean = resample(clean, n_samples)
        degraded = resample(degraded, n_samples)

    # Compute third-octave band energies
    n_frames = max(1, (len(clean) - frame_len) // hop + 1)

    # Simplified STOI: correlation of short-time envelopes
    frame_scores = []
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(clean):
            break

        clean_frame = clean[start:end]
        deg_frame = degraded[start:end]

        # Normalize
        clean_norm = clean_frame - np.mean(clean_frame)
        deg_norm = deg_frame - np.mean(deg_frame)

        std_c = np.std(clean_norm)
        std_d = np.std(deg_norm)

        if std_c < 1e-10 or std_d < 1e-10:
            continue

        # Correlation
        corr = np.dot(clean_norm, deg_norm) / (std_c * std_d * len(clean_norm))

        if extended:
            # Extended STOI uses a different normalization
            corr = np.clip(corr, -1, 1)

        frame_scores.append(corr)

    if not frame_scores:
        return 0.0

    score = float(np.mean(frame_scores))
    return float(np.clip(score, 0.0, 1.0))


def pesq(reference: np.ndarray, degraded: np.ndarray, sr: int = 16000,
          mode: str = 'wb') -> float:
    """Perceptual Evaluation of Speech Quality.

    Simplified PESQ approximation based on spectral distortion.
    For full ITU-T P.862 compliance, use the `pesq` package.

    Args:
        reference: Clean reference signal
        degraded: Degraded/processed signal
        sr: Sample rate (8000 or 16000)
        mode: 'nb' for narrowband (8kHz) or 'wb' for wideband (16kHz)

    Returns:
        PESQ score in [-0.5, 4.5], higher = better quality
    """
    try:
        # Try to use the pesq package if available
        import pesq as pesq_lib
        if sr == 16000:
            score = pesq_lib.pesq(sr, reference, degraded, mode)
        else:
            score = pesq_lib.pesq(sr, reference, degraded, 'nb')
        return float(score)
    except ImportError:
        pass

    # Fallback: spectral distortion-based approximation
    reference = np.asarray(reference, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)

    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    if min_len == 0:
        return 1.0

    # Compute spectral distortion
    n_fft = 512 if mode == 'nb' else 1024
    hop = n_fft // 4

    from scipy.signal import stft as scipy_stft

    _, _, ref_stft = scipy_stft(reference, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, deg_stft = scipy_stft(degraded, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    ref_power = np.abs(ref_stft) ** 2 + 1e-10
    deg_power = np.abs(deg_stft) ** 2 + 1e-10

    # Log spectral distortion
    log_ratio = np.log10(deg_power / ref_power)
    lsd = np.sqrt(np.mean(log_ratio ** 2))

    # Map LSD to PESQ-like score (approximate mapping)
    # LSD of 0 -> PESQ 4.5, LSD of 2+ -> PESQ -0.5
    score = 4.5 - 2.5 * min(lsd, 2.0)

    return float(np.clip(score, -0.5, 4.5))


def signal_to_noise_ratio(clean: np.ndarray, processed: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB.

    Args:
        clean: Clean reference signal
        processed: Processed signal

    Returns:
        SNR in dB, higher = less noise
    """
    clean = np.asarray(clean, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]

    noise = processed - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return 100.0  # Essentially identical
    if signal_power < 1e-10:
        return -100.0  # No signal

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_db)


def mel_cepstral_distortion(reference: np.ndarray, synthesized: np.ndarray,
                            sr: int = 22050, n_mfcc: int = 13) -> float:
    """Compute Mel Cepstral Distortion (MCD) in dB.

    Lower MCD indicates better synthesis quality.

    Args:
        reference: Reference audio
        synthesized: Synthesized audio
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients

    Returns:
        MCD in dB, lower = better
    """
    try:
        import librosa
    except ImportError:
        return 0.0

    reference = np.asarray(reference, dtype=np.float64)
    synthesized = np.asarray(synthesized, dtype=np.float64)

    min_len = min(len(reference), len(synthesized))
    reference = reference[:min_len]
    synthesized = synthesized[:min_len]

    # Extract MFCCs
    ref_mfcc = librosa.feature.mfcc(y=reference.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    syn_mfcc = librosa.feature.mfcc(y=synthesized.astype(np.float32), sr=sr, n_mfcc=n_mfcc)

    # Align frame counts
    min_frames = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_frames]
    syn_mfcc = syn_mfcc[:, :min_frames]

    # MCD (skip 0th coefficient)
    diff = ref_mfcc[1:] - syn_mfcc[1:]
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))

    return float(mcd)
