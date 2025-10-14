"""Evaluation metrics for voice synthesis."""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import librosa
from scipy import signal


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                   sample_rate: int = 44100) -> Dict[str, float]:
    """Compute comprehensive metrics for voice synthesis.

    Args:
        predictions: Predicted audio or features
        targets: Target audio or features
        sample_rate: Sample rate for audio

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['mse'] = mean_squared_error(predictions, targets)
    metrics['mae'] = mean_absolute_error(predictions, targets)
    metrics['snr'] = signal_to_noise_ratio(predictions, targets)

    # Spectral metrics
    if predictions.dim() == 1 or (predictions.dim() == 2 and predictions.shape[0] == 1):
        # Audio waveform
        metrics['spectral_convergence'] = spectral_convergence(predictions, targets)
        metrics['log_spectral_distance'] = log_spectral_distance(predictions, targets)
        metrics['mcd'] = mel_cepstral_distortion(predictions, targets, sample_rate)

    # Perceptual metrics
    metrics['pesq'] = pesq_score(predictions, targets, sample_rate)
    metrics['stoi'] = stoi_score(predictions, targets, sample_rate)

    return metrics


def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean squared error.

    Args:
        pred: Predictions
        target: Targets

    Returns:
        MSE value
    """
    return torch.mean((pred - target) ** 2).item()


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean absolute error.

    Args:
        pred: Predictions
        target: Targets

    Returns:
        MAE value
    """
    return torch.mean(torch.abs(pred - target)).item()


def signal_to_noise_ratio(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute signal-to-noise ratio.

    Args:
        pred: Predicted signal
        target: Target signal

    Returns:
        SNR in dB
    """
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean((pred - target) ** 2)

    if noise_power > 0:
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    else:
        return float('inf')


def spectral_convergence(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute spectral convergence.

    Args:
        pred: Predicted waveform
        target: Target waveform

    Returns:
        Spectral convergence value
    """
    # Compute spectrograms
    pred_spec = torch.stft(pred.flatten(), n_fft=2048, hop_length=512,
                           window=torch.hann_window(2048),
                           return_complex=True)
    target_spec = torch.stft(target.flatten(), n_fft=2048, hop_length=512,
                             window=torch.hann_window(2048),
                             return_complex=True)

    # Magnitude spectrograms
    pred_mag = torch.abs(pred_spec)
    target_mag = torch.abs(target_spec)

    # Spectral convergence
    sc = torch.norm(pred_mag - target_mag, p='fro') / torch.norm(target_mag, p='fro')

    return sc.item()


def log_spectral_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute log spectral distance.

    Args:
        pred: Predicted waveform
        target: Target waveform

    Returns:
        Log spectral distance
    """
    # Compute spectrograms
    pred_spec = torch.stft(pred.flatten(), n_fft=2048, hop_length=512,
                           window=torch.hann_window(2048),
                           return_complex=True)
    target_spec = torch.stft(target.flatten(), n_fft=2048, hop_length=512,
                             window=torch.hann_window(2048),
                             return_complex=True)

    # Log magnitude spectrograms
    pred_log = torch.log(torch.abs(pred_spec) + 1e-7)
    target_log = torch.log(torch.abs(target_spec) + 1e-7)

    # LSD
    lsd = torch.sqrt(torch.mean((pred_log - target_log) ** 2))

    return lsd.item()


def mel_cepstral_distortion(pred: torch.Tensor, target: torch.Tensor,
                           sample_rate: int) -> float:
    """Compute mel-cepstral distortion.

    Args:
        pred: Predicted waveform
        target: Target waveform
        sample_rate: Sample rate

    Returns:
        MCD value
    """
    # Convert to numpy
    pred_np = pred.cpu().numpy() if pred.is_cuda else pred.numpy()
    target_np = target.cpu().numpy() if target.is_cuda else target.numpy()

    # Flatten
    pred_np = pred_np.flatten()
    target_np = target_np.flatten()

    # Compute MFCCs
    pred_mfcc = librosa.feature.mfcc(y=pred_np, sr=sample_rate, n_mfcc=13)
    target_mfcc = librosa.feature.mfcc(y=target_np, sr=sample_rate, n_mfcc=13)

    # Align lengths
    min_len = min(pred_mfcc.shape[1], target_mfcc.shape[1])
    pred_mfcc = pred_mfcc[:, :min_len]
    target_mfcc = target_mfcc[:, :min_len]

    # Compute MCD
    diff = pred_mfcc - target_mfcc
    mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0))) * (10.0 / np.log(10.0)) * np.sqrt(2.0)

    return float(mcd)


def pesq_score(pred: torch.Tensor, target: torch.Tensor,
              sample_rate: int) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality) score.

    Args:
        pred: Predicted waveform
        target: Target waveform
        sample_rate: Sample rate

    Returns:
        PESQ score (placeholder implementation)
    """
    # This would require the pesq library
    # For now, return a placeholder based on correlation
    pred_np = pred.cpu().numpy() if pred.is_cuda else pred.numpy()
    target_np = target.cpu().numpy() if target.is_cuda else target.numpy()

    correlation = np.corrcoef(pred_np.flatten(), target_np.flatten())[0, 1]

    # Map correlation to PESQ-like range [1, 4.5]
    pesq = 1.0 + 3.5 * max(0, correlation)

    return float(pesq)


def stoi_score(pred: torch.Tensor, target: torch.Tensor,
              sample_rate: int) -> float:
    """Compute STOI (Short-Term Objective Intelligibility) score.

    Args:
        pred: Predicted waveform
        target: Target waveform
        sample_rate: Sample rate

    Returns:
        STOI score (placeholder implementation)
    """
    # This would require the pystoi library
    # For now, return a placeholder based on spectral similarity
    pred_np = pred.cpu().numpy() if pred.is_cuda else pred.numpy()
    target_np = target.cpu().numpy() if target.is_cuda else target.numpy()

    # Compute spectrograms
    _, _, pred_spec = signal.spectrogram(pred_np.flatten(), fs=sample_rate)
    _, _, target_spec = signal.spectrogram(target_np.flatten(), fs=sample_rate)

    # Normalize
    pred_spec = pred_spec / (np.max(pred_spec) + 1e-10)
    target_spec = target_spec / (np.max(target_spec) + 1e-10)

    # Compute similarity
    min_shape = min(pred_spec.shape[1], target_spec.shape[1])
    pred_spec = pred_spec[:, :min_shape]
    target_spec = target_spec[:, :min_shape]

    similarity = 1.0 - np.mean(np.abs(pred_spec - target_spec))

    return float(similarity)


def compute_f0_metrics(pred_f0: np.ndarray, target_f0: np.ndarray) -> Dict[str, float]:
    """Compute F0 (fundamental frequency) metrics.

    Args:
        pred_f0: Predicted F0 contour
        target_f0: Target F0 contour

    Returns:
        F0 metrics
    """
    # Filter out unvoiced frames
    voiced_mask = (target_f0 > 0) & (pred_f0 > 0)

    if np.sum(voiced_mask) == 0:
        return {
            'f0_rmse': float('inf'),
            'f0_correlation': 0.0,
            'vuv_error': 1.0
        }

    pred_voiced = pred_f0[voiced_mask]
    target_voiced = target_f0[voiced_mask]

    # RMSE in Hz
    f0_rmse = np.sqrt(np.mean((pred_voiced - target_voiced) ** 2))

    # Correlation
    f0_corr = np.corrcoef(pred_voiced, target_voiced)[0, 1]

    # Voiced/Unvoiced error rate
    pred_vuv = pred_f0 > 0
    target_vuv = target_f0 > 0
    vuv_error = np.mean(pred_vuv != target_vuv)

    return {
        'f0_rmse': float(f0_rmse),
        'f0_correlation': float(f0_corr),
        'vuv_error': float(vuv_error)
    }