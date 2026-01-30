"""CUDA kernel interface with PyTorch fallbacks.

Provides GPU-accelerated operations when custom kernels are compiled,
falls back to PyTorch operations otherwise.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to load compiled CUDA extensions
CUDA_KERNELS_AVAILABLE = False
_cuda_module = None

try:
    import auto_voice._cuda_kernels as _cuda_module
    CUDA_KERNELS_AVAILABLE = True
    logger.info("Custom CUDA kernels loaded successfully")
except ImportError:
    logger.info("Custom CUDA kernels not available, using PyTorch fallbacks")

# Kernel execution metrics
_kernel_metrics: List[Dict[str, Any]] = []


def get_kernel_metrics() -> List[Dict[str, Any]]:
    """Get performance metrics for executed CUDA kernels."""
    return _kernel_metrics


def reset_kernel_metrics():
    """Reset kernel execution metrics."""
    global _kernel_metrics
    _kernel_metrics = []


def pitch_detect_gpu(audio_tensor, sample_rate: int, hop_length: int = 512,
                     fmin: float = 50.0, fmax: float = 1100.0):
    """GPU-accelerated pitch detection.

    Falls back to PyTorch implementation if custom kernels unavailable.
    """
    import torch

    if CUDA_KERNELS_AVAILABLE and audio_tensor.is_cuda:
        try:
            result = _cuda_module.pitch_detect(audio_tensor, sample_rate, hop_length, fmin, fmax)
            _kernel_metrics.append({
                'name': 'pitch_detect',
                'calls': 1,
                'device': str(audio_tensor.device),
            })
            return result
        except Exception as e:
            logger.warning(f"CUDA pitch_detect failed, falling back: {e}")

    # PyTorch fallback - autocorrelation-based pitch detection
    n_frames = audio_tensor.shape[-1] // hop_length
    f0 = torch.zeros(n_frames, device=audio_tensor.device)

    for i in range(n_frames):
        start = i * hop_length
        end = min(start + hop_length * 4, audio_tensor.shape[-1])
        frame = audio_tensor[..., start:end]

        if frame.numel() < hop_length:
            continue

        # Simple autocorrelation
        frame_norm = frame - frame.mean()
        corr = torch.nn.functional.conv1d(
            frame_norm.unsqueeze(0).unsqueeze(0),
            frame_norm.unsqueeze(0).unsqueeze(0),
            padding=frame_norm.shape[-1] - 1
        ).squeeze()

        # Find first peak after min lag
        min_lag = int(sample_rate / fmax)
        max_lag = min(int(sample_rate / fmin), corr.shape[-1] // 2)

        if max_lag > min_lag and max_lag < corr.shape[-1]:
            search_region = corr[corr.shape[-1] // 2 + min_lag:corr.shape[-1] // 2 + max_lag]
            if search_region.numel() > 0:
                peak_idx = search_region.argmax() + min_lag
                if peak_idx > 0:
                    f0[i] = sample_rate / peak_idx.float()

    return f0


def synthesis_gpu(features, speaker_embedding, sample_rate: int):
    """GPU-accelerated waveform synthesis.

    Falls back to simple overlap-add if custom kernels unavailable.
    """
    import torch

    if CUDA_KERNELS_AVAILABLE and features.is_cuda:
        try:
            result = _cuda_module.synthesis(features, speaker_embedding, sample_rate)
            _kernel_metrics.append({
                'name': 'synthesis',
                'calls': 1,
                'device': str(features.device),
            })
            return result
        except Exception as e:
            logger.warning(f"CUDA synthesis failed, falling back: {e}")

    # PyTorch fallback - simple feature-to-waveform via iSTFT
    n_fft = 2048
    hop_length = 512

    if features.dim() == 2:
        # Assume features are mel-spectrogram-like, use Griffin-Lim approximation
        magnitude = features.unsqueeze(0) if features.dim() == 2 else features
        phase = torch.randn_like(magnitude) * 2 * 3.14159 - 3.14159
        stft = magnitude * torch.exp(1j * phase)
        waveform = torch.istft(stft, n_fft=n_fft, hop_length=hop_length)
        return waveform

    return features
