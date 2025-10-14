"""Voice analysis and characterization."""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from .features import FeatureExtractor
from .processor import AudioProcessor

logger = logging.getLogger(__name__)


class VoiceAnalyzer:
    """Analyze voice characteristics and patterns."""

    def __init__(self, sample_rate: int = 44100, device: Optional[torch.device] = None):
        """Initialize voice analyzer.

        Args:
            sample_rate: Sample rate
            device: Device for processing
        """
        self.sample_rate = sample_rate
        self.device = device or torch.device('cpu')
        self.feature_extractor = FeatureExtractor(sample_rate, device)
        self.processor = AudioProcessor(sample_rate, device)

    def analyze_voice_characteristics(self, audio: torch.Tensor) -> Dict:
        """Analyze comprehensive voice characteristics.

        Args:
            audio: Input audio tensor

        Returns:
            Dictionary of voice characteristics
        """
        # Normalize audio
        audio = self.processor.normalize_audio(audio)

        # Extract features
        features = self.feature_extractor.extract_all_features(audio)

        # Analyze pitch characteristics
        pitch_stats = self._analyze_pitch_statistics(features['pitch'])

        # Analyze timbre
        timbre = self._analyze_timbre(features['mfcc'], features['mel_spectrogram'])

        # Analyze rhythm
        rhythm = self._analyze_rhythm(audio)

        # Voice quality assessment
        quality = self._assess_voice_quality(audio, features)

        return {
            'pitch_statistics': pitch_stats,
            'timbre': timbre,
            'rhythm': rhythm,
            'voice_quality': quality,
            'formants': features['formants'].tolist(),
            'spectral_centroid_mean': features['spectral_centroid'].mean().item(),
            'spectral_bandwidth_mean': features['spectral_bandwidth'].mean().item()
        }

    def _analyze_pitch_statistics(self, pitch: torch.Tensor) -> Dict:
        """Analyze pitch statistics.

        Args:
            pitch: Pitch contour tensor

        Returns:
            Pitch statistics
        """
        # Filter out zeros (unvoiced frames)
        voiced_pitch = pitch[pitch > 0]

        if len(voiced_pitch) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0
            }

        return {
            'mean': voiced_pitch.mean().item(),
            'std': voiced_pitch.std().item(),
            'min': voiced_pitch.min().item(),
            'max': voiced_pitch.max().item(),
            'range': (voiced_pitch.max() - voiced_pitch.min()).item()
        }

    def _analyze_timbre(self, mfcc: torch.Tensor,
                       mel_spec: torch.Tensor) -> Dict:
        """Analyze voice timbre.

        Args:
            mfcc: MFCC features
            mel_spec: Mel spectrogram

        Returns:
            Timbre characteristics
        """
        # MFCC statistics
        mfcc_mean = mfcc.mean(dim=-1)
        mfcc_std = mfcc.std(dim=-1)

        # Spectral contrast
        mel_mean = mel_spec.mean(dim=-1)
        mel_std = mel_spec.std(dim=-1)

        # Brightness (high frequency energy ratio)
        high_freq_energy = mel_spec[mel_spec.shape[0] // 2:].mean()
        low_freq_energy = mel_spec[:mel_spec.shape[0] // 2].mean()
        brightness = high_freq_energy / (low_freq_energy + 1e-10)

        return {
            'mfcc_mean': mfcc_mean.tolist(),
            'mfcc_std': mfcc_std.tolist(),
            'brightness': brightness.item(),
            'spectral_contrast': (mel_std.mean() / (mel_mean.mean() + 1e-10)).item()
        }

    def _analyze_rhythm(self, audio: torch.Tensor) -> Dict:
        """Analyze rhythmic patterns.

        Args:
            audio: Input audio tensor

        Returns:
            Rhythm characteristics
        """
        # Compute onset strength
        hop_length = 512
        onset_env = self._compute_onset_envelope(audio, hop_length)

        # Tempo estimation (simplified)
        tempo = self._estimate_tempo(onset_env)

        # Rhythm regularity
        regularity = self._compute_rhythm_regularity(onset_env)

        return {
            'tempo_bpm': tempo,
            'rhythm_regularity': regularity,
            'onset_rate': len(onset_env[onset_env > onset_env.mean()]) / \
                         (audio.shape[-1] / self.sample_rate)
        }

    def _compute_onset_envelope(self, audio: torch.Tensor,
                               hop_length: int) -> torch.Tensor:
        """Compute onset strength envelope.

        Args:
            audio: Input audio tensor
            hop_length: Hop length for analysis

        Returns:
            Onset envelope
        """
        # Compute spectral flux
        spec = torch.stft(audio.flatten(), n_fft=2048, hop_length=hop_length,
                         window=torch.hann_window(2048).to(self.device),
                         return_complex=True)
        magnitude = torch.abs(spec)

        # Compute flux (positive differences)
        flux = torch.diff(magnitude, dim=1)
        flux = torch.relu(flux).sum(dim=0)

        return flux

    def _estimate_tempo(self, onset_env: torch.Tensor) -> float:
        """Estimate tempo from onset envelope.

        Args:
            onset_env: Onset strength envelope

        Returns:
            Estimated tempo in BPM
        """
        # Autocorrelation for tempo
        onset_np = onset_env.cpu().numpy() if onset_env.is_cuda else onset_env.numpy()

        # Compute autocorrelation
        corr = np.correlate(onset_np, onset_np, mode='full')
        corr = corr[len(corr) // 2:]

        # Find peaks in tempo range (60-200 BPM)
        hop_duration = 512 / self.sample_rate
        min_lag = int(60 / (200 * 60 * hop_duration))  # 200 BPM
        max_lag = int(60 / (60 * 60 * hop_duration))   # 60 BPM

        if min_lag < len(corr) and max_lag < len(corr):
            corr_tempo = corr[min_lag:max_lag]
            if len(corr_tempo) > 0:
                peak_lag = np.argmax(corr_tempo) + min_lag
                tempo = 60 / (peak_lag * hop_duration)
                return float(tempo)

        return 120.0  # Default tempo

    def _compute_rhythm_regularity(self, onset_env: torch.Tensor) -> float:
        """Compute rhythm regularity score.

        Args:
            onset_env: Onset strength envelope

        Returns:
            Regularity score (0-1)
        """
        # Find peaks
        onset_np = onset_env.cpu().numpy() if onset_env.is_cuda else onset_env.numpy()
        threshold = np.mean(onset_np) + np.std(onset_np)
        peaks = np.where(onset_np > threshold)[0]

        if len(peaks) < 2:
            return 0.0

        # Compute inter-onset intervals
        intervals = np.diff(peaks)

        if len(intervals) == 0:
            return 0.0

        # Regularity as inverse of coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval > 0:
            cv = std_interval / mean_interval
            regularity = 1.0 / (1.0 + cv)
            return float(regularity)

        return 0.0

    def _assess_voice_quality(self, audio: torch.Tensor, features: Dict) -> Dict:
        """Assess overall voice quality.

        Args:
            audio: Input audio tensor
            features: Extracted features

        Returns:
            Voice quality assessment
        """
        # Signal-to-noise ratio estimation
        snr = self._estimate_snr(audio)

        # Harmonic-to-noise ratio
        hnr = self._estimate_hnr(audio)

        # Voice presence ratio
        voiced_frames = (features['pitch'] > 0).float().mean().item()

        return {
            'snr_db': snr,
            'hnr_db': hnr,
            'voiced_ratio': voiced_frames,
            'clarity_score': min(1.0, (snr / 40.0) * 0.5 + (hnr / 20.0) * 0.3 + voiced_frames * 0.2)
        }

    def _estimate_snr(self, audio: torch.Tensor) -> float:
        """Estimate signal-to-noise ratio.

        Args:
            audio: Input audio tensor

        Returns:
            Estimated SNR in dB
        """
        audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        audio_flat = audio_np.flatten()

        # Simple energy-based SNR estimation
        signal_power = np.mean(audio_flat ** 2)

        # Estimate noise from quiet parts
        threshold = np.percentile(np.abs(audio_flat), 10)
        noise = audio_flat[np.abs(audio_flat) < threshold]

        if len(noise) > 0:
            noise_power = np.mean(noise ** 2)
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return float(snr)

        return 40.0  # Default high SNR

    def _estimate_hnr(self, audio: torch.Tensor) -> float:
        """Estimate harmonic-to-noise ratio.

        Args:
            audio: Input audio tensor

        Returns:
            Estimated HNR in dB
        """
        # Simplified HNR estimation using autocorrelation
        audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        audio_flat = audio_np.flatten()

        # Compute autocorrelation
        corr = np.correlate(audio_flat, audio_flat, mode='full')
        corr = corr[len(corr) // 2:]

        # Find first peak (fundamental period)
        d = np.diff(corr)
        if len(np.where(d > 0)[0]) > 0:
            start = np.where(d > 0)[0][0]
            if start < len(corr) - 1:
                peak_val = np.max(corr[start:])
                noise_floor = np.mean(corr[int(len(corr) * 0.8):])

                if noise_floor > 0:
                    hnr = 10 * np.log10(peak_val / noise_floor)
                    return float(hnr)

        return 15.0  # Default moderate HNR

    def compare_voices(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Dict:
        """Compare two voice samples.

        Args:
            audio1: First audio sample
            audio2: Second audio sample

        Returns:
            Comparison results
        """
        # Analyze both voices
        char1 = self.analyze_voice_characteristics(audio1)
        char2 = self.analyze_voice_characteristics(audio2)

        # Compute similarities
        pitch_diff = abs(char1['pitch_statistics']['mean'] -
                        char2['pitch_statistics']['mean'])

        # Timbre similarity (cosine similarity of MFCC means)
        mfcc1 = torch.tensor(char1['timbre']['mfcc_mean'])
        mfcc2 = torch.tensor(char2['timbre']['mfcc_mean'])
        timbre_sim = torch.cosine_similarity(mfcc1, mfcc2, dim=0).item()

        # Rhythm similarity
        tempo_diff = abs(char1['rhythm']['tempo_bpm'] -
                        char2['rhythm']['tempo_bpm'])

        return {
            'pitch_difference_hz': pitch_diff,
            'timbre_similarity': timbre_sim,
            'tempo_difference_bpm': tempo_diff,
            'overall_similarity': (timbre_sim * 0.5 +
                                 (1.0 - min(pitch_diff / 100, 1.0)) * 0.3 +
                                 (1.0 - min(tempo_diff / 50, 1.0)) * 0.2)
        }