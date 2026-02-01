"""Realtime voice conversion pipeline for live karaoke.

Architecture: Audio -> ContentVec (16kHz) -> RMVPE -> SimpleDecoder -> HiFiGAN (22kHz)

Target latency breakdown:
- ContentVec: ~40ms (content feature extraction)
- RMVPE: ~20ms (pitch extraction)
- SimpleDecoder: ~10ms (mel generation)
- HiFiGAN: ~20ms (waveform synthesis)
- Total: <100ms for live performance

This pipeline is optimized for low-latency streaming inference, using
lightweight components that maintain quality while meeting realtime constraints.
"""
import logging
import time
from collections import deque
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SimpleDecoder(nn.Module):
    """Lightweight decoder for realtime voice conversion.

    Converts content features (from ContentVec) and pitch features (from RMVPE)
    into mel spectrograms, conditioned on speaker embedding.

    Architecture optimized for <10ms inference:
    - Linear projections (no convolutions for speed)
    - FiLM conditioning for speaker identity
    - Single hidden layer with GELU activation

    Args:
        content_dim: ContentVec feature dimension (default 768)
        pitch_dim: Pitch embedding dimension (default 256)
        speaker_dim: Speaker embedding dimension (default 256)
        n_mels: Output mel spectrogram bins (default 80 for HiFiGAN)
        hidden_dim: Hidden layer dimension (default 256)
    """

    def __init__(
        self,
        content_dim: int = 768,
        pitch_dim: int = 256,
        speaker_dim: int = 256,
        n_mels: int = 80,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.pitch_dim = pitch_dim
        self.speaker_dim = speaker_dim
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim

        # Input projection: content + pitch -> hidden
        self.input_proj = nn.Linear(content_dim + pitch_dim, hidden_dim)

        # Speaker conditioning (FiLM: Feature-wise Linear Modulation)
        self.speaker_gamma = nn.Linear(speaker_dim, hidden_dim)
        self.speaker_beta = nn.Linear(speaker_dim, hidden_dim)

        # Output projection: hidden -> mel
        self.output_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_mels),
        )

    def forward(
        self,
        content: torch.Tensor,
        pitch: torch.Tensor,
        speaker: torch.Tensor,
    ) -> torch.Tensor:
        """Generate mel spectrogram from content, pitch, and speaker.

        Args:
            content: [B, T, content_dim] content features from ContentVec
            pitch: [B, T, pitch_dim] pitch embeddings
            speaker: [B, speaker_dim] speaker embedding (L2-normalized)

        Returns:
            [B, n_mels, T] mel spectrogram for vocoder
        """
        # Concatenate content and pitch
        x = torch.cat([content, pitch], dim=-1)

        # Project to hidden dimension
        h = self.input_proj(x)

        # Apply FiLM speaker conditioning
        gamma = self.speaker_gamma(speaker).unsqueeze(1)
        beta = self.speaker_beta(speaker).unsqueeze(1)
        h = h * (1 + gamma) + beta

        # Project to mel
        mel = self.output_proj(h)

        # Transpose to [B, n_mels, T] for vocoder
        return mel.transpose(1, 2)


class RealtimePipeline:
    """Realtime voice conversion pipeline for live karaoke.

    Orchestrates:
    1. ContentVec encoder - extracts speaker-independent content (16kHz input)
    2. RMVPE pitch extractor - extracts F0 contour
    3. SimpleDecoder - generates mel spectrogram conditioned on speaker
    4. HiFiGAN vocoder - synthesizes waveform (22kHz output)

    Usage:
        pipeline = RealtimePipeline()
        pipeline.set_speaker_embedding(embedding)

        for chunk in audio_stream:
            output = pipeline.process_chunk(chunk)
            play(output)

    Args:
        device: Torch device (default: CUDA if available)
        contentvec_model: HuggingFace model ID or local path
        vocoder_checkpoint: Path to HiFiGAN checkpoint
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        contentvec_model: Optional[str] = None,
        vocoder_checkpoint: Optional[str] = None,
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Sample rates
        self.sample_rate = 16000
        self.output_sample_rate = 22050

        # Latency tracking
        self._latency_history: Dict[str, deque] = {
            'content_encoder': deque(maxlen=100),
            'pitch_extractor': deque(maxlen=100),
            'decoder': deque(maxlen=100),
            'vocoder': deque(maxlen=100),
            'total': deque(maxlen=100),
        }

        # Speaker embedding
        self._speaker_embedding: Optional[torch.Tensor] = None

        # Initialize components
        self._init_content_encoder(contentvec_model)
        self._init_pitch_extractor()
        self._init_decoder()
        self._init_vocoder(vocoder_checkpoint)

        logger.info(
            f"RealtimePipeline initialized on {self.device}: "
            f"ContentVec -> RMVPE -> SimpleDecoder -> HiFiGAN"
        )

    def _init_content_encoder(self, model_id: Optional[str]):
        """Initialize ContentVec encoder."""
        from ..models.encoder import ContentVecEncoder

        self._content_encoder = ContentVecEncoder(
            output_dim=768,
            layer=12,
            pretrained=model_id,
            device=self.device,
        )
        self._content_encoder.to(self.device)
        logger.debug("ContentVec encoder initialized")

    def _init_pitch_extractor(self):
        """Initialize RMVPE pitch extractor."""
        from ..models.pitch import RMVPEPitchExtractor
        from ..models.encoder import PitchEncoder

        self._pitch_extractor = RMVPEPitchExtractor(
            device=self.device,
            hop_size=320,
            sample_rate=16000,
        )
        self._pitch_extractor.to(self.device)

        self._pitch_encoder = PitchEncoder(output_size=256)
        self._pitch_encoder.to(self.device)

        logger.debug("RMVPE pitch extractor initialized")

    def _init_decoder(self):
        """Initialize SimpleDecoder."""
        self._decoder = SimpleDecoder(
            content_dim=768,
            pitch_dim=256,
            speaker_dim=256,
            n_mels=80,
            hidden_dim=256,
        )
        self._decoder.to(self.device)
        self._decoder.eval()
        logger.debug("SimpleDecoder initialized")

    def _init_vocoder(self, checkpoint: Optional[str]):
        """Initialize HiFiGAN vocoder."""
        from ..models.vocoder import HiFiGANVocoder

        self._vocoder = HiFiGANVocoder(device=self.device)
        if checkpoint:
            self._vocoder.load_checkpoint(checkpoint)
        logger.debug("HiFiGAN vocoder initialized")

    def set_speaker_embedding(self, embedding: np.ndarray) -> None:
        """Set target speaker embedding for voice conversion.

        Args:
            embedding: 256-dim speaker embedding (should be L2-normalized)
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]

        self._speaker_embedding = torch.from_numpy(embedding).to(self.device)
        logger.info("Speaker embedding set")

    def clear_speaker(self) -> None:
        """Clear speaker embedding (audio will pass through unchanged)."""
        self._speaker_embedding = None
        logger.info("Speaker cleared")

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Process audio chunk through voice conversion pipeline.

        Args:
            audio: Input audio at 16kHz, float32, mono

        Returns:
            Converted audio at 22kHz, float32
        """
        total_start = time.perf_counter()

        # Passthrough if no speaker set
        if self._speaker_embedding is None:
            return audio.astype(np.float32)

        audio = np.asarray(audio, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 1. Content encoding (~40ms)
            t0 = time.perf_counter()
            content = self._content_encoder.encode(audio_tensor)
            self._latency_history['content_encoder'].append(
                time.perf_counter() - t0
            )

            # 2. Pitch extraction (~20ms)
            t0 = time.perf_counter()
            f0 = self._pitch_extractor.extract(audio_tensor)
            pitch = self._pitch_encoder(f0)
            self._latency_history['pitch_extractor'].append(
                time.perf_counter() - t0
            )

            # 3. Frame alignment
            n_frames = min(content.shape[1], pitch.shape[1])
            if n_frames == 0:
                return np.zeros(
                    int(len(audio) * self.output_sample_rate / self.sample_rate),
                    dtype=np.float32
                )

            content = content[:, :n_frames, :]
            pitch = pitch[:, :n_frames, :]

            # 4. Decoder (~10ms)
            t0 = time.perf_counter()
            mel = self._decoder(content, pitch, self._speaker_embedding)
            self._latency_history['decoder'].append(time.perf_counter() - t0)

            # 5. Vocoder (~20ms)
            t0 = time.perf_counter()
            output = self._vocoder.synthesize(mel)
            self._latency_history['vocoder'].append(time.perf_counter() - t0)

        # Convert to numpy
        output_np = output.squeeze(0).cpu().numpy()

        # Normalize output
        peak = np.abs(output_np).max()
        if peak > 0.95:
            output_np = output_np * (0.95 / peak)
        elif peak > 0:
            output_np = output_np * (0.9 / peak)

        self._latency_history['total'].append(time.perf_counter() - total_start)

        return output_np.astype(np.float32)

    def get_latency_metrics(self) -> Dict[str, float]:
        """Get average latency for each pipeline component.

        Returns:
            Dict with latency in milliseconds for each component
        """
        metrics = {}
        for name, history in self._latency_history.items():
            if history:
                avg_seconds = np.mean(list(history))
                metrics[f'{name}_ms'] = avg_seconds * 1000
            else:
                metrics[f'{name}_ms'] = 0.0
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        latency = self.get_latency_metrics()
        return {
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'output_sample_rate': self.output_sample_rate,
            'has_speaker': self._speaker_embedding is not None,
            **latency,
        }
