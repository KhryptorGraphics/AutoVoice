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
        """Initialize ContentVec encoder with error handling.

        Raises:
            RuntimeError: If ContentVec fails to load (missing model, OOM, etc.)
        """
        try:
            from ..models.encoder import ContentVecEncoder

            self._content_encoder = ContentVecEncoder(
                output_dim=768,
                layer=12,
                pretrained=model_id,
                device=self.device,
            )
            self._content_encoder.to(self.device)
            logger.debug("ContentVec encoder initialized")
        except FileNotFoundError as e:
            logger.error(f"ContentVec model file not found: {e}")
            raise RuntimeError("Failed to initialize ContentVec: model file missing") from e
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during ContentVec loading")
            torch.cuda.empty_cache()
            self._log_gpu_memory()
            raise RuntimeError("Insufficient GPU memory for ContentVec encoder") from e
        except Exception as e:
            logger.error(f"Unexpected error loading ContentVec: {e}")
            raise RuntimeError(f"Failed to initialize ContentVec: {e}") from e

    def _init_pitch_extractor(self):
        """Initialize RMVPE pitch extractor with error handling.

        Raises:
            RuntimeError: If pitch extractor fails to initialize
        """
        try:
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
        except FileNotFoundError as e:
            logger.error(f"RMVPE model file not found: {e}")
            raise RuntimeError("Failed to initialize RMVPE: model file missing") from e
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during RMVPE loading")
            torch.cuda.empty_cache()
            self._log_gpu_memory()
            raise RuntimeError("Insufficient GPU memory for RMVPE pitch extractor") from e
        except Exception as e:
            logger.error(f"Unexpected error loading RMVPE: {e}")
            raise RuntimeError(f"Failed to initialize RMVPE: {e}") from e

    def _init_decoder(self):
        """Initialize SimpleDecoder with error handling.

        Raises:
            RuntimeError: If decoder fails to initialize
        """
        try:
            self._decoder = SimpleDecoder(
                content_dim=768,
                pitch_dim=256,
                speaker_dim=256,
                n_mels=80,
                hidden_dim=256,
            )
            self._decoder.to(self.device)
            self._decoder.train(False)  # Set to evaluation mode
            logger.debug("SimpleDecoder initialized")
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during SimpleDecoder loading")
            torch.cuda.empty_cache()
            self._log_gpu_memory()
            raise RuntimeError("Insufficient GPU memory for SimpleDecoder") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing SimpleDecoder: {e}")
            raise RuntimeError(f"Failed to initialize SimpleDecoder: {e}") from e

    def _init_vocoder(self, checkpoint: Optional[str]):
        """Initialize HiFiGAN vocoder with error handling.

        Raises:
            RuntimeError: If vocoder fails to initialize
        """
        try:
            from ..models.vocoder import HiFiGANVocoder

            self._vocoder = HiFiGANVocoder(device=self.device)
            if checkpoint:
                self._vocoder.load_checkpoint(checkpoint)
            logger.debug("HiFiGAN vocoder initialized")
        except FileNotFoundError as e:
            logger.error(f"HiFiGAN checkpoint not found: {e}")
            raise RuntimeError("Failed to initialize HiFiGAN: checkpoint missing") from e
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during HiFiGAN loading")
            torch.cuda.empty_cache()
            self._log_gpu_memory()
            raise RuntimeError("Insufficient GPU memory for HiFiGAN vocoder") from e
        except Exception as e:
            logger.error(f"Unexpected error loading HiFiGAN: {e}")
            raise RuntimeError(f"Failed to initialize HiFiGAN: {e}") from e

    def set_speaker_embedding(self, embedding: np.ndarray) -> None:
        """Set target speaker embedding for voice conversion with validation.

        Args:
            embedding: 256-dim speaker embedding (will be auto-normalized)

        Raises:
            ValueError: If embedding has wrong shape or contains invalid values
        """
        embedding = np.asarray(embedding, dtype=np.float32)

        # Flatten if needed
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        # Validate shape
        if embedding.shape[0] != 256:
            raise ValueError(
                f"Speaker embedding must be 256-dimensional, got {embedding.shape[0]}"
            )

        # Validate values
        if not np.isfinite(embedding).all():
            raise ValueError("Speaker embedding contains NaN or Inf values")

        # Auto-normalize
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            raise ValueError("Speaker embedding has zero norm")

        if not np.isclose(norm, 1.0, atol=0.01):
            logger.debug(f"Speaker embedding not L2-normalized (norm={norm:.3f}), normalizing")
            embedding = embedding / norm

        # Convert to tensor [1, 256]
        self._speaker_embedding = torch.from_numpy(embedding[np.newaxis, :]).to(self.device)
        logger.info(f"Speaker embedding set (norm={norm:.3f})")

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

        Note:
            On GPU error, returns passthrough audio instead of crashing.
            Empty input returns silence.
        """
        total_start = time.perf_counter()

        # Passthrough if no speaker set
        if self._speaker_embedding is None:
            return audio.astype(np.float32)

        audio = np.asarray(audio, dtype=np.float32)

        # Input validation
        if audio.size == 0:
            logger.warning("Empty audio chunk received, returning silence")
            silence_len = int(0.1 * self.output_sample_rate)  # 100ms silence
            return np.zeros(silence_len, dtype=np.float32)

        if not np.isfinite(audio).all():
            logger.error("Non-finite values in input audio")
            raise ValueError("Input audio contains NaN or Inf values")

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        try:
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

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM during chunk processing, falling back to passthrough")
            torch.cuda.empty_cache()
            self._log_gpu_memory()
            return audio.astype(np.float32)

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during processing: {e}")
                torch.cuda.empty_cache()
                return audio.astype(np.float32)
            raise

        except Exception as e:
            logger.error(f"Unexpected error in process_chunk: {e}", exc_info=True)
            return audio.astype(np.float32)

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

    def _log_gpu_memory(self) -> None:
        """Log current GPU memory state for debugging."""
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                reserved = torch.cuda.memory_reserved(self.device) / 1e9
                logger.error(
                    f"GPU memory state: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved"
                )
            except Exception as e:
                logger.warning(f"Could not log GPU memory: {e}")
