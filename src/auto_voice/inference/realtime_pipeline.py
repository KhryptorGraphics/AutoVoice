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
import os
import time
from collections import deque
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 80-mel universal HiFiGAN matching this pipeline's 22050/256 output contract;
# fetched by scripts/download_pretrained_models.py
DEFAULT_HIFIGAN_CHECKPOINT = os.path.join(
    'models', 'pretrained', 'generator_universal.pth.tar'
)

from ..models.feature_contract import (
    DEFAULT_CONTENT_DIM,
    DEFAULT_PITCH_DIM,
    DEFAULT_SPEAKER_DIM,
)

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
        pitch_dim: Pitch embedding dimension (default 768)
        speaker_dim: Speaker embedding dimension (default 256)
        n_mels: Output mel spectrogram bins (default 80 for HiFiGAN)
        hidden_dim: Hidden layer dimension (default 256)
    """

    def __init__(
        self,
        content_dim: int = DEFAULT_CONTENT_DIM,
        pitch_dim: int = DEFAULT_PITCH_DIM,
        speaker_dim: int = DEFAULT_SPEAKER_DIM,
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

    Buffer Management:
    - Maintains circular buffers (maxlen=100) for latency tracking per component
    - Tracks: content_encoder, pitch_extractor, decoder, vocoder, total
    - Used for realtime performance monitoring via get_latency_metrics()

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
        self._consecutive_chunk_failures = 0

        # Optional trained per-profile decoder (CoMoSVC artifact); when set it
        # replaces SimpleDecoder for mel generation
        self._voice_model: Optional[nn.Module] = None

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

        Args:
            model_id: HuggingFace model ID or local path to ContentVec weights.
                     If None, uses default pretrained model.

        Raises:
            RuntimeError: If ContentVec fails to load (missing model, OOM, etc.)
        """
        try:
            from ..models.encoder import ContentVecEncoder

            self._content_encoder = ContentVecEncoder(
                output_dim=DEFAULT_CONTENT_DIM,
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

            self._pitch_encoder = PitchEncoder(output_size=DEFAULT_PITCH_DIM)
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
                content_dim=DEFAULT_CONTENT_DIM,
                pitch_dim=DEFAULT_PITCH_DIM,
                speaker_dim=DEFAULT_SPEAKER_DIM,
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

        Args:
            checkpoint: Path to HiFiGAN checkpoint file (.ckpt or .pt).
                       If None, uses default pretrained weights.

        Raises:
            RuntimeError: If vocoder fails to initialize
        """
        try:
            from ..models.vocoder import HiFiGANVocoder

            self._vocoder = HiFiGANVocoder(device=self.device)
            if not checkpoint and os.path.exists(DEFAULT_HIFIGAN_CHECKPOINT):
                checkpoint = DEFAULT_HIFIGAN_CHECKPOINT
            if checkpoint:
                # load_checkpoint returns False instead of raising; a random
                # vocoder produces noise, so a configured checkpoint must load
                if not self._vocoder.load_checkpoint(checkpoint):
                    raise RuntimeError(f"HiFiGAN checkpoint failed to load: {checkpoint}")
                logger.info(f"HiFiGAN vocoder loaded from {checkpoint}")
            else:
                logger.warning(
                    "HiFiGAN vocoder initialized WITHOUT weights (no checkpoint "
                    f"given and {DEFAULT_HIFIGAN_CHECKPOINT} not found)"
                )
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

    def load_voice_model(self, model_path: str) -> None:
        """Load a profile's trained model to drive mel generation.

        Accepts the training artifacts this repo produces (full-model and
        self-contained adapter checkpoints); the artifact family and dims are
        derived from the state dict. The trained decoder replaces
        SimpleDecoder in process_chunk. Trained artifacts are 80-mel, which
        matches this pipeline's HiFiGAN output contract.

        Raises:
            FileNotFoundError: If model_path does not exist.
            RuntimeError: If the checkpoint is unrecognized or deltas-only.
        """
        from .model_manager import build_voice_model_from_checkpoint

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model checkpoint not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self._voice_model = build_voice_model_from_checkpoint(
            checkpoint, model_path, self.device
        )
        logger.info(f"RealtimePipeline serving trained voice model from {model_path}")

    def clear_voice_model(self) -> None:
        """Drop any loaded per-profile model (pipeline instances are cached
        and reused across jobs; a stale model must not leak between profiles)."""
        self._voice_model = None

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
                expected_len = int(round(len(audio) * self.output_sample_rate / self.sample_rate))
                if n_frames == 0:
                    return np.zeros(expected_len, dtype=np.float32)

                content = content[:, :n_frames, :]
                pitch = pitch[:, :n_frames, :]

                # Retime features from the input analysis grid (16k, hop 320:
                # 20ms/frame) to the vocoder's output grid (22.05k, hop 256:
                # ~11.6ms/frame). Without this the vocoder renders 20ms of
                # content in 11.6ms and the output is time-compressed by ~43%.
                vocoder_hop = 256
                target_frames = max(1, int(round(expected_len / vocoder_hop)))
                if target_frames != n_frames:
                    content = F.interpolate(
                        content.transpose(1, 2), size=target_frames,
                        mode='linear', align_corners=False,
                    ).transpose(1, 2)
                    pitch = F.interpolate(
                        pitch.transpose(1, 2), size=target_frames,
                        mode='linear', align_corners=False,
                    ).transpose(1, 2)

                # 4. Decoder (~10ms) — trained per-profile model when loaded,
                # otherwise the untrained SimpleDecoder placeholder
                t0 = time.perf_counter()
                voice_model = getattr(self, '_voice_model', None)
                if voice_model is not None:
                    mel = voice_model.infer(content, pitch, self._speaker_embedding)
                else:
                    mel = self._decoder(content, pitch, self._speaker_embedding)
                self._latency_history['decoder'].append(time.perf_counter() - t0)

                # 5. Vocoder (~20ms)
                t0 = time.perf_counter()
                output = self._vocoder.synthesize(mel)
                self._latency_history['vocoder'].append(time.perf_counter() - t0)

            # Convert to numpy and enforce the duration contract exactly
            output_np = output.squeeze(0).cpu().numpy()
            if len(output_np) > expected_len:
                output_np = output_np[:expected_len]
            elif len(output_np) < expected_len:
                output_np = np.pad(output_np, (0, expected_len - len(output_np)))

            # Normalize output
            peak = np.abs(output_np).max()
            if peak > 0.95:
                output_np = output_np * (0.95 / peak)
            elif peak > 0:
                output_np = output_np * (0.9 / peak)

            self._latency_history['total'].append(time.perf_counter() - total_start)

            self._consecutive_chunk_failures = 0
            return output_np.astype(np.float32)

        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during chunk processing, falling back to passthrough")
            torch.cuda.empty_cache()
            self._log_gpu_memory()
            self._note_chunk_failure(e)
            return audio.astype(np.float32)

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during processing: {e}")
                torch.cuda.empty_cache()
                self._note_chunk_failure(e)
                return audio.astype(np.float32)
            raise

        except Exception as e:
            logger.error(f"Unexpected error in process_chunk: {e}", exc_info=True)
            self._note_chunk_failure(e)
            return audio.astype(np.float32)

    # Per-chunk passthrough absorbs transient errors, but a failure streak
    # means the whole stream is unconverted audio — fail instead.
    MAX_CONSECUTIVE_CHUNK_FAILURES = 5

    def _note_chunk_failure(self, exc: Exception) -> None:
        failures = getattr(self, '_consecutive_chunk_failures', 0) + 1
        self._consecutive_chunk_failures = failures
        if failures >= self.MAX_CONSECUTIVE_CHUNK_FAILURES:
            raise RuntimeError(
                f"{failures} consecutive chunk conversion failures; "
                "refusing to keep returning passthrough audio"
            ) from exc

    def get_latency_metrics(self) -> Dict[str, float]:
        """Get average latency for each pipeline component from circular buffers.

        Computes rolling average from last 100 measurements per component.
        Useful for monitoring realtime performance and detecting bottlenecks.

        Returns:
            Dict with keys: content_encoder_ms, pitch_extractor_ms, decoder_ms,
            vocoder_ms, total_ms. Returns 0.0 if no measurements recorded yet.
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
        """Get comprehensive pipeline metrics including latency and configuration.

        Returns:
            Dict containing:
            - device: Current torch device (cuda/cpu)
            - sample_rate: Input audio sample rate (16000)
            - output_sample_rate: Output audio sample rate (22050)
            - has_speaker: Whether speaker embedding is loaded
            - *_ms: Average latency per component (from circular buffers)
        """
        latency = self.get_latency_metrics()
        return {
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'output_sample_rate': self.output_sample_rate,
            'has_speaker': self._speaker_embedding is not None,
            **latency,
        }

    def _log_gpu_memory(self) -> None:
        """Log current GPU memory state for debugging OOM errors.

        Logs allocated and reserved memory in GB. Called after CUDA errors
        to help diagnose memory issues during realtime processing.
        """
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
