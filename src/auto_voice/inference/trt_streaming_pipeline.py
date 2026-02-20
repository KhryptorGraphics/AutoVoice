"""TensorRT-optimized real-time streaming voice conversion pipeline.

Combines TRT inference engines with overlap-add synthesis for
ultra-low-latency (<50ms) live voice conversion.

Target: Jetson Thor (SM 11.0, 16GB GPU memory, CUDA 13.0)
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class TRTStreamingPipeline:
    """TensorRT-optimized streaming voice conversion pipeline.

    Uses preloaded TRT engines for minimal latency during live conversion.
    Includes overlap-add synthesis for glitch-free continuous output.

    Args:
        engine_dir: Directory containing TRT engine files
        chunk_size_ms: Chunk duration in milliseconds (default: 100ms)
        overlap_ratio: Overlap between chunks (default: 0.5)
        sample_rate: Audio sample rate (default: 24000)
        device: Torch device for processing
    """

    def __init__(
        self,
        engine_dir: str,
        chunk_size_ms: int = 100,
        overlap_ratio: float = 0.5,
        sample_rate: int = 24000,
        device: Optional[torch.device] = None,
    ):
        self.engine_dir = Path(engine_dir)
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.overlap_ratio = overlap_ratio

        # Calculate chunk and hop sizes
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)
        self.hop_size = int(self.chunk_size * (1 - overlap_ratio))
        self.overlap_size = self.chunk_size - self.hop_size

        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TRT contexts (lazy-loaded)
        self._content_ctx = None
        self._pitch_ctx = None
        self._decoder_ctx = None
        self._vocoder_ctx = None
        self._engines_loaded = False

        # Overlap-add buffer
        self.overlap_buffer: Optional[torch.Tensor] = None
        self.crossfade_window = self._create_crossfade_window()

        # Latency tracking
        self._latency_history: List[float] = []
        self._max_latency_history = 100

        # Minimum chunk size (10ms)
        self._min_chunk_size = int(sample_rate * 0.01)

        logger.info(f"TRTStreamingPipeline initialized (engines at {engine_dir})")

    @staticmethod
    def engines_available(engine_dir: str) -> bool:
        """Check if TRT engines exist in the specified directory.

        Args:
            engine_dir: Directory to check for engine files

        Returns:
            True if all required engines exist
        """
        engine_dir = Path(engine_dir)
        required = [
            'content_extractor.trt',
            'pitch_extractor.trt',
            'decoder.trt',
            'vocoder.trt',
        ]
        return all((engine_dir / name).exists() for name in required)

    def _create_crossfade_window(self) -> torch.Tensor:
        """Create Hann crossfade window for overlap-add synthesis."""
        if self.overlap_size <= 0:
            return torch.ones(1, device=self.device)

        fade = torch.hann_window(self.overlap_size * 2, device=self.device)
        fade_in = fade[:self.overlap_size]
        fade_out = fade[self.overlap_size:]

        return torch.stack([fade_in, fade_out])

    def load_engines(self):
        """Preload TRT engines for fast inference.

        Raises:
            RuntimeError: If engines not found or TensorRT unavailable
        """
        if self._engines_loaded:
            return

        from .trt_pipeline import TRTInferenceContext

        engine_paths = {
            'content': self.engine_dir / 'content_extractor.trt',
            'pitch': self.engine_dir / 'pitch_extractor.trt',
            'decoder': self.engine_dir / 'decoder.trt',
            'vocoder': self.engine_dir / 'vocoder.trt',
        }

        # Check all engines exist
        missing = [name for name, path in engine_paths.items() if not path.exists()]
        if missing:
            raise RuntimeError(
                f"Missing TRT engines: {missing}. Run engine builder first."
            )

        # Load engines
        start = time.time()
        self._content_ctx = TRTInferenceContext(str(engine_paths['content']))
        self._pitch_ctx = TRTInferenceContext(str(engine_paths['pitch']))
        self._decoder_ctx = TRTInferenceContext(str(engine_paths['decoder']))
        self._vocoder_ctx = TRTInferenceContext(str(engine_paths['vocoder']))
        self._engines_loaded = True

        elapsed = (time.time() - start) * 1000
        logger.info(f"TRT engines loaded in {elapsed:.0f}ms")

    def _resample(self, audio: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        """Resample audio tensor."""
        if from_sr == to_sr:
            return audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
            target_len = int(audio.shape[2] * to_sr / from_sr)
            resampled = F.interpolate(
                audio, size=target_len, mode='linear', align_corners=False
            )
            return resampled.squeeze(0).squeeze(0)
        return audio

    def _encode_pitch(self, f0: torch.Tensor) -> torch.Tensor:
        """Encode F0 to pitch embeddings using sinusoidal encoding."""
        B, T = f0.shape
        log_f0 = torch.log2(f0.clamp(min=1.0))
        log_f0_norm = (log_f0 - 5.6) / (10.1 - 5.6)
        log_f0_norm = log_f0_norm.clamp(0, 1)
        freqs = torch.arange(1, 129, device=f0.device).float()
        phase = log_f0_norm.unsqueeze(-1) * freqs * torch.pi
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

    def process_chunk(
        self,
        audio_chunk: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Process a single audio chunk using TRT engines.

        Args:
            audio_chunk: Input audio tensor (samples,) or (1, samples)
            speaker_embedding: Target speaker embedding (256-dim)

        Returns:
            Converted audio chunk with overlap-add applied

        Raises:
            RuntimeError: If engines not loaded or invalid inputs
        """
        if not self._engines_loaded:
            self.load_engines()

        start_time = time.time()

        # Validate inputs
        if audio_chunk.numel() < self._min_chunk_size:
            raise RuntimeError(
                f"Chunk too short: {audio_chunk.numel()} samples, "
                f"minimum is {self._min_chunk_size}"
            )

        if speaker_embedding.shape[-1] != 256:
            raise RuntimeError(
                f"Speaker embedding must be 256-dim, got {speaker_embedding.shape[-1]}"
            )

        # Ensure correct shape
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.squeeze(0)

        # Move to device
        audio_chunk = audio_chunk.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        with torch.no_grad():
            # Resample to 16kHz for content/pitch extraction
            audio_16k = self._resample(audio_chunk, self.sample_rate, 16000)

            # Content extraction
            content_out = self._content_ctx.infer({'audio': audio_16k.unsqueeze(0)})
            content_features = content_out['features']

            # Pitch extraction
            pitch_out = self._pitch_ctx.infer({'audio': audio_16k.unsqueeze(0)})
            f0 = pitch_out['f0']

            # Frame alignment
            n_frames = min(content_features.shape[1], f0.shape[1])
            content_features = content_features[:, :n_frames, :]
            f0_aligned = f0[:, :n_frames]
            pitch_embeddings = self._encode_pitch(f0_aligned)

            # Decoding
            decoder_out = self._decoder_ctx.infer({
                'content': content_features,
                'pitch': pitch_embeddings,
                'speaker': speaker_embedding,
            })
            mel = decoder_out['mel']

            # Vocoding
            vocoder_out = self._vocoder_ctx.infer({'mel': mel})
            converted = vocoder_out['audio'].squeeze()

        # Apply overlap-add synthesis
        output = self._apply_overlap_add(converted)

        # Track latency
        elapsed_ms = (time.time() - start_time) * 1000
        self._latency_history.append(elapsed_ms)
        if len(self._latency_history) > self._max_latency_history:
            self._latency_history.pop(0)

        # Clamp output
        output = torch.clamp(output, -1.0, 1.0)

        return output

    def _apply_overlap_add(self, converted: torch.Tensor) -> torch.Tensor:
        """Apply overlap-add synthesis for continuous output."""
        if self.overlap_size <= 0 or self.overlap_buffer is None:
            if self.overlap_size > 0 and converted.shape[0] >= self.overlap_size:
                self.overlap_buffer = converted[-self.overlap_size:].clone()
            return converted

        output = converted.clone()
        output_len = converted.shape[0]

        if output_len >= self.overlap_size:
            fade_in = self.crossfade_window[0][:min(self.overlap_size, output_len)]
            fade_out = self.crossfade_window[1][:min(self.overlap_size, self.overlap_buffer.shape[0])]

            overlap_len = min(self.overlap_size, output_len, self.overlap_buffer.shape[0])
            output[:overlap_len] = (
                output[:overlap_len] * fade_in[:overlap_len] +
                self.overlap_buffer[:overlap_len] * fade_out[:overlap_len]
            )

            self.overlap_buffer = converted[-self.overlap_size:].clone()

        return output

    def reset(self):
        """Reset overlap buffer and latency history."""
        self.overlap_buffer = None
        self._latency_history.clear()

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics from recent chunks."""
        if not self._latency_history:
            return {'min_ms': 0.0, 'max_ms': 0.0, 'avg_ms': 0.0}

        return {
            'min_ms': min(self._latency_history),
            'max_ms': max(self._latency_history),
            'avg_ms': sum(self._latency_history) / len(self._latency_history),
        }

    def get_engine_memory_usage(self) -> int:
        """Get total memory usage of all TRT engines in bytes."""
        if not self._engines_loaded:
            return 0
        total = 0
        total += self._content_ctx.get_memory_usage()
        total += self._pitch_ctx.get_memory_usage()
        total += self._decoder_ctx.get_memory_usage()
        total += self._vocoder_ctx.get_memory_usage()
        return total
