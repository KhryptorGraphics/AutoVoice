#!/usr/bin/env python3
"""Realtime Voice Conversion Pipeline for AutoVoice.

Optimized for low-latency karaoke applications.
Architecture: ContentVec -> RMVPE -> Simple Decoder -> HiFiGAN

Design choices for low latency:
- ContentVec (lighter than Whisper) for content extraction
- Streaming-friendly chunk processing
- HiFiGAN vocoder (faster than BigVGAN)
- FP16 inference throughout
"""

import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class RealtimeConfig:
    """Configuration for realtime pipeline."""
    sample_rate: int = 22050  # Lower SR for speed
    hop_length: int = 256
    chunk_size_ms: int = 100  # Process 100ms chunks
    overlap_ms: int = 20  # 20ms crossfade overlap
    content_dim: int = 768
    speaker_dim: int = 256
    mel_channels: int = 80
    device: str = "cuda"
    fp16: bool = True


class RealtimeVoiceConverter:
    """Low-latency voice conversion for karaoke applications.

    Pipeline:
    1. ContentVec extracts linguistic content (speaker-invariant)
    2. RMVPE extracts pitch (F0) contour
    3. Simple decoder combines content + pitch + speaker embedding
    4. HiFiGAN synthesizes waveform

    Optimizations:
    - Chunk-based streaming processing
    - FP16 inference
    - Warm model caches
    - Minimal CPU-GPU transfers
    """

    def __init__(self, config: Optional[RealtimeConfig] = None):
        self.config = config or RealtimeConfig()
        self.device = torch.device(self.config.device)

        # Models (lazy loaded)
        self._contentvec = None
        self._rmvpe = None
        self._vocoder = None
        self._decoder = None

        # Chunk parameters
        self.chunk_samples = int(self.config.sample_rate * self.config.chunk_size_ms / 1000)
        self.overlap_samples = int(self.config.sample_rate * self.config.overlap_ms / 1000)

        # Buffers for streaming
        self._prev_chunk = None

        logger.info(f"RealtimeVoiceConverter initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Sample rate: {self.config.sample_rate}Hz")
        logger.info(f"  Chunk size: {self.config.chunk_size_ms}ms ({self.chunk_samples} samples)")

    def _load_contentvec(self):
        """Load ContentVec encoder for content extraction."""
        if self._contentvec is not None:
            return self._contentvec

        logger.info("Loading ContentVec encoder...")
        try:
            from auto_voice.models.encoder import ContentVecEncoder
            self._contentvec = ContentVecEncoder(
                output_dim=self.config.content_dim,
                pretrained="lengyue233/content-vec-best",
                device=self.device
            )
            self._contentvec.to(self.device)

            # FP16 is handled inside ContentVecEncoder on first encode call
            # The model is lazily loaded, so we can't convert to fp16 here

        except ImportError:
            # Fallback: use HuBERT directly
            logger.warning("ContentVec not available, using HuBERT fallback")
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            self._contentvec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/hubert-base-ls960"
            )
            self._contentvec = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self._contentvec = self._contentvec.to(self.device)
            if self.config.fp16:
                self._contentvec = self._contentvec.half()

        return self._contentvec

    def _load_rmvpe(self):
        """Load RMVPE pitch extractor."""
        if self._rmvpe is not None:
            return self._rmvpe

        logger.info("Loading RMVPE pitch extractor...")
        try:
            from auto_voice.models.pitch import RMVPEPitchExtractor
            rmvpe_path = Path("models/pretrained/rmvpe.pt")
            self._rmvpe = RMVPEPitchExtractor(
                pretrained=str(rmvpe_path) if rmvpe_path.exists() else None,
                device=self.device
            )
            self._rmvpe.to(self.device)
            if self.config.fp16:
                self._rmvpe = self._rmvpe.half()
        except ImportError:
            # Fallback: use Seed-VC's RMVPE
            logger.info("Using Seed-VC RMVPE implementation")
            from modules.rmvpe import RMVPE
            from hf_utils import load_custom_model_from_hf
            model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
            self._rmvpe = RMVPE(model_path, is_half=self.config.fp16, device=self.device)

        return self._rmvpe

    def _load_vocoder(self):
        """Load HiFiGAN vocoder for fast synthesis."""
        if self._vocoder is not None:
            return self._vocoder

        logger.info("Loading HiFiGAN vocoder...")
        try:
            from modules.hifigan.generator import HiFTGenerator
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            import yaml

            hift_config_path = Path(__file__).parent.parent / "models/seed-vc/configs/hifigan.yml"
            if hift_config_path.exists():
                hift_config = yaml.safe_load(open(hift_config_path, 'r'))
                self._vocoder = HiFTGenerator(
                    **hift_config['hift'],
                    f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor'])
                )
            else:
                # Minimal HiFiGAN config
                self._vocoder = HiFTGenerator()

            # Try to load pretrained weights
            from hf_utils import load_custom_model_from_hf
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            self._vocoder.load_state_dict(torch.load(hift_path, map_location='cpu'))

            self._vocoder = self._vocoder.to(self.device)
            if self.config.fp16:
                self._vocoder = self._vocoder.half()

        except Exception as e:
            logger.warning(f"HiFiGAN loading failed: {e}, using BigVGAN fallback")
            from auto_voice.models.vocoder import BigVGANVocoder
            self._vocoder = BigVGANVocoder(device=self.device)

        return self._vocoder

    def _build_simple_decoder(self):
        """Build lightweight decoder for realtime inference.

        Simple architecture: Linear projections + small transformer
        Maps (content + pitch + speaker) -> mel spectrogram
        """
        if self._decoder is not None:
            return self._decoder

        logger.info("Building realtime decoder...")

        class SimpleDecoder(nn.Module):
            def __init__(self, content_dim=768, speaker_dim=256, mel_dim=80):
                super().__init__()
                hidden_dim = 512

                # Project inputs
                self.content_proj = nn.Linear(content_dim, hidden_dim)
                self.speaker_proj = nn.Linear(speaker_dim, hidden_dim)
                self.pitch_proj = nn.Linear(1, hidden_dim)

                # Small transformer for temporal modeling
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=4, dim_feedforward=1024,
                    dropout=0.1, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

                # Output projection
                self.output = nn.Linear(hidden_dim, mel_dim)

            def forward(self, content, pitch, speaker):
                """
                Args:
                    content: [B, T, content_dim]
                    pitch: [B, T, 1]
                    speaker: [B, speaker_dim]
                Returns:
                    mel: [B, T, mel_dim]
                """
                # Project all inputs to hidden dim
                c = self.content_proj(content)
                p = self.pitch_proj(pitch)
                s = self.speaker_proj(speaker).unsqueeze(1).expand(-1, content.size(1), -1)

                # Combine
                x = c + p + s

                # Temporal modeling
                x = self.transformer(x)

                # Output
                mel = self.output(x)
                return mel

        self._decoder = SimpleDecoder(
            content_dim=self.config.content_dim,
            speaker_dim=self.config.speaker_dim,
            mel_dim=self.config.mel_channels
        )
        self._decoder = self._decoder.to(self.device)
        if self.config.fp16:
            self._decoder = self._decoder.half()

        return self._decoder

    def extract_content(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract content features from audio."""
        contentvec = self._load_contentvec()

        # Resample to 16kHz
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        with torch.no_grad():
            dtype = torch.float16 if self.config.fp16 else torch.float32
            with torch.amp.autocast('cuda', dtype=dtype):
                if hasattr(self, '_contentvec_extractor'):
                    # HuBERT fallback path
                    inputs = self._contentvec_extractor(
                        audio.squeeze(0).cpu().numpy(),
                        return_tensors="pt",
                        sampling_rate=16000
                    ).to(self.device)
                    outputs = contentvec(inputs.input_values.to(dtype))
                    features = outputs.last_hidden_state
                else:
                    features = contentvec.encode(audio.float())

        return features.float()

    def extract_pitch(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract F0 pitch contour."""
        rmvpe = self._load_rmvpe()

        # Resample to 16kHz
        if sr != 16000:
            audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        else:
            audio_16k = audio

        if audio_16k.dim() > 1:
            audio_16k = audio_16k.squeeze()

        with torch.no_grad():
            if hasattr(rmvpe, 'infer_from_audio'):
                # Seed-VC RMVPE
                audio_np = audio_16k.cpu().numpy() if audio_16k.is_cuda else audio_16k.numpy()
                f0 = rmvpe.infer_from_audio(audio_np, thred=0.03)
                f0 = torch.from_numpy(f0).to(self.device).unsqueeze(0)
            else:
                # AutoVoice RMVPE - ensure input dtype matches model
                audio_input = audio_16k.unsqueeze(0)
                if self.config.fp16:
                    audio_input = audio_input.half()
                else:
                    audio_input = audio_input.float()
                f0 = rmvpe.extract(audio_input)

        return f0.float()

    def convert_chunk(
        self,
        audio_chunk: torch.Tensor,
        sr: int,
        speaker_embedding: torch.Tensor,
        pitch_shift: float = 0.0
    ) -> torch.Tensor:
        """Convert a single chunk of audio (low-latency path).

        Args:
            audio_chunk: Audio tensor [samples]
            sr: Sample rate
            speaker_embedding: Target speaker [256]
            pitch_shift: Semitone shift

        Returns:
            Converted audio chunk
        """
        decoder = self._build_simple_decoder()
        vocoder = self._load_vocoder()

        # Move to device
        if not audio_chunk.is_cuda:
            audio_chunk = audio_chunk.to(self.device)
        if not speaker_embedding.is_cuda:
            speaker_embedding = speaker_embedding.to(self.device)

        # Extract features
        content = self.extract_content(audio_chunk, sr)  # [1, T_c, 768]
        pitch = self.extract_pitch(audio_chunk, sr)  # [1, T_p]

        # Apply pitch shift
        if abs(pitch_shift) > 0.1:
            shift_ratio = 2 ** (pitch_shift / 12)
            pitch = pitch * shift_ratio

        # Align lengths
        n_frames = min(content.size(1), pitch.size(1))
        content = content[:, :n_frames, :]
        pitch = pitch[:, :n_frames].unsqueeze(-1)  # [1, T, 1]

        # Decode to mel
        with torch.no_grad():
            dtype = torch.float16 if self.config.fp16 else torch.float32
            with torch.amp.autocast('cuda', dtype=dtype):
                mel = decoder(content, pitch, speaker_embedding.unsqueeze(0))  # [1, T, 80]
                mel = mel.transpose(1, 2)  # [1, 80, T]

                # Synthesize waveform
                if hasattr(vocoder, 'forward'):
                    audio_out = vocoder(mel.float())
                else:
                    audio_out = vocoder.synthesize(mel.float())

        return audio_out.squeeze().float()

    def convert_streaming(
        self,
        audio: np.ndarray,
        sr: int,
        speaker_embedding: np.ndarray,
        pitch_shift: float = 0.0,
        callback=None
    ) -> np.ndarray:
        """Convert audio with streaming (chunk-by-chunk) for low latency.

        Args:
            audio: Source audio numpy array
            sr: Sample rate
            speaker_embedding: Target speaker [256]
            pitch_shift: Semitone shift
            callback: Optional callback(chunk) for streaming output

        Returns:
            Converted audio
        """
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        speaker_tensor = torch.from_numpy(speaker_embedding).float().to(self.device)

        # Resample to target SR
        if sr != self.config.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.config.sample_rate
            )
            sr = self.config.sample_rate

        total_samples = audio_tensor.size(0)
        output_chunks = []

        # Process in chunks with overlap
        pos = 0
        chunk_idx = 0

        while pos < total_samples:
            chunk_end = min(pos + self.chunk_samples, total_samples)
            chunk = audio_tensor[pos:chunk_end]

            # Pad last chunk if needed
            if chunk.size(0) < self.chunk_samples:
                chunk = F.pad(chunk, (0, self.chunk_samples - chunk.size(0)))

            # Convert chunk
            start_time = time.time()
            converted = self.convert_chunk(chunk, sr, speaker_tensor, pitch_shift)
            latency = (time.time() - start_time) * 1000

            if chunk_idx % 10 == 0:
                logger.debug(f"Chunk {chunk_idx}: latency {latency:.1f}ms")

            # Crossfade with previous chunk
            if self._prev_chunk is not None and self.overlap_samples > 0:
                fade_out = torch.linspace(1, 0, self.overlap_samples, device=self.device)
                fade_in = torch.linspace(0, 1, self.overlap_samples, device=self.device)

                # Blend overlap region
                converted[:self.overlap_samples] = (
                    self._prev_chunk[-self.overlap_samples:] * fade_out +
                    converted[:self.overlap_samples] * fade_in
                )

            # Store for next iteration
            self._prev_chunk = converted.clone()

            # Output (excluding overlap that will be blended with next)
            if pos + self.chunk_samples < total_samples:
                output = converted[:-self.overlap_samples]
            else:
                output = converted[:chunk_end - pos]  # Last chunk

            output_chunks.append(output.cpu().numpy())

            if callback:
                callback(output.cpu().numpy())

            pos += self.chunk_samples - self.overlap_samples
            chunk_idx += 1

        return np.concatenate(output_chunks)

    def convert_full(
        self,
        audio: np.ndarray,
        sr: int,
        speaker_embedding: np.ndarray,
        pitch_shift: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """Convert full audio file (non-streaming).

        Args:
            audio: Source audio
            sr: Sample rate
            speaker_embedding: Target speaker embedding
            pitch_shift: Pitch shift in semitones

        Returns:
            (converted_audio, output_sample_rate)
        """
        logger.info(f"Converting {len(audio)/sr:.1f}s audio (non-streaming)...")
        start_time = time.time()

        converted = self.convert_streaming(audio, sr, speaker_embedding, pitch_shift)

        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio) / sr)
        logger.info(f"Conversion complete: {elapsed:.2f}s (RTF: {rtf:.3f})")

        return converted, self.config.sample_rate

    def unload(self):
        """Unload models to free GPU memory."""
        self._contentvec = None
        self._rmvpe = None
        self._vocoder = None
        self._decoder = None
        self._prev_chunk = None
        torch.cuda.empty_cache()
        logger.info("Models unloaded")


def load_speaker_embedding(profile_id: str) -> np.ndarray:
    """Load speaker embedding from profile."""
    embedding_path = f"data/voice_profiles/{profile_id}.npy"
    if not Path(embedding_path).exists():
        raise FileNotFoundError(f"Speaker embedding not found: {embedding_path}")
    return np.load(embedding_path)


def main():
    """Test the realtime pipeline."""
    print("\n" + "=" * 60)
    print("  REALTIME VOICE CONVERSION PIPELINE TEST")
    print("=" * 60 + "\n")

    os.chdir(Path(__file__).parent.parent)

    WILLIAM_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
    CONOR_ID = "9679a6ec-e6e2-43c4-b64e-1f004fed34f9"

    # Initialize converter
    config = RealtimeConfig(
        sample_rate=22050,
        chunk_size_ms=100,
        overlap_ms=20,
        fp16=True
    )
    converter = RealtimeVoiceConverter(config)

    # Load test audio
    vocals_path = f"data/separated/{WILLIAM_ID}/vocals.wav"
    if not Path(vocals_path).exists():
        print(f"Test vocals not found: {vocals_path}")
        return

    audio, sr = librosa.load(vocals_path, sr=None, mono=True)
    print(f"Source: {len(audio)/sr:.1f}s @ {sr}Hz")

    # Load target embedding
    try:
        target_embedding = load_speaker_embedding(CONOR_ID)
        print(f"Target: Conor embedding {target_embedding.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Creating dummy embedding for testing...")
        target_embedding = np.random.randn(256).astype(np.float32)

    # Convert
    print("\nConverting (realtime mode)...")
    converted, out_sr = converter.convert_full(
        audio, sr, target_embedding, pitch_shift=0.5
    )

    # Save output
    output_dir = Path("data/conversions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "william_as_conor_REALTIME.wav"
    sf.write(str(output_path), converted, out_sr)
    print(f"\nSaved: {output_path}")

    # Cleanup
    converter.unload()
    print("\nRealtime pipeline test complete!")


if __name__ == "__main__":
    main()
