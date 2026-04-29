"""SOTA Singing Voice Conversion Pipeline.

End-to-end pipeline connecting all SOTA components:
  MelBandRoFormer → ContentVec → RMVPE → CoMoSVC → BigVGAN

Sample rate flow:
  Input (any SR) → 44.1kHz (separator) → 16kHz (content+pitch) → mel → 24kHz (vocoder)

Frame alignment:
  ContentVec: 50fps at 16kHz (hop=320)
  RMVPE: 50fps at 16kHz (hop=320)
  Both produce aligned frame sequences for the decoder.

No fallback behavior: raises RuntimeError on failure.
"""
import logging
import time
from typing import Callable, Dict, Optional, Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..audio.separator import MelBandRoFormer
from ..models.encoder import ContentVecEncoder
from ..models.pitch import RMVPEPitchExtractor
from ..models.svc_decoder import CoMoSVCDecoder
from ..models.vocoder import BigVGANVocoder
from ..models.adapter_manager import (
    AdapterManager,
    AdapterManagerConfig,
    SUPPORTED_SPEAKER_EMBEDDING_DIMS,
)
from ..models.hq_adapter_bridge import HQLoRAAdapterBridge, AdapterBridgeConfig

if TYPE_CHECKING:
    from ..storage.voice_profiles import VoiceProfileStore

logger = logging.getLogger(__name__)

# Minimum input duration (100ms)
MIN_DURATION_SAMPLES_24K = 2400  # 100ms at 24kHz


class SOTAConversionPipeline:
    """SOTA singing voice conversion pipeline.

    Orchestrates:
      1. Vocal separation (MelBandRoFormer @ 44.1kHz)
      2. Content extraction (ContentVec @ 16kHz → 768-dim)
      3. Pitch extraction (RMVPE @ 16kHz → F0 + voicing)
      4. SVC decoding (CoMoSVC → mel spectrogram)
      5. Waveform synthesis (BigVGAN @ 24kHz)

    All components run sequentially to minimize GPU memory usage.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        n_steps: int = 1,
        profile_store: Optional["VoiceProfileStore"] = None,
        profile_id: Optional[str] = None,
        require_gpu: bool = True,
    ):
        """Initialize pipeline with all SOTA components.

        Args:
            device: Target device (defaults to CUDA if available)
            n_steps: Consistency model steps (1=fast, 4=quality)
            profile_store: Voice profile storage for loading trained weights
            profile_id: Profile ID to load LoRA weights from (requires profile_store)
            require_gpu: If True, raise RuntimeError if CUDA unavailable

        Raises:
            RuntimeError: If require_gpu is True and CUDA is not available
        """
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SOTAConversionPipeline")

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.n_steps = n_steps
        self._current_stage = None
        self._profile_store = profile_store
        self._profile_id = profile_id
        self._current_speaker_id: Optional[str] = None
        self._speaker_embedding: Optional[torch.Tensor] = None

        # Initialize adapter manager for dynamic speaker switching
        self._adapter_manager = AdapterManager(
            AdapterManagerConfig(device=str(self.device))
        )

        # Initialize HQ adapter bridge for trained HQ LoRA adapters
        # HQ adapters use a different architecture (standalone MLP vs layer-injection)
        self._hq_adapter_bridge = HQLoRAAdapterBridge(
            AdapterBridgeConfig(device=str(self.device))
        )
        self._use_hq_adapter = False  # Flag to indicate if HQ adapter is active

        # Initialize components (all moved to device)
        self.separator = MelBandRoFormer(device=self.device).to(self.device)
        self.content_extractor = ContentVecEncoder(
            output_dim=768, layer=12, device=self.device
        ).to(self.device)
        self.pitch_extractor = RMVPEPitchExtractor(
            device=self.device
        ).to(self.device)
        self.decoder = CoMoSVCDecoder(device=self.device).to(self.device)
        self.vocoder = BigVGANVocoder(
            pretrained=None, device=self.device
        ).to(self.device)

        # Load LoRA weights if profile has trained model
        if profile_store is not None and profile_id is not None:
            self._load_profile_lora(profile_store, profile_id)

        logger.info(f"SOTAConversionPipeline initialized on {self.device}")

    def _load_profile_lora(
        self, profile_store: "VoiceProfileStore", profile_id: str
    ) -> None:
        """Load LoRA weights from profile if available.

        Args:
            profile_store: Voice profile storage
            profile_id: Profile ID to load weights from

        Raises:
            RuntimeError: If loading or applying LoRA weights fails
        """
        if not profile_store.has_trained_model(profile_id):
            logger.info(f"Profile {profile_id} has no trained model, skipping LoRA")
            return

        try:
            # Load weights
            lora_state = profile_store.load_lora_weights(profile_id)

            # Inject LoRA into decoder
            self.decoder.inject_lora(rank=8, alpha=16)

            # Load weights into decoder
            self.decoder.load_lora_state_dict(lora_state)

            self._current_speaker_id = profile_id
            logger.info(f"Loaded LoRA weights from profile {profile_id}")
        except Exception as e:
            logger.error(f"Failed to load LoRA weights for {profile_id}: {e}")
            raise RuntimeError(f"Failed to load LoRA weights: {e}") from e

    def set_speaker(self, profile_id: str) -> None:
        """Dynamically switch to a different speaker by loading their LoRA adapter.

        This method allows changing the target voice after pipeline initialization
        without recreating the entire pipeline. Loads both the LoRA adapter weights
        and the stored speaker embedding from the profile.

        Prefers HQ adapters (standalone MLP) over layer-injection adapters when available.
        HQ adapters transform content features before decoding, while standard adapters
        modify decoder layers directly.

        The speaker embedding is validated for correct shape and L2 normalization.
        If the embedding is not normalized, it will be normalized automatically with
        a warning.

        Args:
            profile_id: UUID of the voice profile to switch to

        Raises:
            FileNotFoundError: If no adapter or embedding exists for the profile
            ValueError: If embedding has an unsupported shape
            RuntimeError: If adapter loading or application fails
        """
        if profile_id == self._current_speaker_id:
            logger.debug(f"Speaker {profile_id} already loaded, skipping")
            return

        # Check for HQ adapter first (preferred)
        has_hq_adapter = self._hq_adapter_bridge.has_adapter(profile_id)
        has_standard_adapter = self._adapter_manager.has_adapter(profile_id)

        if not has_hq_adapter and not has_standard_adapter:
            raise FileNotFoundError(
                f"No trained adapter found for profile: {profile_id}"
            )

        try:
            # Load speaker embedding first (needed for both adapter types)
            self._speaker_embedding = self._adapter_manager.load_speaker_embedding(
                profile_id,
                as_tensor=True,
            )

            # Prefer HQ adapter if available
            if has_hq_adapter:
                self._adapter_manager.release_active_artifact()
                # Load HQ adapter (standalone MLP architecture)
                self._hq_adapter_bridge.load_adapter(profile_id)
                self._use_hq_adapter = True
                logger.info(f"Loaded HQ adapter for speaker: {profile_id}")
            else:
                # Fall back to standard layer-injection adapter
                artifact = self._adapter_manager.swap_artifact(
                    profile_id,
                    artifact_type="adapter",
                )

                # Ensure decoder has LoRA layers injected
                if not hasattr(self.decoder, 'lora_adapters') or not self.decoder.lora_adapters:
                    self.decoder.inject_lora(rank=8, alpha=16)

                # Apply adapter weights to decoder
                self._adapter_manager.apply_adapter(self.decoder, artifact.handle)
                self._use_hq_adapter = False
                logger.info(f"Loaded standard adapter for speaker: {profile_id}")

            self._current_speaker_id = profile_id
            logger.info(f"Switched to speaker: {profile_id} (embedding loaded)")

        except Exception as e:
            logger.error(f"Failed to set speaker {profile_id}: {e}")
            raise RuntimeError(f"Failed to switch speaker: {e}") from e

    def get_current_speaker(self) -> Optional[str]:
        """Get the currently loaded speaker profile ID.

        Useful for tracking which voice is currently active in the pipeline.
        Returns None if no speaker has been set via set_speaker() or if
        clear_speaker() was called.

        Returns:
            Profile ID of the current speaker, or None if no speaker loaded
        """
        return self._current_speaker_id

    def get_speaker_embedding(self) -> Optional[torch.Tensor]:
        """Get the currently loaded speaker embedding.

        Returns the stored profile speaker embedding as loaded from disk. Shipped
        profiles currently use mixed-width embeddings, so callers should not assume
        a fixed dimension.

        Returns:
            L2-normalized speaker embedding tensor on device, or None if no speaker loaded
        """
        return self._speaker_embedding

    def _prepare_decoder_speaker_embedding(
        self,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt a stored speaker embedding to the decoder's expected width."""
        if speaker_embedding.dim() != 1:
            raise RuntimeError(
                f"Speaker embedding must be 1D, got {speaker_embedding.shape}"
            )

        source_dim = speaker_embedding.shape[0]
        if source_dim not in SUPPORTED_SPEAKER_EMBEDDING_DIMS:
            raise RuntimeError(
                "Speaker embedding must have a supported width "
                f"{SUPPORTED_SPEAKER_EMBEDDING_DIMS}, got {speaker_embedding.shape}"
            )

        target_dim = int(getattr(self.decoder, "speaker_dim", source_dim))
        speaker = speaker_embedding.to(self.device)

        if source_dim == target_dim:
            return speaker.unsqueeze(0)

        if source_dim < target_dim:
            logger.warning(
                "Padding speaker embedding from %s to %s for decoder conditioning",
                source_dim,
                target_dim,
            )
            padding = torch.zeros(
                target_dim - source_dim,
                dtype=speaker.dtype,
                device=speaker.device,
            )
            return torch.cat([speaker, padding], dim=0).unsqueeze(0)

        logger.warning(
            "Truncating speaker embedding from %s to %s for decoder conditioning",
            source_dim,
            target_dim,
        )
        return speaker[:target_dim].unsqueeze(0)

    def clear_speaker(self) -> None:
        """Clear the current speaker adapter, reverting to base model.

        This zeros out LoRA contributions without removing the LoRA layers
        and clears the stored speaker embedding. Handles both HQ and standard
        adapter architectures.

        Returns:
            None
        """
        if self._current_speaker_id is None:
            logger.debug("No speaker loaded, nothing to clear")
            return

        # Clear HQ adapter if it was used
        if self._use_hq_adapter:
            self._hq_adapter_bridge.clear()
            self._use_hq_adapter = False
        else:
            self._adapter_manager.remove_adapter(self.decoder)
            self._adapter_manager.release_active_artifact()

        self._speaker_embedding = None
        self._current_speaker_id = None
        logger.info("Speaker cleared, reverted to base model")

    def _resample(self, audio: torch.Tensor, from_sr: int,
                  to_sr: int) -> torch.Tensor:
        """Resample audio tensor between sample rates.

        Uses linear interpolation to change the sample rate. If from_sr equals
        to_sr, returns the input unchanged. Handles both mono [T] and stereo
        [C, T] audio tensors.

        Args:
            audio: [T] or [C, T] audio tensor
            from_sr: Source sample rate in Hz
            to_sr: Target sample rate in Hz

        Returns:
            Resampled audio tensor with same channel configuration as input
        """
        if from_sr == to_sr:
            return audio

        # Use linear interpolation for resampling
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            target_len = int(audio.shape[2] * to_sr / from_sr)
            resampled = F.interpolate(
                audio, size=target_len, mode='linear', align_corners=False
            )
            return resampled.squeeze(0).squeeze(0)
        else:
            # [C, T] → [1, C, T]
            audio = audio.unsqueeze(0)
            target_len = int(audio.shape[2] * to_sr / from_sr)
            resampled = F.interpolate(
                audio, size=target_len, mode='linear', align_corners=False
            )
            return resampled.squeeze(0)

    def _to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono by averaging channels.

        Args:
            audio: [T] or [C, T] audio tensor

        Returns:
            [T] mono audio tensor

        Raises:
            RuntimeError: If audio has unexpected shape (not 1D or 2D)
        """
        if audio.dim() == 1:
            return audio
        if audio.dim() == 2:
            return audio.mean(dim=0)
        raise RuntimeError(
            f"Unexpected audio shape: {audio.shape}. Expected [T] or [C, T]."
        )

    def _encode_pitch(self, f0: torch.Tensor) -> torch.Tensor:
        """Encode F0 values to the decoder pitch embedding width.

        Uses Fourier features to create a rich representation of pitch:
        1. Convert F0 to log2 scale and normalize to [0, 1]
        2. Generate sin/cos harmonics up to the decoder pitch dimension
        3. Return phase-shifted sinusoids that encode pitch continuously

        This approach provides a continuous pitch representation that's more
        expressive than discrete bins, allowing the decoder to interpolate
        smoothly between pitch values.

        Args:
            f0: [B, T] F0 in Hz (unvoiced frames should be 0 or low values)

        Returns:
            [B, T, decoder.pitch_dim] pitch embeddings using Fourier features
        """
        B, T = f0.shape

        # Log-scale F0 (with floor for unvoiced)
        log_f0 = torch.log2(f0.clamp(min=1.0))  # [B, T]
        # Normalize to [0, 1] range (50Hz → 1100Hz = ~5.6 → ~10.1 in log2)
        log_f0_norm = (log_f0 - 5.6) / (10.1 - 5.6)
        log_f0_norm = log_f0_norm.clamp(0, 1)

        # Create pitch features: sin/cos harmonics sized to the decoder contract.
        pitch_dim = int(getattr(self.decoder, "pitch_dim", 768))
        half_dim = max(1, pitch_dim // 2)
        freqs = torch.arange(1, half_dim + 1, device=f0.device).float()
        phase = log_f0_norm.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0) * torch.pi
        pitch_embed = torch.cat([
            torch.sin(phase),
            torch.cos(phase),
        ], dim=-1)

        return pitch_embed[:, :, :pitch_dim]

    def convert(self, audio: torch.Tensor, sample_rate: int,
                speaker_embedding: torch.Tensor,
                on_progress: Optional[Callable[[str, float], None]] = None
                ) -> Dict[str, Any]:
        """Convert audio to target speaker voice.

        Executes the complete SOTA voice conversion pipeline:
        1. Vocal separation using MelBandRoFormer at 44.1kHz
        2. Content feature extraction using ContentVec at 16kHz
        3. Pitch extraction using RMVPE at 16kHz
        4. Frame alignment and pitch encoding to the decoder pitch contract
        5. Mel spectrogram generation using CoMoSVC decoder
        6. Waveform synthesis using BigVGAN vocoder at 24kHz

        The output is normalized to peak amplitude of 0.95 to prevent clipping
        while maintaining reasonable loudness.

        Args:
            audio: [T] or [C, T] input audio tensor (any sample rate)
            sample_rate: Input sample rate in Hz
            speaker_embedding: 1D target speaker embedding (192-d or 256-d today)
            on_progress: Optional callback(stage_name, progress_fraction) for tracking

        Returns:
            Dict with:
                - audio: [T] output waveform at 24kHz, normalized to peak 0.95
                - sample_rate: 24000 (output sample rate)
                - metadata: Dict containing processing_time, n_steps, n_frames, output_duration, device

        Raises:
            RuntimeError: If audio is empty, too short (<100ms), has wrong dimensions,
                         or if any component fails during processing
        """
        start_time = time.time()

        # Validate inputs
        if audio.numel() == 0:
            raise RuntimeError("Empty audio input")

        # Convert to mono
        audio_mono = self._to_mono(audio)

        # Check minimum duration (normalize to 24kHz equivalent)
        duration_samples_24k = int(len(audio_mono) * 24000 / sample_rate)
        if duration_samples_24k < MIN_DURATION_SAMPLES_24K:
            raise RuntimeError(
                f"Audio too short: {duration_samples_24k} samples at 24kHz "
                f"(minimum {MIN_DURATION_SAMPLES_24K})"
            )

        # Validate speaker embedding
        speaker = self._prepare_decoder_speaker_embedding(speaker_embedding)

        def report(stage: str, progress: float):
            self._current_stage = stage
            if on_progress:
                on_progress(stage, progress)

        # Stage 1: Vocal separation (44.1kHz)
        report('separation', 0.0)
        audio_44k = self._resample(audio_mono, sample_rate, 44100)
        audio_44k = audio_44k.to(self.device)

        with torch.no_grad():
            vocals_44k = self.separator.extract_vocals(
                audio_44k.unsqueeze(0)  # [1, T]
            )  # [1, T]
            vocals_44k = vocals_44k.squeeze(0)  # [T]
        report('separation', 0.2)

        # Stage 2: Content extraction (16kHz)
        report('content_extraction', 0.2)
        vocals_16k = self._resample(vocals_44k, 44100, 16000)

        with torch.no_grad():
            content_features = self.content_extractor.encode(
                vocals_16k.unsqueeze(0)  # [1, T]
            )  # [1, N_frames, 768]
        report('content_extraction', 0.4)

        # Stage 3: Pitch extraction (16kHz)
        report('pitch_extraction', 0.4)
        with torch.no_grad():
            f0 = self.pitch_extractor.extract(
                vocals_16k.unsqueeze(0)  # [1, T]
            )  # [1, N_frames]
        report('pitch_extraction', 0.6)

        # Frame alignment: ensure content and pitch have same length
        n_frames = min(content_features.shape[1], f0.shape[1])
        content_features = content_features[:, :n_frames, :]  # [1, N, 768]
        f0_aligned = f0[:, :n_frames]  # [1, N]

        # Apply HQ adapter transformation if active
        # HQ adapters transform content features before the decoder
        if self._use_hq_adapter and self._hq_adapter_bridge.get_current_adapter() is not None:
            content_features = self._hq_adapter_bridge.transform(
                content_features,
                speaker_embedding  # Pass speaker for conditioning
            )
            logger.debug("Applied HQ adapter transformation to content features")

        # Encode pitch to the decoder pitch contract.
        pitch_embeddings = self._encode_pitch(f0_aligned)

        # Stage 4: SVC decoding (mel generation)
        report('decoding', 0.6)
        with torch.no_grad():
            mel = self.decoder.infer(
                content_features, pitch_embeddings, speaker,
                n_steps=self.n_steps
            )  # [1, n_mels, N]
        report('decoding', 0.8)

        # Upsample mel to match vocoder frame rate
        # ContentVec: 50fps (16kHz, hop=320) → BigVGAN: 93.75fps (24kHz, hop=256)
        # Need ~1.875x temporal upsampling of mel frames
        vocoder_hop = getattr(self.vocoder, 'hop_size', 256)
        vocoder_sr = getattr(self.vocoder, 'sample_rate', 24000)
        target_fps = vocoder_sr / vocoder_hop  # 93.75 for BigVGAN 24kHz
        content_fps = 50.0  # 16kHz / hop=320
        upsample_factor = target_fps / content_fps  # ~1.875

        if upsample_factor > 1.01:  # Only upsample if needed
            target_mel_frames = int(mel.shape[2] * upsample_factor)
            mel = F.interpolate(
                mel, size=target_mel_frames, mode='linear', align_corners=False
            )

        # Stage 5: Vocoder (mel → 24kHz waveform)
        report('vocoder', 0.8)
        with torch.no_grad():
            waveform = self.vocoder.synthesize(mel)  # [1, T_audio]
        waveform = waveform.squeeze(0)  # [T_audio]

        # Normalize output to reasonable level (-0.95 to 0.95 peak)
        # Untrained models may produce very quiet output
        peak = waveform.abs().max()
        if peak > 1e-6:
            target_peak = 0.95
            waveform = waveform * (target_peak / peak)

        report('vocoder', 1.0)

        elapsed = time.time() - start_time
        output_duration = len(waveform) / 24000

        logger.info(
            f"SOTA conversion complete: {output_duration:.2f}s audio "
            f"in {elapsed:.2f}s ({elapsed/output_duration:.1f}x realtime)"
        )

        return {
            'audio': waveform,
            'sample_rate': 24000,
            'metadata': {
                'processing_time': elapsed,
                'n_steps': self.n_steps,
                'n_frames': n_frames,
                'output_duration': output_duration,
                'device': str(self.device),
            },
        }
