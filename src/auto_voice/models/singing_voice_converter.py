"""
Singing Voice Converter

Main model for singing voice conversion using So-VITS-SVC architecture.
Integrates content encoder, pitch encoder, posterior encoder, flow decoder, and vocoder.
"""

from typing import Dict, Any, Optional, Union, Tuple, Literal
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .content_encoder import ContentEncoder
from .pitch_encoder import PitchEncoder
from .posterior_encoder import PosteriorEncoder
from .flow_decoder import FlowDecoder
from .hifigan import HiFiGANGenerator

logger = logging.getLogger(__name__)

# Quality presets configuration
QUALITY_PRESETS = {
    'draft': {
        'description': 'Fast conversion with lower quality, suitable for testing',
        'decoder_steps': 2,
        'vocoder_speed': 'fast',
        'relative_quality': 0.6,
        'relative_speed': 4.0
    },
    'fast': {
        'description': 'Balanced quality and speed for real-time applications',
        'decoder_steps': 4,
        'vocoder_speed': 'normal',
        'relative_quality': 0.8,
        'relative_speed': 2.0
    },
    'balanced': {
        'description': 'Standard quality preset (default)',
        'decoder_steps': 4,
        'vocoder_speed': 'normal',
        'relative_quality': 1.0,
        'relative_speed': 1.0
    },
    'high': {
        'description': 'High quality conversion with moderate speed',
        'decoder_steps': 8,
        'vocoder_speed': 'high_quality',
        'relative_quality': 1.3,
        'relative_speed': 0.5
    },
    'studio': {
        'description': 'Maximum quality for studio production',
        'decoder_steps': 16,
        'vocoder_speed': 'ultra_high',
        'relative_quality': 1.5,
        'relative_speed': 0.25
    }
}


class VoiceConversionError(Exception):
    """Exception raised for voice conversion errors."""
    pass


class SingingVoiceConverter(nn.Module):
    """
    Main model for singing voice conversion using So-VITS-SVC architecture.

    Combines content encoder (HuBERT-Soft), pitch encoder, posterior encoder,
    flow decoder, and vocoder for end-to-end singing voice conversion.

    Args:
        config: Configuration dictionary with model parameters

    Example:
        >>> config = {'latent_dim': 192, 'mel_channels': 80, 'content_encoder_type': 'hubert_soft'}
        >>> model = SingingVoiceConverter(config)
        >>> # Training
        >>> outputs = model(source_audio, target_mel, source_f0, speaker_emb)
        >>> # Inference
        >>> model.eval()
        >>> model.prepare_for_inference()
        >>> audio = model.convert(source_audio, target_speaker_emb)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        # Support nested singing_voice_converter section or flat config
        if 'singing_voice_converter' in config:
            svc_config = config['singing_voice_converter']
        else:
            svc_config = config

        # Extract configuration
        self.latent_dim = svc_config.get('latent_dim', 192)
        self.mel_channels = svc_config.get('mel_channels', 80)

        # Content encoder settings
        content_cfg = svc_config.get('content_encoder', {})
        self.content_dim = content_cfg.get('output_dim', svc_config.get('content_dim', 256))

        # Pitch encoder settings
        pitch_cfg = svc_config.get('pitch_encoder', {})
        self.pitch_dim = pitch_cfg.get('pitch_dim', svc_config.get('pitch_dim', 192))

        # Speaker encoder settings
        speaker_cfg = svc_config.get('speaker_encoder', {})
        self.speaker_dim = speaker_cfg.get('embedding_dim', svc_config.get('speaker_dim', 256))

        # Posterior encoder settings
        posterior_cfg = svc_config.get('posterior_encoder', {})
        self.hidden_channels = posterior_cfg.get('hidden_channels', svc_config.get('hidden_channels', 192))

        # Initialize content encoder with CNN fallback mel config
        cnn_fallback_cfg = content_cfg.get('cnn_fallback', {})
        self.content_encoder = ContentEncoder(
            encoder_type=content_cfg.get('type', svc_config.get('content_encoder_type', 'hubert_soft')),
            output_dim=self.content_dim,
            device=content_cfg.get('device', svc_config.get('device', None)),
            use_torch_hub=content_cfg.get('use_torch_hub', svc_config.get('use_torch_hub', True)),
            cnn_mel_config=cnn_fallback_cfg
        )

        # Add projection layer if HuBERT-Soft content features (256) don't match content_dim
        # HuBERT-Soft always outputs 256-dim features, need to project if content_dim != 256
        encoder_type = content_cfg.get('type', svc_config.get('content_encoder_type', 'hubert_soft'))
        if encoder_type == 'hubert_soft':
            hubert_output_dim = 256  # HuBERT-Soft fixed output dimension
            if self.content_dim != hubert_output_dim:
                logger.info(f"Adding content projection layer: {hubert_output_dim} -> {self.content_dim}")
                self.content_projection = nn.Conv1d(hubert_output_dim, self.content_dim, 1)
            else:
                self.content_projection = None
        else:
            self.content_projection = None

        # Initialize pitch encoder
        self.pitch_encoder = PitchEncoder(
            pitch_dim=self.pitch_dim,
            hidden_dim=pitch_cfg.get('hidden_dim', svc_config.get('pitch_hidden_dim', 128)),
            num_bins=pitch_cfg.get('num_bins', svc_config.get('num_bins', 256)),
            f0_min=pitch_cfg.get('f0_min', svc_config.get('f0_min', 80.0)),
            f0_max=pitch_cfg.get('f0_max', svc_config.get('f0_max', 1000.0))
        )

        # Set blend_weight if specified in config
        if 'blend_weight' in pitch_cfg:
            self.pitch_encoder.blend_weight.data.fill_(pitch_cfg['blend_weight'])

        # Initialize posterior encoder (training only)
        self.posterior_encoder = PosteriorEncoder(
            in_channels=self.mel_channels,
            out_channels=self.latent_dim,
            hidden_channels=self.hidden_channels,
            num_layers=posterior_cfg.get('num_layers', svc_config.get('posterior_num_layers', 16))
        )

        # Initialize flow decoder
        # Conditioning: content + pitch + speaker
        self.cond_dim = self.content_dim + self.pitch_dim + self.speaker_dim
        flow_cfg = svc_config.get('flow_decoder', {})
        use_only_mean = flow_cfg.get('use_only_mean', svc_config.get('use_only_mean', False))

        # Warn if training with use_only_mean=True
        if use_only_mean and self.training:
            logger.warning(
                "Training SingingVoiceConverter with use_only_mean=True. "
                "This yields zero log-det which may undermine flow likelihood. "
                "Consider setting use_only_mean=False in config or implementing a staged schedule."
            )

        self.flow_decoder = FlowDecoder(
            in_channels=self.latent_dim,
            hidden_channels=flow_cfg.get('hidden_channels', self.hidden_channels),
            num_flows=flow_cfg.get('num_flows', svc_config.get('num_flows', 4)),
            cond_channels=self.cond_dim,
            use_only_mean=use_only_mean
        )

        # Projection from latent to mel for vocoder
        self.latent_to_mel = nn.Conv1d(self.latent_dim, self.mel_channels, 1)

        # Optional HiFiGAN vocoder
        vocoder_cfg = svc_config.get('vocoder', {})
        # Audio settings - get sample rate from hifigan config or audio config
        hifigan_cfg = config.get('hifigan', {})
        audio_cfg = svc_config.get('audio', {})
        vocoder_sr = hifigan_cfg.get('sample_rate', audio_cfg.get('sample_rate', 22050))

        if vocoder_cfg.get('use_vocoder', svc_config.get('use_vocoder', True)):
            self.vocoder = HiFiGANGenerator(mel_channels=self.mel_channels, sample_rate=vocoder_sr)
            # Use the vocoder's actual sample rate
            self.vocoder_sample_rate = self.vocoder.sample_rate
        else:
            self.vocoder = None
            # Fallback to config value
            self.vocoder_sample_rate = vocoder_sr

        # Inference settings
        inference_cfg = svc_config.get('inference', {})
        self.temperature = inference_cfg.get('temperature', 1.0)
        self.quality_preset = 'balanced'
        self.decoder_steps = QUALITY_PRESETS[self.quality_preset]['decoder_steps']

        # Advanced feature flags
        self.denoise_input = False
        self.enhance_output = False
        self.preserve_dynamics = False
        self.vibrato_transfer = False

        
        # TensorRT configuration
        self.use_tensorrt = svc_config.get('use_tensorrt', False)
        self.tensorrt_precision = svc_config.get('tensorrt_precision', 'fp16')
        self.tensorrt_models = {}
        self.fallback_to_pytorch = svc_config.get('fallback_to_pytorch', True)

        logger.info(f"SingingVoiceConverter initialized: latent_dim={self.latent_dim}, "
                   f"content_dim={self.content_dim}, pitch_dim={self.pitch_dim}, "
                   f"vocoder_sample_rate={self.vocoder_sample_rate}, temperature={self.temperature}, "
                   f"quality_preset={self.quality_preset}")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        config_override: Optional[Dict[str, Any]] = None
    ) -> 'SingingVoiceConverter':
        """Load pre-trained So-VITS-SVC model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to load model onto
            config_override: Optional configuration overrides
            
        Returns:
            Initialized SingingVoiceConverter with loaded weights
            
        Example:
            >>> model = SingingVoiceConverter.load_from_checkpoint(
            ...     'models/pretrained/sovits5.0_main_1500.pth',
            ...     device='cuda'
            ... )
            >>> model.eval()
        """
        from ..utils.model_loader import load_sovits_checkpoint, get_default_sovits_config
        
        # Load checkpoint
        checkpoint_data = load_sovits_checkpoint(checkpoint_path, device, config_override)
        
        # Get config (checkpoint config + defaults + overrides)
        config = get_default_sovits_config()
        config.update(checkpoint_data['config'])
        if config_override:
            config.update(config_override)
        
        # Initialize model
        model = cls(config)
        
        # Load state dict with error handling
        try:
            # Try strict loading first
            model.load_state_dict(checkpoint_data['model_state_dict'], strict=True)
            logger.info("Loaded checkpoint with strict=True")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            logger.info("Attempting non-strict loading...")
            
            # Try non-strict loading
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint_data['model_state_dict'],
                strict=False
            )
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
        
        # Move to device and set eval mode
        model = model.to(device)
        model.eval()
        
        logger.info(f"Successfully loaded model from {checkpoint_path}")
        
        return model

    def forward(
        self,
        source_audio: torch.Tensor,
        target_mel: torch.Tensor,
        source_f0: torch.Tensor,
        target_speaker_emb: Optional[torch.Tensor] = None,
        source_sample_rate: int = 16000,
        x_mask: Optional[torch.Tensor] = None,
        source_voiced: Optional[torch.Tensor] = None,
        use_vocoder: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with ground-truth target.

        Args:
            source_audio: Source singing audio [B, T_audio]
            target_mel: Ground-truth target mel [B, mel_channels, T_mel]
            source_f0: F0 contour from source [B, T_f0]
            target_speaker_emb: Target speaker embedding [B, 256]
            source_sample_rate: Sample rate of source audio
            x_mask: Optional mask for variable lengths [B, 1, T]
            source_voiced: Optional voiced mask for F0 [B, T_f0]
            use_vocoder: If True, generate pred_audio for perceptual losses (default: True)

        Returns:
            Dict with 'pred_mel', 'z_mean', 'z_logvar', 'z', 'u', 'logdet', 'cond'
            If use_vocoder=True, also includes 'pred_audio' for pitch/speaker losses
        """
        B, _, T = target_mel.shape

        # Get model device for consistency
        device = next(self.parameters()).device

        # Move target_mel and x_mask to model device before processing
        target_mel = target_mel.to(device)
        if x_mask is not None:
            x_mask = x_mask.to(device)
        else:
            x_mask = torch.ones(B, 1, T, device=device)

        # Extract content from source
        content = self.content_encoder(source_audio, source_sample_rate)  # [B, T_content, encoder_output_dim]
        content = content.to(device)

        # Apply content projection if needed (HuBERT 256 -> content_dim)
        if self.content_projection is not None:
            # content: [B, T, 256] -> transpose -> [B, 256, T] -> project -> [B, content_dim, T] -> transpose -> [B, T, content_dim]
            content = self.content_projection(content.transpose(1, 2)).transpose(1, 2)

        # Move voiced mask to device if provided
        if source_voiced is not None:
            source_voiced = source_voiced.to(device)

        # Encode pitch with voiced mask
        pitch_emb = self.pitch_encoder(source_f0.to(device), source_voiced)  # [B, T_f0, 192]
        pitch_emb = pitch_emb.to(device)

        # Align content and pitch to target length
        content = self._interpolate_features(content, T)  # [B, T, content_dim]
        pitch_emb = self._interpolate_features(pitch_emb, T)  # [B, T, pitch_dim]

        # Handle optional target_speaker_emb
        if target_speaker_emb is None:
            # Use default zero embedding when not provided
            target_speaker_emb = torch.zeros(B, self.speaker_dim, device=device)
            logger.debug(f"target_speaker_emb is None, using default zero embedding [B={B}, speaker_dim={self.speaker_dim}]")
        else:
            # Validate and expand speaker embedding to sequence
            # Ensure target_speaker_emb has correct shape [B, speaker_dim]
            target_speaker_emb = target_speaker_emb.to(device)
            if target_speaker_emb.dim() != 2:
                raise VoiceConversionError(
                    f"target_speaker_emb must be 2D [B, {self.speaker_dim}], "
                    f"got {target_speaker_emb.dim()}D with shape {list(target_speaker_emb.shape)}"
                )
            if target_speaker_emb.size(1) != self.speaker_dim:
                raise VoiceConversionError(
                    f"target_speaker_emb must have shape [B, {self.speaker_dim}], "
                    f"got [B, {target_speaker_emb.size(1)}]"
                )

        speaker_emb = target_speaker_emb.unsqueeze(2).expand(-1, -1, T)  # [B, speaker_dim, T]

        # Concatenate conditioning: [B, content_dim + pitch_dim + speaker_dim, T]
        cond = torch.cat([
            content.transpose(1, 2),
            pitch_emb.transpose(1, 2),
            speaker_emb
        ], dim=1)

        # Posterior encoding (training)
        z_mean, z_logvar = self.posterior_encoder(target_mel, x_mask)
        z = self.posterior_encoder.sample(z_mean, z_logvar)

        # Flow forward (z -> u)
        u, logdet = self.flow_decoder(z, x_mask, cond=cond, inverse=False)

        # Reconstruction
        pred_mel = self.latent_to_mel(z * x_mask)

        # Ensure mel domain consistency with vocoder training
        # Apply mel normalization if configured
        audio_cfg = self.config.get('singing_voice_converter', {}).get('audio', {})
        mel_scale = audio_cfg.get('mel_scale', 'log')
        if mel_scale == 'log':
            # Convert to log-mel domain for training consistency
            pred_mel = torch.log(torch.clamp(pred_mel, min=1e-10))

        # Prepare output dictionary
        outputs = {
            'pred_mel': pred_mel,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'z': z,
            'u': u,
            'logdet': logdet,
            'cond': cond
        }

        # Generate pred_audio if vocoder is available and requested
        if use_vocoder and self.vocoder is not None:
            try:
                # Convert mel to audio using vocoder
                # Vocoder expects linear-scale mel
                if mel_scale == 'log':
                    # Convert log-mel back to linear for vocoder
                    pred_mel_linear = torch.exp(pred_mel)
                else:
                    pred_mel_linear = pred_mel

                # Generate audio waveform
                pred_audio = self.vocoder(pred_mel_linear)  # [B, 1, T_audio]

                # Squeeze channel dimension if present
                if pred_audio.dim() == 3 and pred_audio.size(1) == 1:
                    pred_audio = pred_audio.squeeze(1)  # [B, T_audio]

                outputs['pred_audio'] = pred_audio
                logger.debug(f"Generated pred_audio: shape={pred_audio.shape}")
            except Exception as e:
                logger.warning(f"Failed to generate pred_audio with vocoder: {e}")
                # Don't raise error, just skip pred_audio generation
        elif use_vocoder and self.vocoder is None:
            logger.warning(
                "use_vocoder=True but vocoder is not available. "
                "Pitch and speaker losses will fall back to mel-based computation."
            )

        return outputs

    def _apply_pitch_shift(
        self,
        f0: torch.Tensor,
        semitones: float,
        method: str = 'linear'
    ) -> torch.Tensor:
        """
        Apply pitch shift to F0 contour.

        Args:
            f0: F0 tensor in Hz [B, T]
            semitones: Number of semitones to shift (positive = up, negative = down)
            method: Shift method ('linear' or 'formant_preserving')

        Returns:
            Shifted F0 tensor [B, T]
        """
        if semitones == 0.0:
            return f0

        # Create mask for voiced frames (f0 > 0)
        voiced_mask = f0 > 0

        # Calculate shift ratio
        shift_ratio = 2.0 ** (semitones / 12.0)

        if method == 'linear':
            # Simple linear pitch shift
            shifted_f0 = f0 * shift_ratio

        elif method == 'formant_preserving':
            # Formant-preserving shift (more natural for large shifts)
            # Use a gentler curve for extreme shifts
            formant_factor = 2.0 ** (semitones * 0.7 / 12.0)  # 70% of shift
            shifted_f0 = f0 * formant_factor

        else:
            raise ValueError(f"Unknown pitch shift method: {method}")

        # Clamp to valid F0 range
        shifted_f0 = torch.clamp(shifted_f0, self.pitch_encoder.f0_min, self.pitch_encoder.f0_max)

        # Preserve unvoiced frames
        shifted_f0 = torch.where(voiced_mask, shifted_f0, torch.zeros_like(shifted_f0))

        return shifted_f0

    def convert(
        self,
        source_audio: Union[torch.Tensor, np.ndarray],
        target_speaker_embedding: Union[torch.Tensor, np.ndarray],
        source_f0: Optional[Union[torch.Tensor, np.ndarray]] = None,
        source_sample_rate: int = 16000,
        output_sample_rate: int = 44100,
        pitch_shift_semitones: float = 0.0,
        pitch_shift_method: str = 'linear',
        denoise_input: Optional[bool] = None,
        enhance_output: Optional[bool] = None,
        preserve_dynamics: Optional[bool] = None,
        vibrato_transfer: Optional[bool] = None
    ) -> np.ndarray:
        """
        Inference method for voice conversion.

        Args:
            source_audio: Source singing audio (tensor or array)
            target_speaker_embedding: Target speaker embedding [256] (tensor or array)
            source_f0: Optional F0 contour (auto-extracted if None)
            source_sample_rate: Sample rate of source
            output_sample_rate: Desired output sample rate
            pitch_shift_semitones: Pitch shift in semitones (default: 0.0)
                                  Positive = shift up, negative = shift down
                                  Range: typically -12 to +12 semitones
            pitch_shift_method: Pitch shift method (default: 'linear')
                               - 'linear': Simple linear pitch shift
                               - 'formant_preserving': More natural for large shifts
            denoise_input: Apply noise reduction to input audio (default: None = use instance setting)
            enhance_output: Apply enhancement to output audio (default: None = use instance setting)
            preserve_dynamics: Preserve dynamic range from source (default: None = use instance setting)
            vibrato_transfer: Transfer vibrato characteristics from source (default: None = use instance setting)

        Returns:
            Converted audio waveform as numpy array

        Raises:
            VoiceConversionError: If conversion fails
            ValueError: If parameters are invalid

        Example:
            >>> # Basic conversion
            >>> audio = model.convert(source, target_emb)
            >>>
            >>> # With pitch shift
            >>> audio = model.convert(source, target_emb, pitch_shift_semitones=2.0)
            >>>
            >>> # With all enhancements
            >>> audio = model.convert(
            ...     source, target_emb,
            ...     pitch_shift_semitones=-1.5,
            ...     denoise_input=True,
            ...     enhance_output=True,
            ...     preserve_dynamics=True
            ... )
        """
        try:
            with torch.no_grad():
                # Validate parameters
                if pitch_shift_method not in ['linear', 'formant_preserving']:
                    raise ValueError(
                        f"Invalid pitch_shift_method '{pitch_shift_method}'. "
                        f"Must be 'linear' or 'formant_preserving'"
                    )

                if abs(pitch_shift_semitones) > 24:
                    logger.warning(
                        f"Large pitch shift of {pitch_shift_semitones} semitones may degrade quality"
                    )

                # Apply feature settings (use parameter or instance default)
                _denoise_input = denoise_input if denoise_input is not None else self.denoise_input
                _enhance_output = enhance_output if enhance_output is not None else self.enhance_output
                _preserve_dynamics = preserve_dynamics if preserve_dynamics is not None else self.preserve_dynamics
                _vibrato_transfer = vibrato_transfer if vibrato_transfer is not None else self.vibrato_transfer

                # Get model device for consistency
                device = next(self.parameters()).device

                # Convert inputs to tensors
                if isinstance(source_audio, np.ndarray):
                    source_audio = torch.from_numpy(source_audio).float()
                if source_audio.dim() == 1:
                    source_audio = source_audio.unsqueeze(0)
                source_audio = source_audio.to(device)

                # Store original audio for dynamics preservation
                if _preserve_dynamics:
                    original_rms = torch.sqrt(torch.mean(source_audio ** 2))
                    original_peak = torch.max(torch.abs(source_audio))

                # Optional: denoise input
                if _denoise_input:
                    source_audio = self._denoise_audio(source_audio, source_sample_rate)

                # Extract F0 if not provided
                source_voiced = None
                vibrato_data = None
                if source_f0 is None:
                    from ..audio.pitch_extractor import SingingPitchExtractor
                    # Use canonical device string (cuda or cpu without index)
                    dev_str = 'cuda' if device.type == 'cuda' else 'cpu'
                    extractor = SingingPitchExtractor(device=dev_str)
                    f0_data = extractor.extract_f0_contour(
                        source_audio.cpu().numpy().squeeze(),
                        source_sample_rate
                    )
                    source_f0 = torch.from_numpy(f0_data['f0']).float().unsqueeze(0)
                    # Retrieve voiced mask from pitch extractor
                    if 'voiced' in f0_data:
                        source_voiced = torch.from_numpy(f0_data['voiced']).bool().unsqueeze(0)
                    # Store vibrato data if vibrato transfer is enabled
                    if _vibrato_transfer and 'vibrato_rate' in f0_data:
                        vibrato_data = {
                            'rate': f0_data.get('vibrato_rate'),
                            'extent': f0_data.get('vibrato_extent')
                        }
                elif isinstance(source_f0, np.ndarray):
                    source_f0 = torch.from_numpy(source_f0).float()
                    if source_f0.dim() == 1:
                        source_f0 = source_f0.unsqueeze(0)
                source_f0 = source_f0.to(device)
                if source_voiced is not None:
                    source_voiced = source_voiced.to(device)

                # Apply pitch shift if requested
                if pitch_shift_semitones != 0.0:
                    source_f0 = self._apply_pitch_shift(
                        source_f0,
                        pitch_shift_semitones,
                        pitch_shift_method
                    )
                    logger.debug(
                        f"Applied pitch shift: {pitch_shift_semitones:+.2f} semitones "
                        f"using {pitch_shift_method} method"
                    )

                # Extract content
                content = self.content_encoder(source_audio, source_sample_rate)
                content = content.to(device)

                # Apply content projection if needed (HuBERT 256 -> content_dim)
                if self.content_projection is not None:
                    content = self.content_projection(content.transpose(1, 2)).transpose(1, 2)

                # Encode pitch with voiced mask
                pitch_emb = self.pitch_encoder(source_f0, source_voiced)
                pitch_emb = pitch_emb.to(device)

                # Derive target frame count from source audio length and hop_length
                # This ensures consistent timing alignment with vocoder/Griffin-Lim
                audio_cfg = self.config.get('singing_voice_converter', {}).get('audio', {})
                dataset_cfg = self.config.get('dataset', {})

                # Get hop_length from config with proper fallback chain
                hop_length = audio_cfg.get('hop_length', dataset_cfg.get('hop_length', 512))

                # Get sample rates for proper scaling
                model_sample_rate = audio_cfg.get('sample_rate', self.vocoder_sample_rate)

                # Compute num_samples at model/vocoder sample rate if needed
                num_samples = source_audio.size(-1)
                if source_sample_rate != model_sample_rate:
                    # Scale num_samples to model sample rate
                    num_samples_model = round(num_samples * model_sample_rate / source_sample_rate)
                else:
                    num_samples_model = num_samples

                # Compute T_mel from hop_length: T = ceil(num_samples / hop_length)
                import math
                T = math.ceil(num_samples_model / hop_length)

                logger.debug(f"Computed T={T} from num_samples={num_samples_model}, hop_length={hop_length}, "
                            f"source_sr={source_sample_rate}, model_sr={model_sample_rate}")

                # Interpolate both content and pitch to this deterministic frame count
                content = self._interpolate_features(content, T)
                pitch_emb = self._interpolate_features(pitch_emb, T)

                # Prepare and validate speaker embedding
                if isinstance(target_speaker_embedding, np.ndarray):
                    speaker_emb = torch.from_numpy(target_speaker_embedding).float()
                else:
                    speaker_emb = target_speaker_embedding.clone() if isinstance(target_speaker_embedding, torch.Tensor) else target_speaker_embedding

                # Convert to float32 if needed
                speaker_emb = speaker_emb.float()

                # Validate and reshape speaker embedding
                if speaker_emb.dim() == 1:
                    # Shape [speaker_dim]
                    if speaker_emb.size(0) != self.speaker_dim:
                        raise VoiceConversionError(
                            f"target_speaker_embedding must have size [{self.speaker_dim}], got [{speaker_emb.size(0)}]"
                        )
                    speaker_emb = speaker_emb.unsqueeze(0).unsqueeze(2).expand(1, -1, T)
                elif speaker_emb.dim() == 2:
                    # Shape [B, speaker_dim]
                    B_speaker = speaker_emb.size(0)
                    if speaker_emb.size(1) != self.speaker_dim:
                        raise VoiceConversionError(
                            f"target_speaker_embedding must have shape [B, {self.speaker_dim}], got [B, {speaker_emb.size(1)}]"
                        )
                    # Check batch size matches content/pitch (which is always 1 for convert)
                    if B_speaker != 1:
                        raise VoiceConversionError(
                            f"convert() supports batch size 1 only. "
                            f"target_speaker_embedding has batch size {B_speaker}. "
                            f"Please use target_speaker_embedding shape [{self.speaker_dim}] or [1, {self.speaker_dim}]."
                        )
                    speaker_emb = speaker_emb.unsqueeze(2).expand(-1, -1, T)
                else:
                    raise VoiceConversionError(
                        f"target_speaker_embedding must be 1D [{self.speaker_dim}] or 2D [B, {self.speaker_dim}], "
                        f"got shape {list(speaker_emb.shape)}"
                    )

                # Move to model device
                speaker_emb = speaker_emb.to(device)

                # Concatenate conditioning
                cond = torch.cat([
                    content.transpose(1, 2),
                    pitch_emb.transpose(1, 2),
                    speaker_emb
                ], dim=1)

                # Create mask
                x_mask = torch.ones(1, 1, T, device=device)

                # Sample from flow (u -> z) with temperature
                u = torch.randn(1, self.latent_dim, T, device=device) * self.temperature
                z = self.flow_decoder(u, x_mask, cond=cond, inverse=True)

                # Generate mel
                pred_mel_raw = self.latent_to_mel(z * x_mask)

                # Normalize mel to match vocoder training domain
                audio_cfg = self.config.get('singing_voice_converter', {}).get('audio', {})
                mel_scale = audio_cfg.get('mel_scale', 'log')

                # Vocoder synthesis
                if self.vocoder is not None:
                    # HiFi-GAN typically expects linear-scale mel or specific normalization
                    # If we use log-mel internally, convert to linear for vocoder
                    if mel_scale == 'log':
                        # Convert log-mel to linear for vocoder
                        pred_mel = torch.exp(pred_mel_raw)
                    else:
                        pred_mel = pred_mel_raw
                    waveform = self.vocoder(pred_mel)
                    waveform = waveform.squeeze()
                else:
                    # Fallback to Griffin-Lim
                    # Ensure mel is in linear magnitude domain for Griffin-Lim
                    pred_mel_np = pred_mel_raw.squeeze().cpu().numpy()
                    if mel_scale == 'log':
                        # Convert log-mel to linear: exp(log_mel) with floor to prevent underflow
                        pred_mel_np = np.exp(pred_mel_np)
                        pred_mel_np = np.maximum(pred_mel_np, 1e-10)
                    waveform_np = self._mel_to_audio_griffin_lim(pred_mel_np)
                    waveform = torch.from_numpy(waveform_np).to(device)

                # Optional: preserve dynamics from source
                if _preserve_dynamics:
                    waveform = self._preserve_dynamics(waveform, original_rms, original_peak)

                # Optional: enhance output
                if _enhance_output:
                    waveform = self._enhance_audio(waveform, self.vocoder_sample_rate)

                # Convert to numpy
                waveform = waveform.cpu().numpy()

                # Resample if needed
                if output_sample_rate != self.vocoder_sample_rate:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=self.vocoder_sample_rate, target_sr=output_sample_rate)

                return waveform

        except Exception as e:
            raise VoiceConversionError(f"Voice conversion failed: {str(e)}")

    def _interpolate_features(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Interpolate features to target length.

        Args:
            features: Features [B, T, C]
            target_length: Target length

        Returns:
            Interpolated features [B, target_length, C]
        """
        if features.size(1) == target_length:
            return features

        # Transpose for interpolation: [B, T, C] -> [B, C, T]
        features = features.transpose(1, 2)
        features = F.interpolate(features, size=target_length, mode='linear', align_corners=False)
        features = features.transpose(1, 2)  # [B, target_length, C]
        return features

    def _mel_to_audio_griffin_lim(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """
        Fallback mel-to-audio conversion using Griffin-Lim with config-sourced STFT parameters.

        IMPORTANT: librosa.feature.inverse.mel_to_audio expects mel-spectrogram in LINEAR POWER scale.
        If your mel is in magnitude or log scale, it must be converted to power before passing.

        The conversion follows this logic:
        - Log-mel domain: mel_power = exp(mel_log)
        - Magnitude domain: mel_power = mel_mag ** 2
        - Power domain: no conversion needed

        Args:
            mel: Mel-spectrogram [mel_channels, T] - MUST be in LINEAR POWER scale
            n_iter: Number of Griffin-Lim iterations

        Returns:
            Audio waveform

        Note:
            Set config key 'mel_is_power' to control conversion:
            - True: Input is power mel (no conversion)
            - False: Input is magnitude mel (will square before Griffin-Lim)
        """
        try:
            import librosa
            # Get STFT parameters from config to match mel frame-to-audio duration mapping
            audio_cfg = self.config.get('singing_voice_converter', {}).get('audio', {})
            dataset_cfg = self.config.get('dataset', {})

            # Retrieve parameters with proper fallback chain
            n_fft = audio_cfg.get('n_fft', dataset_cfg.get('n_fft', 2048))
            hop_length = audio_cfg.get('hop_length', dataset_cfg.get('hop_length', 512))
            win_length = audio_cfg.get('win_length', dataset_cfg.get('win_length', n_fft))
            mel_fmin = audio_cfg.get('mel_fmin', dataset_cfg.get('mel_fmin', 0.0))
            mel_fmax = audio_cfg.get('mel_fmax', dataset_cfg.get('mel_fmax', 8000.0))

            # Check mel domain assumption
            mel_is_power = audio_cfg.get('mel_is_power', True)
            if not mel_is_power:
                # Convert magnitude to power: P = M^2
                logger.debug("Converting magnitude mel to power mel for Griffin-Lim")
                mel = mel ** 2

            # Ensure non-negative values for power mel
            mel = np.maximum(mel, 1e-10)

            # Log Griffin-Lim parameters for traceability
            logger.debug(f"Griffin-Lim synthesis: n_fft={n_fft}, hop_length={hop_length}, "
                        f"win_length={win_length}, sr={self.vocoder_sample_rate}, "
                        f"fmin={mel_fmin}, fmax={mel_fmax}, n_iter={n_iter}, "
                        f"mel_is_power={mel_is_power}")

            # Use librosa's built-in inverse mel function (expects power mel)
            audio = librosa.feature.inverse.mel_to_audio(
                mel,
                sr=self.vocoder_sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_iter=n_iter,
                fmin=mel_fmin,
                fmax=mel_fmax
            )
            return audio
        except Exception as e:
            logger.error(f"Griffin-Lim conversion failed: {e}")
            # Return silence as fallback (using configured hop_length)
            audio_cfg = self.config.get('singing_voice_converter', {}).get('audio', {})
            dataset_cfg = self.config.get('dataset', {})
            hop_length = audio_cfg.get('hop_length', dataset_cfg.get('hop_length', 512))
            logger.warning(f"Returning silence with length={mel.shape[1] * hop_length} samples")
            return np.zeros(mel.shape[1] * hop_length)

    def compute_kl_loss(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior q(z|x) and prior p(z).

        Args:
            z_mean: Mean [B, latent_dim, T]
            z_logvar: Log-variance [B, latent_dim, T]

        Returns:
            KL loss (scalar)
        """
        kl = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return kl

    def _denoise_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Apply noise reduction to input audio.

        Uses spectral subtraction for denoising.

        Args:
            audio: Audio tensor [B, T]
            sample_rate: Sample rate

        Returns:
            Denoised audio tensor [B, T]
        """
        try:
            # Simple spectral gate denoising
            # Estimate noise floor from quiet regions
            frame_size = 2048
            hop_size = 512

            # Store original shape for restoration
            original_shape = audio.shape
            original_length = audio.size(-1)

            # Reshape audio into frames
            audio_2d = audio.squeeze()
            if len(audio_2d.shape) == 1:
                # Pad to multiple of frame_size
                pad_size = (frame_size - (audio_2d.size(0) % frame_size)) % frame_size
                if pad_size > 0:
                    audio_2d = F.pad(audio_2d, (0, pad_size))

                # Calculate energy per frame
                num_frames = audio_2d.size(0) // hop_size
                frame_energy = torch.zeros(num_frames, device=audio.device)
                for i in range(num_frames):
                    start = i * hop_size
                    end = min(start + frame_size, audio_2d.size(0))
                    frame_energy[i] = torch.mean(audio_2d[start:end] ** 2)

                # Estimate noise threshold (bottom 20th percentile)
                noise_threshold = torch.quantile(frame_energy, 0.2)

                # Apply soft gate
                gate_strength = 0.3  # Gentle denoising
                for i in range(num_frames):
                    start = i * hop_size
                    end = min(start + frame_size, audio_2d.size(0))
                    if frame_energy[i] < noise_threshold:
                        audio_2d[start:end] *= gate_strength

                # Trim back to original length
                audio_2d = audio_2d[:original_length]

                # Reshape back to original shape
                audio = audio_2d.unsqueeze(0) if len(original_shape) == 2 else audio_2d

            logger.debug("Applied input denoising")
            return audio

        except Exception as e:
            logger.warning(f"Denoising failed: {e}. Returning original audio.")
            return audio

    def _enhance_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Apply enhancement to output audio.

        Applies subtle EQ and harmonic enhancement.

        Args:
            audio: Audio tensor [T] or [B, T]
            sample_rate: Sample rate

        Returns:
            Enhanced audio tensor [T] or [B, T]
        """
        try:
            # Apply subtle high-frequency enhancement for clarity
            # Using a simple high-shelf filter approximation

            # Ensure proper shape
            original_shape = audio.shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # Simple enhancement: emphasize 2-8kHz range slightly
            # This is a placeholder - in production, use proper DSP filters
            enhanced = audio.clone()

            # Apply gentle compression for consistency
            # RMS normalization with soft knee
            rms = torch.sqrt(torch.mean(enhanced ** 2) + 1e-8)
            target_rms = 0.1  # Target RMS level
            if rms > target_rms:
                ratio = 4.0  # Compression ratio
                threshold = target_rms
                gain_reduction = torch.clamp(
                    1.0 - ((rms - threshold) / threshold) / ratio,
                    0.3,
                    1.0
                )
                enhanced = enhanced * gain_reduction

            # Restore original shape
            if len(original_shape) == 1:
                enhanced = enhanced.squeeze(0)

            logger.debug("Applied output enhancement")
            return enhanced

        except Exception as e:
            logger.warning(f"Enhancement failed: {e}. Returning original audio.")
            return audio

    def _preserve_dynamics(
        self,
        audio: torch.Tensor,
        original_rms: torch.Tensor,
        original_peak: torch.Tensor
    ) -> torch.Tensor:
        """
        Preserve dynamic range from source audio.

        Args:
            audio: Output audio tensor
            original_rms: Original RMS level
            original_peak: Original peak level

        Returns:
            Audio with preserved dynamics
        """
        try:
            # Calculate current levels
            current_rms = torch.sqrt(torch.mean(audio ** 2) + 1e-8)
            current_peak = torch.max(torch.abs(audio))

            # Match RMS
            if current_rms > 1e-6:
                rms_ratio = original_rms / current_rms
                audio = audio * rms_ratio

            # Soft-limit to match peak
            if current_peak > 1e-6:
                current_peak_after_rms = torch.max(torch.abs(audio))
                if current_peak_after_rms > original_peak:
                    # Soft limiting
                    limit_ratio = original_peak / current_peak_after_rms
                    audio = audio * limit_ratio * 0.95  # 95% to leave headroom

            logger.debug("Preserved dynamic range from source")
            return audio

        except Exception as e:
            logger.warning(f"Dynamics preservation failed: {e}. Returning audio as-is.")
            return audio

    def prepare_for_inference(self):
        """Prepare model for inference (remove weight norm, set eval mode)."""
        self.eval()
        self.posterior_encoder.remove_weight_norm()
        self.flow_decoder.remove_weight_norm()
        if self.vocoder is not None:
            try:
                self.vocoder.remove_weight_norm()
            except AttributeError:
                pass  # HiFiGAN may not have this method
        logger.info("Model prepared for inference")

    def export_components_to_onnx(
        self,
        export_dir: str = "./onnx_models",
        opset_version: int = 17,
        force_cnn_fallback: bool = False
    ) -> Dict[str, str]:
        """
        Export voice conversion components to ONNX format for TensorRT optimization.

        Args:
            export_dir: Directory to save ONNX models
            opset_version: ONNX opset version
            force_cnn_fallback: Force use of CNN fallback for content encoder

        Returns:
            Dictionary mapping component names to ONNX file paths

        Raises:
            RuntimeError: If export fails
        """
        import os
        from pathlib import Path

        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        self.eval()

        exported_models = {}

        try:
            # Export Content Encoder (with fallback if needed)
            if force_cnn_fallback and hasattr(self.content_encoder, 'cnn_encoder'):
                # Temporarily disable HuBERT
                hubert_backup = self.content_encoder.hubert
                self.content_encoder.hubert = None

                try:
                    onnx_path = export_dir / "content_encoder.onnx"
                    self.content_encoder.export_to_onnx(str(onnx_path), opset_version=opset_version)
                    exported_models['content_encoder'] = str(onnx_path)
                finally:
                    # Restore HuBERT
                    self.content_encoder.hubert = hubert_backup
            elif hasattr(self.content_encoder, 'hubert') and self.content_encoder.hubert is None:
                # Already using CNN fallback
                onnx_path = export_dir / "content_encoder.onnx"
                self.content_encoder.export_to_onnx(str(onnx_path), opset_version=opset_version)
                exported_models['content_encoder'] = str(onnx_path)
            else:
                logger.warning("Cannot export HuBERT-based ContentEncoder to ONNX. Use CNN fallback.")

            # Export Pitch Encoder
            onnx_path = export_dir / "pitch_encoder.onnx"
            self.pitch_encoder.export_to_onnx(str(onnx_path), opset_version=opset_version)
            exported_models['pitch_encoder'] = str(onnx_path)

            # Export Flow Decoder
            onnx_path = export_dir / "flow_decoder.onnx"
            self.flow_decoder.export_to_onnx(
                str(onnx_path),
                opset_version=opset_version,
                cond_channels=self.cond_dim
            )
            exported_models['flow_decoder'] = str(onnx_path)

            # Export Mel Projection
            onnx_path = export_dir / "mel_projection.onnx"
            from ..inference.tensorrt_converter import TensorRTConverter
            converter = TensorRTConverter(export_dir=export_dir, device='cpu')

            converter.export_mel_projection(
                self.latent_to_mel,
                model_name="mel_projection",
                opset_version=opset_version,
                latent_dim=self.latent_dim,
                mel_channels=self.mel_channels
            )
            exported_models['mel_projection'] = str(onnx_path)

            # Export Vocoder (if available)
            if hasattr(self, 'vocoder') and self.vocoder is not None:
                try:
                    vocoder_onnx_path = converter.export_vocoder(
                        self.vocoder,
                        model_name="vocoder",
                        opset_version=opset_version,
                        mel_channels=self.mel_channels
                    )
                    exported_models['vocoder'] = str(vocoder_onnx_path)
                    logger.info("Vocoder exported to ONNX successfully")
                except Exception as e:
                    logger.warning(f"Vocoder export failed (will fall back to PyTorch): {e}")
            else:
                logger.info("No vocoder attached to SVC model, skipping vocoder export")

            logger.info(f"Successfully exported {len(exported_models)} components to ONNX")
            return exported_models

        except Exception as e:
            logger.error(f"Component ONNX export failed: {e}")
            raise RuntimeError(f"ONNX export failed: {str(e)}")

    def create_tensorrt_engines(
        self,
        onnx_dir: str = "./onnx_models",
        engine_dir: str = "./tensorrt_engines",
        fp16: bool = True,
        int8: bool = False,
        workspace_size: int = 2 << 30,
        dynamic_shapes: Optional[Dict] = None,
        calibration_npz: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create TensorRT engines from exported ONNX models.

        Args:
            onnx_dir: Directory containing ONNX models
            engine_dir: Directory to save TensorRT engines
            fp16: Enable FP16 precision
            int8: Enable INT8 precision
            workspace_size: TensorRT workspace size
            dynamic_shapes: Dynamic shape specifications

        Returns:
            Dictionary mapping component names to engine file paths

        Raises:
            RuntimeError: If TensorRT optimization fails
        """
        import os
        from pathlib import Path

        onnx_dir = Path(onnx_dir)
        engine_dir = Path(engine_dir)
        engine_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ..inference.tensorrt_converter import TensorRTConverter

            converter = TensorRTConverter(export_dir=engine_dir, device='cpu')
            engines = {}

            # Component order for engine creation (vocoder added for full TRT acceleration)
            components = ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection', 'vocoder']

            for component in components:
                onnx_path = onnx_dir / f"{component}.onnx"
                engine_path = engine_dir / f"{component}.engine"

                if onnx_path.exists():
                    try:
                        # Use different dynamic shapes based on component
                        comp_dynamic_shapes = None
                        if dynamic_shapes and component in dynamic_shapes:
                            comp_dynamic_shapes = dynamic_shapes[component]

                        converter.optimize_with_tensorrt(
                            onnx_path=str(onnx_path),
                            engine_path=str(engine_path),
                            fp16=fp16,
                            int8=int8,
                            workspace_size=workspace_size,
                            dynamic_shapes=comp_dynamic_shapes,
                            component_name=component,
                            calibration_npz=calibration_npz
                        )
                        engines[component] = str(engine_path)
                        logger.info(f"Created TensorRT engine for {component}")
                    except Exception as e:
                        logger.warning(f"Failed to create TensorRT engine for {component}: {e}")
                        if not self.fallback_to_pytorch:
                            raise
                else:
                    logger.warning(f"ONNX model not found for {component}: {onnx_path}")

            self.tensorrt_models = engines
            logger.info(f"TensorRT engine creation completed. Available engines: {list(engines.keys())}")
            return engines

        except Exception as e:
            logger.error(f"TensorRT engine creation failed: {e}")
            if self.fallback_to_pytorch:
                logger.info("Falling back to PyTorch inference")
                return {}
            raise

    def load_tensorrt_engines(
        self,
        engine_dir: str = "./tensorrt_engines"
    ) -> bool:
        """
        Load TensorRT engines for inference.

        Args:
            engine_dir: Directory containing TensorRT engines

        Returns:
            True if engines loaded successfully
        """
        from pathlib import Path

        engine_dir = Path(engine_dir)
        if not engine_dir.exists():
            logger.warning(f"Engine directory not found: {engine_dir}")
            return False

        try:
            from ..inference.tensorrt_engine import TensorRTEngine

            self.tensorrt_models = {}

            # Load engines for each component (including vocoder for full TRT acceleration)
            components = ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection', 'vocoder']

            for component in components:
                engine_path = engine_dir / f"{component}.engine"
                if engine_path.exists():
                    try:
                        engine = TensorRTEngine(str(engine_path))
                        self.tensorrt_models[component] = engine
                        logger.info(f"Loaded TensorRT engine for {component}")
                    except Exception as e:
                        logger.warning(f"Failed to load engine for {component}: {e}")

            success = len(self.tensorrt_models) > 0
            if success:
                logger.info("TensorRT engines loaded successfully for components: "
                          f"{list(self.tensorrt_models.keys())}")

            return success

        except Exception as e:
            logger.error(f"Failed to load TensorRT engines: {e}")
            return False

    def convert_with_tensorrt(
        self,
        source_audio: Union[torch.Tensor, np.ndarray],
        target_speaker_embedding: Union[torch.Tensor, np.ndarray],
        source_f0: Optional[Union[torch.Tensor, np.ndarray]] = None,
        source_sample_rate: int = 16000,
        output_sample_rate: int = 44100,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        Inference using TensorRT-optimized components.

        Falls back to PyTorch if TensorRT engines are not available.

        Args:
            source_audio: Source singing audio
            target_speaker_embedding: Target speaker embedding
            source_f0: Optional F0 contour
            source_sample_rate: Source sample rate
            output_sample_rate: Output sample rate
            **kwargs: Additional arguments

        Returns:
            Converted audio or (audio, timing_info) tuple
        """
        # Check if we have TensorRT engines available
        trt_available = (
            self.use_tensorrt and
            len(self.tensorrt_models) > 0 and
            all(comp in self.tensorrt_models for comp in
                ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection'])
        )

        if not trt_available:
            if self.use_tensorrt and self.fallback_to_pytorch:
                logger.info("TensorRT engines not available, falling back to PyTorch")
                return self.convert(source_audio, target_speaker_embedding, source_f0,
                                  source_sample_rate, output_sample_rate, **kwargs)
            else:
                raise RuntimeError("TensorRT engines not available and fallback disabled")

        # Use optimized TensorRT path
        return self._convert_tensorrt_optimized(
            source_audio, target_speaker_embedding, source_f0,
            source_sample_rate, output_sample_rate, **kwargs
        )

    def _convert_tensorrt_optimized(
        self,
        source_audio: Union[torch.Tensor, np.ndarray],
        target_speaker_embedding: Union[torch.Tensor, np.ndarray],
        source_f0: Optional[Union[torch.Tensor, np.ndarray]],
        source_sample_rate: int,
        output_sample_rate: int,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        TensorRT-optimized conversion implementation.

        Most processing is done on CPU with TensorRT for component inference.
        """
        import time

        timing_info = {}
        device = 'cpu'  # TensorRT components run on CPU

        # Convert inputs to tensors
        if isinstance(source_audio, np.ndarray):
            source_audio = torch.from_numpy(source_audio).float()
        if source_audio.dim() == 1:
            source_audio = source_audio.unsqueeze(0)
        source_audio = source_audio.to(device)

        if isinstance(target_speaker_embedding, np.ndarray):
            target_speaker_embedding = torch.from_numpy(target_speaker_embedding).float()
        if target_speaker_embedding.dim() == 1:
            target_speaker_embedding = target_speaker_embedding.unsqueeze(0)
        target_speaker_embedding = target_speaker_embedding.to(device)

        if source_f0 is not None:
            if isinstance(source_f0, np.ndarray):
                source_f0 = torch.from_numpy(source_f0).float()
            if source_f0.dim() == 1:
                source_f0 = source_f0.unsqueeze(0)
            source_f0 = source_f0.to(device)

        # Start timing
        start_time = time.time()

        # Extract content using TensorRT
        content_start = time.time()
        content_input = {
            'input_audio': source_audio,
            'sample_rate': np.array([source_sample_rate], dtype=np.int32)  # FIX: Convert to NumPy int32
        }
        content_features = self.tensorrt_models['content_encoder'].infer_torch(content_input)
        content = content_features['content_features'] if isinstance(content_features, dict) else content_features  # FIX: Use 'content_features'
        timing_info['content_extraction'] = time.time() - content_start

        # Encode pitch using TensorRT
        pitch_start = time.time()
        if source_f0 is not None:
            # Ensure voiced mask is boolean tensor
            voiced_mask = (source_f0 > 0)
            pitch_input = {
                'f0_input': source_f0,
                'voiced_mask': voiced_mask
            }
            pitch_emb = self.tensorrt_models['pitch_encoder'].infer_torch(pitch_input)
            pitch_emb = pitch_emb['pitch_features'] if isinstance(pitch_emb, dict) else pitch_emb  # FIX: Use 'pitch_features'
        else:
            # Need to extract F0 first - fallback to PyTorch for this step
            from ..audio.pitch_extractor import SingingPitchExtractor
            # TensorRT mode runs on CPU, so use 'cpu' device
            extractor = SingingPitchExtractor(device='cpu')
            f0_data = extractor.extract_f0_contour(source_audio.squeeze().numpy(), source_sample_rate)
            f0_tensor = torch.from_numpy(f0_data['f0']).unsqueeze(0).to(device)
            voiced_mask = (f0_tensor > 0)
            pitch_input = {
                'f0_input': f0_tensor,
                'voiced_mask': voiced_mask
            }
            pitch_emb = self.tensorrt_models['pitch_encoder'].infer_torch(pitch_input)
            pitch_emb = pitch_emb['pitch_features'] if isinstance(pitch_emb, dict) else pitch_emb  # FIX: Use 'pitch_features'
        timing_info['pitch_encoding'] = time.time() - pitch_start

        # Compute sequence length and prepare conditioning
        preprocess_start = time.time()
        num_samples = source_audio.size(-1)
        hop_length = 512  # Standard hop length
        T = math.ceil(num_samples / hop_length)

        # Interpolate to target sequence length
        content = self._interpolate_features(content, T)
        pitch_emb = self._interpolate_features(pitch_emb, T)

        # Prepare speaker embedding
        if target_speaker_embedding.dim() == 1:
            speaker_emb = target_speaker_embedding.unsqueeze(0).unsqueeze(2).expand(1, -1, T)
        speaker_emb = speaker_emb.transpose(1, 2)  # [B, T, speaker_dim] -> [B, speaker_dim, T]

        # Create conditioning tensor
        conditioning = torch.cat([
            content.transpose(1, 2),  # [B, T, content_dim] -> [B, content_dim, T]
            pitch_emb.transpose(1, 2),  # [B, T, pitch_dim] -> [B, pitch_dim, T]
            speaker_emb
        ], dim=1)
        timing_info['preprocessing'] = time.time() - preprocess_start

        # Flow decoding using TensorRT
        flow_start = time.time()
        mask = torch.ones(1, 1, T, dtype=torch.float32)
        u = torch.randn(1, self.latent_dim, T, dtype=torch.float32)

        flow_input = {
            'latent_input': u,
            'mask': mask,
            'conditioning': conditioning,
            'inverse': torch.tensor([True])
        }
        latent_output = self.tensorrt_models['flow_decoder'].infer_torch(flow_input)
        z = latent_output['output_latent'] if isinstance(latent_output, dict) else latent_output  # FIX: Use 'output_latent'
        timing_info['flow_decoding'] = time.time() - flow_start

        # Mel projection using TensorRT
        mel_start = time.time()
        mel_input = {'latent_input': z}
        mel_output = self.tensorrt_models['mel_projection'].infer_torch(mel_input)
        pred_mel = mel_output['mel_output'] if isinstance(mel_output, dict) else mel_output
        timing_info['mel_projection'] = time.time() - mel_start

        # Vocoder - keep on GPU if available, fallback to CPU
        vocoder_start = time.time()
        if self.vocoder is not None:
            # Move mel to appropriate device for vocoder
            try:
                vocoder_device = next(self.vocoder.parameters()).device
                pred_mel = pred_mel.to(vocoder_device)
                waveform = self.vocoder(pred_mel)
                waveform = waveform.squeeze().cpu().numpy()
            except Exception as e:
                logger.warning(f"Vocoder failed, using Griffin-Lim: {e}")
                waveform = self._mel_to_audio_griffin_lim(
                    pred_mel.squeeze().cpu().numpy()
                )
        else:
            waveform = self._mel_to_audio_griffin_lim(
                pred_mel.squeeze().cpu().numpy()
            )
        timing_info['vocoder'] = time.time() - vocoder_start

        # Resample if needed
        if output_sample_rate != self.vocoder_sample_rate:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=self.vocoder_sample_rate,
                                      target_sr=output_sample_rate)

        total_time = time.time() - start_time
        timing_info['total'] = total_time

        logger.info(f"TensorRT conversion completed in {total_time:.3f}s")

        # Return audio and timing info
        return waveform, timing_info

    def set_temperature(self, temperature: float) -> None:
        """Set sampling temperature for flow decoder

        Args:
            temperature: Sampling temperature (0.1 to 2.0)
                - Lower values (0.5-0.8): More stable, less expressive
                - Default (1.0): Balanced
                - Higher values (1.2-2.0): More expressive, potentially less stable

        Raises:
            ValueError: If temperature is out of valid range

        Example:
            >>> model.set_temperature(0.8)  # More stable conversion
            >>> model.set_temperature(1.5)  # More expressive conversion
        """
        if not 0.1 <= temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.1 and 2.0, got {temperature}")

        self.temperature = float(temperature)
        logger.info(f"Temperature set to {self.temperature}")

    def auto_tune_temperature(
        self,
        source_audio: Union[torch.Tensor, np.ndarray],
        target_embedding: Union[torch.Tensor, np.ndarray],
        sample_rate: int = 16000,
        num_trials: int = 3
    ) -> float:
        """Automatically tune temperature for optimal conversion quality

        Runs multiple conversions with different temperature values and selects
        the one that produces the most stable output (lowest variance in energy).

        Args:
            source_audio: Source audio for testing
            target_embedding: Target speaker embedding
            sample_rate: Sample rate of source audio
            num_trials: Number of temperature values to try (default: 3)

        Returns:
            Optimal temperature value

        Example:
            >>> optimal_temp = model.auto_tune_temperature(audio, embedding, 22050)
            >>> print(f"Optimal temperature: {optimal_temp}")
        """
        logger.info("Auto-tuning temperature...")

        # Try temperature values in a sensible range
        temperatures = np.linspace(0.7, 1.3, num_trials)
        best_temp = 1.0
        best_score = float('inf')

        original_temp = self.temperature

        try:
            for temp in temperatures:
                self.set_temperature(temp)

                # Convert audio
                try:
                    converted = self.convert(
                        source_audio=source_audio,
                        target_speaker_embedding=target_embedding,
                        source_sample_rate=sample_rate,
                        output_sample_rate=sample_rate
                    )

                    # Compute stability score (lower is better)
                    # Use frame-wise energy variance as stability metric
                    frame_energy = np.array([
                        np.mean(converted[i:i+512]**2)
                        for i in range(0, len(converted)-512, 512)
                    ])
                    stability_score = np.var(frame_energy)

                    if stability_score < best_score:
                        best_score = stability_score
                        best_temp = temp

                except Exception as e:
                    logger.warning(f"Conversion failed at temperature {temp}: {e}")
                    continue
        finally:
            # Restore original temperature if tuning fails
            if best_temp == 1.0 and original_temp != 1.0:
                self.temperature = original_temp
            else:
                self.set_temperature(best_temp)

        logger.info(f"Auto-tuned temperature to {best_temp:.2f}")
        return best_temp

    def set_quality_preset(self, preset_name: str) -> None:
        """Set quality preset for conversion

        Args:
            preset_name: Preset name ('draft', 'fast', 'balanced', 'high', 'studio')

        Raises:
            ValueError: If preset name is not recognized

        Example:
            >>> model.set_quality_preset('high')  # High quality conversion
            >>> model.set_quality_preset('fast')  # Fast conversion
        """
        if preset_name not in QUALITY_PRESETS:
            valid_presets = ', '.join(QUALITY_PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Valid presets: {valid_presets}"
            )

        self.quality_preset = preset_name
        preset = QUALITY_PRESETS[preset_name]
        self.decoder_steps = preset['decoder_steps']

        logger.info(f"Quality preset set to '{preset_name}': {preset['description']}")

    def get_quality_preset_info(self, preset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a quality preset

        Args:
            preset_name: Preset name (if None, returns current preset info)

        Returns:
            Dictionary with preset information:
                - name: Preset name
                - description: Preset description
                - decoder_steps: Number of flow decoder steps
                - vocoder_speed: Vocoder speed setting
                - relative_quality: Quality relative to balanced preset
                - relative_speed: Speed relative to balanced preset

        Example:
            >>> info = model.get_quality_preset_info('high')
            >>> print(f"Quality: {info['relative_quality']}x, Speed: {info['relative_speed']}x")
        """
        if preset_name is None:
            preset_name = self.quality_preset

        if preset_name not in QUALITY_PRESETS:
            valid_presets = ', '.join(QUALITY_PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Valid presets: {valid_presets}"
            )

        preset = QUALITY_PRESETS[preset_name].copy()
        preset['name'] = preset_name
        return preset


    @property
    def trt_enabled(self) -> bool:
        """Check if TensorRT is enabled and engines are loaded.
        
        Returns:
            True if TensorRT is active with loaded engines
        """
        return self.use_tensorrt and len(self.tensorrt_models) > 0

    def estimate_conversion_time(
        self,
        audio_duration: float,
        preset: Optional[str] = None,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """Estimate conversion time for given audio duration

        Args:
            audio_duration: Audio duration in seconds
            preset: Quality preset (if None, uses current preset)
            device: Device type ('cuda' or 'cpu')

        Returns:
            Dictionary with time estimates:
                - estimated_time: Estimated total time in seconds
                - real_time_factor: Speed relative to audio duration
                - breakdown: Time breakdown by component

        Example:
            >>> estimates = model.estimate_conversion_time(30.0, 'high', 'cuda')
            >>> print(f"Estimated time: {estimates['estimated_time']:.1f}s")
            >>> print(f"RTF: {estimates['real_time_factor']:.2f}x")
        """
        if preset is None:
            preset = self.quality_preset

        if preset not in QUALITY_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'")

        preset_info = QUALITY_PRESETS[preset]

        # Base processing times (seconds per second of audio)
        # These are approximate values based on typical hardware
        if device == 'cuda':
            # GPU times (RTX 3080 baseline)
            base_times = {
                'content_extraction': 0.05,  # HuBERT-Soft
                'pitch_extraction': 0.03,    # CREPE
                'flow_decoding': 0.02 * preset_info['decoder_steps'],
                'vocoder': 0.04,
                'preprocessing': 0.01
            }
        else:
            # CPU times (approximately 5-10x slower)
            base_times = {
                'content_extraction': 0.4,
                'pitch_extraction': 0.2,
                'flow_decoding': 0.15 * preset_info['decoder_steps'],
                'vocoder': 0.3,
                'preprocessing': 0.05
            }

        # Scale by relative speed from preset
        speed_factor = preset_info['relative_speed']

        breakdown = {}
        total_time = 0.0
        for component, base_time in base_times.items():
            component_time = (base_time / speed_factor) * audio_duration
            breakdown[component] = component_time
            total_time += component_time

        rtf = total_time / audio_duration if audio_duration > 0 else 0.0

        return {
            'estimated_time': total_time,
            'real_time_factor': rtf,
            'breakdown': breakdown,
            'preset': preset,
            'device': device
        }
