"""Audio output routing for dual-channel karaoke output.

Routes audio streams to separate speaker and headphone outputs:
- Speaker: Instrumental + converted voice (for audience)
- Headphone: Original song (for performer to follow along)

Supports device selection and gain control for each channel.
"""
import logging
from typing import Optional, Tuple, List, Dict, Any

import torch

logger = logging.getLogger(__name__)


class AudioOutputRouter:
    """Routes audio to separate speaker and headphone outputs.

    Speaker output: instrumental + converted voice (audience hears)
    Headphone output: original song (performer follows along)

    Args:
        sample_rate: Audio sample rate (default: 24000)
        speaker_gain: Initial speaker channel gain (default: 1.0)
        headphone_gain: Initial headphone channel gain (default: 1.0)
        voice_gain: Converted voice gain in speaker mix (default: 1.0)
        instrumental_gain: Instrumental gain in speaker mix (default: 0.8)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        speaker_gain: float = 1.0,
        headphone_gain: float = 1.0,
        voice_gain: float = 1.0,
        instrumental_gain: float = 0.8,
    ):
        self.sample_rate = sample_rate

        # Channel gains
        self.speaker_gain = speaker_gain
        self.headphone_gain = headphone_gain
        self.voice_gain = voice_gain
        self.instrumental_gain = instrumental_gain

        # Channel enable flags
        self.speaker_enabled = True
        self.headphone_enabled = True

        # Device indices (None = system default)
        self.speaker_device: Optional[int] = None
        self.headphone_device: Optional[int] = None

        logger.info("AudioOutputRouter initialized")

    def set_channel_config(
        self,
        speaker_gain: Optional[float] = None,
        headphone_gain: Optional[float] = None,
        voice_gain: Optional[float] = None,
        instrumental_gain: Optional[float] = None,
        speaker_enabled: Optional[bool] = None,
        headphone_enabled: Optional[bool] = None,
    ):
        """Configure channel settings.

        Args:
            speaker_gain: Speaker output gain (0.0 to 2.0)
            headphone_gain: Headphone output gain (0.0 to 2.0)
            voice_gain: Converted voice gain in speaker mix
            instrumental_gain: Instrumental gain in speaker mix
            speaker_enabled: Enable/disable speaker output
            headphone_enabled: Enable/disable headphone output
        """
        if speaker_gain is not None:
            self.speaker_gain = max(0.0, min(2.0, speaker_gain))
        if headphone_gain is not None:
            self.headphone_gain = max(0.0, min(2.0, headphone_gain))
        if voice_gain is not None:
            self.voice_gain = max(0.0, min(2.0, voice_gain))
        if instrumental_gain is not None:
            self.instrumental_gain = max(0.0, min(2.0, instrumental_gain))
        if speaker_enabled is not None:
            self.speaker_enabled = speaker_enabled
        if headphone_enabled is not None:
            self.headphone_enabled = headphone_enabled

    def set_devices(
        self,
        speaker_device: Optional[int] = None,
        headphone_device: Optional[int] = None,
    ):
        """Set output device indices.

        Args:
            speaker_device: Device index for speaker output (None = default)
            headphone_device: Device index for headphone output (None = default)
        """
        self.speaker_device = speaker_device
        self.headphone_device = headphone_device
        logger.info(
            f"Output devices set: speaker={speaker_device}, headphone={headphone_device}"
        )

    def route(
        self,
        converted_voice: torch.Tensor,
        instrumental: torch.Tensor,
        original_song: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route audio to speaker and headphone outputs.

        Args:
            converted_voice: Converted voice audio tensor
            instrumental: Separated instrumental track tensor
            original_song: Original song audio tensor (for performer)

        Returns:
            Tuple of (speaker_output, headphone_output) tensors
        """
        # Ensure all inputs are same length
        min_len = min(
            converted_voice.shape[0],
            instrumental.shape[0],
            original_song.shape[0]
        )
        converted_voice = converted_voice[:min_len]
        instrumental = instrumental[:min_len]
        original_song = original_song[:min_len]

        # Speaker output: instrumental + converted voice (for audience)
        if self.speaker_enabled:
            speaker_out = (
                converted_voice * self.voice_gain +
                instrumental * self.instrumental_gain
            ) * self.speaker_gain

            # Normalize to prevent clipping
            max_val = speaker_out.abs().max()
            if max_val > 1.0:
                speaker_out = speaker_out / max_val
        else:
            speaker_out = torch.zeros_like(converted_voice)

        # Headphone output: original song (for performer to follow)
        if self.headphone_enabled:
            headphone_out = original_song * self.headphone_gain

            # Normalize to prevent clipping
            max_val = headphone_out.abs().max()
            if max_val > 1.0:
                headphone_out = headphone_out / max_val
        else:
            headphone_out = torch.zeros_like(original_song)

        return speaker_out, headphone_out

    def get_config(self) -> Dict[str, Any]:
        """Get current router configuration.

        Returns:
            Dict with all router settings
        """
        return {
            'sample_rate': self.sample_rate,
            'speaker_gain': self.speaker_gain,
            'headphone_gain': self.headphone_gain,
            'voice_gain': self.voice_gain,
            'instrumental_gain': self.instrumental_gain,
            'speaker_enabled': self.speaker_enabled,
            'headphone_enabled': self.headphone_enabled,
            'speaker_device': self.speaker_device,
            'headphone_device': self.headphone_device,
        }


def list_audio_devices(device_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available audio devices.

    Args:
        device_type: Filter by type ('input', 'output', or None for all)

    Returns:
        List of device info dicts with device_id, name, type, channels, sample_rate, is_default
    """
    devices = []

    try:
        import sounddevice as sd

        device_list = sd.query_devices()
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        for idx, device in enumerate(device_list):
            # Check for input device
            if device['max_input_channels'] > 0:
                if device_type is None or device_type == 'input':
                    devices.append({
                        'index': idx,
                        'device_id': str(idx),
                        'name': device['name'],
                        'type': 'input',
                        'channels': device['max_input_channels'],
                        'sample_rate': int(device['default_samplerate']),
                        'is_default': idx == default_input,
                    })

            # Check for output device
            if device['max_output_channels'] > 0:
                if device_type is None or device_type == 'output':
                    devices.append({
                        'index': idx,
                        'device_id': str(idx),
                        'name': device['name'],
                        'type': 'output',
                        'channels': device['max_output_channels'],
                        'sample_rate': int(device['default_samplerate']),
                        'is_default': idx == default_output,
                    })

    except ImportError:
        logger.warning("sounddevice not available, returning empty device list")
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")

    return devices


def get_default_device() -> Optional[int]:
    """Get the default audio output device index.

    Returns:
        Default output device index, or None if unavailable
    """
    try:
        import sounddevice as sd
        return sd.default.device[1]
    except (ImportError, Exception):
        return None
