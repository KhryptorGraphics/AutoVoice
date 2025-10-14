"""High-level voice synthesis interface."""

import torch
import torchaudio
import numpy as np
from typing import Optional, Dict, Union
import logging
from ..models import VoiceTransformer, Vocoder, VoiceEncoder
from ..audio import AudioProcessor, FeatureExtractor
from .inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class VoiceSynthesizer:
    """High-level interface for voice synthesis."""

    def __init__(self, model_path: str, vocoder_path: str,
                device: Optional[torch.device] = None):
        """Initialize voice synthesizer.

        Args:
            model_path: Path to acoustic model checkpoint
            vocoder_path: Path to vocoder checkpoint
            device: Device for inference
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models
        self.acoustic_model = VoiceTransformer()
        self.acoustic_engine = InferenceEngine(
            self.acoustic_model, self.device, model_path
        )

        self.vocoder = Vocoder()
        self.vocoder_engine = InferenceEngine(
            self.vocoder, self.device, vocoder_path
        )

        # Audio processors
        self.audio_processor = AudioProcessor(device=self.device)
        self.feature_extractor = FeatureExtractor(device=self.device)

        logger.info("Voice synthesizer initialized")

    def synthesize(self, text: Optional[str] = None,
                  features: Optional[torch.Tensor] = None,
                  speaker_id: Optional[int] = None,
                  pitch_scale: float = 1.0,
                  speed_scale: float = 1.0) -> torch.Tensor:
        """Synthesize speech.

        Args:
            text: Input text (if using text-to-speech)
            features: Input acoustic features
            speaker_id: Speaker ID for multi-speaker synthesis
            pitch_scale: Pitch scaling factor
            speed_scale: Speed scaling factor

        Returns:
            Synthesized waveform
        """
        # Get acoustic features
        if text is not None:
            features = self._text_to_features(text)
        elif features is None:
            raise ValueError("Either text or features must be provided")

        # Apply pitch and speed modifications
        if pitch_scale != 1.0:
            features = self._modify_pitch(features, pitch_scale)
        if speed_scale != 1.0:
            features = self._modify_speed(features, speed_scale)

        # Prepare inputs
        inputs = {
            'x': features.unsqueeze(0),
            'speaker_id': torch.tensor([speaker_id], device=self.device) if speaker_id else None
        }

        # Generate acoustic features
        acoustic_output = self.acoustic_engine.infer(inputs)

        # Generate waveform with vocoder
        if acoustic_output.dim() == 3:
            acoustic_output = acoustic_output.squeeze(0).transpose(0, 1)

        waveform = self.vocoder_engine.infer(acoustic_output.unsqueeze(0))

        return waveform.squeeze()

    def clone_voice(self, reference_audio: Union[str, torch.Tensor],
                   target_text: Optional[str] = None,
                   target_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Clone voice from reference audio.

        Args:
            reference_audio: Path to reference audio or waveform tensor
            target_text: Target text to synthesize
            target_features: Target acoustic features

        Returns:
            Synthesized waveform with cloned voice
        """
        # Load reference audio
        if isinstance(reference_audio, str):
            ref_waveform, _ = self.audio_processor.load_audio(reference_audio)
        else:
            ref_waveform = reference_audio

        # Extract speaker embedding
        speaker_embedding = self._extract_speaker_embedding(ref_waveform)

        # Get target features
        if target_text is not None:
            target_features = self._text_to_features(target_text)
        elif target_features is None:
            # Use reference features if no target provided
            target_features = self.feature_extractor.extract_mel_spectrogram(ref_waveform)

        # Synthesize with speaker embedding
        waveform = self._synthesize_with_embedding(target_features, speaker_embedding)

        return waveform

    def convert_voice(self, source_audio: Union[str, torch.Tensor],
                     target_speaker: Union[int, torch.Tensor]) -> torch.Tensor:
        """Convert voice from source to target speaker.

        Args:
            source_audio: Source audio path or waveform
            target_speaker: Target speaker ID or embedding

        Returns:
            Converted waveform
        """
        # Load source audio
        if isinstance(source_audio, str):
            source_waveform, _ = self.audio_processor.load_audio(source_audio)
        else:
            source_waveform = source_audio

        # Extract content features
        content_features = self._extract_content_features(source_waveform)

        # Get target speaker embedding
        if isinstance(target_speaker, int):
            target_embedding = self._get_speaker_embedding(target_speaker)
        else:
            target_embedding = target_speaker

        # Synthesize with target speaker
        waveform = self._synthesize_with_embedding(content_features, target_embedding)

        return waveform

    def _text_to_features(self, text: str) -> torch.Tensor:
        """Convert text to acoustic features.

        Args:
            text: Input text

        Returns:
            Acoustic features
        """
        # Placeholder - would use text-to-phoneme and duration model
        # For now, return random features
        length = len(text) * 10  # Rough estimate
        features = torch.randn(length, 80, device=self.device)
        return features

    def _extract_speaker_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from waveform.

        Args:
            waveform: Input waveform

        Returns:
            Speaker embedding
        """
        # Extract mel spectrogram
        mel = self.feature_extractor.extract_mel_spectrogram(waveform)

        # Use encoder if available
        if hasattr(self, 'speaker_encoder'):
            _, embedding = self.speaker_encoder(mel.transpose(0, 1).unsqueeze(0))
            return embedding.squeeze(0)
        else:
            # Simple averaging as placeholder
            return mel.mean(dim=-1)

    def _extract_content_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract content features from waveform.

        Args:
            waveform: Input waveform

        Returns:
            Content features
        """
        # Extract mel spectrogram
        mel = self.feature_extractor.extract_mel_spectrogram(waveform)

        # Remove speaker information (simplified)
        content = mel - mel.mean(dim=-1, keepdim=True)

        return content.transpose(0, 1)

    def _get_speaker_embedding(self, speaker_id: int) -> torch.Tensor:
        """Get speaker embedding for ID.

        Args:
            speaker_id: Speaker ID

        Returns:
            Speaker embedding
        """
        # Use speaker embedding from model
        if hasattr(self.acoustic_model, 'speaker_embedding'):
            embedding = self.acoustic_model.speaker_embedding(
                torch.tensor([speaker_id], device=self.device)
            )
            return embedding.squeeze(0)
        else:
            # Random embedding as placeholder
            return torch.randn(256, device=self.device)

    def _synthesize_with_embedding(self, features: torch.Tensor,
                                  embedding: torch.Tensor) -> torch.Tensor:
        """Synthesize with speaker embedding.

        Args:
            features: Acoustic features
            embedding: Speaker embedding

        Returns:
            Synthesized waveform
        """
        # Prepare inputs
        batch_size, seq_len = features.shape[:2]

        # Expand embedding to match sequence length
        embedding_expanded = embedding.unsqueeze(0).unsqueeze(1)
        embedding_expanded = embedding_expanded.expand(1, seq_len, -1)

        # Combine features and embedding
        if features.dim() == 2:
            features = features.unsqueeze(0)

        # Generate acoustic output
        acoustic_output = self.acoustic_model(features)

        # Generate waveform
        if acoustic_output.dim() == 3:
            acoustic_output = acoustic_output.squeeze(0).transpose(0, 1)

        waveform = self.vocoder_engine.infer(acoustic_output.unsqueeze(0))

        return waveform.squeeze()

    def _modify_pitch(self, features: torch.Tensor, scale: float) -> torch.Tensor:
        """Modify pitch of features.

        Args:
            features: Input features
            scale: Pitch scaling factor

        Returns:
            Modified features
        """
        # Simple frequency scaling in mel space
        if scale == 1.0:
            return features

        # Apply scaling to mel frequencies
        scaled = features * scale

        return scaled

    def _modify_speed(self, features: torch.Tensor, scale: float) -> torch.Tensor:
        """Modify speed of features.

        Args:
            features: Input features
            scale: Speed scaling factor

        Returns:
            Modified features
        """
        if scale == 1.0:
            return features

        # Resample temporal dimension
        orig_len = features.shape[0]
        new_len = int(orig_len / scale)

        indices = torch.linspace(0, orig_len - 1, new_len, device=self.device)
        indices = indices.long().clamp(0, orig_len - 1)

        scaled = features[indices]

        return scaled

    def save_audio(self, waveform: torch.Tensor, path: str,
                  sample_rate: int = 44100):
        """Save waveform to audio file.

        Args:
            waveform: Audio waveform
            path: Output file path
            sample_rate: Sample rate
        """
        # Ensure proper shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Normalize
        waveform = self.audio_processor.normalize_audio(waveform)

        # Save
        torchaudio.save(path, waveform.cpu(), sample_rate)
        logger.info(f"Audio saved to {path}")


class BatchVoiceSynthesizer(VoiceSynthesizer):
    """Batch processing voice synthesizer."""

    def batch_synthesize(self, texts: list, speaker_ids: Optional[list] = None,
                        batch_size: int = 8) -> list:
        """Synthesize multiple texts in batch.

        Args:
            texts: List of input texts
            speaker_ids: List of speaker IDs
            batch_size: Batch size for processing

        Returns:
            List of synthesized waveforms
        """
        waveforms = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_speakers = speaker_ids[i:i + batch_size] if speaker_ids else [None] * len(batch_texts)

            # Process batch
            batch_waveforms = []
            for text, speaker in zip(batch_texts, batch_speakers):
                waveform = self.synthesize(text=text, speaker_id=speaker)
                batch_waveforms.append(waveform)

            waveforms.extend(batch_waveforms)

        return waveforms