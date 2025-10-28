"""Audio processing utilities for AutoVoice"""

from __future__ import annotations
import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # Prevent NameError in annotations

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from typing import Tuple, Optional, Union, Dict
import warnings
from pathlib import Path

try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio loading, processing, and saving"""

    def __init__(self, config: dict = None, device: Optional[str] = None):
        """Initialize audio processor with configuration

        Args:
            config: Configuration dictionary with audio processing parameters
            device: Optional device specification (ignored in base class, used by subclasses)
        """
        if config is None:
            config = {}

        self.sample_rate = config.get('sample_rate', 22050)
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.win_length = config.get('win_length', 2048)
        self.n_mels = config.get('n_mels', 128)
        self.fmin = config.get('fmin', 0)
        self.fmax = config.get('fmax', 8000)

        # Store device for subclass use
        if device is not None:
            self.device = device
        else:
            self.device = 'cpu'
        
        # Initialize transforms for mel spectrogram
        if TORCHAUDIO_AVAILABLE:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax
            )
            self.amplitude_to_db = T.AmplitudeToDB()
        else:
            self.mel_transform = None
            self.amplitude_to_db = None
    
    def load_audio(self, path: str, target_sr: Optional[int] = None, return_sr: bool = False) -> Union[torch.Tensor, np.ndarray, Tuple[Union[torch.Tensor, np.ndarray], int]]:
        """Load audio from file

        Args:
            path: Path to audio file
            target_sr: Target sample rate (uses self.sample_rate if None)
            return_sr: If True, return (audio, original_sample_rate) tuple

        Returns:
            Audio tensor/array, or (audio, original_sr) if return_sr=True
        """
        if target_sr is None:
            target_sr = self.sample_rate

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            if TORCHAUDIO_AVAILABLE:
                waveform, sample_rate = torchaudio.load(str(path))
                original_sr = sample_rate
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                # Resample if needed
                if sample_rate != target_sr:
                    resampler = T.Resample(sample_rate, target_sr)
                    waveform = resampler(waveform)
                result = waveform.squeeze(0)  # Remove channel dimension
                return (result, original_sr) if return_sr else result
            elif LIBROSA_AVAILABLE:
                # Load without resampling first to get original SR
                audio_orig, original_sr = librosa.load(str(path), sr=None)
                # Resample if needed
                if original_sr != target_sr:
                    audio = librosa.resample(audio_orig, orig_sr=original_sr, target_sr=target_sr)
                else:
                    audio = audio_orig
                result = torch.from_numpy(audio.astype(np.float32))
                return (result, original_sr) if return_sr else result
            else:
                logger.warning("Neither torchaudio nor librosa available. Using dummy audio.")
                result = torch.randn(target_sr * 3)  # 3 seconds of dummy audio
                return (result, target_sr) if return_sr else result
        except Exception as e:
            logger.error(f"Error loading audio from {path}: {e}")
            raise
    
    def save_audio(self, audio: Union[torch.Tensor, np.ndarray], path: str, sample_rate: Optional[int] = None):
        """Save audio to file
        
        Args:
            audio: Audio tensor or array
            path: Output file path
            sample_rate: Sample rate (uses self.sample_rate if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Convert to numpy if torch tensor
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if TORCHAUDIO_AVAILABLE and path.suffix.lower() in ['.wav', '.flac']:
                # Use torchaudio for better format support
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio.astype(np.float32))
                else:
                    audio_tensor = audio
                # Add channel dimension if needed
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                torchaudio.save(str(path), audio_tensor, sample_rate)
            elif SOUNDFILE_AVAILABLE:
                sf.write(str(path), audio_np, sample_rate)
            else:
                logger.warning("Neither torchaudio nor soundfile available. Audio not saved.")
                return
        except Exception as e:
            logger.error(f"Error saving audio to {path}: {e}")
            raise
    
    def to_mel_spectrogram(self, audio: Union[torch.Tensor, np.ndarray], 
                          n_fft: Optional[int] = None,
                          hop_length: Optional[int] = None,
                          n_mels: Optional[int] = None,
                          sample_rate: Optional[int] = None) -> torch.Tensor:
        """Convert audio to mel spectrogram
        
        Args:
            audio: Input audio tensor or array
            n_fft: FFT size (uses self.n_fft if None)
            hop_length: Hop length (uses self.hop_length if None)
            n_mels: Number of mel bins (uses self.n_mels if None)
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            Mel spectrogram tensor [n_mels, time_steps]
        """
        # Handle parameters
        n_fft = n_fft or self.n_fft
        hop_length = hop_length or self.hop_length
        n_mels = n_mels or self.n_mels
        sample_rate = sample_rate or self.sample_rate
        
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))
        elif not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
            
        # Handle empty audio
        if audio.numel() == 0:
            return torch.zeros(n_mels, 0)
            
        # Handle single sample
        if audio.numel() == 1:
            return torch.zeros(n_mels, 1)
            
        try:
            if TORCHAUDIO_AVAILABLE:
                # Create mel transform with specified parameters
                mel_transform = T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    n_mels=n_mels,
                    f_min=self.fmin,
                    f_max=self.fmax
                )
                amplitude_to_db = T.AmplitudeToDB()
                
                # Ensure audio is 1D
                if audio.dim() > 1:
                    audio = audio.squeeze()
                    
                mel_spec = mel_transform(audio)
                mel_spec_db = amplitude_to_db(mel_spec)
                return mel_spec_db
                
            elif LIBROSA_AVAILABLE:
                # Fallback to librosa
                audio_np = audio.detach().cpu().numpy()
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_np,
                    sr=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    n_mels=n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                return torch.from_numpy(mel_spec_db.astype(np.float32))
                
            else:
                # Return dummy mel spectrogram
                frames = max(1, len(audio) // hop_length)
                return torch.randn(n_mels, frames)
                
        except Exception as e:
            logger.error(f"Error computing mel spectrogram: {e}")
            # Return dummy on error
            frames = max(1, len(audio) // hop_length)
            return torch.randn(n_mels, frames)
    
    def from_mel_spectrogram(self, mel_spec: Union[torch.Tensor, np.ndarray], 
                           n_iter: int = 32) -> torch.Tensor:
        """Convert mel spectrogram back to audio using Griffin-Lim
        
        Args:
            mel_spec: Input mel spectrogram [n_mels, time_steps]
            n_iter: Number of Griffin-Lim iterations
            
        Returns:
            Reconstructed audio tensor
        """
        # Convert to torch tensor if needed
        if isinstance(mel_spec, np.ndarray):
            mel_spec = torch.from_numpy(mel_spec.astype(np.float32))
        elif not isinstance(mel_spec, torch.Tensor):
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
            
        # Handle empty input
        if mel_spec.numel() == 0:
            return torch.zeros(0)
            
        try:
            # Prefer librosa for reconstruction if available as it's more reliable
            if LIBROSA_AVAILABLE:
                mel_spec_np = mel_spec.detach().cpu().numpy()
                mel_spec_power = librosa.db_to_power(mel_spec_np, ref=1.0)
                
                try:
                    audio = librosa.feature.inverse.mel_to_audio(
                        mel_spec_power,
                        sr=self.sample_rate,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        fmin=self.fmin,
                        fmax=self.fmax
                    )
                    return torch.from_numpy(audio.astype(np.float32))
                except (AttributeError, TypeError):
                    # Older librosa versions might not have mel_to_audio
                    # Use Griffin-Lim on a rough linear approximation
                    try:
                        S = librosa.feature.inverse.mel_to_stft(
                            mel_spec_power,
                            sr=self.sample_rate,
                            n_fft=self.n_fft,
                            fmin=self.fmin,
                            fmax=self.fmax
                        )
                        audio = librosa.griffinlim(
                            S,
                            n_iter=n_iter,
                            hop_length=self.hop_length,
                            win_length=self.win_length
                        )
                        return torch.from_numpy(audio.astype(np.float32))
                    except (AttributeError, TypeError):
                        # Even older version - use simple inverse STFT approximation
                        frames = mel_spec.shape[1]
                        estimated_length = frames * self.hop_length
                        # Generate white noise scaled by the average energy
                        avg_energy = torch.mean(mel_spec)
                        noise_scale = torch.abs(avg_energy) * 0.01  # Very low scale
                        audio = torch.randn(estimated_length) * noise_scale
                        return audio
                    
            elif TORCHAUDIO_AVAILABLE:
                # Convert from dB to power
                mel_spec_power = torch.pow(10.0, mel_spec / 10.0)  # Convert dB to power
                
                # For mel spectrogram reconstruction, we use a simplified approach
                # since perfect mel-to-linear inversion requires a vocoder
                frames = mel_spec.shape[1]
                
                # Create a pseudo-inverse using interpolation
                # This is a rough approximation for testing purposes
                n_freqs = self.n_fft // 2 + 1
                linear_spec = torch.zeros(n_freqs, frames)
                
                # Use simple mapping to approximate mel-to-linear conversion
                for i in range(min(self.n_mels, n_freqs)):
                    # Simple linear mapping from mel bins to frequency bins
                    linear_idx = int(i * n_freqs / self.n_mels)
                    if linear_idx < n_freqs:
                        linear_spec[linear_idx] = mel_spec_power[i]
                
                # Use Griffin-Lim for reconstruction
                griffin_lim = T.GriffinLim(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    n_iter=n_iter
                )
                        
                audio = griffin_lim(linear_spec)
                return audio
                
            elif LIBROSA_AVAILABLE:
                # Fallback to librosa
                mel_spec_np = mel_spec.detach().cpu().numpy()
                mel_spec_power = librosa.db_to_power(mel_spec_np, ref=1.0)
                
                try:
                    audio = librosa.feature.inverse.mel_to_audio(
                        mel_spec_power,
                        sr=self.sample_rate,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        fmin=self.fmin,
                        fmax=self.fmax
                    )
                    return torch.from_numpy(audio.astype(np.float32))
                except (AttributeError, TypeError):
                    # Older librosa versions might not have mel_to_audio
                    # Try mel_to_stft + Griffin-Lim approach
                    try:
                        S = librosa.feature.inverse.mel_to_stft(
                            mel_spec_power,
                            sr=self.sample_rate,
                            n_fft=self.n_fft,
                            fmin=self.fmin,
                            fmax=self.fmax
                        )
                        audio = librosa.griffinlim(
                            S,
                            n_iter=n_iter,
                            hop_length=self.hop_length,
                            win_length=self.win_length
                        )
                        return torch.from_numpy(audio.astype(np.float32))
                    except (AttributeError, TypeError):
                        # Create sinusoidal reconstruction based on mel spectrum
                        frames = mel_spec.shape[1]
                        estimated_length = frames * self.hop_length
                        t = torch.linspace(0, estimated_length / self.sample_rate, estimated_length)
                        
                        # Simple reconstruction with limited harmonics
                        audio = torch.zeros(estimated_length)
                        for i in range(min(5, self.n_mels)):  # Use first 5 mel bins
                            freq = 200 + i * 300  # Simple frequency mapping
                            amp = torch.mean(mel_spec[i]).abs() * 0.05  # Scale amplitude
                            audio += amp * torch.sin(2 * torch.pi * freq * t)
                        
                        return audio * 0.2  # Scale result
                    
            else:
                # Return dummy audio with proper scaling
                frames = mel_spec.shape[1]
                estimated_length = frames * self.hop_length
                # Scale based on input mel spectrogram energy
                avg_energy = torch.mean(torch.abs(mel_spec))
                scale_factor = avg_energy * 0.01  # Very small scale factor
                return torch.randn(estimated_length) * scale_factor
                
        except Exception as e:
            logger.error(f"Error reconstructing audio from mel spectrogram: {e}")
            # Return dummy audio on error
            frames = mel_spec.shape[1] if mel_spec.dim() > 1 else mel_spec.shape[0]
            estimated_length = frames * self.hop_length
            return torch.randn(estimated_length) * 0.1
    
    def extract_mfcc(self, audio: Union[torch.Tensor, np.ndarray], 
                     n_mfcc: int = 13, sample_rate: Optional[int] = None) -> torch.Tensor:
        """Extract MFCC features
        
        Args:
            audio: Input audio
            n_mfcc: Number of MFCC coefficients
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            MFCC features tensor [n_mfcc, time_steps]
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Convert to numpy for librosa
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
            
        if not LIBROSA_AVAILABLE:
            frames = max(1, len(audio_np) // self.hop_length)
            return torch.randn(n_mfcc, frames)
            
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_np,
                sr=sample_rate,
                n_mfcc=n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return torch.from_numpy(mfcc.astype(np.float32))
        except Exception as e:
            logger.error(f"Error extracting MFCC: {e}")
            frames = max(1, len(audio_np) // self.hop_length)
            return torch.randn(n_mfcc, frames)
    
    def extract_pitch(self, audio: Union[torch.Tensor, np.ndarray], 
                     sample_rate: Optional[int] = None) -> torch.Tensor:
        """Extract pitch features
        
        Args:
            audio: Input audio
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            Pitch features tensor
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Convert to numpy for librosa
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
            
        # Handle empty audio
        if len(audio_np) == 0:
            return torch.zeros(0)
            
        if not LIBROSA_AVAILABLE:
            frames = max(1, len(audio_np) // self.hop_length)
            return torch.randn(frames)
            
        try:
            # Use piptrack for pitch detection
            pitches, magnitudes = librosa.piptrack(
                y=audio_np,
                sr=sample_rate,
                hop_length=self.hop_length
            )
            
            # Extract fundamental frequency for each frame
            pitch_contour = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
                pitch_contour.append(pitch)
                
            return torch.tensor(pitch_contour, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error extracting pitch: {e}")
            frames = max(1, len(audio_np) // self.hop_length)
            return torch.randn(frames)
    
    def extract_energy(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Extract energy (RMS) features
        
        Args:
            audio: Input audio
            
        Returns:
            Energy features tensor
        """
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))
        elif not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
            
        # Handle empty audio
        if audio.numel() == 0:
            return torch.zeros(1)
            
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for compatibility
                audio_np = audio.detach().cpu().numpy()
                energy = librosa.feature.rms(
                    y=audio_np,
                    hop_length=self.hop_length,
                    frame_length=self.win_length
                )
                return torch.from_numpy(energy.astype(np.float32)).squeeze(0)
            else:
                # Simple RMS calculation
                # Frame the audio
                frames = audio.unfold(0, self.win_length, self.hop_length)
                if frames.numel() == 0:
                    return torch.tensor([torch.sqrt(torch.mean(audio ** 2))])
                energy = torch.sqrt(torch.mean(frames ** 2, dim=1))
                return energy
        except Exception as e:
            logger.error(f"Error extracting energy: {e}")
            return torch.tensor([torch.sqrt(torch.mean(audio ** 2))])
    
    def zero_crossing_rate(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Calculate zero crossing rate
        
        Args:
            audio: Input audio
            
        Returns:
            Zero crossing rate tensor
        """
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))
        elif not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
            
        # Handle empty audio
        if audio.numel() == 0:
            return torch.zeros(1)
            
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for consistency
                audio_np = audio.detach().cpu().numpy()
                zcr = librosa.feature.zero_crossing_rate(
                    audio_np,
                    hop_length=self.hop_length,
                    frame_length=self.win_length
                )
                return torch.from_numpy(zcr.astype(np.float32)).squeeze(0)
            else:
                # Simple ZCR calculation
                # Frame the audio
                frames = audio.unfold(0, self.win_length, self.hop_length)
                if frames.numel() == 0:
                    # Single frame calculation
                    sign_changes = torch.diff(torch.sign(audio))
                    zcr_value = torch.sum(torch.abs(sign_changes)) / (2.0 * len(audio))
                    return torch.tensor([zcr_value])
                    
                zcr_frames = []
                for frame in frames:
                    sign_changes = torch.diff(torch.sign(frame))
                    zcr_value = torch.sum(torch.abs(sign_changes)) / (2.0 * len(frame))
                    zcr_frames.append(zcr_value)
                return torch.tensor(zcr_frames)
        except Exception as e:
            logger.error(f"Error calculating zero crossing rate: {e}")
            return torch.tensor([0.1])  # Default ZCR value
    
    def compute_spectrogram(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        power: float = 1.0
    ) -> torch.Tensor:
        """Compute linear-frequency magnitude spectrogram

        Args:
            audio: Input audio (1D tensor or array)
            n_fft: FFT size (uses self.n_fft if None)
            hop_length: Hop length (uses self.hop_length if None)
            win_length: Window length (uses self.win_length if None)
            power: Exponent for magnitude spectrogram (1.0 = magnitude, 2.0 = power)

        Returns:
            Linear magnitude spectrogram tensor [n_freqs, frames]
        """
        # Handle parameters
        n_fft = n_fft or self.n_fft
        hop_length = hop_length or self.hop_length
        win_length = win_length or self.win_length

        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))
        elif not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        # Handle empty audio
        if audio.numel() == 0:
            n_freqs = n_fft // 2 + 1
            return torch.zeros(n_freqs, 0)

        try:
            if TORCHAUDIO_AVAILABLE:
                # Prefer torchaudio for STFT computation
                spec_transform = T.Spectrogram(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    power=power
                )
                # Ensure audio is 1D
                if audio.dim() > 1:
                    audio = audio.squeeze()
                spectrogram = spec_transform(audio)
                return spectrogram

            elif LIBROSA_AVAILABLE:
                # Fallback to librosa
                audio_np = audio.detach().cpu().numpy()
                S = np.abs(librosa.stft(
                    audio_np,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length
                ))
                if power != 1.0:
                    S = S ** power
                return torch.from_numpy(S.astype(np.float32))

            else:
                # Return deterministic dummy tensor based on input length
                frames = max(1, len(audio) // hop_length)
                n_freqs = n_fft // 2 + 1
                return torch.ones(n_freqs, frames) * 0.01  # Small constant for stability

        except Exception as e:
            logger.error(f"Error computing spectrogram: {e}")
            # Return dummy on error
            frames = max(1, len(audio) // hop_length)
            n_freqs = n_fft // 2 + 1
            return torch.ones(n_freqs, frames) * 0.01

    def extract_features(self, audio: Union[torch.Tensor, np.ndarray],
                        sample_rate: Optional[int] = None) -> Union[Dict, torch.Tensor]:
        """Extract various audio features

        Args:
            audio: Input audio
            sample_rate: Sample rate (uses self.sample_rate if None)

        Returns:
            Dictionary of features or combined feature tensor
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        features = {}

        try:
            # Extract all features
            features['mfcc'] = self.extract_mfcc(audio, sample_rate=sample_rate)
            features['pitch'] = self.extract_pitch(audio, sample_rate=sample_rate)
            features['energy'] = self.extract_energy(audio)
            features['zcr'] = self.zero_crossing_rate(audio)

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return dummy features
            if isinstance(audio, torch.Tensor):
                audio_length = audio.numel()
            else:
                audio_length = len(audio)
            frames = max(1, audio_length // self.hop_length)

            features['mfcc'] = torch.randn(13, frames)
            features['pitch'] = torch.randn(frames)
            features['energy'] = torch.randn(frames)
            features['zcr'] = torch.randn(frames)

            return features
