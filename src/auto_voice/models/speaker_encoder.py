"""Speaker embedding extraction using resemblyzer for voice cloning"""

from __future__ import annotations
import logging
import threading
from typing import Optional, Union, List, Dict, Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpeakerEncodingError(Exception):
    """Exception raised when speaker encoding fails"""
    pass


class SpeakerEncoder:
    """Extract 256-dimensional speaker embeddings from audio using pre-trained resemblyzer model

    This class provides speaker embedding extraction using the resemblyzer library, which implements
    a pre-trained speaker verification model (GE2E loss-based). The embeddings can be used for:
    - Voice cloning and voice conversion
    - Speaker verification and identification
    - Voice profiling and similarity comparison

    Features:
        - Production-ready pre-trained model (no training required)
        - 256-dimensional speaker embeddings
        - GPU-accelerated inference
        - Deterministic embeddings for same input
        - Thread-safe inference
        - Batch processing support

    Example:
        >>> encoder = SpeakerEncoder(device='cuda')
        >>> embedding = encoder.extract_embedding('voice.wav')
        >>> print(embedding.shape)  # (256,)
        >>>
        >>> # Compare two voices
        >>> emb1 = encoder.extract_embedding('voice1.wav')
        >>> emb2 = encoder.extract_embedding('voice2.wav')
        >>> similarity = encoder.compute_similarity(emb1, emb2)
        >>> print(f"Similarity: {similarity:.3f}")

    Attributes:
        device (str): Device for processing ('cuda', 'cpu')
        gpu_manager: Optional GPUManager for GPU acceleration
        encoder: Resemblyzer VoiceEncoder instance
        lock (threading.RLock): Thread safety lock
    """

    def __init__(
        self,
        device: Optional[str] = None,
        gpu_manager: Optional[Any] = None
    ):
        """Initialize SpeakerEncoder with resemblyzer VoiceEncoder

        Args:
            device: Optional device string ('cuda', 'cpu')
            gpu_manager: Optional GPUManager instance for GPU acceleration

        Raises:
            SpeakerEncodingError: If resemblyzer is not available
        """
        if not RESEMBLYZER_AVAILABLE:
            raise SpeakerEncodingError(
                "resemblyzer is not available. Install with: pip install resemblyzer"
            )

        if not NUMPY_AVAILABLE:
            raise SpeakerEncodingError("numpy is required for speaker encoding")

        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.gpu_manager = gpu_manager

        # Set device
        if device is not None:
            self.device = device
        elif gpu_manager is not None and hasattr(gpu_manager, 'device'):
            self.device = str(gpu_manager.device) if hasattr(gpu_manager.device, '__str__') else 'cpu'
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Initialize resemblyzer encoder
        try:
            # Resemblyzer automatically uses GPU if CUDA is available
            self.encoder = VoiceEncoder(device=self.device)
            self.logger.info(f"SpeakerEncoder initialized with device={self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize resemblyzer encoder: {e}")
            raise SpeakerEncodingError(f"Failed to initialize encoder: {e}")

    def extract_embedding(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Extract 256-dimensional speaker embedding from audio

        This is the primary method for extracting speaker embeddings. It handles
        various input formats (file paths, numpy arrays, torch tensors) and
        validates audio quality for optimal embedding extraction.

        Args:
            audio: Audio as numpy array, torch tensor, or file path
                  - File path: Resemblyzer handles loading and preprocessing
                  - Array/tensor: Should be mono, float32, sample rate 16kHz-48kHz
            sample_rate: Sample rate (optional, required for array/tensor input)

        Returns:
            256-dimensional speaker embedding as numpy array

        Raises:
            SpeakerEncodingError: If extraction fails or audio is invalid

        Example:
            >>> # From file
            >>> embedding = encoder.extract_embedding('voice.wav')
            >>>
            >>> # From array
            >>> audio, sr = librosa.load('voice.wav', sr=16000)
            >>> embedding = encoder.extract_embedding(audio, sample_rate=sr)
        """
        with self.lock:
            try:
                # Handle file path input
                if isinstance(audio, str):
                    # Resemblyzer handles loading and preprocessing
                    wav = preprocess_wav(audio)
                    embedding = self.encoder.embed_utterance(wav)

                # Handle torch tensor input
                elif TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
                    # Convert to numpy
                    audio_np = audio.detach().cpu().numpy()

                    # Ensure mono
                    if audio_np.ndim > 1:
                        audio_np = audio_np.mean(axis=0)

                    # Ensure float32 and normalized
                    audio_np = audio_np.astype(np.float32)
                    if np.abs(audio_np).max() > 1.0:
                        audio_np = audio_np / np.abs(audio_np).max()

                    # Validate audio duration (recomm 5-60 seconds)
                    if sample_rate is not None:
                        duration = len(audio_np) / sample_rate
                        if duration < 1.0:
                            self.logger.warning(
                                f"Audio duration ({duration:.1f}s) is very short. "
                                "Recommend at least 5 seconds for reliable embeddings."
                            )
                        elif duration < 5.0:
                            self.logger.info(
                                f"Audio duration ({duration:.1f}s) is short. "
                                "Optimal duration is 10-60 seconds."
                            )
                        elif duration > 60.0:
                            self.logger.info(
                                f"Audio duration ({duration:.1f}s) is long. "
                                "Using first 60 seconds recommended for consistency."
                            )

                    # Preprocess with resemblyzer's preprocess_wav for consistency
                    if sample_rate is None:
                        raise SpeakerEncodingError("sample_rate is required for torch.Tensor input")
                    preprocessed_wav = preprocess_wav(audio_np, source_sr=sample_rate)

                    # Extract embedding
                    embedding = self.encoder.embed_utterance(preprocessed_wav)

                # Handle numpy array input
                elif isinstance(audio, np.ndarray):
                    # Ensure mono
                    if audio.ndim > 1:
                        audio = audio.mean(axis=0)

                    # Ensure float32 and normalized
                    audio = audio.astype(np.float32)
                    if np.abs(audio).max() > 1.0:
                        audio = audio / np.abs(audio).max()

                    # Validate audio duration
                    if sample_rate is not None:
                        duration = len(audio) / sample_rate
                        if duration < 1.0:
                            self.logger.warning(
                                f"Audio duration ({duration:.1f}s) is very short. "
                                "Recommend at least 5 seconds for reliable embeddings."
                            )
                        elif duration < 5.0:
                            self.logger.info(
                                f"Audio duration ({duration:.1f}s) is short. "
                                "Optimal duration is 10-60 seconds."
                            )

                    # Preprocess with resemblyzer's preprocess_wav for consistency
                    if sample_rate is None:
                        raise SpeakerEncodingError("sample_rate is required for np.ndarray input")
                    preprocessed_wav = preprocess_wav(audio, source_sr=sample_rate)

                    # Extract embedding
                    embedding = self.encoder.embed_utterance(preprocessed_wav)

                else:
                    raise SpeakerEncodingError(
                        f"Unsupported audio type: {type(audio)}. "
                        "Expected str, np.ndarray, or torch.Tensor"
                    )

                # Validate embedding
                if embedding.shape != (256,):
                    raise SpeakerEncodingError(
                        f"Invalid embedding shape: {embedding.shape}. Expected (256,)"
                    )

                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    raise SpeakerEncodingError("Embedding contains NaN or Inf values")

                # Ensure embedding is normalized (L2 norm = 1)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                return embedding

            except SpeakerEncodingError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to extract speaker embedding: {e}")
                raise SpeakerEncodingError(f"Embedding extraction failed: {e}")

    def extract_embeddings_batch(
        self,
        audio_list: List[Union[np.ndarray, str]],
        sample_rate: Optional[int] = None,
        on_error: str = 'zero'
    ) -> List[Optional[np.ndarray]]:
        """Extract speaker embeddings from multiple audio files/arrays

        Args:
            audio_list: List of audio arrays or file paths
            sample_rate: Sample rate for array inputs
            on_error: Error handling strategy:
                      - 'zero': Return zero vector (default, backward compatible)
                      - 'none': Return None
                      - 'raise': Raise exception

        Returns:
            List of 256-dimensional embeddings (or None if on_error='none')

        Example:
            >>> embeddings = encoder.extract_embeddings_batch([
            ...     'voice1.wav',
            ...     'voice2.wav',
            ...     'voice3.wav'
            ... ], on_error='none')
            >>> print(len([e for e in embeddings if e is not None]))  # 3
        """
        embeddings = []

        for i, audio in enumerate(audio_list):
            try:
                embedding = self.extract_embedding(audio, sample_rate)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Failed to extract embedding for item {i}: {e}")

                if on_error == 'raise':
                    raise
                elif on_error == 'none':
                    embeddings.append(None)
                else:  # 'zero' (default)
                    embeddings.append(np.zeros(256, dtype=np.float32))

        if len(audio_list) > 5:
            successful = len([e for e in embeddings if e is not None and not np.allclose(e, 0)])
            self.logger.info(f"Batch extracted {successful}/{len(embeddings)} embeddings")

        return embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two speaker embeddings

        Cosine similarity ranges from -1 (opposite) to 1 (identical).
        For speaker embeddings:
        - 0.7-0.95: Same speaker (typical range)
        - 0.5-0.7: Similar voices or same speaker with different conditions
        - <0.5: Different speakers

        Args:
            embedding1: First speaker embedding (256,)
            embedding2: Second speaker embedding (256,)

        Returns:
            Cosine similarity score (float in [-1, 1])

        Raises:
            SpeakerEncodingError: If embeddings have invalid shape

        Example:
            >>> emb1 = encoder.extract_embedding('voice1.wav')
            >>> emb2 = encoder.extract_embedding('voice2.wav')
            >>> similarity = encoder.compute_similarity(emb1, emb2)
            >>> print(f"Similarity: {similarity:.3f}")
        """
        # Validate shapes
        if embedding1.shape != (256,):
            raise SpeakerEncodingError(
                f"Invalid embedding1 shape: {embedding1.shape}. Expected (256,)"
            )
        if embedding2.shape != (256,):
            raise SpeakerEncodingError(
                f"Invalid embedding2 shape: {embedding2.shape}. Expected (256,)"
            )

        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            self.logger.warning("Zero-norm embedding detected")
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Clamp to [-1, 1] to handle numerical errors
        similarity = np.clip(similarity, -1.0, 1.0)

        self.logger.debug(f"Computed similarity: {similarity:.3f}")

        return float(similarity)

    def is_same_speaker(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.75
    ) -> bool:
        """Determine if two embeddings belong to the same speaker

        Uses cosine similarity with a configurable threshold.
        Default threshold of 0.75 works well for most cases.

        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            threshold: Similarity threshold for same speaker (default: 0.75)

        Returns:
            True if embeddings likely belong to same speaker

        Example:
            >>> emb1 = encoder.extract_embedding('voice1.wav')
            >>> emb2 = encoder.extract_embedding('voice2.wav')
            >>> is_same = encoder.is_same_speaker(emb1, emb2, threshold=0.75)
            >>> print(f"Same speaker: {is_same}")
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        self.logger.debug(
            f"Similarity: {similarity:.3f}, Threshold: {threshold:.3f}, "
            f"Same speaker: {similarity >= threshold}"
        )
        return similarity >= threshold

    def get_embedding_stats(self, embedding: np.ndarray) -> Dict[str, float]:
        """Compute statistics for embedding vector

        Useful for debugging and validation.

        Args:
            embedding: Speaker embedding (256,)

        Returns:
            Dictionary with statistics:
                - mean: Mean value
                - std: Standard deviation
                - min: Minimum value
                - max: Maximum value
                - norm: L2 norm

        Example:
            >>> embedding = encoder.extract_embedding('voice.wav')
            >>> stats = encoder.get_embedding_stats(embedding)
            >>> print(f"Norm: {stats['norm']:.3f}")
        """
        if embedding.shape != (256,):
            raise SpeakerEncodingError(
                f"Invalid embedding shape: {embedding.shape}. Expected (256,)"
            )

        return {
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'min': float(np.min(embedding)),
            'max': float(np.max(embedding)),
            'norm': float(np.linalg.norm(embedding))
        }
