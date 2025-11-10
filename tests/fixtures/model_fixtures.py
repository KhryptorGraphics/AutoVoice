"""Model-related pytest fixtures for AutoVoice testing.

Provides mock models, trained checkpoints, and model factories for testing
voice conversion models, encoders, decoders, and vocoders.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, MagicMock


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_voice_model():
    """Mock VoiceModel for testing without loading actual weights.

    Provides a mock model with predictable outputs for testing pipelines
    without requiring trained model weights.

    Examples:
        model = mock_voice_model
        output = model.forward(input_tensor)  # Returns zeros of correct shape
    """
    class MockVoiceModel(nn.Module):
        def __init__(self, hidden_size=256, output_size=80):
            super().__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.device = 'cpu'

        def forward(self, x, speaker_embedding=None):
            """Mock forward pass."""
            batch_size = x.shape[0]
            seq_len = x.shape[1] if x.ndim > 2 else 100

            # Return tensor of expected shape
            return torch.zeros(batch_size, seq_len, self.output_size)

        def to(self, device):
            """Mock device transfer."""
            self.device = str(device)
            return self

        def eval(self):
            """Mock eval mode."""
            return self

        def train(self, mode=True):
            """Mock train mode."""
            return self

    return MockVoiceModel()


@pytest.fixture
def mock_encoder():
    """Mock audio encoder for testing feature extraction.

    Returns a mock encoder that produces deterministic embeddings
    from input audio features.

    Examples:
        encoder = mock_encoder
        embedding = encoder.encode(mel_spec)  # Returns fixed-size embedding
    """
    class MockEncoder(nn.Module):
        def __init__(self, input_size=80, embedding_size=256):
            super().__init__()
            self.input_size = input_size
            self.embedding_size = embedding_size
            self.num_calls = 0

        def forward(self, x):
            """Mock encoding."""
            self.num_calls += 1
            batch_size = x.shape[0]
            return torch.randn(batch_size, self.embedding_size)

        def encode(self, x):
            """Alias for forward."""
            return self.forward(x)

    return MockEncoder()


@pytest.fixture
def mock_decoder():
    """Mock audio decoder for testing waveform synthesis.

    Returns a mock decoder that produces audio from embeddings.

    Examples:
        decoder = mock_decoder
        audio = decoder.decode(embedding)  # Returns synthetic waveform
    """
    class MockDecoder(nn.Module):
        def __init__(self, embedding_size=256, sample_rate=22050):
            super().__init__()
            self.embedding_size = embedding_size
            self.sample_rate = sample_rate
            self.num_calls = 0

        def forward(self, embedding, length=None):
            """Mock decoding."""
            self.num_calls += 1
            batch_size = embedding.shape[0]

            # Generate 1 second of audio if length not specified
            audio_length = length if length is not None else self.sample_rate

            return torch.randn(batch_size, audio_length)

        def decode(self, embedding, length=None):
            """Alias for forward."""
            return self.forward(embedding, length)

    return MockDecoder()


@pytest.fixture
def mock_vocoder():
    """Mock vocoder for testing mel-to-waveform conversion.

    Returns a mock vocoder that converts mel-spectrograms to audio.

    Examples:
        vocoder = mock_vocoder
        audio = vocoder.infer(mel_spec)  # Returns waveform
    """
    class MockVocoder(nn.Module):
        def __init__(self, n_mels=80, hop_length=256):
            super().__init__()
            self.n_mels = n_mels
            self.hop_length = hop_length
            self.num_calls = 0

        def forward(self, mel):
            """Mock vocoding."""
            self.num_calls += 1
            batch_size, n_mels, n_frames = mel.shape
            audio_length = n_frames * self.hop_length

            return torch.randn(batch_size, audio_length)

        def infer(self, mel):
            """Inference method."""
            return self.forward(mel)

    return MockVocoder()


# ============================================================================
# Checkpoint Fixtures
# ============================================================================

@pytest.fixture
def trained_model_checkpoint(tmp_path: Path):
    """Create a mock trained model checkpoint for testing.

    Generates a complete checkpoint file with model state, optimizer state,
    training metadata, and configuration.

    Examples:
        ckpt_path = trained_model_checkpoint
        model.load_checkpoint(ckpt_path)
    """
    def factory(
        model_type: str = 'voice_transformer',
        epoch: int = 100,
        include_optimizer: bool = True,
        include_scheduler: bool = False,
        **kwargs
    ) -> Path:
        """Create checkpoint file.

        Args:
            model_type: Type of model checkpoint
            epoch: Training epoch number
            include_optimizer: Include optimizer state
            include_scheduler: Include LR scheduler state
            **kwargs: Additional checkpoint data

        Returns:
            Path to checkpoint file
        """
        # Create mock model state dict
        model_state = {
            f'layer_{i}.weight': torch.randn(256, 256)
            for i in range(4)
        }

        checkpoint = {
            'model_state_dict': model_state,
            'epoch': epoch,
            'global_step': epoch * 1000,
            'config': {
                'model_type': model_type,
                'hidden_size': 256,
                'num_layers': 4,
                'sample_rate': 22050,
                **kwargs.get('config', {})
            },
            'metrics': {
                'train_loss': 0.5,
                'val_loss': 0.6,
                'best_val_loss': 0.55,
            }
        }

        if include_optimizer:
            checkpoint['optimizer_state_dict'] = {
                'state': {},
                'param_groups': [{
                    'lr': 0.001,
                    'weight_decay': 0.0001,
                }]
            }

        if include_scheduler:
            checkpoint['scheduler_state_dict'] = {
                'last_epoch': epoch,
                '_step_count': epoch + 1,
            }

        # Add any additional kwargs
        checkpoint.update(kwargs)

        # Save checkpoint
        ckpt_path = tmp_path / f'{model_type}_epoch_{epoch}.pt'
        torch.save(checkpoint, ckpt_path)

        return ckpt_path

    return factory


@pytest.fixture
def model_config_factory():
    """Factory for generating model configurations.

    Returns a callable that generates configuration dicts for different
    model types with sensible defaults.

    Examples:
        config = model_config_factory('transformer', hidden_size=512)
        config = model_config_factory('vocoder', n_mels=128)
    """
    def factory(model_type: str = 'transformer', **overrides) -> Dict[str, Any]:
        """Generate model configuration.

        Args:
            model_type: Type of model
            **overrides: Override default config values

        Returns:
            Configuration dictionary
        """
        configs = {
            'transformer': {
                'hidden_size': 256,
                'num_layers': 4,
                'num_heads': 4,
                'dropout': 0.1,
                'max_seq_length': 512,
            },
            'vocoder': {
                'n_mels': 80,
                'hop_length': 256,
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4],
                'resblock_kernel_sizes': [3, 7, 11],
            },
            'encoder': {
                'input_size': 80,
                'hidden_size': 256,
                'num_layers': 3,
                'bidirectional': True,
            },
            'decoder': {
                'embedding_size': 256,
                'hidden_size': 512,
                'num_layers': 3,
                'output_size': 80,
            }
        }

        if model_type not in configs:
            raise ValueError(f"Unknown model type: {model_type}")

        config = configs[model_type].copy()
        config.update(overrides)

        return config

    return factory


# ============================================================================
# Model Testing Utilities
# ============================================================================

@pytest.fixture
def model_forward_tester():
    """Utility for testing model forward passes.

    Provides methods to test:
    - Input/output shapes
    - Gradient flow
    - Memory consumption
    - Output distribution

    Examples:
        tester = model_forward_tester
        tester.test_forward(model, input_tensor, expected_shape=(16, 100, 80))
    """
    class ModelForwardTester:
        def test_forward(
            self,
            model: nn.Module,
            input_tensor: torch.Tensor,
            expected_shape: Optional[tuple] = None,
            check_gradients: bool = True,
            check_nan: bool = True
        ) -> Dict[str, Any]:
            """Test model forward pass.

            Args:
                model: Model to test
                input_tensor: Input tensor
                expected_shape: Expected output shape
                check_gradients: Check gradient flow
                check_nan: Check for NaN values

            Returns:
                Dict with test results
            """
            results = {}

            # Forward pass
            model.train()
            output = model(input_tensor)
            results['output_shape'] = output.shape
            results['output_dtype'] = output.dtype

            # Check shape
            if expected_shape is not None:
                results['shape_correct'] = output.shape == expected_shape

            # Check for NaN/Inf
            if check_nan:
                results['has_nan'] = torch.isnan(output).any().item()
                results['has_inf'] = torch.isinf(output).any().item()

            # Check gradients
            if check_gradients:
                try:
                    loss = output.mean()
                    loss.backward()

                    has_grads = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            has_grads.append(param.grad is not None)

                    results['gradient_flow'] = all(has_grads)
                    results['num_params_with_grad'] = sum(has_grads)
                except Exception as e:
                    results['gradient_error'] = str(e)

            return results

        def test_batch_processing(
            self,
            model: nn.Module,
            input_shape: tuple,
            batch_sizes: List[int] = [1, 4, 16, 32]
        ) -> Dict[int, float]:
            """Test model with different batch sizes.

            Args:
                model: Model to test
                input_shape: Input shape (without batch dimension)
                batch_sizes: List of batch sizes to test

            Returns:
                Dict mapping batch_size -> inference_time
            """
            import time

            results = {}
            model.eval()

            with torch.no_grad():
                for batch_size in batch_sizes:
                    input_tensor = torch.randn(batch_size, *input_shape)

                    # Warmup
                    _ = model(input_tensor)

                    # Timed run
                    start = time.perf_counter()
                    _ = model(input_tensor)
                    elapsed = time.perf_counter() - start

                    results[batch_size] = elapsed

            return results

    return ModelForwardTester()


@pytest.fixture
def parameter_counter():
    """Utility for counting model parameters.

    Examples:
        counter = parameter_counter
        total, trainable = counter.count(model)
    """
    def count(model: nn.Module) -> Dict[str, int]:
        """Count model parameters.

        Args:
            model: PyTorch model

        Returns:
            Dict with parameter counts
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'layers': len(list(model.parameters()))
        }

    return count


__all__ = [
    'mock_voice_model',
    'mock_encoder',
    'mock_decoder',
    'mock_vocoder',
    'trained_model_checkpoint',
    'model_config_factory',
    'model_forward_tester',
    'parameter_counter',
]
