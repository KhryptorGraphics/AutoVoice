"""
Test graceful fallbacks for speaker and pitch loss classes
"""

import pytest
import torch
import logging
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_pitch_loss_with_unavailable_components():
    """Test PitchConsistencyLoss when components are unavailable"""
    # Mock VC_COMPONENTS_AVAILABLE as False
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import PitchConsistencyLoss

        # Create loss instance
        pitch_loss = PitchConsistencyLoss(device='cpu')

        # Verify internal flags
        assert not pitch_loss._extractor_available
        assert pitch_loss.pitch_extractor is None

        # Test forward pass returns zero without exception
        pred_audio = torch.randn(2, 16000)
        source_f0 = torch.randn(2, 100)

        loss = pitch_loss(pred_audio, source_f0, sample_rate=16000)

        assert loss.item() == 0.0
        assert loss.device.type == 'cpu'


def test_speaker_loss_with_unavailable_components():
    """Test SpeakerSimilarityLoss when components are unavailable"""
    # Mock VC_COMPONENTS_AVAILABLE as False
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import SpeakerSimilarityLoss

        # Create loss instance
        speaker_loss = SpeakerSimilarityLoss(device='cpu')

        # Verify internal flags
        assert not speaker_loss._encoder_available
        assert speaker_loss.speaker_encoder is None

        # Test forward pass returns zero without exception
        pred_audio = torch.randn(2, 16000)
        target_emb = torch.randn(2, 256)

        loss = speaker_loss(pred_audio, target_emb, sample_rate=16000)

        assert loss.item() == 0.0
        assert loss.device.type == 'cpu'


def test_pitch_loss_with_initialization_failure():
    """Test PitchConsistencyLoss when initialization fails"""
    # Mock initialization to raise exception
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', True):
        with patch('auto_voice.training.trainer.SingingPitchExtractor') as mock_extractor:
            mock_extractor.side_effect = RuntimeError("Initialization failed")

            from auto_voice.training.trainer import PitchConsistencyLoss

            # Create loss instance (should handle exception)
            pitch_loss = PitchConsistencyLoss(device='cpu')

            # Verify fallback state
            assert not pitch_loss._extractor_available
            assert pitch_loss.pitch_extractor is None

            # Test forward pass returns zero
            pred_audio = torch.randn(2, 16000)
            source_f0 = torch.randn(2, 100)

            loss = pitch_loss(pred_audio, source_f0)
            assert loss.item() == 0.0


def test_speaker_loss_with_initialization_failure():
    """Test SpeakerSimilarityLoss when initialization fails"""
    # Mock initialization to raise exception
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', True):
        with patch('auto_voice.training.trainer.SpeakerEncoder') as mock_encoder:
            mock_encoder.side_effect = RuntimeError("Initialization failed")

            from auto_voice.training.trainer import SpeakerSimilarityLoss

            # Create loss instance (should handle exception)
            speaker_loss = SpeakerSimilarityLoss(device='cpu')

            # Verify fallback state
            assert not speaker_loss._encoder_available
            assert speaker_loss.speaker_encoder is None

            # Test forward pass returns zero
            pred_audio = torch.randn(2, 16000)
            target_emb = torch.randn(2, 256)

            loss = speaker_loss(pred_audio, target_emb)
            assert loss.item() == 0.0


def test_pitch_loss_warning_logged_once_init():
    """Test that warning is logged once during initialization"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        with patch('auto_voice.training.trainer.logger') as mock_logger:
            from auto_voice.training.trainer import PitchConsistencyLoss

            # Create loss instance
            pitch_loss = PitchConsistencyLoss(device='cpu')

            # Verify warning was logged during init
            assert mock_logger.warning.call_count == 1
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "SingingPitchExtractor not available" in warning_msg or "disabled" in warning_msg


def test_speaker_loss_warning_logged_once_init():
    """Test that warning is logged once during initialization"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        with patch('auto_voice.training.trainer.logger') as mock_logger:
            from auto_voice.training.trainer import SpeakerSimilarityLoss

            # Create loss instance
            speaker_loss = SpeakerSimilarityLoss(device='cpu')

            # Verify warning was logged during init
            assert mock_logger.warning.call_count == 1
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "SpeakerEncoder not available" in warning_msg or "disabled" in warning_msg


def test_pitch_loss_warning_logged_once_forward():
    """Test that warning is logged once during forward pass"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import PitchConsistencyLoss

        pitch_loss = PitchConsistencyLoss(device='cpu')

        # Reset warned flag to test forward warning
        pitch_loss._warned = False

        with patch('auto_voice.training.trainer.logger') as mock_logger:
            pred_audio = torch.randn(2, 16000)
            source_f0 = torch.randn(2, 100)

            # First call - should log warning
            loss1 = pitch_loss(pred_audio, source_f0)
            assert loss1.item() == 0.0
            assert mock_logger.warning.call_count == 1

            # Second call - should not log warning again
            loss2 = pitch_loss(pred_audio, source_f0)
            assert loss2.item() == 0.0
            assert mock_logger.warning.call_count == 1  # Still 1, not 2


def test_speaker_loss_warning_logged_once_forward():
    """Test that warning is logged once during forward pass"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import SpeakerSimilarityLoss

        speaker_loss = SpeakerSimilarityLoss(device='cpu')

        # Reset warned flag to test forward warning
        speaker_loss._warned = False

        with patch('auto_voice.training.trainer.logger') as mock_logger:
            pred_audio = torch.randn(2, 16000)
            target_emb = torch.randn(2, 256)

            # First call - should log warning
            loss1 = speaker_loss(pred_audio, target_emb)
            assert loss1.item() == 0.0
            assert mock_logger.warning.call_count == 1

            # Second call - should not log warning again
            loss2 = speaker_loss(pred_audio, target_emb)
            assert loss2.item() == 0.0
            assert mock_logger.warning.call_count == 1  # Still 1, not 2


def test_pitch_loss_multiple_forward_calls_no_repeated_warnings():
    """Test multiple forward calls don't repeat warnings"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import PitchConsistencyLoss

        pitch_loss = PitchConsistencyLoss(device='cpu')
        pitch_loss._warned = False

        with patch('auto_voice.training.trainer.logger') as mock_logger:
            pred_audio = torch.randn(2, 16000)
            source_f0 = torch.randn(2, 100)

            # Call forward 5 times
            for _ in range(5):
                loss = pitch_loss(pred_audio, source_f0)
                assert loss.item() == 0.0

            # Should only log warning once
            assert mock_logger.warning.call_count == 1


def test_speaker_loss_multiple_forward_calls_no_repeated_warnings():
    """Test multiple forward calls don't repeat warnings"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import SpeakerSimilarityLoss

        speaker_loss = SpeakerSimilarityLoss(device='cpu')
        speaker_loss._warned = False

        with patch('auto_voice.training.trainer.logger') as mock_logger:
            pred_audio = torch.randn(2, 16000)
            target_emb = torch.randn(2, 256)

            # Call forward 5 times
            for _ in range(5):
                loss = speaker_loss(pred_audio, target_emb)
                assert loss.item() == 0.0

            # Should only log warning once
            assert mock_logger.warning.call_count == 1


def test_pitch_loss_returns_tensor_on_correct_device():
    """Test that loss returns tensor on correct device"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import PitchConsistencyLoss

        pitch_loss = PitchConsistencyLoss(device='cpu')

        # Test CPU
        pred_audio_cpu = torch.randn(2, 16000)
        source_f0_cpu = torch.randn(2, 100)
        loss_cpu = pitch_loss(pred_audio_cpu, source_f0_cpu)
        assert loss_cpu.device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            pred_audio_cuda = torch.randn(2, 16000).cuda()
            source_f0_cuda = torch.randn(2, 100).cuda()
            loss_cuda = pitch_loss(pred_audio_cuda, source_f0_cuda)
            assert loss_cuda.device.type == 'cuda'


def test_speaker_loss_returns_tensor_on_correct_device():
    """Test that loss returns tensor on correct device"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import SpeakerSimilarityLoss

        speaker_loss = SpeakerSimilarityLoss(device='cpu')

        # Test CPU
        pred_audio_cpu = torch.randn(2, 16000)
        target_emb_cpu = torch.randn(2, 256)
        loss_cpu = speaker_loss(pred_audio_cpu, target_emb_cpu)
        assert loss_cpu.device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            pred_audio_cuda = torch.randn(2, 16000).cuda()
            target_emb_cuda = torch.randn(2, 256).cuda()
            loss_cuda = speaker_loss(pred_audio_cuda, target_emb_cuda)
            assert loss_cuda.device.type == 'cuda'


def test_pitch_loss_no_exception_on_forward():
    """Test that no exception is raised during forward pass"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import PitchConsistencyLoss

        pitch_loss = PitchConsistencyLoss(device='cpu')

        # Various tensor sizes
        test_cases = [
            (torch.randn(1, 8000), torch.randn(1, 50)),
            (torch.randn(4, 16000), torch.randn(4, 100)),
            (torch.randn(8, 32000), torch.randn(8, 200)),
        ]

        for pred_audio, source_f0 in test_cases:
            loss = pitch_loss(pred_audio, source_f0)
            assert loss.item() == 0.0
            assert isinstance(loss, torch.Tensor)


def test_speaker_loss_no_exception_on_forward():
    """Test that no exception is raised during forward pass"""
    with patch('auto_voice.training.trainer.VC_COMPONENTS_AVAILABLE', False):
        from auto_voice.training.trainer import SpeakerSimilarityLoss

        speaker_loss = SpeakerSimilarityLoss(device='cpu')

        # Various tensor sizes
        test_cases = [
            (torch.randn(1, 8000), torch.randn(1, 256)),
            (torch.randn(4, 16000), torch.randn(4, 256)),
            (torch.randn(8, 32000), torch.randn(8, 256)),
        ]

        for pred_audio, target_emb in test_cases:
            loss = speaker_loss(pred_audio, target_emb)
            assert loss.item() == 0.0
            assert isinstance(loss, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
