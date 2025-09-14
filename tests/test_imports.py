"""Test basic imports and functionality."""
import pytest
import torch


def test_pytorch_available():
    """Test that PyTorch is available."""
    assert torch.__version__ is not None


def test_cuda_available():
    """Test CUDA availability."""
    # This test will pass even if CUDA is not available
    # It just checks that the function exists
    cuda_available = torch.cuda.is_available()
    assert isinstance(cuda_available, bool)


def test_package_imports():
    """Test that main package components can be imported."""
    try:
        from auto_voice.web.app import create_app
        from auto_voice.utils.config_loader import load_config
        assert create_app is not None
        assert load_config is not None
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_model_imports():
    """Test that model classes can be imported."""
    try:
        from auto_voice.models import VoiceTransformer, HiFiGANGenerator
        assert VoiceTransformer is not None
        assert HiFiGANGenerator is not None
    except ImportError as e:
        pytest.fail(f"Failed to import model modules: {e}")


def test_audio_processor_imports():
    """Test audio processor imports."""
    try:
        from auto_voice.audio import AudioProcessor, GPUAudioProcessor
        assert AudioProcessor is not None
        assert GPUAudioProcessor is not None
    except ImportError as e:
        pytest.fail(f"Failed to import audio modules: {e}")


def test_training_imports():
    """Test training module imports."""
    try:
        from auto_voice.training import VoiceTrainer, VoiceLoss
        assert VoiceTrainer is not None
        assert VoiceLoss is not None
    except ImportError as e:
        pytest.fail(f"Failed to import training modules: {e}")


def test_inference_imports():
    """Test inference module imports."""
    try:
        from auto_voice.inference import VoiceInferenceEngine
        assert VoiceInferenceEngine is not None
    except ImportError as e:
        pytest.fail(f"Failed to import inference modules: {e}")


def test_config_loading():
    """Test config loading functionality."""
    from auto_voice.utils.config_loader import load_config

    # Test loading default config
    config = load_config()
    assert isinstance(config, dict)
    assert 'server' in config
    assert 'host' in config['server']
    assert 'port' in config['server']


def test_app_creation():
    """Test Flask app creation."""
    from auto_voice.web.app import create_app

    app, socketio = create_app()
    assert app is not None
    assert socketio is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])