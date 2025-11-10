"""
Tests for model registry infrastructure.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from auto_voice.models import (
    ModelRegistry,
    ModelConfig,
    ModelType,
    ModelLoader,
    HuBERTModel,
    HiFiGANModel,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating a model config."""
        config = ModelConfig(
            name='test_model',
            model_type=ModelType.HUBERT,
            version='1.0.0',
            url='https://example.com/model.pt',
            requires_gpu=False,
            min_memory_gb=2.0
        )

        assert config.name == 'test_model'
        assert config.model_type == ModelType.HUBERT
        assert config.version == '1.0.0'
        assert config.requires_gpu is False

    def test_model_config_serialization(self):
        """Test config serialization to/from dict."""
        config = ModelConfig(
            name='test_model',
            model_type=ModelType.HIFIGAN,
            version='1.0.0',
            metadata={'test': 'data'}
        )

        # To dict
        config_dict = config.to_dict()
        assert config_dict['model_type'] == 'hifigan'
        assert config_dict['metadata']['test'] == 'data'

        # From dict
        restored = ModelConfig.from_dict(config_dict)
        assert restored.name == config.name
        assert restored.model_type == ModelType.HIFIGAN


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def registry(self, temp_dir):
        """Create a model registry for testing."""
        config_path = temp_dir / 'models.yaml'
        return ModelRegistry(
            model_dir=str(temp_dir / 'models'),
            config_path=str(config_path),
            use_mock=True
        )

    def test_registry_initialization(self, registry):
        """Test registry initializes correctly."""
        assert registry.model_dir.exists()
        assert len(registry._configs) > 0

    def test_list_models(self, registry):
        """Test listing available models."""
        models = registry.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_model_config(self, registry):
        """Test getting model configuration."""
        models = registry.list_models()
        config = registry.get_config(models[0])

        assert isinstance(config, ModelConfig)
        assert config.name == models[0]

    def test_get_model_path(self, registry):
        """Test getting model path."""
        models = registry.list_models()
        path = registry.get_model_path(models[0])

        assert isinstance(path, Path)

    def test_load_hubert_mock(self, registry):
        """Test loading HuBERT in mock mode."""
        model = registry.load_hubert()

        assert model is not None
        assert isinstance(model, HuBERTModel)

    def test_load_hifigan_mock(self, registry):
        """Test loading HiFi-GAN in mock mode."""
        model = registry.load_hifigan()

        assert model is not None
        assert isinstance(model, HiFiGANModel)

    def test_load_speaker_encoder_mock(self, registry):
        """Test loading speaker encoder in mock mode."""
        model = registry.load_speaker_encoder()

        assert model is not None

    def test_model_caching(self, registry):
        """Test that models are cached after first load."""
        model1 = registry.load_hubert()
        model2 = registry.load_hubert()

        # Should return the same instance
        assert model1 is model2

    def test_warmup_models(self, registry):
        """Test model warmup functionality."""
        # Should not raise any errors
        registry.warmup_models()

        # Check that models are loaded
        assert len(registry._models) > 0

    def test_clear_cache(self, registry):
        """Test clearing model cache."""
        registry.load_hubert()
        assert len(registry._models) > 0

        registry.clear_cache()
        assert len(registry._models) == 0


class TestHuBERTModel:
    """Test HuBERT model wrapper."""

    def test_mock_model_creation(self):
        """Test creating mock HuBERT model."""
        model = HuBERTModel(use_mock=True)

        assert model is not None
        assert model.use_mock is True

    def test_mock_feature_extraction(self):
        """Test mock feature extraction."""
        model = HuBERTModel(use_mock=True)

        # Generate random audio
        audio = np.random.randn(16000).astype(np.float32)

        # Extract features
        features = model.extract_features(audio)

        assert features is not None
        assert features.ndim == 3  # (batch, time, features)
        assert features.shape[2] == 768  # HuBERT base hidden size

    def test_model_callable(self):
        """Test model is callable."""
        model = HuBERTModel(use_mock=True)
        audio = np.random.randn(16000).astype(np.float32)

        features = model(audio)

        assert features is not None


class TestHiFiGANModel:
    """Test HiFi-GAN model wrapper."""

    def test_mock_model_creation(self):
        """Test creating mock HiFi-GAN model."""
        model = HiFiGANModel(use_mock=True)

        assert model is not None
        assert model.use_mock is True

    def test_mock_synthesis(self):
        """Test mock audio synthesis."""
        model = HiFiGANModel(use_mock=True)

        # Generate random mel spectrogram
        mel = np.random.randn(80, 100).astype(np.float32)

        # Synthesize audio
        audio = model.synthesize(mel)

        assert audio is not None
        assert audio.ndim == 1  # Waveform
        assert len(audio) > 0

    def test_model_callable(self):
        """Test model is callable."""
        model = HiFiGANModel(use_mock=True)
        mel = np.random.randn(80, 100).astype(np.float32)

        audio = model(mel)

        assert audio is not None


class TestSpeakerEncoder:
    """Test speaker encoder model wrapper."""

    def test_mock_model_creation(self):
        """Test creating mock speaker encoder."""
        from auto_voice.models.speaker_encoder import SpeakerEncoderModel

        model = SpeakerEncoderModel(use_mock=True)

        assert model is not None
        assert model.use_mock is True

    def test_mock_encoding(self):
        """Test mock speaker encoding."""
        from auto_voice.models.speaker_encoder import SpeakerEncoderModel

        model = SpeakerEncoderModel(use_mock=True)

        # Generate random audio
        audio = np.random.randn(16000).astype(np.float32)

        # Extract embedding
        embedding = model.encode(audio)

        assert embedding is not None
        assert embedding.shape == (256,)  # Resemblyzer embedding size
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5  # Normalized

    def test_similarity_computation(self):
        """Test computing similarity between embeddings."""
        from auto_voice.models.speaker_encoder import SpeakerEncoderModel

        model = SpeakerEncoderModel(use_mock=True)

        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = np.random.randn(16000).astype(np.float32)

        emb1 = model.encode(audio1)
        emb2 = model.encode(audio2)

        similarity = model.compute_similarity(emb1, emb2)

        assert 0.0 <= similarity <= 1.0

    def test_deterministic_encoding(self):
        """Test that same audio gives same embedding."""
        from auto_voice.models.speaker_encoder import SpeakerEncoderModel

        model = SpeakerEncoderModel(use_mock=True)

        audio = np.random.randn(16000).astype(np.float32)

        emb1 = model.encode(audio)
        emb2 = model.encode(audio)

        # Should be deterministic
        np.testing.assert_array_equal(emb1, emb2)


class TestModelLoader:
    """Test model loader functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def loader(self, temp_dir):
        """Create a model loader for testing."""
        return ModelLoader(temp_dir)

    def test_loader_creation(self, loader, temp_dir):
        """Test creating model loader."""
        assert loader.model_dir == temp_dir
        assert temp_dir.exists()

    def test_load_models_with_config(self, loader):
        """Test loading models with configuration."""
        config = ModelConfig(
            name='test_hubert',
            model_type=ModelType.HUBERT,
            version='1.0.0',
            local_path=None  # Will trigger mock fallback
        )

        # Should not raise error even without real model
        # (will fall back to mock)
        model = loader.load_hubert(config)

        assert model is not None
