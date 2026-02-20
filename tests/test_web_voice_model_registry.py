"""Tests for voice model registry module - Target 70% coverage.

Tests for VoiceModelRegistry and speaker embedding extraction.
"""
import pytest
import torch
import numpy as np
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestExtractSpeakerEmbedding:
    """Test extract_speaker_embedding function."""

    @pytest.mark.smoke
    def test_extract_embedding_basic(self):
        """Test basic embedding extraction."""
        from auto_voice.web.voice_model_registry import extract_speaker_embedding

        # Create test audio (1 second at 24kHz)
        audio = torch.randn(24000) * 0.1

        embedding = extract_speaker_embedding(audio, sample_rate=24000)

        # Should return 256-dim embedding
        assert embedding.shape == (256,)
        assert embedding.dtype == torch.float32

    def test_extract_embedding_stereo_input(self):
        """Test embedding extraction from stereo audio."""
        from auto_voice.web.voice_model_registry import extract_speaker_embedding

        # Stereo audio (2 channels)
        audio = torch.randn(2, 24000) * 0.1

        embedding = extract_speaker_embedding(audio, sample_rate=24000)

        # Should handle stereo and return 256-dim
        assert embedding.shape == (256,)

    def test_extract_embedding_l2_normalized(self):
        """Test that embedding is L2 normalized."""
        from auto_voice.web.voice_model_registry import extract_speaker_embedding

        audio = torch.randn(24000) * 0.1

        embedding = extract_speaker_embedding(audio, sample_rate=24000)

        # Check L2 norm is approximately 1
        norm = embedding.norm().item()
        assert abs(norm - 1.0) < 0.01

    def test_extract_embedding_different_sample_rates(self):
        """Test embedding extraction at different sample rates."""
        from auto_voice.web.voice_model_registry import extract_speaker_embedding

        # Test at different sample rates
        for sr in [16000, 22050, 44100, 48000]:
            audio = torch.randn(sr * 2) * 0.1  # 2 seconds

            embedding = extract_speaker_embedding(audio, sample_rate=sr)

            assert embedding.shape == (256,)

    def test_extract_embedding_short_audio(self):
        """Test embedding extraction from short audio."""
        from auto_voice.web.voice_model_registry import extract_speaker_embedding

        # Very short audio (100ms)
        audio = torch.randn(2400) * 0.1

        embedding = extract_speaker_embedding(audio, sample_rate=24000)

        # Should still work
        assert embedding.shape == (256,)

    def test_extract_embedding_int_dtype_conversion(self):
        """Test embedding extraction handles int dtype."""
        from auto_voice.web.voice_model_registry import extract_speaker_embedding

        # Int audio that needs conversion
        audio = torch.randint(-32768, 32767, (24000,), dtype=torch.int16)

        embedding = extract_speaker_embedding(audio.float() / 32768.0, sample_rate=24000)

        assert embedding.shape == (256,)


class TestVoiceModelRegistry:
    """Test VoiceModelRegistry class."""

    @pytest.fixture
    def empty_registry(self, tmp_path):
        """Create registry with empty models directory."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        return VoiceModelRegistry(models_dir=str(models_dir))

    @pytest.fixture
    def registry_with_models(self, tmp_path):
        """Create registry with some pretrained models."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        # Create fake model files
        for name in ['artist_one', 'artist_two']:
            model_path = models_dir / f'{name}.pt'
            embedding = torch.randn(256)
            torch.save({'embedding': embedding}, str(model_path))

        return VoiceModelRegistry(models_dir=str(models_dir))

    @pytest.mark.smoke
    def test_init_empty_dir(self, empty_registry):
        """Test initialization with empty directory."""
        assert len(empty_registry._pretrained_models) == 0
        assert len(empty_registry._extracted_models) == 0

    def test_init_with_pretrained(self, registry_with_models):
        """Test initialization scans pretrained models."""
        assert len(registry_with_models._pretrained_models) == 2
        assert 'artist_one' in registry_with_models._pretrained_models
        assert 'artist_two' in registry_with_models._pretrained_models

    def test_init_nonexistent_dir(self, tmp_path):
        """Test initialization with non-existent directory."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'nonexistent'

        # Should not raise, just log info
        registry = VoiceModelRegistry(models_dir=str(models_dir))

        assert len(registry._pretrained_models) == 0


class TestListModels:
    """Test list_models method."""

    @pytest.fixture
    def registry(self, tmp_path):
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        model_path = models_dir / 'test_model.pt'
        torch.save({'embedding': torch.randn(256)}, str(model_path))

        return VoiceModelRegistry(models_dir=str(models_dir))

    def test_list_models_includes_pretrained(self, registry):
        """Test list_models includes pretrained models."""
        models = registry.list_models()

        assert len(models) >= 1
        model_ids = [m['id'] for m in models]
        assert 'test_model' in model_ids

    def test_list_models_includes_extracted(self, registry):
        """Test list_models includes extracted models."""
        # Register an extracted model
        embedding = torch.randn(256)
        registry.register_extracted_model('Test Extracted', embedding)

        models = registry.list_models()

        # Should include both pretrained and extracted
        model_types = [m['type'] for m in models]
        assert 'pretrained' in model_types
        assert 'extracted' in model_types

    def test_list_models_structure(self, registry):
        """Test list_models returns correct structure."""
        models = registry.list_models()

        for model in models:
            assert 'id' in model
            assert 'name' in model
            assert 'type' in model


class TestGetModel:
    """Test get_model method."""

    @pytest.fixture
    def registry(self, tmp_path):
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        model_path = models_dir / 'test_artist.pt'
        torch.save({'embedding': torch.randn(256)}, str(model_path))

        return VoiceModelRegistry(models_dir=str(models_dir))

    def test_get_model_pretrained(self, registry):
        """Test getting pretrained model."""
        model = registry.get_model('test_artist')

        assert model is not None
        assert model['id'] == 'test_artist'
        assert model['type'] == 'pretrained'
        assert 'embedding_dim' in model

    def test_get_model_extracted(self, registry):
        """Test getting extracted model."""
        embedding = torch.randn(256)
        model_id = registry.register_extracted_model('My Model', embedding, 'song123')

        model = registry.get_model(model_id)

        assert model is not None
        assert model['type'] == 'extracted'
        assert model['source_song_id'] == 'song123'

    def test_get_model_not_found(self, registry):
        """Test getting non-existent model."""
        model = registry.get_model('nonexistent_model')

        assert model is None


class TestGetEmbedding:
    """Test get_embedding method."""

    @pytest.fixture
    def registry(self, tmp_path):
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        # Create model with embedding
        embedding = torch.randn(256)
        model_path = models_dir / 'test_speaker.pt'
        torch.save({'embedding': embedding}, str(model_path))

        return VoiceModelRegistry(models_dir=str(models_dir))

    def test_get_embedding_pretrained(self, registry):
        """Test getting embedding from pretrained model."""
        embedding = registry.get_embedding('test_speaker')

        assert embedding is not None
        assert embedding.shape == (256,)
        assert embedding.dtype == torch.float32

    def test_get_embedding_extracted(self, registry):
        """Test getting embedding from extracted model."""
        original_embedding = torch.randn(256)
        model_id = registry.register_extracted_model('Test', original_embedding)

        embedding = registry.get_embedding(model_id)

        assert embedding is not None
        assert torch.allclose(embedding, original_embedding)

    def test_get_embedding_not_found(self, registry):
        """Test getting embedding for non-existent model."""
        embedding = registry.get_embedding('nonexistent')

        assert embedding is None

    def test_get_embedding_corrupted_file(self, tmp_path):
        """Test handling corrupted model file."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        # Create corrupted file
        corrupted_path = models_dir / 'corrupted.pt'
        corrupted_path.write_bytes(b'not a valid pytorch file')

        registry = VoiceModelRegistry(models_dir=str(models_dir))
        embedding = registry.get_embedding('corrupted')

        assert embedding is None

    def test_get_embedding_alternative_key(self, tmp_path):
        """Test loading embedding with alternative key name."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        # Save with 'speaker_embedding' key instead of 'embedding'
        embedding = torch.randn(256)
        model_path = models_dir / 'alt_key.pt'
        torch.save({'speaker_embedding': embedding}, str(model_path))

        registry = VoiceModelRegistry(models_dir=str(models_dir))
        loaded = registry.get_embedding('alt_key')

        assert loaded is not None
        assert loaded.shape == (256,)


class TestRegisterExtractedModel:
    """Test register_extracted_model method."""

    @pytest.fixture
    def registry(self, tmp_path):
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        return VoiceModelRegistry(models_dir=str(models_dir))

    def test_register_model_basic(self, registry):
        """Test basic model registration."""
        embedding = torch.randn(256)
        model_id = registry.register_extracted_model('Test Model', embedding)

        assert model_id.startswith('extracted_')
        assert model_id in registry._extracted_models

    def test_register_model_with_source_song(self, registry):
        """Test registering model with source song ID."""
        embedding = torch.randn(256)
        model_id = registry.register_extracted_model(
            'Test Model', embedding, source_song_id='song_abc123'
        )

        model = registry._extracted_models[model_id]
        assert model['source_song_id'] == 'song_abc123'

    def test_register_model_squeeze_embedding(self, registry):
        """Test that embedding is squeezed if needed."""
        # 2D embedding (batch dimension)
        embedding = torch.randn(1, 256)
        model_id = registry.register_extracted_model('Test', embedding)

        stored = registry._extracted_models[model_id]['embedding']
        assert stored.dim() == 1
        assert stored.shape == (256,)

    def test_register_model_wrong_dim_raises(self, registry):
        """Test that wrong embedding dimension raises error."""
        embedding = torch.randn(128)  # Wrong dimension

        with pytest.raises(ValueError, match="must be 256-dim"):
            registry.register_extracted_model('Test', embedding)

    def test_register_model_unique_ids(self, registry):
        """Test that each registration gets unique ID."""
        embedding = torch.randn(256)

        id1 = registry.register_extracted_model('Model 1', embedding)
        id2 = registry.register_extracted_model('Model 2', embedding)

        assert id1 != id2

    def test_register_model_clones_embedding(self, registry):
        """Test that embedding is cloned, not referenced."""
        embedding = torch.randn(256)
        model_id = registry.register_extracted_model('Test', embedding)

        # Modify original
        embedding.fill_(0)

        # Stored should be unchanged
        stored = registry._extracted_models[model_id]['embedding']
        assert not torch.allclose(stored, torch.zeros(256))


class TestExtractAndRegisterFromAudio:
    """Test extract_and_register_from_audio method."""

    @pytest.fixture
    def registry(self, tmp_path):
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        return VoiceModelRegistry(models_dir=str(models_dir))

    def test_extract_and_register_basic(self, registry):
        """Test extracting and registering from audio."""
        audio = torch.randn(24000 * 5) * 0.1  # 5 seconds

        model_id = registry.extract_and_register_from_audio(
            audio, 'Test Voice', sample_rate=24000
        )

        assert model_id.startswith('extracted_')
        model = registry.get_model(model_id)
        assert model is not None
        assert model['name'] == 'Test Voice'

    def test_extract_and_register_with_source_song(self, registry):
        """Test extracting with source song ID."""
        audio = torch.randn(24000 * 3) * 0.1

        model_id = registry.extract_and_register_from_audio(
            audio, 'Voice', sample_rate=24000, source_song_id='song123'
        )

        model = registry.get_model(model_id)
        assert model['source_song_id'] == 'song123'

    def test_extract_and_register_different_sample_rates(self, registry):
        """Test extraction at different sample rates."""
        for sr in [16000, 22050, 44100]:
            audio = torch.randn(sr * 2) * 0.1

            model_id = registry.extract_and_register_from_audio(
                audio, f'Voice_{sr}', sample_rate=sr
            )

            assert model_id.startswith('extracted_')


class TestModelScanning:
    """Test model file scanning."""

    def test_scan_pth_files(self, tmp_path):
        """Test scanning .pth files."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        # Create .pth file
        model_path = models_dir / 'model.pth'
        torch.save({'embedding': torch.randn(256)}, str(model_path))

        registry = VoiceModelRegistry(models_dir=str(models_dir))

        assert 'model' in registry._pretrained_models

    def test_scan_pt_files(self, tmp_path):
        """Test scanning .pt files."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        # Create .pt file
        model_path = models_dir / 'model.pt'
        torch.save({'embedding': torch.randn(256)}, str(model_path))

        registry = VoiceModelRegistry(models_dir=str(models_dir))

        assert 'model' in registry._pretrained_models

    def test_scan_ignores_other_files(self, tmp_path):
        """Test that non-model files are ignored."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        # Create non-model files
        (models_dir / 'readme.txt').write_text('readme')
        (models_dir / 'config.json').write_text('{}')
        (models_dir / 'model.pt').write_bytes(b'')  # Empty but valid extension

        registry = VoiceModelRegistry(models_dir=str(models_dir))

        # Should only scan .pt/.pth files (model.pt)
        # readme.txt and config.json should be ignored
        assert 'readme' not in registry._pretrained_models
        assert 'config' not in registry._pretrained_models

    def test_model_name_from_filename(self, tmp_path):
        """Test that model name is derived from filename."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        models_dir = tmp_path / 'voice_models'
        models_dir.mkdir()

        model_path = models_dir / 'famous_artist_voice.pt'
        torch.save({'embedding': torch.randn(256)}, str(model_path))

        registry = VoiceModelRegistry(models_dir=str(models_dir))
        model = registry._pretrained_models['famous_artist_voice']

        # Name should have underscores converted and be title cased
        assert model['name'] == 'Famous Artist Voice'
