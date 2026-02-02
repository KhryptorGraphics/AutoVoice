"""Tests for HQ LoRA Adapter Bridge.

Tests the adapter bridge that handles the architecture mismatch between
trained HQ adapters (standalone MLP) and the expected layer-injection format.
"""

import pytest
import torch
from pathlib import Path

from auto_voice.models.hq_adapter_bridge import (
    HQVoiceLoRAAdapter,
    HQLoRAAdapterBridge,
    AdapterBridgeConfig,
    DEFAULT_HQ_CONFIG,
    get_hq_adapter_bridge,
)


class TestHQVoiceLoRAAdapter:
    """Test the HQVoiceLoRAAdapter module."""

    def test_adapter_creation(self):
        """Test adapter can be created with default config."""
        adapter = HQVoiceLoRAAdapter(**DEFAULT_HQ_CONFIG)
        assert adapter is not None
        assert adapter.input_dim == 768
        assert adapter.output_dim == 768
        assert adapter.num_layers == 6

    def test_adapter_forward_pass(self):
        """Test forward pass with correct shapes."""
        adapter = HQVoiceLoRAAdapter(**DEFAULT_HQ_CONFIG)
        adapter.train(False)

        # Input: [B, T, 768]
        content = torch.randn(2, 100, 768)
        output = adapter(content)

        assert output.shape == (2, 100, 768)
        assert not torch.isnan(output).any()

    def test_adapter_with_speaker_embedding(self):
        """Test forward pass with speaker conditioning."""
        adapter = HQVoiceLoRAAdapter(**DEFAULT_HQ_CONFIG)
        adapter.train(False)

        content = torch.randn(2, 100, 768)
        speaker = torch.randn(2, 256)  # 256-dim speaker embedding

        output = adapter(content, speaker)

        assert output.shape == (2, 100, 768)
        assert not torch.isnan(output).any()

    def test_lora_state_dict_roundtrip(self):
        """Test saving and loading LoRA state."""
        adapter = HQVoiceLoRAAdapter(**DEFAULT_HQ_CONFIG)

        # Get state
        state = adapter.get_lora_state_dict()
        assert len(state) == 12  # 6 layers * 2 (A and B)

        # Check keys
        for i in range(6):
            assert f'lora_{i}_A' in state
            assert f'lora_{i}_B' in state

        # Modify state
        for key in state:
            state[key] = torch.randn_like(state[key])

        # Load state
        adapter.load_lora_state_dict(state)

        # Verify loaded correctly
        new_state = adapter.get_lora_state_dict()
        for key in state:
            assert torch.allclose(state[key], new_state[key])


class TestHQLoRAAdapterBridge:
    """Test the HQ LoRA Adapter Bridge."""

    @pytest.fixture
    def temp_adapter_dir(self, tmp_path):
        """Create temporary adapter directory."""
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()
        return adapter_dir

    @pytest.fixture
    def mock_adapter_file(self, temp_adapter_dir):
        """Create a mock adapter file for testing."""
        profile_id = "test-profile-123"

        # Create adapter
        adapter = HQVoiceLoRAAdapter(**DEFAULT_HQ_CONFIG)
        lora_state = adapter.get_lora_state_dict()

        # Save checkpoint
        checkpoint = {
            'profile_id': profile_id,
            'artist': 'Test Artist',
            'config': DEFAULT_HQ_CONFIG,
            'lora_state': lora_state,
            'epoch': 100,
            'loss': 0.01,
        }

        adapter_path = temp_adapter_dir / f"{profile_id}_hq_lora.pt"
        torch.save(checkpoint, adapter_path)

        return profile_id, adapter_path

    def test_bridge_initialization(self, temp_adapter_dir):
        """Test bridge can be initialized."""
        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        assert bridge is not None
        assert bridge.device == torch.device('cpu')

    def test_list_adapters_empty(self, temp_adapter_dir):
        """Test listing adapters when directory is empty."""
        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        adapters = bridge.list_adapters()
        assert adapters == []

    def test_list_adapters(self, temp_adapter_dir, mock_adapter_file):
        """Test listing adapters when adapters exist."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        adapters = bridge.list_adapters()
        assert profile_id in adapters

    def test_has_adapter(self, temp_adapter_dir, mock_adapter_file):
        """Test checking for adapter existence."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        assert bridge.has_adapter(profile_id) is True
        assert bridge.has_adapter("nonexistent") is False

    def test_load_adapter(self, temp_adapter_dir, mock_adapter_file):
        """Test loading an adapter."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        adapter = bridge.load_adapter(profile_id)
        assert adapter is not None
        assert isinstance(adapter, HQVoiceLoRAAdapter)
        assert bridge.get_current_profile() == profile_id

    def test_load_adapter_not_found(self, temp_adapter_dir):
        """Test loading nonexistent adapter raises error."""
        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        with pytest.raises(FileNotFoundError):
            bridge.load_adapter("nonexistent-profile")

    def test_transform(self, temp_adapter_dir, mock_adapter_file):
        """Test content transformation with loaded adapter."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        # Load adapter
        bridge.load_adapter(profile_id)

        # Transform content
        content = torch.randn(1, 50, 768)
        output = bridge.transform(content)

        assert output.shape == (1, 50, 768)
        assert not torch.isnan(output).any()

    def test_transform_without_adapter_raises(self, temp_adapter_dir):
        """Test transform without loaded adapter raises error."""
        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        content = torch.randn(1, 50, 768)

        with pytest.raises(RuntimeError, match="No adapter loaded"):
            bridge.transform(content)

    def test_clear_adapter(self, temp_adapter_dir, mock_adapter_file):
        """Test clearing the current adapter."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        # Load and then clear
        bridge.load_adapter(profile_id)
        assert bridge.get_current_profile() == profile_id

        bridge.clear()
        assert bridge.get_current_profile() is None
        assert bridge.get_current_adapter() is None

    def test_adapter_caching(self, temp_adapter_dir, mock_adapter_file):
        """Test adapter is cached on second load."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu',
            cache_size=3
        )
        bridge = HQLoRAAdapterBridge(config)

        # Load twice
        adapter1 = bridge.load_adapter(profile_id)
        adapter2 = bridge.load_adapter(profile_id)

        # Should return same instance from cache
        assert adapter1 is adapter2

    def test_get_adapter_info(self, temp_adapter_dir, mock_adapter_file):
        """Test getting adapter metadata."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cpu'
        )
        bridge = HQLoRAAdapterBridge(config)

        info = bridge.get_adapter_info(profile_id)

        assert info is not None
        assert info['profile_id'] == profile_id
        assert info['artist'] == 'Test Artist'
        assert info['epoch'] == 100


@pytest.mark.cuda
class TestHQAdapterBridgeCUDA:
    """CUDA-specific tests for adapter bridge."""

    @pytest.fixture
    def temp_adapter_dir(self, tmp_path):
        """Create temporary adapter directory."""
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()
        return adapter_dir

    @pytest.fixture
    def mock_adapter_file(self, temp_adapter_dir):
        """Create a mock adapter file for testing."""
        profile_id = "cuda-test-profile"

        adapter = HQVoiceLoRAAdapter(**DEFAULT_HQ_CONFIG)
        lora_state = adapter.get_lora_state_dict()

        checkpoint = {
            'profile_id': profile_id,
            'config': DEFAULT_HQ_CONFIG,
            'lora_state': lora_state,
        }

        adapter_path = temp_adapter_dir / f"{profile_id}_hq_lora.pt"
        torch.save(checkpoint, adapter_path)

        return profile_id, adapter_path

    def test_transform_on_cuda(self, temp_adapter_dir, mock_adapter_file):
        """Test content transformation on CUDA."""
        profile_id, _ = mock_adapter_file

        config = AdapterBridgeConfig(
            adapters_dir=temp_adapter_dir,
            device='cuda'
        )
        bridge = HQLoRAAdapterBridge(config)
        bridge.load_adapter(profile_id)

        content = torch.randn(1, 50, 768).cuda()
        output = bridge.transform(content)

        assert output.device.type == 'cuda'
        assert output.shape == (1, 50, 768)
        assert not torch.isnan(output).any()
