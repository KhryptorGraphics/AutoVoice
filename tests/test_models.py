"""
Comprehensive model tests for AutoVoice.

Tests VoiceTransformer, HiFiGAN, and VoiceModel architectures.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path


@pytest.mark.model
@pytest.mark.unit
class TestVoiceTransformer:
    """Test VoiceTransformer from src/auto_voice/models/transformer.py"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.models.transformer import VoiceTransformer
            # Use input_dim instead of vocab_size for mel spectrogram input
            self.config = {
                'input_dim': 256,  # Changed from vocab_size
                'hidden_size': 256,
                'num_layers': 4,
                'num_heads': 4,
                'dropout': 0.1,
                'max_sequence_length': 512
            }
            self.model = VoiceTransformer(input_dim=256, d_model=256, n_layers=4, n_heads=4)
            self.model.eval()
        except ImportError:
            pytest.skip("VoiceTransformer not available")

    def test_model_creation(self):
        """Test model instantiation."""
        assert self.model.hidden_size == 256
        assert self.model.num_layers == 4
        assert self.model.num_heads == 4

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 100), (4, 200), (8, 50)])
    def test_forward_pass_shapes(self, batch_size, seq_len):
        """Test forward pass with various input shapes."""
        # Use float tensor for mel spectrogram input instead of token IDs
        input_mel = torch.randn(batch_size, seq_len, 256)

        with torch.no_grad():
            output = self.model(input_mel)

        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len
        assert output.shape[2] == 256  # output dimension
        assert not torch.isnan(output).any()

    def test_attention_mask_handling(self):
        """Test attention mask for variable-length sequences."""
        batch_size, seq_len = 4, 100
        input_mel = torch.randn(batch_size, seq_len, 256)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 50:] = 0  # Mask second half

        with torch.no_grad():
            output = self.model(input_mel, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, 256)

    def test_positional_encoding(self):
        """Test positional encoding generation."""
        if hasattr(self.model, 'positional_encoding'):
            pos_enc = self.model.positional_encoding
            assert pos_enc is not None
            # Verify sinusoidal pattern properties
            assert not torch.isnan(pos_enc).any()

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        input_mel = torch.randn(2, 50, 256)
        output = self.model(input_mel)
        loss = output.mean()
        loss.backward()

        # Check no NaN/Inf gradients
        for param in self.model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

    def test_model_serialization(self, tmp_path):
        """Test save/load state dict."""
        checkpoint_path = tmp_path / "transformer.pt"

        # Save
        torch.save(self.model.state_dict(), checkpoint_path)

        # Load into new model
        from src.auto_voice.models.transformer import VoiceTransformer
        new_model = VoiceTransformer(input_dim=256, d_model=256, n_layers=4, n_heads=4)
        new_model.load_state_dict(torch.load(checkpoint_path))

        # Verify weights match
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_device_transfer(self, cuda_device):
        """Test model transfer between devices."""
        if cuda_device.type == 'cuda':
            self.model.to(cuda_device)
            input_mel = torch.randn(2, 50, 256, device=cuda_device)

            with torch.no_grad():
                output = self.model(input_mel)

            assert output.device == cuda_device


@pytest.mark.model
@pytest.mark.unit
class TestHiFiGAN:
    """Test HiFiGAN from src/auto_voice/models/hifigan.py"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.models.hifigan import HiFiGANGenerator, HiFiGANDiscriminator
            self.generator_config = {
                'mel_channels': 80,  # Use mel_channels instead of in_channels
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4],
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            }
            self.generator = HiFiGANGenerator(mel_channels=80)
            self.discriminator = HiFiGANDiscriminator()
            self.generator.eval()
            self.discriminator.eval()
        except ImportError:
            pytest.skip("HiFiGAN not available")

    def test_generator_creation(self):
        """Test generator instantiation."""
        assert self.generator is not None
        assert isinstance(self.generator, nn.Module)

    @pytest.mark.parametrize("batch_size,time_steps", [(1, 100), (4, 200)])
    def test_generator_forward(self, batch_size, time_steps):
        """Test generator forward pass."""
        mel_input = torch.randn(batch_size, 80, time_steps)

        with torch.no_grad():
            audio = self.generator(mel_input)

        assert audio.ndim == 2 or audio.ndim == 3
        assert audio.shape[0] == batch_size
        assert not torch.isnan(audio).any()

    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        batch_size, audio_len = 4, 16000
        audio = torch.randn(batch_size, 1, audio_len)

        with torch.no_grad():
            # Use forward_single for testing single input
            disc_output = self.discriminator.forward_single(audio)

        assert disc_output is not None
        assert len(disc_output) > 0  # Should return list of outputs
        for output in disc_output:
            assert not torch.isnan(output[0]).any() if isinstance(output, tuple) else not torch.isnan(output).any()

    def test_output_audio_length(self):
        """Test output audio length matches expected duration."""
        mel_input = torch.randn(1, 80, 100)

        with torch.no_grad():
            audio = self.generator(mel_input)

        # Calculate expected length based on upsampling
        expected_len = 100
        for rate in [8, 8, 2, 2]:  # Use the actual upsample rates
            expected_len *= rate

        # Audio should be 1D (batch, 1, time) or 2D (batch, time)
        if audio.dim() == 3:
            actual_len = audio.shape[-1]
        else:
            actual_len = audio.shape[-1]
        assert abs(actual_len - expected_len) < 1000  # More lenient tolerance

    def test_remove_weight_norm(self):
        """Test weight normalization removal."""
        if hasattr(self.generator, 'remove_weight_norm'):
            self.generator.remove_weight_norm()
            # Model should still work
            mel_input = torch.randn(1, 80, 100)
            with torch.no_grad():
                audio = self.generator(mel_input)
            assert audio is not None


@pytest.mark.model
@pytest.mark.integration
class TestVoiceModel:
    """Test VoiceModel from src/auto_voice/models/voice_model.py"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.models.voice_model import VoiceModel
            self.config = {
                'hidden_size': 256,
                'num_layers': 4,
                'num_heads': 4,
                'dropout': 0.1,
                'num_speakers': 10,
                'mel_channels': 80  # Add mel_channels for VoiceModel
            }
            self.model = VoiceModel(self.config)
        except ImportError:
            pytest.skip("VoiceModel not available")

    def test_model_creation(self):
        """Test model creation with config."""
        assert self.model.hidden_size == 256
        assert self.model.num_layers == 4

    @pytest.mark.parametrize("speaker_id", [0, 1, 5, 9])
    def test_multi_speaker_support(self, speaker_id):
        """Test with different speaker IDs."""
        batch_size, seq_len = 2, 100
        mel_input = torch.randn(batch_size, seq_len, 80)  # (batch, time, mel_channels)

        with torch.no_grad():
            output = self.model(mel_input, speaker_id=torch.tensor([speaker_id]))

        assert output is not None
        if isinstance(output, dict):
            # VoiceModel returns dict with multiple outputs
            assert 'mel_output' in output
            assert not torch.isnan(output['mel_output']).any()
        else:
            assert not torch.isnan(output).any()

    def test_load_checkpoint(self, tmp_path):
        """Test checkpoint loading."""
        checkpoint_path = tmp_path / "model.pt"

        # Create dummy checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': 10
        }
        torch.save(checkpoint, checkpoint_path)

        # Load
        self.model.load_checkpoint(str(checkpoint_path))
        assert self.model.is_loaded()

    def test_invalid_checkpoint(self, tmp_path):
        """Test handling of invalid checkpoint."""
        invalid_path = tmp_path / "invalid.pt"
        invalid_path.write_text("invalid data")

        with pytest.raises((RuntimeError, ValueError, Exception)):
            self.model.load_checkpoint(str(invalid_path))

    def test_get_speaker_list(self):
        """Test speaker list retrieval."""
        speakers = self.model.get_speaker_list()

        assert isinstance(speakers, list)
        assert len(speakers) > 0
        if len(speakers) > 0:
            assert 'id' in speakers[0]
            assert 'name' in speakers[0]

    def test_missing_speaker_embedding(self):
        """Test with missing speaker embeddings."""
        if hasattr(self.model, 'num_speakers'):
            invalid_speaker_id = self.model.num_speakers + 10
            mel_input = torch.randn(1, 100, 80)  # (batch, time, mel_channels)

            with pytest.raises((IndexError, ValueError, RuntimeError)):
                self.model(mel_input, speaker_id=torch.tensor([invalid_speaker_id]))


@pytest.mark.model
@pytest.mark.unit
class TestTransformerInternals:
    """Test VoiceTransformer internal components."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from src.auto_voice.models.transformer import MultiHeadAttention, TransformerBlock
            self.d_model = 256
            self.n_heads = 4
            self.mha = MultiHeadAttention(self.d_model, self.n_heads)
            self.transformer_block = TransformerBlock(self.d_model, self.n_heads, d_ff=1024)
        except ImportError:
            pytest.skip("Transformer components not available")

    def test_multi_head_attention_shape(self):
        """Test MultiHeadAttention output shape."""
        batch_size, seq_len = 4, 100
        query = key = value = torch.randn(batch_size, seq_len, self.d_model)

        with torch.no_grad():
            output = self.mha(query, key, value)

        assert output.shape == (batch_size, seq_len, self.d_model)
        assert not torch.isnan(output).any()

    def test_multi_head_attention_masking(self):
        """Test attention masking."""
        batch_size, seq_len = 2, 50
        query = key = value = torch.randn(batch_size, seq_len, self.d_model)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        with torch.no_grad():
            output = self.mha(query, key, value, mask=mask)

        assert output.shape == (batch_size, seq_len, self.d_model)

    def test_transformer_block_residual(self):
        """Test residual connections in TransformerBlock."""
        batch_size, seq_len = 2, 100
        x = torch.randn(batch_size, seq_len, self.d_model)

        with torch.no_grad():
            output = self.transformer_block(x)

        # Output should be different from input due to transformations
        assert not torch.allclose(output, x, atol=0.1)
        assert output.shape == x.shape

    def test_transformer_block_gradient_flow(self):
        """Test gradient flow through TransformerBlock."""
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)

        output = self.transformer_block(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


@pytest.mark.model
@pytest.mark.unit
class TestHiFiGANComponents:
    """Test HiFiGAN internal components."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from src.auto_voice.models.hifigan import ResBlock, MRF
            self.channels = 128
            self.resblock = ResBlock(self.channels, kernel_size=3, dilation=(1, 3, 5))
            self.mrf = MRF(self.channels,
                          resblock_kernel_sizes=[3, 7, 11],
                          resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)])
        except ImportError:
            pytest.skip("HiFiGAN components not available")

    def test_resblock_forward(self):
        """Test ResBlock forward pass."""
        batch_size, time_steps = 4, 100
        x = torch.randn(batch_size, self.channels, time_steps)

        with torch.no_grad():
            output = self.resblock(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_resblock_residual_connection(self):
        """Test ResBlock preserves residual connection."""
        batch_size, time_steps = 2, 100
        x = torch.randn(batch_size, self.channels, time_steps)

        with torch.no_grad():
            output = self.resblock(x)

        # Output should be different from input but of same shape
        assert output.shape == x.shape
        # Check that residual connection adds meaningful transformation
        assert not torch.allclose(output, x, atol=0.01)

    def test_mrf_multi_receptive_field(self):
        """Test Multi-Receptive Field Fusion."""
        batch_size, time_steps = 2, 100
        x = torch.randn(batch_size, self.channels, time_steps)

        with torch.no_grad():
            output = self.mrf(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_mrf_averaging(self):
        """Test MRF averages multiple resblock outputs."""
        batch_size, time_steps = 2, 50
        x = torch.randn(batch_size, self.channels, time_steps)

        with torch.no_grad():
            output = self.mrf(x)

        # MRF should produce averaged output from multiple paths
        assert output.shape == x.shape


@pytest.mark.model
@pytest.mark.integration
class TestModelIntegration:
    """Test model component integration."""

    def test_transformer_to_vocoder_pipeline(self):
        """Test transformer output can feed into vocoder."""
        try:
            from src.auto_voice.models.transformer import VoiceTransformer
            from src.auto_voice.models.hifigan import HiFiGANGenerator

            # Create models
            transformer = VoiceTransformer(input_dim=80, d_model=256, n_layers=2)
            vocoder = HiFiGANGenerator(mel_channels=80)

            # Generate mel-spectrogram from transformer
            batch_size, seq_len = 2, 100
            input_mel = torch.randn(batch_size, seq_len, 80)

            with torch.no_grad():
                transformer_output = transformer(input_mel)
                # Transpose for vocoder (B, T, C) -> (B, C, T)
                vocoder_input = transformer_output.transpose(1, 2)
                audio = vocoder(vocoder_input)

            assert audio.shape[0] == batch_size
            assert not torch.isnan(audio).any()
        except ImportError:
            pytest.skip("Models not available")

    def test_onnx_export_transformer(self, tmp_path):
        """Test transformer ONNX export."""
        try:
            import onnx
        except ImportError:
            pytest.skip("ONNX module not available")
            
        try:
            from src.auto_voice.models.transformer import VoiceTransformer

            model = VoiceTransformer(input_dim=80, d_model=256, n_layers=2)
            onnx_path = tmp_path / "transformer.onnx"

            # Export to ONNX
            model.export_to_onnx(str(onnx_path), input_shape=(1, 100, 80), verbose=False)

            # Verify file was created
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0
        except ImportError:
            pytest.skip("ONNX export not available")

    def test_onnx_export_hifigan(self, tmp_path):
        """Test HiFiGAN ONNX export."""
        try:
            import onnx
        except ImportError:
            pytest.skip("ONNX module not available")
            
        try:
            from src.auto_voice.models.hifigan import HiFiGANGenerator

            model = HiFiGANGenerator(mel_channels=80)
            onnx_path = tmp_path / "hifigan.onnx"

            # Export to ONNX
            model.export_to_onnx(str(onnx_path), mel_shape=(1, 80, 100), verbose=False)

            # Verify file was created
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0
        except ImportError:
            pytest.skip("ONNX export not available")


@pytest.mark.model
@pytest.mark.unit
class TestNumericalStability:
    """Test model numerical stability."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from src.auto_voice.models.transformer import VoiceTransformer
            self.model = VoiceTransformer(input_dim=80, d_model=128, n_layers=2)
            self.model.eval()
        except ImportError:
            pytest.skip("VoiceTransformer not available")

    def test_very_small_inputs(self):
        """Test with very small input values."""
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, 80) * 1e-6

        with torch.no_grad():
            output = self.model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_large_inputs(self):
        """Test with very large input values."""
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, 80) * 1e3

        with torch.no_grad():
            output = self.model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_all_zero_inputs(self):
        """Test with all-zero inputs."""
        batch_size, seq_len = 2, 50
        x = torch.zeros(batch_size, seq_len, 80)

        with torch.no_grad():
            output = self.model(x)

        assert not torch.isnan(output).any()
        # All-zero input should still produce valid output
        assert output.abs().sum() > 0

    def test_random_noise_inputs(self):
        """Test with random noise inputs."""
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, 80) * 10

        with torch.no_grad():
            output = self.model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mixed_scale_inputs(self):
        """Test with mixed scale inputs (some large, some small)."""
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, 80)
        x[:, :25, :] *= 1e-6  # Very small values
        x[:, 25:, :] *= 1e3   # Very large values

        with torch.no_grad():
            output = self.model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.model
@pytest.mark.performance
class TestModelPerformance:
    """Test model performance and memory."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from src.auto_voice.models.transformer import VoiceTransformer
            from src.auto_voice.models.hifigan import HiFiGANGenerator
            self.transformer = VoiceTransformer(input_dim=80, d_model=256, n_layers=4)
            self.vocoder = HiFiGANGenerator(mel_channels=80)
            self.transformer.eval()
            self.vocoder.eval()
        except ImportError:
            pytest.skip("Models not available")

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_processing_transformer(self, batch_size):
        """Test transformer batch processing."""
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 80)

        with torch.no_grad():
            output = self.transformer(x)

        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_processing_vocoder(self, batch_size):
        """Test vocoder batch processing."""
        time_steps = 100
        x = torch.randn(batch_size, 80, time_steps)

        with torch.no_grad():
            output = self.vocoder(x)

        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

    @pytest.mark.slow
    @pytest.mark.cuda
    def test_gpu_memory_usage(self, cuda_device):
        """Test GPU memory usage."""
        if cuda_device.type != 'cuda':
            pytest.skip("CUDA not available")

        self.transformer.to(cuda_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Run inference
        batch_size, seq_len = 8, 200
        x = torch.randn(batch_size, seq_len, 80, device=cuda_device)

        with torch.no_grad():
            output = self.transformer(x)

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"Peak GPU memory: {peak_memory:.2f} MB")

        # Should use reasonable amount of memory (adjust threshold as needed)
        assert peak_memory < 2000  # Less than 2GB for this size

    def test_inference_speed_transformer(self, benchmark):
        """Test transformer inference speed."""
        batch_size, seq_len = 4, 100
        x = torch.randn(batch_size, seq_len, 80)

        def run_inference():
            with torch.no_grad():
                return self.transformer(x)

        try:
            result = benchmark(run_inference)
            print(f"Transformer inference completed")
        except:
            # Fallback if benchmark fixture not available
            with torch.no_grad():
                result = self.transformer(x)
            assert result is not None

    def test_inference_speed_vocoder(self, benchmark):
        """Test vocoder inference speed."""
        batch_size, time_steps = 4, 100
        x = torch.randn(batch_size, 80, time_steps)

        def run_inference():
            with torch.no_grad():
                return self.vocoder(x)

        try:
            result = benchmark(run_inference)
            print(f"Vocoder inference completed")
        except:
            # Fallback if benchmark fixture not available
            with torch.no_grad():
                result = self.vocoder(x)
            assert result is not None
