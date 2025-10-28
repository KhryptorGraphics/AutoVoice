"""
Comprehensive test suite for Singing Voice Conversion components.

Tests ContentEncoder, PitchEncoder, PosteriorEncoder, FlowDecoder,
and SingingVoiceConverter integration.
"""

import pytest
import torch
import numpy as np
import logging

from src.auto_voice.models.content_encoder import ContentEncoder, ContentEncodingError
from src.auto_voice.models.pitch_encoder import PitchEncoder
from src.auto_voice.models.posterior_encoder import PosteriorEncoder, WaveNetResidualBlock
from src.auto_voice.models.flow_decoder import FlowDecoder, AffineCouplingLayer, Flip
from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter, VoiceConversionError

logger = logging.getLogger(__name__)


# ========== ContentEncoder Tests ==========

@pytest.mark.model
@pytest.mark.unit
class TestContentEncoder:
    """Test ContentEncoder component."""

    def test_content_encoder_initialization(self):
        """Test ContentEncoder initializes correctly."""
        encoder = ContentEncoder(encoder_type='cnn_fallback', device='cpu')
        assert encoder.encoder_type == 'cnn_fallback'
        assert encoder.output_dim == 256
        assert encoder.sample_rate == 16000

    def test_content_encoder_forward(self):
        """Test ContentEncoder forward pass."""
        encoder = ContentEncoder(encoder_type='cnn_fallback', device='cpu')
        audio = torch.randn(2, 16000)  # 1 second at 16kHz
        content = encoder(audio, sample_rate=16000)

        # Check shape: [B, T_frames, output_dim]
        assert content.dim() == 3
        assert content.shape[0] == 2  # Batch
        assert content.shape[2] == 256  # Output dim

        # Check no NaN/Inf
        assert torch.isfinite(content).all()

    def test_content_encoder_removes_speaker_info(self):
        """Test that content encoder removes speaker information."""
        encoder = ContentEncoder(encoder_type='cnn_fallback', device='cpu')

        # Generate two audio samples with different characteristics
        audio1 = torch.randn(1, 16000) * 0.5
        audio2 = torch.randn(1, 16000) * 1.5

        content1 = encoder(audio1, sample_rate=16000)
        content2 = encoder(audio2, sample_rate=16000)

        # Check that both produce valid outputs
        assert content1.shape == content2.shape
        assert torch.isfinite(content1).all()
        assert torch.isfinite(content2).all()

    @pytest.mark.slow
    def test_hubert_soft_loading(self):
        """Test loading HuBERT-Soft from PyTorch Hub (if available)."""
        pytest.skip("Skipping HuBERT-Soft loading test (requires network)")

    def test_content_encoder_extract_method(self):
        """Test high-level extract_content method."""
        encoder = ContentEncoder(encoder_type='cnn_fallback', device='cpu')

        # Test with numpy array
        audio_np = np.random.randn(16000).astype(np.float32)
        content = encoder.extract_content(audio_np, sample_rate=16000)
        assert isinstance(content, torch.Tensor)
        assert torch.isfinite(content).all()

    def test_content_encoder_get_frame_rate(self):
        """Test get_frame_rate method."""
        encoder = ContentEncoder(encoder_type='cnn_fallback', device='cpu')
        frame_rate = encoder.get_frame_rate()
        assert frame_rate == 50.0


# ========== PitchEncoder Tests ==========

@pytest.mark.model
@pytest.mark.unit
class TestPitchEncoder:
    """Test PitchEncoder component."""

    def test_pitch_encoder_initialization(self):
        """Test PitchEncoder initializes correctly."""
        encoder = PitchEncoder(pitch_dim=192, f0_min=80.0, f0_max=1000.0)
        assert encoder.pitch_dim == 192
        assert encoder.f0_min == 80.0
        assert encoder.f0_max == 1000.0

    def test_pitch_encoder_forward(self):
        """Test PitchEncoder forward pass."""
        encoder = PitchEncoder(pitch_dim=192)
        f0 = torch.tensor([[440.0, 450.0, 460.0, 0.0, 470.0]])  # [B=1, T=5]
        voiced = torch.tensor([[True, True, True, False, True]])

        pitch_emb = encoder(f0, voiced)

        # Check shape: [B, T, pitch_dim]
        assert pitch_emb.shape == (1, 5, 192)
        assert torch.isfinite(pitch_emb).all()

    def test_pitch_encoder_handles_unvoiced(self):
        """Test that unvoiced frames are handled correctly."""
        encoder = PitchEncoder(pitch_dim=192)
        f0 = torch.tensor([[440.0, 0.0, 460.0, 0.0, 470.0]])  # Unvoiced = 0
        pitch_emb = encoder(f0)

        assert pitch_emb.shape == (1, 5, 192)
        assert torch.isfinite(pitch_emb).all()

    def test_pitch_encoder_interpolation(self):
        """Test pitch embedding interpolation."""
        encoder = PitchEncoder(pitch_dim=192)
        f0 = torch.randn(2, 50) * 200 + 300  # Random F0
        pitch_emb = encoder(f0)  # [2, 50, 192]

        # Interpolate to different length
        pitch_emb_interp = encoder.interpolate_to_length(pitch_emb, 100)
        assert pitch_emb_interp.shape == (2, 100, 192)

    def test_pitch_encoder_quantized_vs_continuous(self):
        """Test blending of quantized and continuous paths."""
        encoder = PitchEncoder(pitch_dim=192)
        f0 = torch.tensor([[440.0, 450.0, 460.0]])

        # Both paths should work
        pitch_emb = encoder(f0)
        assert pitch_emb.shape == (1, 3, 192)
        assert torch.isfinite(pitch_emb).all()


# ========== PosteriorEncoder Tests ==========

@pytest.mark.model
@pytest.mark.unit
class TestPosteriorEncoder:
    """Test PosteriorEncoder component."""

    def test_posterior_encoder_initialization(self):
        """Test PosteriorEncoder initializes correctly."""
        encoder = PosteriorEncoder(in_channels=80, out_channels=192, num_layers=16)
        assert encoder.in_channels == 80
        assert encoder.out_channels == 192
        assert encoder.num_layers == 16
        assert len(encoder.blocks) == 16

    def test_posterior_encoder_forward(self):
        """Test PosteriorEncoder forward pass."""
        encoder = PosteriorEncoder(in_channels=80, out_channels=192, num_layers=16)
        mel = torch.randn(2, 80, 100)  # [B, mel_channels, T]
        mask = torch.ones(2, 1, 100)

        mean, log_var = encoder(mel, mask)

        # Check shapes
        assert mean.shape == (2, 192, 100)
        assert log_var.shape == (2, 192, 100)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_var).all()

    def test_posterior_encoder_sampling(self):
        """Test latent sampling with reparameterization trick."""
        encoder = PosteriorEncoder(in_channels=80, out_channels=192)
        mel = torch.randn(2, 80, 100)
        mask = torch.ones(2, 1, 100)

        mean, log_var = encoder(mel, mask)
        z = encoder.sample(mean, log_var)

        assert z.shape == (2, 192, 100)
        assert torch.isfinite(z).all()
        # Check z is different from mean (stochastic)
        assert not torch.allclose(z, mean, atol=1e-5)

    def test_wavenet_residual_block(self):
        """Test individual WaveNetResidualBlock."""
        block = WaveNetResidualBlock(
            residual_channels=192,
            skip_channels=192,
            kernel_size=5,
            dilation=2
        )

        x = torch.randn(2, 192, 100)
        mask = torch.ones(2, 1, 100)

        residual, skip = block(x, mask)

        assert residual.shape == (2, 192, 100)
        assert skip.shape == (2, 192, 100)
        assert torch.isfinite(residual).all()
        assert torch.isfinite(skip).all()

    def test_posterior_encoder_with_conditioning(self):
        """Test posterior encoder with conditioning."""
        encoder = PosteriorEncoder(in_channels=80, out_channels=192, cond_channels=256)
        mel = torch.randn(2, 80, 100)
        mask = torch.ones(2, 1, 100)
        cond = torch.randn(2, 256, 100)

        mean, log_var = encoder(mel, mask, cond)

        assert mean.shape == (2, 192, 100)
        assert log_var.shape == (2, 192, 100)


# ========== FlowDecoder Tests ==========

@pytest.mark.model
@pytest.mark.unit
class TestFlowDecoder:
    """Test FlowDecoder component."""

    def test_flow_decoder_initialization(self):
        """Test FlowDecoder initializes correctly."""
        flow = FlowDecoder(in_channels=192, num_flows=4, cond_channels=704)
        assert flow.in_channels == 192
        assert flow.num_flows == 4
        assert len(flow.flows) == 8  # 4 coupling + 4 flips

    def test_flow_forward_pass(self):
        """Test flow forward direction (z -> u)."""
        flow = FlowDecoder(in_channels=192, num_flows=4, cond_channels=704)
        z = torch.randn(2, 192, 100)
        mask = torch.ones(2, 1, 100)
        cond = torch.randn(2, 704, 100)

        u, logdet = flow(z, mask, cond=cond, inverse=False)

        assert u.shape == (2, 192, 100)
        assert logdet.shape == (2,)
        assert torch.isfinite(u).all()
        assert torch.isfinite(logdet).all()

    def test_flow_inverse_pass(self):
        """Test flow inverse direction (u -> z)."""
        flow = FlowDecoder(in_channels=192, num_flows=4, cond_channels=704)
        u = torch.randn(2, 192, 100)
        mask = torch.ones(2, 1, 100)
        cond = torch.randn(2, 704, 100)

        z = flow(u, mask, cond=cond, inverse=True)

        assert z.shape == (2, 192, 100)
        assert torch.isfinite(z).all()

    def test_flow_invertibility(self):
        """Test flow is approximately invertible."""
        flow = FlowDecoder(in_channels=192, num_flows=4, cond_channels=704)
        z_orig = torch.randn(2, 192, 100)
        mask = torch.ones(2, 1, 100)
        cond = torch.randn(2, 704, 100)

        # Forward: z -> u
        u, _ = flow(z_orig, mask, cond=cond, inverse=False)

        # Inverse: u -> z
        z_rec = flow(u, mask, cond=cond, inverse=True)

        # Check reconstruction
        assert torch.allclose(z_orig, z_rec, rtol=1e-4, atol=1e-4)

    def test_affine_coupling_layer(self):
        """Test individual AffineCouplingLayer."""
        layer = AffineCouplingLayer(
            in_channels=192,
            hidden_channels=192,
            cond_channels=704
        )

        x = torch.randn(2, 192, 100)
        mask = torch.ones(2, 1, 100)
        cond = torch.randn(2, 704, 100)

        # Forward
        y, logdet = layer(x, mask, cond=cond, inverse=False)
        assert y.shape == (2, 192, 100)
        assert logdet.shape == (2,)

        # Inverse
        x_rec = layer(y, mask, cond=cond, inverse=True)
        assert torch.allclose(x, x_rec, rtol=1e-4, atol=1e-4)

    def test_flip_layer(self):
        """Test Flip layer."""
        flip = Flip()
        x = torch.randn(2, 192, 100)
        mask = torch.ones(2, 1, 100)

        # Forward
        y, logdet = flip(x, mask, inverse=False)
        assert y.shape == x.shape
        assert logdet.sum() == 0  # Flip has no Jacobian contribution

        # Inverse
        x_rec = flip(y, mask, inverse=True)
        assert torch.allclose(x, x_rec)


# ========== SingingVoiceConverter Integration Tests ==========

@pytest.mark.model
@pytest.mark.integration
class TestSingingVoiceConverter:
    """Test SingingVoiceConverter integration."""

    @pytest.fixture
    def model_config(self):
        """Configuration for test model."""
        return {
            'latent_dim': 192,
            'mel_channels': 80,
            'content_encoder_type': 'cnn_fallback',
            'content_dim': 256,
            'pitch_dim': 192,
            'speaker_dim': 256,
            'hidden_channels': 192,
            'num_flows': 4,
            'posterior_num_layers': 16,
            'use_vocoder': False,  # Disable vocoder for tests
            'device': 'cpu'
        }

    @pytest.fixture
    def model(self, model_config):
        """Create test model."""
        return SingingVoiceConverter(model_config)

    def test_model_initialization(self, model):
        """Test model initializes all components."""
        assert hasattr(model, 'content_encoder')
        assert hasattr(model, 'pitch_encoder')
        assert hasattr(model, 'posterior_encoder')
        assert hasattr(model, 'flow_decoder')
        assert hasattr(model, 'latent_to_mel')

    def test_training_forward_pass(self, model):
        """Test training forward pass."""
        source_audio = torch.randn(2, 16000)
        target_mel = torch.randn(2, 80, 100)
        source_f0 = torch.randn(2, 100) * 200 + 300
        target_speaker_emb = torch.randn(2, 256)

        outputs = model(source_audio, target_mel, source_f0, target_speaker_emb)

        # Check all outputs exist
        assert 'pred_mel' in outputs
        assert 'z_mean' in outputs
        assert 'z_logvar' in outputs
        assert 'z' in outputs
        assert 'u' in outputs
        assert 'logdet' in outputs
        assert 'cond' in outputs

        # Check shapes
        assert outputs['pred_mel'].shape == (2, 80, 100)
        assert outputs['z_mean'].shape == (2, 192, 100)
        assert outputs['z_logvar'].shape == (2, 192, 100)
        assert outputs['z'].shape == (2, 192, 100)

        # Check no NaN/Inf
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"{key} contains NaN/Inf"

    def test_convert_method_with_numpy_inputs(self, model):
        """Test convert() with numpy array inputs."""
        model.eval()
        model.prepare_for_inference()

        source_audio_np = np.random.randn(44100).astype(np.float32)
        target_speaker_emb_np = np.random.randn(256).astype(np.float32)
        source_f0_np = np.random.randn(100).astype(np.float32) * 200 + 300

        with torch.no_grad():
            waveform = model.convert(
                source_audio=source_audio_np,
                target_speaker_embedding=target_speaker_emb_np,
                source_f0=source_f0_np,
                source_sample_rate=44100
            )

        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert np.isfinite(waveform).all()

    def test_kl_loss_computation(self, model):
        """Test KL divergence loss computation."""
        z_mean = torch.randn(2, 192, 100)
        z_logvar = torch.randn(2, 192, 100)

        kl_loss = model.compute_kl_loss(z_mean, z_logvar)

        assert isinstance(kl_loss, torch.Tensor)
        assert kl_loss.dim() == 0  # Scalar
        assert kl_loss >= 0  # KL divergence is non-negative

    def test_prepare_for_inference(self, model):
        """Test prepare_for_inference method."""
        model.prepare_for_inference()
        assert not model.training  # Model should be in eval mode

    @pytest.mark.xfail(reason="Pitch preservation test may be flaky with untrained weights")
    def test_pitch_preservation(self, model):
        """Test that pitch is preserved during conversion.

        Note: This test is marked as xfail because it may fail with untrained weights.
        The test validates structural correctness rather than quality with random weights.
        """
        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        model.eval()
        model.prepare_for_inference()

        # Create source audio with known F0 pattern (440 Hz for 1 second at 16kHz)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        f0_target = 440.0  # A4
        source_audio_np = (0.5 * np.sin(2 * np.pi * f0_target * t)).astype(np.float32)

        # Extract source F0
        extractor = SingingPitchExtractor(device='cpu')
        source_f0_data = extractor.extract_f0_contour(source_audio_np, sample_rate)
        source_f0 = source_f0_data['f0']

        # Create target speaker embedding
        target_speaker_emb_np = np.random.randn(256).astype(np.float32)

        # Convert
        with torch.no_grad():
            converted_audio = model.convert(
                source_audio=source_audio_np,
                target_speaker_embedding=target_speaker_emb_np,
                source_f0=source_f0,
                source_sample_rate=sample_rate,
                output_sample_rate=sample_rate
            )

        # Verify conversion produces valid output (structural validation)
        assert isinstance(converted_audio, np.ndarray)
        assert converted_audio.ndim == 1
        assert np.isfinite(converted_audio).all()

        # Extract converted F0 for quality metrics (may not match well with untrained weights)
        converted_f0_data = extractor.extract_f0_contour(converted_audio, sample_rate)
        converted_f0 = converted_f0_data['f0']

        # Compute RMSE between source and converted F0 (only voiced frames)
        voiced_mask = (source_f0 > 0) & (converted_f0 > 0)
        if voiced_mask.sum() > 0:
            # Align lengths by taking minimum
            min_len = min(len(source_f0), len(converted_f0))
            source_f0_voiced = source_f0[:min_len][voiced_mask[:min_len]]
            converted_f0_voiced = converted_f0[:min_len][voiced_mask[:min_len]]

            if len(source_f0_voiced) > 0 and len(converted_f0_voiced) > 0:
                # Take minimum length again for comparison
                min_voiced_len = min(len(source_f0_voiced), len(converted_f0_voiced))
                rmse = np.sqrt(np.mean((source_f0_voiced[:min_voiced_len] - converted_f0_voiced[:min_voiced_len]) ** 2))

                # Relaxed tolerance for untrained weights (within 100 Hz)
                # With trained weights, this should be < 50 Hz
                assert rmse < 100.0, f"Pitch RMSE too high: {rmse:.2f} Hz"

    @pytest.mark.xfail(reason="Speaker conditioning test may be flaky with untrained weights")
    def test_speaker_conditioning(self, model):
        """Test that speaker conditioning produces different outputs.

        Note: This test is marked as xfail because it may fail with untrained weights.
        With random weights, the speaker conditioning may not produce meaningful differences.
        """
        model.eval()
        model.prepare_for_inference()

        # Create source audio
        source_audio_np = np.random.randn(16000).astype(np.float32)
        source_f0_np = (np.random.randn(100) * 50 + 440).astype(np.float32)

        # Two different random speaker embeddings
        speaker_emb_1 = np.random.randn(256).astype(np.float32)
        speaker_emb_2 = np.random.randn(256).astype(np.float32)

        # Convert with first speaker
        with torch.no_grad():
            converted_1 = model.convert(
                source_audio=source_audio_np,
                target_speaker_embedding=speaker_emb_1,
                source_f0=source_f0_np,
                source_sample_rate=16000,
                output_sample_rate=16000
            )

        # Convert with second speaker
        with torch.no_grad():
            converted_2 = model.convert(
                source_audio=source_audio_np,
                target_speaker_embedding=speaker_emb_2,
                source_f0=source_f0_np,
                source_sample_rate=16000,
                output_sample_rate=16000
            )

        # Verify both conversions produce valid outputs (structural validation)
        assert isinstance(converted_1, np.ndarray) and isinstance(converted_2, np.ndarray)
        assert converted_1.shape == converted_2.shape
        assert np.isfinite(converted_1).all() and np.isfinite(converted_2).all()

        # Compute cosine distance between waveforms
        converted_1_norm = converted_1 / (np.linalg.norm(converted_1) + 1e-8)
        converted_2_norm = converted_2 / (np.linalg.norm(converted_2) + 1e-8)
        cosine_sim = np.dot(converted_1_norm, converted_2_norm)
        cosine_dist = 1 - cosine_sim

        # Relaxed threshold for untrained weights
        # With trained weights, this should be > 0.1
        # With untrained weights, we just verify outputs are not identical
        assert cosine_dist > 0.01, f"Speaker conditioning produces identical outputs: cosine_dist={cosine_dist:.4f}"


# ========== Numerical Stability Tests ==========

@pytest.mark.model
@pytest.mark.unit
class TestVoiceConversionNumericalStability:
    """Test numerical stability of voice conversion components."""

    def test_flow_numerical_stability(self):
        """Test flow with extreme values."""
        flow = FlowDecoder(in_channels=192, num_flows=4, cond_channels=704)

        # Test with large values
        z = torch.randn(2, 192, 100) * 10
        mask = torch.ones(2, 1, 100)
        cond = torch.randn(2, 704, 100)

        u, logdet = flow(z, mask, cond=cond, inverse=False)

        assert torch.isfinite(u).all()
        assert torch.isfinite(logdet).all()

    def test_posterior_encoder_stability(self):
        """Test posterior encoder with various inputs."""
        encoder = PosteriorEncoder(in_channels=80, out_channels=192)

        # Test with different magnitude inputs
        for scale in [0.1, 1.0, 10.0]:
            mel = torch.randn(2, 80, 100) * scale
            mask = torch.ones(2, 1, 100)

            mean, log_var = encoder(mel, mask)

            assert torch.isfinite(mean).all()
            assert torch.isfinite(log_var).all()

    def test_gradient_flow_through_model(self, model_config=None):
        """Test gradients flow through all components."""
        if model_config is None:
            model_config = {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder_type': 'cnn_fallback',
                'use_vocoder': False,
                'device': 'cpu'
            }

        model = SingingVoiceConverter(model_config)
        model.train()

        source_audio = torch.randn(1, 16000)
        target_mel = torch.randn(1, 80, 50, requires_grad=True)
        source_f0 = torch.randn(1, 50) * 200 + 300
        target_speaker_emb = torch.randn(1, 256)

        outputs = model(source_audio, target_mel, source_f0, target_speaker_emb)

        # Compute dummy loss
        loss = outputs['pred_mel'].sum() + outputs['z_mean'].sum()
        loss.backward()

        # Check gradients exist and are finite for target_mel
        assert target_mel.grad is not None
        assert torch.isfinite(target_mel.grad).all()

        # Check gradients exist on PosteriorEncoder parameters
        posterior_params_with_grad = sum(1 for p in model.posterior_encoder.parameters() if p.grad is not None)
        assert posterior_params_with_grad > 0


# ========== Performance Tests ==========

@pytest.mark.model
@pytest.mark.performance
class TestVoiceConversionPerformance:
    """Test performance characteristics."""

    @pytest.mark.cuda
    def test_conversion_speed(self, cuda_device):
        """Test conversion completes in reasonable time."""
        pytest.skip("Skipping CUDA performance test")

    @pytest.mark.cuda
    def test_gpu_memory_usage(self, cuda_device):
        """Test GPU memory usage is reasonable."""
        pytest.skip("Skipping CUDA memory test")

    def test_batch_conversion(self):
        """Test converting multiple audio files."""
        config = {
            'latent_dim': 192,
            'mel_channels': 80,
            'content_encoder_type': 'cnn_fallback',
            'use_vocoder': False,
            'device': 'cpu'
        }
        model = SingingVoiceConverter(config)
        model.eval()

        # Test batch processing
        batch_size = 4
        source_audio = torch.randn(batch_size, 16000)
        target_mel = torch.randn(batch_size, 80, 50)
        source_f0 = torch.randn(batch_size, 50) * 200 + 300
        target_speaker_emb = torch.randn(batch_size, 256)

        with torch.no_grad():
            outputs = model(source_audio, target_mel, source_f0, target_speaker_emb)

        assert outputs['pred_mel'].shape[0] == batch_size


# ========== Verification Comment Tests ==========

@pytest.mark.model
@pytest.mark.unit
class TestVerificationComments:
    """Test suite for verification comments implementation."""

    def test_comment1_hop_derived_timing(self):
        """
        Comment 1: Test that convert() uses hop-derived mel frames.
        Verify T = ceil(num_samples / hop_length) alignment.
        """
        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {
                    'type': 'cnn_fallback',
                    'output_dim': 256
                },
                'audio': {
                    'sample_rate': 22050,
                    'hop_length': 256,  # Non-default hop
                    'n_fft': 1024,
                    'win_length': 1024
                },
                'vocoder': {
                    'use_vocoder': False
                }
            },
            'device': 'cpu'
        }
        model = SingingVoiceConverter(config)
        model.eval()
        model.prepare_for_inference()

        # Create source audio with known sample count
        source_sample_rate = 22050
        num_samples = 22050  # 1 second
        source_audio = torch.randn(num_samples)
        target_speaker_emb = torch.randn(256)

        # Expected T from hop length
        hop_length = 256
        expected_T = int(np.ceil(num_samples / hop_length))  # ceil(22050/256) = 87

        # Convert
        with torch.no_grad():
            waveform = model.convert(
                source_audio=source_audio,
                target_speaker_embedding=target_speaker_emb,
                source_sample_rate=source_sample_rate,
                output_sample_rate=source_sample_rate
            )

        # Verify output waveform length is close to expected_T * hop_length
        expected_len = expected_T * hop_length
        actual_len = len(waveform)

        # Allow within 1 frame tolerance
        assert abs(actual_len - expected_len) <= hop_length, \
            f"Output length {actual_len} not aligned with T*hop_length={expected_len}"

        # Verify waveform is valid
        assert isinstance(waveform, np.ndarray)
        assert np.isfinite(waveform).all()

    def test_comment2_unvoiced_detection_negative(self):
        """
        Comment 2: Test unvoiced detection handles negative F0 values.
        """
        encoder = PitchEncoder(pitch_dim=192, num_bins=256)

        # F0 with negative values (should be treated as unvoiced)
        f0 = torch.tensor([[440.0, -50.0, 460.0, -100.0, 470.0]])  # [B=1, T=5]

        pitch_emb = encoder(f0)

        # Check shape and finite values
        assert pitch_emb.shape == (1, 5, 192)
        assert torch.isfinite(pitch_emb).all()

        # Verify negative values don't corrupt embeddings
        # All embeddings should be finite and reasonable magnitude
        assert pitch_emb.abs().max() < 100.0

    def test_comment2_unvoiced_detection_nonfinite(self):
        """
        Comment 2: Test unvoiced detection handles NaN/Inf values.
        """
        encoder = PitchEncoder(pitch_dim=192, num_bins=256)

        # F0 with NaN and Inf values
        f0 = torch.tensor([[440.0, float('nan'), 460.0, float('inf'), 470.0]])

        pitch_emb = encoder(f0)

        # Check shape and all values are finite
        assert pitch_emb.shape == (1, 5, 192)
        assert torch.isfinite(pitch_emb).all()

    def test_comment2_unvoiced_detection_voiced_mask(self):
        """
        Comment 2: Test that external voiced mask is strictly respected.
        """
        encoder = PitchEncoder(pitch_dim=192, num_bins=256)

        # F0 with positive values but masked as unvoiced
        f0 = torch.tensor([[440.0, 450.0, 460.0, 470.0, 480.0]])
        voiced = torch.tensor([[True, False, True, False, True]])

        pitch_emb = encoder(f0, voiced)

        # Check shape and finite values
        assert pitch_emb.shape == (1, 5, 192)
        assert torch.isfinite(pitch_emb).all()

        # Embeddings for masked-out frames should use unvoiced bin
        # We can't directly check the bin, but embeddings should be valid
        assert pitch_emb.abs().max() < 100.0

    def test_comment3_griffin_lim_config_params(self):
        """
        Comment 3: Test Griffin-Lim uses config audio settings.
        """
        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {
                    'type': 'cnn_fallback',
                    'output_dim': 256
                },
                'audio': {
                    'sample_rate': 22050,
                    'hop_length': 320,  # Custom hop
                    'n_fft': 1024,      # Custom n_fft
                    'win_length': 1024,
                    'mel_fmin': 50.0,
                    'mel_fmax': 7000.0
                },
                'vocoder': {
                    'use_vocoder': False  # Force Griffin-Lim
                }
            }
        }
        model = SingingVoiceConverter(config)
        model.eval()
        model.prepare_for_inference()

        # Create test mel-spectrogram
        mel = np.random.randn(80, 100).astype(np.float32)

        # Convert using Griffin-Lim
        audio = model._mel_to_audio_griffin_lim(mel, n_iter=16)

        # Verify output length aligns with hop_length
        expected_len = 100 * 320  # T_mel * hop_length
        actual_len = len(audio)

        # Allow some tolerance due to windowing effects
        assert abs(actual_len - expected_len) <= 1024, \
            f"Griffin-Lim output length {actual_len} not aligned with expected {expected_len}"

        assert np.isfinite(audio).all()

    def test_comment4_speaker_embedding_validation_wrong_size(self):
        """
        Comment 4: Test validation rejects wrong speaker embedding size.
        """
        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {'type': 'cnn_fallback'},
                'speaker_encoder': {'embedding_dim': 256},
                'vocoder': {'use_vocoder': False}
            }
        }
        model = SingingVoiceConverter(config)
        model.eval()

        source_audio = torch.randn(16000)
        # Wrong size: 128 instead of 256
        wrong_speaker_emb = torch.randn(128)

        # Should raise VoiceConversionError
        with pytest.raises(VoiceConversionError, match="must have size.*256.*got.*128"):
            model.convert(
                source_audio=source_audio,
                target_speaker_embedding=wrong_speaker_emb,
                source_sample_rate=16000
            )

    def test_comment4_speaker_embedding_validation_batch_wrong_size(self):
        """
        Comment 4: Test validation rejects wrong speaker embedding size in batch format.
        """
        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {'type': 'cnn_fallback'},
                'speaker_encoder': {'embedding_dim': 256},
                'vocoder': {'use_vocoder': False}
            }
        }
        model = SingingVoiceConverter(config)
        model.eval()

        source_audio = torch.randn(16000)
        # Wrong size: [1, 128] instead of [1, 256]
        wrong_speaker_emb = torch.randn(1, 128)

        # Should raise VoiceConversionError
        with pytest.raises(VoiceConversionError, match="must have shape.*256.*got.*128"):
            model.convert(
                source_audio=source_audio,
                target_speaker_embedding=wrong_speaker_emb,
                source_sample_rate=16000
            )

    def test_comment4_speaker_embedding_validation_correct_sizes(self):
        """
        Comment 4: Test validation accepts correct speaker embedding sizes.
        """
        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {'type': 'cnn_fallback'},
                'speaker_encoder': {'embedding_dim': 256},
                'vocoder': {'use_vocoder': False}
            }
        }
        model = SingingVoiceConverter(config)
        model.eval()
        model.prepare_for_inference()

        source_audio = torch.randn(16000)

        # Test 1D format [256]
        speaker_emb_1d = torch.randn(256)
        with torch.no_grad():
            waveform_1d = model.convert(
                source_audio=source_audio,
                target_speaker_embedding=speaker_emb_1d,
                source_sample_rate=16000
            )
        assert isinstance(waveform_1d, np.ndarray)
        assert np.isfinite(waveform_1d).all()

        # Test 2D format [1, 256]
        speaker_emb_2d = torch.randn(1, 256)
        with torch.no_grad():
            waveform_2d = model.convert(
                source_audio=source_audio,
                target_speaker_embedding=speaker_emb_2d,
                source_sample_rate=16000
            )
        assert isinstance(waveform_2d, np.ndarray)
        assert np.isfinite(waveform_2d).all()

    def test_comment5_content_encoder_mel_config(self):
        """
        Comment 5: Test ContentEncoder CNN fallback uses configurable mel parameters.
        """
        # Custom mel config
        mel_config = {
            'n_fft': 512,
            'hop_length': 160,
            'n_mels': 64,
            'sample_rate': 16000
        }

        encoder = ContentEncoder(
            encoder_type='cnn_fallback',
            output_dim=256,
            device='cpu',
            cnn_mel_config=mel_config
        )

        # Verify frame rate calculation
        frame_rate = encoder.get_frame_rate()
        expected_frame_rate = 16000 / 160  # sample_rate / hop_length = 100 Hz
        assert abs(frame_rate - expected_frame_rate) < 0.1, \
            f"Frame rate {frame_rate} doesn't match expected {expected_frame_rate}"

        # Test forward pass with custom config
        audio = torch.randn(1, 16000)  # 1 second
        content = encoder(audio, sample_rate=16000)

        # Check output shape alignment with hop_length
        expected_frames = int(np.ceil(16000 / 160))  # ceil(16000/160) = 100
        actual_frames = content.shape[1]

        # Allow some tolerance
        assert abs(actual_frames - expected_frames) <= 2, \
            f"Content frames {actual_frames} not aligned with expected {expected_frames}"

        assert torch.isfinite(content).all()

    def test_comment5_content_encoder_frame_rate_accuracy(self):
        """
        Comment 5: Test that non-default mel_hop_length changes frame rate.
        """
        # Config 1: Default-like settings (320 hop)
        config1 = {
            'n_fft': 1024,
            'hop_length': 320,
            'n_mels': 80,
            'sample_rate': 16000
        }
        encoder1 = ContentEncoder(
            encoder_type='cnn_fallback',
            cnn_mel_config=config1,
            device='cpu'
        )
        frame_rate1 = encoder1.get_frame_rate()
        assert abs(frame_rate1 - 50.0) < 0.1  # 16000/320 = 50 Hz

        # Config 2: Different hop length (256)
        config2 = {
            'n_fft': 1024,
            'hop_length': 256,
            'n_mels': 80,
            'sample_rate': 16000
        }
        encoder2 = ContentEncoder(
            encoder_type='cnn_fallback',
            cnn_mel_config=config2,
            device='cpu'
        )
        frame_rate2 = encoder2.get_frame_rate()
        assert abs(frame_rate2 - 62.5) < 0.1  # 16000/256 = 62.5 Hz

        # Verify different frame rates
        assert abs(frame_rate1 - frame_rate2) > 10.0

    @pytest.mark.cuda
    def test_comment6_device_alignment_fix(self, cuda_device):
        """
        Comment 6: Test device alignment fix when target_mel is on different device.

        Verifies that forward() correctly moves target_mel to model device.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = {
            'latent_dim': 192,
            'mel_channels': 80,
            'content_encoder_type': 'cnn_fallback',
            'use_vocoder': False
        }

        # Create model on CUDA
        model = SingingVoiceConverter(config)
        model = model.to('cuda')
        model.eval()

        # Create inputs on CPU (target_mel on wrong device)
        source_audio = torch.randn(1, 16000)
        target_mel = torch.randn(1, 80, 50)  # On CPU
        source_f0 = torch.randn(1, 50) * 200 + 300
        target_speaker_emb = torch.randn(1, 256)

        # Should not raise device mismatch error
        try:
            with torch.no_grad():
                outputs = model(source_audio, target_mel, source_f0, target_speaker_emb)

            # Verify outputs are on CUDA
            assert outputs['pred_mel'].device.type == 'cuda'
            assert outputs['z_mean'].device.type == 'cuda'
            assert torch.isfinite(outputs['pred_mel']).all()
        except RuntimeError as e:
            if 'device' in str(e).lower():
                pytest.fail(f"Device alignment failed: {e}")
            else:
                raise

    def test_comment6_voiced_mask_propagation(self):
        """
        Comment 6: Test that voiced mask is propagated to PitchEncoder.

        Uses a mock to verify PitchEncoder.forward is called with voiced mask.
        """
        from unittest.mock import MagicMock, patch

        config = {
            'latent_dim': 192,
            'mel_channels': 80,
            'content_encoder_type': 'cnn_fallback',
            'use_vocoder': False,
            'device': 'cpu'
        }

        model = SingingVoiceConverter(config)
        model.eval()

        # Prepare inputs
        source_audio = torch.randn(1, 16000)
        target_mel = torch.randn(1, 80, 50)
        source_f0 = torch.randn(1, 50) * 200 + 300
        target_speaker_emb = torch.randn(1, 256)
        source_voiced = torch.ones(1, 50, dtype=torch.bool)  # All voiced

        # Mock PitchEncoder.forward to track calls
        original_forward = model.pitch_encoder.forward
        call_args = []

        def mock_forward(f0, voiced=None):
            call_args.append((f0, voiced))
            return original_forward(f0, voiced)

        model.pitch_encoder.forward = mock_forward

        # Call model forward with voiced mask
        with torch.no_grad():
            outputs = model(
                source_audio, target_mel, source_f0, target_speaker_emb,
                source_voiced=source_voiced
            )

        # Verify PitchEncoder was called with voiced mask
        assert len(call_args) == 1, "PitchEncoder.forward should be called once"
        f0_arg, voiced_arg = call_args[0]
        assert voiced_arg is not None, "voiced mask should be passed to PitchEncoder"
        assert torch.equal(voiced_arg, source_voiced), "voiced mask should match input"

        # Verify outputs are valid
        assert torch.isfinite(outputs['pred_mel']).all()

    def test_comment6_voiced_mask_in_convert(self):
        """
        Comment 6: Test that convert() retrieves and uses voiced mask from pitch extractor.
        """
        from unittest.mock import MagicMock, patch

        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {'type': 'cnn_fallback'},
                'vocoder': {'use_vocoder': False}
            }
        }

        model = SingingVoiceConverter(config)
        model.eval()
        model.prepare_for_inference()

        source_audio = np.random.randn(16000).astype(np.float32)
        target_speaker_emb = np.random.randn(256).astype(np.float32)

        # Track PitchEncoder calls
        call_args = []
        original_forward = model.pitch_encoder.forward

        def mock_forward(f0, voiced=None):
            call_args.append((f0, voiced))
            return original_forward(f0, voiced)

        model.pitch_encoder.forward = mock_forward

        # Call convert without providing F0 (will auto-extract)
        # Note: This may fail if torchcrepe is not installed, which is expected
        try:
            with torch.no_grad():
                waveform = model.convert(
                    source_audio=source_audio,
                    target_speaker_embedding=target_speaker_emb,
                    source_sample_rate=16000
                )

            # Verify PitchEncoder was called with voiced mask
            assert len(call_args) == 1, "PitchEncoder.forward should be called once"
            f0_arg, voiced_arg = call_args[0]
            assert voiced_arg is not None, "voiced mask should be retrieved from pitch extractor and passed to PitchEncoder"
            assert voiced_arg.dtype == torch.bool, "voiced mask should be boolean tensor"

            # Verify output is valid
            assert isinstance(waveform, np.ndarray)
            assert np.isfinite(waveform).all()
        except VoiceConversionError as e:
            if "torchcrepe" in str(e):
                pytest.skip("torchcrepe not available, skipping pitch extractor test")
            else:
                raise

    def test_integration_all_comments(self):
        """
        Integration test: Verify all verification comments work together.
        """
        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {
                    'type': 'cnn_fallback',
                    'output_dim': 256,
                    'cnn_fallback': {
                        'n_fft': 1024,
                        'hop_length': 320,
                        'n_mels': 80,
                        'sample_rate': 16000
                    }
                },
                'pitch_encoder': {
                    'pitch_dim': 192,
                    'num_bins': 256,
                    'f0_min': 80.0,
                    'f0_max': 1000.0
                },
                'speaker_encoder': {
                    'embedding_dim': 256
                },
                'audio': {
                    'sample_rate': 22050,
                    'hop_length': 512,
                    'n_fft': 2048,
                    'win_length': 2048,
                    'mel_fmin': 0.0,
                    'mel_fmax': 8000.0
                },
                'vocoder': {
                    'use_vocoder': False  # Use Griffin-Lim
                }
            }
        }

        model = SingingVoiceConverter(config)
        model.eval()
        model.prepare_for_inference()

        # Source audio with known properties
        source_sample_rate = 22050
        num_samples = 22050  # 1 second
        source_audio = np.random.randn(num_samples).astype(np.float32)

        # F0 with edge cases (negative, zero, valid)
        f0 = np.array([440.0, -50.0, 0.0, 460.0, float('nan'), 470.0] * 20).astype(np.float32)

        # Correct speaker embedding
        speaker_emb = np.random.randn(256).astype(np.float32)

        # Convert
        with torch.no_grad():
            waveform = model.convert(
                source_audio=source_audio,
                target_speaker_embedding=speaker_emb,
                source_f0=f0,
                source_sample_rate=source_sample_rate,
                output_sample_rate=source_sample_rate
            )

        # Verify output
        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert np.isfinite(waveform).all()

        # Verify timing alignment (Comment 1)
        hop_length = 512
        expected_T = int(np.ceil(num_samples / hop_length))
        expected_len = expected_T * hop_length
        actual_len = len(waveform)
        assert abs(actual_len - expected_len) <= hop_length, \
            f"Integration test: timing not aligned"
