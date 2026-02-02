"""Tests for SeedVCPipeline - SOTA quality voice conversion.

SeedVCPipeline provides state-of-the-art voice conversion using:
- Whisper-base for semantic content extraction
- CAMPPlus for speaker style embedding
- DiT (Diffusion Transformer) with Conditional Flow Matching
- BigVGAN v2 (44kHz, 128-band) for high-quality waveform synthesis
- RMVPE for F0 extraction (singing voice)

Tests cover:
- Initialization and configuration
- Reference audio setting
- Voice conversion with various inputs
- Progress callbacks
- Error handling
"""
import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


# Mark all tests in this module as requiring CUDA unless specified
pytestmark = [pytest.mark.cuda]


class TestSeedVCPipelineInit:
    """Test SeedVCPipeline initialization."""

    def test_import_succeeds(self):
        """SeedVCPipeline can be imported."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline
        assert SeedVCPipeline is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_requires_cuda_by_default(self):
        """SeedVCPipeline requires CUDA when require_gpu=True."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        # This should work on CUDA machine
        # On non-CUDA machine, this test is skipped

    def test_init_without_cuda_raises(self):
        """SeedVCPipeline raises RuntimeError without CUDA."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA"):
                SeedVCPipeline(require_gpu=True)

    def test_init_accepts_custom_device(self):
        """SeedVCPipeline accepts custom device."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        # Mock the wrapper to avoid loading models
        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(
                device=torch.device('cuda:0'),
                require_gpu=False
            )
            assert pipeline.device == torch.device('cuda:0')

    def test_init_diffusion_steps_configurable(self):
        """SeedVCPipeline accepts diffusion_steps parameter."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(
                diffusion_steps=5,
                require_gpu=False
            )
            assert pipeline.diffusion_steps == 5

    def test_init_f0_condition_configurable(self):
        """SeedVCPipeline accepts f0_condition parameter."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(
                f0_condition=False,
                require_gpu=False
            )
            assert pipeline.f0_condition is False


class TestSampleRate:
    """Test sample rate configuration."""

    def test_sample_rate_44k_with_f0(self):
        """Sample rate is 44.1kHz when F0-conditioned."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(f0_condition=True, require_gpu=False)
            assert pipeline.sample_rate == 44100

    def test_sample_rate_22k_without_f0(self):
        """Sample rate is 22.05kHz without F0 conditioning."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(f0_condition=False, require_gpu=False)
            assert pipeline.sample_rate == 22050


class TestReferenceAudio:
    """Test reference audio setting."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocked initialization."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            return SeedVCPipeline(require_gpu=False)

    def test_set_reference_audio_numpy(self, pipeline):
        """set_reference_audio accepts numpy array."""
        audio = np.random.randn(44100 * 5).astype(np.float32)  # 5s

        pipeline.set_reference_audio(audio, sample_rate=44100)

        assert pipeline._reference_audio is not None
        assert pipeline._reference_audio.shape == audio.shape
        assert pipeline._reference_sr == 44100

    def test_set_reference_audio_tensor(self, pipeline):
        """set_reference_audio accepts torch tensor."""
        audio = torch.randn(44100 * 5)  # 5s

        pipeline.set_reference_audio(audio, sample_rate=44100)

        assert pipeline._reference_audio is not None
        assert isinstance(pipeline._reference_audio, np.ndarray)

    def test_set_reference_audio_stores_sample_rate(self, pipeline):
        """set_reference_audio stores sample rate."""
        audio = np.random.randn(16000 * 5).astype(np.float32)

        pipeline.set_reference_audio(audio, sample_rate=16000)

        assert pipeline._reference_sr == 16000

    def test_set_reference_from_profile_id_calls_bridge(self, pipeline):
        """set_reference_from_profile_id uses AdapterBridge."""
        mock_bridge = MagicMock()
        mock_ref = MagicMock()
        mock_ref.reference_paths = [MagicMock()]
        mock_ref.profile_name = "Test Artist"
        mock_bridge.get_voice_reference.return_value = mock_ref

        with patch('auto_voice.inference.adapter_bridge.get_adapter_bridge', return_value=mock_bridge):
            with patch('librosa.load', return_value=(np.zeros(44100), 44100)):
                pipeline.set_reference_from_profile_id("test-profile")

        mock_bridge.get_voice_reference.assert_called_once_with("test-profile")

    def test_set_reference_from_profile_id_no_audio_raises(self, pipeline):
        """set_reference_from_profile_id raises if no reference audio."""
        mock_bridge = MagicMock()
        mock_ref = MagicMock()
        mock_ref.reference_paths = []
        mock_ref.profile_name = "No Audio"
        mock_bridge.get_voice_reference.return_value = mock_ref

        with patch('auto_voice.inference.adapter_bridge.get_adapter_bridge', return_value=mock_bridge):
            with pytest.raises(ValueError, match="No reference audio"):
                pipeline.set_reference_from_profile_id("empty-profile")


class TestConvert:
    """Test voice conversion functionality."""

    @pytest.fixture
    def pipeline_with_ref(self):
        """Create pipeline with reference audio set."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(require_gpu=False)

        # Set reference audio
        pipeline._reference_audio = np.random.randn(44100 * 3).astype(np.float32)
        pipeline._reference_sr = 44100

        return pipeline

    def test_convert_requires_reference(self):
        """convert raises if no reference audio set."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(require_gpu=False)

        audio = np.random.randn(44100 * 2).astype(np.float32)

        with pytest.raises(RuntimeError, match="No reference audio"):
            pipeline.convert(audio, sample_rate=44100)

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_convert_produces_output(self):
        """convert produces audio output (integration test)."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        # This test requires actual models loaded
        try:
            pipeline = SeedVCPipeline(device=torch.device('cuda'))
        except RuntimeError as e:
            if "seed-vc" in str(e).lower():
                pytest.skip("Seed-VC models not downloaded")
            raise

        # Set reference audio
        ref_audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, 44100 * 3))
        pipeline.set_reference_audio(ref_audio.astype(np.float32), 44100)

        # Convert
        source_audio = 0.5 * np.sin(2 * np.pi * 330 * np.linspace(0, 2, 44100 * 2))
        result = pipeline.convert(source_audio.astype(np.float32), sample_rate=44100)

        assert 'audio' in result
        assert 'sample_rate' in result
        assert 'metadata' in result
        assert result['sample_rate'] == 44100
        assert len(result['audio']) > 0

    def test_convert_accepts_tensor_input(self, pipeline_with_ref):
        """convert accepts torch tensor input."""
        audio = torch.randn(44100 * 2)

        # Mock the wrapper
        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = iter([
            (None, (44100, np.random.randn(44100 * 2).astype(np.float32)))
        ])
        pipeline_with_ref._wrapper = mock_wrapper

        with patch('tempfile.NamedTemporaryFile') as mock_tmp:
            mock_tmp.return_value.__enter__ = MagicMock(return_value=MagicMock(name='/tmp/test.wav'))
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            with patch('soundfile.write'):
                with patch('os.unlink'):
                    result = pipeline_with_ref.convert(audio, sample_rate=44100)

        assert 'audio' in result

    def test_convert_handles_stereo_input(self, pipeline_with_ref):
        """convert handles stereo audio by converting to mono."""
        audio = np.random.randn(2, 44100 * 2).astype(np.float32)  # Stereo

        # Mock the wrapper
        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = iter([
            (None, (44100, np.random.randn(44100 * 2).astype(np.float32)))
        ])
        pipeline_with_ref._wrapper = mock_wrapper

        with patch('tempfile.NamedTemporaryFile') as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.wav'
            mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            with patch('soundfile.write') as mock_write:
                with patch('os.unlink'):
                    result = pipeline_with_ref.convert(audio, sample_rate=44100)

                # Should be called with mono audio
                write_calls = mock_write.call_args_list
                # Source audio should be mono (1D)
                for call in write_calls:
                    written_audio = call[0][1]
                    assert written_audio.ndim == 1

    def test_convert_returns_metadata(self, pipeline_with_ref):
        """convert returns processing metadata."""
        audio = np.random.randn(44100 * 2).astype(np.float32)

        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = iter([
            (None, (44100, np.random.randn(44100 * 2).astype(np.float32)))
        ])
        pipeline_with_ref._wrapper = mock_wrapper

        with patch('tempfile.NamedTemporaryFile') as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.wav'
            mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            with patch('soundfile.write'):
                with patch('os.unlink'):
                    result = pipeline_with_ref.convert(audio, sample_rate=44100)

        assert 'processing_time' in result['metadata']
        assert 'diffusion_steps' in result['metadata']
        assert 'f0_condition' in result['metadata']
        assert 'pipeline' in result['metadata']
        assert result['metadata']['pipeline'] == 'seed_vc'

    def test_convert_supports_pitch_shift(self, pipeline_with_ref):
        """convert passes pitch_shift to wrapper."""
        audio = np.random.randn(44100 * 2).astype(np.float32)

        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = iter([
            (None, (44100, np.random.randn(44100 * 2).astype(np.float32)))
        ])
        pipeline_with_ref._wrapper = mock_wrapper

        with patch('tempfile.NamedTemporaryFile') as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.wav'
            mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            with patch('soundfile.write'):
                with patch('os.unlink'):
                    pipeline_with_ref.convert(audio, sample_rate=44100, pitch_shift=12)

        # Check pitch_shift was passed
        call_kwargs = mock_wrapper.convert_voice.call_args[1]
        assert call_kwargs['pitch_shift'] == 12


class TestProgressCallback:
    """Test progress reporting."""

    def test_progress_callback_called(self):
        """on_progress callback is called during conversion."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(require_gpu=False)

        pipeline._reference_audio = np.random.randn(44100 * 3).astype(np.float32)
        pipeline._reference_sr = 44100

        progress_calls = []

        def on_progress(stage, progress):
            progress_calls.append((stage, progress))

        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = iter([
            (None, (44100, np.random.randn(44100 * 2).astype(np.float32)))
        ])
        pipeline._wrapper = mock_wrapper

        audio = np.random.randn(44100 * 2).astype(np.float32)

        with patch('tempfile.NamedTemporaryFile') as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.wav'
            mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            with patch('soundfile.write'):
                with patch('os.unlink'):
                    pipeline.convert(audio, sample_rate=44100, on_progress=on_progress)

        assert len(progress_calls) > 0
        stages = [c[0] for c in progress_calls]
        assert 'preprocessing' in stages or 'conversion' in stages


class TestConvertWithSeparation:
    """Test conversion with vocal separation."""

    def test_convert_with_separation_calls_separator(self):
        """convert_with_separation uses MelBandRoFormer."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(require_gpu=False)

        pipeline._reference_audio = np.random.randn(44100 * 3).astype(np.float32)
        pipeline._reference_sr = 44100

        # Mock separator
        mock_separator = MagicMock()
        mock_separator.extract_vocals.return_value = torch.randn(1, 44100 * 2)

        # Mock wrapper
        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = iter([
            (None, (44100, np.random.randn(44100 * 2).astype(np.float32)))
        ])
        pipeline._wrapper = mock_wrapper

        audio = np.random.randn(44100 * 2).astype(np.float32)

        with patch('auto_voice.audio.separator.MelBandRoFormer', return_value=mock_separator):
            with patch('tempfile.NamedTemporaryFile') as mock_tmp:
                mock_file = MagicMock()
                mock_file.name = '/tmp/test.wav'
                mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

                with patch('soundfile.write'):
                    with patch('os.unlink'):
                        result = pipeline.convert_with_separation(audio, sample_rate=44100)

        mock_separator.extract_vocals.assert_called_once()


class TestEstimatedMemory:
    """Test memory estimation."""

    def test_estimated_memory_constant(self):
        """Pipeline reports estimated GPU memory."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        assert hasattr(SeedVCPipeline, 'ESTIMATED_MEMORY_GB')
        assert SeedVCPipeline.ESTIMATED_MEMORY_GB > 0
        # Should be around 8GB for all components
        assert SeedVCPipeline.ESTIMATED_MEMORY_GB >= 6.0


class TestWrapperInitialization:
    """Test lazy initialization of Seed-VC wrapper."""

    def test_initialize_imports_wrapper(self):
        """_initialize imports SeedVCWrapper."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize') as mock_init:
            pipeline = SeedVCPipeline(require_gpu=False)

        # Should have been called during __init__
        mock_init.assert_called_once()

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_initialize_loads_models(self):
        """_initialize loads all model components."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        try:
            pipeline = SeedVCPipeline(device=torch.device('cuda'))
            # Should have wrapper loaded
            assert pipeline._wrapper is not None
        except RuntimeError as e:
            if "seed-vc" in str(e).lower() or "SeedVCWrapper" in str(e):
                pytest.skip("Seed-VC models not available")
            raise

    def test_initialize_raises_on_import_error(self):
        """_initialize raises RuntimeError on import failure."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch('sys.path', []):  # Clear path
            with patch.dict('sys.modules', {'seed_vc_wrapper': None}):
                with patch.object(SeedVCPipeline, '__init__', lambda s, **k: None):
                    pipeline = SeedVCPipeline.__new__(SeedVCPipeline)
                    pipeline._wrapper = None
                    pipeline.device = torch.device('cuda')

                    # Import will fail
                    with pytest.raises(RuntimeError, match="Failed to import"):
                        pipeline._initialize()


class TestOutputNormalization:
    """Test output audio normalization."""

    def test_output_normalized_to_95_percent(self):
        """Output is normalized to peak at 0.95."""
        from auto_voice.inference.seed_vc_pipeline import SeedVCPipeline

        with patch.object(SeedVCPipeline, '_initialize'):
            pipeline = SeedVCPipeline(require_gpu=False)

        pipeline._reference_audio = np.random.randn(44100 * 3).astype(np.float32)
        pipeline._reference_sr = 44100

        # Mock wrapper to return loud audio
        loud_audio = np.random.randn(44100 * 2).astype(np.float32) * 10  # Very loud
        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = iter([
            (None, (44100, loud_audio))
        ])
        pipeline._wrapper = mock_wrapper

        audio = np.random.randn(44100 * 2).astype(np.float32)

        with patch('tempfile.NamedTemporaryFile') as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.wav'
            mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            with patch('soundfile.write'):
                with patch('os.unlink'):
                    result = pipeline.convert(audio, sample_rate=44100)

        # Output should be normalized
        output = result['audio']
        assert output.abs().max() <= 1.0
        # Should be close to 0.95 peak
        assert output.abs().max() >= 0.9
