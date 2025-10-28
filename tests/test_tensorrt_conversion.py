"""Tests for TensorRT conversion and optimization of voice conversion components."""

import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path
import os

# Test imports with fallbacks
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    ort = None


class TestTensorRTConverterInit:
    """Test TensorRT converter initialization and basic functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_converter_initialization(self, temp_dir):
        """Test TensorRT converter initialization."""
        if not TRT_AVAILABLE:
            pytest.skip("TensorRT not available")

        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter

        converter = TensorRTConverter(export_dir=temp_dir / "exports")

        assert converter.export_dir.exists()
        assert converter.device == 'cpu'
        assert converter.trt_logger is not None

    def test_converter_without_tensorrt(self, temp_dir):
        """Test converter initializes when TensorRT not available."""
        if TRT_AVAILABLE:
            pytest.skip("TensorRT is available, cannot test fallback")

        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter

        converter = TensorRTConverter(export_dir=temp_dir / "exports")

        assert converter.export_dir.exists()
        assert converter.device == 'cpu'


class TestContentEncoderONNX:
    """Test ContentEncoder ONNX export functionality."""

    @pytest.fixture
    def content_encoder(self):
        """Create ContentEncoder using CNN fallback for ONNX export."""
        from src.auto_voice.models.content_encoder import ContentEncoder

        # Use CNN fallback since HuBERT is not ONNX-exportable
        encoder = ContentEncoder(
            encoder_type='cnn_fallback',
            output_dim=256,
            device='cpu'
        )
        encoder.eval()
        return encoder

    @pytest.fixture
    def converter(self, tmp_path):
        """Create TensorRT converter instance."""
        if not TRT_AVAILABLE:
            pytest.skip("TensorRT not available")

        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter
        return TensorRTConverter(export_dir=tmp_path / "exports")

    def test_content_encoder_onnx_export(self, content_encoder, converter, tmp_path):
        """Test ContentEncoder ONNX export."""
        onnx_path = converter.export_content_encoder(
            content_encoder,
            model_name='test_content_encoder'
        )

        assert onnx_path.exists()
        assert onnx_path.suffix == '.onnx'

    def test_content_encoder_method_export(self, content_encoder, tmp_path):
        """Test ContentEncoder's built-in ONNX export method."""
        onnx_path = tmp_path / "test_encoder.onnx"

        content_encoder.export_to_onnx(str(onnx_path))

        assert onnx_path.exists()
        assert onnx_path.suffix == '.onnx'

    def test_hubert_not_exportable(self, tmp_path):
        """Test that HuBERT encoder correctly rejects ONNX export."""
        from src.auto_voice.models.content_encoder import ContentEncoder

        # Try to create HuBERT encoder (may fail if torch.hub unavailable)
        try:
            encoder = ContentEncoder(
                encoder_type='hubert_soft',
                output_dim=256,
                device='cpu',
                use_torch_hub=False  # Skip torch.hub loading
            )
        except:
            pytest.skip("HuBERT encoder not available")

        # Should raise error for ONNX export
        with pytest.raises(RuntimeError, match="cannot be exported to ONNX"):
            encoder.export_to_onnx(tmp_path / "hubert.onnx")

    @pytest.mark.skipif(not ORT_AVAILABLE, reason="ONNX Runtime not available")
    def test_onnx_model_validation(self, content_encoder, converter):
        """Test that exported ONNX model is valid."""
        onnx_path = converter.export_content_encoder(content_encoder)

        # Validate with ONNX Runtime
        session = ort.InferenceSession(str(onnx_path))

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        assert len(inputs) > 0
        assert len(outputs) > 0

        # Test inference with sample data
        sample_audio = torch.randn(1, 16000)  # 1 second at 16kHz
        sample_rate = torch.tensor([16000])

        input_data = {
            'input_audio': sample_audio.numpy(),
            'sample_rate': sample_rate.numpy()
        }

        result = session.run(None, input_data)
        assert len(result) > 0

        # Check output shape makes sense
        output = result[0]  # content_features output
        assert output.shape[0] == 1  # batch size
        assert output.shape[2] == content_encoder.output_dim  # feature dim


class TestPitchEncoderONNX:
    """Test PitchEncoder ONNX export functionality."""

    @pytest.fixture
    def pitch_encoder(self):
        """Create PitchEncoder for testing."""
        from src.auto_voice.models.pitch_encoder import PitchEncoder

        encoder = PitchEncoder(
            pitch_dim=192,
            hidden_dim=128,
            device='cpu'
        )
        encoder.eval()
        return encoder

    @pytest.fixture
    def converter(self, tmp_path):
        """Create TensorRT converter instance."""
        if not TRT_AVAILABLE:
            pytest.skip("TensorRT not available")

        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter
        return TensorRTConverter(export_dir=tmp_path / "exports")

    def test_pitch_encoder_onnx_export(self, pitch_encoder, converter):
        """Test PitchEncoder ONNX export."""
        onnx_path = converter.export_pitch_encoder(
            pitch_encoder,
            model_name='test_pitch_encoder'
        )

        assert onnx_path.exists()
        assert onnx_path.suffix == '.onnx'

    def test_pitch_encoder_method_export(self, pitch_encoder, tmp_path):
        """Test PitchEncoder's built-in ONNX export method."""
        onnx_path = tmp_path / "test_pitch.onnx"

        pitch_encoder.export_to_onnx(str(onnx_path))

        assert onnx_path.exists()

    @pytest.mark.skipif(not ORT_AVAILABLE, reason="ONNX Runtime not available")
    def test_pitch_encoder_inference(self, pitch_encoder, converter):
        """Test PitchEncoder ONNX inference."""
        onnx_path = converter.export_pitch_encoder(pitch_encoder)

        session = ort.InferenceSession(str(onnx_path))

        # Test with sample F0 data
        batch_size, time_steps = 1, 50
        f0_data = torch.randn(batch_size, time_steps)

        input_data = {'f0_input': f0_data.numpy()}

        result = session.run(None, input_data)
        pitch_output = result[0]

        assert pitch_output.shape[0] == batch_size
        assert pitch_output.shape[1] == time_steps
        assert pitch_output.shape[2] == pitch_encoder.pitch_dim


class TestFlowDecoderONNX:
    """Test FlowDecoder ONNX export functionality."""

    @pytest.fixture
    def flow_decoder(self):
        """Create FlowDecoder for testing."""
        from src.auto_voice.models.flow_decoder import FlowDecoder

        decoder = FlowDecoder(
            in_channels=192,
            hidden_channels=192,
            num_flows=2,  # Small for testing
            cond_channels=704
        )
        decoder.eval()
        return decoder

    @pytest.fixture
    def converter(self, tmp_path):
        """Create TensorRT converter instance."""
        if not TRT_AVAILABLE:
            pytest.skip("TensorRT not available")

        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter
        return TensorRTConverter(export_dir=tmp_path / "exports")

    def test_flow_decoder_onnx_export(self, flow_decoder, converter):
        """Test FlowDecoder ONNX export."""
        onnx_path = converter.export_flow_decoder(
            flow_decoder,
            model_name='test_flow_decoder',
            cond_channels=704
        )

        assert onnx_path.exists()
        assert onnx_path.suffix == '.onnx'

    def test_flow_decoder_method_export(self, flow_decoder, tmp_path):
        """Test FlowDecoder's built-in ONNX export method."""
        onnx_path = tmp_path / "test_flow.onnx"

        flow_decoder.export_to_onnx(str(onnx_path), cond_channels=704)

        assert onnx_path.exists()

    @pytest.mark.skipif(not ORT_AVAILABLE, reason="ONNX Runtime not available")
    def test_flow_decoder_inference(self, flow_decoder, converter):
        """Test FlowDecoder ONNX inference."""
        onnx_path = converter.export_flow_decoder(flow_decoder, cond_channels=704)

        session = ort.InferenceSession(str(onnx_path))

        # Test with sample data
        batch_size, time_steps = 1, 50
        latent_dim = 192
        cond_dim = 704

        latent_input = torch.randn(batch_size, latent_dim, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)
        conditioning = torch.randn(batch_size, cond_dim, time_steps)

        input_data = {
            'latent_input': latent_input.numpy(),
            'mask': mask.numpy(),
            'conditioning': conditioning.numpy(),
            'inverse': np.array([True])
        }

        result = session.run(None, input_data)
        output = result[0]

        assert output.shape[0] == batch_size
        assert output.shape[1] == latent_dim  # Should preserve latent dimension
        assert output.shape[2] == time_steps


class TestSingingVoiceConverterTensorRT:
    """Test SingingVoiceConverter TensorRT integration."""

    @pytest.fixture
    def singing_voice_converter(self):
        """Create SingingVoiceConverter for testing."""
        from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter

        config = {
            'latent_dim': 192,
            'mel_channels': 80,
            'singing_voice_converter': {
                'content_encoder': {'type': 'cnn_fallback', 'output_dim': 256},
                'pitch_encoder': {'pitch_dim': 192},
                'speaker_encoder': {'embedding_dim': 256},
                'posterior_encoder': {'hidden_channels': 192},
                'flow_decoder': {'hidden_channels': 192, 'num_flows': 2},
                'vocoder': {'use_vocoder': False}
            }
        }

        model = SingingVoiceConverter(config)
        model.eval()
        return model

    def test_export_components_to_onnx(self, singing_voice_converter, tmp_path):
        """Test exporting SVC components to ONNX."""
        exported = singing_voice_converter.export_components_to_onnx(
            export_dir=str(tmp_path / "onnx")
        )

        expected_components = ['content_encoder', 'pitch_encoder', 'flow_decoder', 'mel_projection']
        assert set(exported.keys()) == set(expected_components)

        for component, path in exported.items():
            assert Path(path).exists()
            assert Path(path).suffix == '.onnx'

    def test_tensorrt_support_flags(self, singing_voice_converter):
        """Test TensorRT support configuration."""
        # Initially should have default settings
        assert hasattr(singing_voice_converter, 'use_tensorrt')
        assert hasattr(singing_voice_converter, 'tensorrt_models')
        assert hasattr(singing_voice_converter, 'fallback_to_pytorch')

        # Should be initially disabled or configured per config
        # (depends on config passed to __init__)


class TestTensorRTEngineLoading:
    """Test TensorRT engine loading and inference."""

    @pytest.mark.skipif(not TRT_AVAILABLE, reason="TensorRT not available")
    def test_engine_initialization(self, tmp_path):
        """Test TensorRT engine initialization."""
        from src.auto_voice.inference.tensorrt_engine import TensorRTEngine

        # Create a dummy engine file to test loading
        dummy_engine_path = tmp_path / "dummy.engine"
        dummy_engine_path.write_bytes(b"dummy engine data")

        # Should fail to load invalid engine but not crash
        with pytest.raises(RuntimeError):
            TensorRTEngine(str(dummy_engine_path))

    def test_engine_without_tensorrt(self, tmp_path):
        """Test engine fails gracefully without TensorRT."""
        if TRT_AVAILABLE:
            pytest.skip("TensorRT is available")

        from src.auto_voice.inference.tensorrt_engine import TensorRTEngine

        with pytest.raises(RuntimeError, match="TensorRT not available"):
            TensorRTEngine("dummy_path")


class TestBenchmarkingSetup:
    """Test benchmarking script setup and execution."""

    def test_benchmark_config(self):
        """Test benchmark configuration."""
        if not TRT_AVAILABLE:
            pytest.skip("TensorRT not available")

        from scripts.benchmark_tensorrt import BenchmarkConfig, VoiceConversionBenchmark

        # Mock args object
        class MockArgs:
            def __init__(self):
                self.model_dir = "./models"
                self.output_dir = "./benchmarks"
                self.samples = 10
                self.warmup = 2
                self.audio_lengths = [1.0]
                self.batch_sizes = [1]
                self.precision_modes = ['fp16']
                self.include_memory = False
                self.include_accuracy = False
                self.device = 'cpu'
                self.seed = 42

        args = MockArgs()
        config = BenchmarkConfig(args)

        assert config.sample_count == 10
        assert config.warmup_runs == 2
        assert config.audio_lengths == [1.0]
        assert config.output_dir.exists()

    def test_benchmark_data_generation(self, tmp_path):
        """Test synthetic benchmark data generation."""
        if not TRT_AVAILABLE:
            pytest.skip("TensorRT not available")

        from scripts.benchmark_tensorrt import BenchmarkConfig, VoiceConversionBenchmark

        # Mock args
        class MockArgs:
            model_dir = tmp_path
            output_dir = tmp_path / "results"
            samples = 5
            warmup = 1
            audio_lengths = [1.0]
            batch_sizes = [1]
            precision_modes = ['fp16']
            include_memory = False
            include_accuracy = False
            device = 'cpu'
            seed = 42

        config = BenchmarkConfig(MockArgs())
        benchmark = VoiceConversionBenchmark(config)

        # Generate test data
        test_data = benchmark.generate_test_data(1.0, batch_size=1)

        assert 'audio' in test_data
        assert 'f0' in test_data
        assert 'speaker_emb' in test_data
        assert test_data['sample_rate'] == 16000

        # Check shapes
        assert test_data['audio'].shape[0] == 1  # batch size
        assert test_data['f0'].shape[0] == 1  # batch size
        assert test_data['audio'].shape[1] == 16000  # 1 second at 16kHz


# Integration test (requires all components working together)
@pytest.mark.integration
class TestTensorRTIntegration:
    """Integration tests for complete TensorRT workflow."""

    @pytest.mark.skipif(not TRT_AVAILABLE or not ORT_AVAILABLE,
                        reason="TensorRT and ONNX Runtime required")
    def test_complete_workflow(self, tmp_path):
        """Test complete TensorRT workflow from export to inference."""
        from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter
        from src.auto_voice.inference.tensorrt_converter import TensorRTConverter

        # Setup model
        config = {
            'latent_dim': 192,
            'mel_channels': 80,
            'singing_voice_converter': {
                'content_encoder': {'type': 'cnn_fallback', 'output_dim': 256},
                'pitch_encoder': {'pitch_dim': 192},
                'speaker_encoder': {'embedding_dim': 256},
                'posterior_encoder': {'hidden_channels': 192},
                'flow_decoder': {'hidden_channels': 192, 'num_flows': 2},
                'vocoder': {'use_vocoder': False}
            }
        }

        model = SingingVoiceConverter(config)
        model.eval()

        # Export components
        onnx_dir = tmp_path / "onnx"
        onnx_paths = model.export_components_to_onnx(export_dir=str(onnx_dir))

        assert len(onnx_paths) == 4  # content, pitch, flow, mel_projection

        # Convert to TensorRT (would require GPU for full conversion)
        converter = TensorRTConverter(export_dir=tmp_path / "exports")

        # Verify ONNX models are loadable
        for component, onnx_path in onnx_paths.items():
            session = ort.InferenceSession(onnx_path)

            # Basic inference test
            if component == 'content_encoder':
                test_input = {
                    'input_audio': np.random.randn(1, 16000).astype(np.float32),
                    'sample_rate': np.array([16000])
                }
                result = session.run(None, test_input)
                assert len(result) == 1
                assert result[0].shape[1] > 0  # Has time dimension

            logger.info(f"ONNX validation passed for {component}")


if __name__ == '__main__':
    pytest.main([__file__])
