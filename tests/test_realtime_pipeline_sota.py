"""Tests for SOTA realtime voice conversion pipeline.

Tests the RealtimePipeline architecture:
    Audio -> ContentVec (16kHz) -> RMVPE (pitch) -> SimpleDecoder -> HiFiGAN (22kHz)

Target: <100ms total chunk latency for live karaoke.

Component latency targets:
- ContentVec: ~40ms
- RMVPE: ~20ms
- SimpleDecoder: ~10ms
- HiFiGAN: ~20ms
"""
import time
import pytest
import numpy as np
import torch


class TestSimpleDecoder:
    """Tests for the lightweight SimpleDecoder for realtime inference."""

    def test_simple_decoder_init(self):
        """SimpleDecoder initializes with correct dimensions."""
        from auto_voice.inference.realtime_pipeline import SimpleDecoder

        decoder = SimpleDecoder(
            content_dim=768,
            pitch_dim=256,
            speaker_dim=256,
            n_mels=80,
            hidden_dim=256,
        )

        assert decoder.content_dim == 768
        assert decoder.n_mels == 80
        assert decoder.hidden_dim == 256

    def test_simple_decoder_forward(self):
        """SimpleDecoder produces correct output shape."""
        from auto_voice.inference.realtime_pipeline import SimpleDecoder

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SimpleDecoder().to(device)

        B, T = 1, 50
        content = torch.randn(B, T, 768, device=device)
        pitch = torch.randn(B, T, 256, device=device)
        speaker = torch.randn(B, 256, device=device)

        mel = decoder(content, pitch, speaker)

        assert mel.shape == (B, 80, T), f"Expected (1, 80, 50), got {mel.shape}"
        assert not torch.any(torch.isnan(mel))
        assert not torch.any(torch.isinf(mel))

    def test_simple_decoder_fast_inference(self):
        """SimpleDecoder inference completes in <10ms."""
        from auto_voice.inference.realtime_pipeline import SimpleDecoder

        if not torch.cuda.is_available():
            pytest.skip("CUDA required for latency testing")

        device = torch.device('cuda')
        decoder = SimpleDecoder().to(device)
        decoder.eval()

        B, T = 1, 50
        content = torch.randn(B, T, 768, device=device)
        pitch = torch.randn(B, T, 256, device=device)
        speaker = torch.randn(B, 256, device=device)

        for _ in range(5):
            with torch.no_grad():
                _ = decoder(content, pitch, speaker)
        torch.cuda.synchronize()

        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = decoder(content, pitch, speaker)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        assert avg_ms < 10, f"SimpleDecoder too slow: {avg_ms:.2f}ms (target <10ms)"


class TestRealtimePipeline:
    """Tests for the full realtime voice conversion pipeline."""

    def test_pipeline_init(self):
        """RealtimePipeline initializes with all components."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline()

        assert pipeline.sample_rate == 16000
        assert pipeline.output_sample_rate == 22050
        assert pipeline._content_encoder is not None
        assert pipeline._pitch_extractor is not None
        assert pipeline._decoder is not None
        assert pipeline._vocoder is not None

    def test_pipeline_set_speaker(self):
        """RealtimePipeline can set target speaker embedding."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline()

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        pipeline.set_speaker_embedding(embedding)

        assert pipeline._speaker_embedding is not None
        assert pipeline._speaker_embedding.shape == (1, 256)

    def test_pipeline_process_chunk(self):
        """RealtimePipeline processes audio chunk and returns converted audio."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline()

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.set_speaker_embedding(embedding)

        duration = 1.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        output = pipeline.process_chunk(audio)

        assert output is not None
        assert len(output) > 0
        assert output.dtype == np.float32
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_pipeline_passthrough_no_speaker(self):
        """RealtimePipeline passes through audio when no speaker set."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline()

        audio = np.random.randn(16000).astype(np.float32)
        output = pipeline.process_chunk(audio)

        assert output is not None
        np.testing.assert_array_almost_equal(output, audio)

    def test_pipeline_latency_target(self):
        """RealtimePipeline achieves <100ms latency for 1s chunk."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        if not torch.cuda.is_available():
            pytest.skip("CUDA required for latency testing")

        device = torch.device('cuda')
        pipeline = RealtimePipeline(device=device)

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.set_speaker_embedding(embedding)

        audio = np.random.randn(16000).astype(np.float32)

        # Warmup - need more iterations for CUDA kernels and model warmup
        for _ in range(10):
            _ = pipeline.process_chunk(audio)
        torch.cuda.synchronize()

        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = pipeline.process_chunk(audio)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        assert avg_ms < 100, f"Pipeline too slow: {avg_ms:.2f}ms (target <100ms)"

    def test_pipeline_streaming_chunks(self):
        """RealtimePipeline handles streaming chunks correctly."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline()

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.set_speaker_embedding(embedding)

        chunk_size = 1600
        outputs = []
        for _ in range(10):
            audio = np.random.randn(chunk_size).astype(np.float32)
            output = pipeline.process_chunk(audio)
            outputs.append(output)

        assert all(o is not None for o in outputs)
        assert all(len(o) > 0 for o in outputs)

    def test_pipeline_get_latency_breakdown(self):
        """RealtimePipeline reports per-component latency breakdown."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline()

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.set_speaker_embedding(embedding)

        audio = np.random.randn(16000).astype(np.float32)
        _ = pipeline.process_chunk(audio)

        metrics = pipeline.get_latency_metrics()

        assert 'content_encoder_ms' in metrics
        assert 'pitch_extractor_ms' in metrics
        assert 'decoder_ms' in metrics
        assert 'vocoder_ms' in metrics
        assert 'total_ms' in metrics


class TestContentVecIntegration:
    """Tests for ContentVec encoder integration in realtime pipeline."""

    def test_contentvec_16khz_input(self):
        """ContentVec properly handles 16kHz input."""
        from auto_voice.models.encoder import ContentVecEncoder

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentVecEncoder(
            output_dim=768,
            pretrained=None,
            device=device,
        )

        audio = torch.randn(1, 16000, device=device)

        features = encoder.encode(audio)

        expected_frames = 16000 // 320
        assert features.shape[0] == 1
        assert features.shape[2] == 768
        assert abs(features.shape[1] - expected_frames) < 10


class TestRMVPEIntegration:
    """Tests for RMVPE pitch extractor integration."""

    def test_rmvpe_extracts_f0(self):
        """RMVPE extracts F0 contour from audio."""
        from auto_voice.models.pitch import RMVPEPitchExtractor

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(device=device)
        extractor.to(device)  # Ensure model is on device

        t = torch.linspace(0, 1, 16000, device=device)
        audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)

        f0 = extractor.extract(audio)

        assert f0.shape[0] == 1
        assert f0.shape[1] > 0
        voiced_frames = f0[0, f0[0] > 0]
        if len(voiced_frames) > 0:
            mean_f0 = voiced_frames.mean().item()
            # The lightweight RMVPE path is approximate; keep a tolerance wide
            # enough for backend/runtime variance while still catching obvious drift.
            assert 350 < mean_f0 < 550, f"F0 detection off: {mean_f0}Hz"


class TestHiFiGANIntegration:
    """Tests for HiFiGAN vocoder integration."""

    def test_hifigan_synthesizes_audio(self):
        """HiFiGAN synthesizes audio from mel spectrogram."""
        from auto_voice.models.vocoder import HiFiGANVocoder

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocoder = HiFiGANVocoder(device=device)

        mel = torch.randn(1, 80, 50, device=device)

        audio = vocoder.synthesize(mel)

        expected_len = 50 * 256
        assert audio.shape[0] == 1
        assert abs(audio.shape[1] - expected_len) < 100


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_conversion_pipeline(self):
        """Full pipeline converts audio through all stages."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline = RealtimePipeline(device=device)

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.set_speaker_embedding(embedding)

        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        output = pipeline.process_chunk(audio)

        assert output is not None
        assert len(output) > 0
        assert output.dtype == np.float32
        assert np.abs(output).max() <= 1.0
        assert not np.any(np.isnan(output))

    @pytest.mark.slow
    def test_continuous_streaming(self):
        """Pipeline handles continuous streaming for extended period."""
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline = RealtimePipeline(device=device)

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.set_speaker_embedding(embedding)

        chunk_samples = 1600
        n_chunks = 100

        total_output_samples = 0
        for _ in range(n_chunks):
            audio = np.random.randn(chunk_samples).astype(np.float32)
            output = pipeline.process_chunk(audio)
            total_output_samples += len(output)

        assert total_output_samples > 0
