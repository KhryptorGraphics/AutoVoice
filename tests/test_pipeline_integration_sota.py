"""Tests for SOTA end-to-end pipeline integration (Phase 7).

Validates the full singing voice conversion pipeline:
  audio file → separation → content extraction → pitch extraction →
  SVC decoder → vocoder → output audio file

Tests cover:
- Complete pipeline forward pass
- Sample rate conversions between components
- Various input formats (mono/stereo, different sample rates)
- GPU memory management (sequential processing)
- Progress callbacks
- No fallback behavior
"""
import pytest
import torch
import numpy as np
import tempfile
import os


class TestSOTAPipelineInit:
    """Tests for SOTA pipeline initialization."""

    def test_class_exists(self):
        """SOTAConversionPipeline class should exist."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        assert SOTAConversionPipeline is not None

    def test_init_default(self):
        """Default initialization with all components."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()
        assert pipeline.device is not None
        assert pipeline.separator is not None
        assert pipeline.content_extractor is not None
        assert pipeline.pitch_extractor is not None
        assert pipeline.decoder is not None
        assert pipeline.vocoder is not None

    def test_init_custom_device(self):
        """Initialize on specific device."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        device = torch.device('cpu')
        pipeline = SOTAConversionPipeline(device=device)
        assert pipeline.device == device


class TestSOTAPipelineConvert:
    """Tests for the full convert() pipeline."""

    def _make_test_audio(self, duration=1.0, sr=24000, stereo=False):
        """Create a test audio tensor with a simple sine wave."""
        t = torch.linspace(0, duration, int(sr * duration))
        audio = torch.sin(2 * np.pi * 440 * t)  # 440Hz A4
        if stereo:
            audio = torch.stack([audio, audio * 0.8])  # [2, T]
        return audio, sr

    def test_convert_produces_audio(self):
        """Pipeline produces valid audio output."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio, sr = self._make_test_audio(duration=0.5, sr=24000)
        speaker = torch.randn(256)  # Target speaker embedding

        result = pipeline.convert(audio, sr, speaker)
        assert 'audio' in result
        assert 'sample_rate' in result
        assert result['audio'].dim() == 1  # Mono output
        assert result['audio'].shape[0] > 0
        assert result['sample_rate'] == 24000  # BigVGAN output rate

    def test_convert_output_finite(self):
        """Output audio should be finite (no NaN/Inf)."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio, sr = self._make_test_audio(duration=0.5, sr=24000)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, sr, speaker)
        assert torch.isfinite(result['audio']).all()

    def test_convert_output_bounded(self):
        """Output waveform should be in [-1, 1] range."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio, sr = self._make_test_audio(duration=0.5, sr=24000)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, sr, speaker)
        assert result['audio'].abs().max() <= 1.0

    def test_convert_stereo_input(self):
        """Pipeline handles stereo input (converts to mono internally)."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio, sr = self._make_test_audio(duration=0.5, sr=24000, stereo=True)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, sr, speaker)
        assert result['audio'].dim() == 1  # Output is always mono

    def test_convert_different_sample_rate(self):
        """Pipeline handles 44.1kHz input."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio, sr = self._make_test_audio(duration=0.5, sr=44100)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, sr, speaker)
        assert result['sample_rate'] == 24000  # Always outputs 24kHz

    def test_convert_16khz_input(self):
        """Pipeline handles 16kHz input."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio, sr = self._make_test_audio(duration=0.5, sr=16000)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, sr, speaker)
        assert result['sample_rate'] == 24000


class TestSampleRateConversions:
    """Tests for inter-component sample rate handling."""

    def test_separator_gets_44100(self):
        """Separator component expects 44.1kHz."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()
        assert pipeline.separator.sample_rate == 44100

    def test_content_extractor_gets_16000(self):
        """ContentVec expects 16kHz input."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()
        assert pipeline.content_extractor.sample_rate == 16000

    def test_pitch_extractor_gets_16000(self):
        """RMVPE expects 16kHz input."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()
        assert pipeline.pitch_extractor.sample_rate == 16000

    def test_vocoder_outputs_24000(self):
        """BigVGAN outputs 24kHz audio."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()
        assert pipeline.vocoder.sample_rate == 24000


class TestGPUMemoryManagement:
    """Tests for sequential processing and memory cleanup."""

    def test_sequential_processing(self):
        """Components run sequentially (not all loaded to GPU at once)."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()
        # Pipeline should track processing stages
        assert hasattr(pipeline, '_current_stage')

    def test_convert_returns_metadata(self):
        """Convert returns processing metadata."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio = torch.sin(torch.linspace(0, 1, 12000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, 24000, speaker)
        assert 'metadata' in result
        assert 'processing_time' in result['metadata']
        assert result['metadata']['processing_time'] > 0


class TestProgressCallbacks:
    """Tests for progress reporting."""

    def test_callback_called(self):
        """Progress callback is called during conversion."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio = torch.sin(torch.linspace(0, 1, 12000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        progress_log = []

        def on_progress(stage: str, progress: float):
            progress_log.append((stage, progress))

        result = pipeline.convert(audio, 24000, speaker, on_progress=on_progress)
        assert len(progress_log) > 0

    def test_callback_stages(self):
        """Callback reports all pipeline stages."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio = torch.sin(torch.linspace(0, 1, 12000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        stages_seen = set()

        def on_progress(stage: str, progress: float):
            stages_seen.add(stage)

        pipeline.convert(audio, 24000, speaker, on_progress=on_progress)
        # Should see all major stages
        expected_stages = {'separation', 'content_extraction', 'pitch_extraction', 'decoding', 'vocoder'}
        assert expected_stages.issubset(stages_seen)

    def test_callback_progress_increases(self):
        """Progress values should be monotonically non-decreasing."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio = torch.sin(torch.linspace(0, 1, 12000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        progress_values = []

        def on_progress(stage: str, progress: float):
            progress_values.append(progress)

        pipeline.convert(audio, 24000, speaker, on_progress=on_progress)
        # Each progress value should be >= previous
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]


class TestNoFallbackPipeline:
    """Tests for strict error behavior."""

    def test_empty_audio_raises(self):
        """Empty input should raise RuntimeError."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio = torch.tensor([])
        speaker = torch.randn(256)

        with pytest.raises(RuntimeError):
            pipeline.convert(audio, 24000, speaker)

    def test_too_short_audio_raises(self):
        """Very short audio (< 100ms) should raise RuntimeError."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        # 50ms at 24kHz = 1200 samples (too short for meaningful processing)
        audio = torch.randn(1200)
        speaker = torch.randn(256)

        with pytest.raises(RuntimeError):
            pipeline.convert(audio, 24000, speaker)

    def test_wrong_speaker_dim_raises(self):
        """Wrong speaker embedding dimension should raise RuntimeError."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        pipeline = SOTAConversionPipeline()

        audio = torch.sin(torch.linspace(0, 1, 12000) * 2 * np.pi * 440)
        speaker = torch.randn(128)  # Wrong dim (should be 256)

        with pytest.raises(RuntimeError):
            pipeline.convert(audio, 24000, speaker)
