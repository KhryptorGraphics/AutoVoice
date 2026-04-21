"""Tests for HQ-SVC wrapper (cutting-edge pipeline component).

HQ-SVC provides:
  - Zero-shot singing voice conversion (with target reference audio)
  - Super-resolution: 16kHz → 44.1kHz upsampling

Architecture:
  FACodec (content) → DDSP → Diffusion → NSF-HiFiGAN @ 44.1kHz

TDD: These tests define the expected interface before implementation.
"""
import os
import sys
import pytest
import torch
import numpy as np
import soundfile as sf

# Mark all tests in this module as requiring CUDA
pytestmark = [pytest.mark.cuda, pytest.mark.slow]


def _load_audio_with_soundfile(path: str) -> tuple[torch.Tensor, int]:
    """Load audio without relying on torchaudio's TorchCodec backend."""
    audio, sample_rate = sf.read(path, dtype='float32')
    audio_tensor = torch.from_numpy(audio)
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.transpose(0, 1)
    return audio_tensor, int(sample_rate)


class TestHQSVCWrapperInit:
    """Test HQ-SVC wrapper initialization."""

    def test_import_succeeds(self):
        """Wrapper module can be imported."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper
        assert HQSVCWrapper is not None

    def test_init_creates_instance(self):
        """Wrapper initializes without errors on CUDA device."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = HQSVCWrapper(device=torch.device('cuda'))
        assert wrapper is not None
        assert wrapper.device.type == 'cuda'

    def test_init_loads_models(self):
        """Wrapper loads all required model components."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = HQSVCWrapper(device=torch.device('cuda'))

        # Check all components are loaded
        assert wrapper.fa_encoder is not None, "FACodec encoder not loaded"
        assert wrapper.fa_decoder is not None, "FACodec decoder not loaded"
        assert wrapper.net_g is not None, "HQ-SVC generator not loaded"
        assert wrapper.vocoder is not None, "NSF-HiFiGAN vocoder not loaded"
        assert wrapper.f0_extractor is not None, "RMVPE F0 extractor not loaded"
        assert wrapper.volume_extractor is not None, "Volume extractor not loaded"

    def test_init_without_cuda_raises(self):
        """Wrapper raises error when CUDA unavailable and required."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        # Mock CUDA unavailable
        with pytest.raises(RuntimeError, match="CUDA"):
            HQSVCWrapper(device=torch.device('cpu'), require_gpu=True)

    def test_sample_rates_configured(self):
        """Wrapper has correct sample rate configuration."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = HQSVCWrapper(device=torch.device('cuda'))

        assert wrapper.output_sample_rate == 44100, "Output should be 44.1kHz"
        assert wrapper.encoder_sample_rate == 16000, "FACodec encoder expects 16kHz"


class TestHQSVCSuperResolution:
    """Test 16kHz → 44.1kHz super-resolution mode."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper instance for tests."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper
        return HQSVCWrapper(device=torch.device('cuda'))

    @pytest.fixture
    def audio_16k(self):
        """Create 16kHz test audio (2 seconds of sine wave)."""
        duration = 2.0
        sr = 16000
        t = torch.linspace(0, duration, int(sr * duration))
        # 440Hz sine with some harmonics for realistic audio
        audio = 0.5 * torch.sin(2 * np.pi * 440 * t)
        audio += 0.25 * torch.sin(2 * np.pi * 880 * t)
        audio += 0.125 * torch.sin(2 * np.pi * 1320 * t)
        return audio

    def test_super_resolution_output_shape(self, wrapper, audio_16k):
        """Super-resolution produces 44.1kHz output."""
        result = wrapper.super_resolve(audio_16k, sample_rate=16000)

        # Check output sample rate
        assert result['sample_rate'] == 44100

        # Check duration preserved (within 5%)
        input_duration = len(audio_16k) / 16000
        output_duration = len(result['audio']) / 44100
        assert abs(output_duration - input_duration) < input_duration * 0.05

    def test_super_resolution_output_not_nan(self, wrapper, audio_16k):
        """Super-resolution output contains no NaN values."""
        result = wrapper.super_resolve(audio_16k, sample_rate=16000)

        assert not torch.isnan(result['audio']).any(), "Output contains NaN"
        assert not torch.isinf(result['audio']).any(), "Output contains Inf"

    def test_super_resolution_output_normalized(self, wrapper, audio_16k):
        """Super-resolution output is within normalized range."""
        result = wrapper.super_resolve(audio_16k, sample_rate=16000)

        # Output should be in [-1, 1] range
        assert result['audio'].max() <= 1.0
        assert result['audio'].min() >= -1.0

    def test_super_resolution_returns_metadata(self, wrapper, audio_16k):
        """Super-resolution returns processing metadata."""
        result = wrapper.super_resolve(audio_16k, sample_rate=16000)

        assert 'metadata' in result
        assert 'processing_time' in result['metadata']
        assert result['metadata']['processing_time'] > 0

    def test_super_resolution_handles_short_audio(self, wrapper):
        """Super-resolution handles minimum-length audio (1 second)."""
        # 1 second at 16kHz = 16000 samples (safe minimum for F0 extraction)
        duration = 1.0
        sr = 16000
        t = torch.linspace(0, duration, int(sr * duration))
        # Use a realistic vocal-like sine wave instead of noise
        short_audio = 0.5 * torch.sin(2 * np.pi * 440 * t)

        result = wrapper.super_resolve(short_audio, sample_rate=16000)
        assert result['sample_rate'] == 44100

    def test_super_resolution_rejects_too_short(self, wrapper):
        """Super-resolution rejects audio shorter than 500ms."""
        # 250ms at 16kHz = 4000 samples (too short for reliable F0)
        duration = 0.25
        sr = 16000
        t = torch.linspace(0, duration, int(sr * duration))
        too_short = 0.5 * torch.sin(2 * np.pi * 440 * t)

        with pytest.raises(RuntimeError, match="too short"):
            wrapper.super_resolve(too_short, sample_rate=16000)


class TestHQSVCVoiceConversion:
    """Test zero-shot voice conversion mode."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper instance for tests."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper
        return HQSVCWrapper(device=torch.device('cuda'))

    @pytest.fixture
    def source_audio(self):
        """Create source audio for conversion (3 seconds)."""
        duration = 3.0
        sr = 44100
        t = torch.linspace(0, duration, int(sr * duration))
        # Simulate a vocal with vibrato
        vibrato = 5 * torch.sin(2 * np.pi * 5 * t)
        audio = 0.5 * torch.sin(2 * np.pi * (440 + vibrato) * t)
        return audio

    @pytest.fixture
    def target_audio(self):
        """Create target reference audio (2 seconds)."""
        duration = 2.0
        sr = 44100
        t = torch.linspace(0, duration, int(sr * duration))
        # Different pitch character
        audio = 0.5 * torch.sin(2 * np.pi * 330 * t)
        return audio

    def test_convert_output_shape(self, wrapper, source_audio, target_audio):
        """Voice conversion produces 44.1kHz output."""
        result = wrapper.convert(
            source_audio=source_audio,
            source_sample_rate=44100,
            target_audio=target_audio,
            target_sample_rate=44100,
        )

        assert result['sample_rate'] == 44100

        # Duration should match source (within 5%)
        input_duration = len(source_audio) / 44100
        output_duration = len(result['audio']) / 44100
        assert abs(output_duration - input_duration) < input_duration * 0.05

    def test_convert_output_not_nan(self, wrapper, source_audio, target_audio):
        """Voice conversion output contains no NaN values."""
        result = wrapper.convert(
            source_audio=source_audio,
            source_sample_rate=44100,
            target_audio=target_audio,
            target_sample_rate=44100,
        )

        assert not torch.isnan(result['audio']).any(), "Output contains NaN"

    def test_convert_with_pitch_shift(self, wrapper, source_audio, target_audio):
        """Voice conversion supports explicit pitch shifting."""
        result = wrapper.convert(
            source_audio=source_audio,
            source_sample_rate=44100,
            target_audio=target_audio,
            target_sample_rate=44100,
            pitch_shift=12,  # One octave up
        )

        assert result['sample_rate'] == 44100
        assert 'pitch_shift' in result['metadata']
        assert result['metadata']['pitch_shift'] == 12

    def test_convert_auto_pitch_adjust(self, wrapper, source_audio, target_audio):
        """Voice conversion can auto-adjust pitch to match target."""
        result = wrapper.convert(
            source_audio=source_audio,
            source_sample_rate=44100,
            target_audio=target_audio,
            target_sample_rate=44100,
            auto_pitch=True,
        )

        assert result['sample_rate'] == 44100
        # Auto-computed pitch shift should be recorded
        assert 'pitch_shift' in result['metadata']

    def test_convert_with_speaker_embedding(self, wrapper, source_audio):
        """Voice conversion works with pre-computed speaker embedding."""
        # 256-dim speaker embedding (HQ-SVC format)
        speaker_embedding = torch.randn(256)

        result = wrapper.convert(
            source_audio=source_audio,
            source_sample_rate=44100,
            speaker_embedding=speaker_embedding,
        )

        assert result['sample_rate'] == 44100

    def test_convert_multiple_targets(self, wrapper, source_audio, target_audio):
        """Voice conversion averages multiple target references."""
        # Multiple targets for more robust speaker embedding
        targets = [target_audio, target_audio * 0.9]

        result = wrapper.convert(
            source_audio=source_audio,
            source_sample_rate=44100,
            target_audio=targets,
            target_sample_rate=44100,
        )

        assert result['sample_rate'] == 44100


class TestHQSVCSpeakerEmbedding:
    """Test speaker embedding extraction."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper instance for tests."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper
        return HQSVCWrapper(device=torch.device('cuda'))

    @pytest.fixture
    def reference_audio(self):
        """Create reference audio for embedding extraction."""
        duration = 3.0
        sr = 44100
        t = torch.linspace(0, duration, int(sr * duration))
        audio = 0.5 * torch.sin(2 * np.pi * 440 * t)
        return audio

    def test_extract_embedding_shape(self, wrapper, reference_audio):
        """Speaker embedding has correct shape."""
        embedding = wrapper.extract_speaker_embedding(
            reference_audio, sample_rate=44100
        )

        # HQ-SVC uses 256-dim speaker embeddings from FACodec
        assert embedding.shape == (256,), f"Expected (256,), got {embedding.shape}"

    def test_extract_embedding_normalized(self, wrapper, reference_audio):
        """Speaker embedding is L2-normalized."""
        embedding = wrapper.extract_speaker_embedding(
            reference_audio, sample_rate=44100
        )

        # Check L2 norm is approximately 1
        norm = torch.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Embedding norm {norm} != 1.0"

    def test_extract_embedding_deterministic(self, wrapper, reference_audio):
        """Same audio produces same embedding."""
        emb1 = wrapper.extract_speaker_embedding(reference_audio, sample_rate=44100)
        emb2 = wrapper.extract_speaker_embedding(reference_audio, sample_rate=44100)

        # Should be identical (no randomness in inference mode)
        assert torch.allclose(emb1, emb2, atol=1e-5)

    def test_extract_embedding_multiple_files(self, wrapper, reference_audio):
        """Embedding extraction handles multiple audio files."""
        audios = [reference_audio, reference_audio * 0.9]

        embedding = wrapper.extract_speaker_embedding(
            audios, sample_rate=44100
        )

        # Should return averaged embedding
        assert embedding.shape == (256,)


class TestHQSVCProgressCallback:
    """Test progress reporting."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper instance for tests."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper
        return HQSVCWrapper(device=torch.device('cuda'))

    @pytest.fixture
    def audio(self):
        """Create test audio (2 seconds sine wave at 16kHz)."""
        duration = 2.0
        sr = 16000
        t = torch.linspace(0, duration, int(sr * duration))
        return 0.5 * torch.sin(2 * np.pi * 440 * t)

    def test_progress_callback_called(self, wrapper, audio):
        """Progress callback is invoked during processing."""
        progress_calls = []

        def on_progress(stage: str, progress: float):
            progress_calls.append((stage, progress))

        wrapper.super_resolve(audio, sample_rate=16000, on_progress=on_progress)

        assert len(progress_calls) > 0, "Progress callback not called"

        # Should report stages
        stages = [call[0] for call in progress_calls]
        assert 'encoding' in stages or 'preprocessing' in stages

    def test_progress_increases_monotonically(self, wrapper, audio):
        """Progress values increase monotonically."""
        progress_values = []

        def on_progress(stage: str, progress: float):
            progress_values.append(progress)

        wrapper.super_resolve(audio, sample_rate=16000, on_progress=on_progress)

        # Check monotonic increase
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i-1], \
                f"Progress decreased: {progress_values[i-1]} -> {progress_values[i]}"


class TestHQSVCIntegration:
    """Integration tests with real audio files."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper instance for tests."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper
        return HQSVCWrapper(device=torch.device('cuda'))

    @pytest.mark.skipif(
        not os.path.exists('/home/kp/thordrive/autovoice/test_audio'),
        reason="Test audio directory not found"
    )
    def test_real_audio_super_resolution(self, wrapper):
        """Super-resolution works on real audio file."""
        # Find a test audio file
        test_dir = '/home/kp/thordrive/autovoice/test_audio'
        audio_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]

        if not audio_files:
            pytest.skip("No test audio files found")

        audio_path = os.path.join(test_dir, audio_files[0])
        audio, sr = _load_audio_with_soundfile(audio_path)
        if audio.ndim == 2:
            audio = audio.mean(dim=0)

        result = wrapper.super_resolve(audio, sample_rate=sr)

        assert result['sample_rate'] == 44100
        assert len(result['audio']) > 0

    @pytest.mark.skipif(
        not os.path.exists('/home/kp/thordrive/autovoice/test_audio'),
        reason="Test audio directory not found"
    )
    def test_real_audio_voice_conversion(self, wrapper):
        """Voice conversion works on real audio files."""
        test_dir = '/home/kp/thordrive/autovoice/test_audio'
        audio_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]

        if len(audio_files) < 2:
            pytest.skip("Need at least 2 test audio files")

        source_path = os.path.join(test_dir, audio_files[0])
        target_path = os.path.join(test_dir, audio_files[1])

        source_audio, source_sr = _load_audio_with_soundfile(source_path)
        target_audio, target_sr = _load_audio_with_soundfile(target_path)

        if source_audio.ndim == 2:
            source_audio = source_audio.mean(dim=0)
        if target_audio.ndim == 2:
            target_audio = target_audio.mean(dim=0)

        result = wrapper.convert(
            source_audio=source_audio,
            source_sample_rate=source_sr,
            target_audio=target_audio,
            target_sample_rate=target_sr,
        )

        assert result['sample_rate'] == 44100
        assert len(result['audio']) > 0
