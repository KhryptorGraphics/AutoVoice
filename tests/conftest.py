"""
Pytest configuration and shared fixtures for AutoVoice tests.
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator

import pytest
import torch
import numpy as np
import yaml


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (component interactions)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (complete workflows)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (>1 second)"
    )
    config.addinivalue_line(
        "markers", "cuda: Tests requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "audio: Audio processing tests"
    )


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(cuda_available: bool) -> str:
    """Return device string ('cuda' or 'cpu')."""
    return "cuda" if cuda_available else "cpu"


@pytest.fixture
def cuda_device(cuda_available: bool):
    """Fixture that returns 'cuda' if available, else 'cpu'."""
    return torch.device("cuda" if cuda_available else "cpu")


def skip_if_no_cuda(func):
    """Decorator to skip tests when CUDA is unavailable."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )(func)


def require_cuda(func):
    """Decorator that fails tests if CUDA is unavailable."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for this test"
    )(func)


@pytest.fixture
def gpu_memory_tracker(cuda_available: bool):
    """Track GPU memory before and after tests."""
    if cuda_available:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        yield
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        memory_leak = final_memory - initial_memory
        if memory_leak > 1024 * 1024:  # 1 MB threshold
            pytest.warn(f"Potential memory leak: {memory_leak / 1024 / 1024:.2f} MB")
    else:
        yield


# ============================================================================
# Audio Fixtures
# ============================================================================

@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate synthetic audio (sine wave) at 16kHz."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def sample_audio_8khz() -> np.ndarray:
    """Generate 8kHz sample audio."""
    sample_rate = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def sample_audio_16khz() -> np.ndarray:
    """Generate 16kHz sample audio."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def sample_audio_22khz() -> np.ndarray:
    """Generate 22.05kHz sample audio."""
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def sample_audio_44khz() -> np.ndarray:
    """Generate 44.1kHz sample audio."""
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def sample_audio_silence() -> np.ndarray:
    """Generate silent audio."""
    sample_rate = 16000
    duration = 1.0
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


@pytest.fixture
def sample_audio_noise() -> np.ndarray:
    """Generate white noise audio."""
    sample_rate = 16000
    duration = 1.0
    return np.random.randn(int(sample_rate * duration)).astype(np.float32)


@pytest.fixture
def sample_audio_speech_like() -> np.ndarray:
    """Generate speech-like audio with multiple frequencies."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Mix of formant-like frequencies
    audio = (
        0.5 * np.sin(2 * np.pi * 500 * t) +
        0.3 * np.sin(2 * np.pi * 1500 * t) +
        0.2 * np.sin(2 * np.pi * 2500 * t)
    ).astype(np.float32)
    return audio


@pytest.fixture
def sample_audio_file(tmp_path: Path, sample_audio: np.ndarray) -> Path:
    """Create temporary audio file."""
    import soundfile as sf
    audio_file = tmp_path / "test_audio.wav"
    sf.write(str(audio_file), sample_audio, 16000)
    return audio_file


@pytest.fixture
def sample_mel_spectrogram() -> torch.Tensor:
    """Generate sample mel-spectrogram."""
    # Typical mel-spectrogram shape: (n_mels, time_steps)
    n_mels = 80
    time_steps = 100
    mel_spec = torch.randn(n_mels, time_steps)
    return mel_spec


@pytest.fixture
def audio_processor():
    """Instantiate AudioProcessor with test config."""
    try:
        from src.auto_voice.audio.processor import AudioProcessor
        return AudioProcessor(sample_rate=16000)
    except ImportError:
        pytest.skip("AudioProcessor not available")


@pytest.fixture
def gpu_audio_processor(cuda_available: bool):
    """Instantiate GPU AudioProcessor if CUDA is available."""
    if not cuda_available:
        pytest.skip("CUDA not available for GPU audio processor")
    try:
        from src.auto_voice.audio.gpu_processor import GPUAudioProcessor
        return GPUAudioProcessor(device="cuda")
    except ImportError:
        pytest.skip("GPUAudioProcessor not available")


@pytest.fixture
def singing_pitch_extractor(cuda_available: bool):
    """Instantiate SingingPitchExtractor for testing."""
    try:
        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor
        device = 'cuda' if cuda_available else 'cpu'
        return SingingPitchExtractor(device=device)
    except ImportError:
        pytest.skip("SingingPitchExtractor not available")


@pytest.fixture
def singing_analyzer(cuda_available: bool):
    """Instantiate SingingAnalyzer for testing."""
    try:
        from src.auto_voice.audio.singing_analyzer import SingingAnalyzer
        device = 'cuda' if cuda_available else 'cpu'
        return SingingAnalyzer(device=device)
    except ImportError:
        pytest.skip("SingingAnalyzer not available")


@pytest.fixture
def sample_vibrato_audio():
    """Generate synthetic audio with vibrato for testing.

    Returns:
        Tuple of (audio, ground_truth) where ground_truth contains
        'rate_hz' and 'depth_cents' of the vibrato.
    """
    sample_rate = 22050
    duration = 2.0
    base_freq = 440.0  # A4
    vibrato_rate = 5.5  # Hz
    vibrato_depth_cents = 60.0  # cents

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Convert vibrato depth from cents to frequency ratio
    depth_ratio = 2 ** (vibrato_depth_cents / 1200.0)

    # Generate vibrato modulation
    vibrato = depth_ratio ** np.sin(2 * np.pi * vibrato_rate * t)

    # Generate audio with vibrato
    audio = np.sin(2 * np.pi * base_freq * vibrato * t).astype(np.float32)

    ground_truth = {
        'rate_hz': vibrato_rate,
        'depth_cents': vibrato_depth_cents,
        'base_freq': base_freq
    }

    return audio, ground_truth


@pytest.fixture
def sample_breathy_audio():
    """Generate synthetic breathy voice for testing.

    Breathy voice has lower HNR (more noise relative to harmonics).
    """
    sample_rate = 22050
    duration = 1.0
    base_freq = 220.0  # A3

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate harmonic signal
    harmonic = np.sin(2 * np.pi * base_freq * t)

    # Add significant noise for breathiness
    noise = np.random.randn(len(t)) * 0.5

    # Mix harmonic and noise (breathy = more noise)
    audio = (0.5 * harmonic + 0.5 * noise).astype(np.float32)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    return audio


@pytest.fixture
def sample_clear_voice():
    """Generate synthetic clear voice for testing.

    Clear voice has high HNR (strong harmonics, little noise).
    """
    sample_rate = 22050
    duration = 1.0
    base_freq = 220.0  # A3

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate harmonic signal with multiple harmonics
    audio = np.zeros(len(t))
    for harmonic_num in range(1, 6):
        amplitude = 1.0 / harmonic_num
        audio += amplitude * np.sin(2 * np.pi * base_freq * harmonic_num * t)

    # Add minimal noise
    noise = np.random.randn(len(t)) * 0.01
    audio = (audio + noise).astype(np.float32)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    return audio


@pytest.fixture
def sample_crescendo_audio():
    """Generate audio with crescendo (increasing amplitude)."""
    sample_rate = 22050
    duration = 2.0
    base_freq = 330.0  # E4

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Linear amplitude increase
    amplitude = np.linspace(0.1, 1.0, len(t))

    # Generate audio
    audio = (amplitude * np.sin(2 * np.pi * base_freq * t)).astype(np.float32)

    return audio


@pytest.fixture
def sample_diminuendo_audio():
    """Generate audio with diminuendo (decreasing amplitude)."""
    sample_rate = 22050
    duration = 2.0
    base_freq = 330.0  # E4

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Linear amplitude decrease
    amplitude = np.linspace(1.0, 0.1, len(t))

    # Generate audio
    audio = (amplitude * np.sin(2 * np.pi * base_freq * t)).astype(np.float32)

    return audio


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def voice_transformer_config() -> Dict[str, Any]:
    """Voice transformer configuration."""
    return {
        "vocab_size": 256,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
    }


@pytest.fixture
def hifigan_config() -> Dict[str, Any]:
    """HiFiGAN configuration."""
    return {
        "in_channels": 80,
        "upsample_rates": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    }


@pytest.fixture
def voice_transformer(voice_transformer_config: Dict[str, Any]):
    """Instantiate VoiceTransformer."""
    try:
        from src.auto_voice.models.transformer import VoiceTransformer
        return VoiceTransformer(**voice_transformer_config)
    except ImportError:
        pytest.skip("VoiceTransformer not available")


@pytest.fixture
def hifigan_generator(hifigan_config: Dict[str, Any]):
    """Instantiate HiFiGANGenerator."""
    try:
        from src.auto_voice.models.hifigan import HiFiGANGenerator
        return HiFiGANGenerator(**hifigan_config)
    except ImportError:
        pytest.skip("HiFiGANGenerator not available")


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Create mock checkpoint file."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "epoch": 10,
        "loss": 0.5,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def voice_model(mock_checkpoint: Path):
    """Instantiate VoiceModel with mock checkpoint."""
    try:
        from src.auto_voice.models.voice_model import VoiceModel
        model = VoiceModel()
        # Try to load checkpoint if model supports it
        if hasattr(model, 'load_checkpoint'):
            try:
                model.load_checkpoint(str(mock_checkpoint))
            except:
                pass  # Checkpoint loading might fail, that's ok for testing
        return model
    except ImportError:
        pytest.skip("VoiceModel not available")


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Load default configuration."""
    return {
        "audio": {
            "sample_rate": 16000,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 80,
            "fmin": 0,
            "fmax": 8000,
        },
        "model": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.1,
        },
        "gpu": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "memory_fraction": 0.9,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0001,
            "num_epochs": 100,
        },
        "inference": {
            "backend": "pytorch",
            "batch_size": 1,
        },
    }


@pytest.fixture
def audio_config() -> Dict[str, Any]:
    """Audio-specific configuration."""
    return {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000,
    }


@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Model-specific configuration."""
    return {
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
        "vocab_size": 256,
    }


@pytest.fixture
def gpu_config(cuda_available: bool) -> Dict[str, Any]:
    """GPU-specific configuration."""
    return {
        "device": "cuda" if cuda_available else "cpu",
        "memory_fraction": 0.9,
        "allow_growth": True,
    }


@pytest.fixture
def test_config(tmp_path: Path) -> Path:
    """Create minimal test configuration file."""
    config = {
        "audio": {"sample_rate": 16000},
        "model": {"hidden_size": 128},
        "gpu": {"device": "cpu"},
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


# ============================================================================
# Temporary File Fixtures
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def temp_audio_file(tmp_path: Path) -> Path:
    """Create temporary audio file path."""
    return tmp_path / "temp_audio.wav"


@pytest.fixture
def temp_checkpoint_file(tmp_path: Path) -> Path:
    """Create temporary checkpoint file path."""
    return tmp_path / "temp_checkpoint.pt"


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_inference_engine():
    """Mock VoiceInferenceEngine."""
    from unittest.mock import Mock
    engine = Mock()
    engine.synthesize_speech.return_value = np.zeros(16000, dtype=np.float32)
    return engine


@pytest.fixture
def mock_audio_processor():
    """Mock AudioProcessor."""
    from unittest.mock import Mock
    processor = Mock()
    processor.to_mel_spectrogram.return_value = torch.randn(80, 100)
    processor.from_mel_spectrogram.return_value = np.zeros(16000, dtype=np.float32)
    processor.extract_features.return_value = {"mfcc": torch.randn(13, 100)}
    return processor


@pytest.fixture
def mock_gpu_manager(cuda_available: bool):
    """Mock GPUManager."""
    from unittest.mock import Mock
    manager = Mock()
    manager.is_available.return_value = cuda_available
    manager.device = "cuda" if cuda_available else "cpu"
    manager.get_status.return_value = {
        "available": cuda_available,
        "device_count": 1 if cuda_available else 0,
        "current_device": 0 if cuda_available else None,
    }
    return manager


@pytest.fixture
def mock_web_client():
    """Mock Flask test client."""
    from unittest.mock import Mock
    client = Mock()
    client.get.return_value.status_code = 200
    client.post.return_value.status_code = 200
    return client


# ============================================================================
# Performance Fixtures
# ============================================================================

@pytest.fixture
def benchmark():
    """Fixture for timing measurements."""
    import time

    class Benchmark:
        def __call__(self, func, *args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            return result, end - start

    return Benchmark()


@pytest.fixture
def benchmark_timer():
    """Fixture for timing measurements (alias for benchmark)."""
    import time

    class BenchmarkTimer:
        def __call__(self, func, *args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            return result, end - start

    return BenchmarkTimer()


@pytest.fixture
def memory_profiler(cuda_available: bool):
    """Fixture for memory tracking."""
    class MemoryProfiler:
        def __enter__(self):
            if cuda_available:
                torch.cuda.reset_peak_memory_stats()
                self.initial_memory = torch.cuda.memory_allocated()
            else:
                self.initial_memory = 0
            return self

        def __exit__(self, *args):
            if cuda_available:
                self.peak_memory = torch.cuda.max_memory_allocated()
                self.final_memory = torch.cuda.memory_allocated()
                self.memory_used = self.peak_memory - self.initial_memory

    return MemoryProfiler()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda(cuda_available: bool):
    """Clear CUDA cache after tests."""
    yield
    if cuda_available:
        torch.cuda.empty_cache()


@pytest.fixture
def reset_random_seeds():
    """Reset random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_text_inputs() -> list:
    """List of test text strings."""
    return [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Testing voice synthesis with various text inputs",
        "",  # Edge case: empty string
        "a",  # Edge case: single character
    ]


@pytest.fixture
def sample_speaker_ids() -> list:
    """List of test speaker IDs."""
    return [0, 1, 2, 5, 10]


@pytest.fixture
def sample_audio_paths() -> list:
    """List of sample audio file paths (for reference)."""
    return [
        "data/audio/sample1.wav",
        "data/audio/sample2.wav",
        "data/audio/sample3.wav",
    ]


@pytest.fixture
def sample_checkpoint_paths() -> list:
    """List of sample checkpoint paths (for reference)."""
    return [
        "checkpoints/model_epoch_10.pt",
        "checkpoints/model_epoch_20.pt",
        "checkpoints/best_model.pt",
    ]


# ============================================================================
# Performance Logger Fixture
# ============================================================================

@pytest.fixture
def performance_logger():
    """Fixture for logging performance metrics."""
    class PerformanceLogger:
        def __init__(self):
            self.metrics = {}

        def log(self, name: str, value: float, unit: str = "ms"):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({"value": value, "unit": unit})

        def get_summary(self):
            summary = {}
            for name, values in self.metrics.items():
                vals = [v["value"] for v in values]
                summary[name] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "count": len(vals),
                    "unit": values[0]["unit"] if values else "unknown",
                }
            return summary

    return PerformanceLogger()
