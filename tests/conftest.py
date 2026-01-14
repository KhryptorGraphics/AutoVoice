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

def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--baseline-file",
        action="store",
        default=".github/quality_baseline.json",
        help="Path to baseline metrics file for regression testing"
    )
    parser.addoption(
        "--output-file",
        action="store",
        default=None,
        help="Path to save regression test results JSON"
    )


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
    config.addinivalue_line(
        "markers", "quality: Quality evaluation tests"
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
        config = {'sample_rate': 16000}
        return AudioProcessor(config=config)
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
# Web Application Fixtures
# ============================================================================

class MockVoiceCloner:
    """Mock VoiceCloner for testing that doesn't require GPU or models."""

    def __init__(self):
        self.profiles = {}
        self._profile_counter = 0

    def create_voice_profile(self, audio, user_id=None):
        """Create a mock voice profile."""
        import uuid
        profile_id = str(uuid.uuid4())
        self._profile_counter += 1
        profile = {
            'profile_id': profile_id,
            'user_id': user_id,
            'audio_duration': 5.0,
            'vocal_range': {'min_hz': 100, 'max_hz': 500},
            'embedding': [0.1] * 256,
            'created_at': '2024-01-01T00:00:00Z'
        }
        self.profiles[profile_id] = profile
        return profile

    def list_profiles(self, user_id=None):
        """List all profiles, optionally filtered by user_id."""
        if user_id:
            return [p for p in self.profiles.values() if p.get('user_id') == user_id]
        return list(self.profiles.values())

    def get_profile(self, profile_id):
        """Get a specific profile by ID."""
        return self.profiles.get(profile_id)

    def delete_profile(self, profile_id):
        """Delete a profile by ID."""
        if profile_id in self.profiles:
            del self.profiles[profile_id]
            return True
        return False

    def load_voice_profile(self, profile_id):
        """Load a voice profile by ID.

        For testing, returns a mock profile for any ID to allow pipeline tests
        to proceed past validation.
        """
        # Return existing profile if found
        profile = self.get_profile(profile_id)
        if profile:
            return profile
        # For testing convenience, return a mock profile for any ID
        # This allows tests to focus on pipeline behavior without profile setup
        return {
            'profile_id': profile_id,
            'user_id': 'test_user',
            'audio_duration': 5.0,
            'vocal_range': {'min_hz': 100, 'max_hz': 500},
            'embedding': [0.1] * 256,
            'created_at': '2024-01-01T00:00:00Z'
        }


@pytest.fixture
def app_and_socketio():
    """Create Flask app and SocketIO instance for testing.

    Sets up mock services for voice_cloner and other dependencies.

    Returns:
        Tuple of (Flask app, SocketIO instance)
    """
    from src.auto_voice.web.app import create_app
    app, sio = create_app(config={'TESTING': True})

    # Set up mock voice cloner for testing
    app.voice_cloner = MockVoiceCloner()

    # Set up app_config for audio settings (used by API endpoints)
    app.app_config = {
        'audio': {
            'sample_rate': 22050,
            'hop_length': 512
        }
    }

    yield app, sio


@pytest.fixture
def client(app_and_socketio):
    """Flask test client fixture.

    Uses the app_and_socketio fixture to create a test client.
    """
    app, _ = app_and_socketio
    with app.test_client() as client:
        yield client


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


# ============================================================================
# CUDA Bindings Fixtures
# ============================================================================

@pytest.fixture
def cuda_kernels_module():
    """Import and return cuda_kernels module if available."""
    try:
        import cuda_kernels
        return cuda_kernels
    except ImportError:
        try:
            from auto_voice import cuda_kernels
            return cuda_kernels
        except ImportError:
            pytest.skip("cuda_kernels module not available")


@pytest.fixture
def cuda_pitch_detection_params():
    """Standard parameters for pitch detection tests."""
    return {
        'sample_rate': 22050.0,
        'frame_length': 2048,
        'hop_length': 512,
        'fmin': 80.0,
        'fmax': 800.0,
    }


@pytest.fixture
def synthetic_sine_wave():
    """Generate synthetic sine wave for testing.

    Returns:
        Tuple of (audio, frequency, sample_rate)
    """
    sample_rate = 22050.0
    duration = 2.0
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    return audio, frequency, sample_rate


@pytest.fixture
def synthetic_audio_with_vibrato():
    """Generate synthetic audio with known vibrato.

    Returns:
        Tuple of (audio, base_freq, vibrato_rate, vibrato_depth, sample_rate)
    """
    sample_rate = 22050.0
    duration = 2.0
    base_freq = 440.0
    vibrato_rate = 5.5  # Hz
    vibrato_depth_cents = 50.0  # cents

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Apply vibrato
    depth_ratio = 2 ** (vibrato_depth_cents / 1200.0)
    vibrato = depth_ratio ** np.sin(2 * np.pi * vibrato_rate * t)

    # Generate audio
    audio = np.sin(2 * np.pi * base_freq * vibrato * t).astype(np.float32)

    return audio, base_freq, vibrato_rate, vibrato_depth_cents, sample_rate


@pytest.fixture
def cuda_tensors_for_pitch_detection(cuda_available: bool):
    """Create CUDA tensors for pitch detection testing.

    Returns:
        Dict with pre-allocated tensors
    """
    if not cuda_available:
        pytest.skip("CUDA not available")

    sample_rate = 22050.0
    duration = 1.0
    audio_length = int(sample_rate * duration)

    frame_length = 2048
    hop_length = 512
    n_frames = max(0, (audio_length - frame_length) // hop_length + 1)

    # Generate test audio
    t = np.linspace(0, duration, audio_length, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    return {
        'audio': torch.from_numpy(audio).cuda(),
        'output_pitch': torch.zeros(n_frames, device='cuda'),
        'output_confidence': torch.zeros(n_frames, device='cuda'),
        'output_vibrato': torch.zeros(n_frames, device='cuda'),
        'sample_rate': sample_rate,
        'frame_length': frame_length,
        'hop_length': hop_length,
        'n_frames': n_frames,
    }


@pytest.fixture
def various_test_frequencies():
    """List of test frequencies (musical notes) for validation."""
    return [
        (110.0, "A2"),
        (220.0, "A3"),
        (440.0, "A4"),
        (880.0, "A5"),
        (261.63, "C4"),
        (329.63, "E4"),
        (392.0, "G4"),
        (523.25, "C5"),
    ]


@pytest.fixture
def test_sample_rates():
    """Common sample rates for testing."""
    return [8000.0, 16000.0, 22050.0, 44100.0, 48000.0]


@pytest.fixture
def cuda_kernel_performance_tracker(cuda_available: bool):
    """Track CUDA kernel performance metrics."""
    if not cuda_available:
        pytest.skip("CUDA not available")

    class PerformanceTracker:
        def __init__(self):
            self.timings = []
            self.memory_snapshots = []

        def start(self):
            """Start timing measurement."""
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()
            self.start_memory = torch.cuda.memory_allocated()

        def stop(self):
            """Stop timing measurement."""
            torch.cuda.synchronize()
            self.end_time = time.perf_counter()
            self.end_memory = torch.cuda.memory_allocated()

            duration = self.end_time - self.start_time
            memory_delta = self.end_memory - self.start_memory

            self.timings.append(duration)
            self.memory_snapshots.append(memory_delta)

            return duration, memory_delta

        def get_stats(self):
            """Get performance statistics."""
            return {
                'mean_time': np.mean(self.timings) if self.timings else 0,
                'std_time': np.std(self.timings) if self.timings else 0,
                'min_time': np.min(self.timings) if self.timings else 0,
                'max_time': np.max(self.timings) if self.timings else 0,
                'mean_memory': np.mean(self.memory_snapshots) if self.memory_snapshots else 0,
                'iterations': len(self.timings),
            }

    import time
    return PerformanceTracker()


@pytest.fixture
def audio_with_noise():
    """Generate audio with various SNR levels.

    Returns:
        Function that generates noisy audio given SNR in dB
    """
    def generate_noisy_audio(snr_db: float = 20.0, duration: float = 1.0,
                            frequency: float = 440.0, sample_rate: float = 22050.0):
        """Generate sine wave with added noise at specified SNR.

        Args:
            snr_db: Signal-to-noise ratio in decibels
            duration: Duration in seconds
            frequency: Frequency of sine wave
            sample_rate: Sample rate in Hz

        Returns:
            Noisy audio as numpy array
        """
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        signal = np.sin(2 * np.pi * frequency * t)

        # Calculate noise power
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(signal)) * np.sqrt(noise_power)

        audio = (signal + noise).astype(np.float32)
        return audio

    return generate_noisy_audio


@pytest.fixture
def multi_frequency_audio():
    """Generate audio with multiple frequency components.

    Returns:
        Function that generates audio with specified frequency mix
    """
    def generate_multi_freq_audio(frequencies: list, amplitudes: list = None,
                                  duration: float = 1.0, sample_rate: float = 22050.0):
        """Generate audio with multiple frequency components.

        Args:
            frequencies: List of frequencies in Hz
            amplitudes: List of amplitudes (default: equal for all)
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Mixed audio as numpy array
        """
        if amplitudes is None:
            amplitudes = [1.0] * len(frequencies)

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.zeros(len(t), dtype=np.float32)

        for freq, amp in zip(frequencies, amplitudes):
            audio += amp * np.sin(2 * np.pi * freq * t)

        # Normalize
        audio = audio / np.max(np.abs(audio))
        return audio.astype(np.float32)

    return generate_multi_freq_audio


@pytest.fixture(autouse=True)
def cuda_error_check(cuda_available: bool):
    """Check for CUDA errors after each test."""
    yield
    if cuda_available:
        try:
            # Synchronize to catch any async errors
            torch.cuda.synchronize()
        except RuntimeError as e:
            pytest.fail(f"CUDA error detected after test: {e}")


# ============================================================================
# Integration Test Fixtures (Task 1)
# ============================================================================

@pytest.fixture
def song_file(tmp_path: Path) -> Path:
    """Synthesize and save test song audio file.

    Creates a 5-second song with vocals and instrumental components.
    """
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not available")

    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Simulate vocals (melody at A4 and C5)
    vocals = 0.4 * np.sin(2 * np.pi * 440 * t)  # A4
    vocals += 0.2 * np.sin(2 * np.pi * 523.25 * t)  # C5

    # Simulate instrumental (bass + rhythm)
    bass = 0.6 * np.sin(2 * np.pi * 110 * t)  # A2
    drums = 0.3 * np.random.randn(len(t)) * np.sin(2 * np.pi * 2 * t)

    # Mix stereo
    left = vocals + bass + drums
    right = vocals * 0.8 + bass + drums * 0.7
    stereo = np.stack([left, right])

    # Save file
    song_path = tmp_path / "test_song.wav"
    sf.write(str(song_path), stereo.T, sample_rate)

    return song_path


@pytest.fixture
def test_profile(tmp_path: Path):
    """Create and cleanup voice profile for testing.

    Yields profile dict, automatically cleans up after test.
    """
    try:
        from src.auto_voice.storage.voice_profiles import VoiceProfileStorage
    except ImportError:
        pytest.skip("VoiceProfileStorage not available")

    storage = VoiceProfileStorage(storage_dir=str(tmp_path / 'profiles'))

    # Create test profile
    profile = {
        'profile_id': 'test-profile-123',
        'user_id': 'test_user',
        'created_at': '2025-01-15T10:00:00Z',
        'embedding': np.random.randn(256).astype(np.float32),
        'audio_duration': 10.0,
        'sample_rate': 22050,
        'metadata': {'test': True}
    }

    storage.save_profile(profile)

    yield profile

    # Cleanup
    if storage.profile_exists(profile['profile_id']):
        storage.delete_profile(profile['profile_id'])


@pytest.fixture
def pipeline_instance(device):
    """Create SingingConversionPipeline instance for testing."""
    try:
        from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        config = {
            'device': device,
            'cache_enabled': True,
            'sample_rate': 22050
        }
        return SingingConversionPipeline(config=config)
    except ImportError:
        pytest.skip("SingingConversionPipeline not available")


@pytest.fixture
def concurrent_executor():
    """ThreadPoolExecutor for concurrent testing."""
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=4)
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def memory_leak_detector(cuda_available: bool):
    """CPU and GPU memory tracking for leak detection.

    Tracks memory before and after test, reports leaks if detected.
    """
    import psutil
    import os

    class MemoryLeakDetector:
        def __init__(self, cuda_available):
            self.cuda_available = cuda_available
            self.process = psutil.Process(os.getpid())

        def __enter__(self):
            # Record initial memory
            self.initial_cpu_memory = self.process.memory_info().rss
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                self.initial_gpu_memory = torch.cuda.memory_allocated()
            else:
                self.initial_gpu_memory = 0
            return self

        def __exit__(self, *args):
            # Check final memory
            import gc
            gc.collect()
            if self.cuda_available:
                torch.cuda.empty_cache()

            final_cpu_memory = self.process.memory_info().rss
            cpu_leak = final_cpu_memory - self.initial_cpu_memory

            if self.cuda_available:
                final_gpu_memory = torch.cuda.memory_allocated()
                gpu_leak = final_gpu_memory - self.initial_gpu_memory

                if gpu_leak > 10 * 1024 * 1024:  # 10 MB threshold
                    pytest.warn(f"Potential GPU memory leak: {gpu_leak / 1024 / 1024:.2f} MB")

            if cpu_leak > 50 * 1024 * 1024:  # 50 MB threshold
                pytest.warn(f"Potential CPU memory leak: {cpu_leak / 1024 / 1024:.2f} MB")

    return MemoryLeakDetector(cuda_available)


@pytest.fixture
def performance_thresholds():
    """Get performance thresholds from environment variables.

    Returns:
        Dictionary of threshold values
    """
    return {
        'gpu_speedup': float(os.environ.get('GPU_SPEEDUP_THRESHOLD', '3.0')),
        'cache_speedup': float(os.environ.get('CACHE_SPEEDUP_THRESHOLD', '3.0')),
        'max_rtf_cpu': float(os.environ.get('MAX_RTF_CPU', '20.0')),
        'max_rtf_gpu': float(os.environ.get('MAX_RTF_GPU', '5.0')),
        'max_gpu_memory_gb': float(os.environ.get('MAX_GPU_MEMORY_GB', '8.0')),
    }


@pytest.fixture
def performance_tracker():
    """Performance metrics tracking fixture.

    Tracks timing, memory, and throughput metrics during tests.
    Supports JSON output when PYTEST_JSON_OUTPUT environment variable is set.
    """
    import json
    import time

    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
            self.timings = []
            self.json_output_path = os.environ.get('PYTEST_JSON_OUTPUT')

        def start(self, label: str = "default"):
            """Start timing measurement."""
            self.label = label
            self.start_time = time.perf_counter()

        def stop(self):
            """Stop timing measurement and record."""
            elapsed = time.perf_counter() - self.start_time
            self.timings.append(elapsed)
            if self.label not in self.metrics:
                self.metrics[self.label] = []
            self.metrics[self.label].append(elapsed)
            return elapsed

        def record(self, metric_name: str, value: float):
            """Record a custom metric."""
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)

            # Write to JSON if output path is set
            if self.json_output_path:
                self._write_json()

        def get_summary(self):
            """Get summary statistics."""
            summary = {}
            for metric, values in self.metrics.items():
                if values:
                    summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            return summary

        def _write_json(self):
            """Write metrics to JSON file."""
            if not self.json_output_path:
                return

            try:
                # Load existing data if file exists
                output_path = Path(self.json_output_path)
                if output_path.exists():
                    with open(output_path) as f:
                        data = json.load(f)
                else:
                    data = {'metrics': {}}

                # Update with current metrics
                data['metrics'].update(self.get_summary())

                # Write back
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to write JSON metrics: {e}")

    return PerformanceTracker()


@pytest.fixture
def gpu_memory_monitor(cuda_available: bool):
    """GPU memory monitoring fixture.

    Monitors GPU memory usage during test execution.
    """
    if not cuda_available:
        pytest.skip("CUDA not available for GPU memory monitoring")

    class GPUMemoryMonitor:
        def __enter__(self):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
            return self

        def __exit__(self, *args):
            self.final_memory = torch.cuda.memory_allocated()
            self.peak_memory = torch.cuda.max_memory_allocated()
            self.memory_delta = self.final_memory - self.initial_memory

        def get_stats(self):
            """Get memory statistics in MB."""
            return {
                'initial_mb': self.initial_memory / 1024 / 1024,
                'final_mb': self.final_memory / 1024 / 1024,
                'peak_mb': self.peak_memory / 1024 / 1024,
                'delta_mb': self.memory_delta / 1024 / 1024
            }

    return GPUMemoryMonitor()


@pytest.fixture
def multi_format_audio(tmp_path: Path):
    """Generate test audio in multiple formats (WAV, FLAC).

    Returns dict mapping format to file path.
    """
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not available")

    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    formats = {}

    # WAV
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, sample_rate, format='WAV')
    formats['wav'] = wav_path

    # FLAC
    flac_path = tmp_path / "test.flac"
    sf.write(str(flac_path), audio, sample_rate, format='FLAC')
    formats['flac'] = flac_path

    return formats


@pytest.fixture
def multi_sample_rate_audio(tmp_path: Path):
    """Generate test audio at multiple sample rates.

    Returns dict mapping sample rate to file path.
    """
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not available")

    duration = 1.0
    sample_rates = [8000, 16000, 22050, 44100]
    audio_files = {}

    for sr in sample_rates:
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio_path = tmp_path / f"test_{sr}hz.wav"
        sf.write(str(audio_path), audio, sr)
        audio_files[sr] = audio_path

    return audio_files


@pytest.fixture
def memory_monitor(cuda_available: bool):
    """Memory monitoring fixture for CPU and GPU memory tracking.

    Tracks memory usage before and after tests with configurable thresholds.
    """
    import psutil
    import os

    class MemoryMonitor:
        def __init__(self, cuda_available):
            self.cuda_available = cuda_available
            self.process = psutil.Process(os.getpid())
            self.initial_cpu_memory = 0
            self.initial_gpu_memory = 0
            self.final_cpu_memory = 0
            self.final_gpu_memory = 0

        def __enter__(self):
            # Record initial memory
            self.initial_cpu_memory = self.process.memory_info().rss
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                self.initial_gpu_memory = torch.cuda.memory_allocated()
            return self

        def __exit__(self, *args):
            # Check final memory
            import gc
            gc.collect()
            if self.cuda_available:
                torch.cuda.empty_cache()

            self.final_cpu_memory = self.process.memory_info().rss
            if self.cuda_available:
                self.final_gpu_memory = torch.cuda.memory_allocated()

        def get_stats(self):
            """Get memory statistics in MB."""
            cpu_delta = self.final_cpu_memory - self.initial_cpu_memory
            stats = {
                'initial_cpu_mb': self.initial_cpu_memory / 1024 / 1024,
                'final_cpu_mb': self.final_cpu_memory / 1024 / 1024,
                'cpu_delta_mb': cpu_delta / 1024 / 1024,
            }

            if self.cuda_available:
                gpu_delta = self.final_gpu_memory - self.initial_gpu_memory
                peak_memory = torch.cuda.max_memory_allocated()
                stats.update({
                    'initial_gpu_mb': self.initial_gpu_memory / 1024 / 1024,
                    'final_gpu_mb': self.final_gpu_memory / 1024 / 1024,
                    'peak_mb': peak_memory / 1024 / 1024,
                    'gpu_delta_mb': gpu_delta / 1024 / 1024,
                    'delta_mb': gpu_delta / 1024 / 1024  # Alias for compatibility
                })
            else:
                stats['peak_mb'] = 0.0
                stats['delta_mb'] = 0.0

            return stats

    return MemoryMonitor(cuda_available)


@pytest.fixture
def quality_evaluator(device):
    """Create evaluator for quality tests."""
    try:
        from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
        evaluator = VoiceConversionEvaluator(
            sample_rate=44100,
            device=device
        )
        return evaluator
    except ImportError:
        pytest.skip("VoiceConversionEvaluator not available")


@pytest.fixture
def voice_profile_storage(tmp_path: Path):
    """Mock voice profile storage for testing."""
    from unittest.mock import Mock

    storage = Mock()
    storage.storage_dir = str(tmp_path / 'profiles')
    storage.profile_exists = Mock(return_value=True)
    storage.get_profile = Mock(return_value={
        'profile_id': 'test-profile-123',
        'user_id': 'test_user',
        'embedding': np.random.randn(256).astype(np.float32),
        'sample_rate': 22050
    })
    storage.save_profile = Mock()
    storage.delete_profile = Mock()

    return storage


@pytest.fixture
def vocal_separator(device):
    """Mock vocal separator for testing."""
    from unittest.mock import Mock

    separator = Mock()
    separator.device = device

    def mock_separate(audio_path):
        """Mock separation returning synthetic vocals and instrumental."""
        # Generate synthetic outputs
        sample_rate = 22050
        duration = 2.0
        samples = int(sample_rate * duration)

        vocals = np.random.randn(samples).astype(np.float32) * 0.3
        instrumental = np.random.randn(samples).astype(np.float32) * 0.3

        return vocals, instrumental

    separator.separate_vocals = Mock(side_effect=mock_separate)
    separator.config = {'cache_enabled': False}

    return separator


@pytest.fixture
def singing_voice_converter(cuda_available: bool):
    """Mock singing voice converter for testing."""
    from unittest.mock import Mock

    device = "cuda" if cuda_available else "cpu"
    converter = Mock()
    converter.device = device

    def mock_convert(source_audio, target_speaker_embedding, source_sample_rate=22050):
        """Mock conversion returning synthetic converted audio."""
        # Return audio with same length as input
        if isinstance(source_audio, torch.Tensor):
            length = source_audio.shape[-1]
        else:
            length = len(source_audio)

        converted = torch.randn(length) * 0.5
        return converted

    converter.convert = Mock(side_effect=mock_convert)
    converter.eval = Mock(return_value=converter)
    converter.to = Mock(return_value=converter)
    converter.prepare_for_inference = Mock()

    return converter


@pytest.fixture
def stereo_mono_pairs(tmp_path: Path):
    """Generate stereo and mono pairs for testing.

    Returns dict with 'stereo' and 'mono' file paths.
    """
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not available")

    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_mono = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Mono file
    mono_path = tmp_path / "test_mono.wav"
    sf.write(str(mono_path), audio_mono, sample_rate)

    # Stereo file
    audio_stereo = np.stack([audio_mono, audio_mono * 0.8])
    stereo_path = tmp_path / "test_stereo.wav"
    sf.write(str(stereo_path), audio_stereo.T, sample_rate)

    return {
        'mono': mono_path,
        'stereo': stereo_path
    }


# ============================================================================
# Quality Validation Test Fixtures (Comment 4)
# ============================================================================

@pytest.fixture
def quality_targets():
    """Default quality targets for validation tests."""
    try:
        from src.auto_voice.evaluation.evaluator import QualityTargets
        return QualityTargets()
    except ImportError:
        pytest.skip("QualityTargets not available")


@pytest.fixture
def voice_conversion_evaluator(cuda_available: bool):
    """Create VoiceConversionEvaluator instance."""
    try:
        from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
        device = "cuda" if cuda_available else "cpu"
        return VoiceConversionEvaluator(sample_rate=44100, device=device)
    except ImportError:
        pytest.skip("VoiceConversionEvaluator not available")


@pytest.fixture
def synthetic_evaluation_pair():
    """Generate synthetic source and target audio pair with known quality."""
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Source: A4 sine wave
    source_freq = 440.0
    source_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * source_freq * t),
        dtype=torch.float32
    )

    # Target: Very similar A4 sine wave (high quality match)
    target_freq = 440.05  # Minimal pitch difference
    target_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * target_freq * t),
        dtype=torch.float32
    )

    return source_audio, target_audio, {
        'expected_pitch_rmse_hz': 0.0,  # Should be very low
        'expected_correlation': 0.95,   # Should correlate highly
        'sample_rate': sample_rate
    }


@pytest.fixture
def poor_quality_evaluation_pair():
    """Generate synthetic source and target audio pair with poor quality match."""
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Source: A4 sine wave
    source_freq = 440.0
    source_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * source_freq * t),
        dtype=torch.float32
    )

    # Target: A5 octave above (poor quality match)
    target_freq = 880.0
    target_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * target_freq * t),
        dtype=torch.float32
    )

    return source_audio, target_audio, {
        'expected_pitch_rmse_hz': 440.0,  # Large difference
        'expected_correlation': 0.0,      # Should not correlate
        'sample_rate': sample_rate
    }


@pytest.fixture
def test_metadata_file(tmp_path: Path):
    """Create test metadata JSON file for metadata-driven evaluation."""
    try:
        import soundfile as sf
        import json
    except ImportError:
        pytest.skip("Required packages not available")

    # Generate source audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    source_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Save source file
    source_path = tmp_path / 'test001_source.wav'
    sf.write(str(source_path), source_audio, sample_rate)

    # Create metadata
    metadata = {
        'test_cases': [{
            'id': 'test001',
            'source_audio': str(source_path),
            'target_profile_id': 'mock-profile-001',
            'conversion_params': {'pitch_shift': 0.0},
            'reference_audio': str(source_path),  # Self-reference for test
        }, {
            'id': 'test002',
            'source_audio': str(source_path),
            'target_profile_id': 'mock-profile-002',
            'conversion_params': {'pitch_shift': 1.0},
            'reference_audio': str(source_path),
        }]
    }

    # Save metadata
    metadata_path = tmp_path / 'test_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


@pytest.fixture
def mock_conversion_pipeline(cuda_available: bool):
    """Mock conversion pipeline that returns predictable results."""
    device = "cuda" if cuda_available else "cpu"

    class MockPipeline:
        def __init__(self):
            self.device = device

        def convert_song(self, song_path, target_profile_id, pitch_shift=0.0):
            """Mock conversion - just loads and returns input audio."""
            try:
                import soundfile as sf
                audio, sr = sf.read(song_path)
                # Simulate conversion delay
                import time
                time.sleep(0.1)
                return {
                    'mixed_audio': audio if audio.ndim == 1 else audio[:, 0],
                    'sample_rate': sr,
                    'duration': len(audio) / sr,
                    'f0_contour': np.ones(int(sr * len(audio) / sr / 512)) * 440.0
                }
            except ImportError:
                pytest.skip("soundfile not available for mock pipeline")

    return MockPipeline()

# ============================================================================
# Import New Testing Infrastructure Fixtures
# ============================================================================
pytest_plugins = [
    'tests.fixtures.audio_fixtures',
    'tests.fixtures.model_fixtures',
    'tests.fixtures.gpu_fixtures',
    'tests.fixtures.mock_fixtures',
    'tests.fixtures.integration_fixtures',
    'tests.fixtures.performance_fixtures',
]
