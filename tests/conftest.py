"""Shared test fixtures for AutoVoice."""
import os
import sys
import tempfile
import shutil
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def reset_separation_backend_hooks():
    """Clear leaked Demucs test doubles between tests.

    Some separation suites patch the module-level ``get_model``/``apply_model``
    hooks in ``auto_voice.audio.separation``. When those mocks leak across test
    boundaries, later integration tests instantiate ``VocalSeparator`` against
    empty MagicMocks instead of the real Demucs backend.
    """
    yield

    try:
        import auto_voice.audio.separation as separation
    except Exception:
        return

    if isinstance(separation.get_model, Mock) or isinstance(separation.apply_model, Mock):
        separation.get_model = None
        separation.apply_model = None


@pytest.fixture(autouse=True)
def restore_test_module_overrides():
    """Restore known sys.modules overrides that some legacy tests leak.

    A few suites patch heavyweight optional dependencies by assigning directly to
    ``sys.modules`` instead of using ``patch.dict``. If those shims survive into
    later tests, they can silently replace real TensorRT or HQ-SVC support
    modules and break unrelated integration coverage.
    """
    tracked_modules = (
        'tensorrt',
        'logger',
        'logger.utils',
        'utils',
        'utils.vocoder',
        'utils.models',
        'utils.models.models_v2_beta',
        'utils.data_preprocessing',
    )
    original_modules = {
        name: sys.modules.get(name)
        for name in tracked_modules
    }

    yield

    for name, original in original_modules.items():
        current = sys.modules.get(name)
        if current is original:
            continue

        is_test_double = isinstance(current, Mock) or (
            isinstance(current, ModuleType)
            and getattr(current, '__file__', None) is None
            and current is not original
        )

        if original is None:
            if is_test_double:
                sys.modules.pop(name, None)
            continue

        if is_test_double or current is None:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def cleanup_cuda_test_state():
    """Release cached CUDA state between tests to reduce cross-suite bleed.

    The full coverage gate exercises many GPU-heavy pipelines before the real
    TensorRT integration suite. Clearing cached CUDA allocations after each test
    keeps those long runs from poisoning later TensorRT engine builds.

    Avoid ``torch.cuda.synchronize()`` here. On some lightweight API suites the
    CUDA runtime is nominally available but not in a safe state for a blocking
    global synchronize during fixture teardown, which can wedge pytest after all
    assertions have already completed.
    """
    yield

    try:
        import gc
        import torch
    except Exception:
        return

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def reset_karaoke_api_test_state():
    """Clear leaked karaoke module globals between tests.

    The karaoke API still exposes a small compatibility cache for output device
    selection. Some tests mutate that module-global cache directly, which can
    otherwise bleed into later tests that use a fresh ``AppStateStore`` and
    expect default routing targets.
    """
    yield

    try:
        from auto_voice.web import karaoke_api
    except Exception:
        return

    karaoke_api._device_config['speaker_device'] = None
    karaoke_api._device_config['headphone_device'] = None
    karaoke_api._uploaded_songs.clear()
    karaoke_api._separation_jobs.clear()
    karaoke_api._active_sessions.clear()


@pytest.fixture
def sample_audio():
    """Generate a simple sine wave audio sample."""
    sr = 22050
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440Hz sine wave with some harmonics
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    audio = audio.astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_audio_file(sample_audio, tmp_path):
    """Create a temporary audio file."""
    import soundfile as sf
    audio, sr = sample_audio
    path = str(tmp_path / "test_audio.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def short_audio():
    """Very short audio (1 second) for edge case testing."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def short_audio_file(short_audio, tmp_path):
    """Create a short audio file (below minimum duration for cloning)."""
    import soundfile as sf
    audio, sr = short_audio
    path = str(tmp_path / "short_audio.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def profiles_dir(tmp_path):
    """Temporary directory for voice profiles."""
    d = tmp_path / "voice_profiles"
    d.mkdir()
    return str(d)


@pytest.fixture
def flask_app():
    """Create a test Flask app with ML components disabled."""
    from auto_voice.web.app import create_app
    app, socketio = create_app(config={
        'TESTING': True,
        'singing_conversion_enabled': False,
        'voice_cloning_enabled': False,
    })
    return app


@pytest.fixture
def flask_app_full():
    """Create a test Flask app with ML components enabled."""
    from auto_voice.web.app import create_app
    app, socketio = create_app(config={'TESTING': True})
    return app


@pytest.fixture
def client(flask_app):
    """Flask test client without ML components."""
    return flask_app.test_client()


@pytest.fixture
def client_full(flask_app_full):
    """Flask test client with ML components."""
    return flask_app_full.test_client()


@pytest.fixture
def voice_cloner(profiles_dir):
    """VoiceCloner instance with temp profile storage."""
    from auto_voice.inference.voice_cloner import VoiceCloner
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return VoiceCloner(device=device, profiles_dir=profiles_dir)


@pytest.fixture
def singing_pipeline(voice_cloner):
    """SingingConversionPipeline with ModelManager pre-loaded (random weights)."""
    from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
    from auto_voice.inference.model_manager import ModelManager
    from auto_voice.models.so_vits_svc import SoVitsSvc
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline = SingingConversionPipeline(
        device=device, voice_cloner=voice_cloner,
        config={'speaker_id': 'default'}
    )

    # Pre-load ModelManager with random-weight models for testing
    # Use content_dim=768 to match ContentVec Layer 12 output
    mm = ModelManager(device=device, config={'speaker_id': 'default'})
    mm.load()  # Random weights
    model = SoVitsSvc({'content_dim': 768, 'pitch_dim': 768})
    model.to(device)
    mm._sovits_models['default'] = model
    pipeline._model_manager = mm

    return pipeline


@pytest.fixture
def audio_processor():
    """AudioProcessor instance."""
    from auto_voice.audio.processor import AudioProcessor
    return AudioProcessor(sample_rate=22050)


@pytest.fixture
def profile_store(profiles_dir):
    """VoiceProfileStore instance."""
    from auto_voice.storage.voice_profiles import VoiceProfileStore
    return VoiceProfileStore(profiles_dir=profiles_dir)


# Multi-speaker fixtures for diarization testing
@pytest.fixture
def multi_speaker_synthetic(tmp_path):
    """Create synthetic multi-speaker audio (2 speakers, alternating)."""
    from tests.fixtures import create_synthetic_multi_speaker

    output_path = str(tmp_path / "multi_speaker_synthetic.wav")
    fixture = create_synthetic_multi_speaker(
        output_path=output_path,
        durations=[
            ("SPEAKER_00", 2.0),
            ("SPEAKER_01", 2.0),
            ("SPEAKER_00", 1.5),
            ("SPEAKER_01", 1.5),
        ],
    )
    return fixture


@pytest.fixture
def multi_speaker_three(tmp_path):
    """Create synthetic multi-speaker audio with 3 speakers."""
    from tests.fixtures import create_synthetic_multi_speaker

    output_path = str(tmp_path / "multi_speaker_three.wav")
    fixture = create_synthetic_multi_speaker(
        output_path=output_path,
        durations=[
            ("SPEAKER_00", 2.0),
            ("SPEAKER_01", 1.5),
            ("SPEAKER_02", 2.0),
            ("SPEAKER_00", 1.0),
            ("SPEAKER_01", 1.5),
        ],
    )
    return fixture


@pytest.fixture
def duet_fixture(tmp_path):
    """Create duet fixture from real audio samples (Conor + William).

    Returns None if quality samples are not available.
    """
    from tests.fixtures.multi_speaker_fixtures import create_duet_fixture

    return create_duet_fixture(output_dir=str(tmp_path))


@pytest.fixture
def interview_fixture(tmp_path):
    """Create interview-style fixture with longer speaker turns.

    Returns None if quality samples are not available.
    """
    from tests.fixtures.multi_speaker_fixtures import create_interview_fixture

    return create_interview_fixture(output_dir=str(tmp_path))
