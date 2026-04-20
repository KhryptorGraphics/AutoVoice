"""Coverage tests for KaraokeManager — mock separator and audio I/O."""
import pytest
import torch
import numpy as np
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_separator():
    sep = MagicMock()
    vocals = torch.randn(1, 1, 44100)
    instrumental = torch.randn(1, 1, 44100)
    sep.separate.return_value = (vocals, instrumental)
    return sep


@pytest.fixture
def karaoke_manager(tmp_path):
    from auto_voice.web.karaoke_manager import KaraokeManager
    mgr = KaraokeManager(
        device=torch.device('cpu'),
        output_dir=str(tmp_path / "output"),
        max_workers=1,
    )
    return mgr


class TestKaraokeManagerInit:
    def test_init_creates_output_dir(self, tmp_path):
        from auto_voice.web.karaoke_manager import KaraokeManager
        output = tmp_path / "karaoke"
        mgr = KaraokeManager(output_dir=str(output))
        assert output.exists()

    def test_init_defaults(self, karaoke_manager):
        assert karaoke_manager.max_workers == 1

    def test_init_custom(self, tmp_path):
        from auto_voice.web.karaoke_manager import KaraokeManager
        mgr = KaraokeManager(output_dir=str(tmp_path), max_workers=4)
        assert mgr.max_workers == 4


class TestKaraokeManagerSeparation:
    def test_start_separation(self, karaoke_manager, mock_separator, tmp_path):
        # Create a test audio file
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        with patch.object(karaoke_manager, '_get_separator', return_value=mock_separator):
            with patch('auto_voice.web.karaoke_manager.load_audio') as mock_load:
                mock_load.return_value = (torch.randn(1, 44100), 44100)
                with patch('auto_voice.web.karaoke_manager.save_audio'):
                    started = karaoke_manager.start_separation("job1", str(audio_path))
                    assert started is True

    def test_start_separation_duplicate_job(self, karaoke_manager, tmp_path):
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        karaoke_manager._jobs['job1'] = {'status': 'queued'}
        started = karaoke_manager.start_separation("job1", str(audio_path))
        assert started is False

    def test_get_job_status(self, karaoke_manager):
        karaoke_manager._jobs['job1'] = {'status': 'completed', 'progress': 100}
        status = karaoke_manager.get_job_status('job1')
        assert status['status'] == 'completed'

    def test_get_job_status_not_found(self, karaoke_manager):
        status = karaoke_manager.get_job_status('nonexistent')
        assert status is None

    def test_get_separated_paths_completed(self, karaoke_manager):
        karaoke_manager._jobs['job1'] = {
            'status': 'completed',
            'vocals_path': '/tmp/vocals.wav',
            'instrumental_path': '/tmp/instrumental.wav',
        }
        paths = karaoke_manager.get_separated_paths('job1')
        assert paths == ('/tmp/vocals.wav', '/tmp/instrumental.wav')

    def test_get_separated_paths_not_completed(self, karaoke_manager):
        karaoke_manager._jobs['job1'] = {'status': 'processing'}
        paths = karaoke_manager.get_separated_paths('job1')
        assert paths is None

    def test_get_separated_paths_not_found(self, karaoke_manager):
        paths = karaoke_manager.get_separated_paths('nonexistent')
        assert paths is None

    def test_separation_failure(self, karaoke_manager, tmp_path):
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        with patch('auto_voice.web.karaoke_manager.load_audio', side_effect=Exception("Load failed")):
            karaoke_manager.start_separation("fail_job", str(audio_path))
            time.sleep(0.5)  # Wait for thread
            status = karaoke_manager.get_job_status('fail_job')
            assert status is not None
            assert status['status'] == 'failed'

    def test_progress_callback(self, tmp_path):
        callbacks = []
        from auto_voice.web.karaoke_manager import KaraokeManager
        mgr = KaraokeManager(
            output_dir=str(tmp_path),
            max_workers=1,
            progress_callback=lambda jid, prog, st: callbacks.append((jid, prog, st))
        )
        # Must create the job first so _update_job finds it
        mgr._jobs['test'] = {'status': 'queued', 'progress': 0}
        mgr._update_job('test', status='processing', progress=50)
        assert len(callbacks) >= 1
        assert callbacks[0][1] == 50


class TestKaraokeManagerShutdown:
    def test_shutdown(self, karaoke_manager):
        karaoke_manager.shutdown()


class TestLoadAudio:
    def test_load_audio_wav(self, tmp_path):
        """Test load_audio with a real WAV file."""
        import soundfile as sf
        from auto_voice.web.karaoke_manager import load_audio
        # Create test WAV
        data = np.random.randn(44100).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), data, 44100)
        audio, sr = load_audio(str(wav_path))
        assert audio.shape[0] == 1  # mono
        assert sr == 44100


class TestSaveAudio:
    def test_save_audio_wav(self, tmp_path):
        from auto_voice.web.karaoke_manager import save_audio
        audio = torch.randn(1, 44100)
        out_path = str(tmp_path / "out.wav")
        save_audio(out_path, audio, 44100)
        assert Path(out_path).exists()


class TestResampleAudio:
    def test_resample_identity(self):
        from auto_voice.web.karaoke_manager import _resample_audio
        audio = torch.randn(1, 44100)
        out = _resample_audio(audio, 44100, 44100)
        assert torch.equal(out, audio)

    def test_resample_downsample(self):
        from auto_voice.web.karaoke_manager import _resample_audio
        audio = torch.randn(1, 44100)
        out = _resample_audio(audio, 44100, 22050)
        assert out.shape[-1] == 22050
