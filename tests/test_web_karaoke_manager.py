"""Focused unit tests for karaoke_manager.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def test_load_audio_uses_soundfile_for_mono():
    from auto_voice.web.karaoke_manager import load_audio

    mono = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    with patch('auto_voice.web.karaoke_manager.sf.read', return_value=(mono, 22050)):
        audio, sr = load_audio('/tmp/test.wav')

    assert sr == 22050
    assert tuple(audio.shape) == (1, 3)
    assert torch.allclose(audio.squeeze(0), torch.from_numpy(mono))


def test_load_audio_falls_back_to_librosa_for_stereo():
    from auto_voice.web.karaoke_manager import load_audio

    stereo = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32)
    with patch('auto_voice.web.karaoke_manager.sf.read', side_effect=RuntimeError('sf failed')):
        with patch('librosa.load', return_value=(stereo, 44100)):
            audio, sr = load_audio('/tmp/test.mp3')

    assert sr == 44100
    assert tuple(audio.shape) == (2, 3)
    assert torch.allclose(audio, torch.from_numpy(stereo))


def test_save_audio_transposes_channels_for_soundfile(tmp_path: Path):
    from auto_voice.web.karaoke_manager import save_audio

    path = tmp_path / 'out.wav'
    audio = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)

    with patch('auto_voice.web.karaoke_manager.sf.write') as mock_write:
        save_audio(str(path), audio, 24000)

    written = mock_write.call_args[0][1]
    assert written.shape == (2, 2)
    assert np.allclose(written, np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32))


@pytest.fixture
def manager(tmp_path: Path):
    from auto_voice.web.karaoke_manager import KaraokeManager

    return KaraokeManager(
        device=torch.device('cpu'),
        output_dir=str(tmp_path / 'karaoke'),
        max_workers=1,
    )


def test_get_separator_lazy_loads_and_caches(manager):
    from auto_voice.web.karaoke_manager import KaraokeManager

    separator_instance = MagicMock()
    separator_instance.to.return_value = separator_instance
    separator_instance.eval.return_value = separator_instance
    fake_module = types.SimpleNamespace(MelBandRoFormer=MagicMock(return_value=separator_instance))

    with patch.dict(sys.modules, {'auto_voice.audio.separator': fake_module}):
        first = KaraokeManager._get_separator(manager)
        second = KaraokeManager._get_separator(manager)

    assert first is separator_instance
    assert second is separator_instance
    fake_module.MelBandRoFormer.assert_called_once()
    separator_instance.to.assert_called_once_with(manager.device)
    separator_instance.eval.assert_called_once()


def test_get_separator_wraps_initialization_failures(manager):
    from auto_voice.web.karaoke_manager import KaraokeManager

    class BrokenSeparator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('load failed')

    fake_module = types.SimpleNamespace(MelBandRoFormer=BrokenSeparator)
    with patch.dict(sys.modules, {'auto_voice.audio.separator': fake_module}):
        with pytest.raises(RuntimeError, match='Separator initialization failed'):
            KaraokeManager._get_separator(manager)


def test_start_separation_rejects_duplicate_job_ids(manager):
    with patch.object(manager._executor, 'submit'):
        assert manager.start_separation('job-1', '/tmp/song.wav') is True
        assert manager.start_separation('job-1', '/tmp/song.wav') is False


def test_run_separation_completes_and_saves_outputs(manager):
    from auto_voice.web.karaoke_manager import KaraokeManager

    manager._jobs['job-1'] = {
        'status': 'queued',
        'progress': 0,
        'input_path': '/tmp/song.wav',
        'vocals_path': None,
        'instrumental_path': None,
        'error': None,
        'started_at': 0.0,
    }
    source_audio = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)
    fake_separator = MagicMock()
    fake_separator.separate.return_value = (
        torch.tensor([[0.2, 0.1]], dtype=torch.float32),
        torch.tensor([[0.8, 0.4]], dtype=torch.float32),
    )

    class FakeResampler:
        def __call__(self, audio):
            return audio

    with patch('auto_voice.web.karaoke_manager.load_audio', return_value=(source_audio, 32000)):
        with patch('auto_voice.web.karaoke_manager.torchaudio.transforms.Resample', return_value=FakeResampler()):
            with patch.object(KaraokeManager, '_get_separator', return_value=fake_separator):
                with patch('auto_voice.web.karaoke_manager.save_audio') as mock_save:
                    manager._run_separation('job-1', '/tmp/song.wav')

    status = manager.get_job_status('job-1')
    assert status['status'] == 'completed'
    assert status['progress'] == 100
    assert status['vocals_path'].endswith('job-1_vocals.wav')
    assert status['instrumental_path'].endswith('job-1_instrumental.wav')
    assert manager.get_separated_paths('job-1') == (
        status['vocals_path'],
        status['instrumental_path'],
    )
    assert mock_save.call_count == 2


def test_run_separation_marks_failure_on_errors(manager):
    manager._jobs['job-err'] = {
        'status': 'queued',
        'progress': 0,
        'input_path': '/tmp/song.wav',
        'vocals_path': None,
        'instrumental_path': None,
        'error': None,
        'started_at': 0.0,
    }

    with patch('auto_voice.web.karaoke_manager.load_audio', side_effect=RuntimeError('boom')):
        manager._run_separation('job-err', '/tmp/song.wav')

    status = manager.get_job_status('job-err')
    assert status['status'] == 'failed'
    assert 'boom' in status['error']


def test_update_job_invokes_callback_and_swallows_callback_errors(manager):
    callback = MagicMock(side_effect=RuntimeError('callback failed'))
    manager.progress_callback = callback
    manager._jobs['job-cb'] = {'status': 'queued', 'progress': 0}

    manager._update_job('job-cb', status='processing', progress=25)

    status = manager.get_job_status('job-cb')
    assert status['status'] == 'processing'
    assert status['progress'] == 25
    callback.assert_called_once_with('job-cb', 25, 'processing')


def test_get_job_status_returns_copy(manager):
    manager._jobs['job-copy'] = {'status': 'queued', 'progress': 5}

    status = manager.get_job_status('job-copy')
    status['progress'] = 99

    assert manager._jobs['job-copy']['progress'] == 5


def test_get_separated_paths_requires_completed_status(manager):
    manager._jobs['job-pending'] = {'status': 'processing'}
    assert manager.get_separated_paths('job-pending') is None


def test_shutdown_stops_executor(manager):
    with patch.object(manager._executor, 'shutdown') as mock_shutdown:
        manager.shutdown()

    mock_shutdown.assert_called_once_with(wait=False)
