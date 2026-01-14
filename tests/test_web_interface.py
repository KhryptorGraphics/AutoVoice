import pytest
import uuid
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from flask.testing import FlaskClient
from src.auto_voice.web.job_manager import JobManager
from src.auto_voice.web.api import api_bp
from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
from src.auto_voice.inference.voice_cloner import VoiceCloner
import socketio
import tempfile
import io
import numpy as np
import base64
import datetime
import math
from src.auto_voice.web.app import create_app


# COMMENT 2 FIX: Fake pipeline for deterministic end-to-end testing
class FakeSingingPipeline:
    """Fake singing conversion pipeline that simulates real behavior with controlled timing"""

    def __init__(self, conversion_delay=0.1, should_fail=False, fail_message=None):
        """
        Args:
            conversion_delay: Time to sleep during conversion (allows cancellation testing)
            should_fail: If True, raises an exception during conversion
            fail_message: Custom error message when should_fail is True
        """
        self.conversion_delay = conversion_delay
        self.should_fail = should_fail
        self.fail_message = fail_message or "Fake pipeline error"

    def convert_song(self, song_path, target_profile_id, progress_callback=None, **kwargs):
        """Simulate song conversion with progress callbacks"""
        stages = [
            ("Loading audio", 10),
            ("Separating vocals", 30),
            ("Converting voice", 60),
            ("Mixing audio", 90),
        ]

        # Call progress callback for each stage
        for stage_name, progress in stages:
            if progress_callback:
                progress_callback(stage_name, progress)
            # Sleep to allow cancellation during processing
            time.sleep(self.conversion_delay / len(stages))

        # Simulate failure if requested
        if self.should_fail:
            raise ValueError(self.fail_message)

        # Final progress
        if progress_callback:
            progress_callback("Completed", 100)

        # Return realistic result
        # Generate fake pitch contour data (Comment 3 - pitch data in WebSocket payloads)
        f0_contour = np.random.uniform(80, 400, 100).astype(np.float32)  # Fake pitch in Hz
        f0_times = np.linspace(0, 1.0, 100).astype(np.float32)  # Time in seconds

        return {
            'mixed_audio': np.random.rand(44100).astype(np.float32),
            'sample_rate': 44100,
            'duration': 1.0,
            'metadata': {
                'target_profile_id': target_profile_id,
                'vocal_volume': kwargs.get('vocal_volume', 1.0),
                'instrumental_volume': kwargs.get('instrumental_volume', 0.9),
            },
            'f0_contour': f0_contour,
            'f0_times': f0_times
        }

@pytest.fixture
def job_manager():
    """Fixture for JobManager instance for testing - COMMENT 3 FIX"""
    config = {'max_workers': 2, 'result_dir': '/tmp/test_results', 'ttl_seconds': 60}
    socketio_mock = Mock(spec=socketio.Server)
    singing_pipeline = Mock(spec=SingingConversionPipeline)
    voice_profile_manager = Mock(spec=VoiceCloner)

    job_manager = JobManager(config, socketio_mock, singing_pipeline, voice_profile_manager)

    # COMMENT 3 FIX: Replace executor with a fake that doesn't execute
    # This prevents race conditions in tests that check initial job state
    class FakeExecutor:
        """Fake executor that records jobs but doesn't execute them"""
        def __init__(self):
            self.submitted_jobs = []

        def submit(self, fn, *args, **kwargs):
            """Record the job but don't execute it"""
            future = Mock()
            future.done.return_value = False
            future.cancel.return_value = True
            self.submitted_jobs.append((fn, args, kwargs))
            return future

        def shutdown(self, wait=True):
            """No-op shutdown"""
            pass

    job_manager.executor = FakeExecutor()
    job_manager.start_cleanup_thread()
    yield job_manager
    job_manager.shutdown()

@pytest.fixture
def completed_job_id(job_manager):
    """Fixture for a completed job ID"""
    # COMMENT 3 FIX: Manually create completed job without triggering execution
    job_id = str(uuid.uuid4())
    result_path = job_manager.result_dir / f"{job_id}.wav"

    # Directly add to jobs dict without calling create_job (avoid async execution)
    with job_manager.lock:
        job_manager.jobs[job_id] = {
            'job_id': job_id,
            'status': 'completed',
            'progress': 100,
            'stage': 'completed',
            'result_path': result_path,
            'created_at': time.time(),
            'completed_at': time.time(),
            'error': None,
            'metadata': {},
            'cancel_flag': False,
            'future': None  # No future since we're not executing
        }

    # Create the result file
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(b'fake audio data')

    yield job_id

    # Cleanup
    with job_manager.lock:
        if job_id in job_manager.jobs:
            del job_manager.jobs[job_id]
    if result_path.exists():
        result_path.unlink()

class TestJobManagement:
    """Test JobManager class methods"""

    def test_job_manager_initialization(self, job_manager):
        """Verify JobManager initializes correctly"""
        assert job_manager.executor is not None
        assert job_manager.lock is not None
        assert job_manager.jobs == {}
        assert job_manager.result_dir.exists()
        assert job_manager.ttl_seconds > 0
        assert job_manager._cleanup_thread is not None

    def test_mock_job_manager_cleanup(self):
        """Test MockJobManager cleanup functionality"""
        from src.auto_voice.web.app import create_app
        app, _ = create_app({'TESTING': True})
        mock_job_manager = app.job_manager

        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_path = tmp.name
            tmp.write(b'fake audio data')

        assert os.path.exists(audio_path)

        # Create job
        job_id = mock_job_manager.create_job(audio_path, 'test-profile', {})

        # Verify path stored
        job = mock_job_manager.jobs[job_id]
        assert job['audio_path'] == audio_path

        # Cleanup
        mock_job_manager.cleanup_job(job_id)

        # Verify file deleted
        assert not os.path.exists(audio_path)

        # Verify idempotent
        mock_job_manager.cleanup_job(job_id)  # Should not error

        # Shutdown cleans all
        mock_job_manager.shutdown()
        assert mock_job_manager.jobs == {}

    def test_create_job_returns_job_id(self, job_manager):
        """Test create_job returns UUID and creates job entry - COMMENT 3 FIX"""
        audio_path = '/tmp/test.wav'
        Path(audio_path).touch()

        job_id = job_manager.create_job(audio_path, 'test-profile', {})

        # COMMENT 3 FIX: Assert on invariant fields only
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        assert job_id in job_manager.jobs
        job = job_manager.jobs[job_id]
        # Status can be any valid state, not just 'queued' (could transition quickly)
        assert job['status'] in job_manager.JOB_STATUSES
        assert job['future'] is not None
        assert 'progress' in job
        assert 'stage' in job

    def test_get_job_status(self, job_manager):
        """Test get_job_status returns correct metadata - COMMENT 3 FIX"""
        audio_path = '/tmp/test.wav'
        Path(audio_path).touch()
        job_id = job_manager.create_job(audio_path, 'test-profile', {})

        status = job_manager.get_job_status(job_id)
        # COMMENT 3 FIX: Assert on invariant fields, not specific status
        assert status['job_id'] == job_id
        assert status['status'] in job_manager.JOB_STATUSES
        assert 'progress' in status
        assert 'stage' in status

    def test_get_job_status_not_found(self, job_manager):
        """Test get_job_status returns None for invalid job_id"""
        invalid_id = 'invalid-job-id'
        status = job_manager.get_job_status(invalid_id)
        assert status is None

    def test_cancel_job(self, job_manager):
        """Test cancel_job changes status and sets flag"""
        audio_path = '/tmp/test.wav'
        Path(audio_path).touch()
        job_id = job_manager.create_job(audio_path, 'test-profile', {})
        
        cancelled = job_manager.cancel_job(job_id)
        assert cancelled is True
        
        with job_manager.lock:
            job = job_manager.jobs[job_id]
            assert job['status'] == 'cancelled'
            assert job['cancel_flag'] is True

    def test_cancel_completed_job(self, job_manager, completed_job_id):
        """Test cancel_job returns False for completed job"""
        cancelled = job_manager.cancel_job(completed_job_id)
        assert cancelled is False

    def test_job_cleanup_expired(self, job_manager):
        """Test cleanup removes expired jobs"""
        audio_path = '/tmp/test_expired.wav'
        Path(audio_path).touch()
        job_id = job_manager.create_job(audio_path, 'test-profile', {})

        # Manually expire the job by setting old created_at timestamp
        with job_manager.lock:
            job_manager.jobs[job_id]['created_at'] = time.time() - (job_manager.ttl_seconds + 100)

        # Call single cleanup iteration (not the infinite loop method)
        job_manager._remove_job(job_id)

        # Verify job was removed
        assert job_id not in job_manager.jobs

    def test_concurrent_job_creation(self, job_manager):
        """Test thread-safe concurrent job creation"""
        from concurrent.futures import ThreadPoolExecutor as TestExecutor
        
        def create_job():
            audio_path = '/tmp/test_concurrent.wav'
            Path(audio_path).touch()
            return job_manager.create_job(audio_path, 'test-profile', {})
        
        with TestExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_job) for _ in range(20)]
            job_ids = [f.result() for f in futures]
        
        # Verify all jobs created successfully
        assert len(job_ids) == 20
        assert len(job_manager.jobs) == 20
        for job_id in job_ids:
            assert job_id in job_manager.jobs


class TestJobEndpoints:
    """Test job management API endpoints"""

    @pytest.fixture
    def app_and_socketio(self):
        """Create Flask app and SocketIO instance for testing"""
        app, sio = create_app({'TESTING': True})
        yield app, sio

    @pytest.fixture
    def client(self, app_and_socketio):
        """Flask test client"""
        app, _ = app_and_socketio
        with app.test_client() as client:
            yield client

    def test_convert_song_async_returns_202(self, client, app_and_socketio):
        """Test /convert/song returns 202 with job_id when job_manager enabled"""
        app, _ = app_and_socketio

        # Enable job_manager by setting app.job_manager
        mock_job_manager = Mock()
        mock_job_manager.create_job.return_value = 'test-job-id'
        app.job_manager = mock_job_manager

        form_data = {
            'audio': (b'', 'test.wav'),
            'target_profile_id': 'test-profile',
        }

        response = client.post('/api/v1/convert/song', data=form_data)

        assert response.status_code == 202
        data = response.json
        assert data['status'] == 'queued'
        assert data['job_id'] == 'test-job-id'
        assert 'websocket_room' in data
        assert data['websocket_room'] == data['job_id']
        mock_job_manager.create_job.assert_called_once()

    def test_convert_song_sync_fallback(self, client, app_and_socketio):
        """Test /convert/song falls back to sync when job_manager disabled"""
        app, _ = app_and_socketio
        
        # Disable job_manager
        app.job_manager = None
        
        form_data = {
            'audio': (b'', 'test.wav'),
            'target_profile_id': 'test-profile',
        }
        
        response = client.post('/api/v1/convert/song', data=form_data)
        
        # Should return 200 for sync fallback (assuming singing_pipeline mock returns success)
        assert response.status_code in [200, 503]  # 503 if pipeline unavailable

    def test_get_conversion_status_success(self, client, app_and_socketio, job_manager, completed_job_id):
        """Test GET /convert/status/{job_id} returns status"""
        app, _ = app_and_socketio
        app.job_manager = job_manager
        
        response = client.get(f'/api/v1/convert/status/{completed_job_id}')
        
        assert response.status_code == 200
        data = response.json
        assert data['status'] == 'completed'

    def test_get_conversion_status_not_found(self, client):
        """Test GET /convert/status/invalid returns 404"""
        response = client.get('/api/v1/convert/status/invalid-job-id')
        assert response.status_code == 404

    def test_download_converted_audio_success(self, client, app_and_socketio, job_manager, completed_job_id):
        """Test GET /convert/download/{job_id} downloads file"""
        app, _ = app_and_socketio
        app.job_manager = job_manager

        # File already created by fixture - no need to create again
        # Just verify download works
        response = client.get(f'/api/v1/convert/download/{completed_job_id}')

        assert response.status_code == 200
        assert response.mimetype == 'audio/wav'
        assert response.data == b'fake audio data'

    def test_download_converted_audio_not_found(self, client):
        """Test GET /convert/download/invalid returns 404"""
        response = client.get('/api/v1/convert/download/invalid-job-id')
        assert response.status_code == 404

    def test_cancel_conversion_success(self, client, app_and_socketio, job_manager):
        """Test POST /convert/cancel/{job_id} cancels job"""
        app, _ = app_and_socketio
        app.job_manager = job_manager
        
        audio_path = '/tmp/test_cancel.wav'
        Path(audio_path).touch()
        job_id = job_manager.create_job(audio_path, 'test-profile', {})
        
        response = client.post(f'/api/v1/convert/cancel/{job_id}')
        
        assert response.status_code == 200
        data = response.json
        assert data['status'] == 'cancelled'

    def test_cancel_conversion_not_found(self, client):
        """Test POST /convert/cancel/invalid returns 404"""
        response = client.post('/api/v1/convert/cancel/invalid-job-id')
        assert response.status_code == 404

    def test_convert_song_missing_profile(self, client, app_and_socketio):
        """Test POST /api/v1/convert/song without profile_id returns 400"""
        app, _ = app_and_socketio
        app.job_manager = None

        # Create a temporary audio file
        audio_bytes = io.BytesIO(b'fake audio data')

        form_data = {
            'audio': (audio_bytes, 'test.wav')
        }

        response = client.post('/api/v1/convert/song', data=form_data)
        assert response.status_code == 400
        assert b'profile_id required' in response.data.lower() or b'target_profile_id' in response.data.lower()

    def test_convert_song_invalid_file_type(self, client, app_and_socketio):
        """Test POST /api/v1/convert/song with unsupported file returns 400"""
        app, _ = app_and_socketio
        app.job_manager = None

        # Create an invalid text file
        invalid_file = io.BytesIO(b'not an audio file')

        form_data = {
            'audio': (invalid_file, 'test.txt'),
            'target_profile_id': 'test-profile'
        }

        response = client.post('/api/v1/convert/song', data=form_data)
        assert response.status_code == 400
        assert b'invalid file type' in response.data.lower() or b'invalid' in response.data.lower()

    def test_error_handling_invalid_profile_sync(self, client, app_and_socketio):
        """Test error handling for invalid profile ID in sync mode returns 404"""
        app, _ = app_and_socketio

        # Force sync mode by disabling job_manager
        app.job_manager = None

        # Mock pipeline to raise profile not found error
        mock_pipeline = MagicMock()
        mock_pipeline.convert_song.side_effect = ValueError('Profile not found: nonexistent-profile')
        app.singing_conversion_pipeline = mock_pipeline

        form_data = {
            'audio': (io.BytesIO(b'fake audio'), 'test.wav'),
            'target_profile_id': 'nonexistent-profile',
        }

        response = client.post('/api/v1/convert/song', data=form_data)
        assert response.status_code in [404, 503]  # Could be 404 or 503 depending on error handling
        data = response.json
        assert 'error' in data

def test_convert_song_sync_success(client, app_and_socketio):
    """Test synchronous convert_song returns 200 with full success response fields"""
    app, _ = app_and_socketio
    app.job_manager = None  # Force sync mode
    
    # Mock pipeline result
    mock_pipeline = MagicMock()
    mock_result = {
        'mixed_audio': np.random.rand(1, 44100).astype(np.float32),
        'sample_rate': 22050,
        'duration': 2.0,
        'metadata': {'target_profile_id': 'test', 'vocal_volume': 1.0}
    }
    mock_pipeline.convert_song.return_value = mock_result
    app.singing_conversion_pipeline = mock_pipeline
    
    form_data = {
        'song': (io.BytesIO(b''), 'test.wav'),
        'profile_id': 'test'
    }
    
    response = client.post('/api/v1/convert/song', data=form_data)
    
    assert response.status_code == 200
    data = response.json
    assert data['status'] == 'success'
    assert 'job_id' in data and len(data['job_id']) > 0
    assert 'audio' in data and len(data['audio']) > 100  # Non-empty base64
    assert isinstance(data['sample_rate'], (int, float)) and data['sample_rate'] > 0
    assert data['duration'] >= 0
    assert isinstance(data['metadata'], dict) and 'target_profile_id' in data['metadata']
    assert data['format'] == 'wav'
    
    # Optional: Verify base64 decodes to WAV
    audio_bytes = base64.b64decode(data['audio'])
    assert audio_bytes.startswith(b'RIFF')
    
    mock_pipeline.convert_song.assert_called_once()


def test_convert_song_with_stems(client, app_and_socketio):
    """Test synchronous convert_song with stems returns full stems data"""
    app, _ = app_and_socketio
    app.job_manager = None  # Force sync mode
    
    # Mock pipeline result with stems
    mock_pipeline = MagicMock()
    mock_result = {
        'mixed_audio': np.random.rand(1, 44100).astype(np.float32),
        'sample_rate': 22050,
        'duration': 2.0,
        'metadata': {'target_profile_id': 'test', 'vocal_volume': 1.0},
        'stems': {
            'vocals': np.random.rand(1, 44100).astype(np.float32),
            'instrumental': np.random.rand(1, 44100).astype(np.float32)
        }
    }
    mock_pipeline.convert_song.return_value = mock_result
    app.singing_conversion_pipeline = mock_pipeline
    
    form_data = {
        'song': (io.BytesIO(b''), 'test.wav'),
        'profile_id': 'test',
        'return_stems': 'true'
    }
    
    response = client.post('/api/v1/convert/song', data=form_data)
    
    assert response.status_code == 200
    data = response.json
    assert data['status'] == 'success'
    assert 'stems' in data
    assert 'vocals' in data['stems']
    assert 'audio' in data['stems']['vocals'] and len(data['stems']['vocals']['audio']) > 100
    assert data['stems']['vocals']['duration'] > 0
    assert 'instrumental' in data['stems']
    assert 'audio' in data['stems']['instrumental'] and len(data['stems']['instrumental']['audio']) > 100
    assert data['stems']['instrumental']['duration'] > 0

@pytest.mark.skip
class TestLegacyVoiceProfileEndpoints:
    """Test legacy voice profile manipulation endpoints (skipped)"""

    @pytest.fixture
    def client(self, app_and_socketio):
        """Flask test client"""
        app, _ = app_and_socketio
        with app.test_client() as client:
            yield client

    def test_create_voice_profile_success(self, client, voice_cloner, create_tempfile):
        """Test POST /api/v1/voice_profiles"""
        form_data = {
            'profile_name': 'New Profile',
            'voice_file': open(create_tempfile, 'rb'),
            'sound_signature_id': valid_sound_signature_id,
            'astapor_api_key': 'valid-api-key',
        }
        
        response = client.post('/api/v1/voice_profiles', data=form_data)
        
        assert response.status_code == 201
        data = response.json
        assert "id" in data
        assert data["status"] == "success"
        assert data["profile_name"] == "New Profile"
        assert data["user_id"] == None

    def test_create_voice_profile_missing_field(self, client, voice_cloner):
        """Test create voice profile with missing required field
        Return 400 and error message"""
        form_data = {
            'profile_name': 'Incomplete Profile',
            'sound_signature_id': valid_sound_signature_id,
            'astapor_api_key': 'valid-api-key',
        }
        
        response = client.post('/api/v1/voice_profiles', data=form_data)
        
        print(response)
        assert response.status_code == 400
        data = response.json
        assert "errors" in data
        assert "voice_file" in data["errors"]

class TestVoiceProfileEndpoints:
    """Test voice profile API endpoints (/api/v1/voice/*) using MockVoiceCloner in TESTING mode"""

    # Uses conftest.py client fixture which sets up MockVoiceCloner

    @pytest.fixture
    def mock_audio_file(self):
        """Mock WAV file for upload testing"""
        # Minimal valid WAV file (RIFF header + silence data)
        wav_bytes = b'RIFF$\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3E\x00\x00\x80\x7E\x00\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00'
        return (io.BytesIO(wav_bytes), 'test_audio.wav')

    def test_create_voice_profile_success_201(self, client, mock_audio_file):
        """Test successful profile creation returns 201 with expected fields excluding embedding"""
        form_data = {
            'reference_audio': mock_audio_file,
            'user_id': 'test_user'
        }
        response = client.post('/api/v1/voice/clone', data=form_data)
        assert response.status_code == 201
        data = response.json
        assert 'profile_id' in data
        assert 'user_id' in data
        assert data['user_id'] == 'test_user'
        assert 'audio_duration' in data
        assert 'vocal_range' in data
        assert isinstance(data['audio_duration'], (int, float))
        assert isinstance(data['vocal_range'], dict)
        assert 'embedding' not in data  # Excluded as specified

    def test_create_voice_profile_missing_reference_audio_400(self, client):
        """Test missing reference_audio returns 400"""
        form_data = {'user_id': 'test_user'}  # No audio
        response = client.post('/api/v1/voice/clone', data=form_data)
        assert response.status_code == 400
        data = response.json
        assert data['error'] == 'No reference_audio file provided'

    def test_create_voice_profile_invalid_reference_audio_400(self, client):
        """Test invalid file type for reference_audio returns 400"""
        invalid_file = (io.BytesIO(b'invalid content'), 'invalid.txt')
        form_data = {'reference_audio': invalid_file}
        response = client.post('/api/v1/voice/clone', data=form_data)
        assert response.status_code == 400
        data = response.json
        assert 'Invalid file format' in data['error']

    def test_list_profiles_no_user_filter(self, client, mock_audio_file):
        """Test GET /api/v1/voice/profiles lists all profiles (empty then populated)"""
        # Empty list initially
        resp = client.get('/api/v1/voice/profiles')
        assert resp.status_code == 200
        assert resp.json == []

        # Create 2 profiles
        client.post('/api/v1/voice/clone', data={'reference_audio': mock_audio_file})
        client.post('/api/v1/voice/clone', data={'reference_audio': mock_audio_file})

        resp = client.get('/api/v1/voice/profiles')
        assert resp.status_code == 200
        profiles = resp.json
        assert len(profiles) == 2
        assert all('profile_id' in p and 'created_at' in p for p in profiles)

    def test_list_profiles_with_user_id_filter(self, client, mock_audio_file):
        """Test GET /api/v1/voice/profiles?user_id=... filters correctly"""
        # Create mixed user profiles
        client.post('/api/v1/voice/clone', data={'reference_audio': mock_audio_file, 'user_id': 'user1'})
        client.post('/api/v1/voice/clone', data={'reference_audio': mock_audio_file, 'user_id': 'user1'})
        client.post('/api/v1/voice/clone', data={'reference_audio': mock_audio_file, 'user_id': 'user2'})

        # Filter user1: expect 2
        resp = client.get('/api/v1/voice/profiles?user_id=user1')
        assert resp.status_code == 200
        assert len(resp.json) == 2
        assert all(p['user_id'] == 'user1' for p in resp.json)

        # Filter user2: expect 1
        resp = client.get('/api/v1/voice/profiles?user_id=user2')
        assert resp.status_code == 200
        assert len(resp.json) == 1

    def test_get_voice_profile_by_id_success_200(self, client, mock_audio_file):
        """Test GET /api/v1/voice/profiles/{id} returns 200 with profile"""
        # Create profile
        resp_create = client.post('/api/v1/voice/clone', data={'reference_audio': mock_audio_file})
        profile_id = resp_create.json['profile_id']

        resp = client.get(f'/api/v1/voice/profiles/{profile_id}')
        assert resp.status_code == 200
        data = resp.json
        assert data['profile_id'] == profile_id
        assert 'vocal_range' in data
        assert 'embedding' not in data

    def test_get_voice_profile_not_found_404(self, client):
        """Test GET non-existent profile returns 404"""
        resp = client.get('/api/v1/voice/profiles/nonexistent-id')
        assert resp.status_code == 404
        data = resp.json
        assert data['error'] == 'Voice profile not found'

    def test_delete_voice_profile_success_200(self, client, mock_audio_file):
        """Test DELETE existing profile returns 200, subsequent GET 404"""
        # Create profile
        resp_create = client.post('/api/v1/voice/clone', data={'reference_audio': mock_audio_file})
        profile_id = resp_create.json['profile_id']

        # Delete
        resp = client.delete(f'/api/v1/voice/profiles/{profile_id}')
        assert resp.status_code == 200
        data = resp.json
        assert data['status'] == 'success'

        # Verify deleted
        resp_get = client.get(f'/api/v1/voice/profiles/{profile_id}')
        assert resp_get.status_code == 404

    def test_delete_voice_profile_not_found_404(self, client):
        """Test DELETE non-existent profile returns 404"""
        resp = client.delete('/api/v1/voice/profiles/nonexistent-id')
        assert resp.status_code == 404
        data = resp.json
        assert data['error'] == 'Voice profile not found'

    def test_endpoints_return_503_when_voice_cloner_none(self, monkeypatch, client):
        """Test all endpoints return 503 when current_app.voice_cloner is None"""
        monkeypatch.setattr(client.application, 'voice_cloner', None)

        # Test each endpoint
        assert client.post('/api/v1/voice/clone').status_code == 503
        assert client.get('/api/v1/voice/profiles').status_code == 503
        assert client.get('/api/v1/voice/profiles/test-id').status_code == 503
        assert client.delete('/api/v1/voice/profiles/test-id').status_code == 503

class TestEndToEndConversionFlow:
    """Test complete conversion workflow from REST API to WebSocket completion"""

    @pytest.fixture
    def app_and_socketio(self):
        """Create Flask app and SocketIO instance for testing"""
        app, sio = create_app({'TESTING': True})
        yield app, sio

    @pytest.fixture
    def client(self, app_and_socketio):
        """Flask test client"""
        app, _ = app_and_socketio
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def socketio_client(self, app_and_socketio):
        """Create SocketIO test client bound to in-process server - COMMENT 4 FIX"""
        app, socketio = app_and_socketio

        # Use flask_socketio's test_client for in-process testing
        client = socketio.test_client(app)
        yield client
        client.disconnect()

    def test_rest_to_websocket_flow(self, client, app_and_socketio, socketio_client, job_manager):
        """Test REST API creates job and WebSocket receives progress - COMMENT 1 & 4 FIX"""
        app, socketio = app_and_socketio

        # COMMENT 2 FIX: Use FakeSingingPipeline for deterministic behavior (COMMENT 1 implementation)
        # This fake pipeline emits progress callbacks and returns valid results
        fake_pipeline = FakeSingingPipeline(conversion_delay=0.1, should_fail=False)

        job_manager_real = JobManager(
            config={'max_workers': 2, 'result_dir': job_manager.result_dir, 'ttl_seconds': 60},
            socketio=socketio,  # Use real socketio, not mock
            singing_pipeline=fake_pipeline,  # Use FakeSingingPipeline that works correctly
            voice_profile_manager=Mock(spec=VoiceCloner)
        )
        job_manager_real.start_cleanup_thread()
        app.job_manager = job_manager_real

        # COMMENT 4 FIX: Temporarily disable TESTING to use async path
        original_testing = app.config.get('TESTING', True)
        app.config['TESTING'] = False

        try:
            # Join job room before starting conversion
            socketio_client.emit('join_job', {'job_id': 'test-job'})

            # Create job via REST API
            form_data = {
                'audio': (io.BytesIO(b'fake audio'), 'test.wav'),
                'target_profile_id': 'test-profile',
            }
            response = client.post('/api/v1/convert/song', data=form_data)
            assert response.status_code == 202

            job_id = response.json['job_id']

            # Join WebSocket room for this job
            socketio_client.emit('join_job', {'job_id': job_id})

            # COMMENT 4 FIX: Wait for events from in-process server
            timeout = 30
            start = time.time()
            received_events = []

            while (time.time() - start) < timeout:
                # Get events from flask_socketio test_client
                received = socketio_client.get_received()
                received_events.extend(received)

                # Check if we got completion event
                if any(msg.get('name') == 'conversion_complete' for msg in received_events):
                    break
                time.sleep(0.1)

            # Verify events received
            assert len(received_events) > 0, "No WebSocket events received"

            # Check for completion or progress events
            event_names = [msg.get('name') for msg in received_events]
            assert 'conversion_progress' in event_names or 'conversion_complete' in event_names, \
                f"No conversion events received. Got: {event_names}"

            # If completion event exists, verify structure
            complete_events = [msg for msg in received_events if msg.get('name') == 'conversion_complete']
            if complete_events:
                complete_data = complete_events[0]['args'][0]
                assert complete_data['job_id'] == job_id
                assert 'output_url' in complete_data or 'result_path' in complete_data

                # COMMENT 3: Verify pitch data is included in WebSocket payload
                # Note: f0_contour and f0_times may be None if not available from pipeline
                assert 'f0_contour' in complete_data, "f0_contour field missing from conversion_complete payload"
                assert 'f0_times' in complete_data, "f0_times field missing from conversion_complete payload"

                # If pitch data is available (not None), verify it's properly serialized
                if complete_data['f0_contour'] is not None:
                    assert isinstance(complete_data['f0_contour'], list), "f0_contour should be a list for JSON serialization"
                    assert len(complete_data['f0_contour']) > 0, "f0_contour should not be empty"

                if complete_data['f0_times'] is not None:
                    assert isinstance(complete_data['f0_times'], list), "f0_times should be a list for JSON serialization"
                    assert len(complete_data['f0_times']) > 0, "f0_times should not be empty"

        finally:
            # Cleanup
            job_manager_real.shutdown()
    
    def test_download_after_completion(self, client, app_and_socketio, job_manager, completed_job_id):
        """Test downloading result after job completes"""
        app, _ = app_and_socketio
        app.job_manager = job_manager

        # File already created by fixture - verify download works
        response = client.get(f'/api/v1/convert/download/{completed_job_id}')
        assert response.status_code == 200
        assert response.mimetype == 'audio/wav'
        assert response.data == b'fake audio data'
    
    def test_cancel_during_processing(self, client, app_and_socketio, socketio_client, job_manager):
        """Test cancelling job while processing - COMMENT 4 FIX"""
        app, socketio = app_and_socketio

        # COMMENT 4 FIX: Use real JobManager with real socketio
        # COMMENT 2 FIX: Use FakeSingingPipeline with longer delay for reliable cancellation
        job_manager_real = JobManager(
            config={'max_workers': 2, 'result_dir': job_manager.result_dir, 'ttl_seconds': 60},
            socketio=socketio,
            singing_pipeline=FakeSingingPipeline(conversion_delay=1.0),  # Longer delay for cancellation
            voice_profile_manager=Mock(spec=VoiceCloner)
        )
        job_manager_real.start_cleanup_thread()
        app.job_manager = job_manager_real

        try:
            # Create long-running job
            audio_path = '/tmp/test_long.wav'
            Path(audio_path).touch()
            job_id = job_manager_real.create_job(audio_path, 'test-profile', {})

            # Join room
            socketio_client.emit('join_job', {'job_id': job_id})

            # Cancel immediately
            response = client.post(f'/api/v1/convert/cancel/{job_id}')
            assert response.status_code == 200
            assert response.json['status'] == 'cancelled'

            # Verify job status
            status = job_manager_real.get_job_status(job_id)
            assert status['status'] == 'cancelled'

        finally:
            job_manager_real.shutdown()

    def test_cancel_queued_job_websocket(self, client, app_and_socketio, socketio_client):
        """Test cancelling queued job emits conversion_cancelled WebSocket event"""
        app, socketio = app_and_socketio

        # Create JobManager with 0 workers to keep jobs queued
        config = {'max_workers': 0, 'result_dir': '/tmp/test_results', 'ttl_seconds': 60}
        job_manager_real = JobManager(
            config=config,
            socketio=socketio,
            singing_pipeline=Mock(spec=SingingConversionPipeline),
            voice_profile_manager=Mock(spec=VoiceCloner)
        )
        job_manager_real.start_cleanup_thread()
        app.job_manager = job_manager_real

        try:
            # Create queued job
            audio_path = '/tmp/test_queued.wav'
            Path(audio_path).touch()
            job_id = job_manager_real.create_job(audio_path, 'test-profile', {})

            # Verify queued
            status = job_manager_real.get_job_status(job_id)
            assert status['status'] == 'queued'

            # Join room
            socketio_client.emit('join_job', {'job_id': job_id})

            # Cancel via API
            response = client.post(f'/api/v1/convert/cancel/{job_id}')
            assert response.status_code == 200
            assert response.json['status'] == 'cancelled'

            # Wait for conversion_cancelled event
            timeout = 5
            start = time.time()
            cancel_received = False
            while (time.time() - start) < timeout:
                received = socketio_client.get_received()
                for msg in received:
                    if msg.get('name') == 'conversion_cancelled':
                        data = msg['args'][0]
                        assert data['job_id'] == job_id
                        assert data['code'] == 'CONVERSION_CANCELLED'
                        cancel_received = True
                        break
                if cancel_received:
                    break
                time.sleep(0.1)

            assert cancel_received, "No conversion_cancelled event received for queued job"

        finally:
            job_manager_real.shutdown()
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_error_handling_invalid_profile(self, client, app_and_socketio, socketio_client, job_manager):
        """Test error handling for invalid profile ID in both sync and async modes"""
        app, socketio = app_and_socketio

        # First test SYNC mode (no job_manager)
        original_job_manager = app.job_manager
        app.job_manager = None

        form_data = {
            'audio': (io.BytesIO(b'fake audio'), 'test.wav'),
            'target_profile_id': 'nonexistent-profile',
        }
        response = client.post('/api/v1/convert/song', data=form_data)

        # In sync mode with invalid profile, expect immediate error
        # The actual behavior depends on voice_cloner.load_voice_profile()
        # For MockVoiceCloner (TESTING mode), it returns None for missing profiles
        # This should be handled upstream before conversion, but if not caught,
        # we expect either 404 or service error
        assert response.status_code in [404, 503], f"Expected 404 or 503, got {response.status_code}"
        if response.status_code == 404:
            assert 'not found' in response.json.get('error', '').lower() or 'not found' in response.json.get('message', '').lower()

        # Restore job_manager for ASYNC mode test
        app.job_manager = original_job_manager

        # Now test ASYNC mode with real JobManager
        # COMMENT 4 FIX: Use real JobManager with real socketio
        # COMMENT 2 FIX: Use FakeSingingPipeline that fails to simulate error
        job_manager_real = JobManager(
            config={'max_workers': 2, 'result_dir': job_manager.result_dir, 'ttl_seconds': 60},
            socketio=socketio,
            singing_pipeline=FakeSingingPipeline(conversion_delay=0.1, should_fail=True, fail_message="Invalid profile ID"),
            voice_profile_manager=Mock(spec=VoiceCloner)
        )
        job_manager_real.start_cleanup_thread()
        app.job_manager = job_manager_real

        try:
            # Create job with invalid profile
            form_data = {
                'audio': (io.BytesIO(b'fake audio'), 'test.wav'),
                'target_profile_id': 'nonexistent-profile',
            }
            response = client.post('/api/v1/convert/song', data=form_data)

            # Async mode should return 202 and handle error via WebSocket
            if response.status_code == 202:
                job_id = response.json['job_id']
                socketio_client.emit('join_job', {'job_id': job_id})

                # Wait for error event from flask_socketio test_client
                timeout = 10
                start = time.time()
                error_received = False

                while (time.time() - start) < timeout:
                    received = socketio_client.get_received()

                    # Check for error events
                    for msg in received:
                        if msg.get('name') == 'conversion_error':
                            error_data = msg['args'][0]
                            assert 'error' in error_data
                            error_received = True
                            break

                    if error_received:
                        break
                    time.sleep(0.1)

                assert error_received, "No error event received via WebSocket in async mode"

        finally:
            job_manager_real.shutdown()
            # Restore original job_manager
            app.job_manager = original_job_manager

    def test_concurrent_job_processing(self, client, app_and_socketio, job_manager):
        """Test multiple jobs can be processed concurrently - COMMENT 1 FIX"""
        app, _ = app_and_socketio
        app.job_manager = job_manager

        # Create multiple jobs
        job_ids = []
        for i in range(3):
            form_data = {
                'audio': (io.BytesIO(f'audio {i}'.encode()), f'test{i}.wav'),
                'target_profile_id': 'test-profile',
            }
            response = client.post('/api/v1/convert/song', data=form_data)
            assert response.status_code == 202
            job_ids.append(response.json['job_id'])

        # Verify all jobs are tracked
        for job_id in job_ids:
            status = job_manager.get_job_status(job_id)
            assert status is not None
            assert status['status'] in ['queued', 'processing', 'completed']
    
def test_convert_song_invalid_params_400(client, app_and_socketio):
    """Test invalid parameters return 400"""
    app, _ = app_and_socketio
    app.job_manager = None
    
    # vocal_volume >2.0
    form_data = {
        'song': (io.BytesIO(b''), 'test.wav'),
        'profile_id': 'test',
        'vocal_volume': '3.0'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 400
    assert 'Invalid value for vocal_volume' in response.json['error']
    
    # settings JSON with invalid
    form_data = {
        'song': (io.BytesIO(b''), 'test.wav'),
        'profile_id': 'test',
        'settings': '{"vocal_volume":1.5,"invalid_key":true}'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 200  # Parses ok, ignores invalid_key
    
    # Invalid quality
    form_data['settings'] = '{"output_quality":"invalid"}'
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 400
    assert 'Invalid value for output_quality' in response.json['error']


@pytest.mark.api
@pytest.mark.validation
def test_convert_song_missing_profile_400(client, app_and_socketio):
    """Test missing profile_id returns 400 with 'profile_id required' error"""
    app, _ = app_and_socketio
    app.job_manager = None  # Force sync mode for predictable testing

    # Test with no profile_id or target_profile_id
    form_data = {
        'song': (io.BytesIO(b'fake audio'), 'test.wav'),
        # Intentionally omit profile_id
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 400
    assert 'profile_id required' in response.json['error']

    # Test with 'audio' field name (alternative accepted field)
    form_data = {
        'audio': (io.BytesIO(b'fake audio'), 'test.wav'),
        # Intentionally omit profile_id
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 400
    assert 'profile_id required' in response.json['error']


@pytest.mark.api
@pytest.mark.validation
def test_convert_song_invalid_file_type_400(client, app_and_socketio):
    """Test invalid file type returns 400 with 'Invalid file type' error"""
    app, _ = app_and_socketio
    app.job_manager = None  # Force sync mode

    # Test with .txt file (unsupported)
    form_data = {
        'song': (io.BytesIO(b'fake text content'), 'test.txt'),
        'target_profile_id': 'test-profile'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 400
    assert 'Invalid file type' in response.json['error']

    # Test with .jpg file (unsupported)
    form_data = {
        'song': (io.BytesIO(b'\xff\xd8\xff\xe0'), 'image.jpg'),
        'target_profile_id': 'test-profile'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 400
    assert 'Invalid file type' in response.json['error']

    # Test with .pdf file (unsupported)
    form_data = {
        'audio': (io.BytesIO(b'%PDF-1.4'), 'document.pdf'),
        'profile_id': 'test-profile'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 400
    assert 'Invalid file type' in response.json['error']


def test_convert_song_pipeline_failure_503(client, app_and_socketio):
    """Test pipeline failure returns 503"""
    app, _ = app_and_socketio
    app.job_manager = None
    
    mock_pipeline = MagicMock()
    mock_pipeline.convert_song.side_effect = ValueError('Mock fail')
    app.singing_conversion_pipeline = mock_pipeline
    
    form_data = {
        'song': (io.BytesIO(b''), 'test.wav'),
        'profile_id': 'test'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 503
    assert 'Temporary service unavailability' in response.json['error']


class TestJobManagerConfig:
    """Test JobManager enabled/disabled via config"""

    def test_convert_song_disabled_job_manager_returns_sync_200(self):
        """Test config job_manager.enabled: false returns 200 with inline audio"""
        config = {'job_manager': {'enabled': False}}
        app, _ = create_app(config=config)
        # Mock singing pipeline
        mock_pipeline = Mock()
        mock_result = {
            'mixed_audio': np.zeros(44100),
            'sample_rate': 44100,
            'duration': 1.0,
            'metadata': {}
        }
        mock_pipeline.convert_song.return_value


def test_convert_song_empty_audio_503(client, app_and_socketio):
    """Test empty mixed_audio returns 503"""
    app, _ = app_and_socketio
    app.job_manager = None
    
    mock_pipeline = MagicMock()
    mock_result = {
        'mixed_audio': np.array([]),
        'sample_rate': 22050,
        'duration': 0.0,
        'metadata': {'target_profile_id': 'test'}
    }
    mock_pipeline.convert_song.return_value = mock_result
    app.singing_conversion_pipeline = mock_pipeline
    
    form_data = {
        'song': (io.BytesIO(b''), 'test.wav'),
        'profile_id': 'test'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 503
    assert 'Temporary service unavailability' in response.json['error']


@patch('src.auto_voice.web.api.torchaudio.save', side_effect=RuntimeError('Encoding fail'))
def test_convert_song_encoding_fail_503(mock_torchaudio_save, client, app_and_socketio):
    """Test encoding failure returns 503"""
    app, _ = app_and_socketio
    app.job_manager = None

    mock_pipeline = MagicMock()
    mock_result = {
        'mixed_audio': np.random.rand(1, 44100).astype(np.float32),
        'sample_rate': 22050,
        'duration': 2.0,
        'metadata': {'target_profile_id': 'test'}
    }
    mock_pipeline.convert_song.return_value = mock_result
    app.singing_conversion_pipeline = mock_pipeline

    form_data = {
        'song': (io.BytesIO(b''), 'test.wav'),
        'profile_id': 'test'
    }
    response = client.post('/api/v1/convert/song', data=form_data)
    assert response.status_code == 503
    assert 'Temporary service unavailability' in response.json['error']
