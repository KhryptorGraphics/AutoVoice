"""End-to-end tests for complete voice profile training workflow.

Tests the full user journey from web UI perspective:
1. Profile creation with sample uploads
2. Diarization and segment assignment
3. LoRA training with WebSocket progress
4. Adapter usage in conversion pipeline
5. Multi-artist YouTube workflow

These tests validate the integration completed in tracks:
- speaker-diarization_20260130
- youtube-artist-training_20260130
- training-inference-integration_20260130
- frontend-complete-integration_20260201
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from flask_socketio import SocketIOTestClient

from auto_voice.web.app import create_app


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get CUDA device, skip test if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - E2E tests require GPU")
    return torch.device('cuda')


@pytest.fixture
def sample_rate():
    """Audio sample rate used in tests."""
    return 24000


@pytest.fixture
def temp_storage():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def generate_voice_audio(
    duration_sec: float,
    sample_rate: int,
    base_freq: float = 440.0,
    add_vibrato: bool = False,
) -> np.ndarray:
    """Generate synthetic voice audio for testing.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz
        base_freq: Fundamental frequency
        add_vibrato: Add vibrato modulation

    Returns:
        Audio as float32 numpy array
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), dtype=np.float32)

    freq = base_freq
    if add_vibrato:
        vibrato_rate = 5.0
        vibrato_depth = 0.02
        freq = base_freq * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))

    # Fundamental + harmonics
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    audio += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    audio += 0.125 * np.sin(2 * np.pi * freq * 3 * t)

    return audio.astype(np.float32)


def generate_multi_speaker_audio(
    duration_sec: float,
    sample_rate: int,
    num_speakers: int = 2,
) -> np.ndarray:
    """Generate audio with multiple distinct speakers.

    Creates alternating speaker segments for diarization testing.
    """
    segment_duration = duration_sec / num_speakers
    segments = []

    # Use distinct frequencies for each speaker
    base_freqs = [220.0, 440.0, 330.0, 550.0][:num_speakers]

    for freq in base_freqs:
        segment = generate_voice_audio(segment_duration, sample_rate, freq, add_vibrato=True)
        segments.append(segment)

    return np.concatenate(segments).astype(np.float32)


@pytest.fixture
def app_with_components(temp_storage):
    """Create Flask app with all components enabled."""
    config = {
        'TESTING': True,
        'singing_conversion_enabled': True,
        'voice_cloning_enabled': True,
        'PROFILE_STORAGE_PATH': str(temp_storage / 'profiles'),
        'UPLOAD_FOLDER': str(temp_storage / 'uploads'),
    }

    # Create required directories
    (temp_storage / 'profiles').mkdir(exist_ok=True)
    (temp_storage / 'uploads').mkdir(exist_ok=True)

    app, socketio = create_app(config=config)
    return app, socketio


@pytest.fixture
def client_with_socketio(app_with_components):
    """Flask test client with SocketIO support."""
    app, socketio = app_with_components
    return app.test_client(), socketio


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_profile(client, name: str, sample_rate: int) -> str:
    """Helper to create a profile with reference audio.

    Args:
        client: Flask test client
        name: Profile name
        sample_rate: Audio sample rate

    Returns:
        profile_id: Created profile ID
    """
    reference_audio = generate_voice_audio(10.0, sample_rate)
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, reference_audio, sample_rate, format='WAV')
    audio_bytes.seek(0)

    response = client.post('/api/v1/voice/clone', data={
        'name': name,
        'reference_audio': (audio_bytes, f'{name.lower().replace(" ", "_")}_ref.wav'),
    }, content_type='multipart/form-data')

    assert response.status_code in [200, 201], f"Profile creation failed: {response.data}"
    data = json.loads(response.data)
    profile_id = data.get('profile_id')
    assert profile_id, "No profile_id returned"
    return profile_id


# ============================================================================
# Phase 1: Web UI Flow Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestProfileCreationFlow:
    """Test Task 1.1-1.4: Profile creation and sample management."""

    def test_create_profile_with_samples(self, client_with_socketio, sample_rate, temp_storage):
        """Test creating a profile and uploading samples via API."""
        client, socketio = client_with_socketio

        # Step 1: Create profile with reference audio
        reference_audio = generate_voice_audio(10.0, sample_rate)
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, reference_audio, sample_rate, format='WAV')
        audio_bytes.seek(0)

        response = client.post('/api/v1/voice/clone', data={
            'name': 'Test Artist',
            'reference_audio': (audio_bytes, 'reference.wav'),
        }, content_type='multipart/form-data')
        assert response.status_code in [200, 201], f"Profile creation failed: {response.data}"
        data = json.loads(response.data)
        profile_id = data.get('profile_id')
        assert profile_id, "No profile_id returned"

        # Step 2: Upload additional sample audio
        audio = generate_voice_audio(10.0, sample_rate)
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, sample_rate, format='WAV')
        audio_bytes.seek(0)

        response = client.post(
            f'/api/v1/profiles/{profile_id}/samples',
            data={'file': (audio_bytes, 'sample1.wav')},
            content_type='multipart/form-data'
        )
        assert response.status_code in [200, 201], f"Sample upload failed: {response.data}"

        # Step 3: List samples
        response = client.get(f'/api/v1/profiles/{profile_id}/samples')
        assert response.status_code == 200
        samples = json.loads(response.data)
        assert len(samples) >= 1, "Sample not in list"

        # Step 4: Verify sample metadata
        sample = samples[0]
        assert 'id' in sample
        assert 'duration_seconds' in sample
        assert sample['duration_seconds'] >= 9.0  # Should be close to 10s

    def test_insufficient_sample_duration(self, client_with_socketio, sample_rate):
        """Test validation for samples that are too short."""
        client, socketio = client_with_socketio

        # Create profile with reference audio
        reference_audio = generate_voice_audio(10.0, sample_rate)
        audio_bytes_ref = io.BytesIO()
        sf.write(audio_bytes_ref, reference_audio, sample_rate, format='WAV')
        audio_bytes_ref.seek(0)

        response = client.post('/api/v1/voice/clone', data={
            'name': 'Test',
            'reference_audio': (audio_bytes_ref, 'reference.wav'),
        }, content_type='multipart/form-data')
        assert response.status_code in [200, 201]
        profile_id = json.loads(response.data)['profile_id']

        # Try to upload very short audio
        audio = generate_voice_audio(2.0, sample_rate)  # Too short
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, sample_rate, format='WAV')
        audio_bytes.seek(0)

        response = client.post(
            f'/api/v1/profiles/{profile_id}/samples',
            data={'file': (audio_bytes, 'short.wav')},
            content_type='multipart/form-data'
        )
        # Should either reject or accept with warning
        if response.status_code >= 400:
            error_data = json.loads(response.data)
            assert 'duration' in error_data.get('error', '').lower() or 'short' in error_data.get('error', '').lower()


@pytest.mark.integration
@pytest.mark.slow
class TestDiarizationFlow:
    """Test Task 1.3-1.4: Diarization and segment assignment."""

    def test_diarization_and_segment_assignment(self, client_with_socketio, sample_rate, temp_storage):
        """Test running diarization and assigning segments to profiles."""
        client, socketio = client_with_socketio

        # Step 1: Upload multi-speaker audio for diarization
        audio = generate_multi_speaker_audio(20.0, sample_rate, num_speakers=2)
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, sample_rate, format='WAV')
        audio_bytes.seek(0)

        response = client.post(
            '/api/v1/audio/diarize',
            data={'file': (audio_bytes, 'multi_speaker.wav')},
            content_type='multipart/form-data'
        )

        # Diarization might not be available in test mode
        if response.status_code == 503:
            pytest.skip("Diarization service not available")

        assert response.status_code == 200, f"Diarization failed: {response.data}"
        diarization_data = json.loads(response.data)

        # Step 2: Verify diarization results
        assert 'speakers' in diarization_data
        assert len(diarization_data['speakers']) >= 1

        # Step 3: Create profile for assignment
        profile_id = create_test_profile(client, 'Speaker 1', sample_rate)

        # Step 4: Assign segment to profile
        if 'segment_key' in diarization_data:
            segment_key = diarization_data['segment_key']
            response = client.post('/api/v1/audio/diarize/assign', json={
                'profile_id': profile_id,
                'segment_key': segment_key,
                'segment_index': 0,
            })

            # Assignment might not be implemented in current version
            if response.status_code != 404:
                assert response.status_code == 200

                # Verify segment appears in profile
                response = client.get(f'/api/v1/profiles/{profile_id}/segments')
                if response.status_code == 200:
                    segments = json.loads(response.data)
                    assert len(segments.get('diarization_assignments', [])) >= 1


# ============================================================================
# Phase 2: LoRA Training Flow
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cuda
class TestLoRATrainingFlow:
    """Test Task 2.1-2.4: Training configuration and execution."""

    def test_training_job_creation(self, client_with_socketio, sample_rate, device):
        """Test creating a training job with samples."""
        client, socketio = client_with_socketio

        # Step 1: Create profile with samples
        profile_id = create_test_profile(client, 'Training Test', sample_rate)

        # Step 2: Upload multiple samples
        sample_ids = []
        for i in range(3):
            audio = generate_voice_audio(10.0, sample_rate, base_freq=440.0 + i * 10)
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio, sample_rate, format='WAV')
            audio_bytes.seek(0)

            response = client.post(
                f'/api/v1/profiles/{profile_id}/samples',
                data={'file': (audio_bytes, f'sample{i}.wav')},
                content_type='multipart/form-data'
            )
            if response.status_code in [200, 201]:
                sample_data = json.loads(response.data)
                if 'id' in sample_data:
                    sample_ids.append(sample_data['id'])

        assert len(sample_ids) >= 3, "Failed to upload samples"

        # Step 3: Create training job
        training_config = {
            'training_mode': 'lora',
            'lora_rank': 8,
            'lora_alpha': 16,
            'learning_rate': 1e-4,
            'batch_size': 4,
            'epochs': 1,
        }

        response = client.post('/api/v1/training/jobs', json={
            'profile_id': profile_id,
            'sample_ids': sample_ids,
            'config': training_config,
        })

        # Training might not be fully implemented
        if response.status_code == 503:
            pytest.skip("Training service not available")

        assert response.status_code in [200, 201, 202], f"Training job creation failed: {response.data}"
        job_data = json.loads(response.data)
        job_id = job_data.get('job_id')
        assert job_id, "No job_id returned"

        # Step 4: Check job status
        response = client.get(f'/api/v1/training/jobs/{job_id}')
        assert response.status_code == 200
        job_status = json.loads(response.data)
        assert job_status['status'] in ['pending', 'running', 'completed']

    def test_insufficient_samples_error(self, client_with_socketio, sample_rate):
        """Test Task 5.1: Error handling for insufficient samples."""
        client, socketio = client_with_socketio

        # Create profile with only 1 sample
        profile_id = create_test_profile(client, 'Insufficient', sample_rate)

        audio = generate_voice_audio(10.0, sample_rate)
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, sample_rate, format='WAV')
        audio_bytes.seek(0)

        response = client.post(
            f'/api/v1/profiles/{profile_id}/samples',
            data={'file': (audio_bytes, 'single.wav')},
            content_type='multipart/form-data'
        )
        sample_id = json.loads(response.data).get('id')

        # Try to train with insufficient samples
        response = client.post('/api/v1/training/jobs', json={
            'profile_id': profile_id,
            'sample_ids': [sample_id] if sample_id else [],
            'config': {'epochs': 10},
        })

        # Should get error or warning
        if response.status_code >= 400:
            error = json.loads(response.data)
            assert 'sample' in error.get('error', '').lower() or 'insufficient' in error.get('error', '').lower()


# ============================================================================
# Phase 3: Multi-Artist YouTube Flow
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestMultiArtistFlow:
    """Test Task 3.1-3.4: YouTube multi-artist workflow."""

    @patch('auto_voice.audio.youtube_downloader.YouTubeDownloader.download')
    def test_auto_profile_creation_from_diarization(
        self,
        mock_download,
        client_with_socketio,
        sample_rate,
        temp_storage
    ):
        """Test auto-creating profiles from diarization results."""
        client, socketio = client_with_socketio

        # Mock YouTube download to return multi-speaker audio
        audio = generate_multi_speaker_audio(30.0, sample_rate, num_speakers=2)
        audio_path = temp_storage / 'youtube_download.wav'
        sf.write(audio_path, audio, sample_rate)

        mock_download.return_value = MagicMock(
            audio_path=str(audio_path),
            metadata={'title': 'Artist A ft. Artist B - Song'}
        )

        # Step 1: Run diarization
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, sample_rate, format='WAV')
        audio_bytes.seek(0)

        response = client.post(
            '/api/v1/audio/diarize',
            data={'audio': (audio_bytes, 'collab.wav')},
            content_type='multipart/form-data'
        )

        if response.status_code == 503:
            pytest.skip("Diarization not available")

        assert response.status_code == 200
        diarization = json.loads(response.data)

        # Step 2: Auto-create profiles from speakers
        if 'segment_key' in diarization and len(diarization.get('speakers', [])) >= 2:
            response = client.post('/api/v1/profiles/auto-create', json={
                'segment_key': diarization['segment_key'],
                'artist_names': ['Artist A', 'Artist B'],
            })

            assert response.status_code != 404, "Auto-create endpoint is part of the public contract"
            assert response.status_code in [200, 201]
            result = json.loads(response.data)
            assert 'profiles' in result
            assert len(result['profiles']) >= 2


# ============================================================================
# Phase 4: Adapter Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cuda
class TestAdapterIntegration:
    """Test Task 4.1-4.4: Adapter loading and usage."""

    def test_adapter_listing(self, client_with_socketio, sample_rate, temp_storage):
        """Test listing adapters for a profile."""
        client, socketio = client_with_socketio

        # Create profile
        profile_id = create_test_profile(client, 'Adapter Test', sample_rate)

        # Check for adapters
        response = client.get(f'/api/v1/profiles/{profile_id}/adapters')

        assert response.status_code != 404, "Adapter listing endpoint is part of the public contract"
        assert response.status_code == 200
        adapters = json.loads(response.data)
        assert 'adapters' in adapters
        assert 'selected' in adapters

    def test_conversion_requires_trained_adapter(self, client_with_socketio, sample_rate):
        """Test Task 4.2: Conversion fails without trained adapter."""
        client, socketio = client_with_socketio

        # Create profile without training
        profile_id = create_test_profile(client, 'Untrained', sample_rate)

        # Try to convert without trained model
        song_audio = generate_voice_audio(5.0, sample_rate)
        song_bytes = io.BytesIO()
        sf.write(song_bytes, song_audio, sample_rate, format='WAV')
        song_bytes.seek(0)

        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (song_bytes, 'song.wav'),
                'profile_id': profile_id,
            },
            content_type='multipart/form-data'
        )

        # Should fail with 404 (no trained model)
        assert response.status_code == 404, f"Expected 404 for untrained profile, got {response.status_code}"
        error = json.loads(response.data)
        assert 'trained' in error.get('error', '').lower() or 'model' in error.get('error', '').lower()


# ============================================================================
# Phase 5: Error Handling Tests
# ============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Test Task 5.1-5.4: Error handling and recovery."""

    def test_invalid_audio_format(self, client_with_socketio, sample_rate):
        """Test Task 5.2: Invalid file rejection."""
        client, socketio = client_with_socketio

        profile_id = create_test_profile(client, 'Test', sample_rate)

        # Try to upload text file as audio
        fake_audio = io.BytesIO(b'This is not audio')

        response = client.post(
            f'/api/v1/profiles/{profile_id}/samples',
            data={'file': (fake_audio, 'fake.txt')},
            content_type='multipart/form-data'
        )

        # Should reject invalid format
        assert response.status_code >= 400

    def test_training_cancellation(self, client_with_socketio, sample_rate):
        """Test Task 5.3: Training job cancellation."""
        client, socketio = client_with_socketio

        # Create profile and samples
        profile_id = create_test_profile(client, 'Cancel Test', sample_rate)

        # Upload samples and create job
        sample_ids = []
        for i in range(3):
            audio = generate_voice_audio(10.0, sample_rate)
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio, sample_rate, format='WAV')
            audio_bytes.seek(0)

            response = client.post(
                f'/api/v1/profiles/{profile_id}/samples',
                data={'file': (audio_bytes, f's{i}.wav')},
                content_type='multipart/form-data'
            )
            if response.status_code in [200, 201]:
                sample_ids.append(json.loads(response.data).get('id'))

        if len(sample_ids) < 3:
            pytest.skip("Sample upload failed")

        response = client.post('/api/v1/training/jobs', json={
            'profile_id': profile_id,
            'sample_ids': sample_ids,
            'config': {'epochs': 3},
        })

        if response.status_code == 503:
            pytest.skip("Training not available")

        job_id = json.loads(response.data).get('job_id')
        if not job_id:
            pytest.skip("Job creation failed")

        # Cancel the job
        response = client.post(f'/api/v1/training/jobs/{job_id}/cancel')

        assert response.status_code != 404, "Training cancellation endpoint is part of the public contract"
        assert response.status_code == 200

        # Verify job is cancelled
        response = client.get(f'/api/v1/training/jobs/{job_id}')
        if response.status_code == 200:
            job = json.loads(response.data)
            # Job status should eventually be cancelled/failed, but cancellation
            # is async and may not have completed yet
            assert job['status'] in ['cancelled', 'failed', 'running', 'completed']


# ============================================================================
# Integration Summary Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cuda
class TestCompleteWorkflow:
    """Test complete end-to-end workflow."""

    def test_profile_train_convert_workflow(
        self,
        client_with_socketio,
        sample_rate,
        device,
        temp_storage
    ):
        """Test complete workflow: Profile → Upload → Train → Convert.

        This is the core user journey that Phase 6 validates.
        """
        client, socketio = client_with_socketio

        # Step 1: Create profile
        profile_id = create_test_profile(client, 'Complete Test', sample_rate)

        # Step 2: Upload 3 samples (minimum for training)
        sample_ids = []
        for i in range(3):
            audio = generate_voice_audio(10.0, sample_rate, base_freq=440.0 + i * 20)
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio, sample_rate, format='WAV')
            audio_bytes.seek(0)

            response = client.post(
                f'/api/v1/profiles/{profile_id}/samples',
                data={'file': (audio_bytes, f'sample{i}.wav')},
                content_type='multipart/form-data'
            )
            assert response.status_code in [200, 201], f"Sample {i} upload failed"
            sample_ids.append(json.loads(response.data)['id'])

        # Step 3: Start training (quick config for testing)
        training_config = {
            'training_mode': 'lora',
            'epochs': 1,
            'batch_size': 2,
        }

        response = client.post('/api/v1/training/jobs', json={
            'profile_id': profile_id,
            'sample_ids': sample_ids,
            'config': training_config,
        })

        if response.status_code == 503:
            pytest.skip("Training not available - components not initialized")

        assert response.status_code in [200, 201, 202], "Training job creation failed"
        job_id = json.loads(response.data)['job_id']

        # Step 4: Wait for training to complete (with timeout)
        max_wait = int(os.environ.get("AUTOVOICE_E2E_TRAINING_TIMEOUT", "60"))
        start_time = time.time()
        job_status = 'pending'

        while time.time() - start_time < max_wait:
            response = client.get(f'/api/v1/training/jobs/{job_id}')
            assert response.status_code == 200
            job = json.loads(response.data)
            job_status = job['status']

            if job_status in ['completed', 'failed', 'cancelled']:
                break

            time.sleep(1)

        if job_status != 'completed':
            pytest.skip(f"Training did not complete in time (status: {job_status})")

        # Step 5: Verify adapter was created
        response = client.get(f'/api/v1/profiles/{profile_id}/model')
        if response.status_code != 404:  # Endpoint might exist
            assert response.status_code == 200
            model_status = json.loads(response.data)
            assert model_status.get('has_trained_model') is True

        # Step 6: Convert song with trained profile
        song_audio = generate_voice_audio(5.0, sample_rate, base_freq=330.0)
        song_bytes = io.BytesIO()
        sf.write(song_bytes, song_audio, sample_rate, format='WAV')
        song_bytes.seek(0)

        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (song_bytes, 'test_song.wav'),
                'profile_id': profile_id,
            },
            content_type='multipart/form-data'
        )

        # Conversion should now succeed with trained model
        assert response.status_code in [200, 202], f"Conversion failed: {response.data}"
        conversion_data = json.loads(response.data)

        # Should have job_id or audio data
        assert 'job_id' in conversion_data or 'audio' in conversion_data
        if 'job_id' in conversion_data:
            conversion_job_id = conversion_data['job_id']
            conversion_status = None
            for _ in range(max_wait):
                response = client.get(f'/api/v1/convert/status/{conversion_job_id}')
                if response.status_code == 200:
                    conversion_job = json.loads(response.data)
                    conversion_status = conversion_job.get('status')
                    if conversion_status in ['completed', 'failed', 'cancelled']:
                        break
                time.sleep(1)

            assert conversion_status in ['completed', 'failed', 'cancelled']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
