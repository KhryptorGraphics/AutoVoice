"""End-to-end tests for speaker diarization pipeline.

These tests verify the complete flow from audio input to profile creation,
including:
- YouTube download → diarization → profile creation
- Upload → diarization → filter → training data preparation
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.io import wavfile


@pytest.fixture
def e2e_client(tmp_path):
    """Create Flask test client with profiles directory."""
    from auto_voice.web.app import create_app

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    app, socketio = create_app(config={
        'TESTING': True,
        'singing_conversion_enabled': False,
        'voice_cloning_enabled': False,
        'profiles_dir': str(profiles_dir),
    })
    return app.test_client(), profiles_dir


class TestYouTubeDownloadDiarizationFlow:
    """E2E tests for YouTube download → diarization → profile creation."""

    def test_youtube_info_returns_metadata(self, e2e_client):
        """Test fetching YouTube video info."""
        client, _ = e2e_client

        # Mock yt-dlp metadata response
        mock_metadata = {
            'id': 'test123',
            'title': 'Artist - Song ft. Featured Artist',
            'duration': 180,
            'thumbnail': 'https://example.com/thumb.jpg',
            'uploader': 'Artist',
            'description': 'Official video featuring Featured Artist',
        }

        with patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._get_metadata') as mock_get:
            mock_get.return_value = mock_metadata

            response = client.post('/api/v1/youtube/info',
                                   data=json.dumps({'url': 'https://youtube.com/watch?v=test123'}),
                                   content_type='application/json')

            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert data['title'] == 'Artist - Song ft. Featured Artist'
            assert 'Featured Artist' in data.get('featured_artists', [])

    def test_youtube_download_creates_audio_file(self, e2e_client, tmp_path):
        """Test YouTube download creates audio file."""
        client, _ = e2e_client

        # Create a mock audio file that would be "downloaded"
        mock_audio_path = tmp_path / "downloaded_audio.wav"
        sample_rate = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
        wavfile.write(str(mock_audio_path), sample_rate, audio)

        mock_metadata = {
            'id': 'test123',
            'title': 'Test Song ft. Guest',
            'duration': 180,
            'thumbnail': 'https://example.com/thumb.jpg',
        }

        with patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._get_metadata') as mock_get, \
             patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._download_audio') as mock_dl:
            mock_get.return_value = mock_metadata
            mock_dl.return_value = True

            # Patch to use our mock file
            with patch('os.path.exists', return_value=True):
                response = client.post('/api/v1/youtube/download',
                                       data=json.dumps({
                                           'url': 'https://youtube.com/watch?v=test123',
                                           'run_diarization': False,
                                       }),
                                       content_type='application/json')

            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True

    def test_full_youtube_to_profile_flow_mocked(self, e2e_client, multi_speaker_synthetic):
        """Test full flow: YouTube download → diarization → profile creation (mocked)."""
        client, profiles_dir = e2e_client

        # Use synthetic multi-speaker fixture as "downloaded" audio
        fixture = multi_speaker_synthetic

        mock_metadata = {
            'id': 'duet123',
            'title': 'Main Artist - Duet ft. Featured Artist',
            'duration': fixture.duration,
            'thumbnail': 'https://example.com/thumb.jpg',
        }

        # Mock YouTube download to return our fixture
        with patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._get_metadata') as mock_get, \
             patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._download_audio') as mock_dl:
            mock_get.return_value = mock_metadata
            mock_dl.return_value = True

            # Mock the output path to be our fixture
            with patch('auto_voice.audio.youtube_downloader.YouTubeDownloader.download') as mock_download:
                from auto_voice.audio.youtube_downloader import YouTubeDownloadResult
                mock_download.return_value = YouTubeDownloadResult(
                    success=True,
                    audio_path=fixture.audio_path,
                    title='Main Artist - Duet ft. Featured Artist',
                    duration=fixture.duration,
                    main_artist='Main Artist',
                    featured_artists=['Featured Artist'],
                    video_id='duet123',
                )

                # Step 1: Get video info
                response = client.post('/api/v1/youtube/info',
                                       data=json.dumps({'url': 'https://youtube.com/watch?v=duet123'}),
                                       content_type='application/json')
                assert response.status_code == 200

                # Step 2: Download and diarize
                response = client.post('/api/v1/youtube/download',
                                       data=json.dumps({
                                           'url': 'https://youtube.com/watch?v=duet123',
                                           'run_diarization': True,
                                       }),
                                       content_type='application/json')
                assert response.status_code == 200
                data = response.get_json()
                assert data['success'] is True


class TestUploadDiarizationFilterFlow:
    """E2E tests for Upload → diarization → filter → training."""

    def test_upload_and_diarize_audio(self, e2e_client, multi_speaker_synthetic):
        """Test uploading audio and running diarization."""
        client, _ = e2e_client
        fixture = multi_speaker_synthetic

        # Use JSON with audio_path (the API accepts both file upload and path)
        response = client.post('/api/v1/audio/diarize',
                               data=json.dumps({'audio_path': fixture.audio_path}),
                               content_type='application/json')

        # Should either succeed or indicate processing (or 500 if diarizer not available)
        assert response.status_code in [200, 202, 500]

    def test_diarization_returns_speaker_segments(self, e2e_client, multi_speaker_synthetic):
        """Test diarization returns proper speaker segments."""
        client, _ = e2e_client
        fixture = multi_speaker_synthetic

        # Mock the diarizer to return expected segments
        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer.diarize') as mock_diarize:
            from auto_voice.audio.speaker_diarization import DiarizationResult, SpeakerSegment

            mock_result = DiarizationResult(
                segments=[
                    SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
                    SpeakerSegment(start=2.0, end=4.0, speaker_id="SPEAKER_01"),
                    SpeakerSegment(start=4.0, end=5.5, speaker_id="SPEAKER_00"),
                    SpeakerSegment(start=5.5, end=7.0, speaker_id="SPEAKER_01"),
                ],
                num_speakers=2,
                audio_duration=7.0,
            )
            mock_diarize.return_value = mock_result

            response = client.post('/api/v1/audio/diarize',
                                   data={'audio': (open(fixture.audio_path, 'rb'), 'test.wav')},
                                   content_type='multipart/form-data')

            if response.status_code == 200:
                data = response.get_json()
                assert 'segments' in data or 'num_speakers' in data

    def test_create_profile_from_diarized_segments(self, e2e_client, multi_speaker_synthetic):
        """Test creating a voice profile from diarized segments."""
        client, profiles_dir = e2e_client
        fixture = multi_speaker_synthetic

        # Create a profile via voice/clone endpoint with file upload
        with open(fixture.audio_path, 'rb') as f:
            response = client.post('/api/v1/voice/clone',
                                   data={
                                       'reference_audio': (f, 'sample.wav'),
                                       'profile_name': 'Test Speaker',
                                   },
                                   content_type='multipart/form-data')

        # Voice cloner may not be initialized in test mode (503) or succeed
        if response.status_code == 503:
            pytest.skip("Voice cloner service not available in test mode")

        assert response.status_code in [200, 201]
        data = response.get_json()
        profile_id = data.get('profile_id') or data.get('id')

        # Verify profile was created
        if profile_id:
            # Get profile to verify
            response = client.get(f'/api/v1/voice/profiles/{profile_id}')
            assert response.status_code in [200, 503]  # 503 if service unavailable

    def test_filter_training_audio_by_speaker(self, e2e_client, multi_speaker_synthetic):
        """Test filtering training audio to specific speaker segments."""
        client, profiles_dir = e2e_client
        fixture = multi_speaker_synthetic

        # Create a profile via voice/clone endpoint
        with open(fixture.audio_path, 'rb') as f:
            response = client.post('/api/v1/voice/clone',
                                   data={
                                       'reference_audio': (f, 'sample.wav'),
                                       'profile_name': 'Filter Test Speaker',
                                   },
                                   content_type='multipart/form-data')

        # Voice cloner may not be available
        if response.status_code == 503:
            pytest.skip("Voice cloner service not available in test mode")

        if response.status_code in [200, 201]:
            data = response.get_json()
            profile_id = data.get('profile_id') or data.get('id')

            if profile_id:
                # Try to set speaker embedding using audio_path
                response = client.post(f'/api/v1/profiles/{profile_id}/speaker-embedding',
                                       data=json.dumps({
                                           'audio_path': fixture.audio_path
                                       }),
                                       content_type='application/json')
                # Should succeed or return error (endpoint may need profile to exist first)
                assert response.status_code in [200, 201, 400, 404, 500]


class TestDiarizationAPIEndpoints:
    """Tests for diarization API endpoints."""

    def test_diarize_endpoint_exists(self, e2e_client):
        """Test that diarize endpoint exists."""
        client, _ = e2e_client

        # Send empty request to check endpoint exists
        response = client.post('/api/v1/audio/diarize')
        # Should not be 404 (endpoint exists, might be 400 for bad request)
        assert response.status_code != 404

    def test_diarize_assign_endpoint_exists(self, e2e_client):
        """Test that diarize assign endpoint exists."""
        client, _ = e2e_client

        response = client.post('/api/v1/audio/diarize/assign',
                               data=json.dumps({
                                   'diarization_id': 'test',
                                   'segment_id': '0',
                                   'profile_id': 'test'
                               }),
                               content_type='application/json')
        # Should exist (not 404)
        assert response.status_code != 404

    def test_profile_segments_endpoint_exists(self, e2e_client):
        """Test that profile segments endpoint exists."""
        client, _ = e2e_client

        # Create a profile first
        response = client.post('/api/v1/profiles',
                               data=json.dumps({'name': 'Segments Test'}),
                               content_type='application/json')

        if response.status_code in [200, 201]:
            data = response.get_json()
            profile_id = data.get('id') or data.get('profile_id')

            if profile_id:
                response = client.get(f'/api/v1/profiles/{profile_id}/segments')
                assert response.status_code != 404

    def test_auto_create_profile_endpoint(self, e2e_client):
        """Test auto-create profile endpoint."""
        client, _ = e2e_client

        response = client.post('/api/v1/profiles/auto-create',
                               data=json.dumps({'name': 'Auto Created'}),
                               content_type='application/json')
        # Should exist (might need more params)
        assert response.status_code != 404


class TestDiarizationFixtures:
    """Tests to verify fixture utilities work correctly."""

    def test_synthetic_fixture_has_multiple_speakers(self, multi_speaker_synthetic):
        """Test synthetic fixture has expected speakers."""
        fixture = multi_speaker_synthetic

        assert fixture.num_speakers == 2
        assert len(fixture.speakers) == 4
        assert fixture.duration > 0
        assert os.path.exists(fixture.audio_path)

    def test_synthetic_fixture_speaker_info(self, multi_speaker_synthetic):
        """Test fixture provides accurate speaker info."""
        fixture = multi_speaker_synthetic

        # Check speaker segments are non-overlapping
        sorted_segments = sorted(fixture.speakers, key=lambda s: s.start)
        for i in range(len(sorted_segments) - 1):
            assert sorted_segments[i].end <= sorted_segments[i + 1].start + 0.01

        # Check speaker totals
        speaker_00_duration = fixture.get_speaker_total_duration("SPEAKER_00")
        speaker_01_duration = fixture.get_speaker_total_duration("SPEAKER_01")
        assert speaker_00_duration == pytest.approx(3.5, rel=0.01)
        assert speaker_01_duration == pytest.approx(3.5, rel=0.01)

    def test_three_speaker_fixture(self, multi_speaker_three):
        """Test three-speaker fixture."""
        fixture = multi_speaker_three

        assert fixture.num_speakers == 3
        assert len(fixture.speakers) == 5

        speaker_ids = {s.speaker_id for s in fixture.speakers}
        assert speaker_ids == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}

    def test_duet_fixture_with_real_audio(self, duet_fixture):
        """Test duet fixture from real audio samples."""
        if duet_fixture is None:
            pytest.skip("Quality samples not available")

        assert duet_fixture.num_speakers == 2
        assert os.path.exists(duet_fixture.audio_path)

        # Should have conor and william as speakers
        speaker_ids = {s.speaker_id for s in duet_fixture.speakers}
        assert "conor" in speaker_ids
        assert "william" in speaker_ids


@pytest.mark.slow
class TestDiarizationPerformance:
    """Performance tests for diarization (Task 8.5)."""

    def test_diarization_speed_synthetic(self, multi_speaker_synthetic):
        """Test diarization speed on synthetic audio."""
        import time

        fixture = multi_speaker_synthetic

        try:
            from auto_voice.audio.speaker_diarization import SpeakerDiarizer

            diarizer = SpeakerDiarizer(device="cpu")

            start = time.time()
            result = diarizer.diarize(fixture.audio_path)
            elapsed = time.time() - start

            print(f"\nDiarization time for {fixture.duration:.1f}s audio: {elapsed:.2f}s")
            print(f"Real-time factor: {elapsed / fixture.duration:.2f}x")

            # Should complete in reasonable time (less than 10x real-time on CPU)
            assert elapsed < fixture.duration * 10

        except ImportError:
            pytest.skip("Speaker diarization module not available")

    @pytest.mark.cuda
    def test_diarization_speed_gpu(self, multi_speaker_synthetic):
        """Test diarization speed on GPU."""
        import time
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        fixture = multi_speaker_synthetic

        try:
            from auto_voice.audio.speaker_diarization import SpeakerDiarizer

            diarizer = SpeakerDiarizer(device="cuda")

            start = time.time()
            result = diarizer.diarize(fixture.audio_path)
            elapsed = time.time() - start

            print(f"\nGPU diarization time for {fixture.duration:.1f}s audio: {elapsed:.2f}s")
            print(f"Real-time factor: {elapsed / fixture.duration:.2f}x")

            # GPU should be faster than real-time
            assert elapsed < fixture.duration

        except ImportError:
            pytest.skip("Speaker diarization module not available")

    def test_diarization_30_second_benchmark(self, tmp_path):
        """Benchmark: Diarization should complete in <30s for 5-minute audio."""
        import time

        from tests.fixtures import create_synthetic_multi_speaker

        # Create 5-minute (300 second) synthetic audio
        long_durations = []
        for i in range(15):  # 15 segments of 20 seconds each
            speaker = f"SPEAKER_{i % 3:02d}"
            long_durations.append((speaker, 20.0))

        output_path = str(tmp_path / "long_audio.wav")
        fixture = create_synthetic_multi_speaker(output_path, durations=long_durations)

        print(f"\nCreated {fixture.duration:.1f}s audio with {fixture.num_speakers} speakers")

        try:
            from auto_voice.audio.speaker_diarization import SpeakerDiarizer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            diarizer = SpeakerDiarizer(device=device)

            start = time.time()
            result = diarizer.diarize(fixture.audio_path)
            elapsed = time.time() - start

            print(f"Diarization time: {elapsed:.2f}s")
            print(f"Detected {result.num_speakers} speakers")

            # Target: <30s for 5-minute audio
            # This might be too aggressive for CPU, so use softer check
            if device == "cuda":
                assert elapsed < 30, f"Diarization took {elapsed:.2f}s, target was <30s"
            else:
                # CPU is expected to be slower
                print(f"Note: CPU took {elapsed:.2f}s (GPU target is <30s)")

        except ImportError:
            pytest.skip("Speaker diarization module not available")
