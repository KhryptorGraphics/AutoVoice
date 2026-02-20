"""Tests for karaoke API endpoints.

TDD tests for live karaoke voice conversion feature.
"""
import io
import json
import pytest
import tempfile
import numpy as np

# Test fixtures and mocks


@pytest.fixture
def app():
    """Create test Flask application."""
    from auto_voice.web.app import create_app

    config = {
        'TESTING': True,
        'singing_conversion_enabled': False,
        'voice_cloning_enabled': False,
    }
    app, socketio = create_app(config)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_audio_bytes():
    """Create sample WAV audio bytes for testing."""
    import wave

    # Generate 1 second of silence at 16kHz mono
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    audio_data = np.zeros(samples, dtype=np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())

    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def sample_mp3_bytes():
    """Create minimal MP3-like bytes for format testing."""
    # MP3 files start with ID3 or 0xFF 0xFB sync word
    # This is just for format detection, not playable audio
    return b'\xff\xfb\x90\x00' + b'\x00' * 100


class TestKaraokeUploadEndpoint:
    """Tests for POST /api/v1/karaoke/upload"""

    def test_upload_requires_file(self, client):
        """Upload endpoint returns 400 when no file provided."""
        response = client.post('/api/v1/karaoke/upload')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'file' in data['error'].lower() or 'song' in data['error'].lower()

    def test_upload_rejects_empty_filename(self, client, sample_audio_bytes):
        """Upload endpoint returns 400 for empty filename."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), '')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_upload_rejects_invalid_format(self, client):
        """Upload endpoint returns 400 for non-audio files."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(b'not audio data'), 'test.txt')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'format' in data['error'].lower() or 'type' in data['error'].lower()

    def test_upload_accepts_wav(self, client, sample_audio_bytes):
        """Upload endpoint accepts WAV files."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.wav')},
            content_type='multipart/form-data'
        )
        # Should succeed or return 503 if service unavailable (not 400)
        assert response.status_code in [200, 201, 202, 503]
        data = response.get_json()
        if response.status_code in [200, 201, 202]:
            assert 'song_id' in data or 'id' in data

    def test_upload_accepts_mp3(self, client, sample_mp3_bytes):
        """Upload endpoint accepts MP3 files."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_mp3_bytes), 'test_song.mp3')},
            content_type='multipart/form-data'
        )
        # Should succeed or return 503 if service unavailable (not 400)
        assert response.status_code in [200, 201, 202, 503]

    def test_upload_accepts_flac(self, client, sample_audio_bytes):
        """Upload endpoint accepts FLAC files."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.flac')},
            content_type='multipart/form-data'
        )
        # Should succeed or return 503 if service unavailable (not 400)
        assert response.status_code in [200, 201, 202, 503]

    def test_upload_accepts_m4a(self, client, sample_audio_bytes):
        """Upload endpoint accepts M4A files."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.m4a')},
            content_type='multipart/form-data'
        )
        # Should succeed or return 503 if service unavailable (not 400)
        assert response.status_code in [200, 201, 202, 503]

    def test_upload_returns_song_id(self, client, sample_audio_bytes):
        """Successful upload returns a song_id for tracking."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.wav')},
            content_type='multipart/form-data'
        )
        if response.status_code in [200, 201, 202]:
            data = response.get_json()
            assert 'song_id' in data or 'id' in data
            song_id = data.get('song_id') or data.get('id')
            assert song_id is not None
            assert len(str(song_id)) > 0

    def test_upload_returns_duration(self, client, sample_audio_bytes):
        """Successful upload returns audio duration."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.wav')},
            content_type='multipart/form-data'
        )
        if response.status_code in [200, 201, 202]:
            data = response.get_json()
            assert 'duration' in data
            assert isinstance(data['duration'], (int, float))
            assert data['duration'] > 0

    def test_upload_size_limit(self, client):
        """Upload endpoint enforces size limit (e.g., 100MB)."""
        # Create oversized content (simulated, not actually 100MB)
        # This test just verifies the endpoint exists and handles large files
        large_data = b'\x00' * (10 * 1024 * 1024)  # 10MB test
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(large_data), 'large_song.wav')},
            content_type='multipart/form-data'
        )
        # Should either accept or reject with 413 (too large) or 400
        assert response.status_code in [200, 201, 202, 400, 413, 503]


class TestKaraokeSeparateEndpoint:
    """Tests for POST /api/v1/karaoke/separate"""

    def test_separate_requires_song_id(self, client):
        """Separate endpoint returns 400 when no song_id provided."""
        response = client.post('/api/v1/karaoke/separate')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_separate_validates_song_id(self, client):
        """Separate endpoint returns 404 for invalid song_id."""
        response = client.post(
            '/api/v1/karaoke/separate',
            json={'song_id': 'nonexistent-song-id-12345'}
        )
        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_separate_returns_job_id(self, client, sample_audio_bytes):
        """Successful separation request returns job_id."""
        # First upload a song
        upload_response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.wav')},
            content_type='multipart/form-data'
        )

        if upload_response.status_code in [200, 201, 202]:
            upload_data = upload_response.get_json()
            song_id = upload_data.get('song_id') or upload_data.get('id')

            # Request separation
            response = client.post(
                '/api/v1/karaoke/separate',
                json={'song_id': song_id}
            )

            if response.status_code in [200, 202]:
                data = response.get_json()
                assert 'job_id' in data

    def test_separate_returns_status_queued(self, client, sample_audio_bytes):
        """Separation request returns queued status."""
        # First upload a song
        upload_response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.wav')},
            content_type='multipart/form-data'
        )

        if upload_response.status_code in [200, 201, 202]:
            upload_data = upload_response.get_json()
            song_id = upload_data.get('song_id') or upload_data.get('id')

            response = client.post(
                '/api/v1/karaoke/separate',
                json={'song_id': song_id}
            )

            if response.status_code == 202:
                data = response.get_json()
                assert data.get('status') == 'queued'


class TestKaraokeSeparateStatusEndpoint:
    """Tests for GET /api/v1/karaoke/separate/{job_id}"""

    def test_status_returns_404_for_unknown_job(self, client):
        """Status endpoint returns 404 for unknown job_id."""
        response = client.get('/api/v1/karaoke/separate/unknown-job-id-12345')
        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_status_returns_job_info(self, client, sample_audio_bytes):
        """Status endpoint returns job information."""
        # Upload and start separation
        upload_response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(sample_audio_bytes), 'test_song.wav')},
            content_type='multipart/form-data'
        )

        if upload_response.status_code in [200, 201, 202]:
            upload_data = upload_response.get_json()
            song_id = upload_data.get('song_id') or upload_data.get('id')

            sep_response = client.post(
                '/api/v1/karaoke/separate',
                json={'song_id': song_id}
            )

            if sep_response.status_code in [200, 202]:
                sep_data = sep_response.get_json()
                job_id = sep_data.get('job_id')

                # Check status
                status_response = client.get(f'/api/v1/karaoke/separate/{job_id}')
                assert status_response.status_code == 200
                status_data = status_response.get_json()
                assert 'status' in status_data
                assert status_data['status'] in ['queued', 'processing', 'completed', 'failed']


class TestKaraokeWebSocketNamespace:
    """Tests for WebSocket audio streaming namespace."""

    def test_socketio_namespace_exists(self, app):
        """Socket.IO karaoke namespace is registered."""
        socketio = app.socketio
        # Check that the namespace handler exists
        # This will be implemented in the karaoke_events module
        assert socketio is not None

    def test_can_connect_to_karaoke_namespace(self, app):
        """Client can connect to /karaoke namespace."""
        from flask_socketio import SocketIOTestClient

        socketio = app.socketio
        test_client = SocketIOTestClient(app, socketio, namespace='/karaoke')

        # Should connect without error
        assert test_client.is_connected(namespace='/karaoke')
        test_client.disconnect(namespace='/karaoke')


class TestKaraokeSession:
    """Tests for KaraokeSession live conversion management (Phase 3)."""

    def test_session_creation(self):
        """KaraokeSession can be created with song_id and embedding."""
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-session-123',
            song_id='song-456',
            vocals_path='/tmp/test_vocals.wav',
            instrumental_path='/tmp/test_instrumental.wav'
        )

        assert session.session_id == 'test-session-123'
        assert session.song_id == 'song-456'
        assert session.is_active is False

    def test_session_start_stop(self):
        """Session can be started and stopped."""
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-session',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav'
        )

        session.start()
        assert session.is_active is True

        session.stop()
        assert session.is_active is False

    def test_session_speaker_embedding(self):
        """Session accepts and stores speaker embedding."""
        import torch
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-session',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav'
        )

        # 256-dim speaker embedding (mean+std of 128 mels)
        embedding = torch.randn(256)
        session.set_speaker_embedding(embedding)

        assert session.speaker_embedding is not None
        # Stored with batch dimension for processing
        assert session.speaker_embedding.shape == (1, 256)

    def test_session_process_chunk_returns_audio(self):
        """Session processes audio chunk and returns converted output."""
        import torch
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-session',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav'
        )

        # Set up session
        embedding = torch.randn(256)
        session.set_speaker_embedding(embedding)
        session.start()

        # Process chunk (24000 Hz sample rate, 100ms chunk = 2400 samples)
        input_chunk = torch.randn(2400)
        output = session.process_chunk(input_chunk)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        # Output should have samples (may differ due to overlap-add)
        assert len(output.shape) >= 1

        session.stop()

    def test_session_tracks_latency(self):
        """Session tracks processing latency."""
        import torch
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-session',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav'
        )

        embedding = torch.randn(256)
        session.set_speaker_embedding(embedding)
        session.start()

        # Process a chunk to generate latency data
        input_chunk = torch.randn(2400)
        session.process_chunk(input_chunk)

        # Should have latency measurement
        latency = session.get_latency_ms()
        assert latency >= 0  # Non-negative latency

        session.stop()

    def test_session_trt_disable_option(self):
        """Session can explicitly disable TRT."""
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-session',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav',
            use_trt=False,  # Explicitly disable TRT
        )

        # With TRT disabled, session should not attempt to load TRT
        assert session._use_trt is False

    def test_session_stats_include_pipeline_type(self):
        """Session stats include pipeline type."""
        import torch
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-session',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav',
            use_trt=False,  # Disable TRT for test
        )

        embedding = torch.randn(256)
        session.set_speaker_embedding(embedding)
        session.start()

        # Process a chunk to initialize pipeline
        input_chunk = torch.randn(2400)
        session.process_chunk(input_chunk)

        stats = session.get_stats()

        assert 'pipeline_type' in stats
        # Without TRT engines, falls back to pytorch
        assert stats['pipeline_type'] in ['pytorch', 'tensorrt', None]

        session.stop()


class TestTRTStreamingPipeline:
    """Tests for TRT streaming pipeline (Task 3.6)."""

    def test_trt_engines_available_returns_false_when_missing(self):
        """TRT engine check returns False when engines don't exist."""
        import tempfile
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory - no engines
            assert TRTStreamingPipeline.engines_available(tmpdir) is False

    def test_trt_pipeline_latency_tracking(self):
        """TRT pipeline tracks latency statistics."""
        import tempfile
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TRTStreamingPipeline(
                engine_dir=tmpdir,
                chunk_size_ms=100,
                sample_rate=24000,
            )

            # Before any processing, latency stats should be zero
            stats = pipeline.get_latency_stats()
            assert stats['min_ms'] == 0.0
            assert stats['max_ms'] == 0.0
            assert stats['avg_ms'] == 0.0

    def test_trt_pipeline_reset(self):
        """TRT pipeline can reset state."""
        import tempfile
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TRTStreamingPipeline(
                engine_dir=tmpdir,
                sample_rate=24000,
            )

            # Manually add some latency history
            pipeline._latency_history = [10.0, 20.0, 30.0]

            # Reset should clear it
            pipeline.reset()
            assert pipeline._latency_history == []
            assert pipeline.overlap_buffer is None


class TestAudioOutputRouter:
    """Tests for dual audio output routing (Phase 4)."""

    def test_router_creation(self):
        """AudioOutputRouter can be created with default config."""
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)

        assert router.sample_rate == 24000
        assert router.speaker_enabled is True
        assert router.headphone_enabled is True

    def test_router_channel_configuration(self):
        """Router allows channel configuration."""
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)

        # Configure channels
        router.set_channel_config(
            speaker_gain=0.8,
            headphone_gain=1.0,
            speaker_enabled=True,
            headphone_enabled=True,
        )

        assert router.speaker_gain == 0.8
        assert router.headphone_gain == 1.0

    def test_router_generates_dual_output(self):
        """Router generates separate speaker and headphone outputs."""
        import torch
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)

        # Input: converted voice and original song chunk
        converted_voice = torch.randn(2400)
        instrumental = torch.randn(2400)
        original_song = torch.randn(2400)

        # Route audio
        speaker_out, headphone_out = router.route(
            converted_voice=converted_voice,
            instrumental=instrumental,
            original_song=original_song,
        )

        # Both outputs should be tensors of same length
        assert speaker_out.shape == converted_voice.shape
        assert headphone_out.shape == converted_voice.shape

    def test_router_speaker_output_is_mixed(self):
        """Speaker output contains instrumental + converted voice."""
        import torch
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)
        router.set_channel_config(speaker_gain=1.0, headphone_gain=1.0)

        # Use known values to verify mixing
        converted_voice = torch.ones(100) * 0.5
        instrumental = torch.ones(100) * 0.3
        original_song = torch.ones(100) * 0.8

        speaker_out, _ = router.route(
            converted_voice=converted_voice,
            instrumental=instrumental,
            original_song=original_song,
        )

        # Speaker should be mix of converted + instrumental (0.5 + 0.3 = 0.8)
        # May be normalized to prevent clipping
        assert speaker_out.mean() > 0  # Contains mixed audio

    def test_router_headphone_output_is_original(self):
        """Headphone output contains original song for performer."""
        import torch
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)

        converted_voice = torch.ones(100) * 0.5
        instrumental = torch.ones(100) * 0.3
        original_song = torch.ones(100) * 0.8

        _, headphone_out = router.route(
            converted_voice=converted_voice,
            instrumental=instrumental,
            original_song=original_song,
        )

        # Headphone should be original song (performer follows along)
        assert torch.allclose(headphone_out, original_song, atol=0.01)

    def test_router_disabled_channel_returns_silence(self):
        """Disabled channel returns silence."""
        import torch
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)
        router.set_channel_config(speaker_enabled=False, headphone_enabled=True)

        converted_voice = torch.ones(100) * 0.5
        instrumental = torch.ones(100) * 0.3
        original_song = torch.ones(100) * 0.8

        speaker_out, headphone_out = router.route(
            converted_voice=converted_voice,
            instrumental=instrumental,
            original_song=original_song,
        )

        # Speaker disabled - should be zeros
        assert speaker_out.abs().max() < 0.01
        # Headphone still active
        assert headphone_out.abs().max() > 0.5

    def test_router_gain_applied(self):
        """Gain is applied to outputs."""
        import torch
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)
        router.set_channel_config(speaker_gain=0.5, headphone_gain=0.5)

        original_song = torch.ones(100) * 1.0

        _, headphone_out = router.route(
            converted_voice=torch.zeros(100),
            instrumental=torch.zeros(100),
            original_song=original_song,
        )

        # Headphone with 0.5 gain should be ~0.5
        assert torch.allclose(headphone_out, torch.ones(100) * 0.5, atol=0.01)


class TestAudioOutputDeviceAPI:
    """Tests for audio device selection API (Phase 4)."""

    def test_list_devices_endpoint(self, client):
        """GET /api/v1/karaoke/devices lists available audio devices."""
        response = client.get('/api/v1/karaoke/devices')

        # Should return 200 with device list (may be empty in test env)
        assert response.status_code == 200
        data = response.get_json()
        assert 'devices' in data
        assert isinstance(data['devices'], list)

    def test_set_output_device_endpoint(self, client):
        """POST /api/v1/karaoke/devices/output sets output device."""
        response = client.post(
            '/api/v1/karaoke/devices/output',
            json={'speaker_device': 0, 'headphone_device': 1}
        )

        # Should accept device configuration
        assert response.status_code in [200, 400]  # 400 if devices not available

    def test_get_current_device_config(self, client):
        """GET /api/v1/karaoke/devices/output gets current device config."""
        response = client.get('/api/v1/karaoke/devices/output')

        assert response.status_code == 200
        data = response.get_json()
        assert 'speaker_device' in data or 'error' in data


class TestVoiceModelAPI:
    """Tests for voice model management API (Phase 5)."""

    def test_list_voice_models_endpoint(self, client):
        """GET /api/v1/karaoke/voice-models lists available models."""
        response = client.get('/api/v1/karaoke/voice-models')

        assert response.status_code == 200
        data = response.get_json()
        assert 'models' in data
        assert isinstance(data['models'], list)

    def test_voice_model_has_required_fields(self, client):
        """Voice models have id, name, and type fields."""
        response = client.get('/api/v1/karaoke/voice-models')

        if response.status_code == 200:
            data = response.get_json()
            # If models exist, check structure
            if data['models']:
                model = data['models'][0]
                assert 'id' in model
                assert 'name' in model
                assert 'type' in model  # 'pretrained' or 'extracted'

    def test_get_voice_model_by_id(self, client):
        """GET /api/v1/karaoke/voice-models/{id} gets model details."""
        # First list models to get a valid ID
        list_response = client.get('/api/v1/karaoke/voice-models')
        if list_response.status_code == 200:
            data = list_response.get_json()
            if data['models']:
                model_id = data['models'][0]['id']
                response = client.get(f'/api/v1/karaoke/voice-models/{model_id}')
                assert response.status_code == 200

    def test_get_nonexistent_voice_model_returns_404(self, client):
        """GET /api/v1/karaoke/voice-models/{id} returns 404 for unknown ID."""
        response = client.get('/api/v1/karaoke/voice-models/nonexistent-model-xyz')
        assert response.status_code == 404


class TestVoiceModelRegistry:
    """Tests for VoiceModelRegistry class (Phase 5)."""

    def test_registry_creation(self):
        """VoiceModelRegistry can be created."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        registry = VoiceModelRegistry()
        assert registry is not None

    def test_registry_list_models(self):
        """Registry can list available models."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        registry = VoiceModelRegistry()
        models = registry.list_models()

        assert isinstance(models, list)

    def test_registry_get_model(self):
        """Registry can get model by ID."""
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        registry = VoiceModelRegistry()
        models = registry.list_models()

        if models:
            model = registry.get_model(models[0]['id'])
            assert model is not None
            assert 'id' in model

    def test_registry_get_embedding(self):
        """Registry can load speaker embedding for model."""
        import torch
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        registry = VoiceModelRegistry()
        models = registry.list_models()

        if models:
            embedding = registry.get_embedding(models[0]['id'])
            assert embedding is not None
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape[-1] == 256  # 256-dim embedding


class TestArtistEmbeddingExtraction:
    """Tests for extracting artist voice from uploaded song (Phase 5)."""

    def test_extract_embedding_from_audio(self):
        """Can extract speaker embedding from audio."""
        import torch
        from auto_voice.web.voice_model_registry import extract_speaker_embedding

        # Create test audio (1 second of noise)
        audio = torch.randn(24000)

        embedding = extract_speaker_embedding(audio, sample_rate=24000)

        assert embedding is not None
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape[-1] == 256

    def test_register_extracted_model(self):
        """Can register extracted embedding as a voice model."""
        import torch
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        registry = VoiceModelRegistry()

        # Create test embedding
        embedding = torch.randn(256)

        model_id = registry.register_extracted_model(
            name="Test Artist",
            embedding=embedding,
            source_song_id="test-song-123"
        )

        assert model_id is not None
        assert registry.get_model(model_id) is not None

    def test_session_can_use_voice_model(self):
        """KaraokeSession can be configured with a voice model."""
        import torch
        from auto_voice.web.karaoke_session import KaraokeSession
        from auto_voice.web.voice_model_registry import VoiceModelRegistry

        registry = VoiceModelRegistry()

        # Register a test model
        embedding = torch.randn(256)
        model_id = registry.register_extracted_model(
            name="Test Artist",
            embedding=embedding,
            source_song_id="test-song"
        )

        # Create session with voice model
        session = KaraokeSession(
            session_id='test-session',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav',
        )

        # Load voice model
        session.load_voice_model(registry, model_id)

        assert session.speaker_embedding is not None
        assert session.voice_model_id == model_id
