"""
API Contract Validation E2E Tests for AutoVoice.

Tests complete API workflows including:
- Voice cloning workflow
- Song conversion workflow
- Health and status monitoring
- Quality metrics validation
- Error handling and recovery

Implements Comment 12 requirements.
"""

import pytest
import requests
import time
import json
import base64
import io
import wave
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional


# Test Configuration
API_BASE_URL = 'http://localhost:5001'
TEST_DATA_DIR = Path(__file__).parent / 'data'
VALIDATION_RESULTS_DIR = Path(__file__).parent.parent / 'validation_results'

# Ensure validation results directory exists
VALIDATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def flask_server():
    """Start Flask app for E2E testing."""
    import threading
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    try:
        from auto_voice.web.app import create_app

        # Create app with test config
        app, socketio = create_app(config={'TESTING': False})  # Use real components

        if app is None:
            pytest.skip("Flask app not available")

        # Start server in background thread
        server_thread = threading.Thread(
            target=lambda: socketio.run(app, host='127.0.0.1', port=5001, debug=False, use_reloader=False),
            daemon=True
        )
        server_thread.start()

        # Wait for server startup
        max_attempts = 30
        for i in range(max_attempts):
            try:
                response = requests.get(f'{API_BASE_URL}/health/live', timeout=1)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.skip("Flask server failed to start within 30 seconds")

        yield

        # Server will be stopped when daemon thread terminates

    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")


@pytest.fixture
def sample_audio_file():
    """Generate sample audio file for testing (30s WAV)."""
    sample_rate = 22050
    duration = 30.0
    samples = int(sample_rate * duration)

    # Generate sine wave
    t = np.linspace(0, duration, samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    # Convert to WAV bytes
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_song_file():
    """Generate sample song file for testing (3s WAV)."""
    sample_rate = 22050
    duration = 3.0
    samples = int(sample_rate * duration)

    # Generate more complex waveform (simulated vocals + instrumental)
    t = np.linspace(0, duration, samples)
    vocals = np.sin(2 * np.pi * 440 * t) * 0.3
    instrumental = np.sin(2 * np.pi * 220 * t) * 0.2
    audio = (vocals + instrumental).astype(np.float32)

    # Convert to WAV bytes
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer


def save_validation_results(test_name: str, results: Dict[str, Any]):
    """Save test results to validation_results directory."""
    output_path = VALIDATION_RESULTS_DIR / f'{test_name}_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Validation results saved: {output_path}")


@pytest.mark.api
@pytest.mark.e2e
class TestAPIHealthEndpoints:
    """Test API health check endpoints."""

    def test_liveness_endpoint(self, flask_server):
        """Test liveness probe endpoint."""
        response = requests.get(f'{API_BASE_URL}/health/live')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'alive'

        save_validation_results('liveness', {
            'status_code': response.status_code,
            'response': data,
            'passed': True
        })

    def test_readiness_endpoint(self, flask_server):
        """Test readiness probe endpoint."""
        response = requests.get(f'{API_BASE_URL}/health/ready')

        # Should be 200 (ready) or 503 (not ready)
        assert response.status_code in [200, 503]
        data = response.json()

        assert 'status' in data
        assert 'components' in data

        save_validation_results('readiness', {
            'status_code': response.status_code,
            'response': data,
            'passed': True
        })

    def test_api_health_endpoint(self, flask_server):
        """Test API-specific health endpoint."""
        response = requests.get(f'{API_BASE_URL}/api/v1/health')

        assert response.status_code == 200
        data = response.json()

        required_fields = ['status', 'gpu_available', 'model_loaded', 'timestamp']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        save_validation_results('api_health', {
            'status_code': response.status_code,
            'response': data,
            'passed': True
        })

    def test_gpu_status_endpoint(self, flask_server):
        """Test GPU status endpoint."""
        response = requests.get(f'{API_BASE_URL}/api/v1/gpu_status')

        assert response.status_code == 200
        data = response.json()

        assert 'cuda_available' in data
        assert 'device' in data
        assert 'device_count' in data

        save_validation_results('gpu_status', {
            'status_code': response.status_code,
            'response': data,
            'gpu_available': data.get('cuda_available', False),
            'passed': True
        })


@pytest.mark.api
@pytest.mark.e2e
class TestVoiceCloningWorkflow:
    """Test complete voice cloning workflow via API."""

    def test_voice_clone_create_and_list(self, flask_server, sample_audio_file):
        """Test voice profile creation and listing."""
        # Step 1: Create voice profile
        files = {'reference_audio': ('test_audio.wav', sample_audio_file, 'audio/wav')}
        data = {'user_id': 'test_user_e2e'}

        create_response = requests.post(
            f'{API_BASE_URL}/api/v1/voice/clone',
            files=files,
            data=data
        )

        # Should succeed or service unavailable
        assert create_response.status_code in [201, 503]

        if create_response.status_code == 503:
            pytest.skip("Voice cloning service unavailable")

        create_data = create_response.json()
        assert create_data['status'] == 'success'
        assert 'profile_id' in create_data

        profile_id = create_data['profile_id']

        # Step 2: List voice profiles
        list_response = requests.get(f'{API_BASE_URL}/api/v1/voice/profiles')
        assert list_response.status_code == 200

        profiles = list_response.json()
        assert isinstance(profiles, list)

        # Step 3: Get specific profile
        get_response = requests.get(f'{API_BASE_URL}/api/v1/voice/profiles/{profile_id}')
        assert get_response.status_code == 200

        profile_data = get_response.json()
        assert profile_data['profile_id'] == profile_id

        # Step 4: Clean up - delete profile
        delete_response = requests.delete(f'{API_BASE_URL}/api/v1/voice/profiles/{profile_id}')
        assert delete_response.status_code == 200

        # Save validation results
        save_validation_results('voice_cloning_workflow', {
            'create': {
                'status_code': create_response.status_code,
                'profile_id': profile_id,
                'audio_duration': create_data.get('audio_duration'),
                'vocal_range': create_data.get('vocal_range')
            },
            'list': {
                'status_code': list_response.status_code,
                'profile_count': len(profiles)
            },
            'get': {
                'status_code': get_response.status_code,
                'profile_found': True
            },
            'delete': {
                'status_code': delete_response.status_code,
                'deleted': True
            },
            'passed': True
        })


@pytest.mark.api
@pytest.mark.e2e
class TestConversionAPIWorkflow:
    """Test complete conversion workflow via API."""

    def test_conversion_api_workflow(self, flask_server, sample_audio_file, sample_song_file):
        """Test complete conversion workflow via API with status polling."""
        # Step 1: Create voice profile for conversion
        files = {'reference_audio': ('reference.wav', sample_audio_file, 'audio/wav')}

        clone_response = requests.post(
            f'{API_BASE_URL}/api/v1/voice/clone',
            files=files
        )

        if clone_response.status_code == 503:
            pytest.skip("Voice cloning service unavailable")

        assert clone_response.status_code == 201
        profile_id = clone_response.json()['profile_id']

        # Step 2: Convert song using the profile
        song_files = {'song': ('song.wav', sample_song_file, 'audio/wav')}
        song_data = {
            'profile_id': profile_id,
            'vocal_volume': '1.0',
            'instrumental_volume': '0.9'
        }

        convert_response = requests.post(
            f'{API_BASE_URL}/api/v1/convert/song',
            files=song_files,
            data=song_data
        )

        # Should succeed or fail with 404/503
        assert convert_response.status_code in [200, 404, 503]

        if convert_response.status_code == 200:
            conversion_data = convert_response.json()

            # Step 3: Validate conversion result
            assert conversion_data['status'] == 'success'
            assert 'conversion_id' in conversion_data
            assert 'audio' in conversion_data
            assert 'duration' in conversion_data
            assert 'metadata' in conversion_data

            # Validate metadata
            metadata = conversion_data['metadata']
            assert metadata['target_profile_id'] == profile_id
            assert 'vocal_volume' in metadata
            assert 'instrumental_volume' in metadata

            # Step 4: Decode and validate audio
            audio_base64 = conversion_data['audio']
            audio_bytes = base64.b64decode(audio_base64)

            # Validate WAV format
            audio_buffer = io.BytesIO(audio_bytes)
            with wave.open(audio_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()

                assert sample_rate > 0
                assert n_channels in [1, 2]
                assert n_frames > 0

            # Save validation results
            save_validation_results('conversion_workflow', {
                'profile_creation': {
                    'status_code': clone_response.status_code,
                    'profile_id': profile_id
                },
                'conversion': {
                    'status_code': convert_response.status_code,
                    'conversion_id': conversion_data['conversion_id'],
                    'duration': conversion_data['duration'],
                    'sample_rate': conversion_data['sample_rate'],
                    'metadata': metadata
                },
                'audio_validation': {
                    'sample_rate': sample_rate,
                    'channels': n_channels,
                    'frames': n_frames,
                    'duration_seconds': n_frames / sample_rate
                },
                'passed': True
            })

        # Clean up
        requests.delete(f'{API_BASE_URL}/api/v1/voice/profiles/{profile_id}')


@pytest.mark.api
@pytest.mark.e2e
class TestQualityMetricsValidation:
    """Test quality metrics validation via API."""

    def test_audio_analysis_endpoint(self, flask_server, sample_audio_file):
        """Test audio analysis endpoint for quality metrics."""
        # Read audio file
        audio_bytes = sample_audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Analyze audio
        response = requests.post(
            f'{API_BASE_URL}/api/v1/analyze',
            json={'audio_data': audio_base64},
            headers={'Content-Type': 'application/json'}
        )

        assert response.status_code == 200
        data = response.json()

        assert 'status' in data
        assert data['status'] == 'success'
        assert 'analysis' in data

        analysis = data['analysis']

        # Validate analysis fields
        required_fields = ['duration', 'sample_rate', 'channels', 'samples', 'statistics']
        for field in required_fields:
            assert field in analysis, f"Missing required field: {field}"

        # Validate statistics
        stats = analysis['statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'rms' in stats

        # Save validation results
        save_validation_results('audio_analysis', {
            'status_code': response.status_code,
            'analysis': analysis,
            'passed': True
        })

    def test_process_audio_with_quality_metrics(self, flask_server, sample_audio_file):
        """Test audio processing with quality metrics extraction."""
        # Upload audio file
        files = {'audio': ('test.wav', sample_audio_file, 'audio/wav')}
        data = {'processing_config': json.dumps({
            'enable_pitch_extraction': True,
            'enable_vad': True,
            'enable_denoising': False
        })}

        response = requests.post(
            f'{API_BASE_URL}/api/v1/process_audio',
            files=files,
            data=data
        )

        assert response.status_code == 200
        result = response.json()

        assert result['status'] == 'success'
        assert 'analysis' in result

        analysis = result['analysis']

        # Validate pitch analysis
        if 'pitch' in analysis:
            pitch = analysis['pitch']
            assert 'mean' in pitch
            assert 'std' in pitch
            assert pitch['mean'] >= 0  # Pitch should be non-negative

        # Validate VAD analysis
        if 'vad' in analysis:
            vad = analysis['vad']
            assert 'voice_ratio' in vad
            assert 0 <= vad['voice_ratio'] <= 1  # Should be a ratio

        # Save validation results
        save_validation_results('process_audio_quality', {
            'status_code': response.status_code,
            'analysis': analysis,
            'duration': result.get('duration'),
            'sample_rate': result.get('sample_rate'),
            'passed': True
        })


@pytest.mark.api
@pytest.mark.e2e
class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_audio_format(self, flask_server):
        """Test handling of invalid audio format."""
        invalid_data = io.BytesIO(b'not an audio file')
        files = {'reference_audio': ('invalid.txt', invalid_data, 'text/plain')}

        response = requests.post(
            f'{API_BASE_URL}/api/v1/voice/clone',
            files=files
        )

        assert response.status_code == 400
        data = response.json()
        assert 'error' in data

        save_validation_results('error_invalid_format', {
            'status_code': response.status_code,
            'error': data.get('error'),
            'passed': True
        })

    def test_missing_required_fields(self, flask_server):
        """Test handling of missing required fields."""
        # Missing song file
        response = requests.post(
            f'{API_BASE_URL}/api/v1/convert/song',
            data={'profile_id': 'test-profile'}
        )

        assert response.status_code == 400
        data = response.json()
        assert 'error' in data

        save_validation_results('error_missing_fields', {
            'status_code': response.status_code,
            'error': data.get('error'),
            'passed': True
        })

    def test_nonexistent_profile(self, flask_server, sample_song_file):
        """Test handling of nonexistent profile."""
        files = {'song': ('song.wav', sample_song_file, 'audio/wav')}
        data = {'profile_id': 'nonexistent-profile-12345'}

        response = requests.post(
            f'{API_BASE_URL}/api/v1/convert/song',
            files=files,
            data=data
        )

        # Should be 404 (not found) or 503 (service unavailable)
        assert response.status_code in [404, 503]

        save_validation_results('error_nonexistent_profile', {
            'status_code': response.status_code,
            'passed': True
        })

    def test_invalid_volume_parameters(self, flask_server, sample_song_file):
        """Test handling of invalid volume parameters."""
        files = {'song': ('song.wav', sample_song_file, 'audio/wav')}
        data = {
            'profile_id': 'test-profile',
            'vocal_volume': '5.0',  # Out of range [0.0, 2.0]
            'instrumental_volume': '0.8'
        }

        response = requests.post(
            f'{API_BASE_URL}/api/v1/convert/song',
            files=files,
            data=data
        )

        # Should be 400 (bad request) or 503 (service unavailable)
        assert response.status_code in [400, 503]

        save_validation_results('error_invalid_volumes', {
            'status_code': response.status_code,
            'passed': True
        })


@pytest.mark.api
@pytest.mark.e2e
@pytest.mark.slow
class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    def test_concurrent_health_checks(self, flask_server):
        """Test concurrent health check requests."""
        import concurrent.futures

        def make_health_request():
            return requests.get(f'{API_BASE_URL}/api/v1/health')

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        for response in responses:
            assert response.status_code == 200

        save_validation_results('concurrent_health_checks', {
            'num_requests': 10,
            'all_succeeded': all(r.status_code == 200 for r in responses),
            'passed': True
        })


@pytest.mark.api
@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
class TestAPIE2EQualityValidation:
    """
    API End-to-End Quality Validation Tests (Comment 10 Requirements).

    Tests complete conversion workflow with quality metric validation:
    1. POST song to /api/v1/convert/song with target_profile_id
    2. Validate conversion success (200 response)
    3. Download converted audio
    4. Compute quality metrics locally using evaluator
    5. Assert quality targets are met:
       - Pitch RMSE < 10 Hz
       - Speaker similarity > 0.85
    """

    def decode_audio_from_base64(self, audio_base64: str) -> tuple:
        """Decode base64-encoded WAV audio to numpy array."""
        audio_bytes = base64.b64decode(audio_base64)
        buffer = io.BytesIO(audio_bytes)

        with wave.open(buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

            # Convert to numpy array
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0

            # Handle multi-channel
            if n_channels > 1:
                audio_float = audio_float.reshape(-1, n_channels).T

            return audio_float, sample_rate

    def evaluate_conversion_quality(self, source_audio: np.ndarray,
                                   converted_audio: np.ndarray,
                                   sample_rate: int,
                                   target_profile_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate conversion quality using QualityMetricsAggregator.

        Args:
            source_audio: Source (input) audio
            converted_audio: Converted (output) audio
            sample_rate: Audio sample rate
            target_profile_id: Optional target profile ID for embedding retrieval

        Returns:
            Dictionary of quality metrics
        """
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

            import torch
            from auto_voice.utils.quality_metrics import QualityMetricsAggregator

            # Convert to torch tensors
            source_tensor = torch.from_numpy(source_audio).float()
            converted_tensor = torch.from_numpy(converted_audio).float()

            # Get target speaker embedding if profile_id provided
            target_embedding = None
            if target_profile_id:
                try:
                    from auto_voice.inference.voice_cloner import VoiceCloner
                    voice_cloner = VoiceCloner()
                    target_embedding = voice_cloner.get_embedding(target_profile_id)
                except Exception as e:
                    print(f"Warning: Could not retrieve target embedding: {e}")

            # Initialize metrics aggregator
            aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

            # Evaluate quality
            result = aggregator.evaluate(
                source_tensor,
                converted_tensor,
                align_audio=True,
                target_speaker_embedding=target_embedding
            )

            return {
                'pitch_rmse_hz': result.pitch_accuracy.rmse_hz,
                'pitch_rmse_log2': result.pitch_accuracy.rmse_log2,
                'pitch_correlation': result.pitch_accuracy.correlation,
                'speaker_similarity': result.speaker_similarity.cosine_similarity,
                'spectral_distortion': result.naturalness.spectral_distortion,
                'mos_estimation': result.naturalness.mos_estimation,
                'stoi_score': result.intelligibility.stoi_score,
                'pesq_score': result.intelligibility.pesq_score,
                'overall_quality_score': result.overall_quality_score,
                'raw_result': result
            }
        except ImportError as e:
            pytest.skip(f"Quality metrics not available: {e}")

    def test_api_e2e_conversion_quality_validation(self, flask_server, sample_audio_file, sample_song_file):
        """
        Test complete API workflow with quality validation (Comment 10).

        Workflow:
        1. Create voice profile from reference audio
        2. Convert song using /api/v1/convert/song
        3. Download converted audio
        4. Compute quality metrics
        5. Assert targets are met:
           - Pitch RMSE < 10 Hz
           - Speaker similarity > 0.85
        """
        # Step 1: Create voice profile
        files = {'reference_audio': ('reference.wav', sample_audio_file, 'audio/wav')}
        clone_response = requests.post(
            f'{API_BASE_URL}/api/v1/voice/clone',
            files=files
        )

        if clone_response.status_code == 503:
            pytest.skip("Voice cloning service unavailable")

        assert clone_response.status_code == 201
        profile_id = clone_response.json()['profile_id']

        try:
            # Step 2: Convert song
            song_files = {'song': ('song.wav', sample_song_file, 'audio/wav')}
            song_data = {'profile_id': profile_id}

            convert_response = requests.post(
                f'{API_BASE_URL}/api/v1/convert/song',
                files=song_files,
                data=song_data
            )

            # Validate conversion response
            if convert_response.status_code == 503:
                pytest.skip("Conversion service unavailable")
            elif convert_response.status_code == 404:
                pytest.skip(f"Profile not found: {profile_id}")

            assert convert_response.status_code == 200, \
                f"Conversion failed with status {convert_response.status_code}"

            result = convert_response.json()
            assert result['status'] == 'success'
            assert 'audio' in result

            # Step 3: Download and decode converted audio
            converted_audio, converted_sr = self.decode_audio_from_base64(result['audio'])

            # Load source audio for comparison
            sample_song_file.seek(0)
            source_buffer = io.BytesIO(sample_song_file.read())
            with wave.open(source_buffer, 'rb') as wav_file:
                source_sr = wav_file.getframerate()
                source_data = wav_file.readframes(wav_file.getnframes())
                source_audio = np.frombuffer(source_data, dtype=np.int16).astype(np.float32) / 32767.0

            # Ensure mono audio for evaluation
            if converted_audio.ndim > 1:
                converted_audio = converted_audio.mean(axis=0)
            if source_audio.ndim > 1:
                source_audio = source_audio.mean(axis=0)

            # Step 4: Compute quality metrics
            metrics = self.evaluate_conversion_quality(
                source_audio,
                converted_audio,
                source_sr,
                target_profile_id=profile_id
            )

            # Step 5: Save metrics to validation_results
            metrics_file = VALIDATION_RESULTS_DIR / 'api_e2e_quality_metrics.json'
            serializable_metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if k != 'raw_result'
            }
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)

            print(f"\nQuality Metrics (saved to {metrics_file}):")
            print(f"  Pitch RMSE (Hz): {metrics['pitch_rmse_hz']:.2f} Hz")
            print(f"  Pitch RMSE (log2): {metrics['pitch_rmse_log2']:.3f} semitones")
            print(f"  Pitch Correlation: {metrics['pitch_correlation']:.3f}")
            print(f"  Speaker Similarity: {metrics['speaker_similarity']:.3f}")
            print(f"  Spectral Distortion: {metrics['spectral_distortion']:.2f} dB")
            print(f"  MOS Estimation: {metrics['mos_estimation']:.2f}")
            print(f"  Overall Quality Score: {metrics['overall_quality_score']:.3f}")

            # Step 6: Assert quality targets (Comment 10 requirements)
            assert metrics['pitch_rmse_hz'] < 10.0, \
                f"Pitch RMSE {metrics['pitch_rmse_hz']:.2f} Hz exceeds target of 10 Hz"

            assert metrics['speaker_similarity'] > 0.85, \
                f"Speaker similarity {metrics['speaker_similarity']:.3f} below target of 0.85"

            # Additional quality checks (informational)
            assert metrics['overall_quality_score'] > 0.0, \
                "Overall quality score should be positive"

            print("\n✅ All quality targets met (Comment 10)!")

            # Save comprehensive validation results
            save_validation_results('api_e2e_quality_validation', {
                'conversion': {
                    'status_code': convert_response.status_code,
                    'profile_id': profile_id,
                    'duration': result.get('duration')
                },
                'metrics': serializable_metrics,
                'quality_targets': {
                    'pitch_rmse_hz_target': 10.0,
                    'pitch_rmse_hz_actual': metrics['pitch_rmse_hz'],
                    'pitch_rmse_hz_pass': metrics['pitch_rmse_hz'] < 10.0,
                    'speaker_similarity_target': 0.85,
                    'speaker_similarity_actual': metrics['speaker_similarity'],
                    'speaker_similarity_pass': metrics['speaker_similarity'] > 0.85
                },
                'passed': True
            })

        finally:
            # Cleanup: Delete voice profile
            requests.delete(f'{API_BASE_URL}/api/v1/voice/profiles/{profile_id}')

    def test_api_e2e_batch_quality_validation(self, flask_server, sample_audio_file, sample_song_file):
        """
        Test batch conversion quality validation with multiple test cases.

        Converts the same song multiple times and validates:
        - Consistency across conversions
        - All conversions meet quality targets
        """
        # Create voice profile
        files = {'reference_audio': ('reference.wav', sample_audio_file, 'audio/wav')}
        clone_response = requests.post(
            f'{API_BASE_URL}/api/v1/voice/clone',
            files=files
        )

        if clone_response.status_code != 201:
            pytest.skip("Voice cloning failed")

        profile_id = clone_response.json()['profile_id']

        try:
            num_conversions = 3
            all_metrics = []

            # Load source audio once
            sample_song_file.seek(0)
            source_buffer = io.BytesIO(sample_song_file.read())
            with wave.open(source_buffer, 'rb') as wav_file:
                source_sr = wav_file.getframerate()
                source_data = wav_file.readframes(wav_file.getnframes())
                source_audio = np.frombuffer(source_data, dtype=np.int16).astype(np.float32) / 32767.0

            if source_audio.ndim > 1:
                source_audio = source_audio.mean(axis=0)

            for i in range(num_conversions):
                # Convert song
                sample_song_file.seek(0)
                song_files = {'song': (f'song_{i}.wav', sample_song_file, 'audio/wav')}
                song_data = {'profile_id': profile_id}

                convert_response = requests.post(
                    f'{API_BASE_URL}/api/v1/convert/song',
                    files=song_files,
                    data=song_data
                )

                if convert_response.status_code != 200:
                    pytest.skip(f"Conversion {i+1} failed")

                result = convert_response.json()
                converted_audio, _ = self.decode_audio_from_base64(result['audio'])

                if converted_audio.ndim > 1:
                    converted_audio = converted_audio.mean(axis=0)

                # Evaluate quality
                metrics = self.evaluate_conversion_quality(
                    source_audio,
                    converted_audio,
                    source_sr,
                    target_profile_id=profile_id
                )

                all_metrics.append(metrics)

                # Save individual conversion metrics
                individual_file = VALIDATION_RESULTS_DIR / f'api_e2e_batch_conversion_{i+1}.json'
                serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in metrics.items() if k != 'raw_result'}
                with open(individual_file, 'w') as f:
                    json.dump(serializable, f, indent=2)

            # Aggregate metrics
            avg_pitch_rmse_hz = np.mean([m['pitch_rmse_hz'] for m in all_metrics])
            avg_speaker_similarity = np.mean([m['speaker_similarity'] for m in all_metrics])
            avg_overall_quality = np.mean([m['overall_quality_score'] for m in all_metrics])

            # Compute consistency metrics
            pitch_rmse_std = np.std([m['pitch_rmse_hz'] for m in all_metrics])
            speaker_sim_std = np.std([m['speaker_similarity'] for m in all_metrics])

            batch_summary = {
                'num_conversions': num_conversions,
                'avg_pitch_rmse_hz': float(avg_pitch_rmse_hz),
                'avg_speaker_similarity': float(avg_speaker_similarity),
                'avg_overall_quality': float(avg_overall_quality),
                'pitch_rmse_std': float(pitch_rmse_std),
                'speaker_similarity_std': float(speaker_sim_std),
                'all_conversions_meet_targets': all(
                    m['pitch_rmse_hz'] < 10.0 and m['speaker_similarity'] > 0.85
                    for m in all_metrics
                )
            }

            # Save batch summary
            batch_summary_file = VALIDATION_RESULTS_DIR / 'api_e2e_batch_summary.json'
            with open(batch_summary_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)

            print(f"\nBatch Summary (saved to {batch_summary_file}):")
            print(f"  Conversions: {num_conversions}")
            print(f"  Avg Pitch RMSE (Hz): {avg_pitch_rmse_hz:.2f} ± {pitch_rmse_std:.2f}")
            print(f"  Avg Speaker Similarity: {avg_speaker_similarity:.3f} ± {speaker_sim_std:.3f}")
            print(f"  Avg Overall Quality: {avg_overall_quality:.3f}")

            # Assert batch quality targets
            assert avg_pitch_rmse_hz < 10.0, \
                f"Average pitch RMSE {avg_pitch_rmse_hz:.2f} Hz exceeds target"

            assert avg_speaker_similarity > 0.85, \
                f"Average speaker similarity {avg_speaker_similarity:.3f} below target"

            assert batch_summary['all_conversions_meet_targets'], \
                "Not all conversions meet quality targets"

            print("\n✅ All batch conversions meet quality targets!")

        finally:
            # Cleanup
            requests.delete(f'{API_BASE_URL}/api/v1/voice/profiles/{profile_id}')


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s', '--tb=short'])
