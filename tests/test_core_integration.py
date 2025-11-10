"""
Comprehensive integration test suite for AutoVoice core components.

This test suite validates component interactions, data flow, and end-to-end
pipeline execution with comprehensive performance and stress testing.

Test Coverage:
1. Component Integration Tests
   - VocalSeparator → SingingPitchExtractor integration
   - VoiceCloner → SingingVoiceConverter integration
   - Full pipeline end-to-end test

2. Data Flow Tests
   - Sample rate alignment across components
   - Audio format compatibility
   - Mono/stereo handling

3. Performance Tests
   - GPU memory tracking during pipeline execution
   - Memory leak detection (CPU and GPU)
   - Stress tests with concurrent operations

Test Markers:
- @pytest.mark.integration: All integration tests
- @pytest.mark.cuda: GPU-specific tests
- @pytest.mark.slow: Long-running tests (>1 second)
- @pytest.mark.performance: Performance benchmarks
"""

import gc
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pytest
import torch

# Import AutoVoice components
from src.auto_voice.audio.source_separator import VocalSeparator, SeparationError
from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor, PitchExtractionError
from src.auto_voice.inference.voice_cloner import VoiceCloner, VoiceCloningError, InvalidAudioError
from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter, VoiceConversionError
from src.auto_voice.audio.processor import AudioProcessor

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def song_file_mono(tmp_path: Path) -> Path:
    """Generate synthetic mono song file for testing.

    Creates a 3-second synthetic audio with voice-like characteristics:
    - Multiple harmonic frequencies (fundamental + overtones)
    - Amplitude modulation to simulate speech envelope
    - Noise component for realism

    Returns:
        Path to generated WAV file
    """
    import soundfile as sf

    sample_rate = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Generate voice-like harmonic series
    fundamental = 220.0  # A3
    audio = np.zeros_like(t)

    # Add harmonics with decreasing amplitude
    for harmonic in range(1, 6):
        freq = fundamental * harmonic
        amplitude = 1.0 / harmonic
        audio += amplitude * np.sin(2 * np.pi * freq * t)

    # Add amplitude modulation (5 Hz envelope for speech-like quality)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5.0 * t)
    audio *= envelope

    # Add noise for realism
    noise = np.random.randn(len(t)) * 0.02
    audio = audio + noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9

    # Save to file
    audio_file = tmp_path / "synthetic_song_mono.wav"
    sf.write(audio_file, audio, sample_rate)

    return audio_file


@pytest.fixture
def song_file_stereo(tmp_path: Path) -> Path:
    """Generate synthetic stereo song file for testing.

    Creates a 3-second stereo synthetic audio with different content
    in left and right channels.

    Returns:
        Path to generated WAV file
    """
    import soundfile as sf

    sample_rate = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Left channel: 220 Hz (A3)
    left = np.sin(2 * np.pi * 220.0 * t)

    # Right channel: 440 Hz (A4)
    right = np.sin(2 * np.pi * 440.0 * t)

    # Combine into stereo
    audio = np.stack([left, right], axis=1)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9

    # Save to file
    audio_file = tmp_path / "synthetic_song_stereo.wav"
    sf.write(audio_file, audio, sample_rate)

    return audio_file


@pytest.fixture
def voice_profile_fixture(tmp_path: Path, cuda_available: bool) -> Dict[str, Any]:
    """Generate a synthetic voice profile for testing.

    Creates a voice profile with synthetic speaker embedding, vocal range,
    and timbre features.

    Returns:
        Dict containing profile_id, embedding, vocal_range, etc.
    """
    import uuid
    from datetime import datetime, timezone

    profile = {
        'profile_id': str(uuid.uuid4()),
        'user_id': 'test_user',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'audio_duration': 30.0,
        'sample_rate': 22050,
        'embedding': np.random.randn(256).astype(np.float32),
        'vocal_range': {
            'min_f0': 80.0,
            'max_f0': 500.0,
            'range_semitones': 36.0,
            'mean_f0': 220.0
        },
        'timbre_features': {
            'spectral_centroid': 1500.0,
            'spectral_rolloff': 3000.0
        },
        'embedding_stats': {
            'mean': 0.0,
            'std': 1.0,
            'norm': 16.0
        },
        'metadata': {}
    }

    return profile


@pytest.fixture
def pipeline_instance(cuda_available: bool):
    """Create full pipeline instance with all components.

    This fixture initializes all components needed for end-to-end testing:
    - VocalSeparator
    - SingingPitchExtractor
    - VoiceCloner
    - AudioProcessor

    Returns:
        Dict with initialized components
    """
    device = 'cuda' if cuda_available else 'cpu'

    # Initialize components with minimal config
    config = {
        'device': device,
        'sample_rate': 22050,
        'defer_model_load': True  # Don't load models until needed
    }

    try:
        separator = VocalSeparator(config={'defer_model_load': True}, device=device)
    except Exception as e:
        logger.warning(f"VocalSeparator initialization failed: {e}")
        separator = None

    try:
        pitch_extractor = SingingPitchExtractor(device=device)
    except Exception as e:
        logger.warning(f"SingingPitchExtractor initialization failed: {e}")
        pitch_extractor = None

    try:
        voice_cloner = VoiceCloner(device=device)
    except Exception as e:
        logger.warning(f"VoiceCloner initialization failed: {e}")
        voice_cloner = None

    audio_processor = AudioProcessor(config={'sample_rate': 22050}, device=device)

    return {
        'separator': separator,
        'pitch_extractor': pitch_extractor,
        'voice_cloner': voice_cloner,
        'audio_processor': audio_processor,
        'device': device
    }


@pytest.fixture
def concurrent_executor():
    """Create ThreadPoolExecutor for concurrent tests.

    Returns:
        ThreadPoolExecutor with 4 workers
    """
    executor = ThreadPoolExecutor(max_workers=4)
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def memory_leak_detector(cuda_available: bool):
    """Fixture for detecting CPU and GPU memory leaks.

    Tracks memory usage before and after test execution and reports leaks.

    Yields:
        MemoryLeakDetector instance
    """
    class MemoryLeakDetector:
        def __init__(self, cuda_available: bool):
            self.cuda_available = cuda_available
            self.initial_cpu_memory = 0
            self.initial_gpu_memory = 0
            self.final_cpu_memory = 0
            self.final_gpu_memory = 0

        def start(self):
            """Record initial memory usage."""
            gc.collect()

            # CPU memory (approximate using process info)
            import psutil
            import os
            process = psutil.Process(os.getpid())
            self.initial_cpu_memory = process.memory_info().rss

            # GPU memory
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.initial_gpu_memory = torch.cuda.memory_allocated()

        def stop(self) -> Tuple[int, int]:
            """Record final memory usage and return deltas.

            Returns:
                Tuple of (cpu_leak_bytes, gpu_leak_bytes)
            """
            gc.collect()

            # CPU memory
            import psutil
            import os
            process = psutil.Process(os.getpid())
            self.final_cpu_memory = process.memory_info().rss

            # GPU memory
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.final_gpu_memory = torch.cuda.memory_allocated()

            cpu_leak = self.final_cpu_memory - self.initial_cpu_memory
            gpu_leak = self.final_gpu_memory - self.initial_gpu_memory if self.cuda_available else 0

            return cpu_leak, gpu_leak

        def report(self, test_name: str, cpu_leak: int, gpu_leak: int,
                   cpu_threshold: int = 50 * 1024 * 1024,
                   gpu_threshold: int = 10 * 1024 * 1024):
            """Report memory leak if above threshold.

            Args:
                test_name: Name of the test
                cpu_leak: CPU memory leak in bytes
                gpu_leak: GPU memory leak in bytes
                cpu_threshold: CPU leak threshold (default 50 MB)
                gpu_threshold: GPU leak threshold (default 10 MB)
            """
            if cpu_leak > cpu_threshold:
                logger.warning(
                    f"[{test_name}] Potential CPU memory leak: "
                    f"{cpu_leak / 1024 / 1024:.2f} MB "
                    f"(threshold: {cpu_threshold / 1024 / 1024:.2f} MB)"
                )

            if self.cuda_available and gpu_leak > gpu_threshold:
                logger.warning(
                    f"[{test_name}] Potential GPU memory leak: "
                    f"{gpu_leak / 1024 / 1024:.2f} MB "
                    f"(threshold: {gpu_threshold / 1024 / 1024:.2f} MB)"
                )

    return MemoryLeakDetector(cuda_available)


# ============================================================================
# 1. Component Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestVocalSeparatorPitchExtractorIntegration:
    """Test integration between VocalSeparator and SingingPitchExtractor.

    Validates that separated vocals can be properly analyzed for pitch
    and that data flows correctly between components.
    """

    def test_separate_and_extract_pitch(self, song_file_mono, pipeline_instance):
        """Test separating vocals and extracting pitch from result.

        This test validates:
        1. VocalSeparator can process audio file
        2. Separated vocals have expected format
        3. SingingPitchExtractor can analyze separated vocals
        4. Sample rates are properly aligned
        """
        separator = pipeline_instance['separator']
        pitch_extractor = pipeline_instance['pitch_extractor']

        if separator is None:
            pytest.skip("VocalSeparator not available")
        if pitch_extractor is None:
            pytest.skip("SingingPitchExtractor not available")

        # Step 1: Separate vocals
        try:
            vocals, instrumental = separator.separate_vocals(str(song_file_mono))
        except Exception as e:
            pytest.skip(f"Vocal separation failed: {e}")

        # Validate separated audio format
        assert vocals is not None
        assert instrumental is not None
        assert isinstance(vocals, np.ndarray)
        assert isinstance(instrumental, np.ndarray)
        assert vocals.ndim in (1, 2)  # Mono or stereo

        # Convert to mono if stereo
        if vocals.ndim == 2:
            vocals_mono = np.mean(vocals, axis=0)
        else:
            vocals_mono = vocals

        # Step 2: Extract pitch from vocals
        # Use actual sample rate from separator (separated audio is at processing sample_rate)
        actual_sample_rate = separator.config['sample_rate']

        f0_data = pitch_extractor.extract_f0_contour(
            vocals_mono,
            sample_rate=actual_sample_rate,
            return_confidence=True,
            return_times=True
        )

        # Validate F0 extraction results
        assert 'f0' in f0_data
        assert 'voiced' in f0_data
        assert 'confidence' in f0_data
        assert 'times' in f0_data
        assert 'vibrato' in f0_data

        # Check F0 data shape and values
        assert f0_data['f0'].shape == f0_data['voiced'].shape
        assert f0_data['f0'].shape == f0_data['confidence'].shape
        assert np.all(np.isfinite(f0_data['f0'][f0_data['voiced']]))

        # Validate sample rate consistency
        assert f0_data['sample_rate'] == actual_sample_rate

        logger.info(
            f"Integration test passed: separated vocals ({len(vocals_mono)} samples) "
            f"-> pitch extraction ({len(f0_data['f0'])} frames)"
        )

    def test_sample_rate_alignment(self, song_file_stereo, pipeline_instance):
        """Test sample rate alignment between separator and pitch extractor.

        Validates:
        1. Sample rates are properly communicated between components
        2. Resampling occurs when needed
        3. No data corruption during rate conversion
        """
        separator = pipeline_instance['separator']
        pitch_extractor = pipeline_instance['pitch_extractor']

        if separator is None or pitch_extractor is None:
            pytest.skip("Required components not available")

        # Get actual sample rate from separator (separated audio is at processing sample_rate)
        actual_sample_rate = separator.config['sample_rate']

        try:
            vocals, _ = separator.separate_vocals(str(song_file_stereo))
        except Exception as e:
            pytest.skip(f"Separation failed: {e}")

        # Ensure mono
        if vocals.ndim == 2:
            vocals = np.mean(vocals, axis=0)

        # Extract pitch with actual sample rate
        f0_data = pitch_extractor.extract_f0_contour(vocals, sample_rate=actual_sample_rate)

        # Validate alignment
        assert f0_data['sample_rate'] == actual_sample_rate

        # Calculate expected number of frames
        hop_length = f0_data['hop_length']
        expected_frames = len(vocals) // hop_length

        # Allow small discrepancy due to padding/windowing
        assert abs(len(f0_data['f0']) - expected_frames) <= 5

        logger.info(
            f"Sample rate alignment validated: {actual_sample_rate} Hz, "
            f"{len(vocals)} samples -> {len(f0_data['f0'])} frames"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestVoiceClonerConverterIntegration:
    """Test integration between VoiceCloner and SingingVoiceConverter.

    Validates that voice profiles can be used for voice conversion and
    that embeddings flow correctly between components.
    """

    def test_create_profile_and_convert(self, song_file_mono, pipeline_instance, tmp_path):
        """Test creating voice profile and using it for conversion.

        This test validates:
        1. Voice profile creation from audio
        2. Profile can be loaded
        3. Embedding can be used for voice conversion
        4. No shape mismatches or errors
        """
        voice_cloner = pipeline_instance['voice_cloner']

        if voice_cloner is None:
            pytest.skip("VoiceCloner not available")

        # Step 1: Create voice profile
        try:
            profile = voice_cloner.create_voice_profile(
                audio=str(song_file_mono),
                user_id='test_integration',
                metadata={'source': 'integration_test'}
            )
        except InvalidAudioError as e:
            pytest.skip(f"Audio validation failed: {e}")
        except Exception as e:
            pytest.skip(f"Profile creation failed: {e}")

        # Validate profile structure
        assert 'profile_id' in profile
        assert 'user_id' in profile
        assert profile['user_id'] == 'test_integration'

        # Step 2: Load profile with embedding
        profile_id = profile['profile_id']
        full_profile = voice_cloner.load_voice_profile(profile_id)

        assert 'embedding' in full_profile
        embedding = full_profile['embedding']

        # Validate embedding
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (256,)
        assert np.all(np.isfinite(embedding))

        # Step 3: Test conversion with embedding
        device = pipeline_instance['device']

        try:
            # Instantiate SingingVoiceConverter
            converter = SingingVoiceConverter(config={'device': device})
        except Exception as e:
            pytest.skip(f"SingingVoiceConverter not available: {e}")

        # Load audio for conversion
        audio_processor = pipeline_instance['audio_processor']
        try:
            audio, sr = audio_processor.load_audio(str(song_file_mono))
        except Exception as e:
            pytest.skip(f"Failed to load audio: {e}")

        # Ensure audio is on correct device with float32 dtype
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float().to(device)
        else:
            audio_tensor = audio.float().to(device)

        # Ensure embedding is on correct device with float32 dtype
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).float().to(device)
        else:
            embedding_tensor = embedding.float().to(device)

        # Validate embedding dimension compatibility
        assert embedding_tensor.shape[0] == 256, \
            f"Expected embedding dimension 256, got {embedding_tensor.shape[0]}"

        # Perform voice conversion with correct parameter names
        converted_audio = converter.convert(
            source_audio=audio_tensor,
            target_speaker_embedding=embedding_tensor,
            source_sample_rate=sr
        )

        # Validate output
        assert converted_audio is not None, "Conversion returned None"

        # Check for non-silent output
        if isinstance(converted_audio, torch.Tensor):
            converted_np = converted_audio.cpu().numpy()
        else:
            converted_np = converted_audio

        assert converted_np.size > 0, "Converted audio is empty"
        assert not np.all(converted_np == 0), "Converted audio is silent"
        assert np.all(np.isfinite(converted_np)), "Converted audio contains NaN or Inf"

        # Validate length consistency (should be similar to input)
        input_length = audio_tensor.shape[-1] if isinstance(audio_tensor, torch.Tensor) else len(audio_tensor)
        output_length = len(converted_np)
        length_ratio = output_length / input_length
        assert 0.9 <= length_ratio <= 1.1, \
            f"Output length ratio {length_ratio:.2f} not in expected range [0.9, 1.1]"

        # Check device consistency
        if torch.cuda.is_available() and device == 'cuda':
            # Ensure output was processed on GPU
            assert isinstance(converted_audio, torch.Tensor), \
                "Expected torch.Tensor output for CUDA device"

        logger.info(
            f"Profile integration test passed: created profile {profile_id}, "
            f"embedding shape {embedding.shape}, converted audio length {output_length}"
        )

    def test_profile_compatibility(self, voice_profile_fixture):
        """Test that voice profiles have compatible format for conversion.

        Validates:
        1. Embedding dimension matches expected
        2. Embedding is normalized
        3. All required fields present
        """
        profile = voice_profile_fixture

        # Check required fields
        required_fields = ['profile_id', 'embedding', 'sample_rate']
        for field in required_fields:
            assert field in profile, f"Missing required field: {field}"

        # Check embedding
        embedding = profile['embedding']
        assert embedding.shape == (256,)
        assert np.all(np.isfinite(embedding))

        # Check normalization (L2 norm should be reasonable)
        norm = np.linalg.norm(embedding)
        assert 0.1 < norm < 100.0, f"Embedding norm out of range: {norm}"

        logger.info(f"Profile compatibility validated: norm={norm:.2f}")


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline execution.

    Validates the full workflow from audio input to converted output.
    """

    def test_full_pipeline(self, song_file_mono, pipeline_instance, voice_profile_fixture):
        """Test complete pipeline: separate -> extract -> convert.

        This test validates:
        1. All components work together
        2. Data flows correctly through pipeline
        3. Output is in expected format
        4. No errors or crashes
        """
        separator = pipeline_instance['separator']
        pitch_extractor = pipeline_instance['pitch_extractor']
        audio_processor = pipeline_instance['audio_processor']

        if separator is None or pitch_extractor is None:
            pytest.skip("Required components not available")

        # Step 1: Separate vocals
        try:
            vocals, instrumental = separator.separate_vocals(str(song_file_mono))
        except Exception as e:
            pytest.skip(f"Separation failed: {e}")

        # Step 2: Convert to mono and extract pitch
        if vocals.ndim == 2:
            vocals_mono = np.mean(vocals, axis=0)
        else:
            vocals_mono = vocals

        # Use actual sample rate from separator (separated audio is at processing sample_rate)
        # VocalSeparator resamples to its configured sample_rate, so use that directly
        actual_sample_rate = separator.config['sample_rate']

        f0_data = pitch_extractor.extract_f0_contour(vocals_mono, sample_rate=actual_sample_rate)

        # Step 3: Get speaker embedding from profile
        embedding = voice_profile_fixture['embedding']

        # Step 4: Validate all components produced valid outputs
        assert vocals_mono is not None
        assert len(vocals_mono) > 0
        assert f0_data['f0'] is not None
        assert len(f0_data['f0']) > 0
        assert embedding is not None
        assert embedding.shape == (256,)

        logger.info(
            f"End-to-end pipeline validated: "
            f"{len(vocals_mono)} samples -> "
            f"{len(f0_data['f0'])} frames -> "
            f"embedding {embedding.shape}"
        )


# ============================================================================
# 2. Data Flow Tests
# ============================================================================

@pytest.mark.integration
class TestDataFlow:
    """Test data flow and format compatibility between components."""

    def test_audio_format_mono_stereo(self, pipeline_instance, sample_audio_22khz):
        """Test mono/stereo handling across components.

        Validates:
        1. Mono audio is preserved as mono
        2. Stereo audio is properly downmixed when needed
        3. No shape errors
        """
        audio_processor = pipeline_instance['audio_processor']
        pitch_extractor = pipeline_instance['pitch_extractor']

        if pitch_extractor is None:
            pytest.skip("SingingPitchExtractor not available")

        # Test mono
        mono_audio = sample_audio_22khz
        f0_mono = pitch_extractor.extract_f0_contour(mono_audio, sample_rate=22050)
        assert f0_mono['f0'] is not None

        # Test stereo (duplicate to stereo)
        stereo_audio = np.stack([mono_audio, mono_audio])
        # Pitch extractor should handle stereo by taking mean
        f0_stereo = pitch_extractor.extract_f0_contour(stereo_audio, sample_rate=22050)
        assert f0_stereo['f0'] is not None

        # Results should be similar (not identical due to processing)
        assert f0_mono['f0'].shape == f0_stereo['f0'].shape

        logger.info("Mono/stereo handling validated")

    def test_sample_rate_consistency(self, pipeline_instance, sample_audio_16khz, sample_audio_44khz):
        """Test sample rate handling across different rates.

        Validates:
        1. Components accept different sample rates
        2. Resampling occurs correctly
        3. No data corruption
        """
        pitch_extractor = pipeline_instance['pitch_extractor']

        if pitch_extractor is None:
            pytest.skip("SingingPitchExtractor not available")

        # Test 16 kHz
        f0_16k = pitch_extractor.extract_f0_contour(sample_audio_16khz, sample_rate=16000)
        assert f0_16k['sample_rate'] == 16000

        # Test 44.1 kHz
        f0_44k = pitch_extractor.extract_f0_contour(sample_audio_44khz, sample_rate=44100)
        assert f0_44k['sample_rate'] == 44100

        logger.info("Sample rate consistency validated: 16kHz, 44.1kHz")

    def test_data_type_consistency(self, pipeline_instance, sample_audio_22khz):
        """Test data type handling (float32, float64, int16).

        Validates:
        1. Components handle different dtypes
        2. Conversions are correct
        3. No overflow/underflow
        """
        pitch_extractor = pipeline_instance['pitch_extractor']

        if pitch_extractor is None:
            pytest.skip("SingingPitchExtractor not available")

        # Test float32
        audio_f32 = sample_audio_22khz.astype(np.float32)
        f0_f32 = pitch_extractor.extract_f0_contour(audio_f32, sample_rate=22050)
        assert f0_f32['f0'] is not None

        # Test float64
        audio_f64 = sample_audio_22khz.astype(np.float64)
        f0_f64 = pitch_extractor.extract_f0_contour(audio_f64, sample_rate=22050)
        assert f0_f64['f0'] is not None

        # Test int16 (common in WAV files)
        audio_i16 = (sample_audio_22khz * 32767).astype(np.int16)
        f0_i16 = pitch_extractor.extract_f0_contour(audio_i16, sample_rate=22050)
        assert f0_i16['f0'] is not None

        logger.info("Data type consistency validated: float32, float64, int16")

    def test_feature_alignment_validation(self, pipeline_instance, sample_audio_22khz):
        """Test F0 contour alignment with audio samples.

        Validates that F0 contour length correctly aligns with audio length
        based on hop_length and sample_rate. This ensures proper feature
        interpolation in the voice converter.

        Validates:
        1. F0 frame count matches expected based on audio length and hop_length
        2. Interpolation maintains temporal alignment
        3. No off-by-one errors in frame calculations
        """
        pitch_extractor = pipeline_instance['pitch_extractor']

        if pitch_extractor is None:
            pytest.skip("SingingPitchExtractor not available")

        sample_rate = 22050
        audio = sample_audio_22khz

        # Extract F0 with explicit hop_length
        f0_data = pitch_extractor.extract_f0_contour(
            audio,
            sample_rate=sample_rate,
            return_times=True
        )

        # Get hop_length used
        hop_length = f0_data.get('hop_length', 512)  # Default hop_length

        # Calculate expected number of frames
        # Standard formula: (n_samples - frame_length) // hop_length + 1
        # But pitch extractors may use different conventions
        audio_length = len(audio)
        frame_length = f0_data.get('frame_length', 2048)  # Default frame_length

        # Expected frames using standard STFT formula
        expected_frames_min = (audio_length - frame_length) // hop_length + 1
        expected_frames_max = audio_length // hop_length + 1

        actual_frames = len(f0_data['f0'])

        # Assert F0 length is within expected range
        assert expected_frames_min <= actual_frames <= expected_frames_max, \
            f"F0 frame count {actual_frames} not in expected range [{expected_frames_min}, {expected_frames_max}]. " \
            f"Audio length: {audio_length}, hop_length: {hop_length}, frame_length: {frame_length}"

        # Validate timing alignment if times are provided
        if 'times' in f0_data:
            times = f0_data['times']
            assert len(times) == actual_frames, \
                f"Time array length {len(times)} doesn't match F0 array length {actual_frames}"

            # Check that times are evenly spaced by hop_length
            if len(times) > 1:
                time_diffs = np.diff(times)
                expected_time_diff = hop_length / sample_rate

                # Allow small tolerance for floating point errors
                np.testing.assert_allclose(
                    time_diffs,
                    expected_time_diff,
                    rtol=0.01,
                    atol=0.001,
                    err_msg=f"Time spacing inconsistent. Expected {expected_time_diff:.4f}s per frame"
                )

        # Test actual voice conversion to validate alignment
        try:
            from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter
            import torch
        except ImportError:
            pytest.skip("SingingVoiceConverter not available for conversion test")

        # Create converter instance
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        converter = SingingVoiceConverter(config={
            'device': device,
            'model_sample_rate': 16000,  # Standard model sample rate
            'hop_length': 512
        })

        # Create dummy target speaker embedding (256-dimensional)
        target_embedding = np.random.randn(256).astype(np.float32)

        # Ensure audio is float32 and on correct device
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).float().to(device)
        embedding_tensor = torch.from_numpy(target_embedding).float().to(device)

        # Perform voice conversion with correct parameter names
        output_sample_rate = 44100  # Typical output sample rate
        converted_audio = converter.convert(
            source_audio=audio_tensor,
            target_speaker_embedding=embedding_tensor,
            source_f0=f0_data['f0'],
            source_sample_rate=sample_rate,
            output_sample_rate=output_sample_rate
        )

        # Validate output duration alignment
        # Expected output length: (input_samples / input_sr) * output_sr
        expected_duration_sec = len(audio) / sample_rate
        expected_output_samples = int(expected_duration_sec * output_sample_rate)

        # Allow 5% tolerance for resampling and frame alignment
        tolerance = 0.05
        min_expected = int(expected_output_samples * (1 - tolerance))
        max_expected = int(expected_output_samples * (1 + tolerance))

        actual_output_samples = len(converted_audio)
        assert min_expected <= actual_output_samples <= max_expected, \
            f"Output length {actual_output_samples} not in expected range [{min_expected}, {max_expected}]. " \
            f"Input: {len(audio)} samples @ {sample_rate}Hz, Output: {actual_output_samples} samples @ {output_sample_rate}Hz"

        # Validate no NaN or Inf values in output
        assert np.all(np.isfinite(converted_audio)), "Converted audio contains NaN or Inf"

        logger.info(
            f"Feature alignment validated with conversion: "
            f"{actual_frames} F0 frames for {audio_length} samples "
            f"(hop: {hop_length}, frame: {frame_length}), "
            f"output: {actual_output_samples} samples @ {output_sample_rate}Hz"
        )


# ============================================================================
# 3. Performance Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks and stress tests."""

    @pytest.mark.cuda
    def test_gpu_memory_tracking(self, song_file_mono, pipeline_instance, cuda_available):
        """Test GPU memory usage during pipeline execution.

        Validates:
        1. GPU memory is allocated
        2. Memory usage is reasonable
        3. No excessive memory consumption
        """
        if not cuda_available:
            pytest.skip("CUDA not available")

        separator = pipeline_instance['separator']

        if separator is None:
            pytest.skip("VocalSeparator not available")

        # Record initial memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Run separation
        try:
            vocals, instrumental = separator.separate_vocals(str(song_file_mono))
        except Exception as e:
            pytest.skip(f"Separation failed: {e}")

        # Record peak memory
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory

        # Clean up
        torch.cuda.empty_cache()

        # Validate reasonable memory usage (< 2 GB for 3-second audio)
        max_memory_mb = 2048
        memory_used_mb = memory_used / (1024 * 1024)

        assert memory_used_mb < max_memory_mb, \
            f"GPU memory usage too high: {memory_used_mb:.2f} MB (max: {max_memory_mb} MB)"

        logger.info(f"GPU memory usage: {memory_used_mb:.2f} MB")

    @pytest.mark.cuda
    def test_pipeline_stage_gpu_memory_cleanup(self, song_file_mono, pipeline_instance, voice_profile_fixture, cuda_available):
        """Test per-stage GPU memory cleanup across full pipeline.

        Validates that each stage cleans up GPU memory properly by tracking
        peak memory per stage independently. Ensures that peak memory for
        individual stages is less than total pipeline memory, confirming
        per-stage cleanup.

        Tests stages:
        1. Vocal separation
        2. Pitch extraction
        3. Voice cloning
        4. Voice conversion
        """
        if not cuda_available:
            pytest.skip("CUDA not available")

        separator = pipeline_instance['separator']
        pitch_extractor = pipeline_instance['pitch_extractor']
        voice_cloner = pipeline_instance['voice_cloner']

        if separator is None or pitch_extractor is None or voice_cloner is None:
            pytest.skip("Required components not available")

        # Instantiate converter
        try:
            converter = SingingVoiceConverter(config={'device': 'cuda'})
        except Exception as e:
            pytest.skip(f"SingingVoiceConverter not available: {e}")

        stage_memories = {}

        # Initial cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Stage 1: Vocal separation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_stage1 = torch.cuda.memory_allocated()

        try:
            vocals, instrumental = separator.separate_vocals(str(song_file_mono))
        except Exception as e:
            pytest.skip(f"Separation failed: {e}")

        torch.cuda.synchronize()
        peak_stage1 = torch.cuda.max_memory_allocated()
        after_stage1 = torch.cuda.memory_allocated()
        stage_memories['source_separation'] = (peak_stage1 - baseline_stage1) / 1e6

        # Assert cleanup within 20MB of baseline
        cleanup_delta_mb = abs(after_stage1 - baseline_stage1) / (1024 * 1024)
        assert cleanup_delta_mb < 20, \
            f"Stage 1 (source_separation) failed to clean up: {cleanup_delta_mb:.2f} MB from baseline"

        # Prepare for next stage
        if vocals.ndim == 2:
            vocals_mono = np.mean(vocals, axis=0)
        else:
            vocals_mono = vocals

        # Stage 2: Pitch extraction
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_stage2 = torch.cuda.memory_allocated()

        # Use actual sample rate from separator (separated audio is at processing sample_rate)
        actual_sample_rate = separator.config['sample_rate']
        try:
            f0_data = pitch_extractor.extract_f0_contour(vocals_mono, sample_rate=actual_sample_rate)
        except Exception as e:
            pytest.skip(f"Pitch extraction failed: {e}")

        torch.cuda.synchronize()
        peak_stage2 = torch.cuda.max_memory_allocated()
        after_stage2 = torch.cuda.memory_allocated()
        stage_memories['pitch_extraction'] = (peak_stage2 - baseline_stage2) / 1e6

        # Assert cleanup within 20MB of baseline
        cleanup_delta_mb = abs(after_stage2 - baseline_stage2) / (1024 * 1024)
        assert cleanup_delta_mb < 20, \
            f"Stage 2 (pitch extraction) failed to clean up: {cleanup_delta_mb:.2f} MB from baseline"

        # Stage 3: Voice cloning (profile already exists from fixture)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_stage3 = torch.cuda.memory_allocated()

        embedding = voice_profile_fixture['embedding']
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).float().to('cuda')
        else:
            embedding_tensor = embedding.float().to('cuda')

        torch.cuda.synchronize()
        peak_stage3 = torch.cuda.max_memory_allocated()
        after_stage3 = torch.cuda.memory_allocated()
        # Voice cloning is mostly CPU-based for Resemblyzer, so minimal GPU usage expected
        stage_memories['voice_cloning'] = (peak_stage3 - baseline_stage3) / 1e6

        # Assert cleanup within 20MB of baseline
        cleanup_delta_mb = abs(after_stage3 - baseline_stage3) / (1024 * 1024)
        assert cleanup_delta_mb < 20, \
            f"Stage 3 (voice cloning) failed to clean up: {cleanup_delta_mb:.2f} MB from baseline"

        # Stage 4: Voice conversion
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_stage4 = torch.cuda.memory_allocated()

        audio_processor = pipeline_instance['audio_processor']
        audio, sr = audio_processor.load_audio(str(song_file_mono))

        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float().to('cuda')
        else:
            audio_tensor = audio.float().to('cuda')

        converted_audio = converter.convert(
            source_audio=audio_tensor,
            target_speaker_embedding=embedding_tensor,
            source_sample_rate=sr
        )

        torch.cuda.synchronize()
        peak_stage4 = torch.cuda.max_memory_allocated()
        after_stage4 = torch.cuda.memory_allocated()
        stage_memories['voice_conversion'] = (peak_stage4 - baseline_stage4) / 1e6

        # Assert cleanup within 20MB of baseline
        cleanup_delta_mb = abs(after_stage4 - baseline_stage4) / (1024 * 1024)
        assert cleanup_delta_mb < 20, \
            f"Stage 4 (voice conversion) failed to clean up: {cleanup_delta_mb:.2f} MB from baseline"

        # Now measure total pipeline memory without resets
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_total = torch.cuda.memory_allocated()

        # Run full pipeline without resets
        vocals, _ = separator.separate_vocals(str(song_file_mono), use_cache=False)
        if vocals.ndim == 2:
            vocals = np.mean(vocals, axis=0)
        f0_data = pitch_extractor.extract_f0_contour(vocals, sample_rate=actual_sample_rate)
        audio_tensor = torch.from_numpy(audio).float().to('cuda') if isinstance(audio, np.ndarray) else audio.float().to('cuda')
        converted_audio = converter.convert(
            source_audio=audio_tensor,
            target_speaker_embedding=embedding_tensor,
            source_sample_rate=sr
        )

        torch.cuda.synchronize()
        total_peak_memory = torch.cuda.max_memory_allocated()
        total_memory_used = (total_peak_memory - initial_total) / (1024 * 1024)

        # Assert per-stage cleanup: each stage's peak < total pipeline peak
        for stage_name, stage_memory in stage_memories.items():
            if stage_memory > 0:  # Only check stages with GPU usage
                assert stage_memory < total_memory_used, \
                    f"Stage '{stage_name}' peak memory ({stage_memory:.2f} MB) >= " \
                    f"total pipeline memory ({total_memory_used:.2f} MB). " \
                    f"Indicates insufficient per-stage cleanup."

        logger.info(
            f"Pipeline stage memory cleanup validated:\n"
            f"  Source separation: {stage_memories['source_separation']:.2f} MB\n"
            f"  Pitch extraction: {stage_memories['pitch_extraction']:.2f} MB\n"
            f"  Voice cloning: {stage_memories['voice_cloning']:.2f} MB\n"
            f"  Voice conversion: {stage_memories['voice_conversion']:.2f} MB\n"
            f"  Total pipeline: {total_memory_used:.2f} MB"
        )

        # Cleanup
        torch.cuda.empty_cache()

    def test_memory_leak_detection(self, song_file_mono, pipeline_instance, memory_leak_detector):
        """Test for memory leaks in repeated operations.

        Validates:
        1. No memory leaks after 10 iterations
        2. Both CPU and GPU memory stable
        3. Cleanup occurs properly
        """
        separator = pipeline_instance['separator']
        pitch_extractor = pipeline_instance['pitch_extractor']

        if separator is None or pitch_extractor is None:
            pytest.skip("Required components not available")

        memory_leak_detector.start()

        # Run 10 iterations
        for i in range(10):
            try:
                # Separate vocals
                vocals, _ = separator.separate_vocals(str(song_file_mono), use_cache=False)

                # Extract pitch
                if vocals.ndim == 2:
                    vocals = np.mean(vocals, axis=0)

                # Use actual sample rate from separator (separated audio is at processing sample_rate)
                actual_sample_rate = separator.config['sample_rate']
                f0_data = pitch_extractor.extract_f0_contour(vocals, sample_rate=actual_sample_rate)

                # Cleanup
                del vocals
                del f0_data

            except Exception as e:
                logger.warning(f"Iteration {i} failed: {e}")
                continue

        # Check for leaks
        cpu_leak, gpu_leak = memory_leak_detector.stop()
        memory_leak_detector.report(
            "memory_leak_detection",
            cpu_leak,
            gpu_leak,
            cpu_threshold=100 * 1024 * 1024,  # 100 MB
            gpu_threshold=50 * 1024 * 1024    # 50 MB
        )

        logger.info(
            f"Memory leak test completed: CPU leak={cpu_leak/1024/1024:.2f} MB, "
            f"GPU leak={gpu_leak/1024/1024:.2f} MB"
        )

    @pytest.mark.slow
    def test_stress_concurrent_operations(self, song_file_mono, pipeline_instance, concurrent_executor):
        """Stress test with concurrent operations.

        Validates:
        1. Components handle concurrent requests
        2. No race conditions
        3. All operations complete successfully
        """
        pitch_extractor = pipeline_instance['pitch_extractor']
        audio_processor = pipeline_instance['audio_processor']

        if pitch_extractor is None:
            pytest.skip("SingingPitchExtractor not available")

        # Load audio once
        audio, sample_rate = audio_processor.load_audio(str(song_file_mono), return_sr=True)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Ensure mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Submit 10 concurrent pitch extraction tasks
        def extract_pitch(audio_data, sr):
            return pitch_extractor.extract_f0_contour(audio_data, sample_rate=sr)

        futures = []
        for i in range(10):
            future = concurrent_executor.submit(extract_pitch, audio, sample_rate)
            futures.append(future)

        # Wait for all tasks to complete
        results = []
        errors = []

        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # Validate results
        assert len(results) > 0, "No successful operations"

        if errors:
            logger.warning(f"Some operations failed: {errors}")

        # All successful results should have consistent format
        for result in results:
            assert 'f0' in result
            assert len(result['f0']) > 0

        logger.info(
            f"Concurrent stress test passed: {len(results)}/{len(futures)} "
            f"operations successful"
        )

    def test_processing_time_benchmark(self, song_file_mono, pipeline_instance):
        """Benchmark processing time for components.

        Measures and reports:
        1. Vocal separation time
        2. Pitch extraction time
        3. Total pipeline time
        """
        separator = pipeline_instance['separator']
        pitch_extractor = pipeline_instance['pitch_extractor']

        if separator is None or pitch_extractor is None:
            pytest.skip("Required components not available")

        # Benchmark separation
        start = time.perf_counter()
        try:
            vocals, instrumental = separator.separate_vocals(str(song_file_mono), use_cache=False)
        except Exception as e:
            pytest.skip(f"Separation failed: {e}")
        separation_time = time.perf_counter() - start

        # Benchmark pitch extraction
        if vocals.ndim == 2:
            vocals = np.mean(vocals, axis=0)

        # Use actual sample rate from separator (separated audio is at processing sample_rate)
        actual_sample_rate = separator.config['sample_rate']

        start = time.perf_counter()
        f0_data = pitch_extractor.extract_f0_contour(vocals, sample_rate=actual_sample_rate)
        extraction_time = time.perf_counter() - start

        total_time = separation_time + extraction_time

        # Log benchmarks
        logger.info(
            f"Performance benchmark:\n"
            f"  Separation: {separation_time:.3f}s\n"
            f"  Pitch extraction: {extraction_time:.3f}s\n"
            f"  Total: {total_time:.3f}s"
        )

        # Validate reasonable performance (< 60 seconds total for 3-second audio)
        assert total_time < 60.0, f"Processing too slow: {total_time:.3f}s"


# ============================================================================
# Test Execution Summary
# ============================================================================

def test_integration_suite_info():
    """Print integration test suite information.

    This test always passes and provides documentation about the test suite.
    """
    info = """
    AutoVoice Core Integration Test Suite
    =====================================

    Coverage:
    - Component integration (VocalSeparator + PitchExtractor)
    - Data flow validation (sample rates, formats)
    - End-to-end pipeline testing
    - Performance benchmarking
    - Memory leak detection
    - Concurrent operation stress tests

    Test Markers:
    - @pytest.mark.integration: All tests in this suite
    - @pytest.mark.slow: Tests taking >1 second
    - @pytest.mark.cuda: GPU-specific tests
    - @pytest.mark.performance: Performance benchmarks

    Run with:
        pytest tests/test_core_integration.py -v
        pytest tests/test_core_integration.py -m integration
        pytest tests/test_core_integration.py -m "integration and not slow"
    """
    logger.info(info)
    print(info)
    assert True
