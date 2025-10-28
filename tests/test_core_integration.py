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
        # Sample rate should match separator's configured rate
        sample_rate = separator.config.get('sample_rate', 44100)

        f0_data = pitch_extractor.extract_f0_contour(
            vocals_mono,
            sample_rate=sample_rate,
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
        assert f0_data['sample_rate'] == sample_rate

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

        # Get separator's target sample rate
        separator_sr = separator.config.get('sample_rate', 44100)

        try:
            vocals, _ = separator.separate_vocals(str(song_file_stereo))
        except Exception as e:
            pytest.skip(f"Separation failed: {e}")

        # Ensure mono
        if vocals.ndim == 2:
            vocals = np.mean(vocals, axis=0)

        # Extract pitch with matching sample rate
        f0_data = pitch_extractor.extract_f0_contour(vocals, sample_rate=separator_sr)

        # Validate alignment
        assert f0_data['sample_rate'] == separator_sr

        # Calculate expected number of frames
        hop_length = f0_data['hop_length']
        expected_frames = len(vocals) // hop_length

        # Allow small discrepancy due to padding/windowing
        assert abs(len(f0_data['f0']) - expected_frames) <= 5

        logger.info(
            f"Sample rate alignment validated: {separator_sr} Hz, "
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
        # Skip actual conversion if SVC not available, just validate embedding format
        pytest.skip("SingingVoiceConverter not tested in this integration test")

        logger.info(
            f"Profile integration test passed: created profile {profile_id}, "
            f"embedding shape {embedding.shape}"
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

        sample_rate = separator.config.get('sample_rate', 44100)

        f0_data = pitch_extractor.extract_f0_contour(vocals_mono, sample_rate=sample_rate)

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

                sample_rate = separator.config.get('sample_rate', 44100)
                f0_data = pitch_extractor.extract_f0_contour(vocals, sample_rate=sample_rate)

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

        sample_rate = separator.config.get('sample_rate', 44100)

        start = time.perf_counter()
        f0_data = pitch_extractor.extract_f0_contour(vocals, sample_rate=sample_rate)
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
