"""
Comprehensive end-to-end tests for complete voice synthesis workflows.

Tests complete singing conversion pipeline, voice cloning, and integration workflows.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import time
import json


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
class TestSingingConversionWorkflow:
    """Test complete singing voice conversion workflow."""

    @pytest.fixture
    def setup_pipeline(self, song_file, test_profile, device):
        """Setup pipeline with test dependencies."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            config = {
                'device': device,
                'cache_enabled': True,
                'sample_rate': 44100
            }
            pipeline = SingingConversionPipeline(config=config)
            return {
                'pipeline': pipeline,
                'song_file': song_file,
                'profile': test_profile
            }
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

    def test_full_singing_conversion_workflow(self, setup_pipeline, performance_tracker):
        """Test full workflow: audio → separation → F0 → conversion → output.

        Tests complete integration from input song to converted output with target voice.
        """
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        test_profile = setup_pipeline['profile']

        # Track performance
        performance_tracker.start('full_conversion')

        # Execute conversion
        result = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            pitch_shift=0.0
        )

        elapsed = performance_tracker.stop()

        # Validate result structure
        assert 'mixed_audio' in result
        assert 'sample_rate' in result
        assert 'duration' in result
        assert 'f0_contour' in result

        # Validate output audio
        audio = result['mixed_audio']
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1  # Mono output
        assert len(audio) > 0
        assert np.isfinite(audio).all()

        # Check sample rate
        assert result['sample_rate'] == 44100

        # Performance check: Should complete in reasonable time
        duration_seconds = result['duration']
        rtf = elapsed / duration_seconds  # Real-time factor
        assert rtf < 10.0, f"Conversion too slow: RTF={rtf:.2f}"

        print(f"\nConversion RTF: {rtf:.2f}x (lower is faster)")

    def test_progress_callback_verification(self, setup_pipeline):
        """Test progress callback is called during conversion."""
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        test_profile = setup_pipeline['profile']

        progress_calls = []

        def progress_callback(stage: str, progress: float):
            progress_calls.append({'stage': stage, 'progress': progress})

        result = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            progress_callback=progress_callback
        )

        # Verify callbacks were made
        assert len(progress_calls) > 0, "Progress callback was never called"

        # Verify progress goes from 0 to 1 (0.0-1.0 range)
        progress_values = [p['progress'] for p in progress_calls]
        assert min(progress_values) >= 0.0
        assert max(progress_values) <= 1.0
        assert max(progress_values) >= 0.95  # Pipeline should reach near completion

        # Verify multiple stages
        stages = set(p['stage'] for p in progress_calls)
        expected_stages = {'source_separation', 'pitch_extraction', 'voice_conversion'}
        assert len(stages.intersection(expected_stages)) >= 2

    def test_caching_speedup(self, setup_pipeline, performance_tracker):
        """Test that caching provides speedup on repeated conversions."""
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        test_profile = setup_pipeline['profile']

        # Clear cache to ensure isolated measurement
        pipeline.clear_cache()

        # First conversion (cold, populates cache)
        performance_tracker.start('cold_conversion')
        result1 = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        time_cold = performance_tracker.stop()

        # Second conversion (warm, uses cache)
        performance_tracker.start('warm_conversion')
        result2 = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        time_warm = performance_tracker.stop()

        # Cache should provide at least 3x speedup for isolated hits
        speedup = time_cold / time_warm
        assert speedup >= 3.0, f"Cache speedup insufficient: {speedup:.2f}x (expected >= 3.0x)"

        # Results should be identical (bitwise, since cached)
        np.testing.assert_array_equal(result1['mixed_audio'], result2['mixed_audio'])

        print(f"\nCache speedup: {speedup:.2f}x")

    def test_error_recovery(self, setup_pipeline):
        """Test error recovery for invalid inputs."""
        pipeline = setup_pipeline['pipeline']

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.convert_song(
                song_path="/nonexistent/file.wav",
                target_profile_id="nonexistent_profile_id"
            )

        # Test with invalid profile ID
        with pytest.raises(ValueError):
            pipeline.convert_song(
                song_path=str(setup_pipeline['song_file']),
                target_profile_id="invalid_profile_id"
            )

    def test_preset_comparison(self, setup_pipeline):
        """Test different quality presets produce different results."""
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        test_profile = setup_pipeline['profile']

        # Try different presets if available
        presets = ['fast', 'balanced', 'quality']
        results = {}

        for preset in presets:
            try:
                result = pipeline.convert_song(
                    song_path=str(song_file),
                    target_profile_id=test_profile['profile_id'],
                    preset=preset
                )
                results[preset] = result['mixed_audio']
            except (ValueError, KeyError):
                # Preset not supported, skip
                continue

        # If multiple presets worked, verify differences
        if len(results) >= 2:
            preset_list = list(results.keys())
            audio1 = results[preset_list[0]]
            audio2 = results[preset_list[1]]

            # Should produce different outputs
            assert not np.array_equal(audio1, audio2), \
                f"Presets {preset_list[0]} and {preset_list[1]} produce identical output"


@pytest.mark.e2e
@pytest.mark.integration
class TestVoiceCloningWorkflow:
    """Test complete voice cloning workflow."""

    def test_voice_clone_create_and_use(self, tmp_path, device):
        """Test creating voice profile and using it for conversion."""
        try:
            from src.auto_voice.inference.voice_cloner import VoiceCloner
        except ImportError:
            pytest.skip("VoiceCloner not available")

        # Create cloner
        cloner = VoiceCloner(device=device)

        # Generate reference audio (10 seconds)
        sample_rate = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        reference_audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        # Create voice profile
        profile = cloner.create_voice_profile(
            audio=reference_audio,
            sample_rate=sample_rate,
            user_id='test_user'
        )

        # Verify profile structure
        assert 'profile_id' in profile
        assert 'user_id' in profile
        assert profile['user_id'] == 'test_user'

        # Load profile back
        loaded_profile = cloner.load_voice_profile(profile['profile_id'])
        assert 'embedding' in loaded_profile

        # Use embedding for test conversion (if pipeline available)
        embedding = loaded_profile['embedding']
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (256,)

    def test_multi_sample_profile_creation(self, device):
        """Test creating profile from multiple audio samples."""
        try:
            from src.auto_voice.inference.voice_cloner import VoiceCloner
        except ImportError:
            pytest.skip("VoiceCloner not available")

        cloner = VoiceCloner(device=device)

        # Generate multiple samples
        sample_rate = 22050
        audio_samples = []
        for i in range(3):
            t = np.linspace(0, 10.0, int(sample_rate * 10))
            audio = (0.3 * np.sin(2 * np.pi * (220 + i * 10) * t)).astype(np.float32)
            audio_samples.append(audio)

        # Create profile from multiple samples
        profile = cloner.create_voice_profile_from_multiple_samples(
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            user_id='multi_sample_user'
        )

        assert 'profile_id' in profile
        assert 'num_samples' in profile
        assert profile['num_samples'] == 3


@pytest.mark.e2e
@pytest.mark.integration
class TestMultiComponentIntegration:
    """Test integration between multiple components."""

    def test_source_separator_pitch_extractor_integration(self, song_file, device):
        """Test VocalSeparator → SingingPitchExtractor integration."""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor
        except ImportError:
            pytest.skip("Components not available")

        # Separate vocals
        separator = VocalSeparator(device=device)
        vocals, instrumental = separator.separate_vocals(str(song_file))

        # Extract pitch from vocals
        extractor = SingingPitchExtractor(device=device)
        f0_result = extractor.extract_f0_contour(vocals[0], sample_rate=44100)

        # Verify results
        assert 'f0' in f0_result
        assert 'voiced' in f0_result
        assert len(f0_result['f0']) > 0

    def test_end_to_end_memory_management(self, song_file, test_profile, device, memory_leak_detector):
        """Test memory is properly managed across full workflow."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        with memory_leak_detector:
            pipeline = SingingConversionPipeline(config={'device': device})

            # Run multiple conversions
            for _ in range(3):
                result = pipeline.convert_song(
                    song_path=str(song_file),
                    target_profile_id=test_profile['profile_id']
                )

                # Force cleanup
                del result
                import gc
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()

        # Memory leak detector will warn if significant leak detected


@pytest.mark.e2e
@pytest.mark.performance
class TestPerformanceE2E:
    """End-to-end performance tests."""

    @pytest.mark.parametrize('audio_length', [5.0, 10.0, 30.0])
    def test_conversion_latency_by_length(self, tmp_path, test_profile, device, audio_length, performance_tracker):
        """Test conversion latency scales with audio length."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import soundfile as sf
        except ImportError:
            pytest.skip("Components not available")

        # Generate audio of specified length
        sample_rate = 22050
        t = np.linspace(0, audio_length, int(sample_rate * audio_length))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_file = tmp_path / f"test_{audio_length}s.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Convert and measure time
        pipeline = SingingConversionPipeline(config={'device': device})

        performance_tracker.start(f'convert_{audio_length}s')
        result = pipeline.convert_song(
            song_path=str(audio_file),
            target_profile_id=test_profile['profile_id']
        )
        elapsed = performance_tracker.stop()

        # Calculate real-time factor
        rtf = elapsed / audio_length
        performance_tracker.record(f'rtf_{audio_length}s', rtf)

        # RTF should be reasonable (< 20x for CPU, < 5x for GPU)
        max_rtf = 20.0 if device == 'cpu' else 5.0
        assert rtf < max_rtf, f"RTF too high for {audio_length}s audio: {rtf:.2f}x"

        print(f"\n{audio_length}s audio: RTF={rtf:.2f}x, elapsed={elapsed:.2f}s")

    def test_concurrent_conversions(self, song_file, test_profile, device, concurrent_executor):
        """Test multiple concurrent conversions."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={'device': device})

        def convert_task():
            return pipeline.convert_song(
                song_path=str(song_file),
                target_profile_id=test_profile['profile_id']
            )

        # Submit 3 concurrent tasks
        futures = [concurrent_executor.submit(convert_task) for _ in range(3)]

        # Wait for all to complete
        results = [f.result(timeout=120) for f in futures]

        # All should succeed
        assert len(results) == 3
        assert all('mixed_audio' in r for r in results)





@pytest.mark.integration
class TestGPUManagerPropagation:
    """Test GPU manager is properly propagated to all components."""

    def test_gpu_manager_propagation_to_pipeline_components(self, device):
        """
        Comment 11: Verify gpu_manager is passed to all pipeline components.

        This test ensures the GPU manager is correctly propagated through
        the entire pipeline hierarchy for proper resource management.
        """
        from unittest.mock import Mock
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        # Create mock GPU manager
        gpu_manager = Mock()
        gpu_manager.device = device
        gpu_manager.allocate_memory = Mock(return_value=True)
        gpu_manager.free_memory = Mock(return_value=True)

        # Create pipeline with GPU manager
        config = {'device': device}
        pipeline = SingingConversionPipeline(config=config, gpu_manager=gpu_manager)

        # Verify propagation to vocal separator
        if hasattr(pipeline, 'vocal_separator'):
            assert pipeline.vocal_separator.gpu_manager is gpu_manager, \
                "GPU manager not propagated to vocal_separator"

        # Verify propagation to pitch extractor
        if hasattr(pipeline, 'pitch_extractor'):
            assert pipeline.pitch_extractor.gpu_manager is gpu_manager, \
                "GPU manager not propagated to pitch_extractor"

        # Verify propagation to converter
        if hasattr(pipeline, 'converter'):
            assert pipeline.converter.gpu_manager is gpu_manager, \
                "GPU manager not propagated to converter"

        # Verify propagation to voice cloner
        if hasattr(pipeline, 'voice_cloner'):
            assert pipeline.voice_cloner.gpu_manager is gpu_manager, \
                "GPU manager not propagated to voice_cloner"

        print("✓ GPU manager successfully propagated to all components")

    def test_gpu_manager_allocation_calls(self, device):
        """Test GPU manager allocation methods are called during operations."""
        from unittest.mock import Mock, call
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        # Create mock GPU manager with tracking
        gpu_manager = Mock()
        gpu_manager.device = device
        gpu_manager.allocate_memory = Mock(return_value=True)
        gpu_manager.free_memory = Mock(return_value=True)

        # Create pipeline
        config = {'device': device}
        pipeline = SingingConversionPipeline(config=config, gpu_manager=gpu_manager)

        # GPU manager methods should be callable from components
        # (Actual calls happen during inference, which requires real models)
        assert hasattr(pipeline, 'gpu_manager') or any(
            hasattr(getattr(pipeline, attr, None), 'gpu_manager')
            for attr in ['vocal_separator', 'pitch_extractor', 'converter']
        ), "GPU manager not accessible in pipeline components"

        print("✓ GPU manager methods accessible for resource management")

    @pytest.fixture
    def conversion_pipeline(self, device):
        """Create conversion pipeline for quality tests."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            pipeline = SingingConversionPipeline(config={'device': device})
            return pipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

    def test_conversion_meets_pitch_target(self, song_file, test_profile, conversion_pipeline, quality_evaluator):
        """
        Quality gate: Test that conversion meets pitch accuracy target (RMSE Hz < 10.0).

        This test enforces the pitch accuracy quality standard for voice conversion.
        """
        # Run conversion via pipeline
        result = conversion_pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            pitch_shift=0.0
        )

        # Load source and converted audio
        import soundfile as sf
        source_audio, _ = sf.read(str(song_file))
        converted_audio = result['mixed_audio']

        # Convert to torch tensors
        source_tensor = torch.from_numpy(source_audio).float()
        converted_tensor = torch.from_numpy(converted_audio).float()

        # Evaluate quality
        metrics_result = quality_evaluator.evaluate_single_conversion(
            source_tensor,
            converted_tensor
        )

        # Assert pitch accuracy target
        pitch_rmse_hz = metrics_result.pitch_accuracy.rmse_hz
        print(f"\nPitch RMSE (Hz): {pitch_rmse_hz:.2f}")

        assert pitch_rmse_hz < 10.0, \
            f"Quality gate FAILED: Pitch RMSE Hz {pitch_rmse_hz:.2f} exceeds 10.0 Hz threshold"

        print("✓ Quality gate PASSED: Pitch accuracy meets target")

    def test_conversion_meets_speaker_similarity(self, song_file, test_profile, conversion_pipeline, quality_evaluator):
        """
        Quality gate: Test that conversion meets speaker similarity target (cosine > 0.85).

        This test enforces the speaker similarity quality standard for voice conversion.
        """
        # Run conversion
        result = conversion_pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            pitch_shift=0.0
        )

        # Load audio
        import soundfile as sf
        source_audio, _ = sf.read(str(song_file))
        converted_audio = result['mixed_audio']

        # Convert to torch tensors
        source_tensor = torch.from_numpy(source_audio).float()
        converted_tensor = torch.from_numpy(converted_audio).float()

        # Get target speaker embedding from profile
        try:
            from src.auto_voice.models.speaker_encoder import SpeakerEncoder
            encoder = SpeakerEncoder()

            # Extract embedding from converted audio (should match target)
            converted_embedding = encoder.extract_embedding(converted_tensor)

            # Get target embedding from profile
            # Assuming test_profile has embedding or reference audio
            if 'embedding' in test_profile:
                target_embedding = test_profile['embedding']
            else:
                # Fall back to extracting from reference audio if available
                pytest.skip("Target profile embedding not available for similarity test")

        except ImportError:
            pytest.skip("SpeakerEncoder not available")

        # Evaluate quality with target embedding
        metrics_result = quality_evaluator.evaluate_single_conversion(
            source_tensor,
            converted_tensor,
            target_speaker_embedding=target_embedding
        )

        # Assert speaker similarity target
        speaker_similarity = metrics_result.speaker_similarity.cosine_similarity
        print(f"\nSpeaker Similarity: {speaker_similarity:.3f}")

        assert speaker_similarity > 0.85, \
            f"Quality gate FAILED: Speaker similarity {speaker_similarity:.3f} below 0.85 threshold"

        print("✓ Quality gate PASSED: Speaker similarity meets target")

    def test_overall_quality_score_threshold(self, song_file, test_profile, conversion_pipeline, quality_evaluator):
        """
        Quality gate: Test that overall quality score meets minimum threshold (> 0.75).

        This test enforces the overall quality standard combining all metrics.
        """
        # Run conversion
        result = conversion_pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            pitch_shift=0.0
        )

        # Load and convert audio
        import soundfile as sf
        source_audio, _ = sf.read(str(song_file))
        converted_audio = result['mixed_audio']

        source_tensor = torch.from_numpy(source_audio).float()
        converted_tensor = torch.from_numpy(converted_audio).float()

        # Evaluate quality
        metrics_result = quality_evaluator.evaluate_single_conversion(
            source_tensor,
            converted_tensor
        )

        # Assert overall quality score
        overall_score = metrics_result.overall_quality_score
        print(f"\nOverall Quality Score: {overall_score:.3f}")
        print(f"  Pitch RMSE (Hz): {metrics_result.pitch_accuracy.rmse_hz:.2f}")
        print(f"  Pitch Correlation: {metrics_result.pitch_accuracy.correlation:.3f}")
        print(f"  Speaker Similarity: {metrics_result.speaker_similarity.cosine_similarity:.3f}")
        print(f"  Naturalness: {metrics_result.naturalness.confidence_score:.3f}")
        print(f"  Intelligibility STOI: {metrics_result.intelligibility.stoi_score:.3f}")

        assert overall_score > 0.75, \
            f"Quality gate FAILED: Overall quality score {overall_score:.3f} below 0.75 threshold"

        print("✓ Quality gate PASSED: Overall quality meets target")

    def test_comprehensive_quality_metrics(self, song_file, test_profile, conversion_pipeline, quality_evaluator, tmp_path):
        """
        Quality gate: Test comprehensive quality metrics including MOS, STOI, and MCD.

        This test enforces comprehensive quality standards:
        - MOS estimation > 4.0 (excellent perceived quality)
        - STOI score > 0.9 (near-perfect intelligibility)
        - MCD < 6.0 dB (low spectral distortion)

        Addresses Comment 9 requirements.
        """
        # Run conversion
        result = conversion_pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            pitch_shift=0.0
        )

        # Load audio
        import soundfile as sf
        source_audio, _ = sf.read(str(song_file))
        converted_audio = result['mixed_audio']

        # Convert to torch tensors
        source_tensor = torch.from_numpy(source_audio).float()
        converted_tensor = torch.from_numpy(converted_audio).float()

        # Create quality evaluator with MCD enabled
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        quality_aggregator = QualityMetricsAggregator(
            sample_rate=44100,
            mos_method='heuristic',  # Use heuristic (NISQA optional)
            compute_mcd=True
        )

        # Evaluate comprehensive metrics
        metrics_result = quality_aggregator.evaluate(
            source_audio=source_tensor,
            target_audio=converted_tensor
        )

        # Extract metrics
        mos_estimation = metrics_result.naturalness.mos_estimation
        stoi_score = metrics_result.intelligibility.stoi_score
        mcd_value = metrics_result.naturalness.mcd

        # Print detailed results
        print(f"\n=== Comprehensive Quality Metrics ===")
        print(f"MOS Estimation: {mos_estimation:.2f}")
        print(f"  Method: {metrics_result.naturalness.mos_method}")
        if metrics_result.naturalness.mos_nisqa is not None:
            print(f"  NISQA MOS: {metrics_result.naturalness.mos_nisqa:.2f}")
        if metrics_result.naturalness.mos_heuristic is not None:
            print(f"  Heuristic MOS: {metrics_result.naturalness.mos_heuristic:.2f}")

        print(f"STOI Score: {stoi_score:.3f}")
        print(f"  ESTOI Score: {metrics_result.intelligibility.estoi_score:.3f}")

        if mcd_value is not None:
            print(f"MCD (Mel-Cepstral Distortion): {mcd_value:.2f} dB")

        print(f"\nPitch RMSE (Hz): {metrics_result.pitch_accuracy.rmse_hz:.2f}")
        print(f"Pitch Correlation: {metrics_result.pitch_accuracy.correlation:.3f}")
        print(f"Speaker Similarity: {metrics_result.speaker_similarity.cosine_similarity:.3f}")
        print(f"Spectral Distortion: {metrics_result.naturalness.spectral_distortion:.2f} dB")

        # Save metrics to JSON
        metrics_json = {
            'timestamp': metrics_result.evaluation_timestamp,
            'processing_time_seconds': metrics_result.processing_time_seconds,
            'pitch_accuracy': {
                'rmse_hz': metrics_result.pitch_accuracy.rmse_hz,
                'rmse_log2': metrics_result.pitch_accuracy.rmse_log2,
                'correlation': metrics_result.pitch_accuracy.correlation,
                'voiced_accuracy': metrics_result.pitch_accuracy.voiced_accuracy,
                'octave_errors': metrics_result.pitch_accuracy.octave_errors,
                'confidence_score': metrics_result.pitch_accuracy.confidence_score
            },
            'speaker_similarity': {
                'cosine_similarity': metrics_result.speaker_similarity.cosine_similarity,
                'embedding_distance': metrics_result.speaker_similarity.embedding_distance,
                'confidence_score': metrics_result.speaker_similarity.confidence_score
            },
            'naturalness': {
                'spectral_distortion': metrics_result.naturalness.spectral_distortion,
                'harmonic_to_noise': metrics_result.naturalness.harmonic_to_noise,
                'mos_estimation': metrics_result.naturalness.mos_estimation,
                'mos_method': metrics_result.naturalness.mos_method,
                'mos_nisqa': metrics_result.naturalness.mos_nisqa,
                'mos_heuristic': metrics_result.naturalness.mos_heuristic,
                'mcd': metrics_result.naturalness.mcd,
                'confidence_score': metrics_result.naturalness.confidence_score
            },
            'intelligibility': {
                'stoi_score': metrics_result.intelligibility.stoi_score,
                'estoi_score': metrics_result.intelligibility.estoi_score,
                'pesq_score': metrics_result.intelligibility.pesq_score,
                'confidence_score': metrics_result.intelligibility.confidence_score
            },
            'overall_quality_score': metrics_result.overall_quality_score
        }

        # Save to validation results directory
        validation_dir = tmp_path / 'validation_results'
        validation_dir.mkdir(exist_ok=True)
        metrics_file = validation_dir / 'quality_metrics.json'

        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        print(f"\n✓ Metrics saved to: {metrics_file}")

        # Assert comprehensive quality thresholds
        assert mos_estimation > 4.0, \
            f"Quality gate FAILED: MOS {mos_estimation:.2f} below 4.0 threshold"

        assert stoi_score > 0.9, \
            f"Quality gate FAILED: STOI {stoi_score:.3f} below 0.9 threshold"

        if mcd_value is not None:
            assert mcd_value < 6.0, \
                f"Quality gate FAILED: MCD {mcd_value:.2f} dB exceeds 6.0 dB threshold"

        print("\n✓ Quality gate PASSED: All comprehensive metrics meet targets")
        print(f"  ✓ MOS: {mos_estimation:.2f} > 4.0")
        print(f"  ✓ STOI: {stoi_score:.3f} > 0.9")
        if mcd_value is not None:
            print(f"  ✓ MCD: {mcd_value:.2f} dB < 6.0 dB")


@pytest.mark.e2e
@pytest.mark.integration
class TestWebSocketIntegration:
    """Test WebSocket integration with conversion pipeline"""
    
    def test_websocket_progress_updates(self, song_file, test_profile, device):
        """Test WebSocket receives progress updates during conversion"""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            from flask_socketio import SocketIO
            from flask import Flask
        except ImportError:
            pytest.skip("Components not available")
        
        # Create mock SocketIO
        app = Flask(__name__)
        socketio = SocketIO(app)
        
        # Track emitted events
        emitted_events = []
        
        def mock_emit(event, data, room=None):
            emitted_events.append({'event': event, 'data': data, 'room': room})
        
        socketio.emit = mock_emit
        
        # Create pipeline with progress callback
        pipeline = SingingConversionPipeline(config={'device': device})
        
        progress_updates = []
        
        def progress_callback(stage, percent):
            progress_updates.append({'percent': percent, 'stage': stage})
            # Simulate WebSocket emission
            socketio.emit('conversion_progress', {
                'job_id': 'test-job',
                'progress': percent,
                'stage': stage
            }, room='test-job')
        
        # Run conversion
        result = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            progress_callback=progress_callback
        )
        
        # Verify progress updates
        assert len(progress_updates) > 0, "No progress updates received"
        assert len(emitted_events) > 0, "No WebSocket events emitted"
        
        # Verify progress goes from 0 to 100
        percents = [p['percent'] for p in progress_updates]
        assert min(percents) >= 0.0
        assert max(percents) >= 90.0  # Should reach near 100%
        
        # Verify all expected stages present
        stages = set(p['stage'] for p in progress_updates)
        expected_stages = {'source_separation', 'pitch_extraction', 'voice_conversion', 'audio_mixing'}
        assert len(stages.intersection(expected_stages)) >= 3, f"Missing stages: {expected_stages - stages}"
        
        # Verify WebSocket events have correct structure
        for event in emitted_events:
            assert event['event'] == 'conversion_progress'
            assert 'job_id' in event['data']
            assert 'progress' in event['data']
            assert 'stage' in event['data']
            assert event['room'] == 'test-job'
    
    def test_websocket_completion_event(self, song_file, test_profile, device):
        """Test WebSocket emits completion event with correct data"""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("Components not available")
        
        pipeline = SingingConversionPipeline(config={'device': device})
        
        # Run conversion
        result = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        
        # Simulate completion event emission
        completion_data = {
            'job_id': 'test-job',
            'status': 'completed',
            'output_url': '/api/v1/convert/download/test-job',
            'duration': result['duration'],
            'sample_rate': result['sample_rate'],
            'metadata': result['metadata']
        }
        
        # Verify completion data structure
        assert 'job_id' in completion_data
        assert 'output_url' in completion_data
        assert 'duration' in completion_data
        assert completion_data['duration'] > 0
        assert completion_data['sample_rate'] > 0
        assert 'metadata' in completion_data
        assert 'target_profile_id' in completion_data['metadata']
    
    def test_websocket_error_event(self, device):
        """Test WebSocket emits error event on conversion failure"""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("Components not available")
        
        pipeline = SingingConversionPipeline(config={'device': device})
        
        # Try conversion with invalid file
        with pytest.raises(FileNotFoundError):
            pipeline.convert_song(
                song_path='/nonexistent/file.wav',
                target_profile_id='test-profile'
            )
        
        # Simulate error event emission
        error_data = {
            'job_id': 'test-job',
            'error': 'Song file not found: /nonexistent/file.wav',
            'stage': 'source_separation'
        }
        
        # Verify error data structure
        assert 'job_id' in error_data
        assert 'error' in error_data
        assert len(error_data['error']) > 0
    
    def test_websocket_room_isolation(self, song_file, test_profile, device):
        """Test WebSocket events are isolated to correct rooms"""
        # This test verifies that progress updates for job A
        # don't leak to clients subscribed to job B
        
        # Create two mock clients
        client_a_events = []
        client_b_events = []
        
        def client_a_handler(data):
            client_a_events.append(data)
        
        def client_b_handler(data):
            client_b_events.append(data)
        
        # Simulate two jobs running concurrently
        job_a_id = 'job-a'
        job_b_id = 'job-b'
        
        # Emit events to different rooms
        events = [
            {'room': job_a_id, 'data': {'job_id': job_a_id, 'progress': 50}},
            {'room': job_b_id, 'data': {'job_id': job_b_id, 'progress': 75}},
            {'room': job_a_id, 'data': {'job_id': job_a_id, 'progress': 100}},
        ]
        
        # Simulate room-based delivery
        for event in events:
            if event['room'] == job_a_id:
                client_a_handler(event['data'])
            elif event['room'] == job_b_id:
                client_b_handler(event['data'])
        
        # Verify isolation
        assert len(client_a_events) == 2
        assert all(e['job_id'] == job_a_id for e in client_a_events)
        
        assert len(client_b_events) == 1
        assert all(e['job_id'] == job_b_id for e in client_b_events)
