"""Validation tests for new testing infrastructure fixtures.

This module validates that all new fixtures work correctly and can be
used in actual tests. Serves as both documentation and smoke tests.
"""

import pytest
import numpy as np
import torch
from pathlib import Path


# ============================================================================
# Audio Fixture Tests
# ============================================================================

class TestAudioFixtures:
    """Test audio generation fixtures."""

    def test_sample_audio_factory(self, sample_audio_factory):
        """Test audio factory generates various audio types."""
        # Sine wave
        sine = sample_audio_factory('sine', frequency=440, duration=1.0)
        assert isinstance(sine, np.ndarray)
        assert sine.dtype == np.float32
        assert len(sine) == 22050  # 1 second at 22050 Hz

        # Harmonics
        harmonics = sample_audio_factory('harmonics', fundamental=220, num_harmonics=5)
        assert isinstance(harmonics, np.ndarray)

        # Speech-like
        speech = sample_audio_factory('speech_like', formants=[800, 1200, 2500])
        assert isinstance(speech, np.ndarray)

        # Noise
        noise = sample_audio_factory('noise', noise_type='white')
        assert isinstance(noise, np.ndarray)

    def test_audio_file_factory(self, audio_file_factory, sample_audio_factory):
        """Test audio file creation."""
        audio = sample_audio_factory('sine', duration=0.5)
        filepath = audio_file_factory('test.wav', audio, sample_rate=22050)

        assert filepath.exists()
        assert filepath.suffix == '.wav'

    def test_multi_channel_audio(self, multi_channel_audio):
        """Test multi-channel audio generation."""
        # Stereo
        stereo = multi_channel_audio(num_channels=2, relationship='identical')
        assert stereo.shape[1] == 2

        # Phase shifted
        phase = multi_channel_audio(num_channels=4, relationship='phase_shifted')
        assert phase.shape[1] == 4

    def test_audio_batch_generator(self, audio_batch_generator):
        """Test batch generation."""
        batches = list(audio_batch_generator(
            batch_size=8,
            num_batches=3,
            audio_type='harmonics'
        ))

        assert len(batches) == 3
        assert len(batches[0]) == 8

    def test_corrupted_audio_samples(self, corrupted_audio_samples):
        """Test corrupted audio samples fixture."""
        samples = corrupted_audio_samples

        assert 'clipped' in samples
        assert 'silent' in samples
        assert 'dc_offset' in samples
        assert 'inf_values' in samples

        # Verify corruption
        assert np.max(samples['clipped']) <= 1.0
        assert np.sum(samples['silent']) == 0


# ============================================================================
# Model Fixture Tests
# ============================================================================

class TestModelFixtures:
    """Test model mock fixtures."""

    def test_mock_voice_model(self, mock_voice_model):
        """Test mock voice model."""
        model = mock_voice_model

        # Test forward pass
        input_tensor = torch.randn(4, 100, 80)
        output = model(input_tensor)

        assert output.shape[0] == 4
        assert output.shape[-1] == 80

    def test_mock_encoder(self, mock_encoder):
        """Test mock encoder."""
        encoder = mock_encoder

        input_data = torch.randn(16, 80)
        embedding = encoder.encode(input_data)

        assert embedding.shape == (16, 256)
        assert encoder.num_calls == 1

    def test_trained_model_checkpoint(self, trained_model_checkpoint):
        """Test checkpoint creation."""
        ckpt_path = trained_model_checkpoint(
            model_type='voice_transformer',
            epoch=50,
            include_optimizer=True
        )

        assert ckpt_path.exists()

        # Load and verify
        checkpoint = torch.load(ckpt_path)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['epoch'] == 50

    def test_model_forward_tester(self, model_forward_tester, mock_voice_model):
        """Test model forward tester."""
        model = mock_voice_model
        input_tensor = torch.randn(4, 100, 80)

        results = model_forward_tester.test_forward(
            model,
            input_tensor,
            expected_shape=(4, 100, 80),
            check_gradients=False
        )

        assert 'output_shape' in results
        assert results['shape_correct']


# ============================================================================
# GPU Fixture Tests
# ============================================================================

class TestGPUFixtures:
    """Test GPU-related fixtures."""

    @pytest.mark.cuda
    def test_gpu_context_manager(self, gpu_context_manager):
        """Test GPU context manager."""
        with gpu_context_manager() as ctx:
            tensor = torch.randn(100, 100, device='cuda')
            result = tensor * 2

        assert ctx.peak_memory_mb >= 0

    @pytest.mark.cuda
    def test_cuda_memory_tracker(self, cuda_memory_tracker):
        """Test CUDA memory tracker."""
        tracker = cuda_memory_tracker

        tracker.start()
        tensor = torch.randn(1000, 1000, device='cuda')
        tracker.checkpoint('after_allocation')
        del tensor
        stats = tracker.stop()

        assert 'peak_mb' in stats
        assert len(stats['snapshots']) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multi_gpu_config(self, multi_gpu_config):
        """Test multi-GPU configuration."""
        config = multi_gpu_config

        assert config.num_gpus >= 1
        devices = config.get_devices()
        assert len(devices) > 0


# ============================================================================
# Mock Fixture Tests
# ============================================================================

class TestMockFixtures:
    """Test mock object fixtures."""

    def test_mock_file_system(self, mock_file_system):
        """Test mock file system."""
        fs = mock_file_system

        # Write
        fs.write('test.txt', 'Hello World')
        assert fs.exists('test.txt')

        # Read
        content = fs.read('test.txt')
        assert content == 'Hello World'

        # Delete
        fs.delete('test.txt')
        assert not fs.exists('test.txt')

    def test_mock_audio_loader(self, mock_audio_loader):
        """Test mock audio loader."""
        loader = mock_audio_loader

        audio, sr = loader.load('fake_song.wav')

        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        assert loader.load_count == 1

    def test_mock_cache_manager(self, mock_cache_manager):
        """Test mock cache manager."""
        cache = mock_cache_manager

        # Set and get
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'

        # Stats
        stats = cache.get_stats()
        assert stats['hits'] >= 1
        assert stats['size'] >= 1

    def test_mock_database(self, mock_database):
        """Test mock database."""
        db = mock_database

        # Insert
        user_id = db.insert('users', {'name': 'Alice', 'age': 30})
        assert user_id == 1

        # Select
        users = db.select('users', where={'name': 'Alice'})
        assert len(users) == 1
        assert users[0]['age'] == 30


# ============================================================================
# Integration Fixture Tests
# ============================================================================

class TestIntegrationFixtures:
    """Test integration testing fixtures."""

    def test_pipeline_test_suite(self, pipeline_test_suite, sample_audio_factory):
        """Test pipeline test suite."""
        suite = pipeline_test_suite

        audio = sample_audio_factory('sine', duration=1.0)
        suite.add_test_case(
            'test1',
            audio,
            {'profile_id': 'test'},
            expected_metrics={'snr': 20.0}
        )

        assert len(suite.test_cases) == 1

    def test_end_to_end_workflow(self, end_to_end_workflow):
        """Test E2E workflow."""
        workflow = end_to_end_workflow

        workflow.setup_test_data(audio_duration=1.0)
        assert workflow.input_audio is not None
        assert len(workflow.input_audio) > 0

        workflow.set_final_output(workflow.input_audio)
        summary = workflow.get_summary()

        assert summary['has_final_output']

    def test_data_flow_validator(self, data_flow_validator):
        """Test data flow validator."""
        validator = data_flow_validator

        data = np.random.randn(16, 80)
        validator.add_checkpoint(
            'encoder_output',
            data,
            expected_shape=(16, 80),
            expected_dtype=np.ndarray
        )

        assert validator.validate_all()


# ============================================================================
# Performance Fixture Tests
# ============================================================================

class TestPerformanceFixtures:
    """Test performance testing fixtures."""

    def test_performance_benchmarker(self, performance_benchmarker):
        """Test performance benchmarker."""
        bench = performance_benchmarker

        def dummy_func():
            import time
            time.sleep(0.001)

        stats = bench.benchmark(dummy_func, iterations=10, warmup=2)

        assert 'mean' in stats
        assert 'std' in stats
        assert stats['iterations'] == 10

    def test_resource_profiler(self, resource_profiler):
        """Test resource profiler."""
        profiler = resource_profiler

        with profiler.profile():
            # Allocate some memory
            data = np.random.randn(1000, 1000)
            result = data * 2

        summary = profiler.get_summary()

        assert 'duration' in summary
        assert 'cpu' in summary
        assert 'memory' in summary

    def test_throughput_tester(self, throughput_tester, sample_audio_factory):
        """Test throughput tester."""
        tester = throughput_tester

        audio = sample_audio_factory('sine', duration=1.0)

        def process_func(audio_data):
            # Dummy processing
            return audio_data * 2

        rtf = tester.measure_rtf(
            process_func,
            audio,
            sample_rate=22050,
            iterations=5
        )

        assert 'mean_rtf' in rtf
        assert 'audio_duration' in rtf

    def test_regression_tester(self, regression_tester):
        """Test regression tester."""
        tester = regression_tester

        # Set baseline
        tester.set_baseline('test_metric', 0.05)

        # Measure
        def fast_func():
            import time
            time.sleep(0.01)

        current = tester.measure('test_metric', fast_func, iterations=5)

        # Should pass (0.01 < 0.05)
        assert tester.check_regression('test_metric', tolerance=0.1)


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_audio_assertions(self, sample_audio_factory):
        """Test audio assertion helpers."""
        from tests.utils import (
            assert_audio_equal,
            assert_audio_normalized,
            compute_snr,
            compute_similarity
        )

        audio1 = sample_audio_factory('sine', frequency=440)
        audio2 = sample_audio_factory('sine', frequency=440)

        # Should pass - identical
        assert_audio_equal(audio1, audio2, rtol=1e-5)
        assert_audio_normalized(audio1, max_value=1.0)

        # Similarity
        similarity = compute_similarity(audio1, audio2)
        assert similarity > 0.99  # Very similar

    def test_model_assertions(self, mock_voice_model):
        """Test model assertion helpers."""
        from tests.utils import (
            assert_model_outputs_valid,
            count_parameters
        )

        model = mock_voice_model
        output = model(torch.randn(4, 100, 80))

        assert_model_outputs_valid(
            output,
            check_nan=True,
            check_inf=True
        )

        params = count_parameters(model)
        assert 'total' in params
        assert 'trainable' in params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
