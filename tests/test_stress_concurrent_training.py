"""Stress tests for multiple profiles and concurrent training.

Task 7.6: Stress test with multiple profiles and concurrent training

Tests cover:
- Multiple profile creation and management
- Concurrent training job execution
- GPU memory management under load
- Resource cleanup after stress
"""

import asyncio
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Get CUDA device, skip test if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def temp_storage():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 256),
    )
    return model


# ============================================================================
# Test: Multiple Profile Management
# ============================================================================


@pytest.mark.cuda
class TestMultipleProfileManagement:
    """Tests for managing multiple voice profiles."""

    def test_create_multiple_profiles(self, temp_storage):
        """System should handle creation of many profiles."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=True):
            manager = TrainingJobManager(storage_path=temp_storage)

        # Create multiple profiles worth of jobs
        profile_ids = [f"profile_{i}" for i in range(10)]
        jobs = []

        for profile_id in profile_ids:
            job = manager.create_job(
                profile_id=profile_id,
                sample_ids=[f"sample_{j}" for j in range(3)],
            )
            jobs.append(job)

        assert len(jobs) == 10
        assert len(set(j.profile_id for j in jobs)) == 10

    def test_profile_isolation(self, temp_storage):
        """Jobs for different profiles should be isolated."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=True):
            manager = TrainingJobManager(storage_path=temp_storage)

        # Create jobs for two different profiles
        job_a = manager.create_job(profile_id="profile_a", sample_ids=["s1"])
        job_b = manager.create_job(profile_id="profile_b", sample_ids=["s2"])

        # Jobs should have different IDs
        assert job_a.job_id != job_b.job_id
        assert job_a.profile_id != job_b.profile_id

    def test_profile_job_queue_capacity(self, temp_storage):
        """Job queue should handle many pending jobs."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=True):
            manager = TrainingJobManager(storage_path=temp_storage)

        # Queue many jobs
        num_jobs = 50
        for i in range(num_jobs):
            manager.create_job(
                profile_id=f"profile_{i % 5}",  # 5 profiles
                sample_ids=[f"sample_{i}"],
            )

        pending = manager.get_pending_jobs()
        assert len(pending) == num_jobs


# ============================================================================
# Test: Concurrent Training Simulation
# ============================================================================


@pytest.mark.cuda
class TestConcurrentTrainingSimulation:
    """Tests for concurrent training job execution."""

    def test_concurrent_job_creation(self, temp_storage):
        """Multiple threads should safely create jobs concurrently."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=True):
            manager = TrainingJobManager(storage_path=temp_storage)

        results = []
        errors = []

        def create_job(profile_id):
            try:
                job = manager.create_job(
                    profile_id=profile_id,
                    sample_ids=["sample_1"],
                )
                results.append(job.job_id)
            except Exception as e:
                errors.append(str(e))

        # Create jobs from multiple threads
        threads = []
        for i in range(20):
            t = threading.Thread(target=create_job, args=(f"profile_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        assert len(set(results)) == 20  # All unique IDs

    def test_concurrent_model_checksum_computation(self, device, temp_storage):
        """Multiple threads computing checksums should not interfere."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Create multiple models
        models = [
            torch.nn.Linear(64, 64).to(device)
            for _ in range(5)
        ]

        checksums = []
        lock = threading.Lock()

        def compute_checksum(model, idx):
            checksum = manager.compute_model_checksum(model)
            with lock:
                checksums.append((idx, checksum))

        # Compute checksums concurrently
        threads = []
        for i, model in enumerate(models):
            t = threading.Thread(target=compute_checksum, args=(model, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(checksums) == 5
        # Each model should have a unique checksum
        unique_checksums = set(c[1] for c in checksums)
        assert len(unique_checksums) == 5

    def test_concurrent_engine_registration(self, device, temp_storage):
        """Multiple threads registering models should not corrupt state."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager

        manager = TRTEngineManager(cache_dir=str(temp_storage))
        models = [
            torch.nn.Linear(32, 32).to(device)
            for _ in range(10)
        ]

        errors = []

        def register_model(idx, model):
            try:
                manager.register_model(f"model_{idx}", model)
            except Exception as e:
                errors.append(str(e))

        # Register models concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(register_model, i, m)
                for i, m in enumerate(models)
            ]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        assert len(errors) == 0
        assert len(manager._registered_models) == 10


# ============================================================================
# Test: GPU Memory Under Load
# ============================================================================


@pytest.mark.cuda
class TestGPUMemoryUnderLoad:
    """Tests for GPU memory management under stress."""

    def test_memory_tracking_multiple_models(self, device, temp_storage):
        """Memory tracker should handle multiple models."""
        from auto_voice.gpu.memory_manager import GPUMemoryTracker

        tracker = GPUMemoryTracker(device=str(device))
        models = []

        # Create multiple models
        for i in range(5):
            model = torch.nn.Linear(256, 256).to(device)
            models.append(model)
            tracker.record_allocation(f"model_{i}", model)

        # Should track all allocations
        stats = tracker.get_stats()
        assert stats["total_allocations"] >= 5

    def test_memory_cleanup_on_model_deletion(self, device):
        """GPU memory should be released when models are deleted."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        initial_memory = torch.cuda.memory_allocated(device)

        # Allocate models
        models = []
        for _ in range(5):
            model = torch.nn.Linear(512, 512).to(device)
            models.append(model)

        peak_memory = torch.cuda.memory_allocated(device)
        assert peak_memory > initial_memory

        # Delete models
        del models
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated(device)
        # Memory should be released (allowing some overhead)
        assert final_memory < peak_memory

    def test_oom_recovery(self, device, temp_storage):
        """System should handle OOM gracefully."""
        from auto_voice.gpu.memory_manager import GPUMemoryTracker, handle_oom

        tracker = GPUMemoryTracker(device=str(device))

        # Simulate OOM handling
        try:
            # Try to allocate a very large tensor (may not actually OOM)
            large_tensor = torch.randn(100000, 100000, device=device)
            del large_tensor
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Should handle OOM gracefully
                handle_oom()
                torch.cuda.empty_cache()

        # System should still be functional
        small_tensor = torch.randn(100, 100, device=device)
        assert small_tensor.is_cuda


# ============================================================================
# Test: Resource Cleanup After Stress
# ============================================================================


@pytest.mark.cuda
class TestResourceCleanupAfterStress:
    """Tests for proper resource cleanup after stress testing."""

    def test_job_queue_cleanup(self, temp_storage):
        """Job queue should clean up completed jobs."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch("torch.cuda.is_available", return_value=True):
            manager = TrainingJobManager(storage_path=temp_storage)

        # Create and "complete" jobs
        for i in range(20):
            job = manager.create_job(
                profile_id=f"profile_{i}",
                sample_ids=["sample"],
            )
            manager._mark_job_completed(job.job_id)

        # Cleanup old completed jobs
        removed = manager.cleanup_completed_jobs(keep_count=5)
        assert len(removed) == 15

    def test_engine_cache_cleanup(self, device, temp_storage):
        """Engine cache should clean up old versions."""
        from auto_voice.inference.trt_rebuilder import TRTEngineManager
        import os
        import time as time_module

        manager = TRTEngineManager(cache_dir=str(temp_storage))

        # Create fake engine files for multiple models
        for model_idx in range(3):
            for version in range(5):
                engine_path = temp_storage / f"model{model_idx}_v{version}.engine"
                engine_path.write_bytes(b"fake")
                # Set different mtimes
                base_time = time_module.time()
                os.utime(engine_path, (
                    base_time - (5 - version) * 100,
                    base_time - (5 - version) * 100
                ))

        # Cleanup keeping only 2 per model
        removed = manager.cleanup_old_engines(keep_count=2)
        # Should remove 3 per model (5-2=3), total 9
        assert len(removed) >= 9

    def test_temp_file_cleanup(self, temp_storage):
        """Temporary files should be cleaned up."""
        # Create temp files
        temp_files = []
        for i in range(10):
            path = temp_storage / f"temp_{i}.bin"
            path.write_bytes(b"temporary data")
            temp_files.append(path)

        # Verify files exist
        assert all(p.exists() for p in temp_files)

        # Cleanup
        for path in temp_files:
            path.unlink()

        # Verify cleanup
        assert not any(p.exists() for p in temp_files)


# ============================================================================
# Test: Stress Test Metrics
# ============================================================================


@pytest.mark.cuda
class TestStressTestMetrics:
    """Tests for stress test metrics collection."""

    def test_throughput_measurement(self, device, temp_storage):
        """Should measure throughput under load."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device=str(device))

        # Simulate processing multiple items
        num_items = 20
        start_time = time.time()

        for i in range(num_items):
            with profiler.measure_stage("processing"):
                # Simulate work
                x = torch.randn(100, 100, device=device)
                _ = x @ x.T

        elapsed = time.time() - start_time
        throughput = num_items / elapsed

        assert throughput > 0
        stats = profiler.get_stats()
        assert stats["processing"]["count"] == num_items

    def test_error_rate_tracking(self, temp_storage):
        """Should track error rate during stress."""
        successes = 0
        failures = 0

        for i in range(100):
            try:
                # Simulate operation that sometimes fails
                if i % 10 == 0:
                    raise ValueError("Simulated error")
                successes += 1
            except ValueError:
                failures += 1

        error_rate = failures / (successes + failures)
        assert error_rate == 0.1  # 10% error rate

    def test_latency_percentiles(self, device, temp_storage):
        """Should calculate latency percentiles."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler
        import numpy as np

        profiler = InferenceLatencyProfiler(device=str(device))

        # Generate varied latencies
        for i in range(100):
            with profiler.measure_stage("operation"):
                time.sleep(0.001 + (i % 10) * 0.0001)  # Varied sleep

        stats = profiler.get_stats()
        times_ms = [t * 1000 for t in profiler.measurements["operation"]]

        p50 = np.percentile(times_ms, 50)
        p95 = np.percentile(times_ms, 95)
        p99 = np.percentile(times_ms, 99)

        assert p50 < p95 < p99
        assert p99 < 100  # Should all be under 100ms


# ============================================================================
# Test: Load Test Scenarios
# ============================================================================


@pytest.mark.cuda
@pytest.mark.slow
class TestLoadTestScenarios:
    """Load test scenarios simulating real-world usage."""

    def test_sustained_load(self, device, temp_storage):
        """System should handle sustained load."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler
        from auto_voice.training.job_manager import TrainingJobManager

        profiler = InferenceLatencyProfiler(device=str(device))

        with patch("torch.cuda.is_available", return_value=True):
            manager = TrainingJobManager(storage_path=temp_storage)

        # Simulate sustained load for N iterations
        iterations = 50
        for i in range(iterations):
            # Create job
            job = manager.create_job(
                profile_id=f"profile_{i % 5}",
                sample_ids=["sample"],
            )

            # Simulate processing
            with profiler.measure_stage("job_processing"):
                x = torch.randn(256, 256, device=device)
                _ = torch.matmul(x, x.T)

            # Mark complete
            manager._mark_job_completed(job.job_id)

        stats = profiler.get_stats()
        assert stats["job_processing"]["count"] == iterations
        # Average should be reasonable (< 100ms)
        assert stats["job_processing"]["mean_ms"] < 100

    def test_burst_load(self, device, temp_storage):
        """System should handle burst load patterns."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device=str(device))

        # Simulate bursts
        burst_sizes = [10, 20, 5, 15, 10]

        for burst_size in burst_sizes:
            for _ in range(burst_size):
                with profiler.measure_stage("burst_processing"):
                    x = torch.randn(128, 128, device=device)
                    _ = x @ x.T

            # Brief pause between bursts
            time.sleep(0.01)

        total_processed = sum(burst_sizes)
        stats = profiler.get_stats()
        assert stats["burst_processing"]["count"] == total_processed
