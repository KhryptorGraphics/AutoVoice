"""Integration testing fixtures for end-to-end workflows.

Provides fixtures for testing complete pipelines, component interactions,
and real-world usage scenarios.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# Pipeline Test Fixtures
# ============================================================================

@pytest.fixture
def pipeline_test_suite(tmp_path: Path):
    """Complete test suite for voice conversion pipeline.

    Provides test data, expected outputs, and validation methods for
    testing the full voice conversion pipeline.

    Examples:
        suite = pipeline_test_suite
        result = pipeline.convert(suite.input_audio, suite.target_profile)
        suite.validate_output(result)
    """
    class PipelineTestSuite:
        def __init__(self, base_path: Path):
            self.base_path = base_path
            self.test_cases = []
            self.results = []

        def add_test_case(
            self,
            name: str,
            input_audio: np.ndarray,
            target_profile: Dict[str, Any],
            expected_metrics: Optional[Dict[str, float]] = None
        ):
            """Add test case to suite.

            Args:
                name: Test case name
                input_audio: Input audio samples
                target_profile: Target voice profile
                expected_metrics: Expected quality metrics
            """
            self.test_cases.append({
                'name': name,
                'input_audio': input_audio,
                'target_profile': target_profile,
                'expected_metrics': expected_metrics or {}
            })

        def run_pipeline(self, pipeline_func: callable) -> List[Dict[str, Any]]:
            """Run pipeline on all test cases.

            Args:
                pipeline_func: Pipeline function to test

            Returns:
                List of results
            """
            results = []

            for test_case in self.test_cases:
                try:
                    result = pipeline_func(
                        test_case['input_audio'],
                        test_case['target_profile']
                    )

                    results.append({
                        'name': test_case['name'],
                        'success': True,
                        'output': result,
                        'error': None
                    })

                except Exception as e:
                    results.append({
                        'name': test_case['name'],
                        'success': False,
                        'output': None,
                        'error': str(e)
                    })

            self.results = results
            return results

        def validate_results(self) -> Dict[str, Any]:
            """Validate pipeline results.

            Returns:
                Validation summary
            """
            total = len(self.results)
            successful = sum(1 for r in self.results if r['success'])
            failed = total - successful

            return {
                'total': total,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total if total > 0 else 0,
                'failures': [
                    {'name': r['name'], 'error': r['error']}
                    for r in self.results if not r['success']
                ]
            }

    return PipelineTestSuite(tmp_path)


@pytest.fixture
def end_to_end_workflow():
    """End-to-end workflow testing fixture.

    Tests complete workflows from audio input to final output,
    including all intermediate steps.

    Examples:
        workflow = end_to_end_workflow
        workflow.setup_test_data()
        workflow.run_conversion()
        assert workflow.validate_quality()
    """
    class EndToEndWorkflow:
        def __init__(self):
            self.input_audio = None
            self.target_profile = None
            self.intermediate_outputs = {}
            self.final_output = None
            self.metrics = {}

        def setup_test_data(
            self,
            audio_duration: float = 3.0,
            sample_rate: int = 22050
        ):
            """Setup test audio data.

            Args:
                audio_duration: Audio duration in seconds
                sample_rate: Sample rate in Hz
            """
            num_samples = int(sample_rate * audio_duration)
            t = np.linspace(0, audio_duration, num_samples)

            # Generate speech-like audio
            audio = np.zeros(num_samples, dtype=np.float32)
            for formant in [800, 1200, 2500]:
                audio += np.sin(2 * np.pi * formant * t)

            # Add envelope
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5.0 * t)
            audio *= envelope

            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.9

            self.input_audio = audio
            self.sample_rate = sample_rate

        def set_target_profile(self, profile: Dict[str, Any]):
            """Set target voice profile.

            Args:
                profile: Voice profile dict
            """
            self.target_profile = profile

        def record_intermediate(self, step: str, output: Any):
            """Record intermediate output.

            Args:
                step: Processing step name
                output: Output data
            """
            self.intermediate_outputs[step] = output

        def set_final_output(self, output: Any):
            """Set final output.

            Args:
                output: Final conversion output
            """
            self.final_output = output

        def compute_metrics(self, metrics_func: callable):
            """Compute quality metrics.

            Args:
                metrics_func: Function to compute metrics
            """
            if self.final_output is not None:
                self.metrics = metrics_func(self.input_audio, self.final_output)

        def validate_quality(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
            """Validate output quality.

            Args:
                thresholds: Quality thresholds

            Returns:
                True if all quality checks pass
            """
            if not self.metrics:
                return False

            thresholds = thresholds or {
                'snr_db': 20.0,
                'similarity': 0.7,
                'pitch_error_hz': 50.0
            }

            return all(
                self.metrics.get(metric, 0) >= threshold
                for metric, threshold in thresholds.items()
            )

        def get_summary(self) -> Dict[str, Any]:
            """Get workflow summary.

            Returns:
                Summary dict
            """
            return {
                'input_shape': self.input_audio.shape if self.input_audio is not None else None,
                'num_intermediate_steps': len(self.intermediate_outputs),
                'has_final_output': self.final_output is not None,
                'metrics': self.metrics,
                'intermediate_steps': list(self.intermediate_outputs.keys())
            }

    return EndToEndWorkflow()


@pytest.fixture
def concurrent_pipeline_tester():
    """Test pipeline with concurrent requests.

    Tests thread safety and performance under concurrent load.

    Examples:
        tester = concurrent_pipeline_tester
        results = tester.run_concurrent(pipeline, num_workers=4, num_requests=100)
    """
    class ConcurrentPipelineTester:
        def __init__(self):
            self.results = []
            self.errors = []

        def run_concurrent(
            self,
            pipeline_func: callable,
            test_data: List[Any],
            num_workers: int = 4,
            timeout: Optional[float] = None
        ) -> Dict[str, Any]:
            """Run pipeline concurrently.

            Args:
                pipeline_func: Pipeline function to test
                test_data: List of test inputs
                num_workers: Number of concurrent workers
                timeout: Timeout per request in seconds

            Returns:
                Results dict
            """
            self.results = []
            self.errors = []

            import time
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(pipeline_func, data): i
                    for i, data in enumerate(test_data)
                }

                for future in as_completed(futures, timeout=timeout):
                    idx = futures[future]
                    try:
                        result = future.result()
                        self.results.append({
                            'index': idx,
                            'success': True,
                            'output': result
                        })
                    except Exception as e:
                        self.errors.append({
                            'index': idx,
                            'error': str(e)
                        })

            elapsed = time.time() - start_time

            return {
                'total_requests': len(test_data),
                'successful': len(self.results),
                'failed': len(self.errors),
                'elapsed_time': elapsed,
                'requests_per_sec': len(test_data) / elapsed,
                'errors': self.errors
            }

        def stress_test(
            self,
            pipeline_func: callable,
            single_input: Any,
            duration: float = 10.0,
            num_workers: int = 4
        ) -> Dict[str, Any]:
            """Stress test pipeline for duration.

            Args:
                pipeline_func: Pipeline function
                single_input: Input to repeat
                duration: Test duration in seconds
                num_workers: Number of workers

            Returns:
                Stress test results
            """
            import time

            start_time = time.time()
            iteration_times = []
            errors = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                while time.time() - start_time < duration:
                    iter_start = time.time()

                    try:
                        future = executor.submit(pipeline_func, single_input)
                        future.result(timeout=5.0)
                        iteration_times.append(time.time() - iter_start)

                    except Exception as e:
                        errors.append(str(e))

            return {
                'duration': time.time() - start_time,
                'total_iterations': len(iteration_times),
                'throughput': len(iteration_times) / (time.time() - start_time),
                'avg_latency': np.mean(iteration_times) if iteration_times else 0,
                'p50_latency': np.percentile(iteration_times, 50) if iteration_times else 0,
                'p95_latency': np.percentile(iteration_times, 95) if iteration_times else 0,
                'p99_latency': np.percentile(iteration_times, 99) if iteration_times else 0,
                'errors': errors,
                'error_rate': len(errors) / len(iteration_times) if iteration_times else 1.0
            }

    return ConcurrentPipelineTester()


@pytest.fixture
def data_flow_validator():
    """Validate data flow through pipeline components.

    Ensures data shapes, types, and values are correct at each step.

    Examples:
        validator = data_flow_validator
        validator.add_checkpoint('encoder_output', output, expected_shape=(16, 256))
        assert validator.validate_all()
    """
    class DataFlowValidator:
        def __init__(self):
            self.checkpoints = []
            self.validations = []

        def add_checkpoint(
            self,
            name: str,
            data: Any,
            expected_shape: Optional[tuple] = None,
            expected_dtype: Optional[type] = None,
            expected_range: Optional[tuple] = None
        ):
            """Add validation checkpoint.

            Args:
                name: Checkpoint name
                data: Data to validate
                expected_shape: Expected shape
                expected_dtype: Expected data type
                expected_range: Expected value range (min, max)
            """
            self.checkpoints.append({
                'name': name,
                'data': data,
                'expected_shape': expected_shape,
                'expected_dtype': expected_dtype,
                'expected_range': expected_range
            })

        def validate_all(self) -> bool:
            """Validate all checkpoints.

            Returns:
                True if all validations pass
            """
            self.validations = []

            for checkpoint in self.checkpoints:
                validation = self._validate_checkpoint(checkpoint)
                self.validations.append(validation)

            return all(v['passed'] for v in self.validations)

        def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
            """Validate single checkpoint.

            Args:
                checkpoint: Checkpoint dict

            Returns:
                Validation result
            """
            data = checkpoint['data']
            errors = []

            # Shape validation
            if checkpoint['expected_shape'] is not None:
                if hasattr(data, 'shape'):
                    if data.shape != checkpoint['expected_shape']:
                        errors.append(
                            f"Shape mismatch: {data.shape} != {checkpoint['expected_shape']}"
                        )

            # Dtype validation
            if checkpoint['expected_dtype'] is not None:
                actual_dtype = type(data)
                if hasattr(data, 'dtype'):
                    actual_dtype = data.dtype

                if actual_dtype != checkpoint['expected_dtype']:
                    errors.append(
                        f"Dtype mismatch: {actual_dtype} != {checkpoint['expected_dtype']}"
                    )

            # Range validation
            if checkpoint['expected_range'] is not None:
                min_val, max_val = checkpoint['expected_range']
                if hasattr(data, 'min') and hasattr(data, 'max'):
                    data_min = float(data.min())
                    data_max = float(data.max())

                    if data_min < min_val or data_max > max_val:
                        errors.append(
                            f"Range violation: [{data_min}, {data_max}] not in [{min_val}, {max_val}]"
                        )

            return {
                'name': checkpoint['name'],
                'passed': len(errors) == 0,
                'errors': errors
            }

        def get_report(self) -> Dict[str, Any]:
            """Get validation report.

            Returns:
                Validation report
            """
            total = len(self.validations)
            passed = sum(1 for v in self.validations if v['passed'])

            return {
                'total_checkpoints': total,
                'passed': passed,
                'failed': total - passed,
                'validations': self.validations
            }

    return DataFlowValidator()


__all__ = [
    'pipeline_test_suite',
    'end_to_end_workflow',
    'concurrent_pipeline_tester',
    'data_flow_validator',
]
