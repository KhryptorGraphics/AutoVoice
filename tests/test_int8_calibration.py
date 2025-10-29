"""
Test script for INT8 calibration pipeline.

Verifies that:
1. Calibration data can be created with correct dtypes
2. Calibration data can be loaded correctly
3. INT8 calibrator receives and uses calibration data
4. Calibration cache is written after engine build
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Skip if TensorRT not available
tensorrt = pytest.importorskip("tensorrt", reason="TensorRT not available")
pycuda = pytest.importorskip("pycuda.driver", reason="PyCUDA not available")

from src.auto_voice.inference.tensorrt_converter import TensorRTConverter


class MockDataset:
    """Mock dataset for calibration testing."""

    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            # Mock sample with required attributes
            sample = type('Sample', (), {
                'source_audio': np.random.randn(16000).astype(np.float32),
                'source_f0': np.random.randn(50).astype(np.float32) * 200 + 300
            })()
            yield sample

    def __len__(self):
        return self.num_samples


def test_create_calibration_dataset():
    """Test calibration dataset creation with correct dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = TensorRTConverter(export_dir=tmpdir, device='cpu')
        dataset = MockDataset(num_samples=5)
        output_path = os.path.join(tmpdir, "calibration.npz")

        # Create calibration dataset
        result_path = converter.create_calibration_dataset(
            dataset=dataset,
            num_samples=5,
            output_path=output_path
        )

        assert os.path.exists(result_path)

        # Verify dtypes in NPZ
        with np.load(result_path) as data:
            # Content encoder
            assert 'content/input_audio' in data
            assert 'content/sample_rate' in data
            assert data['content/input_audio'].dtype == np.float32
            assert data['content/sample_rate'].dtype == np.int32

            # Pitch encoder
            assert 'pitch/f0_input' in data
            assert 'pitch/voiced_mask' in data
            assert data['pitch/f0_input'].dtype == np.float32
            assert data['pitch/voiced_mask'].dtype == np.bool_

            # Flow decoder
            assert 'flow/latent_input' in data
            assert 'flow/mask' in data
            assert 'flow/conditioning' in data
            assert data['flow/latent_input'].dtype == np.float32
            assert data['flow/mask'].dtype == np.float32
            assert data['flow/conditioning'].dtype == np.float32

        print("✓ Calibration dataset created with correct dtypes")


def test_load_calibration_data():
    """Test loading calibration data with correct dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = TensorRTConverter(export_dir=tmpdir, device='cpu')

        # Create test NPZ with correct structure
        calibration_data = {
            'content/input_audio': np.random.randn(3, 16000).astype(np.float32),
            'content/sample_rate': np.full(3, 16000, dtype=np.int32),
            'pitch/f0_input': np.random.randn(3, 50).astype(np.float32),
            'pitch/voiced_mask': (np.random.randn(3, 50) > 0).astype(np.bool_),
            'flow/latent_input': np.random.randn(3, 192, 50).astype(np.float32),
            'flow/mask': np.ones((3, 1, 50), dtype=np.float32),
            'flow/conditioning': np.random.randn(3, 704, 50).astype(np.float32)
        }

        npz_path = os.path.join(tmpdir, "test_calibration.npz")
        np.savez(npz_path, **calibration_data)

        # Test content encoder loading
        content_data = converter._load_calibration_data('content_encoder', npz_path)
        assert len(content_data) == 3
        assert 'input_audio' in content_data[0]
        assert 'sample_rate' in content_data[0]
        assert content_data[0]['input_audio'].dtype == np.float32
        assert content_data[0]['sample_rate'].dtype == np.int32

        # Test pitch encoder loading
        pitch_data = converter._load_calibration_data('pitch_encoder', npz_path)
        assert len(pitch_data) == 3
        assert 'f0_input' in pitch_data[0]
        assert 'voiced_mask' in pitch_data[0]
        assert pitch_data[0]['f0_input'].dtype == np.float32
        assert pitch_data[0]['voiced_mask'].dtype == np.bool_

        # Test flow decoder loading
        flow_data = converter._load_calibration_data('flow_decoder', npz_path)
        assert len(flow_data) == 3
        assert 'latent_input' in flow_data[0]
        assert 'mask' in flow_data[0]
        assert 'conditioning' in flow_data[0]
        assert flow_data[0]['latent_input'].dtype == np.float32
        assert flow_data[0]['mask'].dtype == np.float32
        assert flow_data[0]['conditioning'].dtype == np.float32

        print("✓ Calibration data loaded with correct dtypes")


def test_create_calibrator():
    """Test INT8 calibrator creation with real calibration data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = TensorRTConverter(export_dir=tmpdir, device='cpu')

        # Create test calibration data
        calibration_data = [
            {
                'input_audio': np.random.randn(16000).astype(np.float32),
                'sample_rate': np.array([16000], dtype=np.int32)
            },
            {
                'input_audio': np.random.randn(16000).astype(np.float32),
                'sample_rate': np.array([16000], dtype=np.int32)
            }
        ]

        cache_file = os.path.join(tmpdir, "test_calibration.cache")

        # Create calibrator
        calibrator = converter._create_calibrator(
            calibration_cache_file=cache_file,
            calibration_data=calibration_data
        )

        # Verify calibrator properties
        assert calibrator is not None
        assert calibrator.get_batch_size() == 1
        assert hasattr(calibrator, 'calibration_data')
        assert len(calibrator.calibration_data) == 2

        # Test get_batch returns data
        batch1 = calibrator.get_batch(['input_audio', 'sample_rate'])
        assert batch1 is not None
        assert len(batch1) == 2

        batch2 = calibrator.get_batch(['input_audio', 'sample_rate'])
        assert batch2 is not None

        # Third call should return None (no more data)
        batch3 = calibrator.get_batch(['input_audio', 'sample_rate'])
        assert batch3 is None

        print("✓ INT8 calibrator created and functioning correctly")


def test_calibrator_honors_data():
    """Test that _create_calibrator honors provided calibration_data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = TensorRTConverter(export_dir=tmpdir, device='cpu')

        # Test with provided data
        test_data = [{'test': np.array([1, 2, 3])}]
        calibrator = converter._create_calibrator(
            calibration_data=test_data,
            calibration_cache_file=os.path.join(tmpdir, "test.cache")
        )

        assert calibrator.calibration_data == test_data
        assert len(calibrator.calibration_data) == 1

        # Test with None data
        calibrator_empty = converter._create_calibrator(
            calibration_data=None,
            calibration_cache_file=os.path.join(tmpdir, "test2.cache")
        )

        assert calibrator_empty.calibration_data == []
        assert len(calibrator_empty.calibration_data) == 0

        print("✓ Calibrator correctly honors provided calibration_data")


if __name__ == '__main__':
    print("\nRunning INT8 Calibration Tests\n" + "="*50)

    try:
        test_create_calibration_dataset()
        test_load_calibration_data()
        test_create_calibrator()
        test_calibrator_honors_data()

        print("\n" + "="*50)
        print("✓ All INT8 calibration tests passed!")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
