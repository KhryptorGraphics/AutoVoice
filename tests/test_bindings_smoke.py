"""Smoke test for CUDA kernel bindings after rebuild

This script verifies that the CUDA extension bindings are properly exposed.
Run this after successfully building the extension with: pip install -e .
"""

import sys
import pytest

@pytest.mark.smoke
@pytest.mark.unit
def test_cuda_kernels_import():
    """Test that cuda_kernels module can be imported"""
    try:
        import cuda_kernels
        print("✓ cuda_kernels imported successfully")
        assert True
    except ImportError as e:
        print(f"✗ Failed to import cuda_kernels: {e}")
        print("  Try: from auto_voice import cuda_kernels")
        try:
            from auto_voice import cuda_kernels
            print("✓ auto_voice.cuda_kernels imported successfully")
            assert True
        except ImportError as e2:
            print(f"✗ Failed to import auto_voice.cuda_kernels: {e2}")
            pytest.fail(f"Failed to import cuda_kernels: {e2}")

@pytest.mark.smoke
@pytest.mark.unit
def test_bindings_exposed():
    """Test that the new bindings are exposed"""
    try:
        import cuda_kernels
    except ImportError:
        try:
            from auto_voice import cuda_kernels
        except ImportError:
            print("✗ Cannot import cuda_kernels module")
            pytest.fail("Cannot import cuda_kernels module")

    # Check for the new functions
    functions_to_check = [
        'launch_pitch_detection',
        'launch_vibrato_analysis'
    ]

    missing_functions = []
    for func_name in functions_to_check:
        if hasattr(cuda_kernels, func_name):
            print(f"✓ {func_name} is available")
        else:
            print(f"✗ {func_name} is NOT available")
            missing_functions.append(func_name)

    assert len(missing_functions) == 0, f"Missing functions: {missing_functions}"

@pytest.mark.smoke
@pytest.mark.unit
@pytest.mark.cuda
def test_function_callable():
    """Test that the functions can be called with proper signatures"""
    try:
        import torch
        import cuda_kernels
    except ImportError:
        try:
            import torch
            from auto_voice import cuda_kernels
        except ImportError:
            print("✗ Cannot import required modules")
            pytest.fail("Cannot import required modules")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping callable test")
        pytest.skip("CUDA not available")

    try:
        # Create dummy tensors
        n_samples = 16000
        sample_rate = 16000.0
        frame_length = 2048
        hop_length = 256
        n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

        audio = torch.zeros(n_samples, device='cuda')
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        f0_min = 50.0
        f0_max = 1000.0
        confidence_threshold = 0.7

        # Try to call the function
        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            f0_min, f0_max, confidence_threshold
        )

        print("✓ launch_pitch_detection callable with correct signature")

        # Test vibrato analysis
        vibrato_rate = torch.zeros(n_frames, device='cuda')
        vibrato_depth = torch.zeros(n_frames, device='cuda')

        cuda_kernels.launch_vibrato_analysis(
            output_pitch, vibrato_rate, vibrato_depth,
            hop_length, int(sample_rate)
        )

        print("✓ launch_vibrato_analysis callable with correct signature")
        assert True

    except Exception as e:
        print(f"✗ Function call failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Function call failed: {e}")

@pytest.mark.smoke
@pytest.mark.unit
@pytest.mark.cuda
def test_input_validation():
    """Test that input validation works correctly"""
    try:
        import torch
        import cuda_kernels
    except ImportError:
        try:
            import torch
            from auto_voice import cuda_kernels
        except ImportError:
            print("✗ Cannot import required modules")
            pytest.fail("Cannot import required modules")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping validation test")
        pytest.skip("CUDA not available")

    try:
        n_samples = 16000
        sample_rate = 16000.0
        frame_length = 2048
        hop_length = 256
        n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

        # Test 1: Invalid frame_length
        try:
            audio = torch.zeros(n_samples, device='cuda')
            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            f0_min = 50.0
            f0_max = 1000.0
            confidence_threshold = 0.7

            cuda_kernels.launch_pitch_detection(
                audio, output_pitch, output_confidence, output_vibrato,
                sample_rate, -1, hop_length,  # Invalid frame_length
                f0_min, f0_max, confidence_threshold
            )
            print("✗ Should have raised exception for invalid frame_length")
            pytest.fail("Should have raised exception for invalid frame_length")
        except RuntimeError as e:
            if "frame_length must be > 0" in str(e):
                print("✓ Invalid frame_length raises exception with correct message")
            else:
                print(f"✗ Wrong exception message for frame_length: {e}")
                pytest.fail(f"Wrong exception message for frame_length: {e}")

        # Test 2: CPU tensors
        try:
            audio_cpu = torch.zeros(n_samples)  # CPU tensor
            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            f0_min = 50.0
            f0_max = 1000.0
            confidence_threshold = 0.7

            cuda_kernels.launch_pitch_detection(
                audio_cpu, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                f0_min, f0_max, confidence_threshold
            )
            print("✗ Should have raised exception for CPU tensor")
            pytest.fail("Should have raised exception for CPU tensor")
        except RuntimeError as e:
            if "must be on CUDA device" in str(e):
                print("✓ CPU tensor raises exception with correct message")
            else:
                print(f"✗ Wrong exception message for CPU tensor: {e}")
                pytest.fail(f"Wrong exception message for CPU tensor: {e}")

        # Test 3: Non-contiguous tensors
        try:
            audio = torch.zeros(n_samples, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            f0_min = 50.0
            f0_max = 1000.0
            confidence_threshold = 0.7

            cuda_kernels.launch_pitch_detection(
                audio, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                f0_min, f0_max, confidence_threshold
            )
            print("✗ Should have raised exception for non-contiguous tensor")
            pytest.fail("Should have raised exception for non-contiguous tensor")
        except RuntimeError as e:
            if "must be contiguous" in str(e):
                print("✓ Non-contiguous tensor raises exception with correct message")
            else:
                print(f"✗ Wrong exception message for non-contiguous tensor: {e}")
                pytest.fail(f"Wrong exception message for non-contiguous tensor: {e}")

        # Test 4: Wrong dtype
        try:
            audio = torch.zeros(n_samples, device='cuda', dtype=torch.float64)  # float64 instead of float32
            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            f0_min = 50.0
            f0_max = 1000.0
            confidence_threshold = 0.7

            cuda_kernels.launch_pitch_detection(
                audio, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                f0_min, f0_max, confidence_threshold
            )
            print("✗ Should have raised exception for wrong dtype")
            pytest.fail("Should have raised exception for wrong dtype")
        except RuntimeError as e:
            if "must be float32" in str(e):
                print("✓ Wrong dtype raises exception with correct message")
            else:
                print(f"✗ Wrong exception message for dtype: {e}")
                pytest.fail(f"Wrong exception message for dtype: {e}")

        # Test 5: Vibrato analysis validation
        try:
            pitch_contour = torch.zeros(n_frames, device='cuda')
            vibrato_rate = torch.zeros(n_frames, device='cuda')
            vibrato_depth = torch.zeros(n_frames, device='cuda')

            cuda_kernels.launch_vibrato_analysis(
                pitch_contour, vibrato_rate, vibrato_depth,
                -1, int(sample_rate)  # Invalid hop_length
            )
            print("✗ Should have raised exception for invalid hop_length in vibrato_analysis")
            pytest.fail("Should have raised exception for invalid hop_length in vibrato_analysis")
        except RuntimeError as e:
            if "hop_length must be > 0" in str(e):
                print("✓ Invalid hop_length in vibrato_analysis raises exception")
            else:
                print(f"✗ Wrong exception message for hop_length in vibrato_analysis: {e}")
                pytest.fail(f"Wrong exception message for hop_length in vibrato_analysis: {e}")

        print("✓ All validation tests passed!")
        assert True

    except Exception as e:
        print(f"✗ Validation test failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Validation test failed unexpectedly: {e}")


@pytest.mark.smoke
@pytest.mark.unit
@pytest.mark.cuda
def test_boundary_values():
    """Test boundary values for parameters"""
    try:
        import torch
        import cuda_kernels
    except ImportError:
        try:
            import torch
            from auto_voice import cuda_kernels
        except ImportError:
            print("✗ Cannot import required modules")
            pytest.fail("Cannot import required modules")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping boundary test")
        pytest.skip("CUDA not available")

    try:
        # Test minimum valid parameters
        n_samples = 2048
        sample_rate = 8000.0
        frame_length = 512
        hop_length = 128
        n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

        audio = torch.randn(n_samples, device='cuda')
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        f0_min = 50.0
        f0_max = 1000.0
        confidence_threshold = 0.7

        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            f0_min, f0_max, confidence_threshold
        )
        print("✓ Minimum parameters test passed")

        # Test maximum valid parameters
        n_samples = 441000  # 10 seconds at 44.1kHz
        sample_rate = 44100.0
        frame_length = 4096
        hop_length = 1024
        n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

        audio = torch.randn(n_samples, device='cuda')
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        f0_min = 50.0
        f0_max = 1000.0
        confidence_threshold = 0.7

        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            f0_min, f0_max, confidence_threshold
        )
        print("✓ Maximum parameters test passed")

        # Test single frame
        n_samples = frame_length
        n_frames = 1

        audio = torch.randn(n_samples, device='cuda')
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        f0_min = 50.0
        f0_max = 1000.0
        confidence_threshold = 0.7

        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            f0_min, f0_max, confidence_threshold
        )
        print("✓ Single frame test passed")

        assert True

    except Exception as e:
        print(f"✗ Boundary value test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Boundary value test failed: {e}")


@pytest.mark.smoke
@pytest.mark.unit
@pytest.mark.cuda
def test_stress_large_tensors():
    """Test with large tensors to stress GPU memory"""
    try:
        import torch
        import cuda_kernels
    except ImportError:
        try:
            import torch
            from auto_voice import cuda_kernels
        except ImportError:
            print("✗ Cannot import required modules")
            pytest.fail("Cannot import required modules")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping stress test")
        pytest.skip("CUDA not available")

    try:
        # Test with very large audio (30 seconds at 44.1kHz)
        n_samples = 44100 * 30
        sample_rate = 44100.0
        frame_length = 2048
        hop_length = 512
        n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

        print(f"  Testing with {n_samples} samples ({n_frames} frames)...")

        audio = torch.randn(n_samples, device='cuda')
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Check initial memory
        initial_memory = torch.cuda.memory_allocated()

        f0_min = 50.0
        f0_max = 1000.0
        confidence_threshold = 0.7

        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            f0_min, f0_max, confidence_threshold
        )

        # Check final memory
        final_memory = torch.cuda.memory_allocated()
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)

        print(f"✓ Large tensor test passed (memory increase: {memory_increase:.2f} MB)")
        assert True

    except Exception as e:
        print(f"✗ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Stress test failed: {e}")


@pytest.mark.smoke
@pytest.mark.unit
@pytest.mark.cuda
def test_empty_and_edge_cases():
    """Test empty audio and edge cases"""
    try:
        import torch
        import cuda_kernels
    except ImportError:
        try:
            import torch
            from auto_voice import cuda_kernels
        except ImportError:
            print("✗ Cannot import required modules")
            pytest.fail("Cannot import required modules")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping edge case test")
        pytest.skip("CUDA not available")

    try:
        sample_rate = 16000.0
        frame_length = 2048
        hop_length = 256

        # Test with silent audio (all zeros)
        n_samples = 16000
        n_frames = max(0, (n_samples - frame_length) // hop_length + 1)

        audio = torch.zeros(n_samples, device='cuda')
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        f0_min = 50.0
        f0_max = 1000.0
        confidence_threshold = 0.7

        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            f0_min, f0_max, confidence_threshold
        )

        # Silent audio should have zero pitch
        assert torch.all(output_pitch == 0.0), "Silent audio should have zero pitch"
        print("✓ Silent audio test passed")

        # Test with very low amplitude audio
        audio = torch.randn(n_samples, device='cuda') * 1e-6
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        f0_min = 50.0
        f0_max = 1000.0
        confidence_threshold = 0.7

        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            f0_min, f0_max, confidence_threshold
        )
        print("✓ Low amplitude audio test passed")

        assert True

    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Edge case test failed: {e}")


def main():
    """Run all smoke tests"""
    print("=" * 60)
    print("CUDA Kernel Bindings Smoke Test")
    print("=" * 60)

    test_results = []

    print("\n[1] Testing module import...")
    test_results.append(test_cuda_kernels_import())

    print("\n[2] Testing bindings exposed...")
    test_results.append(test_bindings_exposed())

    print("\n[3] Testing function callable...")
    test_results.append(test_function_callable())

    print("\n[4] Testing input validation...")
    test_results.append(test_input_validation())

    print("\n[5] Testing boundary values...")
    test_results.append(test_boundary_values())

    print("\n[6] Testing stress with large tensors...")
    test_results.append(test_stress_large_tensors())

    print("\n[7] Testing empty and edge cases...")
    test_results.append(test_empty_and_edge_cases())

    print("\n" + "=" * 60)
    if all(test_results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
