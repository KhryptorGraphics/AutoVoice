"""Smoke test for CUDA kernel bindings after rebuild

This script verifies that the CUDA extension bindings are properly exposed.
Run this after successfully building the extension with: pip install -e .
"""

import sys

def test_cuda_kernels_import():
    """Test that cuda_kernels module can be imported"""
    try:
        import cuda_kernels
        print("✓ cuda_kernels imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import cuda_kernels: {e}")
        print("  Try: from auto_voice import cuda_kernels")
        try:
            from auto_voice import cuda_kernels
            print("✓ auto_voice.cuda_kernels imported successfully")
            return True
        except ImportError as e2:
            print(f"✗ Failed to import auto_voice.cuda_kernels: {e2}")
            return False

def test_bindings_exposed():
    """Test that the new bindings are exposed"""
    try:
        import cuda_kernels
    except ImportError:
        try:
            from auto_voice import cuda_kernels
        except ImportError:
            print("✗ Cannot import cuda_kernels module")
            return False

    # Check for the new functions
    functions_to_check = [
        'launch_pitch_detection',
        'launch_vibrato_analysis'
    ]

    all_present = True
    for func_name in functions_to_check:
        if hasattr(cuda_kernels, func_name):
            print(f"✓ {func_name} is available")
        else:
            print(f"✗ {func_name} is NOT available")
            all_present = False

    return all_present

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
            return False

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping callable test")
        return True

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

        # Try to call the function
        cuda_kernels.launch_pitch_detection(
            audio, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length
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
        return True

    except Exception as e:
        print(f"✗ Function call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

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

    print("\n" + "=" * 60)
    if all(test_results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
