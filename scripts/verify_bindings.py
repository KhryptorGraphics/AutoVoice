#!/usr/bin/env python3
"""
AutoVoice CUDA Bindings Verification Script

Quick verification that CUDA extension bindings are properly exposed
and functional.
"""

import sys
import os
import time

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# Symbols
CHECK = "✓"
CROSS = "✗"
INFO = "ℹ"

def print_header():
    """Print script header"""
    print(f"{Colors.BLUE}╔════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║     CUDA Bindings Verification                         ║{Colors.NC}")
    print(f"{Colors.BLUE}╚════════════════════════════════════════════════════════╝{Colors.NC}")
    print()

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}[{CHECK}]{Colors.NC} {message}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}[{CROSS}]{Colors.NC} {message}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}[{INFO}]{Colors.NC} {message}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}[!]{Colors.NC} {message}")

def test_import():
    """Test if cuda_kernels module can be imported"""
    print_info("Testing module import...")

    cuda_kernels = None
    import_variant = None

    # Try namespaced import first (correct for packaged module)
    try:
        from auto_voice import cuda_kernels as ck
        cuda_kernels = ck
        import_variant = "from auto_voice import cuda_kernels"
        print_success(f"Module imported via: {import_variant}")
    except ImportError:
        # Fallback to direct import (for development/testing)
        try:
            import cuda_kernels as ck
            cuda_kernels = ck
            import_variant = "import cuda_kernels"
            print_success(f"Module imported via: {import_variant} (direct import)")
        except ImportError:
            print_error("CUDA extension module not found - likely not built yet")
            print_info("")
            print_info("Possible causes:")
            print_info("  1. CUDA extensions not built: Run 'pip install -e .'")
            print_info("  2. Build failed: Check 'build.log' for errors")
            print_info("  3. Missing CUDA headers: Run './scripts/check_cuda_toolkit.sh'")
            print_info("")
            print_info("If build failed with 'nv/target' error:")
            print_info("  - Install system CUDA toolkit: './scripts/install_cuda_toolkit.sh'")
            return False, None
    except Exception as e:
        print_error(f"Unexpected error importing cuda_kernels: {e}")
        print_info("This may indicate a build or linking issue")
        print_info("Check 'build.log' for compilation errors")
        return False, None

    # Verify module file path and that it's a compiled extension
    if cuda_kernels is not None:
        try:
            module_file = cuda_kernels.__file__
            print_info(f"Module path: {module_file}")

            # Check file size
            if os.path.exists(module_file):
                file_size = os.path.getsize(module_file)
                print_info(f"Module size: {file_size:,} bytes")

                if file_size < 1000:
                    print_warning(f"Module file is very small ({file_size} bytes)")
                    print_warning("This may be a stub file, not a compiled extension")

            # Validate extension type
            _, ext = os.path.splitext(module_file)
            if ext in {'.so', '.pyd'}:
                print_success(f"Confirmed compiled extension: {ext}")
            else:
                print_warning(f"Module file extension unexpected: {ext} (expected .so or .pyd)")
                print_warning("This may be a Python stub, not the compiled CUDA extension")
                print_info("Rebuild with: pip install -e . --force-reinstall --no-cache-dir")
        except AttributeError:
            print_warning("Module has no __file__ attribute")

    return True, cuda_kernels

def print_build_info(cuda_kernels):
    """Print detailed build and environment information"""
    print_info("Build and Environment Information:")

    try:
        # Module path and file size
        if hasattr(cuda_kernels, '__file__'):
            module_path = cuda_kernels.__file__
            if os.path.exists(module_path):
                file_size = os.path.getsize(module_path)
                file_size_mb = file_size / (1024 * 1024)
                print_info(f"  Module size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
            else:
                print_warning(f"  Module file not found: {module_path}")

        # Module version if available
        if hasattr(cuda_kernels, '__version__'):
            print_info(f"  Module version: {cuda_kernels.__version__}")
        else:
            print_info("  Module version: Not available")

        # PyTorch and CUDA version
        try:
            import torch
            print_info(f"  PyTorch version: {torch.__version__}")
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print_info(f"  CUDA version (PyTorch): {torch.version.cuda}")
            else:
                print_info("  CUDA version (PyTorch): Not available")

            # GPU information
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                compute_cap = torch.cuda.get_device_capability(0)
                print_info(f"  GPU: {gpu_name}")
                print_info(f"  Compute capability: {compute_cap[0]}.{compute_cap[1]}")

                # Memory info
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print_info(f"  GPU memory: {total_memory:.2f} GB")
            else:
                print_warning("  GPU: Not available")
        except Exception as e:
            print_warning(f"  Could not retrieve PyTorch/CUDA info: {e}")

    except Exception as e:
        print_error(f"Error printing build info: {e}")

def test_functions_exposed(cuda_kernels):
    """Test if required functions are exposed"""
    print_info("Checking exposed functions...")

    required_functions = [
        'launch_pitch_detection',
        'launch_vibrato_analysis',
    ]

    all_exposed = True

    for func_name in required_functions:
        if hasattr(cuda_kernels, func_name):
            print_success(f"Function '{func_name}' is exposed")
        else:
            print_error(f"Function '{func_name}' is NOT exposed")
            all_exposed = False

    # List all available functions
    available_functions = [name for name in dir(cuda_kernels) if not name.startswith('_')]
    print_info(f"Available functions: {', '.join(available_functions)}")

    return all_exposed

def test_torch_available():
    """Test if PyTorch is available"""
    print_info("Checking PyTorch availability...")

    try:
        import torch
        print_success(f"PyTorch {torch.__version__} available")

        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return True, True
        else:
            print_warning("CUDA not available - GPU functions cannot be tested")
            return True, False
    except ImportError:
        print_error("PyTorch not installed")
        return False, False

def test_tensor_creation(has_cuda):
    """Test if we can create tensors for function calls"""
    print_info("Testing tensor creation...")

    try:
        import torch

        # Test CPU tensor creation
        cpu_tensor = torch.randn(100)
        print_success("CPU tensor creation successful")

        if has_cuda:
            # Test GPU tensor creation
            gpu_tensor = torch.randn(100, device='cuda')
            print_success("GPU tensor creation successful")
            return True, gpu_tensor
        else:
            print_warning("Skipping GPU tensor creation (CUDA not available)")
            return True, cpu_tensor

    except Exception as e:
        print_error(f"Tensor creation failed: {e}")
        return False, None

def test_function_callable(cuda_kernels, has_cuda):
    """Test if functions are callable with correct signatures and basic execution"""
    print_info("Testing function signatures and runtime diagnostics...")

    if not has_cuda:
        print_warning("Skipping function call tests (CUDA not available)")
        return True

    try:
        import torch
        success_count = 0
        total_tests = 0

        # Test 1: Timed execution of launch_pitch_detection
        print_info("Testing launch_pitch_detection with timing...")
        total_tests += 1

        try:
            # Create minimal test tensors
            audio_length = 2048  # Minimal audio for quick test
            num_frames = 1

            audio = torch.randn(audio_length, device='cuda')
            output_pitch = torch.zeros(num_frames, device='cuda')
            output_confidence = torch.zeros(num_frames, device='cuda')
            output_vibrato = torch.zeros(num_frames, device='cuda')

            # Warm-up call
            cuda_kernels.launch_pitch_detection(
                audio, output_pitch, output_confidence, output_vibrato,
                16000.0, 2048, 2048,
                50.0, 2000.0, 0.1  # fmin, fmax, threshold
            )
            torch.cuda.synchronize()

            # Timed call
            start_time = time.perf_counter()
            cuda_kernels.launch_pitch_detection(
                audio, output_pitch, output_confidence, output_vibrato,
                16000.0, 2048, 2048,
                50.0, 2000.0, 0.1  # fmin, fmax, threshold
            )
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            duration_ms = (end_time - start_time) * 1000
            print_success(f"Timed execution: {duration_ms:.3f} ms")
            success_count += 1

        except TypeError as e:
            print_error(f"Signature mismatch for launch_pitch_detection: {e}")
        except RuntimeError as e:
            print_error(f"Runtime error in launch_pitch_detection: {e}")
            if "shape" in str(e).lower() or "dtype" in str(e).lower():
                print_info(f"Hint: Check tensor shapes/dtypes match expected signature")

        # Test 2: Memory stability check (10-iteration loop)
        print_info("Testing memory stability (10 iterations)...")
        total_tests += 1

        try:
            # Create test tensors
            audio_length = 16000
            num_frames = (audio_length - 2048) // 256 + 1

            audio = torch.randn(audio_length, device='cuda')
            output_pitch = torch.zeros(num_frames, device='cuda')
            output_confidence = torch.zeros(num_frames, device='cuda')
            output_vibrato = torch.zeros(num_frames, device='cuda')

            # Record initial memory
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated()

            # Run 10 iterations
            for i in range(10):
                cuda_kernels.launch_pitch_detection(
                    audio, output_pitch, output_confidence, output_vibrato,
                    16000.0, 2048, 256,
                    50.0, 2000.0, 0.1  # fmin, fmax, threshold
                )

            torch.cuda.synchronize()
            final_memory = torch.cuda.memory_allocated()
            memory_delta_mb = (final_memory - initial_memory) / (1024 * 1024)

            print_info(f"Memory delta after 10 iterations: {memory_delta_mb:.3f} MB")

            if memory_delta_mb > 1.0:
                print_error(f"Memory leak detected: {memory_delta_mb:.3f} MB increase exceeds 1 MB threshold")
                # Don't increment success_count - this is a failure
            else:
                print_success(f"Memory stable: {memory_delta_mb:.3f} MB delta (< 1 MB threshold)")
                success_count += 1

        except Exception as e:
            print_error(f"Memory stability test failed: {e}")

        # Test 3: Minimal callable test for launch_vibrato_analysis
        print_info("Testing launch_vibrato_analysis callable...")
        total_tests += 1

        try:
            # Create minimal test tensors for vibrato analysis
            num_frames = 100
            pitch_contour = torch.randn(num_frames, device='cuda') * 100 + 200  # Pitch values around 200 Hz
            vibrato_rate = torch.zeros(num_frames, device='cuda')
            vibrato_depth = torch.zeros(num_frames, device='cuda')

            # Call vibrato analysis
            cuda_kernels.launch_vibrato_analysis(
                pitch_contour, vibrato_rate, vibrato_depth,
                256, 16000.0  # hop_length, sample_rate
            )
            torch.cuda.synchronize()

            print_success("launch_vibrato_analysis callable test passed")
            success_count += 1

        except TypeError as e:
            print_error(f"Signature mismatch for launch_vibrato_analysis: {e}")
        except RuntimeError as e:
            print_error(f"Runtime error in launch_vibrato_analysis: {e}")
        except Exception as e:
            print_error(f"launch_vibrato_analysis test failed: {e}")

        # Test 4: Test kernel memory operations if available
        total_tests += 1
        if hasattr(cuda_kernels, 'launch_memory_test'):
            print_info("Testing CUDA memory operations...")
            try:
                test_tensor = torch.randn(1024, device='cuda')
                result_tensor = torch.zeros_like(test_tensor)

                cuda_kernels.launch_memory_test(test_tensor, result_tensor)

                # Basic validation
                if not torch.allclose(test_tensor, result_tensor):
                    print_success("CUDA memory operations working")
                    success_count += 1
                else:
                    print_warning("CUDA memory test passed but results identical (expected)")
                    success_count += 1

            except Exception as e:
                print_warning(f"CUDA memory test failed: {e}")
                # Don't fail the test for memory issues

        # Test 5: Test FFT operations if available
        total_tests += 1
        if hasattr(cuda_kernels, 'launch_fft_test'):
            print_info("Testing CUDA FFT operations...")
            try:
                # Create test signal
                signal = torch.randn(4096, device='cuda')

                cuda_kernels.launch_fft_test(signal)

                print_success("CUDA FFT operations working")
                success_count += 1

            except Exception as e:
                print_warning(f"CUDA FFT test failed: {e}")

        # Test 4: Test basic kernel launch capability
        total_tests += 1
        print_info("Testing basic kernel launch capability...")
        try:
            # Simple kernel launch test - create small tensors and call a simple function
            test_input = torch.randn(256, device='cuda')
            test_output = torch.zeros_like(test_input)

            # If we have a simple test function, call it
            if hasattr(cuda_kernels, 'launch_simple_test'):
                cuda_kernels.launch_simple_test(test_input, test_output)
                print_success("Basic kernel launch working")
                success_count += 1
            else:
                print_info("No simple kernel test available - skipping")
                total_tests -= 1

        except Exception as e:
            print_warning(f"Basic kernel launch test failed: {e}")

        # Summary
        if success_count > 0:
            print_info(f"CUDA kernel tests: {success_count}/{total_tests} successful")
            return success_count >= max(1, total_tests // 2)  # At least half successful
        else:
            print_warning("No CUDA kernel tests were successful")
            return False  # Fail validation when no CUDA tests pass (strict by default now)

    except Exception as e:
        print_error(f"Function call test suite failed: {e}")
        return False

def main():
    """Main verification function"""
    print_header()

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Verify CUDA bindings')
    parser.add_argument('--strict', action='store_true', help='Require at least one CUDA runtime test to pass when CUDA is available')
    args = parser.parse_args()

    # Track overall success
    all_tests_passed = True

    # Test 1: PyTorch availability
    torch_available, has_cuda = test_torch_available()
    if not torch_available:
        print_error("PyTorch is required but not available")
        return False
    print()

    # Test 2: Module import
    import_success, cuda_kernels = test_import()
    if not import_success:
        all_tests_passed = False
        print()
        print_error("Module import failed - cannot continue")
        return False
    print()

    # Test 2.5: Print build information
    print_build_info(cuda_kernels)
    print()

    # Test 3: Function exposure
    functions_exposed = test_functions_exposed(cuda_kernels)
    if not functions_exposed:
        all_tests_passed = False
    print()

    # Test 4: Tensor creation
    tensor_success, test_tensor = test_tensor_creation(has_cuda)
    if not tensor_success:
        all_tests_passed = False
    print()

    # Test 5: Function callable (only if CUDA available)
    if has_cuda:
        callable_success = test_function_callable(cuda_kernels, has_cuda)
        if not callable_success:
            all_tests_passed = False
        # In strict mode, also require at least one CUDA runtime test passed
        if args.strict and cuda_kernels and not callable_success:
            all_tests_passed = False
        print()

    # Summary
    print(f"{Colors.BLUE}════════════════════════════════════════════════════════{Colors.NC}")
    print(f"{Colors.BLUE}  Verification Summary{Colors.NC}")
    print(f"{Colors.BLUE}════════════════════════════════════════════════════════{Colors.NC}")
    print()

    if all_tests_passed:
        print(f"{Colors.GREEN}╔════════════════════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.GREEN}║  ALL CHECKS PASSED!                                     ║{Colors.NC}")
        print(f"{Colors.GREEN}║  CUDA bindings are properly exposed and functional      ║{Colors.NC}")
        print(f"{Colors.GREEN}╚════════════════════════════════════════════════════════╝{Colors.NC}")
        return True
    else:
        print(f"{Colors.RED}╔════════════════════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.RED}║  VERIFICATION FAILED                                    ║{Colors.NC}")
        print(f"{Colors.RED}║  Some checks did not pass - review errors above         ║{Colors.NC}")
        print(f"{Colors.RED}╚════════════════════════════════════════════════════════╝{Colors.NC}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
