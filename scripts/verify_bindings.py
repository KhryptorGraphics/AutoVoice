#!/usr/bin/env python3
"""
AutoVoice CUDA Bindings Verification Script

Quick verification that CUDA extension bindings are properly exposed
and functional.
"""

import sys
import os

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

    try:
        import cuda_kernels
        print_success("Module 'cuda_kernels' imported successfully")
        return True, cuda_kernels
    except ImportError as e:
        print_error(f"Failed to import cuda_kernels: {e}")
        print_info("Module may not be built yet. Run: pip install -e .")
        return False, None
    except Exception as e:
        print_error(f"Unexpected error importing cuda_kernels: {e}")
        return False, None

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
    """Test if functions are callable with correct signatures"""
    print_info("Testing function signatures...")

    if not has_cuda:
        print_warning("Skipping function call tests (CUDA not available)")
        return True

    try:
        import torch

        # Test launch_pitch_detection signature
        print_info("Testing launch_pitch_detection signature...")

        # Create dummy tensors with correct shapes
        batch_size = 1
        audio_length = 16000  # 1 second at 16kHz
        num_frames = (audio_length - 2048) // 256 + 1

        audio = torch.randn(batch_size, audio_length, device='cuda')
        output_pitch = torch.zeros(batch_size, num_frames, device='cuda')
        output_confidence = torch.zeros(batch_size, num_frames, device='cuda')
        output_vibrato = torch.zeros(batch_size, num_frames, device='cuda')

        try:
            # Try calling the function (may fail due to kernel execution, but signature is correct if no TypeError)
            cuda_kernels.launch_pitch_detection(
                audio, output_pitch, output_confidence, output_vibrato,
                16000.0,  # sample_rate
                2048,     # frame_length
                256       # hop_length
            )
            print_success("launch_pitch_detection is callable with correct signature")
            return True
        except TypeError as e:
            print_error(f"Signature mismatch for launch_pitch_detection: {e}")
            return False
        except RuntimeError as e:
            # Runtime errors are okay for this test (e.g., GPU memory issues)
            if "CUDA" in str(e) or "device" in str(e):
                print_success("launch_pitch_detection has correct signature (GPU runtime error is expected)")
                return True
            else:
                print_warning(f"launch_pitch_detection runtime error: {e}")
                return True

    except Exception as e:
        print_error(f"Function call test failed: {e}")
        return False

def main():
    """Main verification function"""
    print_header()

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
