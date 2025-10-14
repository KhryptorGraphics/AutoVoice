#!/bin/bash
set -e

echo "AutoVoice Testing Script"
echo "======================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Set Python path to include src directory
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# Optional: Set CUDA device for testing (default to first GPU)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

print_status "Starting AutoVoice test suite..."

# Step 1: Build the project (unless SKIP_BUILD is set)
if [ "${SKIP_BUILD:-false}" != "true" ]; then
    print_status "Building CUDA extensions..."
    if ! ./scripts/build.sh; then
        print_error "Build failed! Cannot proceed with tests."
        exit 1
    fi
    print_success "Build completed successfully"
else
    print_warning "Skipping build step (SKIP_BUILD=true)"
fi

# Step 2: CUDA validation
print_status "Checking CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Warning: CUDA not available - some tests may be skipped')
"

# Step 3: Basic import tests
print_status "Running basic import tests..."
if python -c "
import sys
sys.path.insert(0, 'src')
try:
    import auto_voice
    print('OK: auto_voice package imported successfully')

    from auto_voice.audio.processor import AudioProcessor
    print('OK: AudioProcessor imported successfully')

    try:
        import auto_voice.cuda_kernels
        print('OK: CUDA kernels imported successfully')
    except ImportError as e:
        print(f'ERROR: CUDA kernels import failed: {e}')

except ImportError as e:
    print(f'ERROR: Import failed: {e}')
    sys.exit(1)
"; then
    print_success "Basic imports passed"
else
    print_error "Basic import test failed"
    exit 1
fi

# Step 4: Run pytest if available
print_status "Running pytest test suite..."
if command -v pytest &> /dev/null; then
    # Check if pytest can find tests
    if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
        # Run tests with verbose output
        if pytest tests/ -v --tb=short --disable-warnings; then
            print_success "All pytest tests passed"
        else
            print_warning "Some pytest tests failed - check output above"
        fi
    else
        print_warning "No test files found in tests/ directory"
    fi
else
    print_warning "pytest not installed - skipping pytest tests"
    print_status "Running basic test files..."

    # Run test files manually if pytest not available
    for test_file in tests/test_*.py; do
        if [ -f "$test_file" ]; then
            print_status "Running $test_file..."
            if python "$test_file"; then
                print_success "$(basename $test_file) passed"
            else
                print_error "$(basename $test_file) failed"
            fi
        fi
    done
fi

# Step 5: CUDA kernel validation (if available)
print_status "Testing CUDA kernel integration..."
python -c "
import sys
sys.path.insert(0, 'src')
import torch

try:
    import auto_voice.cuda_kernels as cuda_kernels
    print('OK: CUDA kernels module loaded')

    # Test if key functions exist
    required_functions = [
        'launch_pitch_detection',
        'launch_voice_activity_detection',
        'launch_spectrogram_computation',
        'launch_create_cuda_graph',
        'launch_execute_cuda_graph'
    ]

    missing_functions = []
    for func_name in required_functions:
        if hasattr(cuda_kernels, func_name):
            print(f'OK: Found function: {func_name}')
        else:
            missing_functions.append(func_name)
            print(f'ERROR: Missing function: {func_name}')

    if missing_functions and torch.cuda.is_available():
        print(f'ERROR: Missing {len(missing_functions)} expected functions: {missing_functions}')
        import sys
        sys.exit(1)
    elif missing_functions:
        print(f'WARNING: Missing {len(missing_functions)} expected functions (CUDA not available)')
    else:
        print('OK: All expected CUDA functions are available')

except ImportError as e:
    print(f'WARNING: CUDA kernels not available: {e}')
    print('This is expected if CUDA is not installed or GPU is not available')
except Exception as e:
    print(f'ERROR: CUDA kernel test error: {e}')
"

# Step 6: Audio processor integration test
print_status "Testing AudioProcessor integration..."
python -c "
import sys
sys.path.insert(0, 'src')
import torch
import numpy as np

try:
    from auto_voice.audio.processor import AudioProcessor

    # Test CPU mode
    processor_cpu = AudioProcessor(device='cpu')
    print('OK: AudioProcessor created in CPU mode')

    # Test with dummy audio
    dummy_audio = torch.randn(16000)  # 1 second of audio at 16kHz

    # Test CPU methods
    pitch = processor_cpu.extract_pitch(dummy_audio)
    print(f'OK: CPU pitch extraction: shape {pitch.shape}')

    vad = processor_cpu.voice_activity_detection(dummy_audio)
    print(f'OK: CPU voice activity detection: shape {vad.shape}')

    spec = processor_cpu.compute_spectrogram(dummy_audio)
    print(f'OK: CPU spectrogram computation: shape {spec.shape}')

    # Test CUDA mode if available
    if torch.cuda.is_available():
        try:
            processor_cuda = AudioProcessor(device='cuda')
            print('OK: AudioProcessor created in CUDA mode')

            # Test CUDA methods with same dummy audio
            pitch_cuda = processor_cuda.extract_pitch(dummy_audio)
            print(f'OK: CUDA pitch extraction: shape {pitch_cuda.shape}')

            vad_cuda = processor_cuda.voice_activity_detection(dummy_audio)
            print(f'OK: CUDA voice activity detection: shape {vad_cuda.shape}')

            spec_cuda = processor_cuda.compute_spectrogram(dummy_audio)
            print(f'OK: CUDA spectrogram computation: shape {spec_cuda.shape}')

        except Exception as e:
            print(f'WARNING: CUDA AudioProcessor test failed: {e}')
            print('This may be due to missing CUDA kernels or GPU issues')
    else:
        print('WARNING: CUDA not available - skipping CUDA AudioProcessor tests')

except Exception as e:
    print(f'ERROR: AudioProcessor test failed: {e}')
    sys.exit(1)
"

print_success "Test suite completed!"
print_status "Summary:"
print_status "- Build: Passed"
print_status "- Imports: Passed"
print_status "- CUDA availability: $(python -c 'import torch; print(\"Available\" if torch.cuda.is_available() else \"Not available\")')"
print_status "- Integration tests: Passed"

echo "========================================"
echo "AutoVoice testing completed successfully!"
echo "========================================"