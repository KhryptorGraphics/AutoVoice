from setuptools import setup, Extension, find_packages
import os
import sys

def _get_long_description():
    """Get long description from README file with fallback"""
    readme_files = ['README.md', 'docs/README.md', 'src/README.md']
    for readme_file in readme_files:
        if os.path.exists(readme_file):
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                continue
    return 'GPU-accelerated voice synthesis system with CUDA acceleration'

def _wants_cuda_build(argv):
    """
    Determine if the current command requires building CUDA extensions.

    Returns True for build commands (build_ext, install, develop, bdist_wheel, etc.)
    Returns False for metadata-only commands (egg_info, sdist, dist_info, etc.)

    Set AUTO_VOICE_NO_CUDA=1 environment variable to skip CUDA build entirely.
    """
    # Allow explicit skip of CUDA build via environment variable
    if os.environ.get('AUTO_VOICE_NO_CUDA', '').lower() in ('1', 'true', 'yes'):
        return False

    # Commands that require building extensions
    build_commands = {
        'build_ext', 'build', 'install', 'develop',
        'bdist_wheel', 'bdist_egg', 'editable_wheel'
    }

    # Commands that only need metadata (no build required)
    metadata_commands = {
        'egg_info', 'sdist', 'dist_info', 'clean', '--version', '--help'
    }

    # Check if any build command is in argv
    for arg in argv:
        if arg in build_commands:
            return True
        if arg in metadata_commands:
            return False

    # Default: if no recognized command, assume build is needed
    # (e.g., `pip install .` may not pass explicit commands)
    return True

def _validate_cuda_environment():
    """
    Comprehensive validation of CUDA build environment.

    Performs pre-build checks to ensure CUDA extensions can be built successfully.
    Returns (is_valid, error_messages) tuple.
    """
    import os
    import subprocess
    import glob
    from ctypes.util import find_library

    errors = []
    warnings = []

    # Check if CUFFT is disabled for CPU-only builds
    disable_cufft = os.environ.get('DISABLE_CUFFT') == '1'
    if disable_cufft:
        print("CUDA environment validation: CUFFT disabled (CPU-only mode)")
        return True, [], ["CUFFT disabled - only CPU-based operations will be available"]

    # Check for system CUDA toolkit - avoid using CONDA_PREFIX as fallback
    # Allow explicit override via AUTO_VOICE_CUDA_HOME
    cuda_home = os.environ.get('AUTO_VOICE_CUDA_HOME') or os.environ.get('CUDA_HOME') or '/usr/local/cuda'

    # Check nvcc availability
    try:
        nvcc_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if nvcc_result.returncode != 0:
            errors.append("nvcc compiler not found or not working")
        else:
            # Check CUDA version compatibility
            nvcc_output = nvcc_result.stdout
            if "release" in nvcc_output:
                cuda_version_line = [line for line in nvcc_output.split('\n') if "release" in line][0]
                # Extract version from "release X.Y" format (handles both "release 12.8, V12.8.93" and "release 12.8")
                release_parts = cuda_version_line.split("release")
                if len(release_parts) > 1:
                    version_part = release_parts[1].strip().split(',')[0].split()[0]  # Get first part before comma/space
                    cuda_version = version_part
                    cuda_major = int(cuda_version.split('.')[0])
                    cuda_minor = int(cuda_version.split('.')[1])

                    if cuda_major < 11 or (cuda_major == 11 and cuda_minor < 8):
                        errors.append(f"CUDA version {cuda_version} is too old (minimum: 11.8)")
                    else:
                        print(f"  ✓ CUDA compiler version {cuda_version} detected")
                else:
                    warnings.append("Could not parse CUDA version from nvcc output")
            else:
                warnings.append("Could not parse CUDA version from nvcc output")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        errors.append("nvcc command failed - check CUDA toolkit installation")

    # Check critical CUDA headers
    # For nv/target, check both system and conda layouts
    nv_target_paths = [
        os.path.join(cuda_home, 'include', 'nv', 'target'),  # Standard layout
    ]
    # Also check targets/*/include/nv/target for conda-style layouts
    targets_glob = glob.glob(os.path.join(cuda_home, 'targets', '*', 'include', 'nv', 'target'))
    nv_target_paths.extend(targets_glob)

    nv_target_found = False
    nv_target_resolved_path = None
    for nv_path in nv_target_paths:
        if os.path.exists(nv_path):
            nv_target_found = True
            nv_target_resolved_path = nv_path
            print(f"  ✓ Found nv/target at: {nv_path}")
            break

    if not nv_target_found:
        errors.append(f"Critical header directory missing: nv/target (required for CUDA extensions). Checked: {nv_target_paths}")

        # Check if conda CUDA toolkit is installed
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix and os.path.exists(os.path.join(conda_prefix, 'include', 'cuda.h')):
            errors.append("Conda CUDA toolkit detected but incomplete (missing nv/target header)")
            errors.append("Solution: Install system CUDA toolkit with complete headers")
            errors.append("Run: ./scripts/install_cuda_toolkit.sh")
        else:
            errors.append("No CUDA toolkit found with required headers")
            errors.append("Install system CUDA toolkit: ./scripts/install_cuda_toolkit.sh")

    # Check other critical headers - cufft.h is build-blocking unless disabled
    # Check cufft.h separately as it's required for CUDA extensions
    cufft_header = os.path.join(cuda_home, 'include', 'cufft.h')
    if os.environ.get('DISABLE_CUFFT') != '1' and not os.path.isfile(cufft_header):
        errors.append(f"Critical header missing: cufft.h (required for CUDA FFT operations)")
        errors.append(f"Checked: {cufft_header}")
        errors.append("Solution: Install the full CUDA toolkit that includes cuFFT headers")
        errors.append("Run: ./scripts/install_cuda_toolkit.sh")
        errors.append("Or set DISABLE_CUFFT=1 for CPU-only operations")

    # Check other important headers (warnings only)

    # Check other important headers (warnings only)
    other_headers = [
        os.path.join(cuda_home, 'include', 'cuda.h'),
        os.path.join(cuda_home, 'include', 'cuda_runtime.h'),
        os.path.join(cuda_home, 'include', 'device_launch_parameters.h'),
    ]

    missing_headers = []
    for header in other_headers:
        if not os.path.isfile(header):
            missing_headers.append(header)

    if missing_headers:
        warnings.append(f"Some CUDA headers missing: {missing_headers}")

    # Check CUDA libraries using glob patterns to accept versioned libraries
    # Support both lib64 (system) and lib (conda) directories
    lib_dirs = []
    lib64_dir = os.path.join(cuda_home, 'lib64')
    lib_dir = os.path.join(cuda_home, 'lib')

    if os.path.isdir(lib64_dir):
        lib_dirs.append(lib64_dir)
    if os.path.isdir(lib_dir):
        lib_dirs.append(lib_dir)

    # Check for libcudart (accept versioned libraries like libcudart.so.12)
    cudart_found = False
    for lib_path in lib_dirs:
        if glob.glob(os.path.join(lib_path, 'libcudart.so*')):
            cudart_found = True
            break
    if not cudart_found:
        # Fallback to ctypes.util.find_library
        cudart_found = find_library('cudart') is not None

    # Check for libcufft (accept versioned libraries like libcufft.so.11) unless disabled
    cufft_found = False
    if os.environ.get('DISABLE_CUFFT') != '1':
        for lib_path in lib_dirs:
            if glob.glob(os.path.join(lib_path, 'libcufft.so*')):
                cufft_found = True
                break
        if not cufft_found:
            # Fallback to ctypes.util.find_library
            cufft_found = find_library('cufft') is not None

    missing_libs = []
    if not cudart_found:
        missing_libs.append('libcudart.so (or versioned variant)')
    if os.environ.get('DISABLE_CUFFT') != '1' and not cufft_found:
        missing_libs.append('libcufft.so (or versioned variant)')

    if missing_libs:
        errors.append(f"Critical CUDA libraries missing: {missing_libs}")

    return len(errors) == 0, errors, warnings

def _build_cuda_extensions():
    """
    Build CUDA extensions if PyTorch with CUDA is available.

    This function is only called when actually building extensions (not during metadata operations).
    Performs comprehensive pre-build validation and provides detailed error messages.
    Returns (ext_modules, cmdclass) tuple.
    """
    # Check if CUDA is required (fail fast instead of silent fallback)
    require_cuda = os.environ.get('REQUIRE_CUDA') == '1'

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        if require_cuda:
            print("=" * 80)
            print("ERROR: PyTorch with CUDA is required but not installed")
            print("=" * 80)
            print("")
            print("REQUIRE_CUDA=1 is set, but PyTorch is not available.")
            print("")
            print("Install PyTorch with CUDA support:")
            print("  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \\")
            print("    --index-url https://download.pytorch.org/whl/cu121")
            print("")
            print("Or use the provided script:")
            print("  ./scripts/setup_pytorch_env.sh")
            print("")
            print("=" * 80)
            raise SystemExit(1)

        print("=" * 80)
        print("WARNING: PyTorch is not installed - skipping CUDA extensions")
        print("=" * 80)
        print("")
        print("AutoVoice requires PyTorch with CUDA support for GPU acceleration.")
        print("")
        print("To enable CUDA extensions:")
        print("  1. Install PyTorch with CUDA support:")
        print("     pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \\")
        print("       --index-url https://download.pytorch.org/whl/cu121")
        print("")
        print("     OR install using the provided script:")
        print("     ./scripts/setup_pytorch_env.sh")
        print("")
        print("For detailed installation instructions, see:")
        print("  - requirements.txt (header section)")
        print("  - PYTORCH_ENVIRONMENT_FIX_REPORT.md")
        print("")
        print("Continuing with CPU-only installation (no CUDA extensions)...")
        print("=" * 80)
        return [], {}

    # Check PyTorch CUDA availability
    if not torch.cuda.is_available():
        if require_cuda:
            print("=" * 80)
            print("ERROR: CUDA is required but not available in PyTorch")
            print("=" * 80)
            print("")
            print("REQUIRE_CUDA=1 is set, but torch.cuda.is_available() returned False.")
            print("")
            print("This could be caused by:")
            print("  1. Missing NVIDIA GPU")
            print("  2. Incorrect PyTorch installation (CPU-only version installed)")
            print("  3. Missing or incompatible NVIDIA drivers")
            print("")
            print("To fix:")
            print("  1. Check GPU: nvidia-smi")
            print("  2. Reinstall PyTorch with CUDA:")
            print("     pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \\")
            print("       --index-url https://download.pytorch.org/whl/cu121")
            print("")
            print("=" * 80)
            raise SystemExit(1)

        print("=" * 80)
        print("WARNING: PyTorch CUDA support not available - skipping CUDA extensions")
        print("=" * 80)
        print("")
        print("PyTorch is installed but CUDA is not available.")
        print("This could be caused by:")
        print("")
        print("  1. Missing NVIDIA GPU")
        print("  2. Incorrect PyTorch installation (need CUDA version)")
        print("  3. Missing or incompatible NVIDIA drivers")
        print("  4. CUDA toolkit not properly installed")
        print("")
        print("To enable CUDA extensions:")
        print("")
        print("  Option A - Check GPU and drivers:")
        print("    nvidia-smi")
        print("    ./scripts/check_cuda_toolkit.sh")
        print("")
        print("  Option B - Reinstall PyTorch with CUDA:")
        print("    pip uninstall torch torchvision torchaudio")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("")
        print("  Option C - Full CUDA environment setup:")
        print("    ./scripts/install_cuda_toolkit.sh")
        print("")
        print("Continuing with CPU-only installation (no CUDA extensions)...")
        print("=" * 80)
        return [], {}

    # Perform comprehensive CUDA environment validation
    print("Validating CUDA build environment...")
    cuda_valid, cuda_errors, cuda_warnings = _validate_cuda_environment()

    if cuda_warnings:
        print("=" * 60)
        print("WARNING: CUDA Environment Issues Detected")
        print("=" * 60)
        for warning in cuda_warnings:
            print(f"  ⚠ {warning}")
        print("")

    if not cuda_valid:
        if require_cuda:
            print("=" * 80)
            print("ERROR: CUDA environment validation failed (REQUIRE_CUDA=1)")
            print("=" * 80)
            print("")
            print("The following CUDA build requirements are not met:")
            print("")
            for error in cuda_errors:
                print(f"  ✗ {error}")
            print("")
            print("To fix these issues:")
            print("  1. Install CUDA toolkit: ./scripts/install_cuda_toolkit.sh")
            print("  2. Verify installation: ./scripts/check_cuda_toolkit.sh")
            print("")
            print("=" * 80)
            raise SystemExit(1)

        print("=" * 80)
        print("ERROR: CUDA Environment Validation Failed")
        print("=" * 80)
        print("")
        print("The following CUDA build requirements are not met:")
        print("")
        for error in cuda_errors:
            print(f"  ✗ {error}")
        print("")
        print("To fix these issues:")
        print("")
        if any("nvcc" in err for err in cuda_errors):
            print("  1. Install/reinstall CUDA toolkit:")
            print("     ./scripts/install_cuda_toolkit.sh")
            print("     OR visit: https://developer.nvidia.com/cuda-toolkit")
            print("")
        if any("header" in err.lower() for err in cuda_errors):
            print("  2. Install system CUDA toolkit with complete headers:")
            print("     ./scripts/install_cuda_toolkit.sh")
            print("")
            if any("nv/target" in err for err in cuda_errors):
                print("     NOTE: Conda CUDA toolkit is often incomplete")
                print("     System CUDA toolkit installation required for building extensions")
                print("")
        if any("library" in err.lower() for err in cuda_errors):
            print("  3. Check CUDA library paths:")
            print("     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
            print("     sudo ldconfig")
            print("")
        print("  4. Verify complete CUDA installation:")
        print("     ./scripts/check_cuda_toolkit.sh")
        print("")
        print("Continuing with CPU-only build (no CUDA extensions)...")
        print("=" * 80)

        return [], {}

    # CUDA home directory - avoid using CONDA_PREFIX as fallback
    # Allow explicit override via AUTO_VOICE_CUDA_HOME
    CUDA_HOME = os.environ.get('AUTO_VOICE_CUDA_HOME') or os.environ.get('CUDA_HOME') or '/usr/local/cuda'

    # Determine actual CUDA paths from where headers were found during validation
    # This ensures we use the correct include/lib directories
    import glob

    # Find the base CUDA directory from nv/target location
    cuda_base_dir = CUDA_HOME
    nv_target_search_paths = [
        os.path.join(CUDA_HOME, 'include', 'nv', 'target'),
    ]
    # Also check targets/*/include/nv/target for conda-style layouts
    nv_target_search_paths.extend(glob.glob(os.path.join(CUDA_HOME, 'targets', '*', 'include', 'nv', 'target')))

    # Find where nv/target actually exists
    resolved_include_base = None
    for nv_path in nv_target_search_paths:
        if os.path.exists(nv_path):
            # Extract the base include directory
            # For standard: /usr/local/cuda/include/nv/target -> /usr/local/cuda/include
            # For conda: /path/targets/x86_64-linux/include/nv/target -> /path/targets/x86_64-linux/include
            resolved_include_base = os.path.dirname(os.path.dirname(nv_path))
            print(f"  ✓ Using include directory: {resolved_include_base}")
            break

    # Build include directories from resolved path
    if resolved_include_base:
        cuda_include_dirs = [
            resolved_include_base,
            os.path.join(resolved_include_base, 'cccl'),  # CUDA 13 CCCL headers
            'src/cuda_kernels'
        ]
    else:
        # Fallback to standard path
        cuda_include_dirs = [
            os.path.join(CUDA_HOME, 'include'),
            os.path.join(CUDA_HOME, 'include', 'cccl'),  # CUDA 13 CCCL headers
            'src/cuda_kernels'
        ]

    # Build library directories list (support both lib64 and lib)
    cuda_library_dirs = []
    lib64_path = os.path.join(CUDA_HOME, 'lib64')
    lib_path = os.path.join(CUDA_HOME, 'lib')

    if os.path.isdir(lib64_path):
        cuda_library_dirs.append(lib64_path)
    if os.path.isdir(lib_path):
        cuda_library_dirs.append(lib_path)

    # Fallback to lib64 if neither exists (will fail later with clear error)
    if not cuda_library_dirs:
        cuda_library_dirs = [lib64_path]

    # Get CUDA architectures from environment or use defaults
    CUDA_ARCH_LIST = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if CUDA_ARCH_LIST is None:
        # Try to get from torch
        try:
            if torch.cuda.is_available():
                CUDA_ARCH_LIST = ';'.join([str(arch) for arch in torch.cuda.get_arch_list()])
            else:
                CUDA_ARCH_LIST = '72;87;110'  # Jetson: Xavier, Orin, Thor
        except:
            CUDA_ARCH_LIST = '72;87;110'  # Fallback: Jetson architectures

    arch_flags = []
    arch_list = []
    for arch in CUDA_ARCH_LIST.split(';'):
        # Remove compute_ and sm_ prefixes if present
        arch_clean = arch.replace('compute_', '').replace('sm_', '')
        # Remove dots (e.g., '8.6' -> '86') and +PTX suffix
        arch_clean = arch_clean.replace('.', '').replace('+PTX', '')
        if arch_clean.isdigit():  # Validate it's a number
            arch_flags.extend(['-gencode', f'arch=compute_{arch_clean},code=sm_{arch_clean}'])
            arch_list.append(arch_clean)

    # If arch_list is empty after processing, use default Jetson architectures
    if not arch_list:
        arch_list = ['72', '87', '110']  # Xavier, Orin, Thor
        for arch in arch_list:
            arch_flags.extend(['-gencode', f'arch=compute_{arch},code=sm_{arch}'])

    # Add PTX fallback for forward compatibility
    if arch_list:
        highest_arch = max(arch_list, key=int)
        arch_flags.extend(['-gencode', f'arch=compute_{highest_arch},code=compute_{highest_arch}'])

        # Define CUDA extension for custom kernels
    cuda_kernels = CUDAExtension(
        name='auto_voice.cuda_kernels',
        sources=[
            'src/cuda_kernels/audio_kernels.cu',
            'src/cuda_kernels/training_kernels.cu',
            'src/cuda_kernels/memory_kernels.cu',
        ] + (['src/cuda_kernels/fft_kernels.cu'] if os.environ.get('DISABLE_CUFFT') != '1' else []),
        include_dirs=cuda_include_dirs,
        library_dirs=cuda_library_dirs,
        libraries=[
            'cudart',
        ] + (['cufft'] if os.environ.get('DISABLE_CUFFT') != '1' else []),
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17', '-Wno-deprecated-declarations'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-std=c++17',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',  # Enable extended lambda support for device code
                '-D__CORRECT_ISO_CPP11_MATH_H_PROTO',  # Fix cmath issues with GCC 14
                '--allow-unsupported-compiler',  # Allow GCC 14 with CUDA 13
                # CUDA 13.0 Thor sm_110 compatibility flags
                '-device-entity-has-hidden-visibility=false',  # CUDA 13.0 changed ELF visibility default
                '--static-global-template-stub=false',  # CUDA 13.0 static global template stubs
                '--diag-suppress=20012',  # Suppress __device__ call from __host__ __device__ errors
                '--diag-suppress=20014',  # Suppress related device function call warnings
                '--diag-suppress=20036',  # Suppress extern "C" linkage warnings
                '--diag-suppress=174',    # Suppress expression has no effect warnings
                '--diag-suppress=20092',  # Suppress deprecation warnings for CUDA 13.0
                '--ptxas-options=-v',  # Verbose PTX assembly
                '-Xcompiler', '-Wno-deprecated-declarations',  # GCC 14 compat
            ] + arch_flags
        },
    )

    return [cuda_kernels], {'build_ext': BuildExtension}

# Build requirements for cuDNN, TensorRT
build_requirements = [
    'setuptools',
    'pybind11',
    'ninja',
]

# Conditionally build CUDA extensions only when needed
# This avoids importing torch during metadata-only operations (egg_info, sdist, etc.)
ext_modules = []
cmdclass = {}

if _wants_cuda_build(sys.argv):
    # Only attempt to build CUDA extensions for build commands
    ext_modules, cmdclass = _build_cuda_extensions()

setup(
    name='auto_voice',
    version='0.1.0',
    author='AutoVoice Team',
        description='GPU-accelerated voice synthesis system with CUDA 12.1',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        # ===========================================================================================
        # PREREQUISITE: PyTorch with CUDA Support
        # ===========================================================================================
        # PyTorch, torchvision, and torchaudio are REQUIRED but must be installed separately
        # BEFORE running `pip install -e .` to avoid version conflicts.
        #
        # Install PyTorch first using the official PyTorch index:
        #   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        #     --index-url https://download.pytorch.org/whl/cu121
        #
        # For detailed installation instructions, see:
        #   - requirements.txt (header section)
        #   - PYTORCH_ENVIRONMENT_FIX_REPORT.md
        # ===========================================================================================

        # Core numerical and audio processing (aligned with requirements.txt)
        'numpy>=1.26',
        'librosa>=0.10,<0.11',
        'soundfile>=0.12,<0.13',
        'scipy>=1.12,<1.14',
        'matplotlib>=3.7,<3.9',

        # Web framework (aligned with requirements.txt)
        'flask>=2.3,<3.0',
        'flask-socketio>=5.3,<6.0',
        'flask-cors>=4.0,<5.0',
        'python-socketio>=5.10,<6.0',

        # Configuration and utilities (aligned with requirements.txt)
        'pyyaml>=6.0,<7.0',
        'psutil>=5.9,<8.0',
        'pynvml>=11.5,<12.0',
        'websockets>=13,<14',
        'aiohttp>=3.9,<4.0',

        # Audio processing utilities (aligned with requirements.txt)
        'webrtcvad>=2.0,<3.0',
        'crepe>=0.0.12',
        'praat-parselmouth>=0.4.0',
        'noisereduce>=3.0.0',

        # GPU monitoring (aligned with requirements.txt)
        'nvitop>=1.3,<2.0',
        'py3nvml>=0.2.0',
    ],
    extras_require={
        # NOTE: pytorch-cu121 extra has been removed to prevent accidental CPU-only installations.
        # PyTorch MUST be installed separately with the correct index URL to get CUDA support.
        #
        # RECOMMENDED INSTALLATION METHODS:
        #
        # Option 1 - Use curated requirements file (RECOMMENDED):
        #   pip install -r requirements-cu121.txt --index-url https://download.pytorch.org/whl/cu121
        #
        # Option 2 - Manual installation:
        #   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        #     --index-url https://download.pytorch.org/whl/cu121
        #
        # Option 3 - Use setup script:
        #   ./scripts/setup_pytorch_env.sh
        #
        # WARNING: Installing without --index-url will install CPU-only PyTorch!
        # See requirements.txt header and PYTORCH_ENVIRONMENT_FIX_REPORT.md for details.

        'tensorrt': [
            # TensorRT requires special installation from NVIDIA index
            # pip install tensorrt~=8.6.0 --index-url https://pypi.nvidia.com
            'onnx>=1.14.0,<1.16.0',
            'onnxruntime-gpu>=1.16.0,<1.18.0',
        ],
        'dev': [
            'pytest>=7.4,<8.0',
            'pytest-cov>=4.1,<5.0',
            'black>=23.7,<24.0',
            'isort>=5.12,<6.0',
        ],
        'triton': [
            'triton>=2.1.0',  # For optimized kernels
        ],
    },
    python_requires='>=3.8',
    zip_safe=False,
)
