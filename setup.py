from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
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
    return 'GPU-accelerated voice synthesis system with CUDA 12.9'

def _check_cuda_availability():
    """Check CUDA availability and provide helpful error messages"""
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. This package requires CUDA for GPU acceleration.")
        print("Please ensure you have:")
        print("  1. NVIDIA GPU with compute capability >= 7.0")
        print("  2. CUDA toolkit installed (CUDA 11.8+ recommended)")
        print("  3. PyTorch with CUDA support installed")
        return False
    return True

# CUDA home directory (adjust if needed)
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')

# Check CUDA availability early and determine build strategy
cuda_available = _check_cuda_availability()
if not cuda_available:
    # Option A: CPU-only install (set ext_modules to empty list)
    # Option B: Raise SystemExit with clear message (uncomment below)
    print("ERROR: CUDA is required for this package. CPU-only installs are not supported.")
    print("Please install CUDA and PyTorch with CUDA support before installing this package.")
    sys.exit(1)

# Get CUDA architectures from environment or use defaults
CUDA_ARCH_LIST = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
if CUDA_ARCH_LIST is None:
    # Try to get from torch
    try:
        if torch.cuda.is_available():
            CUDA_ARCH_LIST = ';'.join([str(arch) for arch in torch.cuda.get_arch_list()])
        else:
            CUDA_ARCH_LIST = '70;75;80;86'  # Common architectures
    except:
        CUDA_ARCH_LIST = '70;75;80;86'  # Fallback to common architectures

arch_flags = []
arch_list = []
for arch in CUDA_ARCH_LIST.split(';'):
    # Remove compute_ and sm_ prefixes if present
    arch_clean = arch.replace('compute_', '').replace('sm_', '')
    # Normalize architecture string by removing dots (e.g., '8.6' -> '86')
    arch_clean = arch_clean.replace('.', '')
    if arch_clean.isdigit():  # Validate it's a number
        arch_flags.extend(['-gencode', f'arch=compute_{arch_clean},code=sm_{arch_clean}'])
        arch_list.append(arch_clean)

# Add PTX fallback for forward compatibility
if arch_list:
    highest_arch = max(arch_list, key=int)
    arch_flags.extend(['-gencode', f'arch=compute_{highest_arch},code=compute_{highest_arch}'])

# Define CUDA extension for custom kernels
cuda_kernels = CUDAExtension(
    name='auto_voice.cuda_kernels',
    sources=[
        'src/cuda_kernels/audio_kernels.cu',
        'src/cuda_kernels/fft_kernels.cu',
        'src/cuda_kernels/training_kernels.cu',
        'src/cuda_kernels/memory_kernels.cu',
        'src/cuda_kernels/kernel_wrappers.cu',
        'src/cuda_kernels/bindings.cpp',
    ],
    include_dirs=[
        os.path.join(CUDA_HOME, 'include'),
        'src/cuda_kernels',
    ],
    library_dirs=[
        os.path.join(CUDA_HOME, 'lib64'),
    ],
    libraries=[
        'cudart',
        'cufft',
    ],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '--expt-relaxed-constexpr',
            '--ptxas-options=-v',  # Verbose PTX assembly
        ] + arch_flags
    },
)

# Build requirements for cuDNN, TensorRT
build_requirements = [
    'torch',
    'setuptools',
    'pybind11',
    'ninja',
]

setup(
    name='auto_voice',
    version='0.1.0',
    author='AutoVoice Team',
    description='GPU-accelerated voice synthesis system with CUDA 12.9',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[cuda_kernels],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        # Core ML/Deep Learning dependencies (required for GPU voice synthesis)
        'torch>=2.0.0,<2.2.0',  # PyTorch is mandatory for this GPU-accelerated system
        'torchaudio>=2.0.0,<2.2.0',
        'torchvision>=0.15.0,<0.17.0',

        # Core numerical and audio processing
        'numpy>=1.24,<1.27',
        'librosa>=0.10,<0.11',
        'soundfile>=0.12,<0.13',
        'scipy>=1.10,<1.12',
        'matplotlib>=3.7,<3.9',

        # Web framework
        'flask>=2.3,<3.0',
        'flask-socketio>=5.3,<6.0',
        'flask-cors>=4.0,<5.0',
        'python-socketio>=5.10,<6.0',

        # Configuration and utilities
        'pyyaml>=6.0,<7.0',
        'psutil>=5.9,<6.0',
        'pynvml>=11.5,<12.0',
        'websockets>=12.0,<13.0',
        'aiohttp>=3.9,<4.0',

        # Audio processing utilities
        'webrtcvad>=2.0,<3.0',
        'crepe>=0.0.12',
        'praat-parselmouth>=0.4.0',
        'noisereduce>=3.0.0',

        # GPU monitoring
        'nvitop>=1.3,<2.0',
        'py3nvml>=0.2.0',
    ],
    extras_require={
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