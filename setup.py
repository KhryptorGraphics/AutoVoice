from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# CUDA home directory (adjust if needed)
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')

# Get CUDA architectures from environment or use defaults
CUDA_ARCH_LIST = os.environ.get('TORCH_CUDA_ARCH_LIST', '80;86;89')
arch_flags = []
for arch in CUDA_ARCH_LIST.split(';'):
    arch_flags.extend(['-gencode', f'arch=compute_{arch},code=sm_{arch}'])

# Define CUDA extension for custom kernels
cuda_kernels = CUDAExtension(
    name='auto_voice.cuda_kernels',
    sources=[
        'src/cuda_kernels/audio_kernels.cu',
        'src/cuda_kernels/fft_kernels.cu',
        'src/cuda_kernels/training_kernels.cu',
        'src/cuda_kernels/memory_kernels.cu',
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
        'cublas',
        'cufft',
        'curand',
        'cusolver',
        'cusparse',
        'nvToolsExt',
    ],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
        ] + arch_flags
    },
    define_macros=[
        ('TORCH_EXTENSION_NAME', 'auto_voice.cuda_kernels'),
    ],
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
    long_description=open('docs/README.md').read() if os.path.exists('docs/README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[cuda_kernels],
    cmdclass={'build_ext': BuildExtension},
    setup_requires=build_requirements,
    install_requires=[
        'numpy==1.24.3',
        'librosa==0.10.1',
        'soundfile==0.12.1',
        'scipy==1.11.4',
        'matplotlib==3.7.2',
        'flask==2.3.3',
        'flask-socketio==5.3.6',
        'pyyaml==6.0.1',
        'psutil==5.9.6',
        'pynvml==11.5.0',
        'websockets==12.0',
        'aiohttp==3.9.1',
        'pytest==7.4.3',
        'black==23.7.0',
        'isort==5.12.0',
    ],
    extras_require={
        'gpu': [
            'torch>=2.1.0',
            'torchaudio>=2.1.0',
            'torchvision>=0.16.0',
            'triton>=2.1.0',
        ],
    },
    python_requires='>=3.8',
    zip_safe=False,
)