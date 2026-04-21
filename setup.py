"""AutoVoice package setup with optional CUDA kernel compilation."""
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# Attempt CUDA extension build
cuda_extensions = []
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda-13.0')
    if os.path.exists(CUDA_HOME) and torch.cuda.is_available():
        cuda_extensions.append(
            CUDAExtension(
                name='auto_voice._cuda_kernels',
                sources=[
                    'src/cuda_kernels/pitch_kernel.cu',
                    'src/cuda_kernels/synthesis_kernel.cu',
                ],
                include_dirs=[
                    'src/cuda_kernels',
                    os.path.join(CUDA_HOME, 'include'),
                ],
                extra_compile_args={
                    'cxx': ['-std=c++17', '-O3'],
                    'nvcc': [
                        '-std=c++17',
                        '-O3',
                        '--expt-extended-lambda',
                        '-gencode=arch=compute_110,code=sm_110',
                        f'-I{os.path.join(CUDA_HOME, "include")}',
                    ],
                },
            )
        )
        cmdclass = {'build_ext': BuildExtension}
    else:
        cmdclass = {}
except (ImportError, Exception):
    cmdclass = {}

setup(
    name='auto-voice',
    version='0.1.0',
    description='GPU-accelerated singing voice conversion and TTS',
    author='AutoVoice Team',
    python_requires='>=3.10',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    ext_modules=cuda_extensions,
    cmdclass=cmdclass,
    install_requires=[
        'torch>=2.0',
        'torchaudio>=2.0',
        'numpy>=1.24',
        'scipy>=1.10',
        'librosa>=0.10',
        'soundfile>=0.12',
        'flask>=3.0',
        'flask-socketio>=5.3',
        'flask-swagger-ui>=4.11',
        'pyyaml>=6.0',
        'apispec>=6.0',
        'apispec-webframeworks>=1.0',
        'marshmallow>=3.0',
        'memkraft>=0.1.0',
        'prometheus_client>=0.19',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-asyncio',
            'pytest-cov',
            'pytest-mock',
            'pytest-timeout',
            'black',
            'isort',
            'mypy',
        ],
        'ml': [
            'transformers>=4.30',
            'resemblyzer>=0.1.3',
            'demucs>=4.0',
            'pynvml>=11.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'autovoice=auto_voice.cli:main',
        ],
    },
)
