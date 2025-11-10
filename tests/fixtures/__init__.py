"""Testing fixtures for AutoVoice test suite.

This module exports all testing fixtures for easy import across test modules.
"""

from .audio_fixtures import *
from .model_fixtures import *
from .gpu_fixtures import *
from .mock_fixtures import *
from .integration_fixtures import *
from .performance_fixtures import *

__all__ = [
    # Audio fixtures
    'sample_audio_factory',
    'audio_file_factory',
    'multi_channel_audio',
    'audio_batch_generator',
    'corrupted_audio_samples',

    # Model fixtures
    'mock_voice_model',
    'mock_encoder',
    'mock_decoder',
    'mock_vocoder',
    'trained_model_checkpoint',

    # GPU fixtures
    'gpu_context_manager',
    'cuda_memory_tracker',
    'multi_gpu_config',

    # Mock fixtures
    'mock_file_system',
    'mock_network_client',
    'mock_cache_manager',

    # Integration fixtures
    'pipeline_test_suite',
    'end_to_end_workflow',

    # Performance fixtures
    'performance_benchmarker',
    'resource_profiler',
]
