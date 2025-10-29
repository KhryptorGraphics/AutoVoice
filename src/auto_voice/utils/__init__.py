"""Utilities module for AutoVoice"""

# Configuration utilities
from .config_loader import (
    load_config,
    load_config_with_defaults,
    load_config_from_file,
    merge_configs,
    load_config_from_env,
    validate_config,
    DEFAULT_CONFIG
)

# Logging utilities
from .logging_config import (
    setup_logging,
    JSONFormatter,
    ColoredFormatter,
    SensitiveDataFilter,
    LogContext,
    log_execution_time
)

# Data processing utilities
from .data_utils import (
    DataCollator,
    AudioCollator,
    DataBatcher,
    DataSampler,
    DataPreprocessor,
    create_data_loader
)

# Metrics and evaluation utilities
from .metrics import (
    AudioMetrics,
    ModelMetrics,
    PerformanceMetrics,
    MetricsAggregator,
    get_global_metrics,
    reset_global_metrics
)

# Quality evaluation utilities
from .quality_metrics import (
    QualityMetricsAggregator,
    PitchAccuracyMetrics,
    SpeakerSimilarityMetrics,
    NaturalnessMetrics,
    IntelligibilityMetrics,
    AudioAligner,
    AudioNormalizer,
    QualityMetricsResult,
    PitchAccuracyResult,
    SpeakerSimilarityResult,
    NaturalnessResult,
    IntelligibilityResult
)

# Visualization utilities
from .visualization import (
    PitchContourVisualizer,
    SpectrogramVisualizer,
    QualityMetricsVisualizer,
    PitchContourData,
    encode_plot_as_base64,
    create_embedded_markdown_image
)

# General helper utilities
from .helpers import (
    StringUtils,
    MathUtils,
    ValidationUtils,
    RetryUtils,
    CacheUtils,
    ensure_dir,
    safe_divide,
    safe_log,
    flatten_dict
)

__all__ = [
    # Config utilities
    'load_config',
    'load_config_with_defaults',
    'load_config_from_file',
    'merge_configs',
    'load_config_from_env',
    'validate_config',
    'DEFAULT_CONFIG',
    
    # Logging utilities
    'setup_logging',
    'JSONFormatter',
    'ColoredFormatter',
    'SensitiveDataFilter',
    'LogContext',
    'log_execution_time',
    
    # Data utilities
    'DataCollator',
    'AudioCollator',
    'DataBatcher',
    'DataSampler',
    'DataPreprocessor',
    'create_data_loader',
    
    # Metrics utilities
    'AudioMetrics',
    'ModelMetrics',
    'PerformanceMetrics',
    'MetricsAggregator',
    'get_global_metrics',
    'reset_global_metrics',

    # Quality evaluation utilities
    'QualityMetricsAggregator',
    'PitchAccuracyMetrics',
    'SpeakerSimilarityMetrics',
    'NaturalnessMetrics',
    'IntelligibilityMetrics',
    'AudioAligner',
    'AudioNormalizer',
    'QualityMetricsResult',
    'PitchAccuracyResult',
    'SpeakerSimilarityResult',
    'NaturalnessResult',
    'IntelligibilityResult',

    # Visualization utilities
    'PitchContourVisualizer',
    'SpectrogramVisualizer',
    'QualityMetricsVisualizer',
    'PitchContourData',
    'encode_plot_as_base64',
    'create_embedded_markdown_image',

    # Helper utilities
    'StringUtils',
    'MathUtils',
    'ValidationUtils',
    'RetryUtils',
    'CacheUtils',
    'ensure_dir',
    'safe_divide',
    'safe_log',
    'flatten_dict'
]
