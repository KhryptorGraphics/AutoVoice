"""Monitoring and observability."""
from .prometheus import (
    record_conversion,
    record_cloning,
    update_gpu_metrics,
    track_inference,
    record_http_request,
    get_metrics,
    get_content_type,
    PROMETHEUS_AVAILABLE,
)

# Quality monitoring (lazy import to avoid circular deps)
try:
    from .quality_monitor import (
        QualityMonitor,
        QualityMetric,
        QualityThresholds,
        QualityAlert,
        QualityHistory,
        get_quality_monitor,
    )
    QUALITY_MONITOR_AVAILABLE = True
except ImportError:
    QUALITY_MONITOR_AVAILABLE = False

__all__ = [
    'record_conversion',
    'record_cloning',
    'update_gpu_metrics',
    'track_inference',
    'record_http_request',
    'get_metrics',
    'get_content_type',
    'PROMETHEUS_AVAILABLE',
    'QUALITY_MONITOR_AVAILABLE',
]

# Add quality monitor exports if available
if QUALITY_MONITOR_AVAILABLE:
    __all__.extend([
        'QualityMonitor',
        'QualityMetric',
        'QualityThresholds',
        'QualityAlert',
        'QualityHistory',
        'get_quality_monitor',
    ])
