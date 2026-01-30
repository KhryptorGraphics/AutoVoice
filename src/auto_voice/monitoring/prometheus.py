"""Prometheus metrics for AutoVoice monitoring.

Tracks inference latency, GPU utilization, conversion counts,
and other operational metrics.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, CollectorRegistry,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed, metrics disabled")


# Default registry for AutoVoice metrics
_registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None


def get_registry():
    """Get the AutoVoice metrics registry."""
    return _registry


if PROMETHEUS_AVAILABLE:
    # Conversion metrics
    CONVERSIONS_TOTAL = Counter(
        'autovoice_conversions_total',
        'Total voice conversions performed',
        ['preset', 'status'],
        registry=_registry,
    )

    CONVERSION_DURATION = Histogram(
        'autovoice_conversion_duration_seconds',
        'Time spent on voice conversion',
        ['preset'],
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        registry=_registry,
    )

    CONVERSION_AUDIO_DURATION = Histogram(
        'autovoice_conversion_audio_duration_seconds',
        'Duration of converted audio',
        buckets=(5, 10, 30, 60, 120, 300, 600),
        registry=_registry,
    )

    # Voice cloning metrics
    CLONING_TOTAL = Counter(
        'autovoice_cloning_total',
        'Total voice cloning operations',
        ['status'],
        registry=_registry,
    )

    PROFILES_ACTIVE = Gauge(
        'autovoice_profiles_active',
        'Number of active voice profiles',
        registry=_registry,
    )

    # GPU metrics
    GPU_MEMORY_USED = Gauge(
        'autovoice_gpu_memory_used_bytes',
        'GPU memory currently used',
        ['device'],
        registry=_registry,
    )

    GPU_MEMORY_TOTAL = Gauge(
        'autovoice_gpu_memory_total_bytes',
        'Total GPU memory',
        ['device'],
        registry=_registry,
    )

    GPU_UTILIZATION = Gauge(
        'autovoice_gpu_utilization_percent',
        'GPU compute utilization',
        ['device'],
        registry=_registry,
    )

    # Inference metrics
    INFERENCE_LATENCY = Histogram(
        'autovoice_inference_latency_seconds',
        'Model inference latency',
        ['model', 'operation'],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        registry=_registry,
    )

    # Request metrics
    HTTP_REQUESTS = Counter(
        'autovoice_http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status'],
        registry=_registry,
    )

    HTTP_REQUEST_DURATION = Histogram(
        'autovoice_http_request_duration_seconds',
        'HTTP request duration',
        ['method', 'endpoint'],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
        registry=_registry,
    )

    # Job queue metrics
    JOBS_IN_QUEUE = Gauge(
        'autovoice_jobs_in_queue',
        'Number of jobs currently queued',
        registry=_registry,
    )

    JOBS_PROCESSING = Gauge(
        'autovoice_jobs_processing',
        'Number of jobs currently processing',
        registry=_registry,
    )

    # System info
    SYSTEM_INFO = Info(
        'autovoice',
        'AutoVoice system information',
        registry=_registry,
    )


def record_conversion(preset: str, duration: float, audio_duration: float,
                      success: bool = True):
    """Record a voice conversion operation."""
    if not PROMETHEUS_AVAILABLE:
        return

    status = 'success' if success else 'error'
    CONVERSIONS_TOTAL.labels(preset=preset, status=status).inc()
    if success:
        CONVERSION_DURATION.labels(preset=preset).observe(duration)
        CONVERSION_AUDIO_DURATION.observe(audio_duration)


def record_cloning(success: bool = True):
    """Record a voice cloning operation."""
    if not PROMETHEUS_AVAILABLE:
        return

    status = 'success' if success else 'error'
    CLONING_TOTAL.labels(status=status).inc()


def update_gpu_metrics():
    """Update GPU metrics from current state."""
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        import torch
        if not torch.cuda.is_available():
            return

        for i in range(torch.cuda.device_count()):
            device_label = f'cuda:{i}'
            mem_info = torch.cuda.mem_get_info(i)
            free, total = mem_info
            used = total - free

            GPU_MEMORY_USED.labels(device=device_label).set(used)
            GPU_MEMORY_TOTAL.labels(device=device_label).set(total)

            # Try pynvml for utilization
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                GPU_UTILIZATION.labels(device=device_label).set(util.gpu)
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"GPU metrics update failed: {e}")


@contextmanager
def track_inference(model: str, operation: str):
    """Context manager to track inference latency."""
    if not PROMETHEUS_AVAILABLE:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        INFERENCE_LATENCY.labels(model=model, operation=operation).observe(elapsed)


def record_http_request(method: str, endpoint: str, status: int, duration: float):
    """Record an HTTP request."""
    if not PROMETHEUS_AVAILABLE:
        return

    HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def set_system_info(version: str, device: str, cuda_version: str = ''):
    """Set system info metric."""
    if not PROMETHEUS_AVAILABLE:
        return

    SYSTEM_INFO.info({
        'version': version,
        'device': device,
        'cuda_version': cuda_version,
    })


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    if not PROMETHEUS_AVAILABLE:
        return b''
    return generate_latest(_registry)


def get_content_type() -> str:
    """Get Prometheus content type header value."""
    if not PROMETHEUS_AVAILABLE:
        return 'text/plain'
    return CONTENT_TYPE_LATEST
