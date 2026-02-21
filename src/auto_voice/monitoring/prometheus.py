"""Prometheus metrics for AutoVoice monitoring.

Tracks inference latency, GPU utilization, conversion counts,
and other operational metrics.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

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


# ============================================================================
# Conversion Analytics - Privacy-Respecting Metrics Aggregation
# ============================================================================

class ConversionAnalytics:
    """Simple, privacy-respecting conversion analytics.

    Only tracks aggregate metrics, no personal data or audio content.
    Provides quick health check data without querying Prometheus.
    """

    def __init__(self):
        self._metrics = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_audio_seconds': 0.0,
            'total_errors': 0,
            'avg_latency_ms': 0.0,
            'avg_audio_duration_s': 0.0,
            'gpu_utilization': 0.0,
            'conversions_by_preset': {},  # preset -> count
        }
        self._conversion_count_for_avg = 0
        self._latency_samples = []
        self._audio_duration_samples = []

    def record_conversion_start(self):
        """Record a new conversion started."""
        self._metrics['total_conversions'] += 1

    def record_conversion_complete(
        self,
        preset: str,
        latency_ms: float,
        audio_duration_s: float,
        success: bool = True
    ):
        """Record conversion completion.

        Args:
            preset: Conversion preset used (e.g., 'quality', 'realtime')
            latency_ms: Processing latency in milliseconds
            audio_duration_s: Duration of converted audio in seconds
            success: Whether conversion succeeded
        """
        if success:
            self._metrics['successful_conversions'] += 1
        else:
            self._metrics['failed_conversions'] += 1

        self._metrics['total_audio_seconds'] += audio_duration_s

        # Track conversions by preset
        self._metrics['conversions_by_preset'][preset] = (
            self._metrics['conversions_by_preset'].get(preset, 0) + 1
        )

        # Update rolling average latency (keep last 1000 samples)
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 1000:
            self._latency_samples.pop(0)
        self._metrics['avg_latency_ms'] = (
            sum(self._latency_samples) / len(self._latency_samples)
        )

        # Update rolling average audio duration
        self._audio_duration_samples.append(audio_duration_s)
        if len(self._audio_duration_samples) > 1000:
            self._audio_duration_samples.pop(0)
        self._metrics['avg_audio_duration_s'] = (
            sum(self._audio_duration_samples) / len(self._audio_duration_samples)
        )

    def record_error(self):
        """Record an error occurrence."""
        self._metrics['total_errors'] += 1

    def update_gpu_utilization(self, utilization_percent: float):
        """Update GPU utilization metric.

        Args:
            utilization_percent: Current GPU utilization (0-100)
        """
        self._metrics['gpu_utilization'] = utilization_percent

    def get_metrics(self) -> Dict[str, Any]:
        """Get current aggregate metrics.

        Returns:
            Dict with keys:
                - total_conversions: Total number of conversions
                - successful_conversions: Number of successful conversions
                - failed_conversions: Number of failed conversions
                - total_audio_minutes: Total audio processed in minutes
                - avg_latency_ms: Average processing latency
                - avg_audio_duration_s: Average audio duration
                - gpu_utilization: Current GPU utilization percent
                - total_errors: Total error count
                - conversions_by_preset: Breakdown by preset
        """
        return {
            'total_conversions': self._metrics['total_conversions'],
            'successful_conversions': self._metrics['successful_conversions'],
            'failed_conversions': self._metrics['failed_conversions'],
            'total_audio_minutes': round(self._metrics['total_audio_seconds'] / 60, 1),
            'avg_latency_ms': round(self._metrics['avg_latency_ms'], 1),
            'avg_audio_duration_s': round(self._metrics['avg_audio_duration_s'], 1),
            'gpu_utilization': round(self._metrics['gpu_utilization'], 1),
            'total_errors': self._metrics['total_errors'],
            'conversions_by_preset': dict(self._metrics['conversions_by_preset']),
        }


# Global analytics instance
_conversion_analytics = ConversionAnalytics()


def get_conversion_analytics() -> Dict[str, Any]:
    """Get conversion usage analytics (for health/monitoring endpoints).

    Returns:
        Dict containing aggregate conversion metrics including:
        total_conversions, avg_latency_ms, gpu_utilization, etc.
    """
    return _conversion_analytics.get_metrics()


def record_conversion(preset: str, duration: float, audio_duration: float,
                      success: bool = True):
    """Record a voice conversion operation.

    Records to both Prometheus metrics (if available) and in-memory analytics.

    Args:
        preset: Conversion preset used (e.g., 'quality', 'realtime')
        duration: Processing duration in seconds
        audio_duration: Duration of converted audio in seconds
        success: Whether conversion succeeded
    """
    # Record to in-memory analytics (always available)
    _conversion_analytics.record_conversion_complete(
        preset=preset,
        latency_ms=duration * 1000,  # Convert seconds to milliseconds
        audio_duration_s=audio_duration,
        success=success
    )

    # Record to Prometheus metrics (if available)
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
    """Update GPU metrics from current state.

    Updates both Prometheus metrics (if available) and in-memory analytics.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return

        # Track first GPU's utilization for analytics
        primary_gpu_util = 0.0

        for i in range(torch.cuda.device_count()):
            device_label = f'cuda:{i}'
            mem_info = torch.cuda.mem_get_info(i)
            free, total = mem_info
            used = total - free

            # Update Prometheus metrics (if available)
            if PROMETHEUS_AVAILABLE:
                GPU_MEMORY_USED.labels(device=device_label).set(used)
                GPU_MEMORY_TOTAL.labels(device=device_label).set(total)

            # Try pynvml for utilization
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                if PROMETHEUS_AVAILABLE:
                    GPU_UTILIZATION.labels(device=device_label).set(util.gpu)

                # Track primary GPU utilization for analytics
                if i == 0:
                    primary_gpu_util = util.gpu
            except Exception:
                pass

        # Update analytics with primary GPU utilization
        _conversion_analytics.update_gpu_utilization(primary_gpu_util)

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
