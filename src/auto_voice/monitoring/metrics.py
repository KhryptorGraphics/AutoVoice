"""Prometheus metrics for AutoVoice monitoring."""

import logging
import time
from functools import wraps
from typing import Callable, Optional
import threading

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not installed, metrics disabled")

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Central metrics collector for AutoVoice."""

    def __init__(self):
        """Initialize metrics collector."""
        if not PROMETHEUS_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True
        self.registry = CollectorRegistry()

        # HTTP Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )

        # WebSocket Metrics
        self.websocket_connections_total = Counter(
            'websocket_connections_total',
            'Total WebSocket connections',
            registry=self.registry
        )

        self.active_websocket_connections = Gauge(
            'active_websocket_connections',
            'Current active WebSocket connections',
            registry=self.registry
        )

        self.websocket_events_total = Counter(
            'websocket_events_total',
            'Total WebSocket events',
            ['event_type'],
            registry=self.registry
        )

        # Synthesis Metrics
        self.synthesis_requests_total = Counter(
            'synthesis_requests_total',
            'Total synthesis requests',
            ['speaker_id', 'success'],
            registry=self.registry
        )

        self.synthesis_duration_seconds = Histogram(
            'synthesis_duration_seconds',
            'Synthesis operation duration in seconds',
            registry=self.registry
        )

        # Audio Processing Metrics
        self.audio_processing_total = Counter(
            'audio_processing_total',
            'Total audio processing operations',
            ['operation', 'success'],
            registry=self.registry
        )

        self.audio_processing_duration_seconds = Histogram(
            'audio_processing_duration_seconds',
            'Audio processing duration in seconds',
            ['operation'],
            registry=self.registry
        )

        # Model Inference Metrics
        self.model_inference_duration_seconds = Histogram(
            'model_inference_duration_seconds',
            'Model inference time in seconds',
            registry=self.registry
        )

        # GPU Metrics
        self.gpu_memory_used_bytes = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory usage in bytes',
            ['device_id'],
            registry=self.registry
        )

        self.gpu_utilization_percent = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['device_id'],
            registry=self.registry
        )

        self.gpu_temperature_celsius = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['device_id'],
            registry=self.registry
        )

        # Model Status
        self.model_loaded = Gauge(
            'model_loaded',
            'Whether model is loaded (1=loaded, 0=not loaded)',
            registry=self.registry
        )

        logger.info("Metrics collector initialized")

    def generate_metrics(self):
        """Generate metrics in Prometheus format."""
        if not self.enabled:
            return b""
        return generate_latest(self.registry)

    def get_content_type(self):
        """Get content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics


def track_time(metric_name: str):
    """
    Decorator to track execution time of a function.

    Args:
        metric_name: Name of the histogram metric to update
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _metrics.enabled:
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Update histogram based on metric name
                if metric_name == 'synthesis_duration_seconds':
                    _metrics.synthesis_duration_seconds.observe(duration)
                elif metric_name == 'audio_processing_duration_seconds':
                    # Extract operation from kwargs or use function name
                    operation = kwargs.get('operation', func.__name__)
                    _metrics.audio_processing_duration_seconds.labels(operation=operation).observe(duration)
                elif metric_name == 'model_inference_duration_seconds':
                    _metrics.model_inference_duration_seconds.observe(duration)

                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise

        return wrapper
    return decorator


def count_calls(metric_name: str, **labels):
    """
    Decorator to count function calls.

    Args:
        metric_name: Name of the counter metric to update
        **labels: Labels to add to the metric
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _metrics.enabled:
                return func(*args, **kwargs)

            try:
                result = func(*args, **kwargs)

                # Update counter based on metric name
                if metric_name == 'synthesis_requests_total':
                    speaker_id = kwargs.get('speaker_id', 'default')
                    _metrics.synthesis_requests_total.labels(
                        speaker_id=speaker_id,
                        success='true'
                    ).inc()
                elif metric_name == 'audio_processing_total':
                    operation = kwargs.get('operation', func.__name__)
                    _metrics.audio_processing_total.labels(
                        operation=operation,
                        success='true'
                    ).inc()

                return result
            except Exception as e:
                # Track failures
                if metric_name == 'synthesis_requests_total':
                    speaker_id = kwargs.get('speaker_id', 'default')
                    _metrics.synthesis_requests_total.labels(
                        speaker_id=speaker_id,
                        success='false'
                    ).inc()
                elif metric_name == 'audio_processing_total':
                    operation = kwargs.get('operation', func.__name__)
                    _metrics.audio_processing_total.labels(
                        operation=operation,
                        success='false'
                    ).inc()

                raise

        return wrapper
    return decorator


def record_synthesis(duration: float, speaker_id: str = 'default', success: bool = True):
    """Record synthesis operation metrics."""
    if not _metrics.enabled:
        return

    _metrics.synthesis_requests_total.labels(
        speaker_id=speaker_id,
        success=str(success).lower()
    ).inc()
    _metrics.synthesis_duration_seconds.observe(duration)


def record_audio_processing(operation: str, duration: float, success: bool = True):
    """Record audio processing metrics."""
    if not _metrics.enabled:
        return

    _metrics.audio_processing_total.labels(
        operation=operation,
        success=str(success).lower()
    ).inc()
    _metrics.audio_processing_duration_seconds.labels(operation=operation).observe(duration)


def update_gpu_metrics(device_id: int = 0):
    """
    Update GPU metrics from performance monitor.

    Args:
        device_id: GPU device ID
    """
    if not _metrics.enabled:
        return

    try:
        import pynvml
        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Memory usage
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        _metrics.gpu_memory_used_bytes.labels(device_id=str(device_id)).set(mem_info.used)

        # GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        _metrics.gpu_utilization_percent.labels(device_id=str(device_id)).set(utilization.gpu)

        # Temperature
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        _metrics.gpu_temperature_celsius.labels(device_id=str(device_id)).set(temperature)

        pynvml.nvmlShutdown()

    except Exception as e:
        logger.warning(f"Failed to update GPU metrics: {e}")


def start_gpu_metrics_collection(interval: int = 10):
    """
    Start periodic GPU metrics collection.

    Args:
        interval: Collection interval in seconds
    """
    if not _metrics.enabled:
        return

    def collect():
        while True:
            try:
                update_gpu_metrics()
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
            time.sleep(interval)

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    logger.info(f"Started GPU metrics collection (interval: {interval}s)")


def set_model_loaded(loaded: bool):
    """
    Set model loaded status.

    Args:
        loaded: Whether model is loaded
    """
    if not _metrics.enabled:
        return

    _metrics.model_loaded.set(1 if loaded else 0)
