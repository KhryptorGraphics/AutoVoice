"""Tests for monitoring/prometheus module."""
import pytest


class TestPrometheusMetrics:
    """Tests for Prometheus metric recording."""

    def test_prometheus_available(self):
        from auto_voice.monitoring import PROMETHEUS_AVAILABLE
        # prometheus_client should be installed
        assert PROMETHEUS_AVAILABLE is True

    def test_record_conversion(self):
        from auto_voice.monitoring import record_conversion
        # Should not raise
        record_conversion('balanced', 5.0, 30.0, success=True)
        record_conversion('draft', 1.0, 10.0, success=False)

    def test_record_cloning(self):
        from auto_voice.monitoring import record_cloning
        record_cloning(success=True)
        record_cloning(success=False)

    def test_track_inference_context(self):
        from auto_voice.monitoring import track_inference
        import time
        with track_inference('sovits', 'forward'):
            time.sleep(0.01)
        # Should complete without error

    def test_record_http_request(self):
        from auto_voice.monitoring import record_http_request
        record_http_request('GET', '/health', 200, 0.05)
        record_http_request('POST', '/api/v1/convert', 500, 1.5)

    def test_get_metrics(self):
        from auto_voice.monitoring import get_metrics, get_content_type
        metrics = get_metrics()
        assert isinstance(metrics, bytes)
        assert len(metrics) > 0
        assert b'autovoice_conversions_total' in metrics

    def test_get_content_type(self):
        from auto_voice.monitoring import get_content_type
        ct = get_content_type()
        assert 'text' in ct

    def test_update_gpu_metrics(self):
        from auto_voice.monitoring import update_gpu_metrics
        # Should not raise even without GPU
        update_gpu_metrics()


class TestPrometheusRegistry:
    """Tests for the metrics registry."""

    def test_registry_exists(self):
        from auto_voice.monitoring.prometheus import get_registry
        registry = get_registry()
        assert registry is not None

    def test_metrics_are_separate(self):
        from auto_voice.monitoring.prometheus import get_metrics
        # Multiple calls should work
        m1 = get_metrics()
        m2 = get_metrics()
        assert len(m1) > 0
        assert len(m2) > 0
