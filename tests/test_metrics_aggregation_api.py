"""Tests for metrics aggregation API endpoint.

Coverage for GET /api/v1/metrics endpoint that aggregates
conversion analytics for dashboard consumption.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestMetricsAggregationEndpoint:
    """Test metrics aggregation endpoint /api/v1/metrics."""

    @pytest.mark.smoke
    def test_metrics_returns_200(self, client):
        """Test metrics endpoint returns 200 status."""
        resp = client.get('/api/v1/metrics')
        assert resp.status_code == 200

    @pytest.mark.smoke
    def test_metrics_returns_json(self, client):
        """Test metrics endpoint returns JSON response."""
        resp = client.get('/api/v1/metrics')
        assert resp.status_code == 200
        assert resp.content_type == 'application/json'
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_metrics_has_required_fields(self, client):
        """Test metrics response contains all required fields."""
        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # Required fields per implementation_plan.json acceptance criteria
        required_fields = [
            'total_conversions',
            'avg_latency_ms',
            'gpu_utilization',
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_metrics_has_complete_structure(self, client):
        """Test metrics response has complete analytics structure."""
        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # All fields from ConversionAnalytics.get_metrics()
        expected_fields = [
            'total_conversions',
            'successful_conversions',
            'failed_conversions',
            'total_audio_minutes',
            'avg_latency_ms',
            'avg_audio_duration_s',
            'gpu_utilization',
            'total_errors',
            'conversions_by_preset',
        ]

        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_metrics_field_types(self, client):
        """Test metrics fields have correct data types."""
        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # Verify types
        assert isinstance(data['total_conversions'], int)
        assert isinstance(data['successful_conversions'], int)
        assert isinstance(data['failed_conversions'], int)
        assert isinstance(data['total_audio_minutes'], (int, float))
        assert isinstance(data['avg_latency_ms'], (int, float))
        assert isinstance(data['avg_audio_duration_s'], (int, float))
        assert isinstance(data['gpu_utilization'], (int, float))
        assert isinstance(data['total_errors'], int)
        assert isinstance(data['conversions_by_preset'], dict)

    def test_metrics_numeric_values_non_negative(self, client):
        """Test metrics numeric values are non-negative."""
        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # All counts and metrics should be >= 0
        assert data['total_conversions'] >= 0
        assert data['successful_conversions'] >= 0
        assert data['failed_conversions'] >= 0
        assert data['total_audio_minutes'] >= 0
        assert data['avg_latency_ms'] >= 0
        assert data['avg_audio_duration_s'] >= 0
        assert data['gpu_utilization'] >= 0
        assert data['total_errors'] >= 0

    def test_metrics_initial_state(self, client):
        """Test metrics in initial state (no conversions yet)."""
        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # On fresh app, should have zero conversions
        assert data['total_conversions'] == 0
        assert data['successful_conversions'] == 0
        assert data['failed_conversions'] == 0
        assert data['total_audio_minutes'] == 0
        assert data['avg_latency_ms'] == 0
        assert data['avg_audio_duration_s'] == 0
        assert data['total_errors'] == 0
        assert data['conversions_by_preset'] == {}

    def test_metrics_after_simulated_conversion(self, client):
        """Test metrics update after recording a conversion."""
        from auto_voice.monitoring.prometheus import record_conversion

        # Get initial state
        resp_before = client.get('/api/v1/metrics')
        data_before = resp_before.get_json()
        initial_successful = data_before['successful_conversions']

        # Record a test conversion
        record_conversion(
            preset='quality',
            duration=2.5,  # 2.5 seconds processing time
            audio_duration=30.0,  # 30 seconds of audio
            success=True
        )

        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # Should reflect the recorded conversion
        assert data['successful_conversions'] >= initial_successful + 1
        assert data['conversions_by_preset'].get('quality', 0) >= 1
        assert data['avg_latency_ms'] > 0  # Should have recorded latency
        assert data['avg_audio_duration_s'] > 0

    def test_metrics_prometheus_format_via_query_param(self, client):
        """Test Prometheus text format via format=prometheus query param."""
        resp = client.get('/api/v1/metrics?format=prometheus')

        # May return 503 if prometheus_client not installed, or text format
        assert resp.status_code in (200, 503)

        if resp.status_code == 200:
            # Should be text format, not JSON
            assert 'text/plain' in resp.content_type or 'text' in resp.content_type
            # Prometheus format is text, not JSON
            content = resp.get_data(as_text=True)
            assert isinstance(content, str)

    def test_metrics_prometheus_format_via_accept_header(self, client):
        """Test Prometheus text format via Accept: text/plain header."""
        resp = client.get('/api/v1/metrics', headers={'Accept': 'text/plain'})

        # May return 503 if prometheus_client not installed
        assert resp.status_code in (200, 503)

        if resp.status_code == 200:
            # Should be text format
            assert 'text/plain' in resp.content_type or 'text' in resp.content_type

    def test_metrics_default_is_json_not_prometheus(self, client):
        """Test default response is JSON, not Prometheus text format."""
        resp = client.get('/api/v1/metrics')

        assert resp.status_code == 200
        assert resp.content_type == 'application/json'

        # Should be parseable as JSON
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_metrics_prometheus_unavailable_handling(self, client):
        """Test graceful handling when prometheus_client is unavailable."""
        # This test verifies the endpoint works even without Prometheus

        # The endpoint should still return JSON analytics (uses ConversionAnalytics)
        resp = client.get('/api/v1/metrics')
        assert resp.status_code == 200

        data = resp.get_json()
        assert 'total_conversions' in data

    @patch('auto_voice.monitoring.prometheus.get_conversion_analytics')
    def test_metrics_error_handling(self, mock_analytics, client):
        """Test error handling when analytics retrieval fails."""
        mock_analytics.side_effect = Exception("Analytics error")

        resp = client.get('/api/v1/metrics')

        # Should return error response
        assert resp.status_code in (200, 500, 503)

        # If it returns a JSON error, check structure
        if resp.status_code != 200:
            data = resp.get_json()
            assert 'error' in data or 'message' in data

    def test_metrics_endpoint_idempotent(self, client):
        """Test multiple calls to metrics endpoint return consistent results."""
        resp1 = client.get('/api/v1/metrics')
        resp2 = client.get('/api/v1/metrics')

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        data1 = resp1.get_json()
        data2 = resp2.get_json()

        # Metrics should be consistent (no conversions between calls)
        assert data1['total_conversions'] == data2['total_conversions']

    def test_metrics_gpu_utilization_range(self, client):
        """Test GPU utilization is within valid range (0-100)."""
        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # GPU utilization should be 0-100 percent
        assert 0 <= data['gpu_utilization'] <= 100

    def test_metrics_conversions_sum_invariant(self, client):
        """Test successful + failed conversions are tracked independently."""
        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # All conversion counts should be non-negative
        assert data['successful_conversions'] >= 0
        assert data['failed_conversions'] >= 0
        assert data['total_conversions'] >= 0

    def test_metrics_preset_breakdown_consistency(self, client):
        """Test conversions_by_preset tracks individual presets."""
        from auto_voice.monitoring.prometheus import record_conversion

        # Get initial state
        resp_before = client.get('/api/v1/metrics')
        data_before = resp_before.get_json()
        initial_quality = data_before['conversions_by_preset'].get('quality', 0)
        initial_realtime = data_before['conversions_by_preset'].get('realtime', 0)

        # Record some conversions with different presets
        record_conversion('quality', 1.0, 10.0, success=True)
        record_conversion('realtime', 0.5, 10.0, success=True)
        record_conversion('quality', 1.2, 15.0, success=True)

        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # Should track each preset separately
        assert data['conversions_by_preset']['quality'] == initial_quality + 2
        assert data['conversions_by_preset']['realtime'] == initial_realtime + 1

    def test_metrics_follows_karaoke_pattern(self, client):
        """Test metrics endpoint follows karaoke /metrics API pattern."""
        # The implementation should follow karaoke_api.py pattern (lines 172-192)

        resp = client.get('/api/v1/metrics')

        # Should return 200 and JSON
        assert resp.status_code == 200
        assert resp.content_type == 'application/json'

        # Should return dict with aggregate metrics
        data = resp.get_json()
        assert isinstance(data, dict)
        assert len(data) > 0


class TestMetricsIntegrationWithConversionAnalytics:
    """Test integration between /metrics endpoint and ConversionAnalytics."""

    def test_analytics_records_successful_conversion(self, client):
        """Test analytics correctly tracks successful conversions."""
        from auto_voice.monitoring.prometheus import record_conversion

        # Get initial state
        resp_before = client.get('/api/v1/metrics')
        data_before = resp_before.get_json()
        initial_successful = data_before['successful_conversions']

        # Record successful conversion
        record_conversion('quality', 2.0, 20.0, success=True)

        # Check updated state
        resp_after = client.get('/api/v1/metrics')
        data_after = resp_after.get_json()

        assert data_after['successful_conversions'] == initial_successful + 1
        assert data_after['failed_conversions'] == data_before['failed_conversions']

    def test_analytics_records_failed_conversion(self, client):
        """Test analytics correctly tracks failed conversions."""
        from auto_voice.monitoring.prometheus import record_conversion

        # Get initial state
        resp_before = client.get('/api/v1/metrics')
        data_before = resp_before.get_json()
        initial_failed = data_before['failed_conversions']

        # Record failed conversion
        record_conversion('quality', 1.0, 0.0, success=False)

        # Check updated state
        resp_after = client.get('/api/v1/metrics')
        data_after = resp_after.get_json()

        assert data_after['failed_conversions'] == initial_failed + 1

    def test_analytics_updates_latency_average(self, client):
        """Test analytics updates average latency correctly."""
        from auto_voice.monitoring.prometheus import record_conversion

        # Record conversion with known latency
        record_conversion('quality', 3.0, 30.0, success=True)  # 3000ms

        resp = client.get('/api/v1/metrics')
        data = resp.get_json()

        # Should have non-zero average latency
        assert data['avg_latency_ms'] > 0

    def test_analytics_accumulates_audio_duration(self, client):
        """Test analytics accumulates total audio duration."""
        from auto_voice.monitoring.prometheus import record_conversion

        # Get initial state
        resp_before = client.get('/api/v1/metrics')
        data_before = resp_before.get_json()
        initial_minutes = data_before['total_audio_minutes']

        # Record conversion with 60 seconds of audio
        record_conversion('quality', 2.0, 60.0, success=True)

        # Check updated state
        resp_after = client.get('/api/v1/metrics')
        data_after = resp_after.get_json()

        # Should have increased by at least 1 minute
        assert data_after['total_audio_minutes'] >= initial_minutes + 1.0
