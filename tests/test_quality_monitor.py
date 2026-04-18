"""Targeted tests for monitoring.quality_monitor."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from auto_voice.monitoring.quality_monitor import (
    AlertLevel,
    QualityAlert,
    QualityHistory,
    QualityMetric,
    QualityMonitor,
    QualityThresholds,
    get_quality_monitor,
)


class TestQualityDataModels:
    def test_thresholds_and_metric_round_trip(self):
        thresholds = QualityThresholds(mcd_max=5.0, mos_min=4.0)
        threshold_dict = thresholds.to_dict()
        loaded_thresholds = QualityThresholds.from_dict(threshold_dict)

        metric = QualityMetric(
            profile_id="profile-1",
            timestamp=datetime(2026, 4, 18, 12, 0, 0),
            speaker_similarity=0.91,
            mcd=3.8,
            f0_correlation=0.95,
            rtf=0.12,
            mos=4.4,
            conversion_id="conv-1",
        )
        metric_dict = metric.to_dict()
        loaded_metric = QualityMetric.from_dict(metric_dict)

        assert loaded_thresholds.mcd_max == 5.0
        assert loaded_thresholds.mos_min == 4.0
        assert loaded_metric.profile_id == "profile-1"
        assert loaded_metric.timestamp == datetime(2026, 4, 18, 12, 0, 0)
        assert loaded_metric.conversion_id == "conv-1"

    def test_history_adders_truncate_and_compute_rolling_average(self):
        history = QualityHistory(profile_id="profile-1")

        for idx in range(1005):
            history.add_metric(
                QualityMetric(
                    profile_id="profile-1",
                    timestamp=datetime.now(),
                    speaker_similarity=0.8 + idx * 0.0001,
                )
            )
        for idx in range(105):
            history.add_alert(
                QualityAlert(
                    alert_id=f"alert-{idx}",
                    profile_id="profile-1",
                    level=AlertLevel.INFO,
                    metric_name="rtf",
                    current_value=0.2,
                    threshold=0.3,
                    message="rtf high",
                )
            )

        rolling = history.compute_rolling_average("speaker_similarity", window_size=5)

        assert len(history.metrics) == 1000
        assert len(history.alerts) == 100
        assert rolling is not None
        assert history.compute_rolling_average("mcd") is None


class TestQualityMonitor:
    def test_load_history_handles_invalid_json_and_save_history_writes_file(self, tmp_path):
        monitor = QualityMonitor(storage_dir=tmp_path)
        bad_file = tmp_path / "broken_quality.json"
        bad_file.write_text("{not valid json")

        history = monitor._load_history("broken")
        assert history.profile_id == "broken"
        assert history.metrics == []

        monitor._histories["broken"] = history
        history.add_metric(QualityMetric(profile_id="broken", timestamp=datetime.now(), mos=4.0))
        monitor._save_history("broken")

        saved = json.loads(bad_file.read_text())
        assert saved["profile_id"] == "broken"
        assert saved["metrics"][0]["mos"] == 4.0

    def test_record_metric_generates_threshold_alerts_and_uses_callback(self, tmp_path):
        callback = Mock()
        monitor = QualityMonitor(storage_dir=tmp_path, alert_callback=callback)

        alerts = monitor.record_metric(
            profile_id="profile-1",
            speaker_similarity=0.7,
            mcd=5.0,
            f0_correlation=0.8,
            rtf=0.4,
            mos=3.2,
            conversion_id="conv-1",
        )

        alert_names = {alert.metric_name for alert in alerts}
        assert {"speaker_similarity", "mcd", "f0_correlation", "rtf"} == alert_names
        assert callback.call_count == 4
        saved_path = tmp_path / "profile-1_quality.json"
        assert saved_path.exists()

    def test_record_metric_swallow_callback_errors(self, tmp_path):
        callback = Mock(side_effect=RuntimeError("callback boom"))
        monitor = QualityMonitor(storage_dir=tmp_path, alert_callback=callback)

        alerts = monitor.record_metric(profile_id="profile-1", speaker_similarity=0.7)

        assert len(alerts) == 1

    def test_detect_degradation_and_history_summary(self, tmp_path):
        monitor = QualityMonitor(storage_dir=tmp_path)
        base = datetime.now() - timedelta(days=10)
        history = QualityHistory(profile_id="profile-1")

        for idx in range(10):
            history.add_metric(
                QualityMetric(
                    profile_id="profile-1",
                    timestamp=base + timedelta(hours=idx),
                    speaker_similarity=0.96,
                    f0_correlation=0.97,
                    mos=4.5,
                    mcd=3.0,
                    rtf=0.15,
                )
            )
        for idx in range(10):
            history.add_metric(
                QualityMetric(
                    profile_id="profile-1",
                    timestamp=base + timedelta(days=1, hours=idx),
                    speaker_similarity=0.70,
                    f0_correlation=0.72,
                    mos=3.4,
                    mcd=4.8,
                    rtf=0.18,
                )
            )

        monitor._histories["profile-1"] = history
        result = monitor.detect_degradation("profile-1")
        summary = monitor.get_quality_summary("profile-1")
        quality_history = monitor.get_quality_history("profile-1", days=30)

        assert result["degradation_detected"] is True
        assert result["recommendation"] == "Retrain LoRA adapter"
        assert summary["status"] == "critical"
        assert summary["unacknowledged_alerts"] > 0
        assert "speaker_similarity" in quality_history["statistics"]
        assert quality_history["total_metrics"] == 20

    def test_get_all_profiles_status_sorts_critical_before_healthy(self, tmp_path):
        monitor = QualityMonitor(storage_dir=tmp_path)

        critical_history = QualityHistory(
            profile_id="critical-profile",
            metrics=[QualityMetric(profile_id="critical-profile", timestamp=datetime.now(), speaker_similarity=0.6)],
            alerts=[
                QualityAlert(
                    alert_id="alert-1",
                    profile_id="critical-profile",
                    level=AlertLevel.CRITICAL,
                    metric_name="speaker_similarity",
                    current_value=0.6,
                    threshold=0.85,
                    message="degraded",
                )
            ],
        )
        healthy_history = QualityHistory(
            profile_id="healthy-profile",
            metrics=[QualityMetric(profile_id="healthy-profile", timestamp=datetime.now(), speaker_similarity=0.95)],
            alerts=[],
        )

        monitor._histories["critical-profile"] = critical_history
        monitor._histories["healthy-profile"] = healthy_history
        monitor._save_history("critical-profile")
        monitor._save_history("healthy-profile")

        statuses = monitor.get_all_profiles_status()

        assert [entry["profile_id"] for entry in statuses] == ["critical-profile", "healthy-profile"]
        assert statuses[0]["status"] == "critical"
        assert statuses[1]["status"] == "healthy"

    def test_get_quality_monitor_returns_singleton(self, tmp_path):
        import auto_voice.monitoring.quality_monitor as quality_module

        with patch.object(quality_module, "_global_monitor", None):
            first = get_quality_monitor()
            second = get_quality_monitor()

        assert first is second
