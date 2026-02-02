"""Quality monitoring for voice conversion profiles.

Tracks quality metrics over time and detects degradation.
Integrates with auto-retraining pipeline when quality drops.

Quality Thresholds (from lora-lifecycle-management spec):
- speaker_similarity_min: 0.85
- mcd_max: 4.5
- f0_correlation_min: 0.90
- rtf_max_realtime: 0.30

Cross-Context Dependencies:
- training-inference-integration_20260130: JobManager for retraining
- lora-lifecycle-management_20260201: Auto-retrain triggers
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QualityThresholds:
    """Quality thresholds for voice conversion.

    Based on SOTA research and project requirements.
    """
    speaker_similarity_min: float = 0.85
    mcd_max: float = 4.5
    f0_correlation_min: float = 0.90
    rtf_max: float = 0.30
    mos_min: float = 3.5

    # Degradation detection (rolling average drops by this much)
    degradation_threshold: float = 0.05  # 5% drop

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "QualityThresholds":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QualityMetric:
    """A single quality measurement."""
    profile_id: str
    timestamp: datetime
    speaker_similarity: Optional[float] = None
    mcd: Optional[float] = None
    f0_correlation: Optional[float] = None
    rtf: Optional[float] = None
    mos: Optional[float] = None
    conversion_id: Optional[str] = None  # ID of conversion this was measured from

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "timestamp": self.timestamp.isoformat(),
            "speaker_similarity": self.speaker_similarity,
            "mcd": self.mcd,
            "f0_correlation": self.f0_correlation,
            "rtf": self.rtf,
            "mos": self.mos,
            "conversion_id": self.conversion_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetric":
        return cls(
            profile_id=data["profile_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            speaker_similarity=data.get("speaker_similarity"),
            mcd=data.get("mcd"),
            f0_correlation=data.get("f0_correlation"),
            rtf=data.get("rtf"),
            mos=data.get("mos"),
            conversion_id=data.get("conversion_id"),
        )


@dataclass
class QualityAlert:
    """Alert for quality degradation or threshold violation."""
    alert_id: str
    profile_id: str
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "profile_id": self.profile_id,
            "level": self.level.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class QualityHistory:
    """Quality history for a profile."""
    profile_id: str
    metrics: List[QualityMetric] = field(default_factory=list)
    alerts: List[QualityAlert] = field(default_factory=list)

    def add_metric(self, metric: QualityMetric) -> None:
        """Add a quality metric to history."""
        self.metrics.append(metric)
        # Keep last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]

    def add_alert(self, alert: QualityAlert) -> None:
        """Add an alert to history."""
        self.alerts.append(alert)
        # Keep last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def get_recent_metrics(self, days: int = 7) -> List[QualityMetric]:
        """Get metrics from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return [m for m in self.metrics if m.timestamp >= cutoff]

    def compute_rolling_average(
        self,
        metric_name: str,
        window_size: int = 10
    ) -> Optional[float]:
        """Compute rolling average for a metric."""
        recent = self.metrics[-window_size:] if self.metrics else []
        values = [
            getattr(m, metric_name)
            for m in recent
            if getattr(m, metric_name, None) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "metrics": [m.to_dict() for m in self.metrics],
            "alerts": [a.to_dict() for a in self.alerts],
        }


class QualityMonitor:
    """Monitor voice conversion quality and detect degradation.

    Features:
    - Track quality metrics per profile over time
    - Compute rolling averages
    - Detect quality degradation trends
    - Generate alerts when thresholds violated
    - Trigger auto-retraining when quality drops
    """

    def __init__(
        self,
        storage_dir: Path = Path("data/quality_history"),
        thresholds: Optional[QualityThresholds] = None,
        alert_callback: Optional[Callable[[QualityAlert], None]] = None,
    ):
        """Initialize quality monitor.

        Args:
            storage_dir: Directory for persisting quality history
            thresholds: Quality thresholds (uses defaults if not provided)
            alert_callback: Optional callback for alerts (e.g., WebSocket emit)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or QualityThresholds()
        self.alert_callback = alert_callback

        self._histories: Dict[str, QualityHistory] = {}
        self._alert_counter = 0

        logger.info(f"QualityMonitor initialized with storage at {self.storage_dir}")

    def _get_history_path(self, profile_id: str) -> Path:
        """Get path to history file for a profile."""
        return self.storage_dir / f"{profile_id}_quality.json"

    def _load_history(self, profile_id: str) -> QualityHistory:
        """Load or create quality history for a profile."""
        if profile_id in self._histories:
            return self._histories[profile_id]

        history_path = self._get_history_path(profile_id)

        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
                history = QualityHistory(
                    profile_id=profile_id,
                    metrics=[QualityMetric.from_dict(m) for m in data.get("metrics", [])],
                    alerts=[],  # Don't persist old alerts
                )
            except Exception as e:
                logger.warning(f"Failed to load history for {profile_id}: {e}")
                history = QualityHistory(profile_id=profile_id)
        else:
            history = QualityHistory(profile_id=profile_id)

        self._histories[profile_id] = history
        return history

    def _save_history(self, profile_id: str) -> None:
        """Save quality history for a profile."""
        if profile_id not in self._histories:
            return

        history = self._histories[profile_id]
        history_path = self._get_history_path(profile_id)

        try:
            with open(history_path, "w") as f:
                json.dump(history.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history for {profile_id}: {e}")

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"alert-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alert_counter}"

    def _check_thresholds(self, metric: QualityMetric) -> List[QualityAlert]:
        """Check if metric violates any thresholds."""
        alerts = []

        # Speaker similarity (higher is better)
        if (metric.speaker_similarity is not None and
            metric.speaker_similarity < self.thresholds.speaker_similarity_min):
            alerts.append(QualityAlert(
                alert_id=self._generate_alert_id(),
                profile_id=metric.profile_id,
                level=AlertLevel.WARNING,
                metric_name="speaker_similarity",
                current_value=metric.speaker_similarity,
                threshold=self.thresholds.speaker_similarity_min,
                message=f"Speaker similarity {metric.speaker_similarity:.3f} below threshold {self.thresholds.speaker_similarity_min}",
            ))

        # MCD (lower is better)
        if metric.mcd is not None and metric.mcd > self.thresholds.mcd_max:
            alerts.append(QualityAlert(
                alert_id=self._generate_alert_id(),
                profile_id=metric.profile_id,
                level=AlertLevel.WARNING,
                metric_name="mcd",
                current_value=metric.mcd,
                threshold=self.thresholds.mcd_max,
                message=f"MCD {metric.mcd:.2f} exceeds threshold {self.thresholds.mcd_max}",
            ))

        # F0 correlation (higher is better)
        if (metric.f0_correlation is not None and
            metric.f0_correlation < self.thresholds.f0_correlation_min):
            alerts.append(QualityAlert(
                alert_id=self._generate_alert_id(),
                profile_id=metric.profile_id,
                level=AlertLevel.WARNING,
                metric_name="f0_correlation",
                current_value=metric.f0_correlation,
                threshold=self.thresholds.f0_correlation_min,
                message=f"F0 correlation {metric.f0_correlation:.3f} below threshold {self.thresholds.f0_correlation_min}",
            ))

        # RTF (lower is better)
        if metric.rtf is not None and metric.rtf > self.thresholds.rtf_max:
            alerts.append(QualityAlert(
                alert_id=self._generate_alert_id(),
                profile_id=metric.profile_id,
                level=AlertLevel.INFO,
                metric_name="rtf",
                current_value=metric.rtf,
                threshold=self.thresholds.rtf_max,
                message=f"Real-time factor {metric.rtf:.3f} exceeds threshold {self.thresholds.rtf_max}",
            ))

        return alerts

    def record_metric(
        self,
        profile_id: str,
        speaker_similarity: Optional[float] = None,
        mcd: Optional[float] = None,
        f0_correlation: Optional[float] = None,
        rtf: Optional[float] = None,
        mos: Optional[float] = None,
        conversion_id: Optional[str] = None,
    ) -> List[QualityAlert]:
        """Record a quality metric for a profile.

        Args:
            profile_id: Voice profile ID
            speaker_similarity: Speaker embedding cosine similarity
            mcd: Mel Cepstral Distortion
            f0_correlation: Pitch correlation
            rtf: Real-time factor
            mos: Mean Opinion Score
            conversion_id: Optional ID of conversion this was measured from

        Returns:
            List of alerts generated from this metric
        """
        metric = QualityMetric(
            profile_id=profile_id,
            timestamp=datetime.now(),
            speaker_similarity=speaker_similarity,
            mcd=mcd,
            f0_correlation=f0_correlation,
            rtf=rtf,
            mos=mos,
            conversion_id=conversion_id,
        )

        history = self._load_history(profile_id)
        history.add_metric(metric)

        # Check thresholds
        alerts = self._check_thresholds(metric)

        # Check for degradation
        degradation_alerts = self._check_degradation(profile_id)
        alerts.extend(degradation_alerts)

        # Record alerts
        for alert in alerts:
            history.add_alert(alert)
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")

        # Save history
        self._save_history(profile_id)

        return alerts

    def _check_degradation(self, profile_id: str) -> List[QualityAlert]:
        """Check for quality degradation trends."""
        alerts = []
        history = self._load_history(profile_id)

        if len(history.metrics) < 20:
            return alerts  # Not enough data

        # Compare recent vs historical average
        recent_metrics = history.metrics[-10:]
        older_metrics = history.metrics[-20:-10]

        for metric_name in ["speaker_similarity", "f0_correlation", "mos"]:
            recent_values = [
                getattr(m, metric_name)
                for m in recent_metrics
                if getattr(m, metric_name, None) is not None
            ]
            older_values = [
                getattr(m, metric_name)
                for m in older_metrics
                if getattr(m, metric_name, None) is not None
            ]

            if not recent_values or not older_values:
                continue

            recent_avg = np.mean(recent_values)
            older_avg = np.mean(older_values)

            # Check for significant drop (higher is better for these metrics)
            if older_avg > 0 and (older_avg - recent_avg) / older_avg > self.thresholds.degradation_threshold:
                alerts.append(QualityAlert(
                    alert_id=self._generate_alert_id(),
                    profile_id=profile_id,
                    level=AlertLevel.CRITICAL,
                    metric_name=metric_name,
                    current_value=recent_avg,
                    threshold=older_avg,
                    message=f"Quality degradation detected: {metric_name} dropped from {older_avg:.3f} to {recent_avg:.3f}",
                ))

        # MCD - lower is better, so check for increase
        recent_mcd = [m.mcd for m in recent_metrics if m.mcd is not None]
        older_mcd = [m.mcd for m in older_metrics if m.mcd is not None]

        if recent_mcd and older_mcd:
            recent_avg = np.mean(recent_mcd)
            older_avg = np.mean(older_mcd)

            if older_avg > 0 and (recent_avg - older_avg) / older_avg > self.thresholds.degradation_threshold:
                alerts.append(QualityAlert(
                    alert_id=self._generate_alert_id(),
                    profile_id=profile_id,
                    level=AlertLevel.CRITICAL,
                    metric_name="mcd",
                    current_value=recent_avg,
                    threshold=older_avg,
                    message=f"Quality degradation detected: MCD increased from {older_avg:.2f} to {recent_avg:.2f}",
                ))

        return alerts

    def get_quality_history(
        self,
        profile_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get quality history for a profile.

        Args:
            profile_id: Voice profile ID
            days: Number of days of history to return

        Returns:
            Quality history with metrics and statistics
        """
        history = self._load_history(profile_id)
        recent_metrics = history.get_recent_metrics(days)

        # Compute statistics
        stats = {}
        for metric_name in ["speaker_similarity", "mcd", "f0_correlation", "rtf", "mos"]:
            values = [
                getattr(m, metric_name)
                for m in recent_metrics
                if getattr(m, metric_name, None) is not None
            ]
            if values:
                stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }

        return {
            "profile_id": profile_id,
            "period_days": days,
            "total_metrics": len(recent_metrics),
            "statistics": stats,
            "metrics": [m.to_dict() for m in recent_metrics],
            "recent_alerts": [a.to_dict() for a in history.alerts[-10:]],
        }

    def get_quality_summary(self, profile_id: str) -> Dict[str, Any]:
        """Get quality summary for a profile.

        Args:
            profile_id: Voice profile ID

        Returns:
            Summary with current status and recommendations
        """
        history = self._load_history(profile_id)

        # Compute rolling averages
        rolling = {}
        for metric_name in ["speaker_similarity", "mcd", "f0_correlation", "mos"]:
            avg = history.compute_rolling_average(metric_name)
            if avg is not None:
                rolling[metric_name] = avg

        # Determine status
        status = "healthy"
        recommendations = []

        if "speaker_similarity" in rolling:
            if rolling["speaker_similarity"] < self.thresholds.speaker_similarity_min:
                status = "degraded"
                recommendations.append("Retrain LoRA to improve speaker similarity")

        if "mcd" in rolling:
            if rolling["mcd"] > self.thresholds.mcd_max:
                status = "degraded"
                recommendations.append("Retrain to reduce spectral distortion")

        # Check for recent critical alerts
        critical_alerts = [
            a for a in history.alerts[-20:]
            if a.level == AlertLevel.CRITICAL and not a.acknowledged
        ]
        if critical_alerts:
            status = "critical"
            recommendations.insert(0, "Address critical quality alerts")

        return {
            "profile_id": profile_id,
            "status": status,
            "rolling_averages": rolling,
            "thresholds": self.thresholds.to_dict(),
            "recommendations": recommendations,
            "unacknowledged_alerts": len(critical_alerts),
            "total_metrics": len(history.metrics),
        }

    def get_all_profiles_status(self) -> List[Dict[str, Any]]:
        """Get status summary for all monitored profiles.

        Returns:
            List of profile summaries
        """
        # Find all history files
        profiles = []

        for history_file in self.storage_dir.glob("*_quality.json"):
            profile_id = history_file.stem.replace("_quality", "")
            summary = self.get_quality_summary(profile_id)
            profiles.append(summary)

        # Sort by status (critical first, then degraded, then healthy)
        status_order = {"critical": 0, "degraded": 1, "healthy": 2}
        profiles.sort(key=lambda p: status_order.get(p["status"], 3))

        return profiles

    def detect_degradation(self, profile_id: str) -> Dict[str, Any]:
        """Explicitly check for quality degradation.

        Args:
            profile_id: Voice profile ID

        Returns:
            Degradation analysis result
        """
        alerts = self._check_degradation(profile_id)
        history = self._load_history(profile_id)

        # Record any new alerts
        for alert in alerts:
            history.add_alert(alert)
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")

        self._save_history(profile_id)

        return {
            "profile_id": profile_id,
            "degradation_detected": len(alerts) > 0,
            "alerts": [a.to_dict() for a in alerts],
            "recommendation": "Retrain LoRA adapter" if alerts else None,
        }


# Global instance
_global_monitor: Optional[QualityMonitor] = None


def get_quality_monitor() -> QualityMonitor:
    """Get or create global QualityMonitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = QualityMonitor()
    return _global_monitor
