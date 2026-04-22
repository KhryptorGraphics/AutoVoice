#!/usr/bin/env python3
"""LoRA Lifecycle Audit Script.

Examines all existing LoRAs to determine training status, quality metrics,
freshness, and retraining recommendations.

Cross-Context Dependencies:
- speaker-diarization_20260130: WavLM embeddings (256-dim)
- training-inference-integration_20260130: AdapterManager
- voice-profile-training_20260124: Profile management

Usage:
    python scripts/audit_loras.py [--json] [--markdown] [--verbose]
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_voice.storage.paths import (
    resolve_data_dir,
    resolve_diarized_audio_dir,
    resolve_profiles_dir,
    resolve_trained_models_dir,
)


@dataclass
class LoRAStatus:
    """Status information for a LoRA adapter."""
    profile_id: str
    profile_name: str
    has_adapter: bool
    adapter_path: Optional[str]
    adapter_type: str  # 'standard', 'nvfp4', 'hq', 'none'
    sample_count: int
    training_epochs: int
    loss_final: float
    created_at: Optional[str]
    trained_at: Optional[str]
    days_since_training: Optional[int]
    is_stale: bool  # >30 days old
    needs_training: bool  # Has samples but no adapter
    needs_retrain: bool  # New samples since training or stale
    quality_ok: bool  # Meets quality thresholds
    speaker_similarity: Optional[float]
    mcd: Optional[float]
    issues: List[str]
    recommendations: List[str]


@dataclass
class AuditSummary:
    """Summary of the LoRA audit."""
    total_profiles: int
    profiles_with_adapters: int
    profiles_needing_training: int
    profiles_needing_retrain: int
    stale_adapters: int
    low_quality_adapters: int
    total_adapters: int
    adapter_types: Dict[str, int]


def resolve_runtime_paths(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve audit runtime paths from the shared data-dir contract."""

    resolved_data_dir = resolve_data_dir(str(data_dir) if data_dir is not None else None)
    return {
        "data_dir": resolved_data_dir,
        "voice_profiles_dir": resolve_profiles_dir(data_dir=str(resolved_data_dir)),
        "trained_models_dir": resolve_trained_models_dir(data_dir=str(resolved_data_dir)),
        "diarized_dir": resolve_diarized_audio_dir(data_dir=str(resolved_data_dir)),
    }


class LoRAAuditor:
    """Audits LoRA adapters across voice profiles.

    Thresholds (from lora-lifecycle-management track):
    - min_samples_for_training: 5
    - retrain_new_samples: 3
    - freshness_days: 30
    - speaker_similarity_min: 0.85
    - mcd_max: 4.5
    - samples_for_full_model: 50
    """

    # Thresholds from spec
    MIN_SAMPLES_FOR_TRAINING = 5
    RETRAIN_NEW_SAMPLES = 3
    FRESHNESS_DAYS = 30
    SPEAKER_SIMILARITY_MIN = 0.85
    MCD_MAX = 4.5
    SAMPLES_FOR_FULL_MODEL = 50

    def __init__(
        self,
        data_dir: str | Path | None = None,
        verbose: bool = False
    ):
        paths = resolve_runtime_paths(data_dir)
        self.data_dir = paths["data_dir"]
        self.verbose = verbose

        self.voice_profiles_dir = paths["voice_profiles_dir"]
        self.trained_models_dir = paths["trained_models_dir"]
        self.diarized_dir = paths["diarized_dir"]

        self._statuses: List[LoRAStatus] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AUDIT] {msg}")

    def _find_all_profiles(self) -> List[Dict[str, Any]]:
        """Find all voice profiles including diarized ones."""
        profiles = []

        # UUID-based profiles (e.g., c572d02c-...-...-...-...json)
        for json_file in self.voice_profiles_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if "profile_id" in data:
                        data["source"] = "uuid_profile"
                        data["source_path"] = str(json_file)
                        profiles.append(data)
            except Exception as e:
                self._log(f"Error reading {json_file}: {e}")

        # Named artist directories with speaker_profiles.json
        for artist_dir in self.voice_profiles_dir.iterdir():
            if artist_dir.is_dir():
                speaker_file = artist_dir / "speaker_profiles.json"
                if speaker_file.exists():
                    try:
                        with open(speaker_file) as f:
                            speaker_data = json.load(f)
                            for speaker_id, speaker_info in speaker_data.items():
                                profile = {
                                    "profile_id": speaker_info.get("profile_id"),
                                    "name": speaker_info.get("profile_name", f"{artist_dir.name}_{speaker_id}"),
                                    "created_at": speaker_info.get("created_at"),
                                    "is_primary": speaker_info.get("is_primary", False),
                                    "source": "diarized_artist",
                                    "source_path": str(speaker_file),
                                    "artist_dir": artist_dir.name,
                                    "speaker_id": speaker_id,
                                }
                                profiles.append(profile)
                    except Exception as e:
                        self._log(f"Error reading {speaker_file}: {e}")

        return profiles

    def _find_adapter(self, profile_id: str) -> Tuple[Optional[Path], str]:
        """Find adapter for profile, checking all adapter types."""
        # Check standard adapter
        standard_path = self.trained_models_dir / f"{profile_id}_adapter.pt"
        if standard_path.exists():
            return standard_path, "standard"

        # Check nvfp4 adapter
        nvfp4_path = self.trained_models_dir / "nvfp4" / f"{profile_id}_nvfp4_lora.pt"
        if nvfp4_path.exists():
            return nvfp4_path, "nvfp4"

        # Check HQ adapter
        hq_path = self.trained_models_dir / "hq" / f"{profile_id}_hq_lora.pt"
        if hq_path.exists():
            return hq_path, "hq"

        return None, "none"

    def _count_samples(self, profile: Dict[str, Any]) -> int:
        """Count training samples for a profile."""
        # Check training_sample_count in profile
        sample_count = profile.get("training_sample_count", 0)
        if sample_count > 0:
            return sample_count

        # Check diarized samples for artist profiles
        artist_dir = profile.get("artist_dir")
        if artist_dir:
            diarized_path = self.diarized_dir / artist_dir
            if diarized_path.exists():
                # Count segment files
                segments = list(diarized_path.glob("*_segment_*.wav"))
                return len(segments)

        return sample_count

    def _get_training_timestamp(self, adapter_path: Optional[Path]) -> Optional[datetime]:
        """Get training timestamp from adapter file."""
        if adapter_path and adapter_path.exists():
            mtime = adapter_path.stat().st_mtime
            return datetime.fromtimestamp(mtime)
        return None

    def _get_quality_metrics(self, profile_id: str) -> Dict[str, Optional[float]]:
        """Get quality metrics for a profile's adapter."""
        # Check for quality metrics in profile metadata
        profile_path = self.voice_profiles_dir / f"{profile_id}.json"
        if profile_path.exists():
            try:
                with open(profile_path) as f:
                    data = json.load(f)
                    return {
                        "speaker_similarity": data.get("speaker_similarity"),
                        "mcd": data.get("mcd"),
                    }
            except Exception:
                pass

        return {"speaker_similarity": None, "mcd": None}

    def audit_profile(self, profile: Dict[str, Any]) -> LoRAStatus:
        """Audit a single profile."""
        profile_id = profile.get("profile_id", "unknown")
        profile_name = profile.get("name", profile_id)

        self._log(f"Auditing profile: {profile_name} ({profile_id})")

        # Find adapter
        adapter_path, adapter_type = self._find_adapter(profile_id)
        has_adapter = adapter_path is not None

        # Count samples
        sample_count = self._count_samples(profile)

        # Get training info
        created_at = profile.get("created_at")
        trained_at = None
        days_since_training = None

        if adapter_path:
            training_ts = self._get_training_timestamp(adapter_path)
            if training_ts:
                trained_at = training_ts.isoformat()
                days_since_training = (datetime.now() - training_ts).days

        # Check quality metrics
        quality_metrics = self._get_quality_metrics(profile_id)
        speaker_similarity = quality_metrics.get("speaker_similarity")
        mcd = quality_metrics.get("mcd")

        # Determine status flags
        is_stale = days_since_training is not None and days_since_training > self.FRESHNESS_DAYS
        needs_training = sample_count >= self.MIN_SAMPLES_FOR_TRAINING and not has_adapter
        needs_retrain = False
        quality_ok = True

        # Check quality thresholds
        if speaker_similarity is not None and speaker_similarity < self.SPEAKER_SIMILARITY_MIN:
            quality_ok = False
        if mcd is not None and mcd > self.MCD_MAX:
            quality_ok = False

        # Check retrain conditions
        if has_adapter:
            if is_stale:
                needs_retrain = True
            if not quality_ok:
                needs_retrain = True

        # Generate issues and recommendations
        issues = []
        recommendations = []

        if needs_training:
            issues.append(f"Has {sample_count} samples but no adapter trained")
            recommendations.append("Train initial LoRA adapter")

        if is_stale:
            issues.append(f"Adapter is {days_since_training} days old (threshold: {self.FRESHNESS_DAYS})")
            recommendations.append("Consider retraining with latest samples")

        if not quality_ok:
            if speaker_similarity and speaker_similarity < self.SPEAKER_SIMILARITY_MIN:
                issues.append(f"Speaker similarity {speaker_similarity:.3f} below threshold {self.SPEAKER_SIMILARITY_MIN}")
            if mcd and mcd > self.MCD_MAX:
                issues.append(f"MCD {mcd:.2f} above threshold {self.MCD_MAX}")
            recommendations.append("Retrain with quality optimization")

        if sample_count >= self.SAMPLES_FOR_FULL_MODEL and adapter_type != "hq":
            recommendations.append(f"Has {sample_count} samples - consider training full HQ model")

        if sample_count < self.MIN_SAMPLES_FOR_TRAINING and not has_adapter:
            issues.append(f"Only {sample_count} samples (need {self.MIN_SAMPLES_FOR_TRAINING} for training)")
            recommendations.append("Collect more audio samples")

        # Get training epochs and loss from profile
        training_epochs = profile.get("training_epochs", 0)
        loss_final = profile.get("loss_final", 0.0)

        return LoRAStatus(
            profile_id=profile_id,
            profile_name=profile_name,
            has_adapter=has_adapter,
            adapter_path=str(adapter_path) if adapter_path else None,
            adapter_type=adapter_type,
            sample_count=sample_count,
            training_epochs=training_epochs,
            loss_final=loss_final,
            created_at=created_at,
            trained_at=trained_at,
            days_since_training=days_since_training,
            is_stale=is_stale,
            needs_training=needs_training,
            needs_retrain=needs_retrain,
            quality_ok=quality_ok,
            speaker_similarity=speaker_similarity,
            mcd=mcd,
            issues=issues,
            recommendations=recommendations,
        )

    def audit_all(self) -> Tuple[List[LoRAStatus], AuditSummary]:
        """Audit all profiles and return statuses with summary."""
        profiles = self._find_all_profiles()
        self._log(f"Found {len(profiles)} profiles to audit")

        statuses = []
        for profile in profiles:
            status = self.audit_profile(profile)
            statuses.append(status)

        self._statuses = statuses

        # Generate summary
        adapter_types = {"standard": 0, "nvfp4": 0, "hq": 0, "none": 0}
        for s in statuses:
            adapter_types[s.adapter_type] += 1

        summary = AuditSummary(
            total_profiles=len(statuses),
            profiles_with_adapters=sum(1 for s in statuses if s.has_adapter),
            profiles_needing_training=sum(1 for s in statuses if s.needs_training),
            profiles_needing_retrain=sum(1 for s in statuses if s.needs_retrain),
            stale_adapters=sum(1 for s in statuses if s.is_stale),
            low_quality_adapters=sum(1 for s in statuses if not s.quality_ok and s.has_adapter),
            total_adapters=sum(1 for s in statuses if s.has_adapter),
            adapter_types=adapter_types,
        )

        return statuses, summary

    def to_json(self) -> str:
        """Export audit results as JSON."""
        statuses, summary = self.audit_all()
        return json.dumps({
            "audit_timestamp": datetime.now().isoformat(),
            "summary": asdict(summary),
            "profiles": [asdict(s) for s in statuses],
        }, indent=2)

    def to_markdown(self) -> str:
        """Export audit results as Markdown."""
        statuses, summary = self.audit_all()

        lines = [
            "# LoRA Audit Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"- **Total Profiles:** {summary.total_profiles}",
            f"- **Profiles with Adapters:** {summary.profiles_with_adapters}",
            f"- **Needing Training:** {summary.profiles_needing_training}",
            f"- **Needing Retrain:** {summary.profiles_needing_retrain}",
            f"- **Stale Adapters (>{self.FRESHNESS_DAYS} days):** {summary.stale_adapters}",
            f"- **Low Quality Adapters:** {summary.low_quality_adapters}",
            "",
            "### Adapter Types",
            "",
            f"- Standard: {summary.adapter_types['standard']}",
            f"- NVFP4: {summary.adapter_types['nvfp4']}",
            f"- HQ: {summary.adapter_types['hq']}",
            f"- None: {summary.adapter_types['none']}",
            "",
        ]

        # Profiles needing action
        action_needed = [s for s in statuses if s.needs_training or s.needs_retrain or not s.quality_ok]
        if action_needed:
            lines.extend([
                "## Action Required",
                "",
                "| Profile | Issues | Recommendations |",
                "|---------|--------|-----------------|",
            ])
            for s in action_needed:
                issues = "; ".join(s.issues) if s.issues else "None"
                recs = "; ".join(s.recommendations) if s.recommendations else "None"
                lines.append(f"| {s.profile_name} | {issues} | {recs} |")
            lines.append("")

        # All profiles
        lines.extend([
            "## All Profiles",
            "",
            "| Profile | Adapter | Type | Samples | Age (days) | Quality |",
            "|---------|---------|------|---------|------------|---------|",
        ])
        for s in statuses:
            adapter = "✓" if s.has_adapter else "✗"
            quality = "✓" if s.quality_ok else "⚠️"
            age = str(s.days_since_training) if s.days_since_training else "-"
            lines.append(f"| {s.profile_name} | {adapter} | {s.adapter_type} | {s.sample_count} | {age} | {quality} |")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit LoRA adapters across voice profiles")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory path")

    args = parser.parse_args()

    auditor = LoRAAuditor(
        data_dir=args.data_dir,
        verbose=args.verbose,
    )

    if args.json:
        print(auditor.to_json())
    elif args.markdown:
        print(auditor.to_markdown())
    else:
        # Default: human-readable summary
        statuses, summary = auditor.audit_all()

        print("\n" + "=" * 60)
        print("LoRA AUDIT REPORT")
        print("=" * 60)
        print(f"\nTotal Profiles: {summary.total_profiles}")
        print(f"With Adapters:  {summary.profiles_with_adapters}")
        print(f"Need Training:  {summary.profiles_needing_training}")
        print(f"Need Retrain:   {summary.profiles_needing_retrain}")
        print(f"Stale (>{LoRAAuditor.FRESHNESS_DAYS}d): {summary.stale_adapters}")
        print(f"Low Quality:    {summary.low_quality_adapters}")

        print("\nAdapter Types:")
        for atype, count in summary.adapter_types.items():
            if count > 0:
                print(f"  {atype}: {count}")

        # Show action items
        action_needed = [s for s in statuses if s.needs_training or s.needs_retrain]
        if action_needed:
            print("\n" + "-" * 60)
            print("ACTION REQUIRED:")
            print("-" * 60)
            for s in action_needed:
                print(f"\n{s.profile_name} ({s.profile_id}):")
                for issue in s.issues:
                    print(f"  ⚠️  {issue}")
                for rec in s.recommendations:
                    print(f"  →  {rec}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
