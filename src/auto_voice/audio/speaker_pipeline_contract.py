"""Shared defaults and result contracts for the speaker pipeline."""

from typing import Any, Dict, List, Mapping, Sequence

DEFAULT_SPEAKER_PIPELINE_ARTISTS: tuple[str, ...] = (
    "conor_maynard",
    "william_singe",
)


def get_default_speaker_pipeline_artists() -> List[str]:
    """Return a fresh list of default artists for speaker pipeline runs."""
    return list(DEFAULT_SPEAKER_PIPELINE_ARTISTS)


def build_speaker_extraction_stats() -> Dict[str, Any]:
    """Canonical matcher extraction stats shape."""
    return {
        "tracks_processed": 0,
        "embeddings_extracted": 0,
        "primary_speakers": 0,
        "featured_speakers": 0,
        "errors": [],
    }


def normalize_speaker_extraction_job_result(
    result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalize extraction-job result fields while preserving existing data."""
    normalized = dict(result or {})
    tracks_processed = int(normalized.get("tracks_processed") or normalized.get("tracks") or 0)
    tracks_total = int(normalized.get("tracks_total") or tracks_processed)
    normalized["tracks_processed"] = tracks_processed
    normalized["tracks_total"] = max(tracks_processed, tracks_total)
    normalized["speakers_detected"] = list(normalized.get("speakers_detected") or [])
    return normalized


def build_speaker_auto_match_stats() -> Dict[str, Any]:
    """Canonical matcher auto-match stats shape."""
    return {
        "clusters_processed": 0,
        "matches_found": 0,
        "matches_made": [],
    }


def build_speaker_metadata_stats() -> Dict[str, Any]:
    """Canonical YouTube metadata population stats shape."""
    return {
        "tracks_processed": 0,
        "tracks_with_metadata": 0,
        "featured_artists_found": 0,
        "errors": [],
    }


def normalize_speaker_metadata_stats(
    stats: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return metadata stats with canonical keys present."""
    normalized = build_speaker_metadata_stats()
    if not stats:
        return normalized

    normalized.update(stats)
    normalized["featured_artists_found"] = int(
        stats.get("featured_artists_found") or stats.get("featured_found") or 0
    )
    normalized["errors"] = list(stats.get("errors") or [])
    return normalized


def normalize_speaker_auto_match_stats(
    stats: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return matcher auto-match stats with the canonical keys present."""
    normalized = build_speaker_auto_match_stats()
    if not stats:
        return normalized

    normalized.update(stats)
    normalized["matches_made"] = list(stats.get("matches_made") or [])
    return normalized


def build_speaker_clustering_stats(
    clusters: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Canonical matcher clustering stats shape."""
    normalized_clusters = []
    for cluster in clusters or ():
        normalized_clusters.append(
            {
                "cluster_id": cluster["cluster_id"],
                "member_count": cluster["member_count"],
                "duration_sec": cluster.get("duration_sec", cluster.get("total_duration_sec")),
            }
        )

    return {
        "clusters_created": len(normalized_clusters),
        "clusters": normalized_clusters,
    }


def build_speaker_pipeline_run_stats() -> Dict[str, Any]:
    """Canonical full matcher pipeline stats shape."""
    return {
        "artists": {},
        "clustering": build_speaker_clustering_stats(),
        "matching": build_speaker_auto_match_stats(),
    }
