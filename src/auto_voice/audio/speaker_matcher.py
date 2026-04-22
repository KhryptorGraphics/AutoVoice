"""
Cross-track speaker matching and clustering for AutoVoice.

This module provides:
- Speaker embedding extraction using WavLM
- Cross-track speaker clustering using cosine similarity
- Auto-matching clusters to featured artist names from metadata

Usage:
    from auto_voice.audio.speaker_matcher import SpeakerMatcher

    matcher = SpeakerMatcher()
    matcher.extract_embeddings_for_artist('conor_maynard')
    clusters = matcher.cluster_speakers(threshold=0.85)
    matcher.auto_match_clusters_to_artists()
"""

import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from auto_voice.storage.paths import (
    resolve_data_dir,
    resolve_diarized_audio_dir,
    resolve_separated_audio_dir,
)

logger = logging.getLogger(__name__)


class SpeakerMatcher:
    """Cross-track speaker matching and clustering.

    Uses WavLM-based embeddings to identify the same speaker across
    different tracks, enabling consistent voice profile assignment.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_duration: float = 30.0,  # Minimum 30s total for a valid cluster
        device: str = 'cuda',
        data_dir: Optional[Path] = None,
    ):
        """Initialize the speaker matcher.

        Args:
            similarity_threshold: Cosine similarity threshold for clustering (0-1)
            min_cluster_duration: Minimum total duration for a valid cluster (seconds)
            device: Device for embedding extraction ('cuda' or 'cpu')
            data_dir: Base runtime data directory for canonical input defaults
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_duration = min_cluster_duration
        self.device = device
        self.data_dir = resolve_data_dir(str(data_dir) if data_dir is not None else None)
        self._encoder = None

    def _get_encoder(self):
        """Lazy-load the WavLM encoder."""
        if self._encoder is None:
            from .speaker_diarization import SpeakerDiarizer
            # Use the same encoder as diarization for consistency
            diarizer = SpeakerDiarizer(device=self.device)
            self._encoder = diarizer
        return self._encoder

    def extract_embedding_from_audio(
        self,
        audio_path: Path,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Extract speaker embedding from audio file or segment.

        Args:
            audio_path: Path to audio file
            start_sec: Start time in seconds (optional)
            end_sec: End time in seconds (optional)

        Returns:
            512-dim normalized embedding
        """
        import librosa

        # Load audio
        if start_sec is not None or end_sec is not None:
            duration = None if end_sec is None else (end_sec - start_sec)
            audio, sr = librosa.load(
                str(audio_path),
                sr=16000,
                mono=True,
                offset=start_sec or 0,
                duration=duration,
            )
        else:
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        # Get embedding
        encoder = self._get_encoder()
        embedding = encoder.extract_embedding(audio)

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def extract_embeddings_for_artist(
        self,
        artist_name: str,
        separated_dir: Optional[Path] = None,
        diarized_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Extract and store embeddings for all speakers in an artist's tracks.

        Args:
            artist_name: Artist name (e.g., "conor_maynard")
            separated_dir: Directory with separated vocals
            diarized_dir: Directory with diarization JSONs

        Returns:
            Statistics dict
        """
        import json
        from ..db.operations import (
            upsert_track, add_speaker_embedding, get_track
        )

        if separated_dir is None:
            separated_dir = resolve_separated_audio_dir(
                data_dir=str(self.data_dir),
                artist_name=artist_name,
            )
        if diarized_dir is None:
            diarized_dir = resolve_diarized_audio_dir(
                data_dir=str(self.data_dir),
                artist_name=artist_name,
            )

        stats = {
            'tracks_processed': 0,
            'embeddings_extracted': 0,
            'primary_speakers': 0,
            'featured_speakers': 0,
            'errors': [],
        }

        # Find all diarization files
        diarization_files = list(diarized_dir.glob('*_diarization.json'))
        logger.info(f"Found {len(diarization_files)} diarization files for {artist_name}")

        for diar_file in diarization_files:
            try:
                # Load diarization
                with open(diar_file) as f:
                    diarization = json.load(f)

                video_id = diar_file.stem.replace('_diarization', '').replace('_vocals', '')

                # Find corresponding audio file
                audio_file = None
                for pattern in [f'{video_id}_vocals.wav', f'{video_id}.wav']:
                    candidate = separated_dir / pattern
                    if candidate.exists():
                        audio_file = candidate
                        break

                if not audio_file:
                    stats['errors'].append(f"Audio not found for {video_id}")
                    continue

                # Ensure track exists in database
                upsert_track(
                    track_id=video_id,
                    artist_name=artist_name,
                    vocals_path=str(audio_file),
                    diarization_path=str(diar_file),
                )

                stats['tracks_processed'] += 1

                # Group segments by speaker
                speaker_segments = defaultdict(list)
                for seg in diarization.get('segments', []):
                    speaker_segments[seg['speaker']].append(seg)

                # Calculate duration and determine primary speaker
                speaker_durations = {}
                for speaker_id, segs in speaker_segments.items():
                    total_dur = sum(s['end'] - s['start'] for s in segs)
                    speaker_durations[speaker_id] = total_dur

                primary_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0] if speaker_durations else None

                # Extract embedding for each speaker
                for speaker_id, segs in speaker_segments.items():
                    duration = speaker_durations[speaker_id]

                    # Skip very short segments
                    if duration < 5.0:
                        continue

                    # Use the longest segment for embedding
                    longest_seg = max(segs, key=lambda s: s['end'] - s['start'])

                    try:
                        embedding = self.extract_embedding_from_audio(
                            audio_file,
                            start_sec=longest_seg['start'],
                            end_sec=longest_seg['end'],
                        )

                        is_primary = (speaker_id == primary_speaker)

                        add_speaker_embedding(
                            track_id=video_id,
                            speaker_id=speaker_id,
                            embedding=embedding,
                            duration_sec=duration,
                            is_primary=is_primary,
                        )

                        stats['embeddings_extracted'] += 1
                        if is_primary:
                            stats['primary_speakers'] += 1
                        else:
                            stats['featured_speakers'] += 1

                        logger.debug(f"Extracted embedding for {video_id}/{speaker_id} "
                                     f"(duration={duration:.1f}s, primary={is_primary})")

                    except Exception as e:
                        stats['errors'].append(f"Embedding extraction failed for {video_id}/{speaker_id}: {e}")

            except Exception as e:
                stats['errors'].append(f"Failed to process {diar_file.name}: {e}")

        return stats

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding (normalized)
            emb2: Second embedding (normalized)

        Returns:
            Cosine similarity (-1 to 1)
        """
        return float(np.dot(emb1, emb2))

    def cluster_speakers(
        self,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Cluster all unclustered speaker embeddings.

        Uses agglomerative clustering with cosine similarity.

        Args:
            threshold: Similarity threshold (default: self.similarity_threshold)

        Returns:
            List of cluster dicts with member embeddings
        """
        from ..db.operations import (
            find_unclustered_embeddings, create_cluster, add_to_cluster
        )

        if threshold is None:
            threshold = self.similarity_threshold

        # Get all unclustered embeddings
        embeddings = find_unclustered_embeddings()
        logger.info(f"Found {len(embeddings)} unclustered embeddings")

        if not embeddings:
            return []

        # Build similarity matrix
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.cosine_similarity(embeddings[i]['embedding'], embeddings[j]['embedding'])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            similarity_matrix[i, i] = 1.0

        # Simple agglomerative clustering
        clusters = []
        assigned = set()

        for i in range(n):
            if i in assigned:
                continue

            # Find all embeddings similar to this one
            cluster_indices = [i]
            assigned.add(i)

            for j in range(i + 1, n):
                if j in assigned:
                    continue

                # Check if similar to any member of current cluster
                for k in cluster_indices:
                    if similarity_matrix[j, k] >= threshold:
                        cluster_indices.append(j)
                        assigned.add(j)
                        break

            # Create cluster
            cluster_members = [embeddings[idx] for idx in cluster_indices]
            total_duration = sum(m['duration_sec'] or 0 for m in cluster_members)

            # Skip clusters with too little audio
            if total_duration < self.min_cluster_duration:
                continue

            # Create cluster in database
            cluster_id = create_cluster(
                name=f"Unknown Speaker {len(clusters) + 1}",
                is_verified=False,
            )

            # Add members with confidence scores
            for idx in cluster_indices:
                # Calculate average similarity to cluster
                sims = [similarity_matrix[idx, k] for k in cluster_indices if k != idx]
                confidence = np.mean(sims) if sims else 1.0

                add_to_cluster(
                    cluster_id=cluster_id,
                    embedding_id=embeddings[idx]['id'],
                    confidence=confidence,
                )

            clusters.append({
                'cluster_id': cluster_id,
                'member_count': len(cluster_indices),
                'total_duration_sec': total_duration,
                'members': cluster_members,
            })

        logger.info(f"Created {len(clusters)} speaker clusters")
        return clusters

    def auto_match_clusters_to_artists(self) -> Dict[str, Any]:
        """Automatically match clusters to featured artist names from metadata.

        Uses track metadata to find likely matches between speaker clusters
        and featured artist names.

        Returns:
            Statistics dict with matches made
        """
        from ..db.operations import (
            get_all_clusters, get_cluster_members, get_featured_artists_for_track,
            update_cluster_name
        )

        stats = {
            'clusters_processed': 0,
            'matches_found': 0,
            'matches_made': [],
        }

        clusters = get_all_clusters()
        logger.info(f"Processing {len(clusters)} clusters for auto-matching")

        for cluster in clusters:
            if cluster['is_verified']:
                continue  # Skip already verified clusters

            stats['clusters_processed'] += 1

            # Get all tracks this cluster appears in
            members = get_cluster_members(cluster['id'])
            track_ids = set(m['track_id'] for m in members)

            # Count featured artist occurrences across these tracks
            artist_counts = defaultdict(int)
            for track_id in track_ids:
                featured = get_featured_artists_for_track(track_id)
                for artist in featured:
                    artist_counts[artist['name']] += 1

            if not artist_counts:
                continue

            # Find most common featured artist
            best_match = max(artist_counts.items(), key=lambda x: x[1])
            artist_name, count = best_match

            # Require the artist to appear in most tracks for a confident match
            match_ratio = count / len(track_ids)

            if match_ratio >= 0.5 and count >= 2:
                # Auto-match: update cluster name
                update_cluster_name(
                    cluster_id=cluster['id'],
                    name=artist_name,
                    is_verified=False,  # Still needs user verification
                )

                stats['matches_found'] += 1
                stats['matches_made'].append({
                    'cluster_id': cluster['id'],
                    'artist_name': artist_name,
                    'confidence': match_ratio,
                    'track_count': count,
                })

                logger.info(f"Auto-matched cluster to '{artist_name}' "
                            f"(confidence={match_ratio:.0%}, {count} tracks)")

        return stats

    def get_cluster_sample_audio(
        self,
        cluster_id: str,
        max_duration: float = 10.0,
    ) -> Tuple[np.ndarray, int]:
        """Get a sample audio clip for a speaker cluster.

        Args:
            cluster_id: Cluster UUID
            max_duration: Maximum duration for sample (seconds)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        import librosa
        from ..db.operations import get_cluster_members

        members = get_cluster_members(cluster_id)
        if not members:
            raise ValueError(f"Cluster {cluster_id} has no members")

        # Get the highest confidence member
        best_member = max(members, key=lambda m: m['confidence'] or 0)

        # Load audio from isolated vocals
        audio_path = best_member.get('isolated_vocals_path')
        if not audio_path or not Path(audio_path).exists():
            raise ValueError(f"No isolated vocals found for cluster {cluster_id}")

        audio, sr = librosa.load(audio_path, sr=22050, mono=True)

        # Find a non-silent section
        # Simple approach: find first section with significant energy
        frame_length = int(sr * 0.1)  # 100ms frames
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio) - frame_length, frame_length)
        ])

        if len(energy) == 0:
            return audio[:int(max_duration * sr)], sr

        # Find frame with max energy and center sample around it
        max_frame = np.argmax(energy)
        center_sample = max_frame * frame_length
        half_duration = int(max_duration * sr / 2)

        start = max(0, center_sample - half_duration)
        end = min(len(audio), center_sample + half_duration)

        return audio[start:end], sr


def run_speaker_matching(
    artists: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full speaker matching pipeline for specified artists.

    Args:
        artists: List of artist names (default: conor_maynard, william_singe)
        data_dir: Base runtime data directory for canonical input defaults

    Returns:
        Combined statistics dict
    """
    if artists is None:
        artists = ['conor_maynard', 'william_singe']

    matcher = SpeakerMatcher(data_dir=data_dir)
    stats = {
        'artists': {},
        'clustering': {},
        'matching': {},
    }

    # Extract embeddings for each artist
    for artist in artists:
        logger.info(f"\nExtracting embeddings for {artist}...")
        artist_stats = matcher.extract_embeddings_for_artist(artist)
        stats['artists'][artist] = artist_stats
        logger.info(f"  Extracted {artist_stats['embeddings_extracted']} embeddings "
                    f"from {artist_stats['tracks_processed']} tracks")

    # Cluster all embeddings
    logger.info("\nClustering speakers across all tracks...")
    clusters = matcher.cluster_speakers()
    stats['clustering'] = {
        'clusters_created': len(clusters),
        'clusters': [
            {
                'cluster_id': c['cluster_id'],
                'member_count': c['member_count'],
                'duration_sec': c['total_duration_sec'],
            }
            for c in clusters
        ]
    }

    # Auto-match to featured artists
    logger.info("\nAuto-matching clusters to featured artists...")
    match_stats = matcher.auto_match_clusters_to_artists()
    stats['matching'] = match_stats

    return stats


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Cross-track speaker matching')
    parser.add_argument('--artist', nargs='+', help='Artist names to process')
    parser.add_argument('--extract-only', action='store_true', help='Only extract embeddings')
    parser.add_argument('--cluster-only', action='store_true', help='Only run clustering')
    parser.add_argument('--match-only', action='store_true', help='Only run auto-matching')
    parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold')
    parser.add_argument('--data-dir', type=Path, default=None, help='Override the runtime data directory')

    args = parser.parse_args()

    matcher = SpeakerMatcher(
        similarity_threshold=args.threshold,
        data_dir=args.data_dir,
    )

    if args.extract_only:
        artists = args.artist or ['conor_maynard', 'william_singe']
        for artist in artists:
            stats = matcher.extract_embeddings_for_artist(artist)
            print(f"\n{artist}: {stats}")

    elif args.cluster_only:
        clusters = matcher.cluster_speakers()
        print(f"\nCreated {len(clusters)} clusters")
        for c in clusters:
            print(f"  {c['cluster_id']}: {c['member_count']} members, {c['total_duration_sec']:.1f}s")

    elif args.match_only:
        stats = matcher.auto_match_clusters_to_artists()
        print(f"\nMatching stats: {stats}")

    else:
        stats = run_speaker_matching(args.artist, data_dir=args.data_dir)
        print(f"\nFull pipeline stats:")
        import json
        print(json.dumps(stats, indent=2, default=str))
