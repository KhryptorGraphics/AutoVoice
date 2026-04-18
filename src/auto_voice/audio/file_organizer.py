"""
File organizer for speaker-identified vocal files.

This module re-organizes extracted vocal files from UUID-based directories
to named artist directories after speaker identification and clustering.

Usage:
    from auto_voice.audio.file_organizer import organize_by_identified_artist

    stats = organize_by_identified_artist()
"""

import json
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


_ISOLATED_TRACK_RE = re.compile(r"^(?P<track_id>.+)_(?P<speaker_id>SPEAKER_[A-Za-z0-9]+)$")


def _parse_isolated_track_filename(wav_file: Path) -> Optional[tuple[str, str]]:
    """Parse `{track_id}_{speaker_id}_isolated.wav` safely.

    Speaker IDs contain an underscore (`SPEAKER_00`), so splitting on the final
    underscore is ambiguous for tracks whose IDs also contain underscores.
    """
    stem = wav_file.stem.replace('_isolated', '')
    match = _ISOLATED_TRACK_RE.match(stem)
    if match is None:
        return None
    return match.group('track_id'), match.group('speaker_id')


class FileOrganizer:
    """Re-organize vocal files by identified artist name."""

    def __init__(
        self,
        training_vocals_dir: Optional[Path] = None,
        voice_profiles_dir: Optional[Path] = None,
    ):
        """Initialize the file organizer.

        Args:
            training_vocals_dir: Directory containing extracted vocals
            voice_profiles_dir: Directory containing voice profiles
        """
        self.training_vocals_dir = training_vocals_dir or Path('data/training_vocals')
        self.voice_profiles_dir = voice_profiles_dir or Path('data/voice_profiles')
        self.training_vocals_dir.mkdir(parents=True, exist_ok=True)
        self.voice_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.featured_dir = self.training_vocals_dir / 'featured'

    def get_cluster_assignments(self) -> Dict[str, Dict[str, Any]]:
        """Get cluster assignments from database.

        Returns:
            Dict mapping cluster_id to {name, is_verified, members: [{track_id, speaker_id, profile_id}]}
        """
        from ..db.operations import (
            get_all_clusters, get_cluster_members, get_embeddings_by_cluster
        )

        clusters = get_all_clusters()
        assignments = {}

        for cluster in clusters:
            members = get_cluster_members(cluster['id'])
            assignments[cluster['id']] = {
                'name': cluster['name'],
                'is_verified': cluster['is_verified'],
                'voice_profile_id': cluster.get('voice_profile_id'),
                'member_count': cluster.get('member_count', len(members)),
                'total_duration_sec': cluster.get('total_duration_sec'),
                'members': members,
            }

        return assignments

    def find_profile_for_tracks(
        self,
        track_ids: List[str],
        speaker_id: str,
    ) -> Optional[str]:
        """Find the profile UUID used for a specific speaker in tracks.

        Searches the featured directory structure to find which profile UUID
        contains files for the given track IDs and speaker.

        Args:
            track_ids: List of track/video IDs
            speaker_id: Speaker ID (e.g., SPEAKER_01)

        Returns:
            Profile UUID if found, None otherwise
        """
        if not self.featured_dir.exists():
            return None

        for profile_dir in self.featured_dir.iterdir():
            if not profile_dir.is_dir():
                continue

            # Check if any expected files exist in this profile directory
            for track_id in track_ids:
                expected_file = profile_dir / f"{track_id}_{speaker_id}_isolated.wav"
                if expected_file.exists():
                    return profile_dir.name

        return None

    def normalize_artist_name(self, name: str) -> str:
        """Normalize artist name for use as directory name.

        Args:
            name: Raw artist name

        Returns:
            Sanitized directory-safe name
        """
        # Replace common problematic characters
        name = name.lower()
        name = name.replace(' ', '_')
        name = name.replace("'", '')
        name = name.replace('"', '')
        name = name.replace('/', '_')
        name = name.replace('\\', '_')
        name = name.replace(':', '_')
        name = name.replace('*', '')
        name = name.replace('?', '')
        name = name.replace('<', '')
        name = name.replace('>', '')
        name = name.replace('|', '_')
        name = re.sub(r'_+', '_', name)

        # Remove leading/trailing underscores
        name = name.strip('_')

        return name

    def organize_by_cluster(self, dry_run: bool = True) -> Dict[str, Any]:
        """Re-organize files based on cluster assignments.

        Args:
            dry_run: If True, only report what would be done

        Returns:
            Statistics dict
        """
        stats = {
            'clusters_processed': 0,
            'profiles_renamed': 0,
            'files_moved': 0,
            'profiles_created': [],
            'errors': [],
            'dry_run': dry_run,
        }

        assignments = self.get_cluster_assignments()
        logger.info(f"Found {len(assignments)} speaker clusters")

        # Group members by their current profile UUID
        profile_to_cluster: Dict[str, str] = {}
        cluster_profiles: Dict[str, List[str]] = defaultdict(list)

        for cluster_id, cluster_info in assignments.items():
            stats['clusters_processed'] += 1

            # Skip clusters without a proper name
            if cluster_info['name'].startswith('Unknown Speaker'):
                continue

            # Get track IDs for this cluster
            track_ids = set()
            speaker_ids = set()
            for member in cluster_info['members']:
                track_ids.add(member['track_id'])
                speaker_ids.add(member['speaker_id'])

            # Try to find the profile UUID for these tracks
            for speaker_id in speaker_ids:
                profile_uuid = self.find_profile_for_tracks(list(track_ids), speaker_id)
                if profile_uuid:
                    if profile_uuid not in profile_to_cluster:
                        profile_to_cluster[profile_uuid] = cluster_id
                        cluster_profiles[cluster_id].append(profile_uuid)

        # Now rename directories based on cluster names
        for cluster_id, profile_uuids in cluster_profiles.items():
            cluster_info = assignments[cluster_id]
            artist_name = self.normalize_artist_name(cluster_info['name'])
            target_dir = self.featured_dir / artist_name

            for profile_uuid in profile_uuids:
                source_dir = self.featured_dir / profile_uuid

                if not source_dir.exists():
                    continue

                if source_dir == target_dir:
                    continue  # Already named correctly

                # Count files to move
                files = list(source_dir.glob('*.wav'))
                stats['files_moved'] += len(files)

                if dry_run:
                    logger.info(f"[DRY RUN] Would rename {source_dir.name} -> {artist_name}")
                    logger.info(f"  ({len(files)} files would be moved)")
                else:
                    try:
                        # If target exists, merge files
                        if target_dir.exists():
                            for f in files:
                                dest = target_dir / f.name
                                if not dest.exists():
                                    shutil.move(str(f), str(dest))
                            # Remove empty source directory
                            if not list(source_dir.iterdir()):
                                source_dir.rmdir()
                        else:
                            # Simple rename
                            source_dir.rename(target_dir)

                        stats['profiles_renamed'] += 1
                        stats['profiles_created'].append({
                            'old_name': profile_uuid,
                            'new_name': artist_name,
                            'cluster_id': cluster_id,
                            'file_count': len(files),
                        })
                        logger.info(f"Renamed {profile_uuid} -> {artist_name}")

                    except Exception as e:
                        stats['errors'].append(f"Failed to rename {profile_uuid}: {e}")
                        logger.error(f"Failed to rename {profile_uuid}: {e}")

        return stats

    def create_speaker_profiles_json(
        self,
        artist_name: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Create speaker_profiles.json for an artist directory.

        Args:
            artist_name: Artist name (e.g., 'conor_maynard')
            dry_run: If True, only report what would be created

        Returns:
            The profile mapping dict
        """
        from ..db.operations import (
            get_tracks_by_artist, get_embeddings_for_track, get_all_clusters,
            get_cluster_members
        )

        artist_dir = self.training_vocals_dir / artist_name
        if not artist_dir.exists():
            return {}

        # Build mapping from track+speaker to cluster name
        clusters = get_all_clusters()
        track_speaker_to_cluster = {}

        for cluster in clusters:
            members = get_cluster_members(cluster['id'])
            for member in members:
                key = (member['track_id'], member['speaker_id'])
                track_speaker_to_cluster[key] = {
                    'cluster_id': cluster['id'],
                    'cluster_name': cluster['name'],
                    'is_verified': cluster['is_verified'],
                    'confidence': member.get('confidence'),
                }

        # Scan files in artist directory
        profiles = {}
        for wav_file in artist_dir.glob('*_isolated.wav'):
            parsed = _parse_isolated_track_filename(wav_file)
            if parsed is None:
                continue

            track_id, speaker_id = parsed
            key = (track_id, speaker_id)

            if key in track_speaker_to_cluster:
                cluster_info = track_speaker_to_cluster[key]
                profiles[wav_file.name] = {
                    'track_id': track_id,
                    'speaker_id': speaker_id,
                    'cluster_id': cluster_info['cluster_id'],
                    'cluster_name': cluster_info['cluster_name'],
                    'is_primary': speaker_id == 'SPEAKER_00',
                    'verified': cluster_info['is_verified'],
                    'confidence': cluster_info.get('confidence'),
                }
            else:
                profiles[wav_file.name] = {
                    'track_id': track_id,
                    'speaker_id': speaker_id,
                    'cluster_id': None,
                    'cluster_name': 'Unknown',
                    'is_primary': speaker_id == 'SPEAKER_00',
                    'verified': False,
                    'confidence': None,
                }

        profile_path = artist_dir / 'speaker_profiles.json'

        if dry_run:
            logger.info(f"[DRY RUN] Would create {profile_path}")
            logger.info(f"  ({len(profiles)} file mappings)")
        else:
            with open(profile_path, 'w') as f:
                json.dump(profiles, f, indent=2)
            logger.info(f"Created {profile_path} with {len(profiles)} mappings")

        return profiles

    def generate_all_profiles(self, dry_run: bool = True) -> Dict[str, Any]:
        """Generate speaker_profiles.json for all artist directories.

        Args:
            dry_run: If True, only report what would be created

        Returns:
            Statistics dict
        """
        stats = {
            'artists_processed': 0,
            'profiles_created': 0,
            'total_mappings': 0,
            'dry_run': dry_run,
        }

        # Process main artist directories
        for artist_dir in self.training_vocals_dir.iterdir():
            if not artist_dir.is_dir():
                continue
            if artist_dir.name in ['featured', 'by_profile']:
                continue

            profiles = self.create_speaker_profiles_json(artist_dir.name, dry_run)
            stats['artists_processed'] += 1
            stats['profiles_created'] += 1
            stats['total_mappings'] += len(profiles)

        # Process featured artist directories
        if self.featured_dir.exists():
            for featured_dir in self.featured_dir.iterdir():
                if not featured_dir.is_dir():
                    continue

                # Create profile for featured artist
                # Use parent path to construct correct path
                rel_path = f"featured/{featured_dir.name}"
                full_path = self.training_vocals_dir / rel_path

                if full_path.exists():
                    # Build simple profile based on files present
                    profiles = {}
                    for wav_file in full_path.glob('*_isolated.wav'):
                        parsed = _parse_isolated_track_filename(wav_file)
                        if parsed is None:
                            continue
                        track_id, speaker_id = parsed
                        profiles[wav_file.name] = {
                            'track_id': track_id,
                            'speaker_id': speaker_id,
                            'artist_name': featured_dir.name,
                            'is_primary': False,
                        }

                    profile_path = full_path / 'speaker_profiles.json'

                    if dry_run:
                        logger.info(f"[DRY RUN] Would create {profile_path}")
                    else:
                        with open(profile_path, 'w') as f:
                            json.dump(profiles, f, indent=2)

                    stats['profiles_created'] += 1
                    stats['total_mappings'] += len(profiles)

        return stats


def organize_by_identified_artist(dry_run: bool = True) -> Dict[str, Any]:
    """Run full organization pipeline.

    1. Run speaker matching to populate database
    2. Organize files by cluster names
    3. Generate speaker_profiles.json files

    Args:
        dry_run: If True, only report what would be done

    Returns:
        Combined statistics
    """
    from .speaker_matcher import run_speaker_matching

    stats = {
        'speaker_matching': {},
        'file_organization': {},
        'profile_generation': {},
    }

    # Step 1: Run speaker matching pipeline
    logger.info("Step 1: Running speaker matching pipeline...")
    matching_stats = run_speaker_matching()
    stats['speaker_matching'] = matching_stats

    # Step 2: Organize files by cluster
    logger.info("\nStep 2: Organizing files by identified artist...")
    organizer = FileOrganizer()
    org_stats = organizer.organize_by_cluster(dry_run=dry_run)
    stats['file_organization'] = org_stats

    # Step 3: Generate speaker profiles
    logger.info("\nStep 3: Generating speaker profile mappings...")
    profile_stats = organizer.generate_all_profiles(dry_run=dry_run)
    stats['profile_generation'] = profile_stats

    return stats


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Organize vocal files by identified artist')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Only report what would be done (default)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the organization')
    parser.add_argument('--profiles-only', action='store_true',
                        help='Only generate speaker_profiles.json files')

    args = parser.parse_args()

    dry_run = not args.execute

    if args.profiles_only:
        organizer = FileOrganizer()
        stats = organizer.generate_all_profiles(dry_run=dry_run)
        print(f"\nProfile generation stats: {stats}")
    else:
        stats = organize_by_identified_artist(dry_run=dry_run)
        print("\n=== Organization Complete ===")
        print(json.dumps(stats, indent=2, default=str))
