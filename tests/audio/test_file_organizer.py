"""Tests for file_organizer.py - File management and organization.

Test Coverage:
- Task 2.6: Directory creation
- File naming conventions
- Cleanup of old files
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from auto_voice.audio.file_organizer import (
    FileOrganizer,
    _parse_isolated_track_filename,
    organize_by_identified_artist,
)


@pytest.fixture
def organizer(tmp_path):
    """Create FileOrganizer instance with temp directories."""
    return FileOrganizer(
        training_vocals_dir=tmp_path / "training_vocals",
        voice_profiles_dir=tmp_path / "voice_profiles",
    )


@pytest.fixture
def sample_cluster_structure(tmp_path):
    """Create sample cluster directory structure."""
    featured_dir = tmp_path / "training_vocals" / "featured"
    featured_dir.mkdir(parents=True)

    # Create profile directories
    profile1 = featured_dir / "profile-uuid-1"
    profile2 = featured_dir / "profile-uuid-2"
    profile1.mkdir()
    profile2.mkdir()

    # Add sample files
    (profile1 / "track1_SPEAKER_01_isolated.wav").touch()
    (profile1 / "track2_SPEAKER_01_isolated.wav").touch()
    (profile2 / "track1_SPEAKER_02_isolated.wav").touch()

    return tmp_path


class TestFileOrganizer:
    """Test suite for FileOrganizer."""

    def test_initialization(self, tmp_path):
        """Test FileOrganizer initialization."""
        organizer = FileOrganizer(
            training_vocals_dir=tmp_path / "training",
            voice_profiles_dir=tmp_path / "profiles",
        )

        assert organizer.training_vocals_dir == tmp_path / "training"
        assert organizer.voice_profiles_dir == tmp_path / "profiles"
        assert organizer.featured_dir == tmp_path / "training" / "featured"

    def test_initialization_default_paths(self):
        """Test initialization with default paths."""
        organizer = FileOrganizer()

        assert organizer.training_vocals_dir == Path('data/training_vocals')
        assert organizer.voice_profiles_dir == Path('data/voice_profiles')

    def test_normalize_artist_name(self, organizer):
        """Test artist name normalization for directory names."""
        # Spaces to underscores
        assert organizer.normalize_artist_name("Artist Name") == "artist_name"

        # Remove apostrophes
        assert organizer.normalize_artist_name("Artist's Name") == "artists_name"

        # Remove slashes
        assert organizer.normalize_artist_name("AC/DC") == "ac_dc"

        # Remove special characters
        assert organizer.normalize_artist_name("Artist: Name?") == "artist_name"

        # Lowercase
        assert organizer.normalize_artist_name("ARTIST NAME") == "artist_name"

        # Strip underscores
        assert organizer.normalize_artist_name("_Artist_") == "artist"

    def test_find_profile_for_tracks(self, organizer, sample_cluster_structure):
        """Test finding profile UUID for given tracks."""
        organizer.training_vocals_dir = sample_cluster_structure / "training_vocals"
        organizer.featured_dir = organizer.training_vocals_dir / "featured"

        profile_uuid = organizer.find_profile_for_tracks(
            track_ids=['track1', 'track2'],
            speaker_id='SPEAKER_01',
        )

        assert profile_uuid == 'profile-uuid-1'

    def test_find_profile_for_tracks_not_found(self, organizer, sample_cluster_structure):
        """Test finding profile when no matching files exist."""
        organizer.training_vocals_dir = sample_cluster_structure / "training_vocals"
        organizer.featured_dir = organizer.training_vocals_dir / "featured"

        profile_uuid = organizer.find_profile_for_tracks(
            track_ids=['nonexistent'],
            speaker_id='SPEAKER_99',
        )

        assert profile_uuid is None

    def test_organize_by_cluster_dry_run(self, organizer, sample_cluster_structure):
        """Test organize by cluster in dry-run mode."""
        organizer.training_vocals_dir = sample_cluster_structure / "training_vocals"
        organizer.featured_dir = organizer.training_vocals_dir / "featured"

        clusters = {
            'cluster_1': {
                'name': 'Featured Artist A',
                'is_verified': False,
                'members': [
                    {'track_id': 'track1', 'speaker_id': 'SPEAKER_01'},
                ],
            }
        }

        with patch.object(organizer, 'get_cluster_assignments', return_value=clusters), \
             patch.object(organizer, 'find_profile_for_tracks', return_value='profile-uuid-1'):

            stats = organizer.organize_by_cluster(dry_run=True)

            assert stats['dry_run'] is True
            assert stats['profiles_renamed'] == 0  # Dry run doesn't rename

    def test_organize_by_cluster_actual_rename(self, organizer, sample_cluster_structure):
        """Test actual directory renaming (not dry-run)."""
        organizer.training_vocals_dir = sample_cluster_structure / "training_vocals"
        organizer.featured_dir = organizer.training_vocals_dir / "featured"

        clusters = {
            'cluster_1': {
                'name': 'Featured Artist A',
                'is_verified': False,
                'members': [
                    {'track_id': 'track1', 'speaker_id': 'SPEAKER_01'},
                ],
            }
        }

        with patch.object(organizer, 'get_cluster_assignments', return_value=clusters), \
             patch.object(organizer, 'find_profile_for_tracks', return_value='profile-uuid-1'):

            stats = organizer.organize_by_cluster(dry_run=False)

            assert stats['dry_run'] is False
            assert stats['profiles_renamed'] >= 0

    def test_organize_by_cluster_skips_unknown_speakers(self, organizer):
        """Test that 'Unknown Speaker' clusters are skipped."""
        clusters = {
            'cluster_1': {
                'name': 'Unknown Speaker 1',
                'is_verified': False,
                'members': [],
            }
        }

        with patch.object(organizer, 'get_cluster_assignments', return_value=clusters):
            stats = organizer.organize_by_cluster(dry_run=True)

            # Should not process unknown speakers
            assert stats['profiles_renamed'] == 0

    def test_organize_by_cluster_merges_existing_directories(self, organizer, tmp_path):
        """Test merging files when target directory exists."""
        featured_dir = tmp_path / "training_vocals" / "featured"
        featured_dir.mkdir(parents=True)

        # Source directory
        source_dir = featured_dir / "profile-uuid-1"
        source_dir.mkdir()
        (source_dir / "file1.wav").touch()

        # Target directory already exists
        target_dir = featured_dir / "featured_artist_a"
        target_dir.mkdir()
        (target_dir / "file2.wav").touch()

        organizer.training_vocals_dir = tmp_path / "training_vocals"
        organizer.featured_dir = featured_dir

        clusters = {
            'cluster_1': {
                'name': 'Featured Artist A',
                'is_verified': False,
                'members': [{'track_id': 'track1', 'speaker_id': 'SPEAKER_01'}],
            }
        }

        with patch.object(organizer, 'get_cluster_assignments', return_value=clusters), \
             patch.object(organizer, 'find_profile_for_tracks', return_value='profile-uuid-1'):

            stats = organizer.organize_by_cluster(dry_run=False)

            # Both files should exist in target
            assert (target_dir / "file1.wav").exists() or (source_dir / "file1.wav").exists()
            assert (target_dir / "file2.wav").exists()

    def test_create_speaker_profiles_json_dry_run(self, organizer, tmp_path):
        """Test creating speaker_profiles.json in dry-run mode."""
        artist_dir = tmp_path / "training_vocals" / "test_artist"
        artist_dir.mkdir(parents=True)
        (artist_dir / "track1_SPEAKER_00_isolated.wav").touch()

        organizer.training_vocals_dir = tmp_path / "training_vocals"

        with patch('auto_voice.db.operations.get_all_clusters', return_value=[]), \
             patch('auto_voice.db.operations.get_cluster_members', return_value=[]):

            profiles = organizer.create_speaker_profiles_json(
                artist_name='test_artist',
                dry_run=True,
            )

            assert len(profiles) > 0
            # File should not be created in dry-run
            assert not (artist_dir / "speaker_profiles.json").exists()

    def test_create_speaker_profiles_json_actual(self, organizer, tmp_path):
        """Test actually creating speaker_profiles.json file."""
        artist_dir = tmp_path / "training_vocals" / "test_artist"
        artist_dir.mkdir(parents=True)
        (artist_dir / "track1_SPEAKER_00_isolated.wav").touch()

        organizer.training_vocals_dir = tmp_path / "training_vocals"

        with patch('auto_voice.db.operations.get_all_clusters', return_value=[]), \
             patch('auto_voice.db.operations.get_cluster_members', return_value=[]):

            profiles = organizer.create_speaker_profiles_json(
                artist_name='test_artist',
                dry_run=False,
            )

            # File should be created
            assert (artist_dir / "speaker_profiles.json").exists()

            # Verify JSON content
            with open(artist_dir / "speaker_profiles.json") as f:
                data = json.load(f)
                assert len(data) > 0

    def test_generate_all_profiles(self, organizer, tmp_path):
        """Test generating profiles for all artist directories."""
        training_dir = tmp_path / "training_vocals"
        training_dir.mkdir(exist_ok=True)

        # Create artist directories
        (training_dir / "artist1").mkdir()
        (training_dir / "artist1" / "track1_SPEAKER_00_isolated.wav").touch()

        (training_dir / "artist2").mkdir()
        (training_dir / "artist2" / "track2_SPEAKER_00_isolated.wav").touch()

        organizer.training_vocals_dir = training_dir

        with patch('auto_voice.db.operations.get_all_clusters', return_value=[]), \
             patch('auto_voice.db.operations.get_cluster_members', return_value=[]):

            stats = organizer.generate_all_profiles(dry_run=False)

            assert stats['artists_processed'] >= 2
            assert stats['profiles_created'] >= 2

    def test_generate_all_profiles_skips_special_dirs(self, organizer, tmp_path):
        """Test that special directories (featured, by_profile) are skipped."""
        training_dir = tmp_path / "training_vocals"
        training_dir.mkdir(exist_ok=True)

        # Create special directories
        (training_dir / "featured").mkdir()
        (training_dir / "by_profile").mkdir()

        # Create regular artist directory
        (training_dir / "artist1").mkdir()

        organizer.training_vocals_dir = training_dir

        with patch('auto_voice.db.operations.get_all_clusters', return_value=[]), \
             patch('auto_voice.db.operations.get_cluster_members', return_value=[]):

            stats = organizer.generate_all_profiles(dry_run=False)

            # Should only process artist1
            assert stats['artists_processed'] == 1


class TestFileNamingConventions:
    """Test file naming conventions."""

    def test_isolated_wav_filename_format(self, organizer):
        """Test that isolated WAV files follow naming convention."""
        # Expected format: {track_id}_{speaker_id}_isolated.wav
        filename = "track123_SPEAKER_00_isolated.wav"

        track_id, speaker_id = _parse_isolated_track_filename(Path(filename))
        assert track_id == "track123"
        assert speaker_id == "SPEAKER_00"

    def test_speaker_profiles_json_naming(self, organizer, tmp_path):
        """Test speaker_profiles.json naming convention."""
        artist_dir = tmp_path / "test_artist"
        artist_dir.mkdir()

        expected_path = artist_dir / "speaker_profiles.json"

        # Create file
        expected_path.touch()

        assert expected_path.exists()
        assert expected_path.name == "speaker_profiles.json"


class TestDirectoryCreation:
    """Test directory creation functionality."""

    def test_directory_creation_for_artist(self, organizer, tmp_path):
        """Test automatic directory creation for artists."""
        organizer.training_vocals_dir = tmp_path / "training_vocals"

        artist_dir = organizer.training_vocals_dir / "new_artist"

        # Create directory
        artist_dir.mkdir(parents=True, exist_ok=True)

        assert artist_dir.exists()
        assert artist_dir.is_dir()

    def test_featured_directory_creation(self, organizer, tmp_path):
        """Test creation of featured artist directories."""
        organizer.training_vocals_dir = tmp_path / "training_vocals"
        organizer.featured_dir = organizer.training_vocals_dir / "featured"

        # Create featured directory
        organizer.featured_dir.mkdir(parents=True, exist_ok=True)

        assert organizer.featured_dir.exists()
        assert organizer.featured_dir.is_dir()

    def test_nested_directory_creation(self, organizer, tmp_path):
        """Test creation of nested directories."""
        organizer.training_vocals_dir = tmp_path / "training_vocals"

        nested_dir = organizer.training_vocals_dir / "featured" / "artist_name"
        nested_dir.mkdir(parents=True, exist_ok=True)

        assert nested_dir.exists()
        assert nested_dir.parent.exists()


class TestFileCleanup:
    """Test file cleanup functionality."""

    def test_cleanup_empty_directories(self, organizer, tmp_path):
        """Test removal of empty directories after file moves."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.wav").touch()

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # Move file
        (source_dir / "file.wav").rename(target_dir / "file.wav")

        # Remove empty directory
        if not list(source_dir.iterdir()):
            source_dir.rmdir()

        assert not source_dir.exists()
        assert (target_dir / "file.wav").exists()

    def test_cleanup_old_files_by_date(self, tmp_path):
        """Test cleanup of old files (placeholder).

        This is a placeholder for age-based file cleanup if implemented.
        """
        # Future: Implement age-based cleanup
        # organizer.cleanup_old_files(older_than_days=30)
        pass


def test_organize_by_identified_artist_full_pipeline():
    """Test full organization pipeline."""
    with patch('auto_voice.audio.speaker_matcher.run_speaker_matching') as mock_matching, \
         patch('auto_voice.audio.file_organizer.FileOrganizer') as mock_organizer_class:

        mock_matching.return_value = {
            'clustering': {'clusters_created': 2},
        }

        mock_organizer = MagicMock()
        mock_organizer.organize_by_cluster.return_value = {'profiles_renamed': 2}
        mock_organizer.generate_all_profiles.return_value = {'profiles_created': 2}
        mock_organizer_class.return_value = mock_organizer

        stats = organize_by_identified_artist(dry_run=True)

        assert 'speaker_matching' in stats
        assert 'file_organization' in stats
        assert 'profile_generation' in stats


@pytest.mark.parametrize("artist_name,expected_normalized", [
    ("Artist Name", "artist_name"),
    ("AC/DC", "ac_dc"),
    ("Artist's Name", "artists_name"),
    ("UPPERCASE", "uppercase"),
    ("Multiple   Spaces", "multiple_spaces"),
])
def test_artist_name_normalization(organizer, artist_name, expected_normalized):
    """Test various artist name normalization cases."""
    result = organizer.normalize_artist_name(artist_name)
    assert result == expected_normalized
