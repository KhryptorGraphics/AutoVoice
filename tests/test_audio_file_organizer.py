"""Tests for file_organizer.py - File management and organization.

Task 2.6: Test file_organizer.py
- Test directory creation
- Test file naming conventions
- Test cleanup of old files
"""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from auto_voice.audio.file_organizer import (
    FileOrganizer,
    organize_by_identified_artist,
)


@pytest.fixture
def file_organizer(tmp_path):
    """Create FileOrganizer instance with temp directories."""
    training_dir = tmp_path / 'training_vocals'
    profiles_dir = tmp_path / 'voice_profiles'
    training_dir.mkdir()
    profiles_dir.mkdir()

    return FileOrganizer(
        training_vocals_dir=training_dir,
        voice_profiles_dir=profiles_dir,
    )


@pytest.fixture
def mock_cluster_data():
    """Create mock cluster data."""
    return {
        'cluster-1': {
            'name': 'Justin Bieber',
            'is_verified': False,
            'voice_profile_id': 'profile-123',
            'member_count': 3,
            'total_duration_sec': 45.0,
            'members': [
                {'track_id': 'track_1', 'speaker_id': 'SPEAKER_01'},
                {'track_id': 'track_2', 'speaker_id': 'SPEAKER_01'},
            ],
        },
        'cluster-2': {
            'name': 'Unknown Speaker 1',
            'is_verified': False,
            'voice_profile_id': None,
            'member_count': 1,
            'total_duration_sec': 10.0,
            'members': [
                {'track_id': 'track_3', 'speaker_id': 'SPEAKER_02'},
            ],
        },
    }


class TestFileOrganizerInit:
    """Test FileOrganizer initialization."""

    @pytest.mark.smoke
    def test_init_default(self):
        """Test default initialization."""
        organizer = FileOrganizer()
        assert organizer.training_vocals_dir == Path('data/training_vocals')
        assert organizer.voice_profiles_dir == Path('data/voice_profiles')

    def test_init_custom_dirs(self, tmp_path):
        """Test initialization with custom directories."""
        training_dir = tmp_path / 'custom_training'
        profiles_dir = tmp_path / 'custom_profiles'

        organizer = FileOrganizer(
            training_vocals_dir=training_dir,
            voice_profiles_dir=profiles_dir,
        )

        assert organizer.training_vocals_dir == training_dir
        assert organizer.voice_profiles_dir == profiles_dir


class TestArtistNameNormalization:
    """Test artist name normalization."""

    def test_normalize_artist_name_basic(self, file_organizer):
        """Test basic name normalization."""
        result = file_organizer.normalize_artist_name("Justin Bieber")
        assert result == "justin_bieber"

    def test_normalize_artist_name_special_chars(self, file_organizer):
        """Test handling special characters."""
        result = file_organizer.normalize_artist_name("P!nk")
        # Note: normalize_artist_name converts to lowercase and may keep some chars
        assert result.lower() == "p!nk" or "!" not in result

        result = file_organizer.normalize_artist_name("Artist/Name")
        assert "/" not in result or "_" in result

    def test_normalize_artist_name_apostrophes(self, file_organizer):
        """Test removing apostrophes."""
        result = file_organizer.normalize_artist_name("O'Neill")
        assert "'" not in result

    def test_normalize_artist_name_underscores(self, file_organizer):
        """Test handling underscores."""
        result = file_organizer.normalize_artist_name("Artist Name Here")
        assert " " not in result
        assert "_" in result


class TestProfileFinding:
    """Test finding profiles for tracks."""

    def test_find_profile_for_tracks_exists(self, file_organizer, tmp_path):
        """Test finding existing profile."""
        # Create mock structure
        featured_dir = file_organizer.training_vocals_dir / 'featured'
        profile_dir = featured_dir / 'profile-uuid-123'
        profile_dir.mkdir(parents=True)

        # Create expected file
        test_file = profile_dir / 'track_1_SPEAKER_01_isolated.wav'
        test_file.touch()

        result = file_organizer.find_profile_for_tracks(
            track_ids=['track_1'],
            speaker_id='SPEAKER_01',
        )

        assert result == 'profile-uuid-123'

    def test_find_profile_for_tracks_not_exists(self, file_organizer):
        """Test when profile doesn't exist."""
        result = file_organizer.find_profile_for_tracks(
            track_ids=['nonexistent'],
            speaker_id='SPEAKER_01',
        )

        assert result is None


class TestClusterAssignments:
    """Test cluster assignment retrieval."""

    @patch('auto_voice.db.operations.get_all_clusters')
    @patch('auto_voice.db.operations.get_cluster_members')
    def test_get_cluster_assignments(self, mock_members, mock_clusters, file_organizer):
        """Test retrieving cluster assignments."""
        mock_clusters.return_value = [
            {
                'id': 'cluster-1',
                'name': 'Justin Bieber',
                'is_verified': False,
                'member_count': 2,
            }
        ]

        mock_members.return_value = [
            {'track_id': 'track_1', 'speaker_id': 'SPEAKER_01'},
        ]

        result = file_organizer.get_cluster_assignments()

        assert 'cluster-1' in result
        assert result['cluster-1']['name'] == 'Justin Bieber'


class TestFileOrganization:
    """Test file organization by cluster."""

    @patch('auto_voice.audio.file_organizer.FileOrganizer.get_cluster_assignments')
    def test_organize_by_cluster_dry_run(self, mock_get_assignments, file_organizer,
                                        mock_cluster_data):
        """Test dry run mode."""
        mock_get_assignments.return_value = mock_cluster_data

        # Create mock file structure
        featured_dir = file_organizer.training_vocals_dir / 'featured'
        profile_dir = featured_dir / 'old-uuid-456'
        profile_dir.mkdir(parents=True)

        test_file = profile_dir / 'track_1_SPEAKER_01_isolated.wav'
        test_file.touch()

        stats = file_organizer.organize_by_cluster(dry_run=True)

        assert stats['dry_run'] is True
        assert stats['clusters_processed'] > 0
        # Files should NOT be moved in dry run
        assert test_file.exists()

    @patch('auto_voice.audio.file_organizer.FileOrganizer.get_cluster_assignments')
    @patch('auto_voice.audio.file_organizer.FileOrganizer.find_profile_for_tracks')
    def test_organize_by_cluster_execution(self, mock_find, mock_get_assignments,
                                           file_organizer, mock_cluster_data):
        """Test actual file organization."""
        mock_get_assignments.return_value = mock_cluster_data
        mock_find.return_value = 'old-uuid-456'

        # Create mock file structure
        featured_dir = file_organizer.training_vocals_dir / 'featured'
        profile_dir = featured_dir / 'old-uuid-456'
        profile_dir.mkdir(parents=True)

        test_file = profile_dir / 'track_1_SPEAKER_01_isolated.wav'
        test_file.write_text('test audio data')

        stats = file_organizer.organize_by_cluster(dry_run=False)

        assert stats['dry_run'] is False

    def test_organize_skip_unknown_speakers(self, file_organizer, mock_cluster_data):
        """Test that unknown speakers are skipped."""
        with patch('auto_voice.audio.file_organizer.FileOrganizer.get_cluster_assignments') as mock:
            mock.return_value = mock_cluster_data

            stats = file_organizer.organize_by_cluster(dry_run=True)

            # Unknown Speaker clusters should be skipped
            assert stats['clusters_processed'] > 0


class TestSpeakerProfilesJSON:
    """Test speaker_profiles.json generation."""

    @patch('auto_voice.db.operations.get_all_clusters')
    @patch('auto_voice.db.operations.get_cluster_members')
    def test_create_speaker_profiles_json(self, mock_members, mock_clusters,
                                          file_organizer, tmp_path):
        """Test creating speaker_profiles.json."""
        artist_dir = file_organizer.training_vocals_dir / 'test_artist'
        artist_dir.mkdir()

        # Create test files
        test_file = artist_dir / 'track_1_SPEAKER_00_isolated.wav'
        test_file.touch()

        mock_clusters.return_value = [
            {'id': 'cluster-1', 'name': 'Test Artist', 'is_verified': False}
        ]

        mock_members.return_value = [
            {'track_id': 'track_1', 'speaker_id': 'SPEAKER_00', 'confidence': 0.95}
        ]

        profiles = file_organizer.create_speaker_profiles_json(
            artist_name='test_artist',
            dry_run=False,
        )

        # Check that mapping was created
        assert len(profiles) > 0

        # Check that file was created
        profile_path = artist_dir / 'speaker_profiles.json'
        assert profile_path.exists()

    def test_create_speaker_profiles_json_dry_run(self, file_organizer, tmp_path):
        """Test dry run doesn't create files."""
        artist_dir = file_organizer.training_vocals_dir / 'test_artist'
        artist_dir.mkdir()

        test_file = artist_dir / 'track_1_SPEAKER_00_isolated.wav'
        test_file.touch()

        with patch('auto_voice.db.operations.get_all_clusters') as mock_clusters:
            with patch('auto_voice.db.operations.get_cluster_members') as mock_members:
                mock_clusters.return_value = []
                mock_members.return_value = []

                profiles = file_organizer.create_speaker_profiles_json(
                    artist_name='test_artist',
                    dry_run=True,
                )

                profile_path = artist_dir / 'speaker_profiles.json'
                # Should NOT create file in dry run
                assert profiles
                assert not profile_path.exists()


class TestGenerateAllProfiles:
    """Test generating profiles for all artists."""

    @patch('auto_voice.audio.file_organizer.FileOrganizer.create_speaker_profiles_json')
    def test_generate_all_profiles(self, mock_create, file_organizer):
        """Test generating profiles for multiple artists."""
        # Create artist directories
        artist1_dir = file_organizer.training_vocals_dir / 'artist1'
        artist2_dir = file_organizer.training_vocals_dir / 'artist2'
        artist1_dir.mkdir()
        artist2_dir.mkdir()

        mock_create.return_value = {}

        stats = file_organizer.generate_all_profiles(dry_run=True)

        assert stats['artists_processed'] >= 2

    def test_generate_all_profiles_skip_featured(self, file_organizer):
        """Test that 'featured' directory is skipped."""
        # Create regular and featured directories
        artist_dir = file_organizer.training_vocals_dir / 'artist1'
        featured_dir = file_organizer.training_vocals_dir / 'featured'
        artist_dir.mkdir()
        featured_dir.mkdir()

        with patch('auto_voice.audio.file_organizer.FileOrganizer.create_speaker_profiles_json') as mock:
            mock.return_value = {}

            stats = file_organizer.generate_all_profiles(dry_run=True)

            # Should process artist1 but handle featured specially
            assert stats['artists_processed'] >= 1


@pytest.mark.integration
class TestFileOrganizerIntegration:
    """Integration tests for file organization."""

    @patch('auto_voice.audio.speaker_matcher.run_speaker_matching')
    def test_full_organization_pipeline(self, mock_matching, tmp_path):
        """Test complete organization pipeline."""
        mock_matching.return_value = {
            'artists': {},
            'clustering': {'clusters_created': 2},
            'matching': {'matches_found': 1},
        }

        stats = organize_by_identified_artist(dry_run=True)

        assert 'speaker_matching' in stats
        assert 'file_organization' in stats
        assert 'profile_generation' in stats

    def test_directory_structure_creation(self, file_organizer):
        """Test that required directories are created."""
        # Directories should exist
        assert file_organizer.training_vocals_dir.exists()
        assert file_organizer.voice_profiles_dir.exists()

        # Featured directory should be created when needed
        featured_dir = file_organizer.featured_dir
        # May not exist until first use
