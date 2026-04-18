"""Additional coverage for file organizer edge branches."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_voice.audio.file_organizer import (
    FileOrganizer,
    _parse_isolated_track_filename,
    organize_by_identified_artist,
)


def test_parse_isolated_track_filename_rejects_invalid_name():
    assert _parse_isolated_track_filename(Path('not_an_isolated_track.wav')) is None


def test_find_profile_for_tracks_skips_non_directories(tmp_path):
    organizer = FileOrganizer(
        training_vocals_dir=tmp_path / 'training',
        voice_profiles_dir=tmp_path / 'profiles',
    )
    organizer.featured_dir.mkdir(parents=True, exist_ok=True)
    (organizer.featured_dir / 'readme.txt').write_text('not a directory')

    result = organizer.find_profile_for_tracks(['track1'], 'SPEAKER_01')

    assert result is None


def test_organize_by_cluster_skips_missing_source_directory(tmp_path):
    organizer = FileOrganizer(
        training_vocals_dir=tmp_path / 'training',
        voice_profiles_dir=tmp_path / 'profiles',
    )
    organizer.featured_dir.mkdir(parents=True, exist_ok=True)
    assignments = {
        'cluster_1': {
            'name': 'Test Artist',
            'members': [{'track_id': 'track1', 'speaker_id': 'SPEAKER_01'}],
        }
    }

    with patch.object(organizer, 'get_cluster_assignments', return_value=assignments), \
         patch.object(organizer, 'find_profile_for_tracks', return_value='missing-profile'):
        stats = organizer.organize_by_cluster(dry_run=False)

    assert stats['profiles_renamed'] == 0
    assert stats['errors'] == []


def test_organize_by_cluster_skips_already_named_profile(tmp_path):
    organizer = FileOrganizer(
        training_vocals_dir=tmp_path / 'training',
        voice_profiles_dir=tmp_path / 'profiles',
    )
    organizer.featured_dir.mkdir(parents=True, exist_ok=True)
    already_named = organizer.featured_dir / 'test_artist'
    already_named.mkdir()
    (already_named / 'track1_SPEAKER_01_isolated.wav').touch()
    assignments = {
        'cluster_1': {
            'name': 'Test Artist',
            'members': [{'track_id': 'track1', 'speaker_id': 'SPEAKER_01'}],
        }
    }

    with patch.object(organizer, 'get_cluster_assignments', return_value=assignments), \
         patch.object(organizer, 'find_profile_for_tracks', return_value='test_artist'):
        stats = organizer.organize_by_cluster(dry_run=False)

    assert stats['profiles_renamed'] == 0
    assert stats['files_moved'] == 0


def test_organize_by_cluster_records_rename_errors(tmp_path):
    organizer = FileOrganizer(
        training_vocals_dir=tmp_path / 'training',
        voice_profiles_dir=tmp_path / 'profiles',
    )
    organizer.featured_dir.mkdir(parents=True, exist_ok=True)
    source_dir = organizer.featured_dir / 'profile-uuid'
    source_dir.mkdir()
    (source_dir / 'track1_SPEAKER_01_isolated.wav').touch()
    assignments = {
        'cluster_1': {
            'name': 'Test Artist',
            'members': [{'track_id': 'track1', 'speaker_id': 'SPEAKER_01'}],
        }
    }

    def fail_rename(self, target):
        raise OSError('rename failed')

    with patch.object(organizer, 'get_cluster_assignments', return_value=assignments), \
         patch.object(organizer, 'find_profile_for_tracks', return_value='profile-uuid'), \
         patch('pathlib.Path.rename', autospec=True, side_effect=fail_rename):
        stats = organizer.organize_by_cluster(dry_run=False)

    assert stats['profiles_renamed'] == 0
    assert 'rename failed' in stats['errors'][0]


def test_create_speaker_profiles_json_returns_empty_for_missing_artist(tmp_path):
    organizer = FileOrganizer(
        training_vocals_dir=tmp_path / 'training',
        voice_profiles_dir=tmp_path / 'profiles',
    )

    with patch('auto_voice.db.operations.get_all_clusters', return_value=[]), \
         patch('auto_voice.db.operations.get_cluster_members', return_value=[]):
        profiles = organizer.create_speaker_profiles_json('missing_artist', dry_run=True)

    assert profiles == {}


def test_create_speaker_profiles_json_skips_invalid_filenames_and_uses_unknown_fallback(tmp_path):
    organizer = FileOrganizer(
        training_vocals_dir=tmp_path / 'training',
        voice_profiles_dir=tmp_path / 'profiles',
    )
    artist_dir = organizer.training_vocals_dir / 'artist'
    artist_dir.mkdir(parents=True)
    valid_file = artist_dir / 'track1_SPEAKER_00_isolated.wav'
    invalid_file = artist_dir / 'broken_name_isolated.wav'
    valid_file.touch()
    invalid_file.touch()

    with patch('auto_voice.db.operations.get_all_clusters', return_value=[]), \
         patch('auto_voice.db.operations.get_cluster_members', return_value=[]):
        profiles = organizer.create_speaker_profiles_json('artist', dry_run=True)

    assert set(profiles) == {valid_file.name}
    assert profiles[valid_file.name]['cluster_name'] == 'Unknown'
    assert profiles[valid_file.name]['verified'] is False


def test_generate_all_profiles_skips_non_directories_and_writes_featured_profiles(tmp_path):
    organizer = FileOrganizer(
        training_vocals_dir=tmp_path / 'training',
        voice_profiles_dir=tmp_path / 'profiles',
    )
    regular_artist_dir = organizer.training_vocals_dir / 'artist'
    regular_artist_dir.mkdir(parents=True)
    (regular_artist_dir / 'track1_SPEAKER_00_isolated.wav').touch()
    (organizer.training_vocals_dir / 'notes.txt').write_text('ignore me')

    featured_artist_dir = organizer.featured_dir / 'featured_artist'
    featured_artist_dir.mkdir(parents=True)
    valid_featured = featured_artist_dir / 'track2_SPEAKER_01_isolated.wav'
    invalid_featured = featured_artist_dir / 'invalid_isolated.wav'
    valid_featured.touch()
    invalid_featured.touch()

    with patch('auto_voice.db.operations.get_all_clusters', return_value=[]), \
         patch('auto_voice.db.operations.get_cluster_members', return_value=[]):
        stats = organizer.generate_all_profiles(dry_run=False)

    featured_profile_path = featured_artist_dir / 'speaker_profiles.json'
    assert stats['artists_processed'] == 1
    assert featured_profile_path.exists()
    data = json.loads(featured_profile_path.read_text())
    assert set(data) == {valid_featured.name}


def test_organize_by_identified_artist_invokes_pipeline_in_order():
    mock_organizer = MagicMock()
    mock_organizer.organize_by_cluster.return_value = {'profiles_renamed': 1}
    mock_organizer.generate_all_profiles.return_value = {'profiles_created': 1}

    with patch('auto_voice.audio.speaker_matcher.run_speaker_matching', return_value={'clusters': 1}) as mock_match, \
         patch('auto_voice.audio.file_organizer.FileOrganizer', return_value=mock_organizer):
        stats = organize_by_identified_artist(dry_run=False)

    mock_match.assert_called_once_with()
    mock_organizer.organize_by_cluster.assert_called_once_with(dry_run=False)
    mock_organizer.generate_all_profiles.assert_called_once_with(dry_run=False)
    assert stats['speaker_matching'] == {'clusters': 1}
