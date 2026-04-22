"""Fetcher and database integration tests for youtube_metadata.py."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from auto_voice.audio.youtube_metadata import (
    VideoMetadata,
    YouTubeMetadataFetcher,
    populate_database_from_files,
)


class TestYouTubeMetadataFetcher:
    """yt-dlp integration and file scanning tests."""

    def test_fetch_metadata_success(self):
        fetcher = YouTubeMetadataFetcher(yt_dlp_path='yt-dlp')
        payload = {
            'title': 'Artist - Song Title',
            'channel': 'ArtistChannel',
            'upload_date': '20260101',
            'duration': 123,
            'description': 'Featuring Guest Artist',
        }

        with patch('auto_voice.audio.youtube_metadata.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(payload),
                stderr='',
            )

            metadata = fetcher.fetch_metadata('abcdefghijk')

        assert metadata == VideoMetadata(
            video_id='abcdefghijk',
            title='Artist - Song Title',
            channel='ArtistChannel',
            upload_date='20260101',
            duration_sec=123.0,
            description='Featuring Guest Artist',
        )

    def test_fetch_metadata_failure_paths(self):
        fetcher = YouTubeMetadataFetcher()

        with patch('auto_voice.audio.youtube_metadata.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='boom')
            assert fetcher.fetch_metadata('abcdefghijk') is None

            mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=30)
            assert fetcher.fetch_metadata('abcdefghijk') is None

            mock_run.side_effect = None
            mock_run.return_value = MagicMock(returncode=0, stdout='{not-json', stderr='')
            assert fetcher.fetch_metadata('abcdefghijk') is None

            mock_run.side_effect = RuntimeError('unexpected')
            assert fetcher.fetch_metadata('abcdefghijk') is None

    @pytest.mark.parametrize(
        ('filename', 'expected'),
        [
            ('abcdefghijk_vocals.wav', 'abcdefghijk'),
            ('abcdefghijk_diarization.json', 'abcdefghijk'),
            ('abcdefghijk_isolated.wav', 'abcdefghijk'),
            ('abcdefghijk_SPEAKER_00.wav', 'abcdefghijk'),
            ('prefix_abcdefghijk_suffix.wav', 'abcdefghijk'),
            ('not-a-video-id.wav', None),
        ],
    )
    def test_extract_video_id_from_filename(self, filename, expected):
        fetcher = YouTubeMetadataFetcher()
        assert fetcher.extract_video_id_from_filename(filename) == expected

    def test_fetch_metadata_for_directory_deduplicates_ids(self, tmp_path):
        fetcher = YouTubeMetadataFetcher()
        directory = tmp_path / 'vocals'
        directory.mkdir()
        (directory / 'abcdefghijk_vocals.wav').touch()
        (directory / 'abcdefghijk.wav').touch()
        (directory / 'xyzxyzxyz12_vocals.wav').touch()

        with patch.object(fetcher, 'fetch_metadata') as mock_fetch:
            mock_fetch.side_effect = [
                VideoMetadata('abcdefghijk', 'A', 'A', '20260101', 1.0, None),
                VideoMetadata('xyzxyzxyz12', 'B', 'B', '20260102', 2.0, None),
            ]

            results = fetcher.fetch_metadata_for_directory(directory)

        assert sorted(results) == ['abcdefghijk', 'xyzxyzxyz12']
        assert mock_fetch.call_count == 2


class TestPopulateDatabaseFromFiles:
    """Database population tests."""

    def test_populate_database_from_files_with_metadata(self, tmp_path):
        separated_dir = tmp_path / 'separated'
        separated_dir.mkdir()
        diarized_dir = tmp_path / 'diarized'
        diarized_dir.mkdir()

        vocal_file = separated_dir / 'abcdefghijk_vocals.wav'
        vocal_file.touch()
        (diarized_dir / 'abcdefghijk_diarization.json').write_text('{}')

        metadata = VideoMetadata(
            video_id='abcdefghijk',
            title='Main Artist ft. Guest Artist - Song Title',
            channel='Main Artist',
            upload_date='20260101',
            duration_sec=180.0,
            description='',
        )

        with patch('auto_voice.audio.youtube_metadata.YouTubeMetadataFetcher.fetch_metadata', return_value=metadata), \
             patch('auto_voice.db.operations.upsert_track') as mock_upsert, \
             patch('auto_voice.db.operations.add_featured_artist') as mock_add_featured:
            stats = populate_database_from_files('main_artist', separated_dir, diarized_dir)

        assert stats['tracks_processed'] == 1
        assert stats['tracks_with_metadata'] == 1
        assert stats['featured_artists_found'] == 1
        assert not stats['errors']
        assert mock_upsert.called
        mock_add_featured.assert_called_once_with(
            track_id='abcdefghijk',
            name='Guest Artist',
            pattern_matched='title',
        )

    def test_populate_database_from_files_without_metadata(self, tmp_path):
        separated_dir = tmp_path / 'separated'
        separated_dir.mkdir()
        vocal_file = separated_dir / 'abcdefghijk_vocals.wav'
        vocal_file.touch()

        with patch('auto_voice.audio.youtube_metadata.YouTubeMetadataFetcher.fetch_metadata', return_value=None), \
             patch('auto_voice.db.operations.upsert_track') as mock_upsert, \
             patch('auto_voice.db.operations.add_featured_artist') as mock_add_featured:
            stats = populate_database_from_files('main_artist', separated_dir)

        assert stats['tracks_processed'] == 1
        assert stats['tracks_with_metadata'] == 0
        assert stats['featured_artists_found'] == 0
        assert stats['errors'] == ['Could not fetch metadata for abcdefghijk']
        assert mock_upsert.called
        mock_add_featured.assert_not_called()

    def test_populate_database_from_files_resolves_from_explicit_data_dir(self, tmp_path):
        data_dir = tmp_path / 'runtime-data'
        separated_dir = data_dir / 'separated_youtube' / 'main_artist'
        separated_dir.mkdir(parents=True)
        diarized_dir = data_dir / 'diarized_youtube' / 'main_artist'
        diarized_dir.mkdir(parents=True)

        vocal_file = separated_dir / 'abcdefghijk_vocals.wav'
        vocal_file.touch()
        diarization_file = diarized_dir / 'abcdefghijk_diarization.json'
        diarization_file.write_text('{}')

        with patch('auto_voice.audio.youtube_metadata.YouTubeMetadataFetcher.fetch_metadata', return_value=None), \
             patch('auto_voice.db.operations.upsert_track') as mock_upsert, \
             patch('auto_voice.db.operations.add_featured_artist') as mock_add_featured:
            stats = populate_database_from_files('main_artist', data_dir=data_dir)

        assert stats['tracks_processed'] == 1
        assert stats['errors'] == ['Could not fetch metadata for abcdefghijk']
        mock_upsert.assert_called_once_with(
            track_id='abcdefghijk',
            artist_name='main_artist',
            vocals_path=str(vocal_file),
            diarization_path=str(diarization_file),
        )
        mock_add_featured.assert_not_called()

    def test_populate_database_from_files_resolves_from_data_dir_env(self, tmp_path, monkeypatch):
        data_dir = tmp_path / 'env-data'
        separated_dir = data_dir / 'separated_youtube' / 'main_artist'
        separated_dir.mkdir(parents=True)
        diarized_dir = data_dir / 'diarized_youtube' / 'main_artist'
        diarized_dir.mkdir(parents=True)

        vocal_file = separated_dir / 'abcdefghijk_vocals.wav'
        vocal_file.touch()
        diarization_file = diarized_dir / 'abcdefghijk_diarization.json'
        diarization_file.write_text('{}')
        monkeypatch.setenv('DATA_DIR', str(data_dir))

        with patch('auto_voice.audio.youtube_metadata.YouTubeMetadataFetcher.fetch_metadata', return_value=None), \
             patch('auto_voice.db.operations.upsert_track') as mock_upsert, \
             patch('auto_voice.db.operations.add_featured_artist') as mock_add_featured:
            stats = populate_database_from_files('main_artist')

        assert stats['tracks_processed'] == 1
        assert stats['errors'] == ['Could not fetch metadata for abcdefghijk']
        mock_upsert.assert_called_once_with(
            track_id='abcdefghijk',
            artist_name='main_artist',
            vocals_path=str(vocal_file),
            diarization_path=str(diarization_file),
        )
        mock_add_featured.assert_not_called()
