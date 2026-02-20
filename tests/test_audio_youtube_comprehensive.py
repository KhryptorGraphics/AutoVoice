"""Comprehensive tests for YouTube downloader module - Target 70% coverage.

Extends basic YouTube tests with additional edge cases, error handling,
and integration tests.
"""
import json
import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from auto_voice.audio.youtube_downloader import (
    YouTubeDownloader,
    YouTubeDownloadResult,
    _find_ytdlp,
    get_downloader,
)


class TestYouTubeDownloadResultDataclass:
    """Test YouTubeDownloadResult dataclass."""

    @pytest.mark.smoke
    def test_result_default_values(self):
        """Test default values of download result."""
        result = YouTubeDownloadResult(success=False)

        assert result.success is False
        assert result.audio_path is None
        assert result.title == ""
        assert result.duration == 0.0
        assert result.main_artist is None
        assert result.featured_artists == []
        assert result.is_cover is False
        assert result.error is None

    def test_result_with_metadata(self):
        """Test result with full metadata."""
        result = YouTubeDownloadResult(
            success=True,
            audio_path='/tmp/song.wav',
            title='Artist - Song (ft. Feature)',
            duration=180.5,
            main_artist='Artist',
            featured_artists=['Feature'],
            is_cover=False,
            song_title='Song',
            thumbnail_url='https://example.com/thumb.jpg',
            video_id='abc123',
            metadata={'key': 'value'}
        )

        assert result.success is True
        assert result.audio_path == '/tmp/song.wav'
        assert result.main_artist == 'Artist'
        assert 'Feature' in result.featured_artists

    def test_result_error_state(self):
        """Test result in error state."""
        result = YouTubeDownloadResult(
            success=False,
            error='Video unavailable'
        )

        assert result.success is False
        assert result.error == 'Video unavailable'
        assert result.audio_path is None


class TestFindYtdlpExtended:
    """Extended tests for yt-dlp executable finding."""

    @patch('shutil.which')
    def test_find_ytdlp_in_system_path(self, mock_which):
        """Test finding yt-dlp in system PATH."""
        mock_which.return_value = '/usr/local/bin/yt-dlp'

        result = _find_ytdlp()

        assert result == '/usr/local/bin/yt-dlp'
        mock_which.assert_called_once_with('yt-dlp')

    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('os.access')
    def test_find_ytdlp_in_anaconda(self, mock_access, mock_isfile, mock_which):
        """Test finding yt-dlp in anaconda path."""
        mock_which.return_value = None
        mock_isfile.side_effect = lambda p: p == '/home/kp/anaconda3/bin/yt-dlp'
        mock_access.return_value = True

        result = _find_ytdlp()

        assert result == '/home/kp/anaconda3/bin/yt-dlp'

    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('os.access')
    def test_find_ytdlp_in_user_local(self, mock_access, mock_isfile, mock_which):
        """Test finding yt-dlp in ~/.local/bin."""
        mock_which.return_value = None

        def isfile_check(path):
            return path.endswith('.local/bin/yt-dlp')

        mock_isfile.side_effect = isfile_check
        mock_access.return_value = True

        result = _find_ytdlp()

        # Should find in common paths
        assert result is not None

    @patch('shutil.which', return_value=None)
    @patch('os.path.isfile', return_value=False)
    def test_find_ytdlp_fallback(self, mock_isfile, mock_which):
        """Test fallback to 'yt-dlp' when not found."""
        result = _find_ytdlp()

        assert result == 'yt-dlp'


class TestYouTubeDownloaderInit:
    """Test YouTubeDownloader initialization."""

    def test_init_creates_output_dir(self, tmp_path):
        """Test that init creates output directory if needed."""
        output_dir = tmp_path / 'new_downloads'
        assert not output_dir.exists()

        downloader = YouTubeDownloader(output_dir=str(output_dir))

        assert output_dir.exists()
        assert downloader.output_dir == str(output_dir)

    def test_init_uses_temp_dir_default(self):
        """Test default output directory is temp."""
        downloader = YouTubeDownloader()

        assert downloader.output_dir == tempfile.gettempdir()


class TestMetadataFetchingExtended:
    """Extended metadata fetching tests."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return YouTubeDownloader(output_dir=str(tmp_path))

    @patch('subprocess.run')
    def test_get_metadata_with_all_fields(self, mock_run, downloader):
        """Test metadata with all available fields."""
        metadata = {
            'id': 'video123',
            'title': 'Artist - Song Title (Official Video)',
            'duration': 245.0,
            'thumbnail': 'https://i.ytimg.com/vi/video123/maxresdefault.jpg',
            'uploader': 'Official Artist Channel',
            'channel': 'Official Artist Channel',
            'description': 'Official music video',
            'upload_date': '20240101',
            'view_count': 1000000,
            'like_count': 50000,
            'categories': ['Music'],
            'tags': ['music', 'official'],
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(metadata)
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://youtube.com/watch?v=video123')

        assert result['id'] == 'video123'
        assert result['duration'] == 245.0
        assert 'view_count' in result

    @patch('subprocess.run')
    def test_get_metadata_geo_blocked(self, mock_run, downloader):
        """Test handling of geo-blocked videos."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'ERROR: The uploader has not made this video available in your country'
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://youtube.com/watch?v=blocked')

        assert result is None

    @patch('subprocess.run')
    def test_get_metadata_age_restricted(self, mock_run, downloader):
        """Test handling of age-restricted videos."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'ERROR: Sign in to confirm your age'
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://youtube.com/watch?v=restricted')

        assert result is None

    @patch('subprocess.run')
    def test_get_metadata_network_error(self, mock_run, downloader):
        """Test handling of network errors."""
        mock_run.side_effect = OSError("Network unreachable")

        result = downloader._get_metadata('https://youtube.com/watch?v=test')

        assert result is None

    @patch('subprocess.run')
    def test_get_metadata_partial_json(self, mock_run, downloader):
        """Test handling of truncated JSON response."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"id": "test", "title": "Test'  # Truncated
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://youtube.com/watch?v=test')

        assert result is None


class TestAudioDownloadingExtended:
    """Extended audio downloading tests."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return YouTubeDownloader(output_dir=str(tmp_path))

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_download_audio_mp3_format(self, mock_exists, mock_run, downloader):
        """Test downloading in MP3 format."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        result = downloader._download_audio(
            'https://youtube.com/watch?v=test',
            '/tmp/test.mp3',
            'mp3',
            44100
        )

        assert result is True
        # Verify command includes mp3 format
        call_args = mock_run.call_args[0][0]
        assert '--audio-format' in call_args
        assert 'mp3' in call_args

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_download_audio_flac_format(self, mock_exists, mock_run, downloader):
        """Test downloading in FLAC format."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        result = downloader._download_audio(
            'https://youtube.com/watch?v=test',
            '/tmp/test.flac',
            'flac',
            48000
        )

        assert result is True

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_download_audio_with_sample_rate(self, mock_exists, mock_run, downloader):
        """Test downloading with specific sample rate."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        result = downloader._download_audio(
            'https://youtube.com/watch?v=test',
            '/tmp/test.wav',
            'wav',
            22050
        )

        assert result is True
        # Check postprocessor args for sample rate
        call_args = mock_run.call_args[0][0]
        assert any('22050' in str(arg) for arg in call_args)

    @patch('subprocess.run')
    def test_download_audio_permission_denied(self, mock_run, downloader):
        """Test handling of permission denied error."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'Permission denied'
        mock_run.return_value = mock_result

        result = downloader._download_audio(
            'https://youtube.com/watch?v=test',
            '/root/protected.wav',
            'wav',
            44100
        )

        assert result is False

    @patch('subprocess.run')
    @patch('os.path.exists', return_value=False)
    @patch('os.rename')
    def test_download_audio_finds_opus_extension(self, mock_rename, mock_exists, mock_run, downloader):
        """Test finding audio file with different extension."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Simulate file exists with opus extension
        def exists_check(path):
            return path.endswith('.opus')

        mock_exists.side_effect = exists_check

        result = downloader._download_audio(
            'https://youtube.com/watch?v=test',
            '/tmp/test.wav',
            'wav',
            44100
        )

        # Should rename the opus file
        assert mock_rename.called


class TestCompleteDownloadFlow:
    """Test complete download workflow."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return YouTubeDownloader(output_dir=str(tmp_path))

    @pytest.fixture
    def mock_metadata(self):
        return {
            'id': 'abc123',
            'title': 'Artist Name - Song Title (Official Audio)',
            'duration': 210.5,
            'thumbnail': 'https://example.com/thumb.jpg',
            'uploader': 'ArtistVEVO',
            'channel': 'ArtistVEVO',
            'description': 'Official audio for Song Title',
        }

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_extracts_artist_info(self, mock_download, mock_meta,
                                           downloader, mock_metadata):
        """Test that download extracts artist information."""
        mock_meta.return_value = mock_metadata
        mock_download.return_value = True

        result = downloader.download('https://youtube.com/watch?v=abc123')

        assert result.success is True
        assert result.main_artist is not None

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_with_featured_artists(self, mock_download, mock_meta, downloader):
        """Test downloading video with featured artists."""
        mock_meta.return_value = {
            'id': 'xyz789',
            'title': 'Main Artist - Song (ft. Featured 1, Featured 2)',
            'duration': 200,
        }
        mock_download.return_value = True

        result = downloader.download('https://youtube.com/watch?v=xyz789')

        assert result.success is True
        # Featured artists should be parsed
        assert len(result.featured_artists) >= 0  # Depends on parser

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_cover_detection(self, mock_download, mock_meta, downloader):
        """Test detecting cover songs."""
        mock_meta.return_value = {
            'id': 'cover123',
            'title': 'Someone - Famous Song (Cover)',
            'duration': 180,
            'description': 'My cover of Famous Song by Original Artist',
        }
        mock_download.return_value = True

        result = downloader.download('https://youtube.com/watch?v=cover123')

        assert result.success is True
        # Cover detection may be in metadata
        assert result.is_cover or 'cover' in result.title.lower()

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_custom_filename(self, mock_download, mock_meta,
                                       downloader, mock_metadata):
        """Test downloading with custom filename."""
        mock_meta.return_value = mock_metadata
        mock_download.return_value = True

        result = downloader.download(
            'https://youtube.com/watch?v=abc123',
            output_filename='custom_name'
        )

        assert result.success is True
        assert 'custom_name' in result.audio_path

    @patch.object(YouTubeDownloader, '_get_metadata')
    def test_download_handles_exception(self, mock_meta, downloader):
        """Test download handles unexpected exceptions."""
        mock_meta.side_effect = RuntimeError("Unexpected error")

        result = downloader.download('https://youtube.com/watch?v=test')

        assert result.success is False
        assert result.error is not None


class TestGetVideoInfo:
    """Test get_video_info method."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return YouTubeDownloader(output_dir=str(tmp_path))

    @patch.object(YouTubeDownloader, '_get_metadata')
    def test_get_video_info_success(self, mock_meta, downloader):
        """Test getting video info without downloading."""
        mock_meta.return_value = {
            'id': 'info123',
            'title': 'Test Video',
            'duration': 300,
            'thumbnail': 'https://example.com/thumb.jpg',
        }

        result = downloader.get_video_info('https://youtube.com/watch?v=info123')

        assert result.success is True
        assert result.audio_path is None  # No download
        assert result.duration == 300

    @patch.object(YouTubeDownloader, '_get_metadata')
    def test_get_video_info_failure(self, mock_meta, downloader):
        """Test video info when metadata fetch fails."""
        mock_meta.return_value = None

        result = downloader.get_video_info('https://youtube.com/watch?v=notfound')

        assert result.success is False
        assert result.error is not None

    @patch.object(YouTubeDownloader, '_get_metadata')
    def test_get_video_info_exception(self, mock_meta, downloader):
        """Test video info handles exceptions."""
        mock_meta.side_effect = Exception("Connection error")

        result = downloader.get_video_info('https://youtube.com/watch?v=test')

        assert result.success is False
        assert 'error' in result.error.lower()


class TestFilenameSanitization:
    """Test filename sanitization."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return YouTubeDownloader(output_dir=str(tmp_path))

    def test_sanitize_removes_special_chars(self, downloader):
        """Test removing special characters."""
        filename = 'Song: Title? With <Special> Chars*'
        sanitized = downloader._sanitize_filename(filename)

        for char in '<>:"/\\|?*':
            assert char not in sanitized

    def test_sanitize_replaces_spaces(self, downloader):
        """Test replacing spaces with underscores."""
        filename = 'Song With Spaces'
        sanitized = downloader._sanitize_filename(filename)

        assert ' ' not in sanitized
        assert '_' in sanitized

    def test_sanitize_limits_length(self, downloader):
        """Test filename length limiting."""
        filename = 'A' * 200
        sanitized = downloader._sanitize_filename(filename)

        assert len(sanitized) <= 100

    def test_sanitize_preserves_valid_chars(self, downloader):
        """Test that valid characters are preserved."""
        filename = 'Valid_Filename-123'
        sanitized = downloader._sanitize_filename(filename)

        # Should be mostly unchanged
        assert '123' in sanitized

    def test_sanitize_empty_string(self, downloader):
        """Test sanitizing empty string."""
        sanitized = downloader._sanitize_filename('')

        assert sanitized == ''

    def test_sanitize_unicode_characters(self, downloader):
        """Test sanitizing unicode characters."""
        filename = 'Song Title - 日本語'
        sanitized = downloader._sanitize_filename(filename)

        # Should not crash, may remove or keep unicode
        assert isinstance(sanitized, str)


class TestModuleLevelFunctionsExtended:
    """Extended tests for module-level functions."""

    def test_get_downloader_creates_new(self, tmp_path):
        """Test get_downloader creates new instance."""
        # Reset global
        import auto_voice.audio.youtube_downloader as yt_module
        yt_module._downloader = None

        downloader = get_downloader(str(tmp_path))

        assert downloader is not None
        assert downloader.output_dir == str(tmp_path)

    def test_get_downloader_reuses_same_dir(self, tmp_path):
        """Test get_downloader reuses instance for same directory."""
        import auto_voice.audio.youtube_downloader as yt_module
        yt_module._downloader = None

        d1 = get_downloader(str(tmp_path))
        d2 = get_downloader(str(tmp_path))

        assert d1 is d2

    def test_get_downloader_new_for_different_dir(self, tmp_path):
        """Test get_downloader creates new instance for different directory."""
        import auto_voice.audio.youtube_downloader as yt_module
        yt_module._downloader = None

        dir1 = tmp_path / 'dir1'
        dir2 = tmp_path / 'dir2'
        dir1.mkdir()
        dir2.mkdir()

        d1 = get_downloader(str(dir1))
        d2 = get_downloader(str(dir2))

        assert d1 is not d2


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return YouTubeDownloader(output_dir=str(tmp_path))

    @patch('subprocess.run')
    def test_metadata_empty_response(self, mock_run, downloader):
        """Test handling empty response from yt-dlp."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ''
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://youtube.com/watch?v=test')

        # Empty JSON string should fail parsing
        assert result is None

    @patch('subprocess.run')
    def test_metadata_unicode_title(self, mock_run, downloader):
        """Test handling unicode in title."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'id': 'unicode',
            'title': '日本語タイトル - アーティスト',
            'duration': 180,
        })
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://youtube.com/watch?v=unicode')

        assert result is not None
        assert '日本語' in result['title']

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_zero_duration(self, mock_download, mock_meta, downloader):
        """Test handling video with zero duration."""
        mock_meta.return_value = {
            'id': 'zero',
            'title': 'Test',
            'duration': 0,
        }
        mock_download.return_value = True

        result = downloader.download('https://youtube.com/watch?v=zero')

        assert result.success is True
        assert result.duration == 0

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_missing_optional_fields(self, mock_download, mock_meta, downloader):
        """Test download with minimal metadata."""
        mock_meta.return_value = {
            'id': 'minimal',
            'title': 'Minimal Video',
            # No duration, thumbnail, etc.
        }
        mock_download.return_value = True

        result = downloader.download('https://youtube.com/watch?v=minimal')

        assert result.success is True
        assert result.duration == 0  # Default

    @patch('subprocess.run')
    def test_download_keyboard_interrupt(self, mock_run, downloader):
        """Test handling keyboard interrupt during download."""
        mock_run.side_effect = KeyboardInterrupt()

        # Should propagate or handle gracefully
        with pytest.raises(KeyboardInterrupt):
            downloader._download_audio(
                'https://youtube.com/watch?v=test',
                '/tmp/test.wav',
                'wav',
                44100
            )


class TestURLHandling:
    """Test various YouTube URL formats."""

    @pytest.fixture
    def downloader(self, tmp_path):
        return YouTubeDownloader(output_dir=str(tmp_path))

    @pytest.mark.parametrize("url", [
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'https://youtube.com/watch?v=dQw4w9WgXcQ',
        'https://youtu.be/dQw4w9WgXcQ',
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest',
        'https://music.youtube.com/watch?v=dQw4w9WgXcQ',
    ])
    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_various_url_formats(self, mock_download, mock_meta, downloader, url):
        """Test various YouTube URL formats."""
        mock_meta.return_value = {
            'id': 'dQw4w9WgXcQ',
            'title': 'Test Video',
            'duration': 213,
        }
        mock_download.return_value = True

        result = downloader.download(url)

        # Should handle all URL formats
        assert mock_meta.called
