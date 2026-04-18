"""Tests for youtube_downloader.py - YouTube audio downloading.

Task 2.4: Test youtube_downloader.py
- Test successful download (5s clip)
- Test format extraction (audio-only)
- Test error handling (404, geo-block, invalid URL)
- Test metadata extraction (title, artist)
"""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from auto_voice.audio.youtube_downloader import (
    YouTubeDownloader,
    YouTubeDownloadResult,
    _find_ytdlp,
    get_downloader,
)


@pytest.fixture
def mock_metadata():
    """Mock YouTube metadata."""
    return {
        'id': 'test_video_id',
        'title': 'Artist - Song Title (ft. Featured Artist)',
        'duration': 180.5,
        'thumbnail': 'https://example.com/thumb.jpg',
        'uploader': 'Artist Channel',
        'channel': 'Artist Channel',
        'description': 'Official music video',
    }


@pytest.fixture
def downloader(tmp_path):
    """Create YouTubeDownloader instance."""
    return YouTubeDownloader(output_dir=str(tmp_path))


class TestYouTubeDownloader:
    """Test YouTubeDownloader initialization."""

    @pytest.mark.smoke
    def test_init_default(self):
        """Test default initialization."""
        downloader = YouTubeDownloader()
        assert Path(downloader.output_dir).exists()

    def test_init_custom_output_dir(self, tmp_path):
        """Test initialization with custom output directory."""
        output_dir = tmp_path / 'custom'
        downloader = YouTubeDownloader(output_dir=str(output_dir))

        assert downloader.output_dir == str(output_dir)
        assert Path(output_dir).exists()


class TestFindYtdlp:
    """Test yt-dlp executable finding."""

    @patch('shutil.which')
    def test_find_ytdlp_in_path(self, mock_which):
        """Test finding yt-dlp in PATH."""
        mock_which.return_value = '/usr/bin/yt-dlp'

        result = _find_ytdlp()
        assert result == '/usr/bin/yt-dlp'

    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('os.access')
    def test_find_ytdlp_common_locations(self, mock_access, mock_isfile, mock_which):
        """Test finding yt-dlp in common locations."""
        mock_which.return_value = None
        mock_isfile.return_value = True
        mock_access.return_value = True

        result = _find_ytdlp()
        assert result is not None

    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_find_ytdlp_fallback(self, mock_isfile, mock_which):
        """Test fallback when yt-dlp not found."""
        mock_which.return_value = None
        mock_isfile.return_value = False

        result = _find_ytdlp()
        assert result == 'yt-dlp'  # Fallback value


class TestMetadataFetching:
    """Test metadata fetching functionality."""

    @patch('subprocess.run')
    def test_get_metadata_success(self, mock_run, downloader, mock_metadata):
        """Test successful metadata fetching."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_metadata)
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://www.youtube.com/watch?v=test_id')

        assert result is not None
        assert result['id'] == 'test_video_id'
        assert result['title'] == mock_metadata['title']

    @patch('subprocess.run')
    def test_get_metadata_failure(self, mock_run, downloader):
        """Test metadata fetching failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'Video not found'
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://www.youtube.com/watch?v=invalid')

        assert result is None

    @patch('subprocess.run')
    def test_get_metadata_timeout(self, mock_run, downloader):
        """Test metadata fetching timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=30)

        result = downloader._get_metadata('https://www.youtube.com/watch?v=test_id')

        assert result is None

    @patch('subprocess.run')
    def test_get_metadata_invalid_json(self, mock_run, downloader):
        """Test handling of invalid JSON response."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'not valid json'
        mock_run.return_value = mock_result

        result = downloader._get_metadata('https://www.youtube.com/watch?v=test_id')

        assert result is None


class TestAudioDownloading:
    """Test audio downloading functionality."""

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_download_audio_success(self, mock_exists, mock_run, downloader):
        """Test successful audio download."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        result = downloader._download_audio(
            'https://www.youtube.com/watch?v=test_id',
            '/tmp/test.wav',
            'wav',
            44100
        )

        assert result is True

    @patch('subprocess.run')
    def test_download_audio_failure(self, mock_run, downloader):
        """Test audio download failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'Download error'
        mock_run.return_value = mock_result

        result = downloader._download_audio(
            'https://www.youtube.com/watch?v=test_id',
            '/tmp/test.wav',
            'wav',
            44100
        )

        assert result is False

    @patch('subprocess.run')
    def test_download_audio_timeout(self, mock_run, downloader):
        """Test audio download timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=300)

        result = downloader._download_audio(
            'https://www.youtube.com/watch?v=test_id',
            '/tmp/test.wav',
            'wav',
            44100
        )

        assert result is False

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.rename')
    def test_download_audio_alternative_extension(self, mock_rename, mock_exists,
                                                  mock_run, downloader):
        """Test handling alternative file extensions."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # File doesn't exist with expected extension
        def exists_side_effect(path):
            if path.endswith('.wav'):
                return False
            elif path.endswith('.m4a'):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        result = downloader._download_audio(
            'https://www.youtube.com/watch?v=test_id',
            '/tmp/test.wav',
            'wav',
            44100
        )

        # Should rename and return success
        assert mock_rename.called


class TestDownloadMethod:
    """Test the main download() method."""

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_success(self, mock_download, mock_get_meta, downloader, mock_metadata):
        """Test successful complete download."""
        mock_get_meta.return_value = mock_metadata
        mock_download.return_value = True

        result = downloader.download('https://www.youtube.com/watch?v=test_id')

        assert result.success is True
        assert result.audio_path is not None
        assert result.title == mock_metadata['title']
        assert result.video_id == 'test_video_id'

    @patch.object(YouTubeDownloader, '_get_metadata')
    def test_download_metadata_failure(self, mock_get_meta, downloader):
        """Test download failure due to metadata fetch error."""
        mock_get_meta.return_value = None

        result = downloader.download('https://www.youtube.com/watch?v=invalid')

        assert result.success is False
        assert result.error is not None

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_audio_failure(self, mock_download, mock_get_meta,
                                   downloader, mock_metadata):
        """Test download failure during audio extraction."""
        mock_get_meta.return_value = mock_metadata
        mock_download.return_value = False

        result = downloader.download('https://www.youtube.com/watch?v=test_id')

        assert result.success is False
        assert result.error is not None

    @patch.object(YouTubeDownloader, '_get_metadata')
    @patch.object(YouTubeDownloader, '_download_audio')
    def test_download_custom_format(self, mock_download, mock_get_meta,
                                    downloader, mock_metadata):
        """Test download with custom audio format."""
        mock_get_meta.return_value = mock_metadata
        mock_download.return_value = True

        result = downloader.download(
            'https://www.youtube.com/watch?v=test_id',
            audio_format='mp3',
            sample_rate=22050
        )

        # Verify download was called with correct format
        assert mock_download.called
        call_args = mock_download.call_args[0]
        assert call_args[2] == 'mp3'
        assert call_args[3] == 22050


class TestGetVideoInfo:
    """Test get_video_info method."""

    @patch.object(YouTubeDownloader, '_get_metadata')
    def test_get_video_info_success(self, mock_get_meta, downloader, mock_metadata):
        """Test successful video info retrieval."""
        mock_get_meta.return_value = mock_metadata

        result = downloader.get_video_info('https://www.youtube.com/watch?v=test_id')

        assert result.success is True
        assert result.audio_path is None  # No download
        assert result.title == mock_metadata['title']
        assert result.duration == mock_metadata['duration']

    @patch.object(YouTubeDownloader, '_get_metadata')
    def test_get_video_info_failure(self, mock_get_meta, downloader):
        """Test video info retrieval failure."""
        mock_get_meta.return_value = None

        result = downloader.get_video_info('https://www.youtube.com/watch?v=invalid')

        assert result.success is False
        assert result.error is not None


class TestFilenameSanitization:
    """Test filename sanitization."""

    def test_sanitize_filename(self, downloader):
        """Test sanitizing filenames."""
        # Test removing invalid characters
        result = downloader._sanitize_filename('Artist - Song: Title (Official)')
        assert ':' not in result
        assert result == 'Artist_-_Song_Title_(Official)'

        # Test length limiting
        long_name = 'A' * 200
        result = downloader._sanitize_filename(long_name)
        assert len(result) <= 100

        # Test space replacement
        result = downloader._sanitize_filename('Artist Song Title')
        assert ' ' not in result
        assert '_' in result


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_get_downloader_singleton(self, tmp_path):
        """Test get_downloader creates singleton."""
        downloader1 = get_downloader(str(tmp_path))
        downloader2 = get_downloader(str(tmp_path))

        assert downloader1 is downloader2

    def test_get_downloader_different_dir(self, tmp_path):
        """Test get_downloader with different directories."""
        dir1 = tmp_path / 'dir1'
        dir2 = tmp_path / 'dir2'

        downloader1 = get_downloader(str(dir1))
        downloader2 = get_downloader(str(dir2))

        assert downloader1 is not downloader2
        assert downloader2.output_dir == str(dir2)


@pytest.mark.integration
class TestYouTubeDownloaderIntegration:
    """Integration tests for complete download workflow."""

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_complete_download_workflow(self, mock_exists, mock_run, tmp_path, mock_metadata):
        """Test complete download from URL to file."""
        # Mock metadata fetch
        metadata_result = MagicMock()
        metadata_result.returncode = 0
        metadata_result.stdout = json.dumps(mock_metadata)

        # Mock audio download
        download_result = MagicMock()
        download_result.returncode = 0

        mock_run.side_effect = [metadata_result, download_result]
        mock_exists.return_value = True

        downloader = YouTubeDownloader(output_dir=str(tmp_path))
        result = downloader.download('https://www.youtube.com/watch?v=test_id')

        assert result.success is True
        assert result.main_artist is not None
        assert len(result.featured_artists) > 0  # Should parse "ft. Featured Artist"

    @patch('subprocess.run')
    def test_error_handling_404(self, mock_run, downloader):
        """Test handling of 404 video not found error."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'ERROR: Video unavailable'
        mock_run.return_value = mock_result

        result = downloader.download('https://www.youtube.com/watch?v=nonexistent')

        assert result.success is False
        assert 'metadata' in result.error.lower() or 'failed' in result.error.lower()

    @patch('subprocess.run')
    def test_error_handling_invalid_url(self, mock_run, downloader):
        """Test handling of invalid URL."""
        mock_run.side_effect = Exception("Invalid URL format")

        result = downloader.download('not-a-valid-url')

        assert result.success is False
        assert result.error is not None
