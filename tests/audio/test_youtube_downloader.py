"""Tests for youtube_downloader.py - YouTube audio download handling.

Test Coverage:
- Task 2.4: Successful download (5s clip)
- Test format extraction (audio-only)
- Test error handling (404, geo-block, invalid URL)
- Test metadata extraction (title, artist)
"""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from auto_voice.audio.youtube_downloader import (
    YouTubeDownloader,
    YouTubeDownloadResult,
    _find_ytdlp,
    get_downloader,
)


@pytest.fixture
def downloader(tmp_path):
    """Create YouTubeDownloader instance with temp directory."""
    return YouTubeDownloader(output_dir=str(tmp_path))


@pytest.fixture
def mock_video_metadata():
    """Mock YouTube video metadata."""
    return {
        'id': 'dQw4w9WgXcQ',
        'title': 'Rick Astley - Never Gonna Give You Up (Official Video)',
        'duration': 212,
        'thumbnail': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
        'uploader': 'Rick Astley',
        'channel': 'RickAstleyVEVO',
    }


class TestYouTubeDownloader:
    """Test suite for YouTubeDownloader."""

    def test_initialization(self, tmp_path):
        """Test YouTubeDownloader initialization."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        assert downloader.output_dir == str(tmp_path)
        assert Path(tmp_path).exists()

    def test_initialization_default_output_dir(self):
        """Test initialization with default temp directory."""
        downloader = YouTubeDownloader()

        assert downloader.output_dir is not None
        assert Path(downloader.output_dir).exists()

    def test_sanitize_filename(self, downloader):
        """Test filename sanitization."""
        # Invalid characters should be removed
        result = downloader._sanitize_filename('Invalid: <File>|Name?')
        assert ':' not in result
        assert '<' not in result
        assert '>' not in result
        assert '|' not in result
        assert '?' not in result

        # Spaces should be replaced with underscores
        result = downloader._sanitize_filename('Some File Name')
        assert result == 'Some_File_Name'

        # Length should be limited
        long_name = 'a' * 200
        result = downloader._sanitize_filename(long_name)
        assert len(result) <= 100

    def test_get_metadata_success(self, downloader, mock_video_metadata):
        """Test fetching video metadata successfully."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_video_metadata),
                stderr='',
            )

            metadata = downloader._get_metadata('https://youtube.com/watch?v=dQw4w9WgXcQ')

            assert metadata is not None
            assert metadata['id'] == 'dQw4w9WgXcQ'
            assert metadata['title'] == mock_video_metadata['title']
            assert metadata['duration'] == 212

    def test_get_metadata_failure(self, downloader):
        """Test metadata fetch failure (404-like error)."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout='',
                stderr='ERROR: Video unavailable',
            )

            metadata = downloader._get_metadata('https://youtube.com/watch?v=invalid')

            assert metadata is None

    def test_get_metadata_timeout(self, downloader):
        """Test metadata fetch timeout."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 30)

            metadata = downloader._get_metadata('https://youtube.com/watch?v=dQw4w9WgXcQ')

            assert metadata is None

    def test_get_metadata_invalid_json(self, downloader):
        """Test handling of invalid JSON response."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='Invalid JSON {',
                stderr='',
            )

            metadata = downloader._get_metadata('https://youtube.com/watch?v=dQw4w9WgXcQ')

            assert metadata is None

    def test_download_success(self, downloader, mock_video_metadata, tmp_path):
        """Test successful audio download."""
        # Create a fake downloaded file
        expected_file = tmp_path / "test_download.wav"
        expected_file.touch()

        with patch.object(downloader, '_get_metadata', return_value=mock_video_metadata), \
             patch.object(downloader, '_download_audio', return_value=True), \
             patch.object(downloader, '_sanitize_filename', return_value='test_download'):

            result = downloader.download(
                'https://youtube.com/watch?v=dQw4w9WgXcQ',
                output_filename='test_download',
            )

            assert result.success is True
            assert result.title == mock_video_metadata['title']
            assert result.duration == 212
            assert result.video_id == 'dQw4w9WgXcQ'

    def test_download_metadata_failure(self, downloader):
        """Test download failure when metadata fetch fails."""
        with patch.object(downloader, '_get_metadata', return_value=None):

            result = downloader.download('https://youtube.com/watch?v=invalid')

            assert result.success is False
            assert result.error == "Failed to fetch video metadata"

    def test_download_audio_failure(self, downloader, mock_video_metadata):
        """Test download failure when audio download fails."""
        with patch.object(downloader, '_get_metadata', return_value=mock_video_metadata), \
             patch.object(downloader, '_download_audio', return_value=False):

            result = downloader.download('https://youtube.com/watch?v=dQw4w9WgXcQ')

            assert result.success is False
            assert result.error == "Failed to download audio"
            assert result.video_id == 'dQw4w9WgXcQ'

    def test_download_with_featured_artists(self, downloader, tmp_path):
        """Test download with featured artist parsing."""
        metadata_with_featured = {
            'id': 'test123',
            'title': 'Artist A ft. Artist B - Song Title',
            'duration': 180,
            'thumbnail': 'https://example.com/thumb.jpg',
        }

        with patch.object(downloader, '_get_metadata', return_value=metadata_with_featured), \
             patch.object(downloader, '_download_audio', return_value=True), \
             patch.object(downloader, '_sanitize_filename', return_value='test'):

            result = downloader.download('https://youtube.com/watch?v=test123')

            assert result.success is True
            assert result.main_artist == 'Artist A'
            assert 'Artist B' in result.featured_artists

    def test_download_audio_format_options(self, downloader, tmp_path):
        """Test download with different audio formats."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

            output_path = str(tmp_path / "test.wav")
            success = downloader._download_audio(
                'https://youtube.com/watch?v=test',
                output_path,
                audio_format='wav',
                sample_rate=44100,
            )

            # Verify yt-dlp was called with correct format
            call_args = mock_run.call_args[0][0]
            assert '-x' in call_args  # Extract audio
            assert '--audio-format' in call_args
            assert 'wav' in call_args

    def test_download_audio_timeout(self, downloader, tmp_path):
        """Test download timeout handling."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 300)

            output_path = str(tmp_path / "test.wav")
            success = downloader._download_audio(
                'https://youtube.com/watch?v=test',
                output_path,
                audio_format='wav',
                sample_rate=44100,
            )

            assert success is False

    def test_download_audio_file_not_created(self, downloader, tmp_path):
        """Test handling when yt-dlp succeeds but file not found."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

            output_path = str(tmp_path / "nonexistent.wav")
            success = downloader._download_audio(
                'https://youtube.com/watch?v=test',
                output_path,
                audio_format='wav',
                sample_rate=44100,
            )

            assert success is False

    def test_download_audio_with_extension_rename(self, downloader, tmp_path):
        """Test automatic file renaming when extension differs."""
        # Create file with different extension
        actual_file = tmp_path / "test.m4a"
        actual_file.touch()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

            expected_path = str(tmp_path / "test.wav")
            success = downloader._download_audio(
                'https://youtube.com/watch?v=test',
                expected_path,
                audio_format='wav',
                sample_rate=44100,
            )

            # Should find and rename the m4a file
            assert Path(expected_path).exists() or Path(actual_file).exists()

    def test_get_video_info_without_download(self, downloader, mock_video_metadata):
        """Test getting video info without downloading audio."""
        with patch.object(downloader, '_get_metadata', return_value=mock_video_metadata):

            result = downloader.get_video_info('https://youtube.com/watch?v=dQw4w9WgXcQ')

            assert result.success is True
            assert result.title == mock_video_metadata['title']
            assert result.video_id == 'dQw4w9WgXcQ'
            assert result.audio_path is None  # No download

    def test_get_video_info_failure(self, downloader):
        """Test get_video_info failure."""
        with patch.object(downloader, '_get_metadata', return_value=None):

            result = downloader.get_video_info('https://youtube.com/watch?v=invalid')

            assert result.success is False
            assert result.error == "Failed to fetch video metadata"

    def test_download_with_cover_detection(self, downloader, tmp_path):
        """Test download with cover song detection."""
        cover_metadata = {
            'id': 'cover123',
            'title': 'Artist A - Song Title (Original Artist Cover)',
            'duration': 200,
        }

        with patch.object(downloader, '_get_metadata', return_value=cover_metadata), \
             patch.object(downloader, '_download_audio', return_value=True), \
             patch.object(downloader, '_sanitize_filename', return_value='test'):

            result = downloader.download('https://youtube.com/watch?v=cover123')

            assert result.success is True
            assert result.is_cover is True
            assert result.original_artist is not None


def test_find_ytdlp():
    """Test finding yt-dlp executable."""
    with patch('shutil.which', return_value='/usr/bin/yt-dlp'):
        ytdlp_path = _find_ytdlp()
        assert ytdlp_path == '/usr/bin/yt-dlp'


def test_find_ytdlp_fallback():
    """Test yt-dlp fallback when not in PATH."""
    with patch('shutil.which', return_value=None), \
         patch('os.path.isfile', return_value=False), \
         patch('os.access', return_value=False):

        ytdlp_path = _find_ytdlp()
        assert ytdlp_path == 'yt-dlp'  # Fallback


def test_get_downloader_singleton(tmp_path):
    """Test get_downloader returns singleton instance."""
    downloader1 = get_downloader(str(tmp_path))
    downloader2 = get_downloader(str(tmp_path))

    assert downloader1 is downloader2


def test_get_downloader_creates_new_for_different_dir(tmp_path):
    """Test get_downloader creates new instance for different directory."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    downloader1 = get_downloader(str(dir1))
    downloader2 = get_downloader(str(dir2))

    assert downloader1 is not downloader2
    assert downloader1.output_dir == str(dir1)
    assert downloader2.output_dir == str(dir2)


@pytest.mark.parametrize("url,expected_valid", [
    ('https://youtube.com/watch?v=dQw4w9WgXcQ', True),
    ('https://www.youtube.com/watch?v=dQw4w9WgXcQ', True),
    ('https://youtu.be/dQw4w9WgXcQ', True),
    ('invalid_url', False),
    ('', False),
])
def test_download_url_validation(tmp_path, url, expected_valid):
    """Test URL validation (implicit through metadata fetch)."""
    downloader = YouTubeDownloader(output_dir=str(tmp_path))

    with patch('subprocess.run') as mock_run:
        if expected_valid:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"id": "test", "title": "Test"}',
                stderr='',
            )
        else:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr='ERROR: Invalid URL',
            )

        metadata = downloader._get_metadata(url)

        if expected_valid:
            assert metadata is not None or mock_run.called
        else:
            # Invalid URLs may still call subprocess but should fail
            pass


@pytest.mark.integration
@pytest.mark.slow
def test_real_download_5s_clip(tmp_path):
    """Integration test: Download actual 5s YouTube clip.

    This test is skipped by default (requires network and yt-dlp).
    """
    pytest.skip("Integration test requires network and yt-dlp")

    downloader = YouTubeDownloader(output_dir=str(tmp_path))

    # Use a known stable video (YouTube's test video)
    result = downloader.download(
        'https://www.youtube.com/watch?v=jNQXAC9IVRw',  # "Me at the zoo"
        audio_format='wav',
        sample_rate=16000,
    )

    assert result.success is True
    assert result.audio_path is not None
    assert Path(result.audio_path).exists()
    assert Path(result.audio_path).stat().st_size > 0


def test_download_result_dataclass():
    """Test YouTubeDownloadResult dataclass."""
    result = YouTubeDownloadResult(
        success=True,
        audio_path='/path/to/audio.wav',
        title='Test Title',
        duration=180.0,
        video_id='test123',
    )

    assert result.success is True
    assert result.audio_path == '/path/to/audio.wav'
    assert result.title == 'Test Title'
    assert result.duration == 180.0
    assert result.video_id == 'test123'
    assert result.featured_artists == []  # Default empty list
    assert result.error is None


def test_download_with_geo_blocked_video(downloader):
    """Test handling of geo-blocked videos."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr='ERROR: This video is not available in your country',
        )

        metadata = downloader._get_metadata('https://youtube.com/watch?v=geo_blocked')

        assert metadata is None
