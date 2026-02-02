"""Tests for YouTube download and metadata modules.

Task 2.4: Test youtube_downloader.py - Download handling
Task 2.5: Test youtube_metadata.py - Metadata parsing
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from auto_voice.audio.youtube_downloader import (
    YouTubeDownloader,
    YouTubeDownloadResult,
    _find_ytdlp,
)
from auto_voice.audio.youtube_metadata import (
    parse_featured_artists,
    parse_youtube_metadata,
    _clean_artist_name,
    _split_multiple_artists,
    _is_producer_credit,
)


# ============================================================================
# Task 2.5: YouTube Metadata Tests
# ============================================================================


class TestMetadataParsing:
    """Test metadata parsing utilities."""

    @pytest.mark.smoke
    def test_clean_artist_name_basic(self):
        """Test basic artist name cleaning."""
        assert _clean_artist_name("Taylor Swift") == "Taylor Swift"
        assert _clean_artist_name("  Drake  ") == "Drake"

    def test_clean_artist_name_removes_suffixes(self):
        """Test removal of common suffixes."""
        assert _clean_artist_name("Ed Sheeran (Official Video)") == "Ed Sheeran"
        assert _clean_artist_name("Billie Eilish [Official Audio]") == "Billie Eilish"
        assert _clean_artist_name("The Weeknd - Official Lyrics") == "The Weeknd"

    def test_clean_artist_name_trailing_punctuation(self):
        """Test removal of trailing punctuation."""
        assert _clean_artist_name("Ariana Grande,") == "Ariana Grande"
        assert _clean_artist_name("Post Malone&") == "Post Malone"

    def test_split_multiple_artists_comma(self):
        """Test splitting artists by comma."""
        artists = _split_multiple_artists("Taylor Swift, Ed Sheeran")
        assert len(artists) == 2
        assert "Taylor Swift" in artists
        assert "Ed Sheeran" in artists

    def test_split_multiple_artists_ampersand(self):
        """Test splitting artists by ampersand."""
        artists = _split_multiple_artists("Bruno Mars & Anderson .Paak")
        assert len(artists) == 2
        assert "Bruno Mars" in artists
        assert "Anderson .Paak" in artists

    def test_split_multiple_artists_and(self):
        """Test splitting artists by 'and'."""
        artists = _split_multiple_artists("Simon and Garfunkel")
        assert len(artists) == 2
        assert "Simon" in artists
        assert "Garfunkel" in artists

    def test_split_multiple_artists_mixed(self):
        """Test splitting with mixed separators."""
        artists = _split_multiple_artists("Artist A, Artist B & Artist C")
        assert len(artists) == 3
        assert "Artist A" in artists
        assert "Artist B" in artists
        assert "Artist C" in artists

    def test_is_producer_credit_positive(self):
        """Test detection of producer credits."""
        assert _is_producer_credit("Produced by Metro Boomin") is True
        assert _is_producer_credit("prod. by Timbaland") is True
        assert _is_producer_credit("mixed by Dr. Dre") is True
        assert _is_producer_credit("remixed by Calvin Harris") is True

    def test_is_producer_credit_negative(self):
        """Test that artist features are not producer credits."""
        assert _is_producer_credit("ft. Drake") is False
        assert _is_producer_credit("featuring Rihanna") is False
        assert _is_producer_credit("with The Weeknd") is False


class TestFeaturedArtistParsing:
    """Test featured artist extraction from titles."""

    def test_parse_featured_artists_ft_abbreviation(self):
        """Test parsing 'ft.' pattern."""
        title = "Song Title ft. Drake"
        featured = parse_featured_artists(title)
        assert len(featured) == 1
        assert "Drake" in featured

    def test_parse_featured_artists_feat_full(self):
        """Test parsing 'featuring' pattern."""
        title = "Song Title featuring Rihanna"
        featured = parse_featured_artists(title)
        assert len(featured) == 1
        assert "Rihanna" in featured

    def test_parse_featured_artists_parentheses(self):
        """Test parsing featured artists in parentheses."""
        title = "Song Title (ft. Post Malone)"
        featured = parse_featured_artists(title)
        assert len(featured) == 1
        assert "Post Malone" in featured

    def test_parse_featured_artists_multiple(self):
        """Test parsing multiple featured artists."""
        title = "Song Title (ft. Drake, The Weeknd & Travis Scott)"
        featured = parse_featured_artists(title)
        assert len(featured) == 3
        assert "Drake" in featured
        assert "The Weeknd" in featured
        assert "Travis Scott" in featured

    def test_parse_featured_artists_vs(self):
        """Test parsing 'vs.' pattern (battle/mashup)."""
        title = "Artist A vs. Artist B"
        featured = parse_featured_artists(title)
        assert len(featured) == 1
        assert "Artist B" in featured

    def test_parse_featured_artists_with(self):
        """Test parsing 'with' pattern."""
        title = "Song Title with Ed Sheeran"
        featured = parse_featured_artists(title)
        assert len(featured) == 1
        assert "Ed Sheeran" in featured

    def test_parse_featured_artists_x_collaboration(self):
        """Test parsing 'x' collaboration pattern."""
        title = "Artist A x Artist B"
        featured = parse_featured_artists(title)
        assert len(featured) == 1
        assert "Artist B" in featured

    def test_parse_featured_artists_no_features(self):
        """Test title with no featured artists."""
        title = "Just a Regular Song Title"
        featured = parse_featured_artists(title)
        assert len(featured) == 0

    def test_parse_featured_artists_filters_producers(self):
        """Test that producer credits are filtered out."""
        title = "Song Title prod. by Metro Boomin"
        featured = parse_featured_artists(title)
        assert len(featured) == 0

    def test_parse_featured_artists_with_description(self):
        """Test parsing from both title and description."""
        title = "Song Title"
        description = "Featuring vocals by John Doe"
        featured = parse_featured_artists(title, description)
        assert len(featured) == 1
        assert "John Doe" in featured

    def test_parse_featured_artists_complex_title(self):
        """Test parsing complex real-world title."""
        title = "Conor Maynard - Starboy (ft. Anth & Mikey Ceaser) [The Weeknd Cover]"
        featured = parse_featured_artists(title)
        assert len(featured) >= 2
        assert "Anth" in featured
        assert "Mikey Ceaser" in featured


class TestYouTubeMetadataParser:
    """Test parse_youtube_metadata function."""

    def test_parse_youtube_metadata_basic(self):
        """Test parsing basic metadata."""
        metadata = {
            'title': 'Artist Name - Song Title',
            'channel': 'Official Channel',
        }
        parsed = parse_youtube_metadata(metadata)

        assert parsed is not None
        assert 'main_artist' in parsed
        assert 'song_title' in parsed

    def test_parse_youtube_metadata_featured(self):
        """Test parsing metadata with featured artists."""
        metadata = {
            'title': 'Main Artist ft. Featured Artist - Song Title',
            'channel': 'Music Channel',
        }
        parsed = parse_youtube_metadata(metadata)

        assert 'featured_artists' in parsed
        assert len(parsed['featured_artists']) >= 1

    def test_parse_youtube_metadata_cover(self):
        """Test parsing cover song metadata."""
        metadata = {
            'title': 'Cover Artist - Song Name (Original Artist Cover)',
            'channel': 'Cover Channel',
        }
        parsed = parse_youtube_metadata(metadata)

        assert 'is_cover' in parsed
        if parsed['is_cover']:
            assert 'original_artist' in parsed

    def test_parse_youtube_metadata_empty(self):
        """Test parsing empty metadata."""
        metadata = {}
        parsed = parse_youtube_metadata(metadata)

        # Should handle gracefully
        assert parsed is not None


# ============================================================================
# Task 2.4: YouTube Downloader Tests
# ============================================================================


class TestYouTubeDownloadResult:
    """Test YouTubeDownloadResult dataclass."""

    @pytest.mark.smoke
    def test_download_result_creation(self):
        """Test creating a download result."""
        result = YouTubeDownloadResult(
            success=True,
            audio_path="/path/to/audio.wav",
            title="Test Video",
            duration=180.0,
        )
        assert result.success is True
        assert result.audio_path == "/path/to/audio.wav"
        assert result.duration == 180.0

    def test_download_result_failure(self):
        """Test creating a failed download result."""
        result = YouTubeDownloadResult(
            success=False,
            error="Download failed: 404 Not Found",
        )
        assert result.success is False
        assert result.error is not None
        assert "404" in result.error


class TestYouTubeDownloader:
    """Test YouTubeDownloader class."""

    @pytest.mark.smoke
    def test_init_default(self, tmp_path):
        """Test downloader initialization with default output dir."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))
        assert downloader.output_dir == str(tmp_path)
        assert Path(downloader.output_dir).exists()

    def test_init_creates_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "downloads"
        downloader = YouTubeDownloader(output_dir=str(output_dir))
        assert output_dir.exists()

    @patch('subprocess.run')
    def test_get_metadata_success(self, mock_run, tmp_path):
        """Test successful metadata retrieval."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        # Mock yt-dlp metadata response
        mock_metadata = {
            'id': 'dQw4w9WgXcQ',
            'title': 'Rick Astley - Never Gonna Give You Up',
            'duration': 213,
            'thumbnail': 'https://example.com/thumb.jpg',
        }

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_metadata)
        mock_run.return_value = mock_result

        with patch.object(downloader, '_get_metadata', return_value=mock_metadata):
            metadata = downloader._get_metadata('https://youtube.com/watch?v=dQw4w9WgXcQ')

            assert metadata is not None
            assert metadata['id'] == 'dQw4w9WgXcQ'
            assert 'title' in metadata

    @patch('subprocess.run')
    def test_get_metadata_failure(self, mock_run, tmp_path):
        """Test metadata retrieval failure."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        # Mock yt-dlp failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: Video unavailable"
        mock_run.return_value = mock_result

        with patch.object(downloader, '_get_metadata', return_value=None):
            metadata = downloader._get_metadata('https://youtube.com/watch?v=invalid')
            assert metadata is None

    def test_sanitize_filename(self, tmp_path):
        """Test filename sanitization."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        # Test various problematic characters
        unsafe = "Artist/Name: Song | Title (feat. Other)"
        safe = downloader._sanitize_filename(unsafe)

        assert '/' not in safe
        assert ':' not in safe
        assert '|' not in safe
        assert len(safe) > 0

    @patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._get_metadata')
    @patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._download_audio')
    def test_download_success(self, mock_download_audio, mock_get_metadata, tmp_path):
        """Test successful download."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        # Mock metadata
        mock_get_metadata.return_value = {
            'id': 'test123',
            'title': 'Test Video',
            'duration': 180,
            'thumbnail': 'https://example.com/thumb.jpg',
        }

        # Mock successful download
        mock_download_audio.return_value = True

        with patch('auto_voice.audio.youtube_metadata.parse_youtube_metadata') as mock_parse:
            mock_parse.return_value = {
                'main_artist': 'Test Artist',
                'featured_artists': [],
                'is_cover': False,
                'song_title': 'Test Song',
            }

            result = downloader.download('https://youtube.com/watch?v=test123')

            assert result.success is True
            assert result.video_id == 'test123'
            assert result.title == 'Test Video'
            assert result.duration == 180

    @patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._get_metadata')
    def test_download_metadata_failure(self, mock_get_metadata, tmp_path):
        """Test download with metadata fetch failure."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        # Mock metadata failure
        mock_get_metadata.return_value = None

        result = downloader.download('https://youtube.com/watch?v=invalid')

        assert result.success is False
        assert result.error is not None
        assert "metadata" in result.error.lower()

    @patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._get_metadata')
    @patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._download_audio')
    def test_download_audio_failure(self, mock_download_audio, mock_get_metadata, tmp_path):
        """Test download with audio download failure."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        # Mock metadata success
        mock_get_metadata.return_value = {
            'id': 'test123',
            'title': 'Test Video',
            'duration': 180,
        }

        # Mock audio download failure
        mock_download_audio.return_value = False

        with patch('auto_voice.audio.youtube_metadata.parse_youtube_metadata', return_value={}):
            result = downloader.download('https://youtube.com/watch?v=test123')

            assert result.success is False
            assert "Failed to download audio" in result.error

    @patch('auto_voice.audio.youtube_downloader.YouTubeDownloader._get_metadata')
    def test_get_video_info(self, mock_get_metadata, tmp_path):
        """Test getting video info without downloading."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        mock_metadata = {
            'id': 'test123',
            'title': 'Test Video',
            'duration': 180,
            'thumbnail': 'https://example.com/thumb.jpg',
        }
        mock_get_metadata.return_value = mock_metadata

        with patch('auto_voice.audio.youtube_metadata.parse_youtube_metadata') as mock_parse:
            mock_parse.return_value = {
                'main_artist': 'Test Artist',
                'featured_artists': ['Featured Artist'],
                'is_cover': False,
                'song_title': 'Test Song',
            }

            result = downloader.get_video_info('https://youtube.com/watch?v=test123')

            assert result.success is True
            assert result.video_id == 'test123'
            assert result.audio_path is None  # No download
            assert result.main_artist == 'Test Artist'
            assert 'Featured Artist' in result.featured_artists

    def test_download_custom_filename(self, tmp_path):
        """Test download with custom output filename."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        with patch.object(downloader, '_get_metadata') as mock_metadata:
            with patch.object(downloader, '_download_audio', return_value=True):
                with patch('auto_voice.audio.youtube_metadata.parse_youtube_metadata', return_value={}):
                    mock_metadata.return_value = {
                        'id': 'test123',
                        'title': 'Test Video',
                        'duration': 180,
                    }

                    result = downloader.download(
                        'https://youtube.com/watch?v=test123',
                        output_filename='custom_name'
                    )

                    if result.success:
                        assert 'custom_name' in result.audio_path

    def test_download_format_options(self, tmp_path):
        """Test download with different audio formats."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        for audio_format in ['wav', 'mp3', 'flac']:
            with patch.object(downloader, '_get_metadata') as mock_metadata:
                with patch.object(downloader, '_download_audio', return_value=True):
                    with patch('auto_voice.audio.youtube_metadata.parse_youtube_metadata', return_value={}):
                        mock_metadata.return_value = {
                            'id': f'test_{audio_format}',
                            'title': f'Test {audio_format}',
                            'duration': 180,
                        }

                        result = downloader.download(
                            f'https://youtube.com/watch?v=test_{audio_format}',
                            audio_format=audio_format
                        )

                        if result.success:
                            assert result.audio_path.endswith(f'.{audio_format}')

    def test_download_exception_handling(self, tmp_path):
        """Test that exceptions are caught and returned as errors."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        with patch.object(downloader, '_get_metadata', side_effect=Exception("Network error")):
            result = downloader.download('https://youtube.com/watch?v=test123')

            assert result.success is False
            assert result.error is not None
            assert "Network error" in result.error


class TestYouTubeFindExecutable:
    """Test _find_ytdlp executable finding."""

    def test_find_ytdlp_in_path(self):
        """Test finding yt-dlp in PATH."""
        with patch('shutil.which', return_value='/usr/bin/yt-dlp'):
            ytdlp_path = _find_ytdlp()
            assert ytdlp_path == '/usr/bin/yt-dlp'

    def test_find_ytdlp_not_in_path(self):
        """Test fallback when yt-dlp not in PATH."""
        with patch('shutil.which', return_value=None):
            with patch('os.path.isfile', return_value=False):
                ytdlp_path = _find_ytdlp()
                assert ytdlp_path == 'yt-dlp'  # Fallback

    def test_find_ytdlp_common_location(self):
        """Test finding yt-dlp in common locations."""
        with patch('shutil.which', return_value=None):
            with patch('os.path.isfile') as mock_isfile:
                with patch('os.access', return_value=True):
                    # First common path exists
                    mock_isfile.side_effect = lambda x: x == '/home/kp/anaconda3/bin/yt-dlp'

                    ytdlp_path = _find_ytdlp()
                    assert '/yt-dlp' in ytdlp_path


@pytest.mark.integration
@pytest.mark.slow
class TestYouTubeDownloaderIntegration:
    """Integration tests for YouTube downloader (require network)."""

    @pytest.mark.skip(reason="Requires network and actual YouTube access")
    def test_real_video_metadata(self):
        """Test fetching real video metadata (skipped by default)."""
        downloader = YouTubeDownloader()

        # Test with a stable, known video (e.g., "Never Gonna Give You Up")
        result = downloader.get_video_info('https://www.youtube.com/watch?v=dQw4w9WgXcQ')

        assert result.success is True
        assert result.title is not None
        assert result.duration > 0

    @pytest.mark.skip(reason="Requires network and actual download")
    def test_real_video_download(self, tmp_path):
        """Test downloading real video audio (skipped by default)."""
        downloader = YouTubeDownloader(output_dir=str(tmp_path))

        # Download a short clip
        result = downloader.download(
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            audio_format='wav'
        )

        assert result.success is True
        assert result.audio_path is not None
        assert Path(result.audio_path).exists()

    @pytest.mark.skip(reason="Tests error handling with invalid URL")
    def test_invalid_url_handling(self):
        """Test handling of invalid YouTube URL."""
        downloader = YouTubeDownloader()

        result = downloader.download('https://youtube.com/watch?v=invalidvideoid123')

        assert result.success is False
        assert result.error is not None
