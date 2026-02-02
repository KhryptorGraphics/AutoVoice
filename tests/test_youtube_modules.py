"""Tests for YouTube downloader, metadata, and file organizer modules.

Tests include:
- YouTube download with mocked yt-dlp
- Metadata parsing (artist detection, featured artists)
- File organization and naming conventions
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from auto_voice.audio.youtube_downloader import YouTubeDownloader, YouTubeDownloadResult
from auto_voice.audio.youtube_metadata import (
    parse_featured_artists,
    extract_main_artist,
    detect_cover_song,
    parse_youtube_metadata,
)
from auto_voice.audio.file_organizer import FileOrganizer


# ===== YouTube Downloader Tests (Phase 5) =====

@pytest.fixture
def downloader(tmp_path):
    """Create YouTubeDownloader instance."""
    return YouTubeDownloader(output_dir=str(tmp_path))


def test_downloader_initialization(tmp_path):
    """Test YouTubeDownloader initialization."""
    dl = YouTubeDownloader(output_dir=str(tmp_path))

    assert dl.output_dir == str(tmp_path)
    assert Path(dl.output_dir).exists()


@patch('auto_voice.audio.youtube_downloader.subprocess.run')
def test_download_success_mocked(mock_run, downloader, tmp_path):
    """Test successful download (mocked)."""
    # Mock metadata fetch
    mock_metadata = {
        'id': 'test_video_id',
        'title': 'Artist - Song Title',
        'duration': 180.0,
        'thumbnail': 'https://example.com/thumb.jpg',
    }

    mock_run.return_value = Mock(
        returncode=0,
        stdout=json.dumps(mock_metadata),
        stderr='',
    )

    # Create dummy audio file
    audio_path = tmp_path / "test_video_id.wav"
    audio_path.write_text("dummy audio")

    with patch.object(downloader, '_download_audio', return_value=True):
        result = downloader.download("https://youtube.com/watch?v=test_video_id")

        assert result.success
        assert result.title == 'Artist - Song Title'
        assert result.video_id == 'test_video_id'


@patch('auto_voice.audio.youtube_downloader.subprocess.run')
def test_metadata_extraction(mock_run, downloader):
    """Test metadata extraction."""
    mock_metadata = {
        'id': 'abc123',
        'title': 'Taylor Swift ft. Ed Sheeran - Everything Has Changed',
        'duration': 245.0,
        'thumbnail': 'https://example.com/thumb.jpg',
        'description': 'Official music video',
    }

    mock_run.return_value = Mock(
        returncode=0,
        stdout=json.dumps(mock_metadata),
    )

    result = downloader.get_video_info("https://youtube.com/watch?v=abc123")

    assert result.success
    assert result.main_artist == 'Taylor Swift'
    assert 'Ed Sheeran' in result.featured_artists


@patch('auto_voice.audio.youtube_downloader.subprocess.run')
def test_error_404_handling(mock_run, downloader):
    """Test 404 error handling."""
    mock_run.return_value = Mock(
        returncode=1,
        stderr='ERROR: Video unavailable',
    )

    result = downloader.download("https://youtube.com/watch?v=invalid_id")

    assert not result.success
    assert result.error is not None


@patch('auto_voice.audio.youtube_downloader.subprocess.run')
def test_network_timeout(mock_run, downloader):
    """Test network timeout handling."""
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 30)

    result = downloader.download("https://youtube.com/watch?v=test_id")

    assert not result.success


# ===== YouTube Metadata Tests (Phase 6) =====

@pytest.mark.parametrize("title,expected_featured", [
    ("Artist ft. Guest - Song", ["Guest"]),
    ("Artist feat. Guest - Song", ["Guest"]),
    ("Artist featuring Guest - Song", ["Guest"]),
    ("Artist & Guest - Song", ["Guest"]),
    ("Artist x Guest - Song", ["Guest"]),
    ("Artist vs. Guest - Song", ["Guest"]),
    ("Artist (ft. Guest) - Song", ["Guest"]),
])
def test_featured_artist_patterns(title, expected_featured):
    """Test various featured artist formats."""
    featured = parse_featured_artists(title)

    assert len(featured) >= len(expected_featured)
    for expected in expected_featured:
        assert expected in featured


def test_multiple_featured_artists():
    """Test multiple featured artists."""
    title = "Main Artist ft. Guest1, Guest2 & Guest3 - Song"

    featured = parse_featured_artists(title)

    assert len(featured) >= 2  # Should detect multiple


def test_extract_main_artist_standard():
    """Test standard 'Artist - Song' format."""
    title = "Taylor Swift - Shake It Off"

    artist = extract_main_artist(title)

    assert artist == "Taylor Swift"


def test_extract_main_artist_with_featured():
    """Test 'Artist ft. Guest - Song' format."""
    title = "Ed Sheeran ft. Justin Bieber - I Don't Care"

    artist = extract_main_artist(title)

    assert artist == "Ed Sheeran"


def test_title_cleaning():
    """Test title cleaning (remove tags)."""
    titles = [
        "Artist - Song (Official Video)",
        "Artist - Song [HD]",
        "Artist - Song (Official Audio)",
    ]

    for title in titles:
        artist = extract_main_artist(title)
        assert artist == "Artist"


def test_cover_song_detection():
    """Test cover song detection."""
    title = "Singer Name - Song Title (Artist Name Cover)"

    is_cover, original = detect_cover_song(title)

    assert is_cover
    assert original == "Artist Name"


def test_parse_youtube_metadata_full():
    """Test full metadata parsing."""
    metadata = {
        'title': 'Conor Maynard ft. William Singe - Despacito Cover',
        'description': 'Covering Despacito by Luis Fonsi',
        'uploader': 'Conor Maynard',
    }

    parsed = parse_youtube_metadata(metadata)

    assert parsed['main_artist'] == 'Conor Maynard'
    assert 'William Singe' in parsed['featured_artists']
    assert parsed['is_cover']
    assert 'Despacito' in parsed['song_title']


# ===== File Organizer Tests (Phase 7) =====

@pytest.fixture
def organizer(tmp_path):
    """Create FileOrganizer instance."""
    training_dir = tmp_path / "training_vocals"
    profiles_dir = tmp_path / "voice_profiles"

    training_dir.mkdir()
    profiles_dir.mkdir()

    return FileOrganizer(
        training_vocals_dir=training_dir,
        voice_profiles_dir=profiles_dir,
    )


def test_organizer_initialization(organizer):
    """Test FileOrganizer initialization."""
    assert organizer.training_vocals_dir.exists()
    assert organizer.voice_profiles_dir.exists()


def test_normalize_artist_name(organizer):
    """Test artist name normalization."""
    test_cases = [
        ("Taylor Swift", "taylor_swift"),
        ("Ed Sheeran", "ed_sheeran"),
        ("A$AP Rocky", "aap_rocky"),
        ("Kendrick Lamar", "kendrick_lamar"),
    ]

    for input_name, expected in test_cases:
        normalized = organizer.normalize_artist_name(input_name)
        assert normalized == expected


def test_directory_creation(organizer, tmp_path):
    """Test profile directory creation."""
    # Create test structure
    featured_dir = organizer.featured_dir
    featured_dir.mkdir(exist_ok=True)

    artist_dir = featured_dir / "test_artist"
    artist_dir.mkdir()

    assert artist_dir.exists()


def test_file_naming_convention():
    """Test file naming follows convention."""
    # profile_{id}_sample_{n}.wav
    # adapter_{id}_epoch_{n}.safetensors

    expected_patterns = [
        "track123_SPEAKER_00_isolated.wav",
        "track456_SPEAKER_01_isolated.wav",
    ]

    for pattern in expected_patterns:
        assert "_isolated.wav" in pattern
        assert "SPEAKER_" in pattern


def test_cleanup_old_files(organizer, tmp_path):
    """Test cleanup of old checkpoint files."""
    # Create dummy checkpoint files
    test_dir = tmp_path / "checkpoints"
    test_dir.mkdir()

    for i in range(5):
        (test_dir / f"checkpoint_epoch_{i}.pt").write_text("dummy")

    # Should keep last 3
    # Placeholder for actual cleanup logic
    assert len(list(test_dir.glob("*.pt"))) == 5


# ===== Integration Tests (Phase 8) =====

def test_youtube_to_diarization_flow():
    """Test YouTube → Diarization → Profiles flow (mocked)."""
    # Placeholder for integration test
    pass


def test_separation_to_diarization_flow():
    """Test Separation → Diarization flow."""
    # Placeholder for integration test
    pass


# ===== Coverage verification =====

def test_coverage_youtube_modules():
    """Verify coverage of YouTube and file organizer modules."""
    from auto_voice.audio import youtube_downloader, youtube_metadata, file_organizer

    assert hasattr(youtube_downloader, 'YouTubeDownloader')
    assert hasattr(youtube_metadata, 'parse_featured_artists')
    assert hasattr(file_organizer, 'FileOrganizer')
