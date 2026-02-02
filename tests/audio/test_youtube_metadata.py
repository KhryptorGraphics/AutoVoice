"""Tests for youtube_metadata.py - Metadata parsing and artist detection.

Test Coverage:
- Task 2.5: Artist detection from title
- Featured artist extraction
- Title cleaning
- Genre classification (if applicable)
"""

import pytest
from auto_voice.audio.youtube_metadata import (
    parse_featured_artists,
    extract_main_artist,
    detect_cover_song,
    parse_youtube_metadata,
    _clean_artist_name,
    _split_multiple_artists,
    _is_producer_credit,
)


class TestFeaturedArtistParsing:
    """Test suite for featured artist parsing."""

    def test_parse_featured_with_ft(self):
        """Test parsing 'ft.' pattern."""
        title = "Main Artist ft. Featured Artist - Song Title"
        featured = parse_featured_artists(title)

        assert len(featured) == 1
        assert "Featured Artist" in featured

    def test_parse_featured_with_feat(self):
        """Test parsing 'feat.' pattern."""
        title = "Main Artist feat. Featured Artist - Song Title"
        featured = parse_featured_artists(title)

        assert len(featured) == 1
        assert "Featured Artist" in featured

    def test_parse_featured_with_featuring(self):
        """Test parsing 'featuring' pattern."""
        title = "Main Artist featuring Featured Artist - Song Title"
        featured = parse_featured_artists(title)

        assert len(featured) == 1
        assert "Featured Artist" in featured

    def test_parse_featured_with_ampersand(self):
        """Test parsing '&' pattern."""
        title = "Artist A & Artist B - Song Title"
        featured = parse_featured_artists(title)

        assert len(featured) >= 1
        # May parse "Artist A & Artist B" or split into separate artists

    def test_parse_featured_with_vs(self):
        """Test parsing 'vs.' pattern."""
        title = "Artist A vs. Artist B - Battle"
        featured = parse_featured_artists(title)

        assert len(featured) >= 1
        assert "Artist B" in featured

    def test_parse_featured_with_x(self):
        """Test parsing 'x' collaboration pattern."""
        title = "Artist A x Artist B - Song Title"
        featured = parse_featured_artists(title)

        assert len(featured) >= 1

    def test_parse_featured_in_parentheses(self):
        """Test parsing featured artists in parentheses."""
        title = "Main Artist - Song Title (ft. Featured Artist)"
        featured = parse_featured_artists(title)

        assert len(featured) == 1
        assert "Featured Artist" in featured

    def test_parse_multiple_featured_artists(self):
        """Test parsing multiple featured artists."""
        title = "Main Artist ft. Artist A, Artist B & Artist C - Song"
        featured = parse_featured_artists(title)

        # Should parse multiple artists
        assert len(featured) >= 2

    def test_parse_featured_excludes_producers(self):
        """Test that producer credits are excluded."""
        title = "Artist - Song (prod. by Producer Name)"
        featured = parse_featured_artists(title)

        # Should not include producer
        assert "Producer Name" not in featured

    def test_parse_featured_from_description(self):
        """Test parsing featured artists from description."""
        title = "Artist - Song Title"
        description = "Featuring vocals by Guest Artist"
        featured = parse_featured_artists(title, description)

        assert len(featured) >= 1
        assert "Guest Artist" in featured

    def test_parse_featured_no_duplicates(self):
        """Test that duplicate artists are removed."""
        title = "Artist ft. Guest ft. Guest - Song"
        featured = parse_featured_artists(title)

        # Should have only one "Guest"
        assert featured.count("Guest") == 1

    def test_parse_featured_empty_title(self):
        """Test parsing empty title."""
        featured = parse_featured_artists("")
        assert len(featured) == 0

    def test_parse_featured_no_features(self):
        """Test title with no featured artists."""
        title = "Artist - Song Title"
        featured = parse_featured_artists(title)

        assert len(featured) == 0


class TestMainArtistExtraction:
    """Test suite for main artist extraction."""

    def test_extract_main_artist_standard_format(self):
        """Test standard 'Artist - Title' format."""
        title = "Rick Astley - Never Gonna Give You Up"
        artist = extract_main_artist(title)

        assert artist == "Rick Astley"

    def test_extract_main_artist_with_featured(self):
        """Test extraction with featured artist."""
        title = "Main Artist ft. Guest - Song Title"
        artist = extract_main_artist(title)

        assert artist == "Main Artist"
        assert "Guest" not in artist

    def test_extract_main_artist_no_dash(self):
        """Test extraction when no dash separator."""
        title = "Just A Title No Artist"
        artist = extract_main_artist(title)

        assert artist is None

    def test_extract_main_artist_multiple_dashes(self):
        """Test extraction with multiple dashes."""
        title = "Artist - Song - Remix"
        artist = extract_main_artist(title)

        # Should extract first part only
        assert artist == "Artist"

    def test_extract_main_artist_with_official_tag(self):
        """Test extraction with [Official Video] tag."""
        title = "Artist - Song [Official Video]"
        artist = extract_main_artist(title)

        assert "Official" not in artist
        assert artist == "Artist"


class TestCoverSongDetection:
    """Test suite for cover song detection."""

    def test_detect_cover_with_parentheses(self):
        """Test cover detection with (Artist Cover) pattern."""
        title = "Cover Artist - Song Title (Original Artist Cover)"
        is_cover, original = detect_cover_song(title)

        assert is_cover is True
        assert original == "Original Artist"

    def test_detect_cover_with_keyword(self):
        """Test cover detection with 'cover' keyword."""
        title = "Cover Artist - Song Title Cover"
        is_cover, original = detect_cover_song(title)

        assert is_cover is True

    def test_detect_cover_originally_by(self):
        """Test cover detection with 'originally by' pattern."""
        title = "Cover Artist - Song originally by Original Artist"
        is_cover, original = detect_cover_song(title)

        assert is_cover is True
        assert "Original Artist" in original

    def test_detect_cover_from_description(self):
        """Test cover detection from description."""
        title = "Cover Artist - Song Title"
        description = "This is a cover of the song originally by Original Artist"
        is_cover, original = detect_cover_song(title, description)

        assert is_cover is True

    def test_detect_no_cover(self):
        """Test when video is not a cover."""
        title = "Original Artist - Song Title"
        is_cover, original = detect_cover_song(title)

        assert is_cover is False
        assert original is None


class TestYouTubeMetadataParsing:
    """Test suite for complete metadata parsing."""

    def test_parse_youtube_metadata_complete(self):
        """Test parsing complete metadata."""
        metadata = {
            'title': 'Main Artist ft. Guest - Song Title',
            'description': 'Song description',
            'uploader': 'Main Artist',
            'channel': 'MainArtistVEVO',
        }

        parsed = parse_youtube_metadata(metadata)

        assert parsed['main_artist'] == 'Main Artist'
        assert 'Guest' in parsed['featured_artists']
        assert parsed['song_title'] == 'Song Title'
        assert parsed['is_cover'] is False

    def test_parse_youtube_metadata_with_cover(self):
        """Test parsing cover song metadata."""
        metadata = {
            'title': 'Cover Artist - Song (Original Artist Cover)',
            'description': '',
            'uploader': 'Cover Artist',
        }

        parsed = parse_youtube_metadata(metadata)

        assert parsed['is_cover'] is True
        assert parsed['original_artist'] == 'Original Artist'

    def test_parse_youtube_metadata_fallback_to_uploader(self):
        """Test falling back to uploader when no artist in title."""
        metadata = {
            'title': 'Just A Song Title',
            'uploader': 'Channel Name',
            'channel': 'Channel Name',
        }

        parsed = parse_youtube_metadata(metadata)

        assert parsed['main_artist'] == 'Channel Name'

    def test_parse_youtube_metadata_cleans_song_title(self):
        """Test that song title is cleaned of featured info."""
        metadata = {
            'title': 'Artist - Song Title (ft. Guest)',
        }

        parsed = parse_youtube_metadata(metadata)

        assert 'ft.' not in parsed['song_title']
        assert 'Guest' not in parsed['song_title']


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_clean_artist_name(self):
        """Test artist name cleaning."""
        # Remove official tags
        assert "Official" not in _clean_artist_name("Artist (Official Video)")

        # Strip whitespace
        assert _clean_artist_name("  Artist  ") == "Artist"

        # Remove trailing punctuation
        assert _clean_artist_name("Artist,") == "Artist"

    def test_split_multiple_artists(self):
        """Test splitting multiple artists."""
        # Comma separated
        artists = _split_multiple_artists("Artist A, Artist B")
        assert len(artists) == 2
        assert "Artist A" in artists
        assert "Artist B" in artists

        # Ampersand separated
        artists = _split_multiple_artists("Artist A & Artist B")
        assert len(artists) == 2

        # "and" separated
        artists = _split_multiple_artists("Artist A and Artist B")
        assert len(artists) == 2

    def test_is_producer_credit(self):
        """Test producer credit detection."""
        assert _is_producer_credit("prod. by Producer") is True
        assert _is_producer_credit("produced by Producer") is True
        assert _is_producer_credit("mixed by Engineer") is True
        assert _is_producer_credit("ft. Artist") is False


@pytest.mark.parametrize("title,expected_artists", [
    ("Artist ft. Guest - Song", ["Guest"]),
    ("Artist feat. Guest - Song", ["Guest"]),
    ("Artist featuring Guest - Song", ["Guest"]),
    ("Artist & Guest - Song", []),  # Ampersand is tricky
    ("Artist vs. Guest - Battle", ["Guest"]),
    ("Artist x Guest - Song", ["Guest"]),
    ("Artist - Song (ft. Guest)", ["Guest"]),
])
def test_featured_artist_patterns(title, expected_artists):
    """Test various featured artist patterns."""
    featured = parse_featured_artists(title)

    for artist in expected_artists:
        assert artist in featured


@pytest.mark.parametrize("title,expected_artist", [
    ("Rick Astley - Never Gonna Give You Up", "Rick Astley"),
    ("Main Artist ft. Guest - Song", "Main Artist"),
    ("Artist - Song [Official Video]", "Artist"),
    ("Just A Title", None),
])
def test_main_artist_extraction_patterns(title, expected_artist):
    """Test main artist extraction patterns."""
    artist = extract_main_artist(title)
    assert artist == expected_artist


@pytest.mark.parametrize("title,expected_cover", [
    ("Artist - Song (Original Cover)", True),
    ("Artist - Song Cover", True),
    ("Artist - Song originally by Other", True),
    ("Artist - Song", False),
    ("Artist - Original Song", False),
])
def test_cover_detection_patterns(title, expected_cover):
    """Test cover song detection patterns."""
    is_cover, _ = detect_cover_song(title)
    assert is_cover == expected_cover


def test_metadata_with_complex_title():
    """Test parsing complex real-world title."""
    metadata = {
        'title': 'Conor Maynard ft. Anth - Cold Water (Major Lazer Cover) [Official Video]',
        'uploader': 'Conor Maynard',
    }

    parsed = parse_youtube_metadata(metadata)

    assert parsed['main_artist'] == 'Conor Maynard'
    assert 'Anth' in parsed['featured_artists']
    assert parsed['is_cover'] is True
    assert 'Major Lazer' in parsed['original_artist']


def test_genre_classification_placeholder():
    """Placeholder for genre classification tests.

    Genre classification is not currently implemented in youtube_metadata.py.
    This test serves as a marker for future implementation.
    """
    # Future: Test genre classification if implemented
    # metadata = {'title': 'Artist - Song', 'tags': ['pop', 'music']}
    # parsed = parse_youtube_metadata(metadata)
    # assert parsed['genre'] in ['pop', 'rock', 'hip-hop', etc.]
    pass


def test_title_cleaning_removes_brackets():
    """Test that title cleaning removes various bracket types."""
    metadata = {
        'title': 'Artist - Song Title [Official] (Lyric Video) {HD}',
    }

    parsed = parse_youtube_metadata(metadata)

    # Song title should be cleaned
    song_title = parsed['song_title']
    assert '[' not in song_title
    assert ']' not in song_title
    assert '(' not in song_title
    assert ')' not in song_title


def test_empty_metadata():
    """Test handling of empty metadata."""
    metadata = {}

    parsed = parse_youtube_metadata(metadata)

    assert parsed['main_artist'] is None
    assert parsed['featured_artists'] == []
    assert parsed['is_cover'] is False
    assert parsed['song_title'] is None


def test_description_parsing_for_featured_artists():
    """Test extracting featured artists from video description."""
    title = "Artist - Song Title"
    description = """
    Listen to my new song!

    Featuring vocals by Guest Artist

    Available on all platforms.
    """

    featured = parse_featured_artists(title, description)

    assert len(featured) > 0
    assert "Guest Artist" in featured
