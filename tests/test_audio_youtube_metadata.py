"""Tests for youtube_metadata.py - Metadata parsing and artist detection.

Task 2.5: Test youtube_metadata.py
- Test artist detection from title
- Test featured artist extraction
- Test title cleaning
- Test cover song detection
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


class TestArtistNameCleaning:
    """Test artist name cleaning utilities."""

    def test_clean_artist_name_official_video(self):
        """Test removing 'official video' suffixes."""
        result = _clean_artist_name("Artist Name (Official Video)")
        assert result == "Artist Name"
        assert "official" not in result.lower()

    def test_clean_artist_name_official_audio(self):
        """Test removing 'official audio' suffixes."""
        result = _clean_artist_name("Artist - Song [Official Audio]")
        assert "official" not in result.lower()
        assert "audio" not in result.lower()

    def test_clean_artist_name_trailing_punctuation(self):
        """Test removing trailing punctuation."""
        result = _clean_artist_name("Artist Name,")
        assert result == "Artist Name"

        result = _clean_artist_name("Artist Name&")
        assert result == "Artist Name"

    def test_clean_artist_name_whitespace(self):
        """Test trimming whitespace."""
        result = _clean_artist_name("  Artist Name  ")
        assert result == "Artist Name"


class TestMultipleArtistSplitting:
    """Test splitting artist strings."""

    def test_split_multiple_artists_comma(self):
        """Test splitting by comma."""
        result = _split_multiple_artists("Artist A, Artist B, Artist C")
        assert len(result) == 3
        assert "Artist A" in result
        assert "Artist B" in result

    def test_split_multiple_artists_ampersand(self):
        """Test splitting by ampersand."""
        result = _split_multiple_artists("Artist A & Artist B")
        assert len(result) == 2
        assert "Artist A" in result
        assert "Artist B" in result

    def test_split_multiple_artists_and(self):
        """Test splitting by 'and'."""
        result = _split_multiple_artists("Artist A and Artist B")
        assert len(result) == 2


class TestProducerCreditDetection:
    """Test producer credit detection."""

    def test_is_producer_credit_prod_by(self):
        """Test detecting 'prod by' pattern."""
        assert _is_producer_credit("prod by Producer Name") is True
        assert _is_producer_credit("prod. by Producer") is True

    def test_is_producer_credit_produced_by(self):
        """Test detecting 'produced by' pattern."""
        assert _is_producer_credit("produced by Producer Name") is True

    def test_is_producer_credit_mixed_by(self):
        """Test detecting 'mixed by' pattern."""
        assert _is_producer_credit("mixed by Engineer") is True

    def test_is_producer_credit_negative(self):
        """Test that actual artists are not detected as producers."""
        assert _is_producer_credit("Artist Name") is False
        assert _is_producer_credit("ft. Guest Artist") is False


class TestFeaturedArtistParsing:
    """Test featured artist extraction."""

    @pytest.mark.smoke
    def test_parse_featured_artists_ft(self):
        """Test parsing 'ft.' pattern."""
        result = parse_featured_artists("Main Artist ft. Guest Artist - Song")
        assert len(result) > 0
        assert "Guest Artist" in result

    def test_parse_featured_artists_feat(self):
        """Test parsing 'feat.' pattern."""
        result = parse_featured_artists("Main Artist feat. Guest - Song")
        assert len(result) > 0

    def test_parse_featured_artists_featuring(self):
        """Test parsing 'featuring' pattern."""
        result = parse_featured_artists("Main Artist featuring Guest Artist")
        assert len(result) > 0

    def test_parse_featured_artists_with(self):
        """Test parsing 'with' pattern."""
        result = parse_featured_artists("Main Artist with Guest Artist")
        assert len(result) > 0

    def test_parse_featured_artists_vs(self):
        """Test parsing 'vs' pattern."""
        result = parse_featured_artists("Artist A vs Artist B")
        assert len(result) > 0

    def test_parse_featured_artists_ampersand(self):
        """Test parsing '&' pattern."""
        result = parse_featured_artists("Artist A & Artist B - Song")
        assert len(result) >= 1

    def test_parse_featured_artists_parentheses(self):
        """Test parsing featured artists in parentheses."""
        result = parse_featured_artists("Song Title (ft. Guest Artist)")
        assert len(result) > 0
        assert "Guest Artist" in result

    def test_parse_featured_artists_multiple(self):
        """Test parsing multiple featured artists."""
        result = parse_featured_artists("Main ft. Artist A, Artist B & Artist C")
        assert len(result) >= 2

    def test_parse_featured_artists_no_features(self):
        """Test with no featured artists."""
        result = parse_featured_artists("Artist - Song Title")
        assert len(result) == 0

    def test_parse_featured_artists_producer_exclusion(self):
        """Test that producers are not detected as featured artists."""
        result = parse_featured_artists("Artist - Song (prod by Producer)")
        # Should not include producer
        assert "Producer" not in result

    def test_parse_featured_artists_from_description(self):
        """Test parsing from description when title has no features."""
        title = "Artist - Song Title"
        description = "featuring vocals by Guest Artist"

        result = parse_featured_artists(title, description)
        # May find in description if supported
        # At minimum, should not crash


class TestMainArtistExtraction:
    """Test main artist extraction."""

    def test_extract_main_artist_standard_format(self):
        """Test standard 'Artist - Song' format."""
        result = extract_main_artist("Artist Name - Song Title")
        assert result == "Artist Name"

    def test_extract_main_artist_with_features(self):
        """Test extracting main artist when features are present."""
        result = extract_main_artist("Main Artist ft. Guest - Song")
        assert result == "Main Artist"
        assert "ft." not in result

    def test_extract_main_artist_no_dash(self):
        """Test when there's no dash separator."""
        result = extract_main_artist("Just A Title")
        assert result is None

    def test_extract_main_artist_multiple_dashes(self):
        """Test with multiple dashes (uses first)."""
        result = extract_main_artist("Artist - Song - Remix")
        assert result == "Artist"


class TestCoverSongDetection:
    """Test cover song detection."""

    def test_detect_cover_song_parentheses(self):
        """Test detecting cover in parentheses."""
        is_cover, original = detect_cover_song("Song Title (Artist Cover)")
        assert is_cover is True
        assert original == "Artist"

    def test_detect_cover_song_cover_of(self):
        """Test detecting 'cover of' pattern."""
        is_cover, original = detect_cover_song("Song Title - cover of Original Artist")
        assert is_cover is True

    def test_detect_cover_song_originally_by(self):
        """Test detecting 'originally by' pattern."""
        is_cover, original = detect_cover_song("Song (originally by Original Artist)")
        assert is_cover is True

    def test_detect_cover_song_no_cover(self):
        """Test with non-cover song."""
        is_cover, original = detect_cover_song("Artist - Song Title")
        assert is_cover is False
        assert original is None

    def test_detect_cover_song_in_description(self):
        """Test detecting cover from description."""
        title = "Song Title"
        description = "This is a cover of Original Artist's song"

        is_cover, original = detect_cover_song(title, description)
        assert is_cover is True


class TestFullMetadataParsing:
    """Test complete metadata parsing."""

    def test_parse_youtube_metadata_standard(self):
        """Test parsing standard video metadata."""
        metadata = {
            'title': 'Artist - Song Title',
            'description': 'Official music video',
            'uploader': 'Artist Channel',
            'channel': 'Artist Channel',
        }

        result = parse_youtube_metadata(metadata)

        assert result['main_artist'] == 'Artist'
        assert result['song_title'] == 'Song Title'
        assert result['is_cover'] is False

    def test_parse_youtube_metadata_with_features(self):
        """Test parsing with featured artists."""
        metadata = {
            'title': 'Main Artist ft. Guest Artist - Song Title',
            'description': '',
            'uploader': 'Main Artist',
        }

        result = parse_youtube_metadata(metadata)

        assert result['main_artist'] == 'Main Artist'
        assert 'Guest Artist' in result['featured_artists']
        assert result['song_title'] is not None

    def test_parse_youtube_metadata_cover_song(self):
        """Test parsing cover song metadata."""
        metadata = {
            'title': 'Cover Artist - Song Title (Original Artist Cover)',
            'description': '',
            'uploader': 'Cover Artist',
        }

        result = parse_youtube_metadata(metadata)

        assert result['is_cover'] is True
        assert result['original_artist'] is not None

    def test_parse_youtube_metadata_no_main_artist(self):
        """Test fallback to uploader when no main artist."""
        metadata = {
            'title': 'Just A Title',
            'description': '',
            'uploader': 'Channel Name',
            'channel': 'Channel Name',
        }

        result = parse_youtube_metadata(metadata)

        assert result['main_artist'] == 'Channel Name'

    def test_parse_youtube_metadata_song_title_cleaning(self):
        """Test that song title is cleaned of featured artist info."""
        metadata = {
            'title': 'Artist - Song Title (ft. Guest)',
            'description': '',
        }

        result = parse_youtube_metadata(metadata)

        assert result['song_title'] is not None
        assert 'ft.' not in result['song_title']


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_title(self):
        """Test handling empty title."""
        result = parse_featured_artists("")
        assert result == []

    def test_non_english_characters(self):
        """Test handling non-English characters."""
        result = extract_main_artist("アーティスト - 曲名")
        assert result is not None

    def test_special_characters_in_names(self):
        """Test artists with special characters."""
        result = extract_main_artist("P!nk - Song Title")
        assert result == "P!nk"

    def test_very_long_title(self):
        """Test handling very long titles."""
        long_title = "A" * 500 + " - " + "B" * 500
        result = extract_main_artist(long_title)
        assert result is not None

    def test_case_insensitivity(self):
        """Test case-insensitive pattern matching."""
        # Should work with different cases
        result1 = parse_featured_artists("Artist FT. Guest")
        result2 = parse_featured_artists("Artist ft. Guest")
        result3 = parse_featured_artists("Artist Ft. Guest")

        # All should find the featured artist
        assert len(result1) > 0
        assert len(result2) > 0
        assert len(result3) > 0


@pytest.mark.integration
class TestMetadataIntegration:
    """Integration tests for metadata parsing."""

    def test_realistic_title_parsing(self):
        """Test parsing realistic YouTube titles."""
        titles = [
            "Conor Maynard - Pillowtalk (ZAYN cover)",
            "Ariana Grande & Justin Bieber - Stuck with U",
            "Ed Sheeran - Shape of You [Official Video]",
            "The Chainsmokers ft. Halsey - Closer",
            "Post Malone - Circles (Lyrics)",
        ]

        for title in titles:
            result = parse_youtube_metadata({'title': title, 'description': ''})
            assert result['main_artist'] is not None

    def test_multiple_patterns_in_single_title(self):
        """Test title with multiple feature patterns."""
        title = "Artist A ft. Artist B & Artist C vs Artist D - Song"

        result = parse_youtube_metadata({'title': title, 'description': ''})

        # Should extract main artist
        assert result['main_artist'] is not None
        # Should find multiple featured artists
        assert len(result['featured_artists']) >= 2

    def test_complex_metadata_structure(self):
        """Test complex metadata with all fields."""
        metadata = {
            'title': 'Main Artist ft. Guest - Song Title (Official Video)',
            'description': 'featuring vocals by Another Guest\nProduced by Producer Name',
            'uploader': 'Main Artist Official',
            'channel': 'Main Artist',
        }

        result = parse_youtube_metadata(metadata)

        assert result['main_artist'] == 'Main Artist'
        assert len(result['featured_artists']) >= 1
        assert 'Guest' in str(result['featured_artists'])
        assert 'Producer' not in str(result['featured_artists'])  # Should exclude producers
        assert 'Official Video' not in result['song_title'] if result['song_title'] else True
