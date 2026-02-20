"""Tests for YouTube metadata parsing and featured artist detection."""
import pytest


class TestParseFeaturedArtists:
    """Test cases for parse_featured_artists function."""

    def test_ft_pattern(self):
        """Test detection of 'ft.' pattern."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Conor Maynard - Pillowtalk ft. William Singe"
        artists = parse_featured_artists(title)
        assert "William Singe" in artists

    def test_feat_pattern(self):
        """Test detection of 'feat.' pattern."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Artist - Song (feat. Other Artist)"
        artists = parse_featured_artists(title)
        assert "Other Artist" in artists

    def test_featuring_pattern(self):
        """Test detection of 'featuring' pattern."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Main Artist - Track Name featuring Guest Singer"
        artists = parse_featured_artists(title)
        assert "Guest Singer" in artists

    def test_vs_pattern(self):
        """Test detection of 'vs.' or 'vs' pattern."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Artist A vs. Artist B - Epic Mashup"
        artists = parse_featured_artists(title)
        assert "Artist B" in artists

    def test_with_pattern(self):
        """Test detection of 'with' pattern."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Main Singer - Duet with Another Singer"
        artists = parse_featured_artists(title)
        assert "Another Singer" in artists

    def test_ampersand_pattern(self):
        """Test detection of '&' pattern."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Artist One & Artist Two - Collaboration"
        artists = parse_featured_artists(title)
        assert "Artist Two" in artists

    def test_x_pattern(self):
        """Test detection of 'x' collaboration pattern."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Producer x Singer - Beat Drop"
        artists = parse_featured_artists(title)
        assert "Singer" in artists

    def test_multiple_featured_artists(self):
        """Test detection of multiple featured artists."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Main - Song ft. Artist A, Artist B & Artist C"
        artists = parse_featured_artists(title)
        assert len(artists) >= 2

    def test_no_featured_artists(self):
        """Test solo artist returns empty list."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Solo Artist - My Song (Official Video)"
        artists = parse_featured_artists(title)
        assert artists == []

    def test_description_parsing(self):
        """Test parsing from description when title has no features."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Main Artist - Song Title"
        description = "Featuring vocals by Guest Singer. Produced by Producer X."
        artists = parse_featured_artists(title, description)
        assert "Guest Singer" in artists

    def test_prod_by_excluded(self):
        """Test that 'prod. by' doesn't count as featured artist."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Rapper - Track (prod. Producer Name)"
        artists = parse_featured_artists(title)
        # Producer should not be in featured artists (they're not singing)
        assert "Producer Name" not in artists

    def test_cover_song_original_artist(self):
        """Test detection of original artist in cover songs."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Conor Maynard - Shape of You (Ed Sheeran Cover)"
        artists = parse_featured_artists(title)
        # Original artist shouldn't be a featured artist in a cover
        assert "Ed Sheeran" not in artists or len(artists) == 0

    def test_case_insensitive(self):
        """Test that patterns are case insensitive."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Artist - Song FT. Featured Artist"
        artists = parse_featured_artists(title)
        assert "Featured Artist" in artists

    def test_parentheses_handling(self):
        """Test patterns inside parentheses."""
        from auto_voice.audio.youtube_metadata import parse_featured_artists

        title = "Main Artist - Song Title (ft. Guest)"
        artists = parse_featured_artists(title)
        assert "Guest" in artists


class TestExtractMainArtist:
    """Test cases for extracting main artist from title."""

    def test_extract_main_artist_simple(self):
        """Test extracting main artist from simple title."""
        from auto_voice.audio.youtube_metadata import extract_main_artist

        title = "Conor Maynard - Pillowtalk"
        artist = extract_main_artist(title)
        assert artist == "Conor Maynard"

    def test_extract_main_artist_with_featured(self):
        """Test extracting main artist when features present."""
        from auto_voice.audio.youtube_metadata import extract_main_artist

        title = "Main Artist ft. Guest - Song Name"
        artist = extract_main_artist(title)
        assert artist == "Main Artist"

    def test_extract_main_artist_no_dash(self):
        """Test title without dash separator returns None."""
        from auto_voice.audio.youtube_metadata import extract_main_artist

        title = "Some Random Video Title"
        artist = extract_main_artist(title)
        assert artist is None


class TestParseYouTubeMetadata:
    """Test the combined metadata parsing function."""

    def test_full_metadata_parsing(self):
        """Test complete metadata parsing with all fields."""
        from auto_voice.audio.youtube_metadata import parse_youtube_metadata

        metadata = {
            'title': 'Conor Maynard - Pillowtalk ft. William Singe (Zayn Cover)',
            'description': 'Cover of Pillowtalk by Zayn. Mixed by Producer X.',
            'uploader': 'Conor Maynard',
            'channel': 'Conor Maynard'
        }

        result = parse_youtube_metadata(metadata)

        assert result['main_artist'] == 'Conor Maynard'
        assert 'William Singe' in result['featured_artists']
        assert result['is_cover'] == True
        assert result['original_artist'] == 'Zayn'
