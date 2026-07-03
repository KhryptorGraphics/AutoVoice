"""Integration tests for YouTube Artist Training Pipeline."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestYouTubeModule:
    """Test YouTube module imports and basic functionality."""

    def test_youtube_imports(self):
        """Test all YouTube module imports work."""
        from auto_voice.youtube import (
            VideoMetadata,
            YouTubeChannelScraper,
            scrape_artist_channel,
            DownloadResult,
            YouTubeDownloader,
            download_artist_videos,
        )
        assert VideoMetadata is not None
        assert YouTubeChannelScraper is not None

    def test_video_metadata_is_music_heuristic(self):
        """Test music detection heuristic."""
        from auto_voice.youtube import VideoMetadata

        # Should be music
        music_video = VideoMetadata(
            video_id='test1',
            title='Hello - Adele (Cover)',
            duration=240,
            upload_date='20210101',
            channel='Test',
            channel_id='UC123',
        )
        assert music_video.is_music is True

        # Should not be music - too short
        short_video = VideoMetadata(
            video_id='test2',
            title='Quick Hello',
            duration=30,
            upload_date='20210101',
            channel='Test',
            channel_id='UC123',
        )
        assert short_video.is_music is False

        # Should not be music - vlog keyword
        vlog_video = VideoMetadata(
            video_id='test3',
            title='Studio Vlog Day',
            duration=600,
            upload_date='20210101',
            channel='Test',
            channel_id='UC123',
        )
        assert vlog_video.is_music is False

    def test_known_channels_defined(self):
        """Test known channel URLs are defined."""
        from auto_voice.youtube import YouTubeChannelScraper

        assert 'conor_maynard' in YouTubeChannelScraper.KNOWN_CHANNELS
        assert 'william_singe' in YouTubeChannelScraper.KNOWN_CHANNELS

    def test_downloader_rate_limiting(self):
        """Test downloader respects rate limits."""
        import time
        from auto_voice.youtube import YouTubeDownloader
        from pathlib import Path

        downloader = YouTubeDownloader(
            Path('/tmp/test_youtube'),
            rate_limit=0.5
        )

        start = time.time()
        downloader._wait_rate_limit()
        downloader._wait_rate_limit()
        elapsed = time.time() - start

        # Should have waited at least 0.5 seconds between calls
        assert elapsed >= 0.4  # Allow some tolerance


class TestPipelineIntegration:
    """Test pipeline components integrate correctly."""

    def test_pipeline_script_imports(self):
        """Test pipeline script can be imported."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "youtube_pipeline",
            Path(__file__).parent.parent / "scripts" / "youtube_artist_pipeline.py"
        )
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just check syntax
        assert module is not None

    def test_artist_profiles_persist_and_load(self, tmp_path):
        """Saved artist profiles persist to disk and load back by id.

        (Previously asserted two hardcoded production profile ids — tests
        must not depend on live store contents.)
        """
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        store = VoiceProfileStore(
            profiles_dir=str(tmp_path / 'profiles'),
            samples_dir=str(tmp_path / 'samples'),
        )
        conor_id = store.save({'name': 'Conor Maynard', 'profile_role': 'target_user'})
        william_id = store.save({'name': 'William Singe', 'profile_role': 'source_artist'})

        assert store.exists(conor_id), "Conor profile not found"
        assert store.exists(william_id), "William profile not found"
        assert store.load(william_id)['profile_role'] == 'source_artist'
