"""Coverage tests for YouTube downloader — mock yt-dlp and subprocess."""
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

@dataclass
class FakeVideoMetadata:
    video_id: str
    title: str
    channel: str = ""
    duration: float = 0.0
    upload_date: str = ""


class TestYouTubeDownloaderInit:
    def test_init_creates_output_dir(self, tmp_path):
        from auto_voice.youtube.downloader import YouTubeDownloader
        output = tmp_path / "audio"
        dl = YouTubeDownloader(output_dir=output)
        assert output.exists()

    def test_init_defaults(self, tmp_path):
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path)
        assert dl.max_workers == 4
        assert dl.rate_limit == 1.0

    def test_init_custom_workers(self, tmp_path):
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path, max_workers=8, rate_limit=0.5)
        assert dl.max_workers == 8
        assert dl.rate_limit == 0.5


class TestYouTubeDownloaderDownload:
    def test_download_already_exists(self, tmp_path):
        """Returns immediately if file already downloaded."""
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path)
        # Create existing file
        (tmp_path / "abc123.wav").touch()
        result = dl.download_audio("abc123", "Test Video")
        assert result.success is True
        assert result.output_path is not None

    def test_download_success(self, tmp_path):
        """Successful download via subprocess."""
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path, rate_limit=0)
        dl._last_download_time = 0  # Skip rate limit
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Create the output file after "download"
            (tmp_path / "testvid.wav").touch()
            result = dl.download_audio("testvid", "Test")
            assert result.success is True
            assert result.video_id == "testvid"

    def test_download_failure(self, tmp_path):
        """Failed download returns error."""
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path, rate_limit=0)
        dl._last_download_time = 0
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Video not found")
            result = dl.download_audio("badvid", "Bad Video")
            assert result.success is False
            assert result.error is not None
            assert "not found" in result.error.lower() or result.error

    def test_download_timeout(self, tmp_path):
        """Timeout returns error."""
        import subprocess
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path, rate_limit=0)
        dl._last_download_time = 0
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=['yt-dlp'], timeout=600)
            result = dl.download_audio("slowvid", "Slow")
            assert result.success is False
            assert "timed out" in result.error.lower()

    def test_download_exception(self, tmp_path):
        """General exception returns error."""
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path, rate_limit=0)
        dl._last_download_time = 0
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = OSError("Network error")
            result = dl.download_audio("errvid", "Error")
            assert result.success is False

    def test_download_output_not_found(self, tmp_path):
        """Success return code but no output file."""
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path, rate_limit=0)
        dl._last_download_time = 0
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Don't create any output file
            result = dl.download_audio("nofilevid", "No File")
            assert result.success is False
            assert "not found" in result.error.lower()


class TestYouTubeDownloaderRateLimit:
    def test_rate_limit_waits(self, tmp_path):
        """Rate limit enforces delay between downloads."""
        from auto_voice.youtube.downloader import YouTubeDownloader
        dl = YouTubeDownloader(output_dir=tmp_path, rate_limit=0.1)
        with patch('time.sleep') as mock_sleep:
            dl._last_download_time = time.time() - 0.05  # 50ms since last
            dl._wait_rate_limit()
            # Should sleep for the remaining time


class TestYouTubeDownloaderBatch:
    def test_batch_downloads_multiple(self, tmp_path):
        """Batch download processes multiple videos."""
        from auto_voice.youtube.downloader import YouTubeDownloader, DownloadResult
        dl = YouTubeDownloader(output_dir=tmp_path, max_workers=2, rate_limit=0)
        videos = [FakeVideoMetadata(video_id=f"vid{i}", title=f"Video {i}") for i in range(3)]
        with patch.object(dl, 'download_audio') as mock_dl:
            mock_dl.side_effect = [
                DownloadResult(video_id="vid0", title="Video 0", success=True, output_path=tmp_path/"vid0.wav"),
                DownloadResult(video_id="vid1", title="Video 1", success=True, output_path=tmp_path/"vid1.wav"),
                DownloadResult(video_id="vid2", title="Video 2", success=False, error="failed"),
            ]
            results = dl.download_batch(videos)
            assert len(results) == 3

    def test_batch_progress_callback(self, tmp_path):
        """Batch download calls progress callback."""
        from auto_voice.youtube.downloader import YouTubeDownloader, DownloadResult
        dl = YouTubeDownloader(output_dir=tmp_path, max_workers=1, rate_limit=0)
        videos = [FakeVideoMetadata(video_id="v1", title="V1")]
        with patch.object(dl, 'download_audio') as mock_dl:
            mock_dl.return_value = DownloadResult(video_id="v1", title="V1", success=True)
            callbacks = []
            dl.download_batch(videos, progress_callback=lambda c, t, r: callbacks.append((c, t)))
            assert len(callbacks) >= 1


class TestDownloadResult:
    def test_download_result_dataclass(self):
        from auto_voice.youtube.downloader import DownloadResult
        r = DownloadResult(video_id="abc", title="Test", success=True, output_path=Path("/tmp/a.wav"))
        assert r.video_id == "abc"
        assert r.success is True
        assert r.error is None

    def test_download_result_failure(self):
        from auto_voice.youtube.downloader import DownloadResult
        r = DownloadResult(video_id="abc", title="Test", success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"
        assert r.output_path is None


class TestConvenienceFunctions:
    def test_download_artist_videos_no_videos(self, tmp_path):
        """Returns empty list when no videos found."""
        from auto_voice.youtube.downloader import download_artist_videos
        with patch('auto_voice.youtube.channel_scraper.scrape_artist_channel', return_value=[]):
            results = download_artist_videos("unknown_artist", output_subdir=str(tmp_path))
            assert results == []
