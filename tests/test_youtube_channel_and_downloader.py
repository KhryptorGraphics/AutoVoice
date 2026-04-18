"""Targeted tests for youtube.channel_scraper and youtube.downloader."""

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from auto_voice.youtube.channel_scraper import (
    VideoMetadata,
    YouTubeChannelScraper,
    scrape_artist_channel,
)
from auto_voice.youtube.downloader import (
    DownloadResult,
    YouTubeDownloader,
    download_artist_videos,
    download_artist_videos_async,
)


class TestChannelScraper:
    def test_video_metadata_training_heuristics(self):
        solo_music = VideoMetadata(
            video_id="abc123",
            title="Original Song (Official Audio)",
            duration=240,
            upload_date="20260418",
            channel="Artist",
            channel_id="chan-1",
        )
        collab = VideoMetadata(
            video_id="abc124",
            title="Original Song ft. Guest",
            duration=240,
            upload_date="20260418",
            channel="Artist",
            channel_id="chan-1",
        )
        long_non_music = VideoMetadata(
            video_id="abc125",
            title="Studio interview",
            duration=1200,
            upload_date="20260418",
            channel="Artist",
            channel_id="chan-1",
        )

        assert solo_music.is_music is True
        assert solo_music.is_solo_artist is True
        assert solo_music.is_valid_for_training is True
        assert collab.is_solo_artist is False
        assert collab.is_valid_for_training is False
        assert long_non_music.is_music is False

    def test_get_channel_videos_filters_non_music_collabs_and_invalid_json(self, tmp_path):
        scraper = YouTubeChannelScraper(output_dir=tmp_path)
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "id": "keep-1",
                        "title": "Great Cover",
                        "duration": 180,
                        "upload_date": "20260418",
                        "channel": "Artist",
                        "channel_id": "chan-1",
                        "view_count": 100,
                        "description": "desc",
                        "tags": ["tag"],
                    }
                ),
                json.dumps(
                    {
                        "id": "drop-1",
                        "title": "Behind The Scenes vlog",
                        "duration": 500,
                        "upload_date": "20260418",
                        "channel": "Artist",
                        "channel_id": "chan-1",
                    }
                ),
                json.dumps(
                    {
                        "id": "drop-2",
                        "title": "Big Song feat. Guest",
                        "duration": 220,
                        "upload_date": "20260418",
                        "channel": "Artist",
                        "channel_id": "chan-1",
                    }
                ),
                "{not-json}",
            ]
        )

        completed = subprocess.CompletedProcess(
            args=["yt-dlp"],
            returncode=0,
            stdout=stdout,
            stderr="",
        )

        with patch("subprocess.run", return_value=completed) as run_mock:
            videos = scraper.get_channel_videos(
                "https://youtube.com/@artist",
                max_videos=3,
                music_only=True,
                solo_only=True,
            )

        assert len(videos) == 1
        assert videos[0].video_id == "keep-1"
        called_cmd = run_mock.call_args.args[0]
        assert "--playlist-end" in called_cmd

    def test_get_channel_videos_errors_and_timeouts_raise_runtime_error(self, tmp_path):
        scraper = YouTubeChannelScraper(output_dir=tmp_path)

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=["yt-dlp"], returncode=1, stdout="", stderr="bad"),
        ):
            with pytest.raises(RuntimeError, match="Failed to scrape channel"):
                scraper.get_channel_videos("https://youtube.com/@artist")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["yt-dlp"], timeout=300)):
            with pytest.raises(RuntimeError, match="timed out"):
                scraper.get_channel_videos("https://youtube.com/@artist")

    def test_save_load_and_scrape_artist_channel_round_trip(self, tmp_path):
        scraper = YouTubeChannelScraper(output_dir=tmp_path)
        videos = [
            VideoMetadata(
                video_id="keep-1",
                title="Great Cover",
                duration=180,
                upload_date="20260418",
                channel="Artist",
                channel_id="chan-1",
                view_count=42,
            )
        ]

        path = scraper.save_metadata(videos, "artist")
        loaded = scraper.load_metadata("artist")

        assert path.exists()
        assert len(loaded) == 1
        assert loaded[0].video_id == "keep-1"
        assert loaded[0].channel == "Artist"

        with pytest.raises(ValueError, match="Unknown artist"):
            scrape_artist_channel("unknown")

        with patch.object(YouTubeChannelScraper, "get_channel_videos", return_value=videos) as get_mock, patch.object(
            YouTubeChannelScraper, "save_metadata"
        ) as save_mock:
            result = scrape_artist_channel("conor_maynard", max_videos=2, solo_only=False)

        assert result == videos
        assert get_mock.call_args.args[0] == YouTubeChannelScraper.KNOWN_CHANNELS["conor_maynard"]
        assert get_mock.call_args.kwargs["solo_only"] is False
        save_mock.assert_called_once()


class TestYouTubeDownloader:
    def test_download_audio_returns_existing_file_without_subprocess(self, tmp_path):
        downloader = YouTubeDownloader(output_dir=tmp_path)
        existing = tmp_path / "video123.wav"
        existing.write_bytes(b"audio")

        with patch("subprocess.run") as run_mock:
            result = downloader.download_audio("video123", "Existing Track")

        assert result.success is True
        assert result.output_path == existing
        run_mock.assert_not_called()

    def test_download_audio_success_failure_timeout_and_exception(self, tmp_path):
        downloader = YouTubeDownloader(output_dir=tmp_path)

        def create_file_and_return(*args, **kwargs):
            (tmp_path / "video-success.wav").write_bytes(b"audio")
            return subprocess.CompletedProcess(args=["yt-dlp"], returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=create_file_and_return):
            success = downloader.download_audio("video-success", "Successful Track")

        assert success.success is True
        assert success.output_path == tmp_path / "video-success.wav"

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=["yt-dlp"], returncode=1, stdout="", stderr="download failed"),
        ):
            failed = downloader.download_audio("video-fail", "Failed Track")

        assert failed.success is False
        assert "download failed" in failed.error

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["yt-dlp"], timeout=600)):
            timed_out = downloader.download_audio("video-timeout", "Timeout Track")

        assert timed_out.success is False
        assert "timed out" in timed_out.error

        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            errored = downloader.download_audio("video-error", "Error Track")

        assert errored.success is False
        assert errored.error == "boom"

    def test_download_audio_reports_missing_output_file(self, tmp_path):
        downloader = YouTubeDownloader(output_dir=tmp_path)

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=["yt-dlp"], returncode=0, stdout="", stderr=""),
        ):
            result = downloader.download_audio("video-missing", "Missing Output")

        assert result.success is False
        assert "Output file not found" in result.error

    def test_download_batch_handles_callback_and_future_exceptions(self, tmp_path):
        downloader = YouTubeDownloader(output_dir=tmp_path, max_workers=2)
        videos = [
            VideoMetadata("ok-1", "Song One", 180, "20260418", "Artist", "chan-1"),
            VideoMetadata("bad-1", "Song Two", 180, "20260418", "Artist", "chan-1"),
        ]
        callback = Mock()

        def fake_download(video_id, title=""):
            if video_id == "bad-1":
                raise RuntimeError("future boom")
            return DownloadResult(video_id=video_id, title=title, success=True, output_path=tmp_path / f"{video_id}.wav")

        with patch.object(downloader, "download_audio", side_effect=fake_download):
            results = downloader.download_batch(videos, progress_callback=callback)

        assert len(results) == 2
        assert sum(1 for result in results if result.success) == 1
        failed = next(result for result in results if not result.success)
        assert failed.error == "future boom"
        assert callback.call_count == 2

    def test_download_artist_videos_and_async_wrapper(self, tmp_path):
        videos = [VideoMetadata("ok-1", "Song One", 180, "20260418", "Artist", "chan-1")]
        results = [DownloadResult(video_id="ok-1", title="Song One", success=True, output_path=tmp_path / "ok-1.wav")]

        with patch("auto_voice.youtube.channel_scraper.scrape_artist_channel", return_value=[]) as scrape_mock:
            empty = download_artist_videos("conor_maynard")

        assert empty == []
        scrape_mock.assert_called_once()

        with patch("auto_voice.youtube.channel_scraper.scrape_artist_channel", return_value=videos), patch.object(
            YouTubeDownloader, "download_batch", return_value=results
        ) as batch_mock:
            downloaded = download_artist_videos("conor_maynard", output_subdir="custom", max_videos=12, max_workers=3)

        assert downloaded == results
        batch_mock.assert_called_once_with(videos)

        with patch("auto_voice.youtube.downloader.download_artist_videos", return_value=results) as sync_mock:
            async_results = asyncio.run(download_artist_videos_async("conor_maynard", "custom", 12, 3))

        assert async_results == results
        sync_mock.assert_called_once_with("conor_maynard", "custom", 12, 3)
