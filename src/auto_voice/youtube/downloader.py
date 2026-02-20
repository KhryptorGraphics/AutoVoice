"""YouTube audio downloader with parallel processing.

Downloads audio from YouTube videos with rate limiting and progress tracking.
"""
import asyncio
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable

from .channel_scraper import VideoMetadata

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation."""
    video_id: str
    title: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class YouTubeDownloader:
    """Downloads audio from YouTube videos with parallel processing."""

    def __init__(self, output_dir: Path, max_workers: int = 4,
                 rate_limit: float = 1.0):
        """Initialize downloader.

        Args:
            output_dir: Directory to store downloaded audio
            max_workers: Maximum parallel downloads
            rate_limit: Minimum seconds between download starts (avoid rate limiting)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self._last_download_time = 0.0

    def _wait_rate_limit(self):
        """Wait if needed to respect rate limit."""
        now = time.time()
        elapsed = now - self._last_download_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_download_time = time.time()

    def download_audio(self, video_id: str, title: str = "") -> DownloadResult:
        """Download audio from a single video.

        Args:
            video_id: YouTube video ID
            title: Video title for logging

        Returns:
            DownloadResult with success status and output path
        """
        self._wait_rate_limit()

        output_template = str(self.output_dir / f'{video_id}.%(ext)s')
        url = f'https://www.youtube.com/watch?v={video_id}'

        # Check if already downloaded
        existing = list(self.output_dir.glob(f'{video_id}.*'))
        if existing:
            logger.info(f"Already downloaded: {video_id}")
            return DownloadResult(
                video_id=video_id,
                title=title,
                success=True,
                output_path=existing[0],
            )

        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',  # Convert to WAV for processing
            '--audio-quality', '0',  # Best quality
            '--no-playlist',
            '--no-warnings',
            '-o', output_template,
            url
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout per video
            )

            duration = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"Download failed for {video_id}: {result.stderr}")
                return DownloadResult(
                    video_id=video_id,
                    title=title,
                    success=False,
                    error=result.stderr[:500],
                    duration_seconds=duration,
                )

            # Find the output file
            output_files = list(self.output_dir.glob(f'{video_id}.*'))
            if not output_files:
                return DownloadResult(
                    video_id=video_id,
                    title=title,
                    success=False,
                    error="Output file not found after download",
                    duration_seconds=duration,
                )

            logger.info(f"Downloaded: {title or video_id} ({duration:.1f}s)")
            return DownloadResult(
                video_id=video_id,
                title=title,
                success=True,
                output_path=output_files[0],
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            return DownloadResult(
                video_id=video_id,
                title=title,
                success=False,
                error="Download timed out after 10 minutes",
            )
        except Exception as e:
            return DownloadResult(
                video_id=video_id,
                title=title,
                success=False,
                error=str(e),
            )

    def download_batch(self, videos: List[VideoMetadata],
                       progress_callback: Optional[Callable[[int, int, DownloadResult], None]] = None
                       ) -> List[DownloadResult]:
        """Download audio from multiple videos in parallel.

        Args:
            videos: List of video metadata to download
            progress_callback: Called with (completed, total, result) after each download

        Returns:
            List of download results
        """
        results = []
        total = len(videos)
        completed = 0

        logger.info(f"Starting batch download of {total} videos with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(self.download_audio, v.video_id, v.title): v
                for v in videos
            }

            for future in as_completed(future_to_video):
                video = future_to_video[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = DownloadResult(
                        video_id=video.video_id,
                        title=video.title,
                        success=False,
                        error=str(e),
                    )

                results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, total, result)
                else:
                    status = "OK" if result.success else f"FAIL: {result.error[:50]}"
                    logger.info(f"[{completed}/{total}] {video.title[:50]}: {status}")

        # Summary
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {success_count}/{total} successful")

        return results


def download_artist_videos(artist_key: str, output_subdir: str = None,
                           max_videos: int = 500, max_workers: int = 4) -> List[DownloadResult]:
    """Convenience function to download all music videos for an artist.

    Args:
        artist_key: Key from KNOWN_CHANNELS ('conor_maynard' or 'william_singe')
        output_subdir: Subdirectory under data/youtube_audio (defaults to artist_key)
        max_videos: Maximum videos to download
        max_workers: Parallel download workers

    Returns:
        List of download results
    """
    from .channel_scraper import scrape_artist_channel

    # First scrape the channel for video list
    videos = scrape_artist_channel(artist_key, max_videos=max_videos)

    if not videos:
        logger.warning(f"No videos found for {artist_key}")
        return []

    # Download audio
    output_dir = Path('data/youtube_audio') / (output_subdir or artist_key)
    downloader = YouTubeDownloader(output_dir, max_workers=max_workers)

    results = downloader.download_batch(videos)

    return results


async def download_artist_videos_async(artist_key: str, output_subdir: str = None,
                                       max_videos: int = 500, max_workers: int = 4
                                       ) -> List[DownloadResult]:
    """Async wrapper for download_artist_videos.

    Runs the download in a thread pool to avoid blocking event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        download_artist_videos,
        artist_key,
        output_subdir,
        max_videos,
        max_workers
    )
