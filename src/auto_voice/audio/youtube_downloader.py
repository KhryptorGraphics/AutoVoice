"""YouTube audio downloader with metadata extraction and diarization integration.

Uses yt-dlp to download audio from YouTube videos and extracts metadata
for featured artist detection.
"""

import logging
import os
import subprocess
import tempfile
import json
import uuid
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from .youtube_metadata import parse_youtube_metadata, parse_featured_artists

logger = logging.getLogger(__name__)


def _find_ytdlp() -> str:
    """Find yt-dlp executable, checking common locations."""
    # First try shutil.which
    ytdlp = shutil.which('yt-dlp')
    if ytdlp:
        return ytdlp

    # Check common conda locations
    common_paths = [
        '/home/kp/anaconda3/bin/yt-dlp',
        '/home/kp/anaconda3/envs/autovoice-thor/bin/yt-dlp',
        '/usr/local/bin/yt-dlp',
        '/usr/bin/yt-dlp',
        os.path.expanduser('~/.local/bin/yt-dlp'),
    ]

    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Fallback to just 'yt-dlp' and let it fail with a clear error
    return 'yt-dlp'


@dataclass
class YouTubeDownloadResult:
    """Result of a YouTube download operation."""
    success: bool
    audio_path: Optional[str] = None
    title: str = ""
    duration: float = 0.0
    main_artist: Optional[str] = None
    featured_artists: List[str] = field(default_factory=list)
    is_cover: bool = False
    original_artist: Optional[str] = None
    song_title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    video_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class YouTubeDownloader:
    """Downloads audio from YouTube videos with metadata extraction."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the downloader.

        Args:
            output_dir: Directory to save downloaded audio. Defaults to temp dir.
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)

    def download(
        self,
        url: str,
        output_filename: Optional[str] = None,
        audio_format: str = "wav",
        sample_rate: int = 44100,
    ) -> YouTubeDownloadResult:
        """Download audio from a YouTube video.

        Args:
            url: YouTube video URL
            output_filename: Optional output filename (without extension)
            audio_format: Output audio format (wav, mp3, flac)
            sample_rate: Output sample rate

        Returns:
            YouTubeDownloadResult with audio path and metadata
        """
        try:
            # First, get metadata without downloading
            metadata = self._get_metadata(url)
            if not metadata:
                return YouTubeDownloadResult(
                    success=False,
                    error="Failed to fetch video metadata"
                )

            # Parse artist information
            parsed = parse_youtube_metadata(metadata)

            # Generate output filename
            video_id = metadata.get('id', str(uuid.uuid4())[:8])
            if not output_filename:
                # Sanitize title for filename
                safe_title = self._sanitize_filename(metadata.get('title', 'download'))
                output_filename = f"{safe_title}_{video_id}"

            output_path = os.path.join(self.output_dir, f"{output_filename}.{audio_format}")

            # Download audio
            success = self._download_audio(url, output_path, audio_format, sample_rate)

            if not success:
                return YouTubeDownloadResult(
                    success=False,
                    error="Failed to download audio",
                    title=metadata.get('title', ''),
                    video_id=video_id,
                )

            return YouTubeDownloadResult(
                success=True,
                audio_path=output_path,
                title=metadata.get('title', ''),
                duration=metadata.get('duration', 0),
                main_artist=parsed.get('main_artist'),
                featured_artists=parsed.get('featured_artists', []),
                is_cover=parsed.get('is_cover', False),
                original_artist=parsed.get('original_artist'),
                song_title=parsed.get('song_title'),
                thumbnail_url=metadata.get('thumbnail'),
                video_id=video_id,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"YouTube download failed: {e}")
            return YouTubeDownloadResult(
                success=False,
                error=str(e)
            )

    def get_video_info(self, url: str) -> YouTubeDownloadResult:
        """Get video information without downloading.

        Args:
            url: YouTube video URL

        Returns:
            YouTubeDownloadResult with metadata only (no audio_path)
        """
        try:
            metadata = self._get_metadata(url)
            if not metadata:
                return YouTubeDownloadResult(
                    success=False,
                    error="Failed to fetch video metadata"
                )

            parsed = parse_youtube_metadata(metadata)

            return YouTubeDownloadResult(
                success=True,
                title=metadata.get('title', ''),
                duration=metadata.get('duration', 0),
                main_artist=parsed.get('main_artist'),
                featured_artists=parsed.get('featured_artists', []),
                is_cover=parsed.get('is_cover', False),
                original_artist=parsed.get('original_artist'),
                song_title=parsed.get('song_title'),
                thumbnail_url=metadata.get('thumbnail'),
                video_id=metadata.get('id'),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return YouTubeDownloadResult(
                success=False,
                error=str(e)
            )

    def _get_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch video metadata using yt-dlp."""
        try:
            ytdlp = _find_ytdlp()
            cmd = [
                ytdlp,
                '--dump-json',
                '--no-download',
                '--no-warnings',
                url
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"yt-dlp metadata failed: {result.stderr}")
                return None

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("yt-dlp metadata timed out")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse yt-dlp output: {e}")
            return None
        except Exception as e:
            logger.error(f"yt-dlp metadata error: {e}")
            return None

    def _download_audio(
        self,
        url: str,
        output_path: str,
        audio_format: str,
        sample_rate: int
    ) -> bool:
        """Download audio using yt-dlp."""
        try:
            # Remove extension from output path for yt-dlp template
            output_base = os.path.splitext(output_path)[0]
            ytdlp = _find_ytdlp()

            cmd = [
                ytdlp,
                '-x',  # Extract audio
                '--audio-format', audio_format,
                '--audio-quality', '0',  # Best quality
                '-o', f"{output_base}.%(ext)s",
                '--no-playlist',
                '--no-warnings',
            ]

            # Add post-processor for sample rate if wav
            if audio_format == 'wav':
                cmd.extend([
                    '--postprocessor-args',
                    f'ffmpeg:-ar {sample_rate}'
                ])

            cmd.append(url)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"yt-dlp download failed: {result.stderr}")
                return False

            # Verify file exists
            if not os.path.exists(output_path):
                # yt-dlp might have created file with different extension
                base = os.path.splitext(output_path)[0]
                for ext in ['.wav', '.mp3', '.m4a', '.webm', '.opus']:
                    alt_path = base + ext
                    if os.path.exists(alt_path):
                        # Rename to expected path if needed
                        if alt_path != output_path:
                            os.rename(alt_path, output_path)
                        return True
                logger.error(f"Downloaded file not found at {output_path}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("yt-dlp download timed out")
            return False
        except Exception as e:
            logger.error(f"yt-dlp download error: {e}")
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string for use as a filename."""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        # Limit length
        filename = filename[:100]
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        return filename


# Module-level instance for convenience
_downloader: Optional[YouTubeDownloader] = None


def get_downloader(output_dir: Optional[str] = None) -> YouTubeDownloader:
    """Get or create a YouTubeDownloader instance."""
    global _downloader
    if _downloader is None or (output_dir and _downloader.output_dir != output_dir):
        _downloader = YouTubeDownloader(output_dir)
    return _downloader
