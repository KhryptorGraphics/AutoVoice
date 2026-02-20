"""YouTube integration for voice training data collection."""

from .channel_scraper import (
    VideoMetadata,
    YouTubeChannelScraper,
    scrape_artist_channel,
)
from .downloader import (
    DownloadResult,
    YouTubeDownloader,
    download_artist_videos,
    download_artist_videos_async,
)

__all__ = [
    'VideoMetadata',
    'YouTubeChannelScraper',
    'scrape_artist_channel',
    'DownloadResult',
    'YouTubeDownloader',
    'download_artist_videos',
    'download_artist_videos_async',
]
