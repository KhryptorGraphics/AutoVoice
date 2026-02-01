"""YouTube channel scraper using yt-dlp.

Discovers and filters videos from artist channels for training data collection.
"""
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video."""
    video_id: str
    title: str
    duration: float  # seconds
    upload_date: str  # YYYYMMDD format
    channel: str
    channel_id: str
    view_count: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def is_music(self) -> bool:
        """Heuristic check if video is likely music content."""
        title_lower = self.title.lower()
        music_keywords = ['cover', 'song', 'music', 'acoustic', 'live', 'mashup',
                          'remix', 'official', 'audio', 'lyrics']
        non_music = ['vlog', 'q&a', 'reaction', 'behind', 'bts', 'tour', 'challenge',
                     'interview', 'prank', 'unboxing', 'haul', 'mukbang', 'asmr']

        # Short videos likely not full songs
        if self.duration < 60:
            return False
        # Very long videos (>15 min) likely compilations or non-music
        if self.duration > 900:
            return False

        # Explicit non-music keywords
        if any(kw in title_lower for kw in non_music):
            return False

        # Has music keywords OR is 1-8 minutes (typical song length with buffer)
        return any(kw in title_lower for kw in music_keywords) or (60 < self.duration < 480)

    @property
    def is_solo_artist(self) -> bool:
        """Check if the primary artist is solo (not a collaboration).

        Returns False if the title clearly indicates another featured artist.
        """
        import re
        title_lower = self.title.lower()

        # Patterns that indicate collaboration where another artist is featured
        collab_patterns = [
            r'\bft\.?\s+\w',           # ft. or ft followed by name
            r'\bfeat\.?\s+\w',         # feat. or feat followed by name
            r'\bfeaturing\s+\w',       # featuring
            r'\bwith\s+[A-Z]',         # with [Name]
            r'\bx\s+[A-Z]',            # Artist x Artist
            r'\b&\s+[A-Z]',            # Artist & Artist
            r'\band\s+[A-Z]',          # Artist and Artist
        ]

        for pattern in collab_patterns:
            if re.search(pattern, self.title, re.IGNORECASE):
                return False

        return True

    @property
    def is_valid_for_training(self) -> bool:
        """Check if video is suitable for voice training.

        Must be music content where the channel owner is the solo/primary artist.
        """
        return self.is_music and self.is_solo_artist


class YouTubeChannelScraper:
    """Scrapes video metadata from YouTube channels using yt-dlp."""

    # Known channel handles/URLs
    KNOWN_CHANNELS = {
        'conor_maynard': 'https://www.youtube.com/@ConorMaynard',
        'william_singe': 'https://www.youtube.com/@WilliamSinge',
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize scraper.

        Args:
            output_dir: Directory to store metadata cache
        """
        self.output_dir = output_dir or Path('data/youtube_metadata')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_channel_videos(self, channel_url: str, max_videos: int = 1000,
                           music_only: bool = True,
                           solo_only: bool = True) -> List[VideoMetadata]:
        """Get all video metadata from a channel.

        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum videos to fetch (default 1000 for full channel)
            music_only: Filter to likely music content only
            solo_only: Filter out collaborations/features (only solo performances)

        Returns:
            List of VideoMetadata objects
        """
        logger.info(f"Scraping videos from {channel_url}")

        # yt-dlp command to extract metadata only
        cmd = [
            'yt-dlp',
            '--flat-playlist',
            '--dump-json',
            '--no-download',
            '--no-warnings',
            '--playlist-end', str(max_videos),
            f'{channel_url}/videos'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )

            if result.returncode != 0:
                logger.error(f"yt-dlp error: {result.stderr}")
                raise RuntimeError(f"Failed to scrape channel: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Channel scraping timed out after 5 minutes")

        videos = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                video = VideoMetadata(
                    video_id=data.get('id', ''),
                    title=data.get('title', ''),
                    duration=float(data.get('duration', 0) or 0),
                    upload_date=data.get('upload_date', ''),
                    channel=data.get('channel', ''),
                    channel_id=data.get('channel_id', ''),
                    view_count=int(data.get('view_count', 0) or 0),
                    description=data.get('description', ''),
                    tags=data.get('tags', []) or [],
                )

                # Apply filters
                if music_only and not video.is_music:
                    logger.debug(f"Skipping non-music: {video.title}")
                    continue
                if solo_only and not video.is_solo_artist:
                    logger.debug(f"Skipping collaboration: {video.title}")
                    continue

                videos.append(video)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse video metadata: {e}")
                continue

        logger.info(f"Found {len(videos)} videos (music_only={music_only})")
        return videos

    def save_metadata(self, videos: List[VideoMetadata], artist_name: str) -> Path:
        """Save video metadata to JSON file.

        Args:
            videos: List of video metadata
            artist_name: Artist name for filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f'{artist_name}_videos.json'
        data = [
            {
                'video_id': v.video_id,
                'title': v.title,
                'duration': v.duration,
                'upload_date': v.upload_date,
                'channel': v.channel,
                'view_count': v.view_count,
                'is_music': v.is_music,
            }
            for v in videos
        ]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved metadata to {output_path}")
        return output_path

    def load_metadata(self, artist_name: str) -> List[VideoMetadata]:
        """Load cached video metadata.

        Args:
            artist_name: Artist name to load

        Returns:
            List of VideoMetadata objects
        """
        path = self.output_dir / f'{artist_name}_videos.json'
        if not path.exists():
            return []

        with open(path) as f:
            data = json.load(f)

        return [
            VideoMetadata(
                video_id=v['video_id'],
                title=v['title'],
                duration=v['duration'],
                upload_date=v['upload_date'],
                channel=v['channel'],
                channel_id='',
                view_count=v.get('view_count', 0),
            )
            for v in data
        ]


def scrape_artist_channel(artist_key: str, max_videos: int = 1000,
                          solo_only: bool = True) -> List[VideoMetadata]:
    """Convenience function to scrape known artist channel.

    Args:
        artist_key: Key from KNOWN_CHANNELS ('conor_maynard' or 'william_singe')
        max_videos: Maximum videos to fetch (default 1000 for full channel)
        solo_only: Filter out collaborations (default True for training)

    Returns:
        List of video metadata
    """
    scraper = YouTubeChannelScraper()

    if artist_key not in scraper.KNOWN_CHANNELS:
        raise ValueError(f"Unknown artist: {artist_key}. Known: {list(scraper.KNOWN_CHANNELS.keys())}")

    channel_url = scraper.KNOWN_CHANNELS[artist_key]
    videos = scraper.get_channel_videos(
        channel_url,
        max_videos=max_videos,
        music_only=True,
        solo_only=solo_only
    )
    scraper.save_metadata(videos, artist_key)

    return videos
