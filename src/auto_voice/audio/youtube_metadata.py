"""YouTube metadata parsing and fetching for featured artist detection.

This module parses YouTube video titles and descriptions to identify:
- Main artist performing the song
- Featured/collaborating artists (ft., feat., vs., with, &, x patterns)
- Cover song detection and original artist identification

It also provides yt-dlp integration for fetching metadata from YouTube.

Usage:
    from auto_voice.audio.youtube_metadata import YouTubeMetadataFetcher

    fetcher = YouTubeMetadataFetcher()
    metadata = fetcher.fetch_metadata("dQw4w9WgXcQ")
    featured = parse_featured_artists(metadata.title)
"""

import re
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# Patterns that indicate featured artists (case insensitive)
# Note: patterns stop at ( [ ] | , & - to avoid capturing song titles
FEATURED_PATTERNS = [
    # "ft." or "ft " - most common
    r'\bft\.?\s+([^(\[\]|,&-]+?)(?:\s*[(\[\]|,&-]|$)',
    # "feat." or "feat "
    r'\bfeat\.?\s+([^(\[\]|,&-]+?)(?:\s*[(\[\]|,&-]|$)',
    # "featuring"
    r'\bfeaturing\s+([^(\[\]|,&-]+?)(?:\s*[(\[\]|,&-]|$)',
    # "with" - but not "mixed with" or "produced with"
    r'(?<!mixed\s)(?<!produced\s)\bwith\s+([^(\[\]|,&-]+?)(?:\s*[(\[\]|,&-]|$)',
    # "vs." or "vs" - battle/mashup
    r'\bvs\.?\s+([^(\[\]|,&-]+?)(?:\s*[(\[\]|,&-]|$)',
    # "&" between artists (after main artist extraction)
    r'\s+&\s+([^(\[\]|,&-]+?)(?:\s*[(\[\]|,&-]|$)',
    # "x" collaboration (space x space)
    r'\s+x\s+([^(\[\]|-]+?)(?:\s*[(\[\]|-]|$)',
]

# Patterns inside parentheses for featured artists
PAREN_FEATURED_PATTERNS = [
    r'\(ft\.?\s+([^)]+)\)',
    r'\(feat\.?\s+([^)]+)\)',
    r'\(featuring\s+([^)]+)\)',
    r'\(with\s+([^)]+)\)',
]

# Patterns to exclude (producers, not singers)
EXCLUDE_PATTERNS = [
    r'\bprod\.?\s+(?:by\s+)?',
    r'\bproduced\s+by\s+',
    r'\bmixed\s+by\s+',
    r'\bremix(?:ed)?\s+by\s+',
]

# Cover song patterns
COVER_PATTERNS = [
    r'\(([^)]+)\s+cover\)',
    r'\bcover\s+of\s+([^(\[\]]+?)(?:\s*[(\[\]]|$)',
    r'originally\s+by\s+([^(\[\]]+?)(?:\s*[(\[\]]|$)',
]

# Description patterns for featured artists
DESCRIPTION_FEATURED_PATTERNS = [
    r'featuring\s+(?:vocals?\s+by\s+)?([^.!\n]+)',
    r'feat(?:uring)?\.?\s+([^.!\n]+)',
    r'vocals?\s+by\s+([^.!\n]+)',
]


def _clean_artist_name(name: str) -> str:
    """Clean and normalize an artist name."""
    # Remove common suffixes
    name = re.sub(r'\s*\(official\s+(?:video|audio|lyrics?)\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\[official\s+(?:video|audio|lyrics?)\]', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*official\s+(?:video|audio|lyrics?)', '', name, flags=re.IGNORECASE)

    # Strip whitespace and trailing punctuation
    name = name.strip()
    name = re.sub(r'[,&|]+$', '', name).strip()

    return name


def _split_multiple_artists(artist_str: str) -> List[str]:
    """Split a string containing multiple artists separated by , & and."""
    # Split by comma, ampersand, conjunctions, or repeated feature markers.
    artists = re.split(
        r'\s*[,&]\s*|\s+and\s+|\s+ft\.?\s+|\s+feat\.?\s+|\s+featuring\s+',
        artist_str,
        flags=re.IGNORECASE,
    )
    return [_clean_artist_name(a) for a in artists if a.strip()]


def _clean_description_artist_credit(text: str) -> str:
    """Normalize description-derived artist credits before splitting."""
    return re.sub(
        r'^(?:vocals?\s+by\s+|voice\s+by\s+|performed\s+by\s+)',
        '',
        text.strip(),
        flags=re.IGNORECASE,
    )


def _is_producer_credit(text: str) -> bool:
    """Check if text is a producer credit rather than a featured artist."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def parse_featured_artists(title: str, description: Optional[str] = None) -> List[str]:
    """Parse featured artists from YouTube video title and description.

    Args:
        title: The video title
        description: Optional video description

    Returns:
        List of featured artist names (empty if none found)
    """
    featured = []

    # Skip if it looks like a producer credit
    if _is_producer_credit(title):
        # Check if only producer pattern, no other feature patterns
        has_feature = any(
            re.search(p, title, re.IGNORECASE)
            for p in ['\\bft\\.', '\\bfeat\\.', '\\bfeaturing\\b', '\\bvs\\.?\\b', '\\bwith\\b', '\\s&\\s', '\\sx\\s']
        )
        if not has_feature:
            return []

    # Try parentheses patterns first (most specific)
    for pattern in PAREN_FEATURED_PATTERNS:
        matches = re.findall(pattern, title, re.IGNORECASE)
        for match in matches:
            artists = _split_multiple_artists(match)
            featured.extend(artists)

    # If found in parentheses, return those
    if featured:
        return [a for a in featured if a and len(a) > 1]

    # Try main patterns
    for pattern in FEATURED_PATTERNS:
        matches = re.findall(pattern, title, re.IGNORECASE)
        for match in matches:
            # Skip if this looks like a producer credit
            match_lower = match.lower().strip()
            if any(word in match_lower for word in ['prod', 'producer', 'remix']):
                continue
            artists = _split_multiple_artists(match)
            featured.extend(artists)

    # Remove duplicates while preserving order
    seen = set()
    unique_featured = []
    for artist in featured:
        artist_lower = artist.lower()
        if artist_lower not in seen and artist and len(artist) > 1:
            seen.add(artist_lower)
            unique_featured.append(artist)

    # If nothing found in title and description provided, try description
    if not unique_featured and description:
        for pattern in DESCRIPTION_FEATURED_PATTERNS:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                # Clean and validate
                artists = _split_multiple_artists(_clean_description_artist_credit(match))
                for artist in artists:
                    artist_lower = artist.lower()
                    if artist_lower not in seen and artist and len(artist) > 1:
                        # Skip common non-artist words
                        if artist_lower not in ['me', 'us', 'them', 'the', 'a', 'an']:
                            seen.add(artist_lower)
                            unique_featured.append(artist)

    return unique_featured


def extract_main_artist(title: str) -> Optional[str]:
    """Extract the main artist from a YouTube video title.

    Expects format: "Artist - Song Title" or "Artist ft. Other - Song"

    Args:
        title: The video title

    Returns:
        Main artist name or None if not found
    """
    # Standard "Artist - Song" format
    if ' - ' not in title:
        return None

    # Split on first dash
    parts = title.split(' - ', 1)
    artist_part = parts[0].strip()

    # Remove featured artist indicators from main artist
    # "Main Artist ft. Guest - Song" -> "Main Artist"
    for pattern in [r'\s+ft\.?\s+.*$', r'\s+feat\.?\s+.*$', r'\s+featuring\s+.*$',
                    r'\s+vs\.?\s+.*$', r'\s+x\s+.*$', r'\s+&\s+.*$', r'\s+with\s+.*$']:
        artist_part = re.sub(pattern, '', artist_part, flags=re.IGNORECASE)

    return _clean_artist_name(artist_part) if artist_part else None


def detect_cover_song(title: str, description: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Detect if a video is a cover song and identify the original artist.

    Args:
        title: The video title
        description: Optional video description

    Returns:
        Tuple of (is_cover, original_artist)
    """
    # Check title for cover patterns
    for pattern in COVER_PATTERNS:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            original_artist = _clean_artist_name(match.group(1))
            return True, original_artist

    # Check if "cover" appears in title
    if re.search(r'\bcover\b', title, re.IGNORECASE):
        return True, None

    # Check description
    if description:
        for pattern in COVER_PATTERNS:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                original_artist = _clean_artist_name(match.group(1))
                return True, original_artist

        if re.search(r'\bcover\b', description, re.IGNORECASE):
            return True, None

    return False, None


def parse_youtube_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Parse YouTube video metadata to extract artist information.

    Args:
        metadata: Dictionary with keys like 'title', 'description', 'uploader', 'channel'

    Returns:
        Dictionary with:
        - main_artist: The primary artist
        - featured_artists: List of featured/collaborating artists
        - is_cover: Whether this is a cover song
        - original_artist: Original artist if this is a cover
        - song_title: The song title (if extractable)
    """
    title = metadata.get('title', '')
    description = metadata.get('description', '')
    uploader = metadata.get('uploader', '')
    channel = metadata.get('channel', '')

    # Extract main artist
    main_artist = extract_main_artist(title)

    # If no main artist from title, try uploader/channel
    if not main_artist:
        main_artist = uploader or channel or None

    # Get featured artists
    featured_artists = parse_featured_artists(title, description)

    # Detect cover song
    is_cover, original_artist = detect_cover_song(title, description)

    # Extract song title (part after the dash)
    song_title = None
    if ' - ' in title:
        parts = title.split(' - ', 1)
        song_title = parts[1].strip() if len(parts) > 1 else None
        # Clean up song title
        if song_title:
            # Remove featured artist info from song title
            song_title = re.sub(r'\s*\(ft\.?\s+[^)]+\)', '', song_title, flags=re.IGNORECASE)
            song_title = re.sub(r'\s*\(feat\.?\s+[^)]+\)', '', song_title, flags=re.IGNORECASE)
            song_title = re.sub(r'\s*ft\.?\s+.*$', '', song_title, flags=re.IGNORECASE)
            song_title = re.sub(r'\s*feat\.?\s+.*$', '', song_title, flags=re.IGNORECASE)
            song_title = re.sub(
                r'\s*[\[\(\{][^\]\)\}]*\b(?:official|lyric|lyrics|audio|video|hd|4k|mv)\b[^\]\)\}]*[\]\)\}]',
                '',
                song_title,
                flags=re.IGNORECASE,
            )
            song_title = _clean_artist_name(song_title)

    return {
        'main_artist': main_artist,
        'featured_artists': featured_artists,
        'is_cover': is_cover,
        'original_artist': original_artist,
        'song_title': song_title,
    }


# ============================================================================
# YouTube Metadata Fetcher (yt-dlp integration)
# ============================================================================

@dataclass
class VideoMetadata:
    """YouTube video metadata."""
    video_id: str
    title: str
    channel: str
    upload_date: str
    duration_sec: float
    description: Optional[str] = None


class YouTubeMetadataFetcher:
    """Fetch YouTube video metadata using yt-dlp."""

    def __init__(self, yt_dlp_path: str = 'yt-dlp'):
        """Initialize the fetcher.

        Args:
            yt_dlp_path: Path to yt-dlp executable
        """
        self.yt_dlp_path = yt_dlp_path

    def fetch_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Fetch metadata for a YouTube video.

        Args:
            video_id: YouTube video ID (e.g., dQw4w9WgXcQ)

        Returns:
            VideoMetadata or None if fetch failed
        """
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            result = subprocess.run(
                [
                    self.yt_dlp_path,
                    '--dump-json',
                    '--no-download',
                    '--no-playlist',
                    url
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"yt-dlp failed for {video_id}: {result.stderr}")
                return None

            data = json.loads(result.stdout)

            return VideoMetadata(
                video_id=video_id,
                title=data.get('title', ''),
                channel=data.get('channel', data.get('uploader', '')),
                upload_date=data.get('upload_date', ''),
                duration_sec=float(data.get('duration', 0)),
                description=data.get('description'),
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout fetching metadata for {video_id}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse yt-dlp output for {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching metadata for {video_id}: {e}")
            return None

    def extract_video_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract YouTube video ID from a filename.

        Expected format: {video_id}_vocals.wav or {video_id}.wav

        Args:
            filename: Filename or path

        Returns:
            Video ID or None
        """
        # Get just the filename without extension
        stem = Path(filename).stem

        # Remove common suffixes
        stem = re.sub(r'_vocals$', '', stem)
        stem = re.sub(r'_diarization$', '', stem)
        stem = re.sub(r'_isolated$', '', stem)
        stem = re.sub(r'_SPEAKER_\d+$', '', stem)

        # YouTube video IDs are 11 characters, alphanumeric plus - and _
        # Check if remaining stem looks like a video ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', stem):
            return stem

        # Fall back to separator-delimited tokens to avoid arbitrary 11-char slices
        for token in re.split(r'[_-]+', stem):
            if re.match(r'^[a-zA-Z0-9_-]{11}$', token):
                return token

        return None

    def fetch_metadata_for_directory(
        self,
        directory: Path,
        file_pattern: str = '*_vocals.wav',
    ) -> Dict[str, VideoMetadata]:
        """Fetch metadata for all video IDs found in a directory.

        Args:
            directory: Directory to scan
            file_pattern: Glob pattern for files

        Returns:
            Dict mapping video_id to VideoMetadata
        """
        results = {}
        seen_ids = set()

        for file_path in directory.glob(file_pattern):
            video_id = self.extract_video_id_from_filename(file_path.name)
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                metadata = self.fetch_metadata(video_id)
                if metadata:
                    results[video_id] = metadata
                    logger.info(f"Fetched metadata for {video_id}: {metadata.title}")

        return results


def populate_database_from_files(
    artist_name: str,
    separated_dir: Path,
    diarized_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Populate the database with track metadata from extracted files.

    Args:
        artist_name: Artist name (e.g., "conor_maynard")
        separated_dir: Directory with separated vocals
        diarized_dir: Directory with diarization JSONs

    Returns:
        Statistics dict
    """
    from ..db.operations import upsert_track, add_featured_artist

    fetcher = YouTubeMetadataFetcher()
    stats = {
        'tracks_processed': 0,
        'tracks_with_metadata': 0,
        'featured_artists_found': 0,
        'errors': [],
    }

    if diarized_dir is None:
        diarized_dir = Path(f'data/diarized_youtube/{artist_name}')

    # Find all vocal files
    vocal_files = list(separated_dir.glob('*_vocals.wav')) + list(separated_dir.glob('*.wav'))
    seen_ids = set()

    for vocal_file in vocal_files:
        video_id = fetcher.extract_video_id_from_filename(vocal_file.name)
        if not video_id or video_id in seen_ids:
            continue

        seen_ids.add(video_id)
        stats['tracks_processed'] += 1

        # Find corresponding diarization file
        diarization_path = None
        for pattern in [f'{video_id}_diarization.json', f'{video_id}_vocals_diarization.json']:
            candidate = diarized_dir / pattern
            if candidate.exists():
                diarization_path = str(candidate)
                break

        # Fetch YouTube metadata
        metadata = fetcher.fetch_metadata(video_id)

        if metadata:
            stats['tracks_with_metadata'] += 1

            # Upsert track
            upsert_track(
                track_id=video_id,
                title=metadata.title,
                channel=metadata.channel,
                upload_date=metadata.upload_date,
                duration_sec=metadata.duration_sec,
                artist_name=artist_name,
                vocals_path=str(vocal_file),
                diarization_path=diarization_path,
            )

            # Parse and add featured artists
            featured = parse_featured_artists(metadata.title, metadata.description)
            for artist in featured:
                add_featured_artist(
                    track_id=video_id,
                    name=artist,
                    pattern_matched='title',
                )
                stats['featured_artists_found'] += 1
                logger.info(f"Found featured artist '{artist}' in '{metadata.title}'")
        else:
            # Still add track without YouTube metadata
            upsert_track(
                track_id=video_id,
                artist_name=artist_name,
                vocals_path=str(vocal_file),
                diarization_path=diarization_path,
            )
            stats['errors'].append(f"Could not fetch metadata for {video_id}")

    return stats


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Fetch YouTube metadata for tracks')
    parser.add_argument('--artist', help='Artist name')
    parser.add_argument('--separated-dir', type=Path, help='Directory with separated vocals')
    parser.add_argument('--test-id', help='Test with a single video ID')
    parser.add_argument('--test-title', help='Test parsing a title')

    args = parser.parse_args()

    fetcher = YouTubeMetadataFetcher()

    if args.test_id:
        metadata = fetcher.fetch_metadata(args.test_id)
        if metadata:
            print(f"Title: {metadata.title}")
            print(f"Channel: {metadata.channel}")
            print(f"Duration: {metadata.duration_sec}s")
            featured = parse_featured_artists(metadata.title, metadata.description)
            print(f"Featured artists: {featured}")
    elif args.test_title:
        featured = parse_featured_artists(args.test_title)
        print(f"Featured artists: {featured}")
    elif args.artist:
        separated_dir = args.separated_dir or Path(f'data/separated_youtube/{args.artist}')
        stats = populate_database_from_files(args.artist, separated_dir)
        print(f"\nStats: {stats}")
