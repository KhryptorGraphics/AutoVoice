"""YouTube metadata parsing for featured artist detection.

This module parses YouTube video titles and descriptions to identify:
- Main artist performing the song
- Featured/collaborating artists (ft., feat., vs., with, &, x patterns)
- Cover song detection and original artist identification
"""

import re
from typing import Dict, List, Optional, Any


# Patterns that indicate featured artists (case insensitive)
FEATURED_PATTERNS = [
    # "ft." or "ft " - most common
    r'\bft\.?\s+([^(\[\]|]+?)(?:\s*[(\[\]|,&]|$)',
    # "feat." or "feat "
    r'\bfeat\.?\s+([^(\[\]|]+?)(?:\s*[(\[\]|,&]|$)',
    # "featuring"
    r'\bfeaturing\s+([^(\[\]|]+?)(?:\s*[(\[\]|,&]|$)',
    # "with" - but not "mixed with" or "produced with"
    r'(?<!mixed\s)(?<!produced\s)\bwith\s+([^(\[\]|]+?)(?:\s*[(\[\]|,&]|$)',
    # "vs." or "vs" - battle/mashup
    r'\bvs\.?\s+([^(\[\]|-]+?)(?:\s*[(\[\]|-]|$)',
    # "&" between artists (after main artist extraction)
    r'\s+&\s+([^(\[\]|-]+?)(?:\s*[(\[\]|-]|$)',
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
    # Split by comma, ampersand, or "and"
    artists = re.split(r'\s*[,&]\s*|\s+and\s+', artist_str, flags=re.IGNORECASE)
    return [_clean_artist_name(a) for a in artists if a.strip()]


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
                artists = _split_multiple_artists(match)
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
            song_title = _clean_artist_name(song_title)

    return {
        'main_artist': main_artist,
        'featured_artists': featured_artists,
        'is_cover': is_cover,
        'original_artist': original_artist,
        'song_title': song_title,
    }
