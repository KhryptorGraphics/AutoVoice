"""Shared web utilities for AutoVoice API."""

ALLOWED_AUDIO_EXTENSIONS = {
    'wav', 'mp3', 'flac', 'ogg', 'opus', 'aac', 'm4a', 'wma', 'aiff', 'webm'
}


def allowed_file(filename: str) -> bool:
    """Check if filename has an allowed audio extension."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_AUDIO_EXTENSIONS
