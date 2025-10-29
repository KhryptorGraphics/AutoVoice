"""Web utility functions for AutoVoice"""

# Shared allowed file extensions for audio uploads
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}


def allowed_file(filename: str, allowed_extensions: set = None) -> bool:
    """Check if the file extension is allowed and filename is safe.

    Args:
        filename: The filename to check
        allowed_extensions: Set of allowed extensions (uses ALLOWED_AUDIO_EXTENSIONS if None)

    Returns:
        True if file is allowed, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_AUDIO_EXTENSIONS

    if not filename or '.' not in filename:
        return False

    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False

    # Check file extension
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in allowed_extensions
