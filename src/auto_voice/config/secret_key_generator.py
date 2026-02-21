"""Secure random secret key generation for Flask and other sensitive configs.

Task 1.1: Add secure random key generation utility function

Generates cryptographically secure random keys using Python's secrets module.
Keys are URL-safe base64 encoded strings suitable for:
- Flask SECRET_KEY
- JWT signing keys
- API tokens
- Session tokens
"""

import secrets
import base64
from typing import Optional


def generate_secret_key(length: int = 32) -> str:
    """Generate a cryptographically secure random secret key.

    Uses Python's secrets module to generate high-entropy random bytes,
    then encodes them as a URL-safe base64 string for safe storage in
    environment variables and config files.

    Args:
        length: Minimum length of random bytes to generate (default: 32).
               The resulting base64 string will be longer (~43 chars for 32 bytes).

    Returns:
        URL-safe base64-encoded random string

    Raises:
        ValueError: If length is less than 32 (minimum security requirement)

    Examples:
        >>> key = generate_secret_key()
        >>> len(key) >= 32
        True
        >>> key1 = generate_secret_key()
        >>> key2 = generate_secret_key()
        >>> key1 != key2  # Each key should be unique
        True
    """
    if length < 32:
        raise ValueError(
            f"Secret key length must be at least 32 bytes for security, got {length}"
        )

    # Generate cryptographically secure random bytes
    random_bytes = secrets.token_bytes(length)

    # Encode as URL-safe base64 (removes padding for cleaner keys)
    key = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')

    return key


def generate_hex_key(length: int = 32) -> str:
    """Generate a cryptographically secure random hex key.

    Alternative to generate_secret_key() that returns a hexadecimal string
    instead of base64. Useful when hex encoding is preferred.

    Args:
        length: Minimum length of random bytes to generate (default: 32).
               The resulting hex string will be 2x this length.

    Returns:
        Hexadecimal-encoded random string

    Raises:
        ValueError: If length is less than 32 (minimum security requirement)

    Examples:
        >>> key = generate_hex_key()
        >>> len(key) >= 64  # 32 bytes = 64 hex chars
        True
        >>> all(c in '0123456789abcdef' for c in key)
        True
    """
    if length < 32:
        raise ValueError(
            f"Secret key length must be at least 32 bytes for security, got {length}"
        )

    return secrets.token_hex(length)


def generate_urlsafe_token(nbytes: Optional[int] = None) -> str:
    """Generate a URL-safe random token.

    Wrapper around secrets.token_urlsafe() with reasonable defaults
    for API tokens and similar use cases.

    Args:
        nbytes: Number of random bytes (default: 32 if None).
               The resulting token will be longer due to base64 encoding.

    Returns:
        URL-safe random token string

    Examples:
        >>> token = generate_urlsafe_token()
        >>> len(token) >= 32
        True
    """
    if nbytes is None:
        nbytes = 32

    return secrets.token_urlsafe(nbytes)
