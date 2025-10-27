"""Voice profile storage and management with JSON metadata and NumPy embeddings"""

from __future__ import annotations
import json
import logging
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


class ProfileStorageError(Exception):
    """Base exception for profile storage errors"""
    pass


class ProfileNotFoundError(ProfileStorageError):
    """Exception raised when profile doesn't exist"""
    pass


class ProfileValidationError(ProfileStorageError):
    """Exception raised when profile structure is invalid"""
    pass


class VoiceProfileStorage:
    """CRUD operations for voice profiles with JSON metadata and NumPy embeddings

    Manages persistent storage of voice profiles using a simple file-based approach:
    - JSON files for metadata (user_id, timestamps, vocal features, etc.)
    - NumPy .npy files for 256-dim speaker embeddings
    - In-memory LRU cache for frequently accessed profiles
    - Atomic write operations for data integrity

    Storage Structure:
        ~/.cache/autovoice/voice_profiles/
        ├── {profile_id}.json           # Metadata
        ├── {profile_id}_embedding.npy  # Speaker embedding (256-dim)
        └── ...

    Features:
        - Thread-safe operations with RLock
        - LRU caching for fast repeated access
        - Atomic writes (temp file + rename)
        - Profile validation on save/load
        - Storage statistics and monitoring

    Example:
        >>> storage = VoiceProfileStorage()
        >>> profile = {
        ...     'profile_id': str(uuid.uuid4()),
        ...     'user_id': 'user123',
        ...     'embedding': np.random.randn(256),
        ...     'vocal_range': {'min_f0': 100.0, 'max_f0': 400.0},
        ...     'created_at': '2025-01-15T10:00:00Z'
        ... }
        >>> storage.save_profile(profile)
        >>> loaded = storage.load_profile(profile['profile_id'])

    Attributes:
        storage_dir (Path): Directory for profile storage
        cache_enabled (bool): Whether in-memory caching is enabled
        cache (OrderedDict): LRU cache for profiles
        lock (threading.RLock): Thread safety lock
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        cache_enabled: bool = True,
        cache_size: int = 100
    ):
        """Initialize VoiceProfileStorage

        Args:
            storage_dir: Optional storage directory path (default: ~/.cache/autovoice/voice_profiles/)
            cache_enabled: Enable in-memory LRU caching
            cache_size: Maximum number of cached profiles
        """
        if not NUMPY_AVAILABLE:
            raise ProfileStorageError("numpy is required for voice profile storage")

        # Set storage directory
        if storage_dir is None:
            storage_dir = '~/.cache/autovoice/voice_profiles/'

        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Cache configuration
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"VoiceProfileStorage initialized: dir={self.storage_dir}, cache_enabled={cache_enabled}")

    def save_profile(self, profile: Dict[str, Any]) -> str:
        """Save voice profile to disk with atomic write

        Args:
            profile: Profile dictionary with required keys:
                    - profile_id: Unique identifier (generated if missing)
                    - embedding: 256-dim numpy array
                    - created_at: ISO timestamp
                    Other optional keys: user_id, vocal_range, timbre_features, metadata

        Returns:
            Profile ID (str)

        Raises:
            ProfileValidationError: If profile structure is invalid
            ProfileStorageError: If save operation fails

        Example:
            >>> profile = {
            ...     'profile_id': str(uuid.uuid4()),
            ...     'embedding': np.random.randn(256),
            ...     'created_at': '2025-01-15T10:00:00Z'
            ... }
            >>> profile_id = storage.save_profile(profile)
        """
        with self.lock:
            try:
                # Generate profile_id if missing
                if 'profile_id' not in profile:
                    profile['profile_id'] = str(uuid.uuid4())

                profile_id = profile['profile_id']

                # Validate profile
                self._validate_profile(profile)

                # Extract embedding (don't store in JSON)
                embedding = profile['embedding']
                profile_meta = {k: v for k, v in profile.items() if k != 'embedding'}

                # Convert numpy types to Python types for JSON serialization
                profile_meta = self._convert_numpy_types(profile_meta)

                # Get file paths
                json_path, embedding_path = self._get_profile_path(profile_id)

                # Atomic write: JSON metadata
                self._atomic_write_json(json_path, profile_meta)

                # Atomic write: Embedding
                self._atomic_write_numpy(embedding_path, embedding)

                # Update cache
                if self.cache_enabled:
                    self.cache[profile_id] = profile
                    self._enforce_cache_limit()

                logger.debug(f"Saved profile: {profile_id}")
                return profile_id

            except ProfileValidationError:
                raise
            except Exception as e:
                logger.error(f"Failed to save profile: {e}")
                raise ProfileStorageError(f"Profile save failed: {e}")

    def load_profile(
        self,
        profile_id: str,
        include_embedding: bool = True
    ) -> Dict[str, Any]:
        """Load voice profile from disk

        Args:
            profile_id: Profile identifier
            include_embedding: Whether to load embedding array (default: True)

        Returns:
            Profile dictionary

        Raises:
            ProfileNotFoundError: If profile doesn't exist
            ProfileStorageError: If load operation fails

        Example:
            >>> profile = storage.load_profile('uuid-1234')
            >>> print(profile['user_id'])
            >>>
            >>> # Load without embedding for efficiency
            >>> profile_meta = storage.load_profile('uuid-1234', include_embedding=False)
        """
        with self.lock:
            # Check cache first
            if self.cache_enabled and profile_id in self.cache:
                logger.debug(f"Cache hit for profile: {profile_id}")
                # Move to end (most recently used)
                self.cache.move_to_end(profile_id)
                cached_profile = self.cache[profile_id]

                if not include_embedding:
                    return {k: v for k, v in cached_profile.items() if k != 'embedding'}
                return cached_profile.copy()

            try:
                # Get file paths
                json_path, embedding_path = self._get_profile_path(profile_id)

                # Check if profile exists
                if not json_path.exists():
                    raise ProfileNotFoundError(f"Profile not found: {profile_id}")

                # Load JSON metadata
                with open(json_path, 'r') as f:
                    profile = json.load(f)

                # Load embedding if requested
                if include_embedding:
                    if not embedding_path.exists():
                        raise ProfileStorageError(
                            f"Embedding file missing for profile: {profile_id}"
                        )
                    embedding = np.load(embedding_path)
                    profile['embedding'] = embedding

                # Update cache
                if self.cache_enabled and include_embedding:
                    self.cache[profile_id] = profile.copy()
                    self._enforce_cache_limit()

                logger.debug(f"Loaded profile: {profile_id}")
                return profile

            except ProfileNotFoundError:
                raise
            except Exception as e:
                logger.error(f"Failed to load profile {profile_id}: {e}")
                raise ProfileStorageError(f"Profile load failed: {e}")

    def list_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all voice profiles, optionally filtered by user_id

        Args:
            user_id: Optional user ID filter

        Returns:
            List of profile dictionaries (without embeddings for efficiency)

        Example:
            >>> all_profiles = storage.list_profiles()
            >>> user_profiles = storage.list_profiles(user_id='user123')
        """
        with self.lock:
            try:
                profiles = []

                # Scan storage directory for JSON files
                for json_path in self.storage_dir.glob("*.json"):
                    # Skip temporary files
                    if json_path.name.endswith('.tmp'):
                        continue

                    try:
                        with open(json_path, 'r') as f:
                            profile = json.load(f)

                        # Filter by user_id if provided
                        if user_id is not None:
                            if profile.get('user_id') != user_id:
                                continue

                        profiles.append(profile)

                    except Exception as e:
                        logger.warning(f"Failed to load profile {json_path.name}: {e}")
                        continue

                # Sort by created_at (newest first)
                profiles.sort(
                    key=lambda p: p.get('created_at', ''),
                    reverse=True
                )

                return profiles

            except Exception as e:
                logger.error(f"Failed to list profiles: {e}")
                raise ProfileStorageError(f"Profile list failed: {e}")

    def delete_profile(self, profile_id: str) -> bool:
        """Delete voice profile from disk

        Args:
            profile_id: Profile identifier

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = storage.delete_profile('uuid-1234')
            >>> print(f"Deleted: {deleted}")
        """
        with self.lock:
            try:
                json_path, embedding_path = self._get_profile_path(profile_id)

                # Check if profile exists
                if not json_path.exists():
                    logger.debug(f"Profile not found for deletion: {profile_id}")
                    return False

                # Delete files
                json_path.unlink(missing_ok=True)
                embedding_path.unlink(missing_ok=True)

                # Remove from cache
                if self.cache_enabled and profile_id in self.cache:
                    del self.cache[profile_id]

                logger.info(f"Deleted profile: {profile_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete profile {profile_id}: {e}")
                raise ProfileStorageError(f"Profile deletion failed: {e}")

    def profile_exists(self, profile_id: str) -> bool:
        """Check if profile exists

        Args:
            profile_id: Profile identifier

        Returns:
            True if profile exists

        Example:
            >>> if storage.profile_exists('uuid-1234'):
            ...     print("Profile exists")
        """
        # Check cache first
        if self.cache_enabled and profile_id in self.cache:
            return True

        # Check disk
        json_path, _ = self._get_profile_path(profile_id)
        return json_path.exists()

    def get_profile_count(self, user_id: Optional[str] = None) -> int:
        """Get count of profiles, optionally filtered by user_id

        Args:
            user_id: Optional user ID filter

        Returns:
            Number of profiles

        Example:
            >>> total = storage.get_profile_count()
            >>> user_count = storage.get_profile_count(user_id='user123')
        """
        profiles = self.list_profiles(user_id=user_id)
        return len(profiles)

    def get_storage_size(self) -> Dict[str, Any]:
        """Get storage statistics

        Returns:
            Dictionary with:
                - total_size_mb: Total size in MB
                - num_profiles: Number of profiles
                - storage_dir: Storage directory path

        Example:
            >>> stats = storage.get_storage_size()
            >>> print(f"Storage: {stats['total_size_mb']:.2f} MB")
        """
        total_size = 0
        num_profiles = 0

        for file_path in self.storage_dir.glob("*"):
            if file_path.is_file() and not file_path.name.endswith('.tmp'):
                total_size += file_path.stat().st_size
                if file_path.suffix == '.json':
                    num_profiles += 1

        return {
            'total_size_mb': total_size / (1024 ** 2),
            'num_profiles': num_profiles,
            'storage_dir': str(self.storage_dir)
        }

    def clear_cache(self):
        """Clear in-memory cache

        Example:
            >>> storage.clear_cache()
        """
        with self.lock:
            self.cache.clear()
            logger.debug("Cache cleared")

    def _validate_profile(self, profile: Dict[str, Any]) -> bool:
        """Validate profile structure

        Args:
            profile: Profile dictionary

        Returns:
            True if valid

        Raises:
            ProfileValidationError: If validation fails
        """
        # Check required fields
        required_fields = ['profile_id', 'embedding', 'created_at']
        for field in required_fields:
            if field not in profile:
                raise ProfileValidationError(f"Missing required field: {field}")

        # Validate profile_id
        if not isinstance(profile['profile_id'], str):
            raise ProfileValidationError("profile_id must be a string")

        # Validate embedding
        embedding = profile['embedding']
        if not isinstance(embedding, np.ndarray):
            raise ProfileValidationError("embedding must be a numpy array")

        if embedding.shape != (256,):
            raise ProfileValidationError(
                f"embedding must have shape (256,), got {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ProfileValidationError("embedding contains NaN or Inf values")

        # Validate created_at
        if not isinstance(profile['created_at'], str):
            raise ProfileValidationError("created_at must be a string")

        return True

    def _get_profile_path(self, profile_id: str) -> Tuple[Path, Path]:
        """Get file paths for profile

        Args:
            profile_id: Profile identifier

        Returns:
            Tuple of (json_path, embedding_path)
        """
        json_path = self.storage_dir / f"{profile_id}.json"
        embedding_path = self.storage_dir / f"{profile_id}_embedding.npy"
        return json_path, embedding_path

    def _atomic_write_json(self, path: Path, data: Dict):
        """Write JSON file atomically

        Args:
            path: Target file path
            data: Data to write
        """
        # Write to temp file
        temp_path = path.with_suffix('.json.tmp')

        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_path.replace(path)

        except Exception as e:
            # Clean up temp file on error
            temp_path.unlink(missing_ok=True)
            raise e

    def _atomic_write_numpy(self, path: Path, array: np.ndarray):
        """Write NumPy file atomically

        Args:
            path: Target file path
            array: Array to write
        """
        # Write to temp file
        temp_path = path.with_suffix('.npy.tmp')

        try:
            np.save(temp_path, array)

            # Atomic rename
            temp_path.replace(path)

        except Exception as e:
            # Clean up temp file on error
            temp_path.unlink(missing_ok=True)
            raise e

    def _enforce_cache_limit(self):
        """Enforce LRU cache size limit"""
        while len(self.cache) > self.cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Evicted from cache: {oldest_key}")

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization

        Args:
            obj: Object to convert

        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
