"""Mock object fixtures for external dependencies.

Provides mocks for file I/O, network operations, caching, and other
external dependencies to enable isolated unit testing.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, MagicMock, patch, mock_open
import io


# ============================================================================
# File System Mocks
# ============================================================================

@pytest.fixture
def mock_file_system(tmp_path: Path):
    """Mock file system for testing file I/O without actual disk operations.

    Provides in-memory file storage with read/write/exists operations.

    Examples:
        fs = mock_file_system
        fs.write('test.txt', 'Hello')
        content = fs.read('test.txt')
        assert fs.exists('test.txt')
    """
    class MockFileSystem:
        def __init__(self, base_path: Path):
            self.base_path = base_path
            self.files = {}  # In-memory file storage
            self.access_log = []  # Track file access

        def write(self, filepath: str, content: Any, mode: str = 'w'):
            """Write content to mock file.

            Args:
                filepath: Relative file path
                content: Content to write
                mode: Write mode ('w', 'wb', etc.)
            """
            full_path = str(self.base_path / filepath)
            self.files[full_path] = content
            self.access_log.append(('write', full_path, mode))

        def read(self, filepath: str, mode: str = 'r'):
            """Read content from mock file.

            Args:
                filepath: Relative file path
                mode: Read mode ('r', 'rb', etc.)

            Returns:
                File content

            Raises:
                FileNotFoundError: If file doesn't exist
            """
            full_path = str(self.base_path / filepath)
            if full_path not in self.files:
                raise FileNotFoundError(f"No such file: {filepath}")

            self.access_log.append(('read', full_path, mode))
            return self.files[full_path]

        def exists(self, filepath: str) -> bool:
            """Check if file exists.

            Args:
                filepath: Relative file path

            Returns:
                True if file exists
            """
            full_path = str(self.base_path / filepath)
            return full_path in self.files

        def delete(self, filepath: str):
            """Delete mock file.

            Args:
                filepath: Relative file path
            """
            full_path = str(self.base_path / filepath)
            if full_path in self.files:
                del self.files[full_path]
                self.access_log.append(('delete', full_path))

        def list_files(self) -> List[str]:
            """List all mock files.

            Returns:
                List of file paths
            """
            return list(self.files.keys())

        def clear(self):
            """Clear all mock files."""
            self.files.clear()
            self.access_log.clear()

        def get_access_log(self) -> List[tuple]:
            """Get file access log.

            Returns:
                List of (operation, filepath, mode) tuples
            """
            return self.access_log.copy()

    return MockFileSystem(tmp_path)


@pytest.fixture
def mock_audio_loader():
    """Mock audio file loader that returns synthetic audio.

    Avoids requiring actual audio files for testing.

    Examples:
        loader = mock_audio_loader
        audio, sr = loader.load('song.wav')
        # Returns synthetic audio instead of loading file
    """
    class MockAudioLoader:
        def __init__(self):
            self.default_sr = 22050
            self.default_duration = 3.0
            self.load_count = 0

        def load(
            self,
            filepath: str,
            sr: Optional[int] = None,
            mono: bool = True,
            duration: Optional[float] = None
        ) -> tuple:
            """Mock audio loading.

            Args:
                filepath: Path to audio file (ignored)
                sr: Sample rate (default 22050)
                mono: Return mono audio
                duration: Audio duration in seconds

            Returns:
                Tuple of (audio, sample_rate)
            """
            self.load_count += 1

            sr = sr if sr is not None else self.default_sr
            dur = duration if duration is not None else self.default_duration

            # Generate synthetic audio
            num_samples = int(sr * dur)
            t = np.linspace(0, dur, num_samples)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            if not mono:
                # Return stereo
                audio = np.stack([audio, audio * 0.8])

            return audio, sr

        def save(self, filepath: str, audio: np.ndarray, sr: int):
            """Mock audio saving.

            Args:
                filepath: Output path (ignored)
                audio: Audio data
                sr: Sample rate
            """
            pass  # No-op for mock

    return MockAudioLoader()


# ============================================================================
# Network Mocks
# ============================================================================

@pytest.fixture
def mock_network_client():
    """Mock network client for testing API calls without actual network access.

    Examples:
        client = mock_network_client
        client.set_response('/api/models', {'status': 'ok'})
        response = client.get('/api/models')
    """
    class MockNetworkClient:
        def __init__(self):
            self.responses = {}
            self.request_log = []
            self.default_status = 200

        def set_response(self, endpoint: str, data: Any, status: int = 200):
            """Set mock response for endpoint.

            Args:
                endpoint: API endpoint
                data: Response data
                status: HTTP status code
            """
            self.responses[endpoint] = {
                'data': data,
                'status': status
            }

        def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
            """Mock GET request.

            Args:
                endpoint: API endpoint
                **kwargs: Request parameters

            Returns:
                Mock response dict
            """
            self.request_log.append(('GET', endpoint, kwargs))

            if endpoint in self.responses:
                return self.responses[endpoint]

            return {
                'data': {},
                'status': 404,
                'error': 'Not found'
            }

        def post(self, endpoint: str, data: Any = None, **kwargs) -> Dict[str, Any]:
            """Mock POST request.

            Args:
                endpoint: API endpoint
                data: Request data
                **kwargs: Request parameters

            Returns:
                Mock response dict
            """
            self.request_log.append(('POST', endpoint, data, kwargs))

            if endpoint in self.responses:
                return self.responses[endpoint]

            return {
                'data': {'received': data},
                'status': self.default_status
            }

        def get_request_log(self) -> List[tuple]:
            """Get request history.

            Returns:
                List of request tuples
            """
            return self.request_log.copy()

        def reset(self):
            """Reset mock client."""
            self.responses.clear()
            self.request_log.clear()

    return MockNetworkClient()


# ============================================================================
# Cache Mocks
# ============================================================================

@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing caching logic without Redis/Memcached.

    Provides in-memory caching with TTL support.

    Examples:
        cache = mock_cache_manager
        cache.set('key', value, ttl=60)
        value = cache.get('key')
    """
    import time

    class MockCacheManager:
        def __init__(self):
            self.cache = {}
            self.hit_count = 0
            self.miss_count = 0
            self.set_count = 0

        def get(self, key: str) -> Optional[Any]:
            """Get value from cache.

            Args:
                key: Cache key

            Returns:
                Cached value or None if not found/expired
            """
            if key not in self.cache:
                self.miss_count += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if entry['expires_at'] is not None:
                if time.time() > entry['expires_at']:
                    del self.cache[key]
                    self.miss_count += 1
                    return None

            self.hit_count += 1
            return entry['value']

        def set(self, key: str, value: Any, ttl: Optional[int] = None):
            """Set cache value.

            Args:
                key: Cache key
                value: Value to cache
                ttl: Time-to-live in seconds
            """
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl

            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'set_at': time.time()
            }

            self.set_count += 1

        def delete(self, key: str):
            """Delete cache entry.

            Args:
                key: Cache key
            """
            if key in self.cache:
                del self.cache[key]

        def clear(self):
            """Clear all cache entries."""
            self.cache.clear()

        def get_stats(self) -> Dict[str, Any]:
            """Get cache statistics.

            Returns:
                Dict with cache stats
            """
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

            return {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'sets': self.set_count,
                'hit_rate': hit_rate,
                'size': len(self.cache)
            }

    return MockCacheManager()


# ============================================================================
# Database Mocks
# ============================================================================

@pytest.fixture
def mock_database():
    """Mock database for testing without actual DB connection.

    Provides basic CRUD operations with in-memory storage.

    Examples:
        db = mock_database
        db.insert('users', {'name': 'Alice', 'age': 30})
        users = db.select('users', where={'age': 30})
    """
    class MockDatabase:
        def __init__(self):
            self.tables = {}
            self.query_log = []

        def insert(self, table: str, data: Dict[str, Any]) -> int:
            """Insert record.

            Args:
                table: Table name
                data: Record data

            Returns:
                Record ID
            """
            if table not in self.tables:
                self.tables[table] = []

            record = data.copy()
            record['id'] = len(self.tables[table]) + 1

            self.tables[table].append(record)
            self.query_log.append(('INSERT', table, data))

            return record['id']

        def select(
            self,
            table: str,
            where: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            """Select records.

            Args:
                table: Table name
                where: Filter conditions

            Returns:
                List of matching records
            """
            if table not in self.tables:
                return []

            self.query_log.append(('SELECT', table, where))

            records = self.tables[table]

            if where:
                # Simple filtering
                records = [
                    r for r in records
                    if all(r.get(k) == v for k, v in where.items())
                ]

            return records.copy()

        def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]):
            """Update records.

            Args:
                table: Table name
                data: Update data
                where: Filter conditions
            """
            if table not in self.tables:
                return

            self.query_log.append(('UPDATE', table, data, where))

            for record in self.tables[table]:
                if all(record.get(k) == v for k, v in where.items()):
                    record.update(data)

        def delete(self, table: str, where: Dict[str, Any]):
            """Delete records.

            Args:
                table: Table name
                where: Filter conditions
            """
            if table not in self.tables:
                return

            self.query_log.append(('DELETE', table, where))

            self.tables[table] = [
                r for r in self.tables[table]
                if not all(r.get(k) == v for k, v in where.items())
            ]

        def get_query_log(self) -> List[tuple]:
            """Get query history.

            Returns:
                List of query tuples
            """
            return self.query_log.copy()

    return MockDatabase()


__all__ = [
    'mock_file_system',
    'mock_audio_loader',
    'mock_network_client',
    'mock_cache_manager',
    'mock_database',
]
