"""
Caching and batch processing for content moderation.

Provides Redis-based caching and batch processing capabilities to improve performance.
"""

import logging
import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Caching will be disabled.")

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager for moderation results."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        enable_cache: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds (default: 1 hour)
            enable_cache: Whether to enable caching
        """
        self.available = REDIS_AVAILABLE and enable_cache
        self.default_ttl = default_ttl
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install with: pip install redis")
            return
        
        if not enable_cache:
            logger.info("Caching disabled by configuration")
            return
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Cache manager initialized: {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.available = False
    
    def _generate_key(self, prefix: str, content_id: str) -> str:
        """Generate cache key."""
        return f"moderation:{prefix}:{content_id}"
    
    def _hash_content(self, content: bytes) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content).hexdigest()
    
    def get(self, prefix: str, content_id: str) -> Optional[Dict]:
        """
        Get cached moderation result.
        
        Args:
            prefix: Cache key prefix (e.g., 'nsfw', 'age_verify')
            content_id: Content identifier
            
        Returns:
            Cached result dict or None
        """
        if not self.available:
            return None
        
        try:
            key = self._generate_key(prefix, content_id)
            cached = self.redis_client.get(key)
            
            if cached:
                logger.debug(f"Cache hit: {key}")
                return json.loads(cached)
            
            logger.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        prefix: str,
        content_id: str,
        result: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache moderation result.
        
        Args:
            prefix: Cache key prefix
            content_id: Content identifier
            result: Result to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if cached successfully
        """
        if not self.available:
            return False
        
        try:
            key = self._generate_key(prefix, content_id)
            ttl = ttl or self.default_ttl
            
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(result)
            )
            
            logger.debug(f"Cached result: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, prefix: str, content_id: str) -> bool:
        """Delete cached result."""
        if not self.available:
            return False
        
        try:
            key = self._generate_key(prefix, content_id)
            self.redis_client.delete(key)
            logger.debug(f"Deleted cache: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_prefix(self, prefix: str) -> int:
        """
        Clear all cached results with given prefix.
        
        Returns:
            Number of keys deleted
        """
        if not self.available:
            return 0
        
        try:
            pattern = f"moderation:{prefix}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cached results for prefix: {prefix}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.available:
            return {'available': False}
        
        try:
            info = self.redis_client.info('stats')
            
            return {
                'available': True,
                'total_keys': self.redis_client.dbsize(),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'available': False, 'error': str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


class BatchProcessor:
    """Batch processor for efficient moderation of multiple items."""
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
        """
        self.batch_size = batch_size
        logger.info(f"Batch processor initialized (batch_size={batch_size})")
    
    def process_batch(
        self,
        items: List[Any],
        processor_func: callable,
        **kwargs
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            processor_func: Function to process each batch
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processed results
        """
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = processor_func(batch, **kwargs)
            results.extend(batch_results)
        
        logger.info(f"Processed {len(items)} items in {len(results) // self.batch_size + 1} batches")
        return results

