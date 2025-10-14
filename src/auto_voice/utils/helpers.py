"""String, math, and general helper utilities for AutoVoice."""

import re
import math
import string
import hashlib
import time
import random
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StringUtils:
    """String manipulation and formatting utilities."""
    
    @staticmethod
    def sanitize_filename(filename: str, replacement: str = "_") -> str:
        """
        Sanitize a string for use as a filename.
        
        Args:
            filename: Original filename
            replacement: Character to replace invalid characters
            
        Returns:
            Sanitized filename
        """
        # Remove/replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, replacement)
        
        # Remove control characters
        filename = ''.join(c for c in filename if ord(c) >= 32)
        
        # Trim whitespace and dots
        filename = filename.strip(' .')
        
        # Ensure not empty
        if not filename:
            filename = "unnamed"
        
        # Limit length
        if len(filename) > 255:
            name, ext = Path(filename).stem, Path(filename).suffix
            max_name_len = 255 - len(ext)
            filename = name[:max_name_len] + ext
        
        return filename
    
    @staticmethod
    def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate string to maximum length with suffix.
        
        Args:
            text: Text to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated string
        """
        if len(text) <= max_length:
            return text
        
        if max_length <= len(suffix):
            return suffix[:max_length]
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def camel_to_snake(text: str) -> str:
        """Convert camelCase to snake_case."""
        # Insert underscore before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        # Insert underscore before uppercase letters followed by lowercase
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
        """Convert snake_case to camelCase."""
        components = text.split('_')
        if capitalize_first:
            return ''.join(word.capitalize() for word in components)
        else:
            return components[0] + ''.join(word.capitalize() for word in components[1:])
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract all numbers from a string."""
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches if match]
    
    @staticmethod
    def format_bytes(size_bytes: int) -> str:
        """Format bytes into human readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"
    
    @staticmethod
    def generate_id(length: int = 8, charset: str = None) -> str:
        """
        Generate a random ID string.
        
        Args:
            length: Length of the ID
            charset: Character set to use (default: alphanumeric)
            
        Returns:
            Random ID string
        """
        if charset is None:
            charset = string.ascii_letters + string.digits
        
        return ''.join(random.choice(charset) for _ in range(length))
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'md5') -> str:
        """
        Generate hash of a string.
        
        Args:
            text: Text to hash
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            Hex digest of the hash
        """
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hasher.update(text.encode('utf-8'))
        return hasher.hexdigest()


class MathUtils:
    """Mathematical utilities and helper functions."""
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(value, max_val))
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b by factor t."""
        return a + t * (b - a)
    
    @staticmethod
    def smooth_step(x: float) -> float:
        """Smooth step function (3x^2 - 2x^3)."""
        x = MathUtils.clamp(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)
    
    @staticmethod
    def db_to_linear(db: float) -> float:
        """Convert decibels to linear scale."""
        return 10 ** (db / 20)
    
    @staticmethod
    def linear_to_db(linear: float, min_db: float = -80.0) -> float:
        """Convert linear scale to decibels."""
        if linear <= 0:
            return min_db
        return 20 * math.log10(linear)
    
    @staticmethod
    def rms(values: List[float]) -> float:
        """Calculate Root Mean Square of values."""
        if not values:
            return 0.0
        return math.sqrt(sum(x * x for x in values) / len(values))
    
    @staticmethod
    def normalize_range(
        value: float,
        from_min: float,
        from_max: float,
        to_min: float = 0.0,
        to_max: float = 1.0
    ) -> float:
        """Normalize value from one range to another."""
        if from_max == from_min:
            return to_min
        
        # Normalize to 0-1
        normalized = (value - from_min) / (from_max - from_min)
        # Scale to target range
        return to_min + normalized * (to_max - to_min)
    
    @staticmethod
    def moving_average(values: List[float], window_size: int) -> List[float]:
        """Calculate moving average with specified window size."""
        if window_size <= 0 or not values:
            return values
        
        result = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window = values[start_idx:i + 1]
            result.append(sum(window) / len(window))
        
        return result
    
    @staticmethod
    def round_to_nearest(value: float, nearest: float) -> float:
        """Round value to nearest multiple."""
        return round(value / nearest) * nearest
    
    @staticmethod
    def is_power_of_2(n: int) -> bool:
        """Check if number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0
    
    @staticmethod
    def next_power_of_2(n: int) -> int:
        """Find the next power of 2 greater than or equal to n."""
        if n <= 0:
            return 1
        
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Calculate Greatest Common Divisor."""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Calculate Least Common Multiple."""
        return abs(a * b) // MathUtils.gcd(a, b)


class ValidationUtils:
    """Input validation utilities."""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return re.match(pattern, url) is not None
    
    @staticmethod
    def is_valid_path(path: str) -> bool:
        """Validate if path is syntactically valid."""
        try:
            Path(path)
            return True
        except (ValueError, OSError):
            return False
    
    @staticmethod
    def validate_range(
        value: Union[int, float],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        inclusive: bool = True
    ) -> bool:
        """Validate if value is within specified range."""
        if min_val is not None:
            if inclusive and value < min_val:
                return False
            elif not inclusive and value <= min_val:
                return False
        
        if max_val is not None:
            if inclusive and value > max_val:
                return False
            elif not inclusive and value >= max_val:
                return False
        
        return True
    
    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Validate if value is of expected type."""
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_dict_keys(
        data: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate dictionary has required keys and only allowed keys.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required keys
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")
        
        # Check for unexpected keys
        if optional_keys is not None:
            allowed_keys = set(required_keys + optional_keys)
            for key in data:
                if key not in allowed_keys:
                    errors.append(f"Unexpected key: {key}")
        
        return len(errors) == 0, errors


class RetryUtils:
    """Retry and backoff utilities."""
    
    @staticmethod
    def exponential_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple = (Exception,)
    ) -> Any:
        """
        Execute function with exponential backoff retry.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for delay
            exceptions: Tuple of exceptions to catch and retry
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        delay = base_delay
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    @staticmethod
    def retry_with_jitter(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.1,
        exceptions: Tuple = (Exception,)
    ) -> Any:
        """Execute function with jittered retry delay."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    # Calculate delay with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = delay * jitter_factor * random.random()
                    total_delay = delay + jitter
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.2f}s")
                    time.sleep(total_delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception


class CacheUtils:
    """Simple in-memory caching utilities."""
    
    def __init__(self, max_size: int = 128, ttl: Optional[float] = None):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self._cache:
            return None
        
        # Check TTL
        if self.ttl is not None:
            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp > self.ttl:
                self.remove(key)
                return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        # Remove oldest item if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = min(self._timestamps.keys(), key=self._timestamps.get)
            self.remove(oldest_key)
        
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Convenience functions
def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value for division by zero."""
    return a / b if b != 0 else default

def safe_log(x: float, base: float = math.e, default: float = float('-inf')) -> float:
    """Safe logarithm with default value for invalid input."""
    if x <= 0:
        return default
    return math.log(x, base)

def flatten_dict(d: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, dict):
            flattened = flatten_dict(v, separator)
            items.extend((f"{new_key}{separator}{sub_k}", sub_v) for sub_k, sub_v in flattened.items())
        else:
            items.append((new_key, v))
    return dict(items)


# Export all classes and functions
__all__ = [
    'StringUtils',
    'MathUtils',
    'ValidationUtils',
    'RetryUtils',
    'CacheUtils',
    'ensure_dir',
    'safe_divide',
    'safe_log',
    'flatten_dict'
]