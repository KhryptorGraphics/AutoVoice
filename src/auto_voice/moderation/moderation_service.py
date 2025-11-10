"""
Comprehensive Content Moderation Service.

Integrates NSFW detection, age verification, deepfake detection, profanity filtering,
database logging, and caching for complete content safety.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib

from .nsfw_detector import NSFWDetector
from .database import ModerationDatabase
from .cache_manager import CacheManager, BatchProcessor
from .age_verifier import AgeVerifier
from .deepfake_detector import DeepfakeDetector
from .profanity_filter import ProfanityFilter

logger = logging.getLogger(__name__)


class ModerationService:
    """
    Comprehensive content moderation service.
    
    Provides:
    - NSFW detection (OpenNSFW2 - 93%+ accuracy)
    - Age verification
    - Deepfake detection
    - Profanity filtering
    - Database audit logging
    - Redis caching
    - Batch processing
    """
    
    def __init__(
        self,
        nsfw_threshold: float = 0.7,
        age_threshold: int = 18,
        deepfake_threshold: float = 0.8,
        database_url: str = "sqlite:///moderation.db",
        redis_url: str = "redis://localhost:6379/0",
        enable_cache: bool = True,
        enable_database: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize moderation service.
        
        Args:
            nsfw_threshold: NSFW detection threshold (0.0-1.0)
            age_threshold: Minimum age requirement
            deepfake_threshold: Deepfake detection threshold (0.0-1.0)
            database_url: Database connection URL
            redis_url: Redis connection URL
            enable_cache: Enable caching
            enable_database: Enable database logging
            batch_size: Batch processing size
        """
        logger.info("Initializing comprehensive moderation service...")
        
        # Initialize components
        self.nsfw_detector = NSFWDetector(threshold=nsfw_threshold)
        self.age_verifier = AgeVerifier(min_age=age_threshold)
        self.deepfake_detector = DeepfakeDetector(threshold=deepfake_threshold)
        self.profanity_filter = ProfanityFilter()
        
        # Initialize database
        self.database = None
        if enable_database:
            self.database = ModerationDatabase(database_url)
        
        # Initialize cache
        self.cache = None
        if enable_cache:
            self.cache = CacheManager(redis_url, enable_cache=True)
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        
        logger.info("Moderation service initialized successfully")
    
    def moderate_image(
        self,
        image_path: str,
        user_id: Optional[str] = None,
        content_id: Optional[str] = None,
        check_nsfw: bool = True,
        check_age: bool = False,
        check_deepfake: bool = False
    ) -> Dict:
        """
        Moderate an image with comprehensive checks.
        
        Args:
            image_path: Path to image file
            user_id: User identifier
            content_id: Content identifier
            check_nsfw: Enable NSFW detection
            check_age: Enable age verification
            check_deepfake: Enable deepfake detection
            
        Returns:
            Moderation result dictionary
        """
        start_time = time.time()
        
        # Generate content ID if not provided
        if content_id is None:
            with open(image_path, 'rb') as f:
                content_id = hashlib.sha256(f.read()).hexdigest()
        
        # Check cache
        if self.cache:
            cached = self.cache.get('image_moderation', content_id)
            if cached:
                logger.info(f"Using cached moderation result for {content_id}")
                return cached
        
        result = {
            'content_id': content_id,
            'content_type': 'image',
            'is_safe': True,
            'flags': [],
            'checks': {}
        }
        
        # NSFW check
        if check_nsfw and self.nsfw_detector.available:
            nsfw_result = self.nsfw_detector.predict_image(image_path)
            result['checks']['nsfw'] = nsfw_result
            
            if nsfw_result.get('is_nsfw', False):
                result['is_safe'] = False
                result['flags'].append('nsfw_content')
            
            # Log to database
            if self.database:
                self.database.log_moderation_event(
                    event_type='nsfw_check',
                    content_type='image',
                    is_flagged=nsfw_result.get('is_nsfw', False),
                    confidence_score=nsfw_result.get('nsfw_probability', 0.0),
                    threshold=nsfw_result.get('threshold', 0.7),
                    detector_model='OpenNSFW2',
                    content_id=content_id,
                    user_id=user_id,
                    action='blocked' if nsfw_result.get('is_nsfw') else 'allowed'
                )
        
        # Age verification
        if check_age and self.age_verifier.available:
            age_result = self.age_verifier.verify_image(image_path)
            result['checks']['age'] = age_result
            
            if not age_result.get('is_adult', True):
                result['is_safe'] = False
                result['flags'].append('underage_content')
        
        # Deepfake detection
        if check_deepfake and self.deepfake_detector.available:
            deepfake_result = self.deepfake_detector.detect(image_path)
            result['checks']['deepfake'] = deepfake_result
            
            if deepfake_result.get('is_deepfake', False):
                result['is_safe'] = False
                result['flags'].append('deepfake_detected')
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        result['processing_time_ms'] = processing_time
        
        # Cache result
        if self.cache:
            self.cache.set('image_moderation', content_id, result, ttl=3600)
        
        logger.info(f"Image moderation complete: {content_id} - Safe: {result['is_safe']}")
        return result
    
    def moderate_text(
        self,
        text: str,
        user_id: Optional[str] = None,
        check_profanity: bool = True
    ) -> Dict:
        """
        Moderate text content.
        
        Args:
            text: Text to moderate
            user_id: User identifier
            check_profanity: Enable profanity filtering
            
        Returns:
            Moderation result dictionary
        """
        start_time = time.time()
        
        result = {
            'content_type': 'text',
            'is_safe': True,
            'flags': [],
            'checks': {}
        }
        
        # Profanity check
        if check_profanity:
            profanity_result = self.profanity_filter.check(text)
            result['checks']['profanity'] = profanity_result
            
            if profanity_result.get('has_profanity', False):
                result['is_safe'] = False
                result['flags'].append('profanity_detected')
                result['cleaned_text'] = profanity_result.get('cleaned_text', text)
        
        result['processing_time_ms'] = (time.time() - start_time) * 1000
        
        return result
    
    def moderate_batch(
        self,
        items: List[Dict],
        item_type: str = 'image'
    ) -> List[Dict]:
        """
        Moderate multiple items in batches.
        
        Args:
            items: List of items to moderate (each with 'path' or 'text' key)
            item_type: Type of items ('image' or 'text')
            
        Returns:
            List of moderation results
        """
        if item_type == 'image':
            return self.batch_processor.process_batch(
                items,
                lambda batch: [self.moderate_image(item['path']) for item in batch]
            )
        elif item_type == 'text':
            return self.batch_processor.process_batch(
                items,
                lambda batch: [self.moderate_text(item['text']) for item in batch]
            )
        else:
            raise ValueError(f"Unsupported item type: {item_type}")
    
    def get_statistics(self, days: int = 7) -> Dict:
        """Get moderation statistics."""
        stats = {
            'cache': self.cache.get_stats() if self.cache else {'available': False},
            'database': self.database.get_moderation_stats(days) if self.database else {}
        }
        
        return stats

