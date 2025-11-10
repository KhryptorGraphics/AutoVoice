"""
NSFW Detection using OpenNSFW2 for high-accuracy content moderation.

OpenNSFW2 provides 93%+ accuracy for NSFW content detection.
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import opennsfw2 as n2
    OPENNSFW2_AVAILABLE = True
except ImportError:
    OPENNSFW2_AVAILABLE = False
    logging.warning("OpenNSFW2 not available. NSFW detection will be disabled.")

logger = logging.getLogger(__name__)


class NSFWDetector:
    """High-accuracy NSFW content detector using OpenNSFW2."""
    
    def __init__(
        self,
        threshold: float = 0.7,
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize NSFW detector.
        
        Args:
            threshold: NSFW probability threshold (0.0-1.0). Default 0.7
            model_path: Optional custom model path
            use_gpu: Whether to use GPU acceleration
        """
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.available = OPENNSFW2_AVAILABLE
        
        if not self.available:
            logger.warning("OpenNSFW2 not available. Install with: pip install opennsfw2")
            return
        
        try:
            # OpenNSFW2 automatically downloads and caches the model
            logger.info("OpenNSFW2 initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenNSFW2: {e}")
            self.available = False
    
    def predict_image(self, image_path: str) -> Dict[str, float]:
        """
        Predict NSFW probability for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with 'nsfw_probability' and 'is_nsfw' keys
        """
        if not self.available:
            return {
                'nsfw_probability': 0.0,
                'is_nsfw': False,
                'error': 'OpenNSFW2 not available'
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get NSFW probability using OpenNSFW2
            nsfw_prob = n2.predict_image(image_path)
            
            result = {
                'nsfw_probability': float(nsfw_prob),
                'is_nsfw': nsfw_prob >= self.threshold,
                'threshold': self.threshold,
                'model': 'OpenNSFW2'
            }
            
            logger.info(f"NSFW detection: {nsfw_prob:.3f} (threshold: {self.threshold})")
            return result
            
        except Exception as e:
            logger.error(f"Error in NSFW detection: {e}")
            return {
                'nsfw_probability': 0.0,
                'is_nsfw': False,
                'error': str(e)
            }
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Batch predict NSFW probability for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        if not self.available:
            return [{
                'nsfw_probability': 0.0,
                'is_nsfw': False,
                'error': 'OpenNSFW2 not available'
            }] * len(image_paths)
        
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)
        
        return results
    
    def set_threshold(self, threshold: float):
        """Update NSFW detection threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        logger.info(f"NSFW threshold updated to {threshold}")

