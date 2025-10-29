"""
Quality evaluation system for voice conversion.

This package provides comprehensive evaluation tools for assessing the quality
of singing voice conversion systems, including pitch accuracy, speaker similarity,
naturalness, and intelligibility metrics.
"""

from .evaluator import VoiceConversionEvaluator

__all__ = ['VoiceConversionEvaluator']
