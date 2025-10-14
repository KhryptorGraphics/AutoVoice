"""Web interface for AutoVoice."""

from .app import create_app
from .api import VoiceAPI

__all__ = ['create_app', 'VoiceAPI']