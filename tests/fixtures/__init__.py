"""Test fixtures for AutoVoice E2E testing."""

from .multi_speaker_fixtures import (
    create_multi_speaker_audio,
    create_synthetic_multi_speaker,
    MultiSpeakerFixture,
    SpeakerInfo,
)

__all__ = [
    "create_multi_speaker_audio",
    "create_synthetic_multi_speaker",
    "MultiSpeakerFixture",
    "SpeakerInfo",
]
