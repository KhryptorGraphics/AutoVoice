"""Shared feature-dimension contract for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_CONTENT_DIM = 768
DEFAULT_PITCH_DIM = 768
DEFAULT_SPEAKER_DIM = 256


@dataclass(frozen=True)
class FeatureContract:
    """Decoder feature dimensions consumed by a voice-conversion model."""

    content_dim: int = DEFAULT_CONTENT_DIM
    pitch_dim: int = DEFAULT_PITCH_DIM
    speaker_dim: int = DEFAULT_SPEAKER_DIM


def feature_contract_from_model(
    model: Any,
    config: Mapping[str, Any] | None = None,
) -> FeatureContract:
    """Resolve the feature contract from config first, then model attributes."""
    payload = dict(config or {})
    return FeatureContract(
        content_dim=int(payload.get("content_dim", getattr(model, "content_dim", DEFAULT_CONTENT_DIM))),
        pitch_dim=int(payload.get("pitch_dim", getattr(model, "pitch_dim", DEFAULT_PITCH_DIM))),
        speaker_dim=int(payload.get("speaker_dim", getattr(model, "speaker_dim", DEFAULT_SPEAKER_DIM))),
    )
