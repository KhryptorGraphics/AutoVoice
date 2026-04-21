"""Helpers for canonical training artifact serialization."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

import torch


LORA_METADATA_KEY = "__autovoice_lora_metadata__"
LORA_FORMAT_VERSION = 2


def build_lora_checkpoint_payload(
    adapter_state: Mapping[str, torch.Tensor],
    *,
    config: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical on-disk payload for LoRA checkpoints.

    The runtime consumes the flat tensor keys directly, while metadata stays
    under a reserved key so loaders can round-trip config without changing the
    adapter application path.
    """
    payload: Dict[str, Any] = {
        key: value.detach().cpu()
        for key, value in adapter_state.items()
    }
    payload[LORA_METADATA_KEY] = {
        "format_version": LORA_FORMAT_VERSION,
        "config": dict(config or {}),
        "metadata": dict(metadata or {}),
    }
    return payload


def extract_lora_state_dict(payload: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract a flat LoRA state dict from canonical or legacy payloads."""
    if "adapters" in payload:
        flattened: Dict[str, torch.Tensor] = {}
        for name, state in dict(payload["adapters"]).items():
            if "lora_A" in state:
                flattened[f"{name}.lora_A"] = state["lora_A"]
            if "lora_B" in state:
                flattened[f"{name}.lora_B"] = state["lora_B"]
        return flattened

    return {
        key: value
        for key, value in payload.items()
        if key != LORA_METADATA_KEY and isinstance(value, torch.Tensor)
    }


def extract_lora_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Load adapter metadata from canonical or legacy payloads."""
    if LORA_METADATA_KEY in payload:
        metadata = payload.get(LORA_METADATA_KEY, {})
        if isinstance(metadata, MutableMapping):
            return dict(metadata)
        return {}

    if "config" in payload:
        config = dict(payload.get("config", {}))
        adapters = dict(payload.get("adapters", {}))
        if adapters:
            first_adapter = next(iter(adapters.values()))
            config.setdefault("rank", first_adapter.get("rank"))
            config.setdefault("alpha", first_adapter.get("alpha"))
        return {
            "format_version": 1,
            "config": config,
            "metadata": {},
        }

    return {
        "format_version": 0,
        "config": {},
        "metadata": {},
    }
