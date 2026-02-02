"""Adapter Bridge for HQ LoRA Adapters.

Bridges the architecture gap between:
- Trained HQ adapters: Standalone 6-layer MLP with keys 'lora_0_A', 'lora_0_B', etc.
- Expected format: Layer-injection format '{module}.adapter.lora_A'

The HQVoiceLoRAAdapter is a content feature transformer that takes ContentVec
features and applies speaker-specific adaptation before the decoder.

Usage:
    bridge = HQLoRAAdapterBridge(device='cuda')
    bridge.load_adapter('profile-uuid')

    # In conversion pipeline:
    content_features = content_encoder.encode(audio)  # [B, T, 768]
    adapted_features = bridge.transform(content_features, speaker_embedding)
    mel = decoder.infer(adapted_features, pitch, speaker)
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# HQ LoRA Architecture (from train_hq_lora.py)
# ============================================================================

class LoRALayer(nn.Module):
    """High-Quality LoRA layer with scaled initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        alpha: float = 256.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices with careful initialization
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Kaiming initialization for better gradient flow
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Start with zero adaptation

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        lora_out = self.dropout(lora_out)
        return base_output + self.scaling * lora_out

    def get_delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


class HQVoiceLoRAAdapter(nn.Module):
    """High-Quality Voice LoRA Adapter.

    Architecture: 768 -> 1024 -> 1024 -> 1024 -> 1024 -> 1024 -> 768
    With residual connections and layer normalization.

    This is a content feature transformer that applies speaker-specific
    adaptation to ContentVec features before passing to the decoder.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 768,
        lora_rank: int = 128,
        lora_alpha: float = 256.0,
        dropout: float = 0.05,
        num_layers: int = 6,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank
        self.num_layers = num_layers

        # Build layer dimensions with smooth transition
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        # Base layers (frozen during LoRA training)
        self.base_layers = nn.ModuleList()
        self.lora_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            # Base linear layer
            base = nn.Linear(dims[i], dims[i + 1])
            self.base_layers.append(base)

            # LoRA adaptation layer
            lora = LoRALayer(
                dims[i], dims[i + 1],
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=dropout,
            )
            self.lora_layers.append(lora)

            # Layer normalization for stability
            self.layer_norms.append(nn.LayerNorm(dims[i + 1]))

        # Speaker conditioning projection
        self.speaker_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Residual projections for dimension mismatches
        self.residual_projs = nn.ModuleList()
        for i in range(num_layers):
            if dims[i] != dims[i + 1]:
                self.residual_projs.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            else:
                self.residual_projs.append(nn.Identity())

        # Freeze base layers
        for layer in self.base_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        content: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform content features with speaker-specific adaptation.

        Args:
            content: [B, T, 768] ContentVec features
            speaker_embedding: Optional [B, 768] or [B, 256] speaker embedding

        Returns:
            [B, T, 768] Adapted content features
        """
        x = content

        for i, (base, lora, ln, res_proj) in enumerate(zip(
            self.base_layers, self.lora_layers, self.layer_norms, self.residual_projs
        )):
            # Residual connection
            residual = res_proj(x)

            # Base + LoRA
            base_out = base(x)
            x = lora(x, base_out)

            # Speaker conditioning after first layer
            if i == 0 and speaker_embedding is not None:
                # Handle different embedding dimensions
                if speaker_embedding.dim() == 1:
                    speaker_embedding = speaker_embedding.unsqueeze(0)

                # Project speaker embedding if needed
                if speaker_embedding.shape[-1] != self.output_dim:
                    # Create a simple projection if needed
                    if not hasattr(self, '_spk_input_proj'):
                        self._spk_input_proj = nn.Linear(
                            speaker_embedding.shape[-1],
                            self.output_dim
                        ).to(speaker_embedding.device)
                        for param in self._spk_input_proj.parameters():
                            param.requires_grad = False
                    speaker_embedding = self._spk_input_proj(speaker_embedding)

                spk = self.speaker_proj(speaker_embedding).unsqueeze(1)
                x = x + spk.expand(-1, x.size(1), -1)

            # Normalization + activation + residual
            x = ln(x)
            if i < len(self.base_layers) - 1:
                x = F.gelu(x)
                x = x + 0.1 * residual  # Scaled residual

        return x

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA weights in the training format."""
        state = {}
        for i, lora in enumerate(self.lora_layers):
            state[f'lora_{i}_A'] = lora.lora_A.data.clone()
            state[f'lora_{i}_B'] = lora.lora_B.data.clone()
        return state

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]):
        """Load LoRA weights from training format."""
        for i, lora in enumerate(self.lora_layers):
            if f'lora_{i}_A' in state:
                lora.lora_A.data = state[f'lora_{i}_A'].to(lora.lora_A.device)
                lora.lora_B.data = state[f'lora_{i}_B'].to(lora.lora_B.device)


# ============================================================================
# Adapter Bridge
# ============================================================================

# Default HQ configuration
DEFAULT_HQ_CONFIG = {
    'input_dim': 768,
    'hidden_dim': 1024,
    'output_dim': 768,
    'num_layers': 6,
    'lora_rank': 128,
    'lora_alpha': 256.0,
    'dropout': 0.05,
}


@dataclass
class AdapterBridgeConfig:
    """Configuration for adapter bridge."""
    adapters_dir: Path = Path("data/trained_models/hq")
    profiles_dir: Path = Path("data/voice_profiles")
    cache_size: int = 3  # Number of adapters to keep in memory
    device: str = "cuda"


class HQLoRAAdapterBridge:
    """Bridge for loading and applying HQ LoRA adapters.

    This bridge loads trained HQ LoRA adapters (which have their own
    architecture) and applies them to content features before the decoder.

    Unlike the layer-injection approach, this uses a standalone adapter
    module that transforms features directly.

    Usage:
        bridge = HQLoRAAdapterBridge(device='cuda')

        # Load adapter for a profile
        if bridge.has_adapter(profile_id):
            bridge.load_adapter(profile_id)

            # Transform content features
            content = content_encoder.encode(audio)  # [B, T, 768]
            adapted = bridge.transform(content, speaker_embedding)
            mel = decoder.infer(adapted, pitch, speaker)
    """

    def __init__(self, config: Optional[AdapterBridgeConfig] = None):
        self.config = config or AdapterBridgeConfig()
        self.device = torch.device(self.config.device)

        # Cache for loaded adapters
        self._adapters: Dict[str, HQVoiceLoRAAdapter] = {}
        self._current_profile: Optional[str] = None
        self._current_adapter: Optional[HQVoiceLoRAAdapter] = None

        # Ensure directories exist
        self.config.adapters_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"HQLoRAAdapterBridge initialized")
        logger.info(f"  Adapters dir: {self.config.adapters_dir}")
        logger.info(f"  Device: {self.device}")

    def list_adapters(self) -> list[str]:
        """List all available HQ adapter profile IDs."""
        adapters = []
        for path in self.config.adapters_dir.glob("*_hq_lora.pt"):
            profile_id = path.stem.replace("_hq_lora", "")
            adapters.append(profile_id)
        return adapters

    def has_adapter(self, profile_id: str) -> bool:
        """Check if an HQ adapter exists for the given profile."""
        adapter_path = self.config.adapters_dir / f"{profile_id}_hq_lora.pt"
        return adapter_path.exists()

    def get_adapter_path(self, profile_id: str) -> Optional[Path]:
        """Get path to adapter file if it exists."""
        path = self.config.adapters_dir / f"{profile_id}_hq_lora.pt"
        return path if path.exists() else None

    def load_adapter(self, profile_id: str) -> HQVoiceLoRAAdapter:
        """Load HQ adapter for a profile.

        Args:
            profile_id: UUID of the voice profile

        Returns:
            Loaded HQVoiceLoRAAdapter ready for inference

        Raises:
            FileNotFoundError: If adapter doesn't exist
            RuntimeError: If loading fails
        """
        # Check cache first
        if profile_id in self._adapters:
            self._current_profile = profile_id
            self._current_adapter = self._adapters[profile_id]
            logger.debug(f"Adapter {profile_id} loaded from cache")
            return self._current_adapter

        # Load from disk
        adapter_path = self.config.adapters_dir / f"{profile_id}_hq_lora.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"No HQ adapter found for profile: {profile_id}")

        logger.info(f"Loading HQ adapter from {adapter_path}")

        # Load checkpoint
        checkpoint = torch.load(
            adapter_path,
            map_location=self.device,
            weights_only=False
        )

        # Get config from checkpoint or use default
        config = checkpoint.get('config', DEFAULT_HQ_CONFIG)

        # Create adapter with config
        adapter = HQVoiceLoRAAdapter(
            input_dim=config.get('input_dim', 768),
            hidden_dim=config.get('hidden_dim', 1024),
            output_dim=config.get('output_dim', 768),
            num_layers=config.get('num_layers', 6),
            lora_rank=config.get('lora_rank', 128),
            lora_alpha=config.get('lora_alpha', 256.0),
            dropout=config.get('dropout', 0.05),
        )
        adapter.to(self.device)

        # Load LoRA weights
        lora_state = checkpoint.get('lora_state', {})
        if not lora_state:
            raise RuntimeError(f"No lora_state found in checkpoint for {profile_id}")

        adapter.load_lora_state_dict(lora_state)

        # Set to inference mode
        adapter.train(False)

        # Manage cache size
        if len(self._adapters) >= self.config.cache_size:
            # Evict oldest (first) entry
            oldest = next(iter(self._adapters))
            del self._adapters[oldest]
            logger.debug(f"Evicted adapter {oldest} from cache")

        # Cache and set current
        self._adapters[profile_id] = adapter
        self._current_profile = profile_id
        self._current_adapter = adapter

        logger.info(f"Loaded HQ adapter for profile {profile_id}")
        return adapter

    def get_current_adapter(self) -> Optional[HQVoiceLoRAAdapter]:
        """Get the currently loaded adapter."""
        return self._current_adapter

    def get_current_profile(self) -> Optional[str]:
        """Get the currently loaded profile ID."""
        return self._current_profile

    def transform(
        self,
        content: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform content features using the current adapter.

        Args:
            content: [B, T, 768] ContentVec features
            speaker_embedding: Optional [B, 256] or [256] speaker embedding

        Returns:
            [B, T, 768] Adapted content features

        Raises:
            RuntimeError: If no adapter is loaded
        """
        if self._current_adapter is None:
            raise RuntimeError("No adapter loaded. Call load_adapter() first.")

        with torch.no_grad():
            return self._current_adapter(
                content.to(self.device),
                speaker_embedding.to(self.device) if speaker_embedding is not None else None,
            )

    def clear(self) -> None:
        """Clear the current adapter."""
        self._current_profile = None
        self._current_adapter = None
        logger.debug("Current adapter cleared")

    def clear_cache(self) -> None:
        """Clear all cached adapters."""
        self._adapters.clear()
        self._current_profile = None
        self._current_adapter = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Adapter cache cleared")

    def get_adapter_info(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about an adapter."""
        adapter_path = self.config.adapters_dir / f"{profile_id}_hq_lora.pt"
        if not adapter_path.exists():
            return None

        checkpoint = torch.load(
            adapter_path,
            map_location='cpu',
            weights_only=False
        )

        return {
            'profile_id': profile_id,
            'artist': checkpoint.get('artist', 'Unknown'),
            'config': checkpoint.get('config', DEFAULT_HQ_CONFIG),
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'precision': checkpoint.get('precision', 'fp32'),
            'trained_on': checkpoint.get('trained_on', ''),
        }


# Global instance for shared use
_global_bridge: Optional[HQLoRAAdapterBridge] = None


def get_hq_adapter_bridge() -> HQLoRAAdapterBridge:
    """Get or create global HQLoRAAdapterBridge instance."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = HQLoRAAdapterBridge()
    return _global_bridge
