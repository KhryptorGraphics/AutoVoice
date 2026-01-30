"""CoMoSVC consistency model decoder for singing voice conversion.

Implements a consistency model decoder based on CoMoSVC (ISCSLP 2024).
Uses Bidirectional Dilated Convolutions (BiDilConv) conditioned on
content features (768-dim ContentVec), pitch embeddings (256-dim),
and speaker embeddings (256-dim mel-statistics).

Key design:
- Consistency distillation enables 1-step inference (matches 50-step diffusion)
- BiDilConv captures long-range temporal patterns in mel spectrograms
- Speaker conditioning via FiLM (Feature-wise Linear Modulation)
- Multi-step inference available for higher quality when latency allows
- LoRA injection for per-voice fine-tuning
- No fallback: raises RuntimeError on failure
"""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BiDilConv(nn.Module):
    """Bidirectional Dilated Convolution block.

    Uses exponentially increasing dilation rates for large receptive field
    while maintaining temporal resolution. Each layer doubles the dilation.

    Architecture: input → [DilConv → GELU → DilConv → residual] × n_layers
    """

    def __init__(self, channels: int = 256, kernel_size: int = 3,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, channels * 2, kernel_size,
                          dilation=dilation, padding=padding),
                nn.GLU(dim=1),  # Halves channels back to `channels`
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 1),
            ))

        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through dilated conv layers with residual connections.

        Args:
            x: [B, channels, T] input features

        Returns:
            [B, channels, T] processed features
        """
        for layer in self.layers:
            residual = x
            out = layer(x)
            # Ensure same length (padding may add/remove samples)
            if out.shape[2] != residual.shape[2]:
                out = out[:, :, :residual.shape[2]]
            x = out + residual

        # Apply layer norm (transpose for channel-last)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation for speaker conditioning.

    Produces scale (gamma) and shift (beta) from speaker embedding
    to modulate intermediate features.
    """

    def __init__(self, speaker_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(speaker_dim, feature_dim)
        self.beta_proj = nn.Linear(speaker_dim, feature_dim)

    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: [B, feature_dim, T] features to modulate
            speaker: [B, speaker_dim] speaker embedding

        Returns:
            [B, feature_dim, T] modulated features
        """
        gamma = self.gamma_proj(speaker).unsqueeze(2)  # [B, feature_dim, 1]
        beta = self.beta_proj(speaker).unsqueeze(2)
        return x * (1 + gamma) + beta


class CoMoSVCDecoder(nn.Module):
    """CoMoSVC consistency model decoder.

    Generates mel spectrograms from content, pitch, and speaker features
    using a consistency model approach. Supports 1-step inference for
    real-time use and multi-step for higher quality.

    Architecture:
    - Input projection: content(768) + pitch(256) → hidden_dim
    - Speaker conditioning: FiLM modulation
    - BiDilConv backbone: 8 layers, exponential dilation
    - Output projection: hidden_dim → n_mels(100)
    - Consistency model: maps noisy mel → clean mel in 1 step

    Reference: CoMoSVC (ISCSLP 2024)
    """

    def __init__(self, content_dim: int = 768, pitch_dim: int = 256,
                 speaker_dim: int = 256, n_mels: int = 100,
                 hidden_dim: int = 512, n_layers: int = 8,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.content_dim = content_dim
        self.pitch_dim = pitch_dim
        self.speaker_dim = speaker_dim
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Input projection: content + pitch → hidden
        self.input_proj = nn.Linear(content_dim + pitch_dim, hidden_dim)

        # Speaker conditioning via FiLM
        self.speaker_film = FiLMConditioning(speaker_dim, hidden_dim)

        # Timestep embedding for consistency model
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # BiDilConv backbone
        self.backbone = BiDilConv(
            channels=hidden_dim,
            kernel_size=3,
            n_layers=n_layers,
            dropout=0.1,
        )

        # Output projection to mel
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, n_mels, 1),
        )

        # LoRA state tracking
        self._lora_injected = False
        self._original_layers: Dict[str, nn.Module] = {}

    def inject_lora(self, rank: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        """Inject LoRA adapters into Linear layers for fine-tuning.

        Wraps Linear layers with LoRALinear for parameter-efficient training.
        Each voice profile gets its own LoRA weights.

        Args:
            rank: LoRA rank (dimension of low-rank decomposition)
            alpha: LoRA alpha (scaling factor)
            dropout: Dropout rate for LoRA layers
        """
        if self._lora_injected:
            logger.debug("LoRA already injected, skipping")
            return

        from ..training.fine_tuning import LoRALinear

        # Store original and wrap with LoRA
        if isinstance(self.input_proj, nn.Linear):
            self._original_layers['input_proj'] = self.input_proj
            self.input_proj = LoRALinear(
                self.input_proj, rank=rank, alpha=alpha, dropout=dropout
            )

        # Also wrap time embedding linear layers
        for i, layer in enumerate(self.time_embed):
            if isinstance(layer, nn.Linear):
                key = f'time_embed.{i}'
                self._original_layers[key] = layer
                self.time_embed[i] = LoRALinear(
                    layer, rank=rank, alpha=alpha, dropout=dropout
                )

        # Wrap FiLM conditioning layers
        if isinstance(self.speaker_film.gamma_proj, nn.Linear):
            self._original_layers['speaker_film.gamma_proj'] = self.speaker_film.gamma_proj
            self.speaker_film.gamma_proj = LoRALinear(
                self.speaker_film.gamma_proj, rank=rank, alpha=alpha, dropout=dropout
            )
        if isinstance(self.speaker_film.beta_proj, nn.Linear):
            self._original_layers['speaker_film.beta_proj'] = self.speaker_film.beta_proj
            self.speaker_film.beta_proj = LoRALinear(
                self.speaker_film.beta_proj, rank=rank, alpha=alpha, dropout=dropout
            )

        self._lora_injected = True
        logger.info(f"LoRA injected with rank={rank}, alpha={alpha}")

    def remove_lora(self) -> None:
        """Remove LoRA adapters and restore original Linear layers."""
        if not self._lora_injected:
            return

        # Restore original layers
        if 'input_proj' in self._original_layers:
            self.input_proj = self._original_layers['input_proj']

        for key, layer in self._original_layers.items():
            if key.startswith('time_embed.'):
                idx = int(key.split('.')[1])
                self.time_embed[idx] = layer
            elif key == 'speaker_film.gamma_proj':
                self.speaker_film.gamma_proj = layer
            elif key == 'speaker_film.beta_proj':
                self.speaker_film.beta_proj = layer

        self._original_layers.clear()
        self._lora_injected = False
        logger.info("LoRA removed, original layers restored")

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only LoRA adapter parameters.

        Returns:
            Dict of LoRA parameter tensors (lora_A, lora_B for each layer)
        """
        if not self._lora_injected:
            raise RuntimeError("LoRA not injected - call inject_lora() first")

        from ..training.fine_tuning import LoRALinear

        lora_state = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[f'{name}.adapter.lora_A'] = module.adapter.lora_A.data.clone()
                lora_state[f'{name}.adapter.lora_B'] = module.adapter.lora_B.data.clone()

        return lora_state

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load LoRA adapter parameters from state dict.

        Args:
            state_dict: Dict containing LoRA parameters
        """
        if not self._lora_injected:
            raise RuntimeError("LoRA not injected - call inject_lora() first")

        from ..training.fine_tuning import LoRALinear

        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                lora_a_key = f'{name}.adapter.lora_A'
                lora_b_key = f'{name}.adapter.lora_B'
                if lora_a_key in state_dict:
                    module.adapter.lora_A.data.copy_(state_dict[lora_a_key])
                if lora_b_key in state_dict:
                    module.adapter.lora_B.data.copy_(state_dict[lora_b_key])

        logger.info(f"Loaded LoRA state dict with {len(state_dict)} parameters")

    def _consistency_function(self, x_t: torch.Tensor, t: torch.Tensor,
                              condition: torch.Tensor,
                              speaker: torch.Tensor) -> torch.Tensor:
        """Consistency function: maps (x_t, t) → x_0.

        The consistency model is trained such that f(x_t, t) = x_0
        for any t along the diffusion trajectory.

        Args:
            x_t: [B, n_mels, T] noisy mel at timestep t
            t: [B, 1] timestep value in [0, 1]
            condition: [B, hidden_dim, T] conditioning features
            speaker: [B, speaker_dim] speaker embedding

        Returns:
            [B, n_mels, T] predicted clean mel
        """
        # Embed timestep
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(2)  # [B, hidden_dim, 1]

        # Combine noisy mel with condition
        # Project mel to hidden dim
        # Handle both nn.Linear and LoRALinear (which wraps original in .original)
        input_weight = (
            self.input_proj.original.weight
            if hasattr(self.input_proj, 'original')
            else self.input_proj.weight
        )
        mel_proj = F.linear(
            x_t.transpose(1, 2),
            input_weight[:, :self.n_mels],
            None,
        ).transpose(1, 2)  # Approximate mel→hidden (reuse weights)

        # Add condition and timestep
        h = condition + t_emb + mel_proj

        # Speaker conditioning
        h = self.speaker_film(h, speaker)

        # BiDilConv processing
        h = self.backbone(h)

        # Output mel
        mel_pred = self.output_proj(h)  # [B, n_mels, T]
        return mel_pred

    def forward(self, content: torch.Tensor, pitch: torch.Tensor,
                speaker: torch.Tensor) -> torch.Tensor:
        """Forward pass: generate mel from content + pitch + speaker.

        Single-step consistency model inference (default mode).

        Args:
            content: [B, T, content_dim] content features from ContentVec
            pitch: [B, T, pitch_dim] pitch embeddings from PitchEncoder
            speaker: [B, speaker_dim] speaker embedding (L2-normalized)

        Returns:
            [B, n_mels, T] predicted mel spectrogram
        """
        return self.infer(content, pitch, speaker, n_steps=1)

    def infer(self, content: torch.Tensor, pitch: torch.Tensor,
              speaker: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Inference with configurable number of consistency steps.

        Args:
            content: [B, T, content_dim]
            pitch: [B, T, pitch_dim]
            speaker: [B, speaker_dim]
            n_steps: Number of denoising steps (1 for consistency, >1 for multi-step)

        Returns:
            [B, n_mels, T] predicted mel spectrogram
        """
        B, T, _ = content.shape

        # Project conditioning: content + pitch → hidden
        cond_input = torch.cat([content, pitch], dim=-1)  # [B, T, content+pitch]
        condition = self.input_proj(cond_input)  # [B, T, hidden_dim]
        condition = condition.transpose(1, 2)  # [B, hidden_dim, T]

        # Initialize from noise
        x = torch.randn(B, self.n_mels, T, device=content.device)

        # Multi-step consistency sampling
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=content.device)

        for i in range(n_steps):
            t = timesteps[i].expand(B, 1)  # Current timestep
            x = self._consistency_function(x, t, condition, speaker)

            # Add noise for next step (except last)
            if i < n_steps - 1:
                t_next = timesteps[i + 1]
                noise = torch.randn_like(x)
                x = x + (t_next ** 0.5) * noise

        return x
