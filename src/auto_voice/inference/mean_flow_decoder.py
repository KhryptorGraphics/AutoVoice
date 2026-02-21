"""Mean Flow Decoder for single-step voice conversion inference.

This module implements the mean flow regression approach from MeanVC, which
enables single-step inference by directly regressing the average velocity field
instead of iteratively solving an ODE.

Key innovation: Instead of computing v(x_t, t) at each timestep and integrating,
we directly predict the mean flow: x1 = x0 + mean_flow(x0, conditions).

Research paper: MeanVC (arXiv:2510.08392)

Architecture:
    ContentVec features → DiT with Mean Flow → Mel spectrogram
    Single step: x1 = x0 + ∫₀¹ v(x_t, t) dt ≈ x0 + mean_v(x0)

This is MUCH faster than iterative CFM (10 steps) or diffusion (30+ steps).
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MeanFlowDecoder(nn.Module):
    """
    Mean flow decoder for single-step voice conversion.

    Instead of iteratively solving the ODE:
        dx/dt = v(x_t, t)
        x1 = x0 + ∫₀¹ v(x_t, t) dt

    We directly regress the mean flow:
        mean_v(x0) ≈ ∫₀¹ v(x_t, t) dt
        x1 = x0 + mean_v(x0)

    This reduces inference to a single forward pass through the network.

    Training:
        The mean flow is learned by:
        1. Standard flow matching loss: E[||v(x_t, t) - (x1 - x0)||²]
        2. Mean flow loss: E[||∫v dt - (x1 - x0)||²]

    Inference:
        Single step: x1 = x0 + model(x0, t=0→1, conditions)
    """

    def __init__(
        self,
        content_dim: int = 512,
        speaker_dim: int = 256,
        mel_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        """
        Initialize mean flow decoder.

        Args:
            content_dim: Dimension of content features (from ASR)
            speaker_dim: Dimension of speaker embedding
            mel_dim: Output mel spectrogram dimension
            hidden_dim: Hidden dimension of transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.x_proj = nn.Linear(mel_dim, hidden_dim)
        self.content_proj = nn.Linear(content_dim, hidden_dim)

        # Speaker conditioning
        self.spk_proj = nn.Linear(speaker_dim, hidden_dim)

        # Time embedding (for t and r in mean flow)
        self.time_embed = TimeEmbedding(hidden_dim)

        # Transformer decoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, mel_dim)

        logger.info(
            f"MeanFlowDecoder initialized: "
            f"{content_dim}D content → {hidden_dim}D → {mel_dim}D mel, "
            f"{num_layers} layers, {num_heads} heads"
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        content: torch.Tensor,
        speaker: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict mean flow from x to x1.

        Args:
            x: Current noisy state [B, T, mel_dim]
            t: Current time (0→1) [B]
            r: Target time (usually 0 for mean flow) [B]
            content: Content features from ASR [B, T, content_dim]
            speaker: Speaker embedding [B, speaker_dim]
            prompt: Optional reference mel for in-context learning [B, T_prompt, mel_dim]

        Returns:
            Mean velocity field [B, T, mel_dim]
        """
        B, T, _ = x.shape

        # Project inputs
        x_emb = self.x_proj(x)  # [B, T, hidden]
        content_emb = self.content_proj(content)  # [B, T, hidden]

        # Time conditioning
        t_emb = self.time_embed(t)  # [B, hidden]
        r_emb = self.time_embed(r)  # [B, hidden]
        time_cond = t_emb + r_emb  # Combine t and r

        # Speaker conditioning
        spk_emb = self.spk_proj(speaker)  # [B, hidden]

        # Combine all conditioning
        # Broadcast time and speaker to all frames
        cond = (
            time_cond.unsqueeze(1) + spk_emb.unsqueeze(1)
        )  # [B, 1, hidden]

        # Combine x, content, and conditioning
        h = x_emb + content_emb + cond  # [B, T, hidden]

        # Add prompt if provided (in-context learning)
        if prompt is not None:
            prompt_emb = self.x_proj(prompt)  # [B, T_prompt, hidden]
            h = torch.cat([prompt_emb, h], dim=1)  # [B, T_prompt + T, hidden]

        # Transformer processing
        h = self.transformer(h)  # [B, T (+T_prompt), hidden]

        # Remove prompt frames if added
        if prompt is not None:
            h = h[:, prompt.size(1):, :]  # [B, T, hidden]

        # Project to velocity field
        v = self.out_proj(h)  # [B, T, mel_dim]

        return v

    def inference_single_step(
        self,
        x0: torch.Tensor,
        content: torch.Tensor,
        speaker: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single-step inference via mean flow.

        Args:
            x0: Initial noise [B, T, mel_dim]
            content: Content features [B, T, content_dim]
            speaker: Speaker embedding [B, speaker_dim]
            prompt: Optional reference mel [B, T_prompt, mel_dim]

        Returns:
            Predicted mel spectrogram [B, T, mel_dim]
        """
        B = x0.size(0)
        device = x0.device

        # Mean flow: integrate from t=1 (noise) to t=0 (data)
        t = torch.ones(B, device=device)
        r = torch.zeros(B, device=device)

        # Predict mean velocity
        v = self.forward(x0, t, r, content, speaker, prompt)

        # Single step: x1 = x0 + (t - r) * v
        x1 = x0 - (t.view(B, 1, 1) - r.view(B, 1, 1)) * v

        return x1

    def inference_two_step(
        self,
        x0: torch.Tensor,
        content: torch.Tensor,
        speaker: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Two-step inference for slightly better quality.

        Args:
            x0: Initial noise [B, T, mel_dim]
            content: Content features [B, T, content_dim]
            speaker: Speaker embedding [B, speaker_dim]
            prompt: Optional reference mel [B, T_prompt, mel_dim]

        Returns:
            Predicted mel spectrogram [B, T, mel_dim]
        """
        B = x0.size(0)
        device = x0.device

        # Step 1: t=1.0 → t=0.8
        t1 = torch.ones(B, device=device) * 1.0
        r1 = torch.ones(B, device=device) * 0.8
        v1 = self.forward(x0, t1, r1, content, speaker, prompt)
        x_mid = x0 - (t1.view(B, 1, 1) - r1.view(B, 1, 1)) * v1

        # Step 2: t=0.8 → t=0.0
        t2 = torch.ones(B, device=device) * 0.8
        r2 = torch.zeros(B, device=device)
        v2 = self.forward(x_mid, t2, r2, content, speaker, prompt)
        x1 = x_mid - (t2.view(B, 1, 1) - r2.view(B, 1, 1)) * v2

        return x1


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (same as DiT/CFM models)."""

    def __init__(self, dim: int, max_period: int = 10000):
        """Initialize time embedding layer.

        Creates sinusoidal positional encodings for continuous time values,
        similar to the positional encoding in transformers but for scalar
        time inputs. The embeddings are processed through an MLP for
        additional expressiveness.

        Args:
            dim: Embedding dimension (output size)
            max_period: Maximum period for sinusoidal encoding (default: 10000)
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] in range [0, 1]

        Returns:
            Time embeddings [B, dim]
        """
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.max_period)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return self.mlp(emb)


# Utility functions for mean flow training
def compute_mean_flow_loss(
    model: MeanFlowDecoder,
    x0: torch.Tensor,
    x1: torch.Tensor,
    content: torch.Tensor,
    speaker: torch.Tensor,
    prompt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dual loss for mean flow training.

    Args:
        model: MeanFlowDecoder instance
        x0: Noise samples [B, T, mel_dim]
        x1: Target mel spectrograms [B, T, mel_dim]
        content: Content features [B, T, content_dim]
        speaker: Speaker embeddings [B, speaker_dim]
        prompt: Optional prompt mel [B, T_prompt, mel_dim]

    Returns:
        fm_loss: Flow matching loss
        mean_loss: Mean flow loss
    """
    B = x0.size(0)
    device = x0.device

    # 1. Flow matching loss at random timestep
    t_random = torch.rand(B, device=device)
    x_t = (1 - t_random.view(B, 1, 1)) * x0 + t_random.view(B, 1, 1) * x1
    target_v = x1 - x0

    v_pred = model.forward(
        x_t, t_random, torch.zeros_like(t_random), content, speaker, prompt
    )
    fm_loss = torch.nn.functional.mse_loss(v_pred, target_v)

    # 2. Mean flow loss (single-step prediction)
    t_ones = torch.ones(B, device=device)
    t_zeros = torch.zeros(B, device=device)

    mean_v = model.forward(x0, t_ones, t_zeros, content, speaker, prompt)
    mean_loss = torch.nn.functional.mse_loss(mean_v, target_v)

    return fm_loss, mean_loss


if __name__ == "__main__":
    # Quick sanity check
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeanFlowDecoder().to(device)

    # Dummy inputs
    B, T = 2, 100
    x0 = torch.randn(B, T, 80, device=device)
    content = torch.randn(B, T, 512, device=device)
    speaker = torch.randn(B, 256, device=device)

    # Test single-step inference
    x1 = model.inference_single_step(x0, content, speaker)
    print(f"Single-step output shape: {x1.shape}")
    assert x1.shape == (B, T, 80), f"Expected (2, 100, 80), got {x1.shape}"

    # Test two-step inference
    x1_2step = model.inference_two_step(x0, content, speaker)
    print(f"Two-step output shape: {x1_2step.shape}")
    assert x1_2step.shape == (B, T, 80), f"Expected (2, 100, 80), got {x1_2step.shape}"

    print("✓ MeanFlowDecoder sanity check passed!")
