"""Conformer encoder for content feature refinement.

Replaces the linear projection in ContentEncoder with a multi-head
self-attention + Conv1D feed-forward network that captures long-range
dependencies in content features. Based on the Amphion SVC Conformer.

Architecture per layer:
    x → LayerNorm → MultiHeadAttention(relative pos) → Dropout → Residual
    x → LayerNorm → Conv1D FFN (GELU) → Dropout → Residual
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerLayerNorm(nn.Module):
    """Channel-wise layer normalization for [B, C, T] tensors."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative position encoding.

    Operates on [B, C, T] tensors. Uses a local window for relative
    position bias, enabling efficient attention on long sequences.
    """

    def __init__(self, channels: int, n_heads: int = 2,
                 window_size: int = 4, dropout: float = 0.1):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

        # Relative position embeddings
        rel_stddev = self.k_channels ** -0.5
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
        )

        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention on [B, C, T] tensor."""
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        out = self._attention(q, k, v)
        return self.conv_o(out)

    def _attention(self, query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor) -> torch.Tensor:
        b, d, t = query.size()
        # Reshape to [B, heads, k_channels, T] then transpose to [B, heads, T, k_channels]
        query = query.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        # Add relative position bias
        rel_k = self._get_relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(query, rel_k.unsqueeze(0).transpose(-2, -1))
        rel_logits = self._relative_to_absolute(rel_logits)
        scores = scores + rel_logits / math.sqrt(self.k_channels)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)

        output = torch.matmul(p_attn, value)

        # Add relative position values
        rel_weights = self._absolute_to_relative(p_attn)
        rel_v = self._get_relative_embeddings(self.emb_rel_v, t)
        output = output + torch.matmul(rel_weights, rel_v.unsqueeze(0))

        # Reshape back to [B, C, T]
        output = output.transpose(2, 3).contiguous().view(b, d, t)
        return output

    def _get_relative_embeddings(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            padded = F.pad(emb, [0, 0, pad_length, pad_length])
        else:
            padded = emb
        return padded[:, slice_start:slice_end]

    def _relative_to_absolute(self, x: torch.Tensor) -> torch.Tensor:
        """Convert relative position scores to absolute."""
        b, heads, length, _ = x.size()
        x = F.pad(x, [0, 1])
        x_flat = x.view(b, heads, length * 2 * length)
        x_flat = F.pad(x_flat, [0, length - 1])
        x_final = x_flat.view(b, heads, length + 1, 2 * length - 1)
        return x_final[:, :, :length, length - 1:]

    def _absolute_to_relative(self, x: torch.Tensor) -> torch.Tensor:
        """Convert absolute position attention to relative."""
        b, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1])
        x_flat = x.view(b, heads, length ** 2 + length * (length - 1))
        x_flat = F.pad(x_flat, [length, 0])
        x_final = x_flat.view(b, heads, length, 2 * length)
        return x_final[:, :, :, 1:]


class ConformerFFN(nn.Module):
    """Feed-forward network with Conv1D and GELU activation."""

    def __init__(self, channels: int, filter_channels: int,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel_size,
                                padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel_size,
                                padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN on [B, C, T] tensor."""
        x = self.conv_1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        return x


class ConformerLayer(nn.Module):
    """Single Conformer layer: attention + FFN with pre-norm residuals."""

    def __init__(self, channels: int, filter_channels: int,
                 n_heads: int = 2, kernel_size: int = 3,
                 window_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_1 = ConformerLayerNorm(channels)
        self.attn = MultiHeadSelfAttention(channels, n_heads, window_size, dropout)
        self.drop_1 = nn.Dropout(dropout)

        self.norm_2 = ConformerLayerNorm(channels)
        self.ffn = ConformerFFN(channels, filter_channels, kernel_size, dropout)
        self.drop_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: pre-norm residual attention + FFN."""
        # Self-attention with residual
        y = self.norm_1(x)
        y = self.attn(y)
        x = x + self.drop_1(y)

        # FFN with residual
        y = self.norm_2(x)
        y = self.ffn(y)
        x = x + self.drop_2(y)

        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder replacing linear projection in ContentEncoder.

    Takes HuBERT features [B, T, 256] and produces enriched content
    features [B, T, output_dim] with long-range dependency modeling.

    Args:
        input_dim: Input feature dimension (HuBERT output = 256)
        hidden_dim: Internal hidden dimension (default 384)
        output_dim: Output feature dimension (default 256)
        n_layers: Number of Conformer layers (default 6)
        n_heads: Number of attention heads (default 2)
        filter_channels: FFN intermediate dimension (default 1536, 4x hidden)
        kernel_size: Conv kernel size in FFN (default 3)
        window_size: Relative attention window (default 4)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 384,
                 output_dim: int = 256, n_layers: int = 6,
                 n_heads: int = 2, filter_channels: Optional[int] = None,
                 kernel_size: int = 3, window_size: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        if filter_channels is None:
            filter_channels = hidden_dim * 4  # Standard 4x expansion

        # Input projection to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Conformer layers
        self.layers = nn.ModuleList([
            ConformerLayer(
                channels=hidden_dim,
                filter_channels=filter_channels,
                n_heads=n_heads,
                kernel_size=kernel_size,
                window_size=window_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = ConformerLayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process content features through Conformer layers.

        Args:
            x: Input features [B, T, input_dim]

        Returns:
            Enriched features [B, T, output_dim]
        """
        # Project to hidden dim: [B, T, input_dim] -> [B, T, hidden_dim]
        x = self.input_proj(x)

        # Transpose to [B, C, T] for Conv1d-based layers
        x = x.transpose(1, 2)

        # Apply Conformer layers
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.final_norm(x)

        # Transpose back to [B, T, C] and project to output dim
        x = x.transpose(1, 2)
        x = self.output_proj(x)

        return x
