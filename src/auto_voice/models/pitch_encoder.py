"""
Pitch Encoder for Singing Voice Conversion

Encodes F0 contour into learned embeddings for voice conversion conditioning.
"""

from typing import Optional, Dict, Union
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PitchEncoder(nn.Module):
    """
    Convert continuous F0 values to learned embeddings.

    Combines quantized and continuous representations for robust pitch conditioning.

    Args:
        pitch_dim: Output embedding dimension (default: 192)
        hidden_dim: Intermediate dimension for projection (default: 128)
        num_bins: Number of quantization bins for F0 (default: 256)
        f0_min: Minimum F0 in Hz for normalization (default: 80.0)
        f0_max: Maximum F0 in Hz for normalization (default: 1000.0)

    Example:
        >>> pitch_encoder = PitchEncoder(pitch_dim=192, f0_min=80.0, f0_max=1000.0)
        >>> f0 = torch.tensor([[440.0, 450.0, 460.0, 0.0, 470.0]])  # [B=1, T=5]
        >>> voiced = torch.tensor([[True, True, True, False, True]])
        >>> pitch_emb = pitch_encoder(f0, voiced)
        >>> print(pitch_emb.shape)  # [1, 5, 192]
    """

    def __init__(
        self,
        pitch_dim: int = 192,
        hidden_dim: int = 128,
        num_bins: int = 256,
        f0_min: float = 80.0,
        f0_max: float = 1000.0
    ):
        super().__init__()

        self.pitch_dim = pitch_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.f0_min = f0_min
        self.f0_max = f0_max

        # Embedding layer for quantized F0 (+1 for unvoiced/0 Hz)
        self.embedding = nn.Embedding(num_bins + 1, pitch_dim)

        # Continuous projection network
        self.continuous_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pitch_dim)
        )

        # Blend weight between quantized and continuous
        self.blend_weight = nn.Parameter(torch.tensor(0.5))

        logger.info(f"PitchEncoder initialized: pitch_dim={pitch_dim}, f0_range=[{f0_min}, {f0_max}]")

    def forward(self, f0: torch.Tensor, voiced: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode F0 contour to pitch embeddings.

        Args:
            f0: F0 tensor in Hz [B, T]
            voiced: Optional voiced mask [B, T]

        Returns:
            Pitch embeddings [B, T, pitch_dim]
        """
        # Step 1: Apply voiced mask if provided - strictly enforce unvoiced frames
        if voiced is not None:
            f0 = torch.where(voiced, f0, torch.zeros_like(f0))

        # Step 2: Create comprehensive unvoiced mask for non-finite, negative, or zero values
        unvoiced_mask = (~torch.isfinite(f0)) | (f0 <= 0)

        # Step 3: Replace non-finite values with 0 for safety
        f0 = torch.where(torch.isfinite(f0), f0, torch.zeros_like(f0))

        # Warn if F0 outside range (only for finite, positive values)
        valid_mask = ~unvoiced_mask
        if torch.any(valid_mask & ((f0 < self.f0_min) | (f0 > self.f0_max))):
            logger.warning(f"F0 values outside range [{self.f0_min}, {self.f0_max}]")

        # Step 4: Normalize F0 with epsilon for finite positive frames
        eps = 1e-6
        f0_norm = (f0 - self.f0_min) / (self.f0_max - self.f0_min)
        # Clamp to [eps, 1.0] for finite positive frames to avoid logit artifacts
        f0_norm = torch.clamp(f0_norm, eps, 1.0)

        # Step 5: Quantization - compute indices for finite positive frames
        f0_bins_idx = (f0_norm * (self.num_bins - 1)).long()
        f0_bins_idx = torch.clamp(f0_bins_idx, 0, self.num_bins - 1)

        # Set unvoiced frames to special unvoiced bin (num_bins)
        f0_bins_idx = torch.where(unvoiced_mask, torch.full_like(f0_bins_idx, self.num_bins), f0_bins_idx)

        quantized_emb = self.embedding(f0_bins_idx)  # [B, T, pitch_dim]

        # Step 6: Continuous path - mask unvoiced frames to zeros
        f0_input = f0_norm.unsqueeze(-1)  # [B, T, 1]
        # Zero out unvoiced frames in continuous path
        f0_input = torch.where(unvoiced_mask.unsqueeze(-1), torch.zeros_like(f0_input), f0_input)
        continuous_emb = self.continuous_proj(f0_input)  # [B, T, pitch_dim]

        # Step 7: Blend both paths
        blend = torch.sigmoid(self.blend_weight)  # Ensure in [0, 1]
        pitch_emb = blend * quantized_emb + (1 - blend) * continuous_emb

        return pitch_emb

    def encode_f0_contour(self, f0_data: Dict) -> torch.Tensor:
        """
        High-level method accepting F0 data dict from SingingPitchExtractor.

        Args:
            f0_data: Dict with 'f0' and 'voiced' arrays

        Returns:
            Pitch embeddings [B, T, pitch_dim]
        """
        f0 = f0_data['f0']
        voiced = f0_data.get('voiced', None)

        # Convert to tensors if numpy
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float()
        if isinstance(voiced, np.ndarray):
            voiced = torch.from_numpy(voiced).bool()

        # Add batch dimension if needed
        if f0.dim() == 1:
            f0 = f0.unsqueeze(0)
        if voiced is not None and voiced.dim() == 1:
            voiced = voiced.unsqueeze(0)

        # Move to same device as module parameters
        device = next(self.parameters()).device
        f0 = f0.to(device)
        if voiced is not None:
            voiced = voiced.to(device)

        return self.forward(f0, voiced)

    def interpolate_to_length(self, pitch_emb: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Interpolate pitch embeddings to match target sequence length.

        Args:
            pitch_emb: Pitch embeddings [B, T, pitch_dim]
            target_length: Target sequence length

        Returns:
            Interpolated embeddings [B, target_length, pitch_dim]
        """
        # Transpose for interpolation: [B, T, pitch_dim] -> [B, pitch_dim, T]
        pitch_emb = pitch_emb.transpose(1, 2)

        # Interpolate
        pitch_emb = F.interpolate(
            pitch_emb,
            size=target_length,
            mode='linear',
            align_corners=False
        )

        # Transpose back: [B, pitch_dim, T] -> [B, T, pitch_dim]
        pitch_emb = pitch_emb.transpose(1, 2)

        return pitch_emb

    def export_to_onnx(
        self,
        onnx_path: str,
        opset_version: int = 17,
        input_sample: Optional[Dict[str, torch.Tensor]] = None
    ) -> str:
        """
        Export PitchEncoder to ONNX format.

        Args:
            onnx_path: Output path for ONNX model
            opset_version: ONNX opset version
            input_sample: Sample inputs dict with 'f0' and 'voiced' tensors
                         If None, creates default sample (50 frames)

        Returns:
            Path to exported ONNX model

        Example:
            >>> pitch_encoder = PitchEncoder(pitch_dim=192)
            >>> pitch_encoder.export_to_onnx('pitch_encoder.onnx')
        """
        self.eval()

        # Create default inputs if not provided
        if input_sample is None:
            f0_sample = torch.randn(1, 50)  # 50 frames at 50Hz
            voiced_sample = torch.ones(1, 50, dtype=torch.bool)  # All voiced
            input_sample = {'f0': f0_sample, 'voiced': voiced_sample}

        device = next(self.parameters()).device
        f0_input = input_sample['f0'].to(device)
        voiced_input = input_sample['voiced'].to(device)

        # Ensure voiced_input is boolean tensor
        if voiced_input.dtype != torch.bool:
            voiced_input = voiced_input.bool()

        # Define dynamic axes for both inputs and output
        dynamic_axes = {
            'f0_input': {0: 'batch_size', 1: 'time_steps'},
            'voiced_mask': {0: 'batch_size', 1: 'time_steps'},
            'pitch_features': {0: 'batch_size', 1: 'time_steps'}
        }

        logger.info(f"Exporting PitchEncoder to ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                self,
                (f0_input, voiced_input),
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['f0_input', 'voiced_mask'],
                output_names=['pitch_features'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"PitchEncoder exported successfully to {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"PitchEncoder ONNX export failed: {e}")
            raise
