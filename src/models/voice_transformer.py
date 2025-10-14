"""Transformer-based voice synthesis model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class VoiceTransformer(nn.Module):
    """Transformer model for voice synthesis."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 512,
                num_layers: int = 6, num_heads: int = 8,
                output_dim: int = 80, dropout: float = 0.1):
        """Initialize VoiceTransformer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Additional components for voice synthesis
        self.pitch_embedding = nn.Linear(1, hidden_dim)
        self.energy_embedding = nn.Linear(1, hidden_dim)
        self.speaker_embedding = nn.Embedding(100, hidden_dim)  # 100 speakers

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pitch: Optional[torch.Tensor] = None,
               energy: Optional[torch.Tensor] = None,
               speaker_id: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (batch, seq_len, input_dim)
            pitch: Pitch values (batch, seq_len, 1)
            energy: Energy values (batch, seq_len, 1)
            speaker_id: Speaker IDs (batch,)
            mask: Attention mask (batch, seq_len)

        Returns:
            Output features
        """
        batch_size, seq_len = x.shape[:2]

        # Input projection
        x = self.input_projection(x)

        # Add pitch and energy embeddings if provided
        if pitch is not None:
            pitch_emb = self.pitch_embedding(pitch)
            x = x + pitch_emb

        if energy is not None:
            energy_emb = self.energy_embedding(energy)
            x = x + energy_emb

        # Add speaker embedding if provided
        if speaker_id is not None:
            speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(1)
            speaker_emb = speaker_emb.expand(-1, seq_len, -1)
            x = x + speaker_emb

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Output projection
        output = self.output_projection(x)

        return output

    def generate(self, context: torch.Tensor, length: int,
                pitch: Optional[torch.Tensor] = None,
                energy: Optional[torch.Tensor] = None,
                speaker_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate voice features autoregressively.

        Args:
            context: Context features (batch, context_len, input_dim)
            length: Number of frames to generate
            pitch: Pitch values for generation
            energy: Energy values for generation
            speaker_id: Speaker ID for generation

        Returns:
            Generated features
        """
        self.eval()
        batch_size = context.shape[0]
        device = context.device

        # Initialize output with context
        output = context.clone()

        with torch.no_grad():
            for i in range(length):
                # Get current sequence
                curr_seq = output[:, -self.max_seq_len:] if hasattr(self, 'max_seq_len') else output

                # Prepare pitch and energy for current position
                curr_pitch = pitch[:, :curr_seq.shape[1]] if pitch is not None else None
                curr_energy = energy[:, :curr_seq.shape[1]] if energy is not None else None

                # Forward pass
                pred = self.forward(curr_seq, curr_pitch, curr_energy, speaker_id)

                # Take last prediction
                next_frame = pred[:, -1:, :]

                # Append to output
                output = torch.cat([output, next_frame], dim=1)

        return output[:, context.shape[1]:]  # Return only generated part


class ConditionalVoiceTransformer(VoiceTransformer):
    """Conditional voice transformer with style control."""

    def __init__(self, *args, num_styles: int = 10, **kwargs):
        """Initialize conditional transformer.

        Args:
            num_styles: Number of voice styles
        """
        super().__init__(*args, **kwargs)

        # Style embeddings
        self.style_embedding = nn.Embedding(num_styles, self.hidden_dim)

        # Emotion embeddings
        self.emotion_embedding = nn.Embedding(7, self.hidden_dim)  # 7 basic emotions

        # Cross-attention for conditioning
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, pitch: Optional[torch.Tensor] = None,
               energy: Optional[torch.Tensor] = None,
               speaker_id: Optional[torch.Tensor] = None,
               style_id: Optional[torch.Tensor] = None,
               emotion_id: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with style and emotion conditioning.

        Args:
            x: Input features
            pitch: Pitch values
            energy: Energy values
            speaker_id: Speaker IDs
            style_id: Style IDs (batch,)
            emotion_id: Emotion IDs (batch,)
            mask: Attention mask

        Returns:
            Output features
        """
        batch_size, seq_len = x.shape[:2]

        # Get base encoding
        x = self.input_projection(x)

        # Add all embeddings
        if pitch is not None:
            x = x + self.pitch_embedding(pitch)

        if energy is not None:
            x = x + self.energy_embedding(energy)

        if speaker_id is not None:
            speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(1)
            x = x + speaker_emb.expand(-1, seq_len, -1)

        if style_id is not None:
            style_emb = self.style_embedding(style_id).unsqueeze(1)
            x = x + style_emb.expand(-1, seq_len, -1)

        if emotion_id is not None:
            emotion_emb = self.emotion_embedding(emotion_id).unsqueeze(1)
            # Use cross-attention for emotion conditioning
            x, _ = self.cross_attention(x, emotion_emb, emotion_emb)

        # Continue with standard transformer processing
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = self.output_projection(x)

        return output