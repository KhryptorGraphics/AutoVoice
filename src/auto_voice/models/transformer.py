"""
Transformer model for voice synthesis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Handle mask dimensions for multi-head attention
        if mask is not None:
            if mask.dim() == 2:
                if mask.shape[0] == batch_size:  # (batch, seq_len)
                    # Convert to (batch, 1, 1, seq_len) for broadcasting
                    mask = mask.unsqueeze(1).unsqueeze(1)
                else:  # (seq_len, seq_len) - causal mask
                    # Add batch dimension: (1, seq_len, seq_len) then broadcast
                    mask = mask.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)
            elif mask.dim() == 3:  # (batch, seq_len, seq_len)
                # Convert to (batch, 1, seq_len, seq_len) for broadcasting
                mask = mask.unsqueeze(1)
            # Invert mask: 1 means attend, 0 means mask out
            mask = mask.bool()
            mask = ~mask  # Invert for F.scaled_dot_product_attention

        # Attention
        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(attn)

class TransformerBlock(nn.Module):
    """Transformer encoder/decoder block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x

class VoiceTransformer(nn.Module):
    """Transformer model for voice synthesis"""

    def __init__(self, input_dim: int = 80, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 5000,
                 dropout: float = 0.1, vocab_size: int = None, hidden_size: int = None,
                 num_layers: int = None, num_heads: int = None, 
                 max_sequence_length: int = None):
        super().__init__()
        
        # Support both parameter styles for compatibility
        if hidden_size is not None:
            d_model = hidden_size
        if num_layers is not None:
            n_layers = num_layers
        if num_heads is not None:
            n_heads = num_heads
        if max_sequence_length is not None:
            max_seq_len = max_sequence_length
        if vocab_size is not None:
            input_dim = vocab_size
            
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_layers = n_layers
        self.num_heads = n_heads
        self.input_dim = input_dim
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding"""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, speaker_id: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle different input formats
        if x.dtype in [torch.long, torch.int64]:  # Token IDs
            # Convert to one-hot encoding for embedding
            x = F.one_hot(x.clamp(0, self.input_dim - 1), num_classes=self.input_dim).float()
        
        # Use attention_mask if provided (for test compatibility)
        if attention_mask is not None:
            mask = attention_mask
            
        seq_len = x.size(1)

        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Output projection
        return self.output_projection(x)

    def forward_for_training(self, mel_spec: torch.Tensor, speaker_id: Optional[torch.Tensor] = None,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training with trainer-compatible signature.

        Args:
            mel_spec: Input mel-spectrogram features (batch, seq_len, input_dim)
            speaker_id: Speaker IDs (optional, not used in base model)
            mask: Attention mask (optional)

        Returns:
            Model predictions (batch, seq_len, output_dim)
        """
        # Call the standard forward method
        # mel_spec is already the input features (x parameter)
        return self.forward(mel_spec, speaker_id, mask)

    def export_to_onnx(self, output_path: str, input_shape: Tuple[int, int, int] = (1, 100, 80),
                      dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                      opset_version: int = 17, verbose: bool = True) -> None:
        """Export model to ONNX format for TensorRT conversion.

        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch, seq_len, input_dim)
            dynamic_axes: Dynamic axes configuration for variable-length sequences
            opset_version: ONNX opset version (17 for TensorRT 8.6+ compatibility)
            verbose: Enable verbose logging
        """
        import os
        from pathlib import Path

        # Set model to eval mode
        self.eval()

        # Create dummy input matching expected format
        batch_size, seq_len, input_dim = input_shape
        dummy_input = torch.randn(batch_size, seq_len, input_dim)

        # Default dynamic axes for variable-length sequences
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        if verbose:
            print(f"Exporting VoiceTransformer to ONNX: {output_path}")
            print(f"  Input shape: {input_shape}")
            print(f"  Dynamic axes: {dynamic_axes}")
            print(f"  Opset version: {opset_version}")

        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=verbose
        )

        if verbose:
            print(f"✓ ONNX export completed: {output_path}")
            print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")