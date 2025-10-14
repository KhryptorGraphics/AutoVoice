"""Voice encoder for speaker embedding and representation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VoiceEncoder(nn.Module):
    """Encoder for extracting voice representations."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256,
                embedding_dim: int = 256, num_layers: int = 3):
        """Initialize voice encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            embedding_dim: Output embedding dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Convolutional layers for local feature extraction
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True)

        # Projection to embedding
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract voice embedding.

        Args:
            x: Input features (batch, time, features)
            lengths: Sequence lengths for each batch item

        Returns:
            Frame-level features and utterance-level embedding
        """
        # Transpose for convolution (batch, features, time)
        x = x.transpose(1, 2)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Transpose back for LSTM (batch, time, features)
        x = x.transpose(1, 2)

        # LSTM processing
        if lengths is not None:
            # Pack sequences for efficient processing
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        lstm_out, (hidden, cell) = self.lstm(x)

        if lengths is not None:
            # Unpack sequences
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )

        # Frame-level features
        frame_features = self.projection(lstm_out)

        # Utterance-level embedding (average pooling)
        if lengths is not None:
            # Mask padding
            mask = torch.arange(lstm_out.size(1), device=lstm_out.device)
            mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).float()
            utterance_embedding = (lstm_out * mask).sum(1) / lengths.unsqueeze(1).float()
        else:
            utterance_embedding = lstm_out.mean(dim=1)

        utterance_embedding = self.projection(utterance_embedding)

        return frame_features, utterance_embedding


class SpeakerEncoder(VoiceEncoder):
    """Specialized encoder for speaker verification."""

    def __init__(self, *args, num_speakers: int = 100, **kwargs):
        """Initialize speaker encoder.

        Args:
            num_speakers: Number of speakers for classification
        """
        super().__init__(*args, **kwargs)

        # Speaker classification head
        self.speaker_classifier = nn.Linear(self.embedding_dim, num_speakers)

        # Similarity scoring
        self.similarity_weight = nn.Parameter(torch.tensor(10.0))
        self.similarity_bias = nn.Parameter(torch.tensor(-5.0))

    def forward_embedding(self, x: torch.Tensor,
                         lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract speaker embedding.

        Args:
            x: Input features
            lengths: Sequence lengths

        Returns:
            Speaker embedding
        """
        _, embedding = super().forward(x, lengths)
        return F.normalize(embedding, p=2, dim=1)

    def similarity(self, embedding1: torch.Tensor,
                  embedding2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score
        """
        # Cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2, dim=-1)

        # Apply learned scaling
        similarity = similarity * self.similarity_weight + self.similarity_bias

        return torch.sigmoid(similarity)

    def classify_speaker(self, x: torch.Tensor,
                        lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classify speaker from audio.

        Args:
            x: Input features
            lengths: Sequence lengths

        Returns:
            Speaker logits
        """
        embedding = self.forward_embedding(x, lengths)
        return self.speaker_classifier(embedding)


class ContentEncoder(nn.Module):
    """Encoder for extracting content (linguistic) features."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256,
                content_dim: int = 128):
        """Initialize content encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            content_dim: Content embedding dimension
        """
        super().__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, content_dim, 3, padding=1)
        )

        # Instance normalization to remove speaker information
        self.instance_norm = nn.InstanceNorm1d(content_dim, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract content features.

        Args:
            x: Input features (batch, time, features)

        Returns:
            Content features
        """
        # Transpose for convolution
        x = x.transpose(1, 2)

        # Encode
        content = self.encoder(x)

        # Remove speaker information
        content = self.instance_norm(content)

        # Transpose back
        content = content.transpose(1, 2)

        return content


class StyleEncoder(nn.Module):
    """Encoder for extracting style features."""

    def __init__(self, input_dim: int = 80, style_dim: int = 128):
        """Initialize style encoder.

        Args:
            input_dim: Input feature dimension
            style_dim: Style embedding dimension
        """
        super().__init__()

        # Reference encoder
        self.reference_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        # Style token attention
        self.gst_tokens = nn.Parameter(torch.randn(10, style_dim))
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)

        # Final projection
        self.projection = nn.Linear(128, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract style embedding.

        Args:
            x: Input features (batch, time, features)

        Returns:
            Style embedding
        """
        # Encode reference
        x = x.transpose(1, 2)
        ref_encoding = self.reference_encoder(x)

        # Global average pooling
        ref_encoding = ref_encoding.mean(dim=-1, keepdim=True).transpose(1, 2)

        # Attention over style tokens
        batch_size = x.shape[0]
        tokens = self.gst_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        style_embed, _ = self.attention(ref_encoding, tokens, tokens)

        # Project to style dimension
        style_embed = self.projection(style_embed.squeeze(1))

        return style_embed