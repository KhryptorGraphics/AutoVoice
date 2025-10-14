"""Loss functions for voice training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VoiceLoss(nn.Module):
    """Combined loss for voice synthesis."""

    def __init__(self, reconstruction_weight: float = 1.0,
                perceptual_weight: float = 0.1,
                adversarial_weight: float = 0.1):
        """Initialize voice loss.

        Args:
            reconstruction_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            adversarial_weight: Weight for adversarial loss
        """
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight

        # Component losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               perceptual_features: Optional[Dict] = None,
               discriminator_outputs: Optional[torch.Tensor] = None) -> Dict:
        """Compute combined loss.

        Args:
            predictions: Predicted features
            targets: Target features
            perceptual_features: Features for perceptual loss
            discriminator_outputs: Outputs from discriminator

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Reconstruction loss
        reconstruction_loss = self.l1_loss(predictions, targets)
        losses['reconstruction'] = reconstruction_loss

        # Spectral loss
        spectral_loss = self._spectral_loss(predictions, targets)
        losses['spectral'] = spectral_loss

        # Perceptual loss
        if perceptual_features is not None:
            perceptual_loss = self._perceptual_loss(
                perceptual_features['pred'],
                perceptual_features['target']
            )
            losses['perceptual'] = perceptual_loss
        else:
            perceptual_loss = 0.0

        # Adversarial loss
        if discriminator_outputs is not None:
            adversarial_loss = -torch.mean(discriminator_outputs)
            losses['adversarial'] = adversarial_loss
        else:
            adversarial_loss = 0.0

        # Combined loss
        total_loss = (self.reconstruction_weight * reconstruction_loss +
                     self.reconstruction_weight * spectral_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.adversarial_weight * adversarial_loss)

        losses['total'] = total_loss

        return losses

    def _spectral_loss(self, predictions: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        """Compute spectral convergence loss.

        Args:
            predictions: Predicted spectrogram
            targets: Target spectrogram

        Returns:
            Spectral loss
        """
        # Compute magnitude difference
        diff = torch.abs(predictions - targets)

        # Spectral convergence
        spectral_convergence = torch.norm(diff) / torch.norm(targets)

        # Log magnitude loss
        log_loss = self.l1_loss(
            torch.log(predictions + 1e-7),
            torch.log(targets + 1e-7)
        )

        return spectral_convergence + log_loss

    def _perceptual_loss(self, pred_features: Dict,
                        target_features: Dict) -> torch.Tensor:
        """Compute perceptual loss using feature matching.

        Args:
            pred_features: Predicted features from perceptual model
            target_features: Target features from perceptual model

        Returns:
            Perceptual loss
        """
        loss = 0.0
        for key in pred_features:
            if key in target_features:
                loss += self.l1_loss(pred_features[key], target_features[key])

        return loss / len(pred_features)


class SpeakerLoss(nn.Module):
    """Loss for speaker embedding and verification."""

    def __init__(self, num_speakers: int = 100, margin: float = 0.2):
        """Initialize speaker loss.

        Args:
            num_speakers: Number of speakers
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.num_speakers = num_speakers
        self.margin = margin

        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor,
               logits: Optional[torch.Tensor] = None) -> Dict:
        """Compute speaker loss.

        Args:
            embeddings: Speaker embeddings (batch, embedding_dim)
            speaker_ids: Speaker IDs (batch,)
            logits: Speaker classification logits

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Classification loss
        if logits is not None:
            classification_loss = self.ce_loss(logits, speaker_ids)
            losses['classification'] = classification_loss
        else:
            classification_loss = 0.0

        # Contrastive loss
        contrastive_loss = self._contrastive_loss(embeddings, speaker_ids)
        losses['contrastive'] = contrastive_loss

        # Total loss
        losses['total'] = classification_loss + contrastive_loss

        return losses

    def _contrastive_loss(self, embeddings: torch.Tensor,
                         speaker_ids: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for speaker embeddings.

        Args:
            embeddings: Speaker embeddings
            speaker_ids: Speaker IDs

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create mask for same speakers
        mask = speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)

        # Positive pairs (same speaker)
        positive_distances = distances * mask.float()
        positive_loss = positive_distances.sum() / mask.sum().clamp(min=1)

        # Negative pairs (different speakers)
        negative_distances = distances * (~mask).float()
        negative_loss = F.relu(self.margin - negative_distances)
        negative_loss = negative_loss.sum() / (~mask).sum().clamp(min=1)

        return positive_loss + negative_loss


class DiscriminatorLoss(nn.Module):
    """Loss for discriminator in adversarial training."""

    def __init__(self):
        """Initialize discriminator loss."""
        super().__init__()

    def forward(self, real_outputs: torch.Tensor,
               fake_outputs: torch.Tensor) -> Dict:
        """Compute discriminator loss.

        Args:
            real_outputs: Discriminator outputs for real samples
            fake_outputs: Discriminator outputs for fake samples

        Returns:
            Dictionary of losses
        """
        # Hinge loss
        real_loss = torch.mean(F.relu(1.0 - real_outputs))
        fake_loss = torch.mean(F.relu(1.0 + fake_outputs))

        total_loss = real_loss + fake_loss

        return {
            'real': real_loss,
            'fake': fake_loss,
            'total': total_loss
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss for joint training."""

    def __init__(self, task_weights: Dict[str, float]):
        """Initialize multi-task loss.

        Args:
            task_weights: Weights for each task
        """
        super().__init__()
        self.task_weights = task_weights

        # Task-specific losses
        self.voice_loss = VoiceLoss()
        self.speaker_loss = SpeakerLoss()

    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        """Compute multi-task loss.

        Args:
            outputs: Model outputs for all tasks
            targets: Target values for all tasks

        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0

        # Voice synthesis loss
        if 'voice' in outputs and 'voice' in self.task_weights:
            voice_losses = self.voice_loss(
                outputs['voice'],
                targets['voice']
            )
            losses['voice'] = voice_losses
            total_loss += self.task_weights['voice'] * voice_losses['total']

        # Speaker embedding loss
        if 'speaker' in outputs and 'speaker' in self.task_weights:
            speaker_losses = self.speaker_loss(
                outputs['speaker']['embeddings'],
                targets['speaker_ids'],
                outputs['speaker'].get('logits')
            )
            losses['speaker'] = speaker_losses
            total_loss += self.task_weights['speaker'] * speaker_losses['total']

        losses['total'] = total_loss

        return losses