"""Training module for AutoVoice models."""
import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VoiceTrainer:
    """Main training class for voice synthesis models."""

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1

        # Initialize distributed training
        if self.is_distributed:
            self.init_distributed()

        # Move model to device and setup DDP
        self.model = self.model.to(device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[device])

        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=config.get('betas', (0.9, 0.99)),
            weight_decay=config.get('weight_decay', 0.01)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )

        # Mixed precision training
        self.use_amp = config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        logger.info(f"Trainer initialized - Rank: {self.rank}, World Size: {self.world_size}")

    def init_distributed(self):
        """Initialize distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
        torch.cuda.set_device(self.rank)

    def train_epoch(self, dataloader, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self.move_batch_to_device(batch)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss = criterion(outputs, batch)
            else:
                outputs = self.model(batch)
                loss = criterion(outputs, batch)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        return total_loss / num_batches

    def validate(self, dataloader, criterion):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_batch_to_device(batch)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch)
                        loss = criterion(outputs, batch)
                else:
                    outputs = self.model(batch)
                    loss = criterion(outputs, batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def move_batch_to_device(self, batch):
        """Move batch data to the appropriate device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [x.to(self.device) if torch.is_tensor(x) else x for x in batch]
        else:
            return batch.to(self.device) if torch.is_tensor(batch) else batch

    def save_checkpoint(self, filepath: str, epoch: int, loss: float,
                       save_optimizer: bool = True):
        """Save model checkpoint."""
        if self.rank != 0:  # Only save on main process
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.get_model_state_dict(),
            'loss': loss,
            'config': self.config
        }

        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.load_model_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint

    def get_model_state_dict(self):
        """Get model state dict, handling DDP wrapper."""
        if hasattr(self.model, 'module'):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict):
        """Load model state dict, handling DDP wrapper."""
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def cleanup_distributed(self):
        """Cleanup distributed training."""
        if self.is_distributed:
            dist.destroy_process_group()


class VoiceLoss(nn.Module):
    """Combined loss for voice synthesis."""

    def __init__(self, l1_weight=1.0, mel_weight=45.0, gan_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.mel_weight = mel_weight
        self.gan_weight = gan_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        """Compute combined loss."""
        # L1 loss
        l1_loss = self.l1_loss(pred['audio'], target['audio'])

        # Mel spectrogram loss
        mel_loss = self.l1_loss(pred['mel'], target['mel'])

        # GAN loss (placeholder)
        gan_loss = torch.tensor(0.0, device=pred['audio'].device)
        if 'disc_real' in pred and 'disc_fake' in pred:
            gan_loss = self.mse_loss(pred['disc_fake'], torch.ones_like(pred['disc_fake']))

        total_loss = (self.l1_weight * l1_loss +
                     self.mel_weight * mel_loss +
                     self.gan_weight * gan_loss)

        return total_loss