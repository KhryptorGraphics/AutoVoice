"""Training logic for voice models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for voice models."""

    def __init__(self, model: nn.Module, device: torch.device,
                learning_rate: float = 1e-4, checkpoint_dir: str = './checkpoints'):
        """Initialize trainer.

        Args:
            model: Model to train
            device: Device for training
            learning_rate: Initial learning rate
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader,
                   criterion: nn.Module) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            criterion: Loss function

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {self.epoch}')
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)

            # Compute loss
            loss = criterion(outputs, batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, dataloader: DataLoader,
                criterion: nn.Module) -> float:
        """Validate model.

        Args:
            dataloader: Validation data loader
            criterion: Loss function

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

                # Compute loss
                loss = criterion(outputs, batch)

                # Update statistics
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             criterion: nn.Module, num_epochs: int,
             save_every: int = 5) -> Dict:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs

        Returns:
            Training statistics
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.epoch = epoch + 1

            # Train
            train_loss = self.train_epoch(train_loader, criterion)
            logger.info(f"Epoch {self.epoch} - Train Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate(val_loader, criterion)
            logger.info(f"Epoch {self.epoch} - Val Loss: {val_loss:.4f}")

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best_model.pth')
                logger.info(f"New best model saved with loss: {val_loss:.4f}")

            # Regular checkpoint
            if self.epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth')

        # Final save
        self.save_checkpoint('final_model.pth')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'final_epoch': self.epoch
        }

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        logger.info(f"Checkpoint loaded from {path}")


class DistributedTrainer(Trainer):
    """Distributed trainer for multi-GPU training."""

    def __init__(self, model: nn.Module, rank: int, world_size: int, *args, **kwargs):
        """Initialize distributed trainer.

        Args:
            model: Model to train
            rank: Process rank
            world_size: Total number of processes
        """
        super().__init__(model, *args, **kwargs)

        self.rank = rank
        self.world_size = world_size

        # Wrap model for distributed training
        self.model = nn.parallel.DistributedDataParallel(
            self.model, device_ids=[rank]
        )

        logger.info(f"Initialized distributed trainer on rank {rank}/{world_size}")

    def train_epoch(self, dataloader: DataLoader,
                   criterion: nn.Module) -> float:
        """Train epoch with distributed synchronization."""
        # Set epoch for distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(self.epoch)

        return super().train_epoch(dataloader, criterion)