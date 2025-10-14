"""
Advanced training utilities for AutoVoice models with multi-GPU support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import time
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
import math

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Basic training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    
    # Loss function settings
    loss_type: str = "mse"  # mse, l1, huber, spectral
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"reconstruction": 1.0, "adversarial": 0.1})
    
    # Optimizer settings
    optimizer_type: str = "adamw"  # adamw, adam, sgd, lion
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler settings
    scheduler_type: str = "cosine"  # cosine, linear, exponential, plateau, onecycle
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 1000
    
    # Mixed precision training
    use_amp: bool = True
    
    # Multi-GPU settings
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # Logging and checkpointing
    log_interval: int = 100
    save_interval: int = 1000
    validate_interval: int = 500
    max_checkpoints: int = 5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Model compilation (PyTorch 2.0+)
    compile_model: bool = False

class LossManager:
    """Manages multiple loss functions for training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.loss_functions = self._create_loss_functions()
        self.loss_weights = config.loss_weights
        
    def _create_loss_functions(self) -> Dict[str, nn.Module]:
        """Create loss functions based on configuration"""
        losses = {}
        
        # Reconstruction losses
        if self.config.loss_type == "mse":
            losses["reconstruction"] = nn.MSELoss()
        elif self.config.loss_type == "l1":
            losses["reconstruction"] = nn.L1Loss()
        elif self.config.loss_type == "huber":
            losses["reconstruction"] = nn.HuberLoss()
        elif self.config.loss_type == "spectral":
            losses["reconstruction"] = SpectralLoss()
        
        # Perceptual loss for audio
        losses["perceptual"] = PerceptualLoss()
        
        # Multi-resolution STFT loss
        losses["stft"] = MultiResolutionSTFTLoss()
        
        return losses
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute weighted combination of losses"""
        losses = {}
        total_loss = 0.0
        
        for loss_name, loss_fn in self.loss_functions.items():
            if loss_name in self.loss_weights and self.loss_weights[loss_name] > 0:
                if loss_name == "reconstruction":
                    loss_value = loss_fn(predictions.get("output", predictions.get("waveform")), 
                                       targets.get("target", targets.get("waveform")))
                elif loss_name == "perceptual":
                    loss_value = loss_fn(predictions.get("output"), targets.get("target"))
                elif loss_name == "stft":
                    loss_value = loss_fn(predictions.get("output"), targets.get("target"))
                else:
                    continue
                
                weighted_loss = self.loss_weights[loss_name] * loss_value
                losses[loss_name] = weighted_loss
                total_loss += weighted_loss
        
        losses["total"] = total_loss
        return losses

class SpectralLoss(nn.Module):
    """Spectral convergence loss for audio"""
    
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_spec = torch.stft(pred.squeeze(), n_fft=self.n_fft, hop_length=self.hop_length, 
                              return_complex=True)
        target_spec = torch.stft(target.squeeze(), n_fft=self.n_fft, hop_length=self.hop_length,
                                return_complex=True)
        
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)
        
        return torch.norm(target_mag - pred_mag, p="fro") / torch.norm(target_mag, p="fro")

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained features"""
    
    def __init__(self):
        super().__init__()
        # Placeholder for perceptual loss implementation
        self.mse = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simplified perceptual loss - in practice, would use pre-trained features
        return self.mse(pred, target)

class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss"""
    
    def __init__(self, resolutions: List[Tuple[int, int]] = None):
        super().__init__()
        if resolutions is None:
            resolutions = [(1024, 256), (2048, 512), (512, 128)]
        self.resolutions = resolutions
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for n_fft, hop_length in self.resolutions:
            pred_spec = torch.stft(pred.squeeze(), n_fft=n_fft, hop_length=hop_length,
                                  return_complex=True)
            target_spec = torch.stft(target.squeeze(), n_fft=n_fft, hop_length=hop_length,
                                    return_complex=True)
            
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)
            
            # Spectral convergence loss
            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / torch.norm(target_mag, p="fro")
            
            # Log STFT magnitude loss
            log_loss = nn.L1Loss()(torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7))
            
            total_loss += sc_loss + log_loss
        
        return total_loss / len(self.resolutions)

class VoiceTrainer:
    """Advanced trainer for voice synthesis models with multi-GPU support"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, 
                 gpu_manager=None, experiment_name: str = "voice_training"):
        """
        Initialize trainer with model and configuration
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            gpu_manager: GPU management utility
            experiment_name: Name for logging and checkpointing
        """
        self.model = model
        self.config = config
        self.gpu_manager = gpu_manager
        self.experiment_name = experiment_name
        
        # Setup device and distributed training
        self._setup_distributed()
        self._setup_device()
        
        # Initialize components
        self.loss_manager = LossManager(config)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.use_amp else None
        
        # Setup model for training
        self._setup_model()
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        logger.info(f"Trainer initialized for {experiment_name}")
        logger.info(f"Device: {self.device}, Distributed: {config.distributed}")
        
    def _setup_distributed(self):
        """Setup distributed training"""
        if self.config.distributed:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.config.local_rank = int(os.environ['LOCAL_RANK'])
                self.config.world_size = int(os.environ['WORLD_SIZE'])
                
                dist.init_process_group(backend='nccl')
                torch.cuda.set_device(self.config.local_rank)
                
                logger.info(f"Distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
            else:
                logger.warning("Distributed training requested but environment not set")
                self.config.distributed = False
    
    def _setup_device(self):
        """Setup training device"""
        if torch.cuda.is_available():
            if self.config.distributed:
                self.device = torch.device(f'cuda:{self.config.local_rank}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            logger.warning("CUDA not available, training on CPU")
    
    def _setup_model(self):
        """Setup model for training"""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Model compilation (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile")
        
        # Distributed data parallel
        if self.config.distributed:
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
            
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        params = self.model.parameters()
        
        if self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_params
            )
        elif self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_params
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.optimizer_params.get('momentum', 0.9),
                **{k: v for k, v in self.config.optimizer_params.items() if k != 'momentum'}
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.config.scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                **self.config.scheduler_params
            )
        elif self.config.scheduler_type.lower() == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                **self.config.scheduler_params
            )
        elif self.config.scheduler_type.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.scheduler_params.get('gamma', 0.95),
                **{k: v for k, v in self.config.scheduler_params.items() if k != 'gamma'}
            )
        elif self.config.scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_params.get('patience', 5),
                **{k: v for k, v in self.config.scheduler_params.items() if k != 'patience'}
            )
        elif self.config.scheduler_type.lower() == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                **self.config.scheduler_params
            )
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging and experiment tracking"""
        self.log_dir = Path(f"logs/{self.experiment_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config.local_rank == 0:  # Only log on main process
            self.writer = SummaryWriter(self.log_dir / "tensorboard")
            
            # Initialize Weights & Biases if available
            if WANDB_AVAILABLE:
                try:
                    wandb.init(
                        project="autovoice",
                        name=self.experiment_name,
                        config=self.config.__dict__
                    )
                    self.use_wandb = True
                except Exception as e:
                    logger.warning(f"Could not initialize wandb: {e}")
                    self.use_wandb = False
            else:
                self.use_wandb = False
        else:
            self.writer = None
            self.use_wandb = False
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with advanced features"""
        self.model.train()
        epoch_losses = {}
        num_batches = len(dataloader)
        
        # Create progress bar
        if self.config.local_rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
        
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                # Get model predictions
                predictions = self._forward_pass(batch)
                
                # Compute losses
                losses = self.loss_manager.compute_loss(predictions, batch)
                loss = losses["total"] / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate (for step-based schedulers)
                if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self._log_training_step(losses, accumulated_loss)
                    accumulated_loss = 0.0
            
            # Update epoch losses
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value.item())
            
            # Update progress bar
            if self.config.local_rank == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Compute average epoch losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate model on validation set"""
        self.model.eval()
        val_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", disable=self.config.local_rank != 0):
                batch = self._move_batch_to_device(batch)
                
                with autocast(enabled=self.config.use_amp):
                    predictions = self._forward_pass(batch)
                    losses = self.loss_manager.compute_loss(predictions, batch)
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in val_losses:
                        val_losses[loss_name] = []
                    val_losses[loss_name].append(loss_value.item())
        
        # Compute average validation losses
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        return avg_losses
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through model"""
        # Handle different batch formats
        if 'features' in batch:
            features = batch['features']
            speaker_id = batch.get('speaker_id', None)
        elif 'mel_spec' in batch:
            features = batch['mel_spec']
            speaker_id = batch.get('speaker_id', None)
        else:
            raise ValueError("Batch must contain 'features' or 'mel_spec'")
        
        # Forward through model
        if hasattr(self.model, 'forward_for_training'):
            output = self.model.forward_for_training(features, speaker_id)
        else:
            # Standard forward pass - handle speaker_id properly
            if speaker_id is not None:
                output = self.model(features, speaker_id)
            else:
                output = self.model(features)
        
        return {"output": output}
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
        return batch
    
    def _log_training_step(self, losses: Dict[str, torch.Tensor], accumulated_loss: float):
        """Log training step metrics"""
        if self.config.local_rank != 0:
            return
        
        # TensorBoard logging
        if self.writer:
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f"train/{loss_name}", loss_value.item(), self.global_step)
            
            self.writer.add_scalar("train/learning_rate", 
                                 self.optimizer.param_groups[0]['lr'], self.global_step)
            self.writer.add_scalar("train/accumulated_loss", accumulated_loss, self.global_step)
        
        # Weights & Biases logging
        if self.use_wandb:
            log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
            log_dict["train/learning_rate"] = self.optimizer.param_groups[0]['lr']
            log_dict["global_step"] = self.global_step
            wandb.log(log_dict, step=self.global_step)
    
    def train(self, train_dataloader, val_dataloader=None, checkpoint_path: Optional[str] = None):
        """Main training loop with comprehensive features"""
        logger.info("Starting training...")
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Set sampler epoch for distributed training
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            # Training
            train_losses = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_losses = {}
            if val_dataloader:
                val_losses = self.validate(val_dataloader)
                
                # Learning rate scheduling (for validation-based schedulers)
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses.get('total', val_losses.get('reconstruction', 0)))
            
            # Logging
            if self.config.local_rank == 0:
                self._log_epoch(epoch, train_losses, val_losses)
            
            # Checkpointing
            if self.config.local_rank == 0 and epoch % (self.config.save_interval // len(train_dataloader)) == 0:
                self.save_checkpoint(epoch, val_losses.get('total', float('inf')))
            
            # Early stopping
            if val_dataloader and self._check_early_stopping(val_losses.get('total', float('inf'))):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        
        # Cleanup
        if self.config.distributed:
            dist.destroy_process_group()
    
    def _log_epoch(self, epoch: int, train_losses: Dict[str, float], val_losses: Dict[str, float]):
        """Log epoch metrics"""
        # Console logging
        train_msg = " | ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
        logger.info(f"Epoch {epoch} - Train: {train_msg}")
        
        if val_losses:
            val_msg = " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
            logger.info(f"Epoch {epoch} - Val: {val_msg}")
        
        # TensorBoard logging
        if self.writer:
            for loss_name, loss_value in train_losses.items():
                self.writer.add_scalar(f"epoch/train_{loss_name}", loss_value, epoch)
            
            for loss_name, loss_value in val_losses.items():
                self.writer.add_scalar(f"epoch/val_{loss_name}", loss_value, epoch)
        
        # Weights & Biases logging
        if self.use_wandb:
            log_dict = {f"epoch/train_{k}": v for k, v in train_losses.items()}
            log_dict.update({f"epoch/val_{k}": v for k, v in val_losses.items()})
            log_dict["epoch"] = epoch
            wandb.log(log_dict, step=self.global_step)
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping criteria"""
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self, epoch: int, val_loss: float = float('inf'), 
                       additional_info: Optional[Dict[str, Any]] = None):
        """Save comprehensive training checkpoint"""
        if self.config.local_rank != 0:
            return
        
        checkpoint_dir = Path(f"checkpoints/{self.experiment_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Model state dict (handle DDP)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'val_loss': val_loss,
            'timestamp': time.time(),
            'pytorch_version': torch.__version__,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, path: str, strict: bool = True, load_optimizer: bool = True) -> Dict[str, Any]:
        """Load comprehensive training checkpoint"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        logger.info(f"Loading checkpoint from {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state dict (handle DDP)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if load_optimizer and self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        
        logger.info(f"Checkpoint loaded: epoch={self.epoch}, step={self.global_step}")
        
        return checkpoint
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints to save disk space"""
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        while len(checkpoints) > self.config.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.info(f"Removed old checkpoint: {oldest}")

# Example usage and factory functions
def create_trainer(model: nn.Module, config_dict: Dict[str, Any], **kwargs) -> VoiceTrainer:
    """Factory function to create trainer from configuration"""
    config = TrainingConfig(**config_dict)
    return VoiceTrainer(model, config, **kwargs)

def get_default_training_config() -> TrainingConfig:
    """Get default training configuration"""
    return TrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=100,
        use_amp=True,
        scheduler_type="cosine",
        loss_type="spectral"
    )