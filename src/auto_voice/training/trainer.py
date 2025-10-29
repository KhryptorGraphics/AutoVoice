"""
Advanced training utilities for AutoVoice models with multi-GPU support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Import voice conversion components
try:
    from ..audio.pitch_extractor import SingingPitchExtractor
    from ..models.speaker_encoder import SpeakerEncoder
    from ..models.singing_voice_converter import SingingVoiceConverter
    VC_COMPONENTS_AVAILABLE = True
except ImportError:
    VC_COMPONENTS_AVAILABLE = False
    SingingPitchExtractor = None
    SpeakerEncoder = None
    SingingVoiceConverter = None

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

    # Voice conversion settings
    voice_conversion_mode: bool = False  # Enable voice conversion training
    sample_rate: int = 44100  # Sample rate for voice conversion (higher than TTS)
    extract_f0: bool = True  # Extract F0 during data loading
    extract_speaker_emb: bool = True  # Extract speaker embeddings

    # Voice conversion loss weights
    vc_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'mel_reconstruction': 45.0,
        'kl_divergence': 1.0,
        'pitch_consistency': 10.0,
        'speaker_similarity': 5.0,
        'flow_likelihood': 1.0,
        'stft': 2.5,
        'adversarial': 0.1  # Adversarial loss weight for GAN training
    })

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


class PitchConsistencyLoss(nn.Module):
    """Compute F0 RMSE between source and converted audio for pitch preservation."""

    def __init__(self, device: Optional[str] = None):
        """Initialize pitch consistency loss.

        Args:
            device: Device for computation ('cuda', 'cpu', etc.)
        """
        super().__init__()
        self.device = device
        self._extractor_available = True
        self._warned = False

        if VC_COMPONENTS_AVAILABLE:
            try:
                self.pitch_extractor = SingingPitchExtractor(device=device)
            except Exception as e:
                self._extractor_available = False
                self.pitch_extractor = None
                if not self._warned:
                    logger.warning(f"Pitch extractor initialization failed: {e}")
                    self._warned = True
        else:
            self._extractor_available = False
            self.pitch_extractor = None
            if not self._warned:
                logger.warning("SingingPitchExtractor not available, pitch consistency loss will be disabled")
                self._warned = True

    def forward(
        self,
        pred_audio: torch.Tensor,
        source_f0: torch.Tensor,
        sample_rate: int = 44100
    ) -> torch.Tensor:
        """Compute pitch consistency loss.

        Args:
            pred_audio: Predicted audio waveform [B, T_audio]
            source_f0: Source F0 contour [B, T_f0]
            sample_rate: Audio sample rate

        Returns:
            F0 RMSE loss
        """
        if not self._extractor_available:
            if not self._warned:
                logger.warning("Pitch loss returning zero (extractor unavailable)")
                self._warned = True
            return torch.tensor(0.0, device=pred_audio.device)

        try:
            # Extract F0 from predicted audio
            batch_size = pred_audio.size(0)
            pred_f0_list = []

            for i in range(batch_size):
                audio_np = pred_audio[i].detach().cpu().numpy()
                f0_result = self.pitch_extractor.extract_f0_contour(audio_np, sample_rate)
                pred_f0_list.append(f0_result['f0'])

            # Convert to tensor
            max_len = max(len(f0) for f0 in pred_f0_list)
            pred_f0 = torch.zeros(batch_size, max_len, device=pred_audio.device)

            for i, f0 in enumerate(pred_f0_list):
                f0_tensor = torch.from_numpy(f0).float().to(pred_audio.device)
                pred_f0[i, :len(f0)] = f0_tensor

            # Align lengths (interpolate to match)
            if pred_f0.size(1) != source_f0.size(1):
                pred_f0 = F.interpolate(
                    pred_f0.unsqueeze(1),
                    size=source_f0.size(1),
                    mode='linear',
                    align_corners=False
                ).squeeze(1)

            # Compute MSE on voiced frames (F0 > 0)
            voiced_mask = (source_f0 > 0) & (pred_f0 > 0)
            if voiced_mask.sum() > 0:
                loss = F.mse_loss(pred_f0[voiced_mask], source_f0[voiced_mask])
            else:
                loss = torch.tensor(0.0, device=pred_audio.device)

            return loss

        except Exception as e:
            logger.warning(f"Pitch consistency loss computation failed: {e}")
            return torch.tensor(0.0, device=pred_audio.device)


class SpeakerSimilarityLoss(nn.Module):
    """Compute cosine distance between target speaker embedding and converted audio embedding."""

    def __init__(self, device: Optional[str] = None):
        """Initialize speaker similarity loss.

        Args:
            device: Device for computation ('cuda', 'cpu', etc.)
        """
        super().__init__()
        self.device = device
        self._encoder_available = True
        self._warned = False

        if VC_COMPONENTS_AVAILABLE:
            try:
                self.speaker_encoder = SpeakerEncoder(device=device)
            except Exception as e:
                self._encoder_available = False
                self.speaker_encoder = None
                if not self._warned:
                    logger.warning(f"Speaker encoder initialization failed: {e}")
                    self._warned = True
        else:
            self._encoder_available = False
            self.speaker_encoder = None
            if not self._warned:
                logger.warning("SpeakerEncoder not available, speaker similarity loss will be disabled")
                self._warned = True

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_speaker_emb: torch.Tensor,
        sample_rate: int = 44100
    ) -> torch.Tensor:
        """Compute speaker similarity loss.

        Args:
            pred_audio: Predicted audio waveform [B, T_audio]
            target_speaker_emb: Target speaker embedding [B, 256]
            sample_rate: Audio sample rate

        Returns:
            Speaker cosine distance loss
        """
        if not self._encoder_available:
            if not self._warned:
                logger.warning("Speaker loss returning zero (encoder unavailable)")
                self._warned = True
            return torch.tensor(0.0, device=pred_audio.device)

        try:
            # Extract speaker embedding from predicted audio
            batch_size = pred_audio.size(0)
            pred_emb_list = []

            for i in range(batch_size):
                audio_np = pred_audio[i].detach().cpu().numpy()
                emb = self.speaker_encoder.extract_embedding(audio_np, sample_rate)
                pred_emb_list.append(emb)

            # Convert to tensor
            pred_emb = torch.from_numpy(np.stack(pred_emb_list)).float().to(pred_audio.device)

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(pred_emb, target_speaker_emb, dim=-1)

            # Convert to distance (1 - similarity)
            distance = 1.0 - cos_sim

            # Return mean distance
            return distance.mean()

        except Exception as e:
            logger.warning(f"Speaker similarity loss computation failed: {e}")
            return torch.tensor(0.0, device=pred_audio.device)


class KLDivergenceLoss(nn.Module):
    """KL divergence loss for variational models."""

    def forward(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            z_mean: Latent mean [B, D, T]
            z_logvar: Latent log variance [B, D, T]

        Returns:
            KL divergence loss
        """
        # KL divergence: -0.5 * sum(1 + log(var) - mean^2 - var)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        # Normalize by batch size and sequence length
        batch_size = z_mean.size(0)
        seq_len = z_mean.size(2) if z_mean.dim() > 2 else 1
        kl_loss = kl_loss / (batch_size * seq_len)

        return kl_loss


class FlowLogLikelihoodLoss(nn.Module):
    """Negative log-likelihood from normalizing flow."""

    def forward(self, logdet: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute flow log-likelihood loss.

        Args:
            logdet: Log determinant from flow [B, T]
            u: Flow output (prior space) [B, D, T]

        Returns:
            Negative log-likelihood loss
        """
        # Prior log-likelihood (standard normal)
        log_p_u = -0.5 * (np.log(2 * np.pi) + u.pow(2))
        log_p_u = log_p_u.sum(dim=1)  # Sum over channels

        # Total log-likelihood
        log_p_z = log_p_u + logdet

        # Negative log-likelihood
        nll = -log_p_z.mean()

        return nll


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
        if getattr(self.config, 'local_rank', 0) == 0:  # Only log on main process
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
        if getattr(self.config, 'local_rank', 0) == 0:
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
            if getattr(self.config, 'local_rank', 0) == 0:
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
            for batch in tqdm(dataloader, desc="Validation", disable=getattr(self.config, 'local_rank', 0) != 0):
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
        if getattr(self.config, 'local_rank', 0) != 0:
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
            if getattr(self.config, 'local_rank', 0) == 0:
                self._log_epoch(epoch, train_losses, val_losses)
            
            # Checkpointing
            if getattr(self.config, 'local_rank', 0) == 0 and epoch % (self.config.save_interval // len(train_dataloader)) == 0:
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
        if getattr(self.config, 'local_rank', 0) != 0:
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


class VoiceConversionTrainer(VoiceTrainer):
    """Specialized trainer for voice conversion with perceptual losses.

    Extends VoiceTrainer to add voice conversion-specific loss computation
    including pitch consistency, speaker similarity, flow-based losses, and adversarial training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        gpu_manager=None,
        experiment_name: str = "voice_conversion_training"
    ):
        """Initialize voice conversion trainer.

        IMPORTANT: The SingingVoiceConverter model MUST generate pred_audio during
        forward pass for pitch_consistency and speaker_similarity losses to work.
        Ensure model.forward() is called with use_vocoder=True (default).

        Args:
            model: SingingVoiceConverter model
            config: Training configuration with voice_conversion_mode=True
            gpu_manager: GPU manager instance
            experiment_name: Name for logging and checkpointing
        """
        # Initialize parent trainer
        super().__init__(model, config, gpu_manager, experiment_name)

        # Override loss manager to include voice conversion losses
        self._setup_vc_losses()

        # Use voice conversion loss weights
        self.vc_loss_weights = config.vc_loss_weights

        # Initialize discriminator for adversarial training
        self._setup_discriminator()

        # Warning flags for missing inputs (one-time warnings)
        self._warned_missing_inputs = False

        logger.info(f"VoiceConversionTrainer initialized with VC-specific losses and adversarial training")

    def _setup_vc_losses(self):
        """Setup voice conversion-specific losses."""
        if not VC_COMPONENTS_AVAILABLE:
            logger.warning("Voice conversion components not available, some losses may be disabled")

        # Create voice conversion loss functions
        self.pitch_loss = PitchConsistencyLoss(device=self.device)
        self.speaker_loss = SpeakerSimilarityLoss(device=self.device)
        self.kl_loss = KLDivergenceLoss()
        self.flow_loss = FlowLogLikelihoodLoss()

        # Keep existing STFT loss
        self.stft_loss = MultiResolutionSTFTLoss()

        logger.info("Voice conversion losses initialized")

    def _setup_discriminator(self):
        """Setup discriminator for adversarial training."""
        from ..models.discriminator import (
            VoiceDiscriminator,
            hinge_discriminator_loss,
            hinge_generator_loss,
            feature_matching_loss
        )

        # Create discriminator
        self.discriminator = VoiceDiscriminator(
            use_spectral_norm=False,  # Can enable for more stable training
            num_scales=3,
            channels=64
        )
        self.discriminator = self.discriminator.to(self.device)

        # Create separate optimizer for discriminator
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            **self.config.optimizer_params
        )

        # Store loss functions
        self.hinge_discriminator_loss = hinge_discriminator_loss
        self.hinge_generator_loss = hinge_generator_loss
        self.feature_matching_loss = feature_matching_loss

        # Feature matching weight (small, for stability)
        self.feature_matching_weight = 2.0

        logger.info("Discriminator and adversarial training components initialized")

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for voice conversion model with safe handling of optional inputs.

        Args:
            batch: Batch dict containing:
                - source_audio: Source audio waveform [B, T_audio]
                - target_mel: Target mel-spectrogram [B, T_mel, 80]
                - source_f0: Source F0 contour [B, T_f0] (optional)
                - target_speaker_emb: Target speaker embedding [B, 256] (optional)
                - mel_mask: Mask for variable lengths [B, 1, T_mel] (optional)

        Returns:
            Dict with model outputs:
                - pred_mel: Predicted mel-spectrogram
                - pred_audio: Predicted audio waveform (for perceptual losses)
                - z_mean: Latent mean
                - z_logvar: Latent log variance
                - z: Sampled latent
                - u: Flow output
                - logdet: Flow log determinant
                - cond: Conditioning features
        """
        # Safe access to optional inputs with defaults
        source_f0 = batch.get('source_f0')
        target_speaker_emb = batch.get('target_speaker_emb')

        # Handle missing source_f0: create zero tensor matching target_mel length
        if source_f0 is None:
            batch_size = batch['target_mel'].size(0)
            mel_length = batch['target_mel'].size(1)
            source_f0 = torch.zeros(batch_size, mel_length, device=batch['target_mel'].device)

            if not self._warned_missing_inputs:
                logger.warning(
                    "source_f0 not found in batch. Using zero tensor as default. "
                    "Pitch consistency loss will be zero for this batch. "
                    "This can happen when dataset is built with extract_f0=False."
                )
                self._warned_missing_inputs = True

        # Handle missing target_speaker_emb: pass None (model handles internally)
        if target_speaker_emb is None:
            if not self._warned_missing_inputs:
                logger.warning(
                    "target_speaker_emb not found in batch. Passing None to model. "
                    "Speaker similarity loss will be zero for this batch. "
                    "This can happen when dataset is built with extract_speaker_emb=False."
                )
                self._warned_missing_inputs = True

        # Call SingingVoiceConverter.forward() with use_vocoder=True
        # This generates pred_audio required for pitch/speaker losses
        outputs = self.model(
            source_audio=batch['source_audio'],
            target_mel=batch['target_mel'],
            source_f0=source_f0,
            target_speaker_emb=target_speaker_emb,
            source_sample_rate=self.config.sample_rate,
            x_mask=batch.get('mel_mask'),
            use_vocoder=True  # REQUIRED for perceptual losses
        )

        return outputs

    def _compute_voice_conversion_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        use_feature_matching: bool = False,
        real_features_list: Optional[List[List[torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all voice conversion losses.

        Args:
            predictions: Model predictions dict
            batch: Input batch dict
            use_feature_matching: If True, add feature matching loss
            real_features_list: Real audio discriminator features (for feature matching)

        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        # 1. Mel reconstruction loss
        if 'pred_mel' in predictions and 'target_mel' in batch:
            mel_loss = F.l1_loss(predictions['pred_mel'], batch['target_mel'])
            losses['mel_reconstruction'] = mel_loss
            total_loss += self.vc_loss_weights['mel_reconstruction'] * mel_loss

        # 2. KL divergence loss
        if 'z_mean' in predictions and 'z_logvar' in predictions:
            kl_loss_val = self.kl_loss(predictions['z_mean'], predictions['z_logvar'])
            losses['kl_divergence'] = kl_loss_val
            total_loss += self.vc_loss_weights['kl_divergence'] * kl_loss_val

        # 3. Flow log-likelihood loss
        if 'logdet' in predictions and 'u' in predictions:
            flow_loss_val = self.flow_loss(predictions['logdet'], predictions['u'])
            losses['flow_likelihood'] = flow_loss_val
            total_loss += self.vc_loss_weights['flow_likelihood'] * flow_loss_val

        # 4. Multi-resolution STFT loss (operates on audio waveforms)
        if 'pred_audio' in predictions and 'target_audio' in batch:
            # STFT loss should operate on raw audio waveforms, not mel spectrograms
            stft_loss_val = self.stft_loss(predictions['pred_audio'], batch['target_audio'])
            losses['stft'] = stft_loss_val
            total_loss += self.vc_loss_weights['stft'] * stft_loss_val
        else:
            # If audio predictions are not available, omit STFT loss
            losses['stft'] = torch.tensor(0.0, device=predictions['pred_mel'].device)

        # 5. Pitch consistency loss
        if 'pred_audio' in predictions and 'source_f0' in batch:
            # Convert predicted output to audio and compute pitch consistency
            try:
                pitch_loss_val = self.pitch_loss(
                    predictions['pred_audio'],
                    batch['source_f0'],
                    sample_rate=self.config.sample_rate
                )
                losses['pitch_consistency'] = pitch_loss_val
                total_loss += self.vc_loss_weights['pitch_consistency'] * pitch_loss_val

                # Assert that pitch loss is contributing
                if pitch_loss_val.item() > 0:
                    logger.debug(f"Pitch consistency loss: {pitch_loss_val.item():.6f}")
            except Exception as e:
                logger.warning(f"Pitch consistency loss computation failed: {e}")
                losses['pitch_consistency'] = torch.tensor(0.0, device=predictions['pred_mel'].device)
        else:
            # Log warning if pred_audio is missing
            if 'pred_audio' not in predictions:
                logger.warning(
                    "pred_audio not in predictions. Ensure SingingVoiceConverter.forward() "
                    "is called with use_vocoder=True. Skipping pitch consistency loss."
                )
            # Skip if audio predictions or F0 not available
            losses['pitch_consistency'] = torch.tensor(0.0, device=predictions['pred_mel'].device)

        # 6. Speaker similarity loss
        # Check if target_speaker_emb is available and not the default zero embedding
        target_speaker_emb = batch.get('target_speaker_emb')
        has_valid_speaker_emb = (
            target_speaker_emb is not None and
            torch.any(target_speaker_emb != 0)  # Not a zero embedding
        )

        if 'pred_audio' in predictions and has_valid_speaker_emb:
            # Compute speaker similarity using predicted audio
            try:
                speaker_loss_val = self.speaker_loss(
                    predictions['pred_audio'],
                    target_speaker_emb,
                    sample_rate=self.config.sample_rate
                )
                losses['speaker_similarity'] = speaker_loss_val
                total_loss += self.vc_loss_weights['speaker_similarity'] * speaker_loss_val
            except Exception as e:
                logger.warning(f"Speaker similarity loss computation failed: {e}")
                losses['speaker_similarity'] = torch.tensor(0.0, device=predictions['pred_mel'].device)
        else:
            # Skip if audio predictions or embeddings not available/valid
            if target_speaker_emb is None or not has_valid_speaker_emb:
                if target_speaker_emb is None:
                    logger.debug("target_speaker_emb is None, setting speaker_similarity loss to zero")
                else:
                    logger.debug("target_speaker_emb is default zero embedding, setting speaker_similarity loss to zero")
            losses['speaker_similarity'] = torch.tensor(0.0, device=predictions['pred_mel'].device)

        # 7. Adversarial loss (generator side)
        # NOTE: pred_audio is needed for discriminator, computed from vocoder if available
        if 'pred_audio' in predictions and self.vc_loss_weights.get('adversarial', 0) > 0:
            # Forward through discriminator
            fake_logits_list, fake_features_list = self.discriminator(predictions['pred_audio'])

            # Compute generator adversarial loss
            adv_loss_val = self.hinge_generator_loss(fake_logits_list)
            losses['adversarial'] = adv_loss_val
            total_loss += self.vc_loss_weights['adversarial'] * adv_loss_val

            # Feature matching loss (for training stability)
            if use_feature_matching and real_features_list is not None:
                fm_loss_val = self.feature_matching_loss(real_features_list, fake_features_list)
                losses['feature_matching'] = fm_loss_val
                total_loss += self.feature_matching_weight * fm_loss_val
            else:
                losses['feature_matching'] = torch.tensor(0.0, device=predictions['pred_mel'].device)
        else:
            losses['adversarial'] = torch.tensor(0.0, device=predictions['pred_mel'].device)
            losses['feature_matching'] = torch.tensor(0.0, device=predictions['pred_mel'].device)

        losses['total'] = total_loss

        return losses

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with voice conversion losses and two-step GAN training.

        Implements proper adversarial training with separate discriminator and generator steps:
        - STEP A: Update discriminator with real/fake pairs
        - STEP B: Update generator with all losses (including adversarial)

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dict with average losses for the epoch
        """
        self.model.train()
        self.discriminator.train()
        epoch_losses = {}
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=(getattr(self.config, 'local_rank', 0) != 0)
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # ==================================================================
            # Step 0: Forward pass (compute predictions once)
            # ==================================================================
            with autocast(enabled=self.config.use_amp):
                predictions = self._forward_pass(batch)

            # Check if adversarial training should be performed
            use_adversarial = (
                'pred_audio' in predictions and
                self.vc_loss_weights.get('adversarial', 0) > 0
            )

            # ==================================================================
            # STEP A: Discriminator Update (if adversarial training enabled)
            # ==================================================================
            d_loss = torch.tensor(0.0, device=self.device)
            real_features_list = None

            if use_adversarial:
                # Freeze generator parameters to prevent gradient flow
                for param in self.model.parameters():
                    param.requires_grad_(False)

                # Zero discriminator gradients
                self.discriminator_optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.config.use_amp):
                    # Get real and fake audio
                    real_audio = batch['target_audio']
                    fake_audio = predictions['pred_audio'].detach()  # Detach to avoid generator gradients

                    # Forward through discriminator
                    real_logits_list, real_features_list = self.discriminator(real_audio)
                    fake_logits_list, _ = self.discriminator(fake_audio)

                    # Compute discriminator loss (hinge loss)
                    d_loss = self.hinge_discriminator_loss(real_logits_list, fake_logits_list)

                # Backward pass for discriminator
                if self.scaler:
                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(self.discriminator_optimizer)
                    self.scaler.update()
                else:
                    d_loss.backward()
                    self.discriminator_optimizer.step()

                # Re-enable generator gradients
                for param in self.model.parameters():
                    param.requires_grad_(True)

            # ==================================================================
            # STEP B: Generator Update (all losses including adversarial)
            # ==================================================================
            with autocast(enabled=self.config.use_amp):
                # Compute all generator losses (including adversarial and feature matching)
                losses = self._compute_voice_conversion_losses(
                    predictions,
                    batch,
                    use_feature_matching=use_adversarial,
                    real_features_list=real_features_list
                )

            # Add discriminator loss to losses dict for logging
            losses['discriminator'] = d_loss

            # Backward pass for generator
            loss = losses['total'] / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

                self.global_step += 1

            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = 0.0
                epoch_losses[loss_name] += loss_value.item()

            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total'].item(),
                'mel': losses.get('mel_reconstruction', torch.tensor(0)).item(),
                'adv_g': losses.get('adversarial', torch.tensor(0)).item(),
                'd_loss': d_loss.item()
            })

            # Log to tensorboard/wandb
            if self.global_step % self.config.log_interval == 0:
                self._log_training_step(losses, losses['total'].item())

        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        return epoch_losses

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate with voice conversion metrics.

        Args:
            dataloader: Validation data loader

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        val_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)

                # Forward pass
                predictions = self._forward_pass(batch)
                losses = self._compute_voice_conversion_losses(predictions, batch)

                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in val_losses:
                        val_losses[loss_name] = 0.0
                    val_losses[loss_name] += loss_value.item()

                num_batches += 1

        # Average losses
        for loss_name in val_losses:
            val_losses[loss_name] /= num_batches

        # Log validation metrics
        if getattr(self.config, 'local_rank', 0) == 0:
            if self.writer:
                for loss_name, loss_value in val_losses.items():
                    self.writer.add_scalar(f"val/{loss_name}", loss_value, self.global_step)

            if self.use_wandb:
                log_dict = {f"val/{k}": v for k, v in val_losses.items()}
                wandb.log(log_dict, step=self.global_step)

        return val_losses


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