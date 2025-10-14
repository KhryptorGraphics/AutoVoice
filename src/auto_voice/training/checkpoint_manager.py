"""
Advanced checkpoint management for AutoVoice models
Handles model saving, loading, versioning, and state management
"""

import torch
import torch.nn as nn
import os
import json
import shutil
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import tempfile
from datetime import datetime, timezone
import pickle
import warnings

logger = logging.getLogger(__name__)

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    # Directory settings
    checkpoint_dir: str = "checkpoints"
    backup_dir: str = "checkpoints_backup"
    max_checkpoints: int = 5
    
    # Saving options
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_random_states: bool = True
    save_model_architecture: bool = True
    
    # Compression and storage
    compress_checkpoints: bool = False
    use_safe_serialization: bool = True
    verify_integrity: bool = True
    
    # Versioning
    versioning_enabled: bool = True
    auto_increment_version: bool = True
    
    # Metadata tracking
    track_git_info: bool = True
    track_system_info: bool = True
    track_model_hash: bool = True

@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint tracking"""
    # Basic info
    timestamp: float
    epoch: int
    global_step: int
    model_name: str
    experiment_name: str
    
    # Performance metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_metric: Optional[float] = None
    
    # Model info
    model_hash: Optional[str] = None
    num_parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    
    # Training state
    learning_rate: Optional[float] = None
    optimizer_state_available: bool = False
    scheduler_state_available: bool = False
    
    # System info
    pytorch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    hostname: Optional[str] = None
    
    # Git info (if available)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    
    # User defined
    notes: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class CheckpointManager:
    """Advanced checkpoint management system"""
    
    def __init__(self, config: CheckpointConfig, experiment_name: str = "default"):
        """
        Initialize checkpoint manager
        
        Args:
            config: Checkpoint configuration
            experiment_name: Name of the experiment for organizing checkpoints
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
        self.backup_dir = Path(config.backup_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if config.versioning_enabled:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoint_history = []
        self.load_checkpoint_history()
        
        # Best model tracking
        self.best_checkpoint = None
        self.best_metric_value = float('inf')
        
        logger.info(f"CheckpointManager initialized for experiment: {experiment_name}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        global_step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save comprehensive checkpoint with metadata
        
        Args:
            model: PyTorch model to save
            epoch: Current epoch
            global_step: Current global step
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            scaler: GradScaler for mixed precision (optional)
            metrics: Training metrics
            metadata: Additional metadata
            is_best: Whether this is the best checkpoint
            checkpoint_name: Custom checkpoint name (optional)
        
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch}_step_{global_step}_{timestamp}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = self._prepare_checkpoint_data(
            model, epoch, global_step, optimizer, scheduler, scaler, 
            metrics, metadata
        )
        
        # Create metadata
        checkpoint_metadata = self._create_metadata(
            model, epoch, global_step, metrics, metadata, checkpoint_name
        )
        checkpoint_data['metadata'] = asdict(checkpoint_metadata)
        
        # Save checkpoint with error handling
        try:
            self._save_checkpoint_safely(checkpoint_data, checkpoint_path)
            
            # Verify checkpoint integrity
            if self.config.verify_integrity:
                self._verify_checkpoint_integrity(checkpoint_path)
            
            # Update checkpoint history
            self._update_checkpoint_history(checkpoint_path, checkpoint_metadata, is_best)
            
            # Handle best model tracking
            if is_best or self._is_new_best(metrics):
                self._save_best_checkpoint(checkpoint_path, checkpoint_metadata)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        strict: bool = True,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint with comprehensive state restoration
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            scaler: GradScaler to load state into (optional)
            strict: Strict loading for model state dict
            device: Device to map tensors to
        
        Returns:
            Loaded checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load checkpoint with device mapping
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Verify checkpoint integrity
            if self.config.verify_integrity:
                self._verify_loaded_checkpoint(checkpoint, checkpoint_path)
            
            # Load model state
            self._load_model_state(model, checkpoint, strict)
            
            # Load optimizer state
            if optimizer and self.config.save_optimizer and 'optimizer_state_dict' in checkpoint:
                self._load_optimizer_state(optimizer, checkpoint)
            
            # Load scheduler state
            if scheduler and self.config.save_scheduler and 'scheduler_state_dict' in checkpoint:
                self._load_scheduler_state(scheduler, checkpoint)
            
            # Load scaler state
            if scaler and 'scaler_state_dict' in checkpoint:
                self._load_scaler_state(scaler, checkpoint)
            
            # Restore random states
            if self.config.save_random_states and 'random_states' in checkpoint:
                self._restore_random_states(checkpoint['random_states'])
            
            # Log checkpoint info
            metadata = checkpoint.get('metadata', {})
            epoch = checkpoint.get('epoch', 'unknown')
            step = checkpoint.get('global_step', 'unknown')
            logger.info(f"Checkpoint loaded successfully: epoch={epoch}, step={step}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the best saved checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None
        
        return self.load_checkpoint(
            str(best_path), model, optimizer, scheduler, scaler, device=device
        )
    
    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint"""
        if not self.checkpoint_history:
            logger.warning("No checkpoints found in history")
            return None
        
        latest_entry = max(self.checkpoint_history, key=lambda x: x['timestamp'])
        checkpoint_path = latest_entry['path']
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Latest checkpoint not found: {checkpoint_path}")
            return None
        
        return self.load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device=device
        )
    
    def list_checkpoints(self, sort_by: str = "timestamp") -> List[Dict[str, Any]]:
        """
        List all available checkpoints
        
        Args:
            sort_by: Sort criterion ('timestamp', 'epoch', 'val_loss')
        
        Returns:
            List of checkpoint information
        """
        if sort_by == "timestamp":
            return sorted(self.checkpoint_history, key=lambda x: x['timestamp'], reverse=True)
        elif sort_by == "epoch":
            return sorted(self.checkpoint_history, key=lambda x: x.get('epoch', 0), reverse=True)
        elif sort_by == "val_loss":
            return sorted(self.checkpoint_history, 
                         key=lambda x: x.get('val_loss', float('inf')))
        else:
            return self.checkpoint_history
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint.get('metadata', {})
        except Exception as e:
            logger.error(f"Failed to load checkpoint info: {e}")
            return None
    
    def remove_checkpoint(self, checkpoint_path: str) -> bool:
        """Remove a specific checkpoint"""
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                
                # Update history
                self.checkpoint_history = [
                    entry for entry in self.checkpoint_history 
                    if entry['path'] != checkpoint_path
                ]
                self._save_checkpoint_history()
                
                logger.info(f"Checkpoint removed: {checkpoint_path}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to remove checkpoint: {e}")
            return False
    
    def create_checkpoint_backup(self, checkpoint_path: str) -> Optional[str]:
        """Create a backup of a specific checkpoint"""
        if not self.config.versioning_enabled:
            logger.warning("Versioning not enabled")
            return None
        
        try:
            backup_path = self.backup_dir / f"backup_{int(time.time())}_{Path(checkpoint_path).name}"
            shutil.copy2(checkpoint_path, backup_path)
            logger.info(f"Checkpoint backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def export_checkpoint(
        self, 
        checkpoint_path: str, 
        export_path: str, 
        include_optimizer: bool = False
    ) -> bool:
        """
        Export checkpoint for inference (remove training-specific data)
        
        Args:
            checkpoint_path: Source checkpoint path
            export_path: Destination path for exported model
            include_optimizer: Whether to include optimizer state
        
        Returns:
            Success status
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create minimal checkpoint for inference
            export_data = {
                'model_state_dict': checkpoint['model_state_dict'],
                'model_config': checkpoint.get('model_config'),
                'metadata': checkpoint.get('metadata'),
                'epoch': checkpoint.get('epoch'),
                'pytorch_version': checkpoint.get('pytorch_version')
            }
            
            if include_optimizer:
                export_data['optimizer_state_dict'] = checkpoint.get('optimizer_state_dict')
                export_data['scheduler_state_dict'] = checkpoint.get('scheduler_state_dict')
            
            # Save exported checkpoint
            torch.save(export_data, export_path)
            logger.info(f"Checkpoint exported: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export checkpoint: {e}")
            return False
    
    def _prepare_checkpoint_data(
        self,
        model: nn.Module,
        epoch: int,
        global_step: int,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        metrics: Optional[Dict[str, float]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare checkpoint data dictionary"""
        # Get model state dict (handle DDP)
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        checkpoint_data = {
            'model_state_dict': model_state_dict,
            'epoch': epoch,
            'global_step': global_step,
            'timestamp': time.time(),
            'pytorch_version': torch.__version__,
        }
        
        # Add optimizer state
        if optimizer and self.config.save_optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler and self.config.save_scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add scaler state
        if scaler:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        # Add random states
        if self.config.save_random_states:
            checkpoint_data['random_states'] = self._get_random_states()
        
        # Add model architecture
        if self.config.save_model_architecture:
            checkpoint_data['model_config'] = self._get_model_config(model)
        
        # Add metrics
        if metrics:
            checkpoint_data.update(metrics)
        
        # Add custom metadata
        if metadata:
            checkpoint_data['custom_metadata'] = metadata
        
        return checkpoint_data
    
    def _create_metadata(
        self,
        model: nn.Module,
        epoch: int,
        global_step: int,
        metrics: Optional[Dict[str, float]],
        metadata: Optional[Dict[str, Any]],
        checkpoint_name: str
    ) -> CheckpointMetadata:
        """Create comprehensive checkpoint metadata"""
        # Basic info
        checkpoint_metadata = CheckpointMetadata(
            timestamp=time.time(),
            epoch=epoch,
            global_step=global_step,
            model_name=model.__class__.__name__,
            experiment_name=self.experiment_name
        )
        
        # Performance metrics
        if metrics:
            checkpoint_metadata.train_loss = metrics.get('train_loss')
            checkpoint_metadata.val_loss = metrics.get('val_loss')
            checkpoint_metadata.best_metric = metrics.get('best_metric')
        
        # Model info
        if self.config.track_model_hash:
            checkpoint_metadata.model_hash = self._compute_model_hash(model)
        checkpoint_metadata.num_parameters = sum(p.numel() for p in model.parameters())
        
        # System info
        if self.config.track_system_info:
            checkpoint_metadata.pytorch_version = torch.__version__
            if torch.cuda.is_available():
                checkpoint_metadata.cuda_version = torch.version.cuda
            import socket
            checkpoint_metadata.hostname = socket.gethostname()
        
        # Git info
        if self.config.track_git_info:
            git_info = self._get_git_info()
            checkpoint_metadata.git_commit = git_info.get('commit')
            checkpoint_metadata.git_branch = git_info.get('branch')
            checkpoint_metadata.git_dirty = git_info.get('dirty')
        
        # User metadata
        if metadata:
            checkpoint_metadata.notes = metadata.get('notes')
            checkpoint_metadata.tags = metadata.get('tags', [])
        
        return checkpoint_metadata
    
    def _save_checkpoint_safely(self, checkpoint_data: Dict[str, Any], checkpoint_path: Path):
        """Save checkpoint with atomic write operation"""
        # Use temporary file for atomic write
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        try:
            torch.save(checkpoint_data, temp_path)
            
            # Atomic move
            temp_path.rename(checkpoint_path)
            
        except Exception as e:
            # Cleanup temporary file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path):
        """Verify checkpoint integrity by loading and basic checks"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required fields
            required_fields = ['model_state_dict', 'epoch', 'global_step', 'timestamp']
            for field in required_fields:
                if field not in checkpoint:
                    raise ValueError(f"Missing required field: {field}")
            
            # Verify model state dict structure
            model_state = checkpoint['model_state_dict']
            if not isinstance(model_state, dict):
                raise ValueError("Invalid model_state_dict format")
            
            logger.debug(f"Checkpoint integrity verified: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Checkpoint integrity check failed: {e}")
            raise
    
    def _verify_loaded_checkpoint(self, checkpoint: Dict[str, Any], checkpoint_path: str):
        """Verify loaded checkpoint data"""
        # Check checkpoint structure
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        
        # Verify model hash if available
        metadata = checkpoint.get('metadata', {})
        if 'model_hash' in metadata and self.config.track_model_hash:
            # Could verify against expected model hash
            pass
        
        logger.debug(f"Loaded checkpoint verification passed: {checkpoint_path}")
    
    def _load_model_state(self, model: nn.Module, checkpoint: Dict[str, Any], strict: bool):
        """Load model state with error handling"""
        try:
            # Handle DDP models
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.debug("Model state loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            if strict:
                raise
            else:
                logger.warning("Continuing with partial model loading due to strict=False")
    
    def _load_optimizer_state(self, optimizer: torch.optim.Optimizer, checkpoint: Dict[str, Any]):
        """Load optimizer state with error handling"""
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.debug("Optimizer state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    def _load_scheduler_state(self, scheduler: torch.optim.lr_scheduler._LRScheduler, checkpoint: Dict[str, Any]):
        """Load scheduler state with error handling"""
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.debug("Scheduler state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
    
    def _load_scaler_state(self, scaler: torch.cuda.amp.GradScaler, checkpoint: Dict[str, Any]):
        """Load GradScaler state with error handling"""
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.debug("GradScaler state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GradScaler state: {e}")
    
    def _get_random_states(self) -> Dict[str, Any]:
        """Capture random states for reproducibility"""
        return {
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': None  # Python random state if needed
        }
    
    def _restore_random_states(self, random_states: Dict[str, Any]):
        """Restore random states for reproducibility"""
        try:
            if 'torch' in random_states and random_states['torch'] is not None:
                torch.set_rng_state(random_states['torch'])
            
            if 'torch_cuda' in random_states and random_states['torch_cuda'] is not None:
                torch.cuda.set_rng_state_all(random_states['torch_cuda'])
            
            if 'numpy' in random_states and random_states['numpy'] is not None:
                np.random.set_state(random_states['numpy'])
            
            logger.debug("Random states restored successfully")
            
        except Exception as e:
            logger.warning(f"Failed to restore random states: {e}")
    
    def _get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model configuration"""
        config = {
            'class_name': model.__class__.__name__,
            'module': model.__class__.__module__,
        }
        
        # Try to get model-specific config
        if hasattr(model, 'config'):
            config['model_config'] = model.config
        elif hasattr(model, 'get_config'):
            config['model_config'] = model.get_config()
        
        return config
    
    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute hash of model architecture for verification"""
        model_str = str(model)
        return hashlib.md5(model_str.encode()).hexdigest()
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information"""
        git_info = {}
        try:
            import subprocess
            
            # Get current commit
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            git_info['commit'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                  capture_output=True, text=True, check=True)
            git_info['branch'] = result.stdout.strip()
            
            # Check if working directory is dirty
            result = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD', '--'],
                                  capture_output=True)
            git_info['dirty'] = result.returncode != 0
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git not available or not in a git repository
            pass
        
        return git_info
    
    def _update_checkpoint_history(self, checkpoint_path: Path, metadata: CheckpointMetadata, is_best: bool):
        """Update checkpoint history tracking"""
        history_entry = {
            'path': str(checkpoint_path),
            'timestamp': metadata.timestamp,
            'epoch': metadata.epoch,
            'global_step': metadata.global_step,
            'train_loss': metadata.train_loss,
            'val_loss': metadata.val_loss,
            'is_best': is_best,
            'model_name': metadata.model_name
        }
        
        self.checkpoint_history.append(history_entry)
        self._save_checkpoint_history()
    
    def _is_new_best(self, metrics: Optional[Dict[str, float]]) -> bool:
        """Check if this is a new best checkpoint"""
        if not metrics or 'val_loss' not in metrics:
            return False
        
        val_loss = metrics['val_loss']
        if val_loss < self.best_metric_value:
            self.best_metric_value = val_loss
            return True
        
        return False
    
    def _save_best_checkpoint(self, checkpoint_path: Path, metadata: CheckpointMetadata):
        """Save or update the best checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pt"
        
        try:
            shutil.copy2(checkpoint_path, best_path)
            self.best_checkpoint = {
                'path': str(best_path),
                'source_path': str(checkpoint_path),
                'metadata': asdict(metadata)
            }
            logger.info(f"New best checkpoint saved: {best_path}")
        except Exception as e:
            logger.error(f"Failed to save best checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        # Get all checkpoint files (excluding best model)
        checkpoint_files = [
            entry for entry in self.checkpoint_history 
            if Path(entry['path']).name != "best_model.pt"
        ]
        
        if len(checkpoint_files) <= self.config.max_checkpoints:
            return
        
        # Sort by timestamp and remove oldest
        checkpoint_files.sort(key=lambda x: x['timestamp'])
        files_to_remove = checkpoint_files[:-self.config.max_checkpoints]
        
        for entry in files_to_remove:
            checkpoint_path = entry['path']
            if os.path.exists(checkpoint_path):
                # Create backup if versioning enabled
                if self.config.versioning_enabled:
                    self.create_checkpoint_backup(checkpoint_path)
                
                # Remove checkpoint
                self.remove_checkpoint(checkpoint_path)
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to file"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.checkpoint_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint history: {e}")
    
    def load_checkpoint_history(self):
        """Load checkpoint history from file"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.checkpoint_history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint history: {e}")
                self.checkpoint_history = []
        else:
            self.checkpoint_history = []

# Factory functions and utilities
def create_checkpoint_manager(config_dict: Dict[str, Any], experiment_name: str = "default") -> CheckpointManager:
    """Factory function to create checkpoint manager from configuration"""
    config = CheckpointConfig(**config_dict)
    return CheckpointManager(config, experiment_name)

def get_default_checkpoint_config() -> CheckpointConfig:
    """Get default checkpoint configuration"""
    return CheckpointConfig(
        max_checkpoints=5,
        save_optimizer=True,
        save_scheduler=True,
        verify_integrity=True,
        versioning_enabled=True
    )

# Example usage
if __name__ == "__main__":
    # Example usage
    config = get_default_checkpoint_config()
    manager = CheckpointManager(config, "test_experiment")
    
    # Simulate saving a checkpoint
    # model = SomeModel()
    # manager.save_checkpoint(model, epoch=1, global_step=100, metrics={'val_loss': 0.5})
    
    print(f"Checkpoint manager initialized: {manager.checkpoint_dir}")