"""Training module for AutoVoice"""

from .trainer import VoiceTrainer, TrainingConfig, LossManager
from .data_pipeline import (
    VoiceDataset, 
    AudioConfig, 
    AudioProcessor,
    VoiceCollator,
    DataAugmentation,
    create_voice_dataloader,
    create_train_val_datasets,
    create_dataloaders,
    preprocess_batch
)
from .checkpoint_manager import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetadata,
    create_checkpoint_manager,
    get_default_checkpoint_config
)

__all__ = [
    # Trainer
    'VoiceTrainer',
    'TrainingConfig', 
    'LossManager',
    
    # Data Pipeline
    'VoiceDataset',
    'AudioConfig',
    'AudioProcessor', 
    'VoiceCollator',
    'DataAugmentation',
    'create_voice_dataloader',
    'create_train_val_datasets',
    'create_dataloaders',
    'preprocess_batch',
    
    # Checkpoint Management
    'CheckpointManager',
    'CheckpointConfig',
    'CheckpointMetadata',
    'create_checkpoint_manager',
    'get_default_checkpoint_config'
]
