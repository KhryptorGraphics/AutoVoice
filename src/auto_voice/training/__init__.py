"""Training module for AutoVoice"""

from .trainer import (
    VoiceTrainer,
    VoiceConversionTrainer,
    TrainingConfig,
    LossManager,
    PitchConsistencyLoss,
    SpeakerSimilarityLoss,
    KLDivergenceLoss,
    FlowLogLikelihoodLoss
)
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
from .dataset import (
    PairedVoiceDataset,
    SingingAugmentation,
    PairedVoiceCollator,
    create_paired_voice_dataloader,
    create_paired_train_val_datasets
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
    'VoiceConversionTrainer',
    'TrainingConfig',
    'LossManager',
    'PitchConsistencyLoss',
    'SpeakerSimilarityLoss',
    'KLDivergenceLoss',
    'FlowLogLikelihoodLoss',

    # Data Pipeline
    'VoiceDataset',
    'PairedVoiceDataset',
    'AudioConfig',
    'AudioProcessor',
    'VoiceCollator',
    'PairedVoiceCollator',
    'DataAugmentation',
    'SingingAugmentation',
    'create_voice_dataloader',
    'create_paired_voice_dataloader',
    'create_train_val_datasets',
    'create_paired_train_val_datasets',
    'create_dataloaders',
    'preprocess_batch',

    # Checkpoint Management
    'CheckpointManager',
    'CheckpointConfig',
    'CheckpointMetadata',
    'create_checkpoint_manager',
    'get_default_checkpoint_config'
]
