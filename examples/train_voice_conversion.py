"""
Training script for Singing Voice Conversion (So-VITS-SVC)

Usage:
    # Train with default config
    python examples/train_voice_conversion.py

    # Train with custom config
    python examples/train_voice_conversion.py --config config/voice_conversion_training.yaml

    # Resume from checkpoint
    python examples/train_voice_conversion.py --resume checkpoints/voice_conversion/checkpoint_epoch_50.pt

    # Distributed training on 4 GPUs
    torchrun --nproc_per_node=4 examples/train_voice_conversion.py --distributed --num-gpus 4

    # Override specific parameters
    python examples/train_voice_conversion.py --epochs 200 --batch-size 16 --lr 1e-4

Adversarial Training:
    The trainer implements two-step GAN training for improved voice quality:
    - STEP A: Discriminator update with real/fake audio pairs
    - STEP B: Generator update with all losses (mel, pitch, speaker, adversarial, etc.)

    Adversarial training is enabled by default with a small weight (0.1). To disable:
    - Set 'adversarial': 0 in config YAML under 'losses' section
    - Or modify the default config below

    The discriminator uses multi-scale architecture and hinge loss for stability.
    Feature matching loss is automatically added during adversarial training
    for improved convergence.

Data Format:
    Metadata JSON should contain paired audio files:
    {
      "pairs": [
        {
          "source_file": "speaker1/song1.wav",
          "target_file": "speaker2/song1.wav",
          "source_speaker_id": "speaker1",
          "target_speaker_id": "speaker2"
        }
      ]
    }
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_voice.training import (
    VoiceConversionTrainer,
    TrainingConfig,
    PairedVoiceDataset,
    SingingAugmentation,
    AudioConfig,
    create_paired_voice_dataloader,
    create_paired_train_val_datasets,
    CheckpointManager,
    CheckpointConfig
)
from src.auto_voice.models import SingingVoiceConverter
from src.auto_voice.gpu import GPUManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for voice conversion training.

    Returns:
        Default config dict
    """
    return {
        'data': {
            'train_data_dir': 'data/voice_conversion/train',
            'val_data_dir': 'data/voice_conversion/val',
            'train_metadata': 'data/voice_conversion/train_pairs.json',
            'val_metadata': 'data/voice_conversion/val_pairs.json',
            'num_workers': 4,
            'cache_size': 500
        },
        'audio': {
            'sample_rate': 44100,
            'n_mels': 80,
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048,
            'max_audio_length': 220500  # 5 seconds at 44.1kHz
        },
        'model': {
            'latent_dim': 192,
            'mel_channels': 80,
            'content_encoder_type': 'hubert_soft',
            'num_flows': 4,
            'use_vocoder': True
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 2e-4,
            'num_epochs': 100,
            'use_amp': True,
            'distributed': False,
            'gradient_clip': 1.0,
            'gradient_accumulation_steps': 2,
            'voice_conversion_mode': True,
            'optimizer_type': 'adamw',
            'scheduler_type': 'cosine',
            'warmup_steps': 1000,
            'log_interval': 100,
            'save_interval': 1000,
            'validate_interval': 5  # Validate every 5 epochs
        },
        'losses': {
            'mel_reconstruction': 45.0,
            'kl_divergence': 1.0,
            'pitch_consistency': 10.0,
            'speaker_similarity': 5.0,
            'flow_likelihood': 1.0,
            'stft': 2.5,
            'adversarial': 0.1  # GAN adversarial loss (set to 0 to disable)
        },
        'checkpoint': {
            'checkpoint_dir': 'checkpoints/voice_conversion',
            'max_checkpoints': 5,
            'save_optimizer': True,
            'verify_integrity': True,
            'versioning_enabled': True
        },
        'augmentation': {
            'pitch_preserving_time_stretch': True,
            'formant_shift': True,
            'noise_injection': True,
            'augmentation_prob': 0.5,
            'pitch_preserving_time_stretch_strict': False
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration.

    Args:
        config: Config dict

    Returns:
        True if valid

    Raises:
        ValueError if invalid
    """
    # Check required sections
    required_sections = ['data', 'audio', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate data paths (only if not using synthetic data)
    if not config.get('use_synthetic_data', False):
        train_data_dir = Path(config['data']['train_data_dir'])
        if not train_data_dir.exists():
            logger.warning(f"Training data directory does not exist: {train_data_dir}")

        train_metadata = Path(config['data']['train_metadata'])
        if not train_metadata.exists():
            logger.warning(f"Training metadata file does not exist: {train_metadata}")

    # Validate model parameters
    if config['model']['latent_dim'] <= 0:
        raise ValueError("latent_dim must be positive")

    if config['model']['mel_channels'] <= 0:
        raise ValueError("mel_channels must be positive")

    # Validate training parameters
    if config['training']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")

    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")

    return True


def print_config(config: Dict[str, Any]):
    """Pretty-print configuration.

    Args:
        config: Config dict
    """
    logger.info("=" * 80)
    logger.info("Voice Conversion Training Configuration")
    logger.info("=" * 80)
    for section, values in config.items():
        logger.info(f"\n[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {values}")
    logger.info("=" * 80)


def setup_logging(log_dir: Path, debug: bool = False):
    """Set up logging to console and file.

    Args:
        log_dir: Directory for log files
        debug: Enable debug logging
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set log level
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def create_synthetic_dataset_demo() -> tuple:
    """Create synthetic paired dataset for testing.

    Returns:
        Tuple of (data_dir, train_metadata, val_metadata)
    """
    import numpy as np
    import soundfile as sf
    import tempfile

    logger.info("Creating synthetic dataset for demo...")

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create synthetic audio files
    sample_rate = 44100
    duration = 2  # seconds
    num_samples = sample_rate * duration

    num_pairs = 20

    pairs = []
    for i in range(num_pairs):
        # Generate synthetic audio (sine waves with different frequencies)
        t = np.linspace(0, duration, num_samples)
        source_freq = 220 + i * 10  # Varying frequency
        target_freq = 330 + i * 15

        source_audio = np.sin(2 * np.pi * source_freq * t).astype(np.float32)
        target_audio = np.sin(2 * np.pi * target_freq * t).astype(np.float32)

        # Normalize amplitude to [-1, 1] range
        source_audio = source_audio / (np.abs(source_audio).max() + 1e-8)
        target_audio = target_audio / (np.abs(target_audio).max() + 1e-8)

        # Apply simple fade in/out to avoid clicks
        fade_len = int(sample_rate * 0.01)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        source_audio[:fade_len] *= fade_in
        source_audio[-fade_len:] *= fade_out
        target_audio[:fade_len] *= fade_in
        target_audio[-fade_len:] *= fade_out

        # Save audio files as WAV using soundfile
        source_file = temp_dir / f"source_{i}.wav"
        target_file = temp_dir / f"target_{i}.wav"

        sf.write(str(source_file), source_audio, sample_rate)
        sf.write(str(target_file), target_audio, sample_rate)

        pairs.append({
            'source_file': str(source_file.relative_to(temp_dir)),
            'target_file': str(target_file.relative_to(temp_dir)),
            'source_speaker_id': f'speaker_{i % 3}',
            'target_speaker_id': f'speaker_{(i + 1) % 3}',
            'duration': duration
        })

    # Create metadata files
    train_pairs = pairs[:16]
    val_pairs = pairs[16:]

    train_metadata = temp_dir / 'train_pairs.json'
    val_metadata = temp_dir / 'val_pairs.json'

    with open(train_metadata, 'w') as f:
        json.dump({'pairs': train_pairs}, f)

    with open(val_metadata, 'w') as f:
        json.dump({'pairs': val_pairs}, f)

    logger.info(f"Created {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")
    logger.info(f"Synthetic data saved to: {temp_dir}")

    return temp_dir, train_metadata, val_metadata


def main(args):
    """Main training function.

    Args:
        args: Command-line arguments
    """
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Override with command-line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.data_dir:
        config['data']['train_data_dir'] = args.data_dir
        config['data']['val_data_dir'] = args.data_dir
    if args.checkpoint_dir:
        config['checkpoint']['checkpoint_dir'] = args.checkpoint_dir
    if args.distributed:
        config['training']['distributed'] = True

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)

    # Print configuration
    print_config(config)

    # Set up logging
    log_dir = Path(config['checkpoint']['checkpoint_dir']) / 'logs'
    setup_logging(log_dir, args.debug)

    # Initialize GPU manager
    gpu_manager = None
    if torch.cuda.is_available():
        try:
            gpu_manager = GPUManager()
            logger.info(f"GPU Manager initialized: {len(gpu_manager.available_gpus)} GPUs available")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU manager: {e}")

    # Create audio config
    audio_config = AudioConfig(
        sample_rate=config['audio']['sample_rate'],
        n_mels=config['audio']['n_mels'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        win_length=config['audio']['win_length']
    )

    # Create augmentation transforms
    train_transforms = []
    if config.get('augmentation', {}).get('pitch_preserving_time_stretch', True):
        train_transforms.append(SingingAugmentation.pitch_preserving_time_stretch)
    if config.get('augmentation', {}).get('formant_shift', True):
        train_transforms.append(SingingAugmentation.formant_shift)
    if config.get('augmentation', {}).get('noise_injection', True):
        train_transforms.append(SingingAugmentation.noise_injection_snr)

    # Get VTLP configuration
    vtlp_enabled = config.get('augmentation', {}).get('vtlp', {}).get('enabled', False)

    # Create datasets
    try:
        # Check if we should use synthetic data
        use_synthetic = args.use_synthetic_data or not Path(config['data']['train_metadata']).exists()

        if use_synthetic:
            logger.info("Using synthetic dataset for demo")
            data_dir, train_metadata, val_metadata = create_synthetic_dataset_demo()
        else:
            data_dir = config['data']['train_data_dir']
            train_metadata = config['data']['train_metadata']
            val_metadata = config['data']['val_metadata']

        train_dataset, val_dataset = create_paired_train_val_datasets(
            data_dir=data_dir,
            train_metadata=str(train_metadata),
            val_metadata=str(val_metadata),
            audio_config=audio_config,
            train_transforms=train_transforms,
            augmentation_prob=config.get('augmentation', {}).get('augmentation_prob', 0.5),
            extract_f0=config['training'].get('extract_f0', True),
            extract_speaker_emb=config['training'].get('extract_speaker_emb', True),
            device=gpu_manager.device if gpu_manager else None,
            gpu_manager=gpu_manager,
            pitch_time_stretch_strict=config.get('augmentation', {}).get('pitch_preserving_time_stretch_strict', False),
            enable_vtlp=vtlp_enabled
        )

        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        raise

    # Create dataloaders
    train_dataloader = create_paired_voice_dataloader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        distributed=config['training'].get('distributed', False)
    )

    val_dataloader = create_paired_voice_dataloader(
        dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        distributed=False
    )

    # Initialize model
    logger.info("Initializing SingingVoiceConverter model...")
    model = SingingVoiceConverter(
        latent_dim=config['model']['latent_dim'],
        mel_channels=config['model']['mel_channels'],
        content_encoder_type=config['model'].get('content_encoder_type', 'hubert_soft'),
        num_flows=config['model'].get('num_flows', 4),
        use_vocoder=config['model'].get('use_vocoder', True)
    )

    # Move model to GPU if available
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Create training config
    training_config = TrainingConfig(
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['num_epochs'],
        use_amp=config['training'].get('use_amp', True),
        distributed=config['training'].get('distributed', False),
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 2),
        voice_conversion_mode=True,
        sample_rate=config['audio']['sample_rate'],
        extract_f0=config['training'].get('extract_f0', True),
        extract_speaker_emb=config['training'].get('extract_speaker_emb', True),
        vc_loss_weights=config.get('losses', {}),
        optimizer_type=config['training'].get('optimizer_type', 'adamw'),
        scheduler_type=config['training'].get('scheduler_type', 'cosine'),
        warmup_steps=config['training'].get('warmup_steps', 1000),
        log_interval=config['training'].get('log_interval', 100),
        save_interval=config['training'].get('save_interval', 1000),
        validate_interval=config['training'].get('validate_interval', 500)
    )

    # Initialize trainer
    logger.info("Initializing VoiceConversionTrainer...")
    trainer = VoiceConversionTrainer(
        model=model,
        config=training_config,
        gpu_manager=gpu_manager,
        experiment_name=args.experiment_name
    )

    # Initialize checkpoint manager
    checkpoint_config = CheckpointConfig(
        checkpoint_dir=config['checkpoint']['checkpoint_dir'],
        max_checkpoints=config['checkpoint'].get('max_checkpoints', 5),
        save_optimizer=config['checkpoint'].get('save_optimizer', True),
        verify_integrity=config['checkpoint'].get('verify_integrity', True),
        versioning_enabled=config['checkpoint'].get('versioning_enabled', True)
    )

    checkpoint_manager = CheckpointManager(checkpoint_config)

    # Resume from checkpoint if specified
    resume_checkpoint_path = None
    if args.resume:
        resume_checkpoint_path = args.resume
        logger.info(f"Will resume training from: {resume_checkpoint_path}")

    # Start training
    logger.info("=" * 80)
    logger.info("Starting Voice Conversion Training")
    logger.info("=" * 80)

    try:
        # Training loop
        best_val_loss = float('inf')
        best_checkpoint_path = None

        for epoch in range(training_config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")

            # Train epoch
            train_losses = trainer.train_epoch(train_dataloader, epoch)
            logger.info(f"Training losses: {train_losses}")

            # Validate at specified epoch intervals
            # validate_interval is interpreted as number of epochs between validations
            if (epoch + 1) % training_config.validate_interval == 0:
                val_losses = trainer.validate(val_dataloader)
                logger.info(f"Validation losses: {val_losses}")

                # Track best checkpoint
                is_best = val_losses['total'] < best_val_loss
                if is_best:
                    best_val_loss = val_losses['total']

                # Save checkpoint with correct API
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    epoch=epoch,
                    global_step=trainer.global_step,
                    metrics={'val_loss': val_losses['total']},
                    is_best=is_best
                )
                logger.info(f"Checkpoint saved: {checkpoint_path}")

                if is_best:
                    best_checkpoint_path = checkpoint_path

        # Training complete
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)

        # Export best model
        if best_checkpoint_path:
            export_destination = Path(config['checkpoint']['checkpoint_dir']) / f"{args.experiment_name}_best.pt"
            success = checkpoint_manager.export_checkpoint(
                checkpoint_path=best_checkpoint_path,
                export_path=str(export_destination)
            )
            if success:
                logger.info(f"Best model exported to: {export_destination}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # Save emergency checkpoint
        emergency_path = Path(config['checkpoint']['checkpoint_dir']) / 'emergency_checkpoint.pt'
        torch.save({
            'model': model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'epoch': trainer.epoch,
            'global_step': trainer.global_step
        }, emergency_path)
        logger.info(f"Emergency checkpoint saved to: {emergency_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train Singing Voice Conversion model')

    # Configuration
    parser.add_argument('--config', type=str, help='Path to config YAML file')

    # Data
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    parser.add_argument('--use-synthetic-data', action='store_true', help='Use synthetic data for demo')

    # Training
    parser.add_argument('--epochs', type=int, help='Number of epochs (override config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (override config)')
    parser.add_argument('--lr', type=float, help='Learning rate (override config)')

    # Checkpoint
    parser.add_argument('--checkpoint-dir', type=str, help='Override checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')

    # Device
    parser.add_argument('--device', type=str, help="Device ('cuda', 'cpu', 'cuda:0')")
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs for distributed training')

    # Logging
    parser.add_argument('--experiment-name', type=str, default='voice_conversion', help='Experiment name for logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        main(args)
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
