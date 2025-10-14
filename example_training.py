"""
Example training script for AutoVoice models
Demonstrates usage of the complete training pipeline
"""

import torch
import torch.nn as nn
from pathlib import Path

from src.auto_voice.training import (
    VoiceTrainer, TrainingConfig,
    VoiceDataset, AudioConfig, create_voice_dataloader, create_train_val_datasets,
    CheckpointManager, CheckpointConfig
)
from src.auto_voice.models.transformer import VoiceTransformer

def main():
    """Main training example"""
    
    # Configuration
    audio_config = AudioConfig(
        sample_rate=22050,
        n_mels=80,
        max_audio_length=8192,
        n_fft=1024,
        hop_length=256
    )
    
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=50,
        use_amp=True,
        distributed=False,
        loss_type="spectral",
        scheduler_type="cosine",
        early_stopping_patience=10,
        gradient_accumulation_steps=2
    )
    
    checkpoint_config = CheckpointConfig(
        checkpoint_dir="checkpoints",
        max_checkpoints=5,
        save_optimizer=True,
        save_scheduler=True,
        verify_integrity=True,
        versioning_enabled=True
    )
    
    # Create model
    model = VoiceTransformer(
        input_dim=audio_config.n_mels,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets (assuming you have audio data)
    data_dir = "data/audio"  # Replace with your audio data directory
    
    if Path(data_dir).exists():
        # Create datasets
        train_dataset, val_dataset = create_train_val_datasets(
            data_dir=data_dir,
            val_split=0.1,
            audio_config=audio_config,
            train_transforms=None,  # Default augmentations will be used
        )
        
        # Create dataloaders
        train_dataloader = create_voice_dataloader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=4,
            distributed=training_config.distributed
        )
        
        val_dataloader = create_voice_dataloader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=4,
            distributed=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_config, 
            experiment_name="voice_transformer_experiment"
        )
        
        # Initialize trainer
        trainer = VoiceTrainer(
            model=model,
            config=training_config,
            experiment_name="voice_transformer_experiment"
        )
        
        # Start training
        try:
            trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader
            )
            
            print("Training completed successfully!")
            
            # Save final checkpoint
            checkpoint_manager.save_checkpoint(
                model=model,
                epoch=training_config.num_epochs,
                global_step=trainer.global_step,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                metrics={"final_training": True},
                is_best=False,
                checkpoint_name="final_checkpoint"
            )
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            
            # Save interruption checkpoint
            checkpoint_manager.save_checkpoint(
                model=model,
                epoch=trainer.epoch,
                global_step=trainer.global_step,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                metadata={"notes": "Training interrupted"},
                checkpoint_name="interrupted_checkpoint"
            )
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please provide a directory with audio files to train on.")
        
        # Demonstrate synthetic data training (for testing)
        print("Running with synthetic data for demonstration...")
        demo_training_synthetic_data(model, training_config, checkpoint_config)

def demo_training_synthetic_data(model, training_config, checkpoint_config):
    """Demonstrate training with synthetic data"""
    
    # Create synthetic dataset
    class SyntheticVoiceDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, seq_len=100, mel_dim=80):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.mel_dim = mel_dim
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate synthetic mel-spectrogram and audio
            mel_spec = torch.randn(self.seq_len, self.mel_dim)
            audio = torch.randn(self.seq_len * 4)  # Assume 4x upsampling
            speaker_id = torch.LongTensor([idx % 10])  # 10 synthetic speakers
            
            return {
                'features': mel_spec,
                'target_features': audio,
                'mel_spec': mel_spec,
                'audio': audio,
                'waveform': audio,
                'speaker_id': speaker_id,
                'lengths': torch.LongTensor([self.seq_len])
            }
    
    # Create synthetic datasets
    train_dataset = SyntheticVoiceDataset(num_samples=800)
    val_dataset = SyntheticVoiceDataset(num_samples=200)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0  # No multiprocessing for synthetic data
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Reduce epochs for demo
    training_config.num_epochs = 5
    training_config.log_interval = 10
    
    # Initialize trainer
    trainer = VoiceTrainer(
        model=model,
        config=training_config,
        experiment_name="synthetic_demo"
    )
    
    # Initialize checkpoint manager  
    checkpoint_manager = CheckpointManager(
        checkpoint_config,
        experiment_name="synthetic_demo"
    )
    
    print("Starting synthetic data training demo...")
    
    try:
        # Train for a few epochs
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        print("Demo training completed!")
        
        # Save demo checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            epoch=training_config.num_epochs,
            global_step=trainer.global_step,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            metrics={"demo_completed": True},
            checkpoint_name="demo_final"
        )
        
        # List saved checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        print(f"\nSaved checkpoints: {len(checkpoints)}")
        for i, ckpt in enumerate(checkpoints):
            print(f"  {i+1}. Epoch {ckpt['epoch']}, Step {ckpt['global_step']}")
        
    except Exception as e:
        print(f"Demo training failed: {e}")
        raise

if __name__ == "__main__":
    main()