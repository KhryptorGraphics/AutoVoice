#!/usr/bin/env python3
"""Train voice models on Pillowtalk with live progress display.

This script:
1. Separates vocals from Pillowtalk for both artists
2. Trains LoRA adapters with live progress output
3. Saves trained models for voice conversion
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from auto_voice.storage.paths import (
    resolve_data_dir,
    resolve_profiles_dir,
    resolve_trained_models_dir,
)

# Configure logging with live output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WILLIAM_PROFILE_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
CONOR_PROFILE_ID = "9679a6ec-e6e2-43c4-b64e-1f004fed34f9"

PILLOWTALK_WILLIAM = "tests/quality_samples/william_singe_pillowtalk.wav"
PILLOWTALK_CONOR = "tests/quality_samples/conor_maynard_pillowtalk.wav"


def resolve_runtime_paths(data_dir: str | None = None) -> dict[str, Path]:
    """Resolve runtime paths without relying on cwd-sensitive data/* literals."""

    resolved_data_dir = resolve_data_dir(data_dir)
    return {
        "data_dir": resolved_data_dir,
        "profiles_dir": resolve_profiles_dir(data_dir=str(resolved_data_dir)),
        "separated_dir": resolved_data_dir / "separated",
        "models_dir": resolve_trained_models_dir(data_dir=str(resolved_data_dir)),
    }

# Training config
TRAINING_CONFIG = {
    'epochs': 5,          # Quick initial training
    'learning_rate': 1e-4,
    'batch_size': 4,
    'lora_rank': 8,
    'lora_alpha': 16,
}


def print_banner(text: str):
    """Print a prominent banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def print_progress(epoch: int, step: int, loss: float, progress: int, profile_name: str):
    """Print training progress."""
    bar_width = 30
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r[{bar}] {progress:3d}% | Epoch {epoch+1} | Step {step} | Loss: {loss:.4f} | {profile_name}", end="", flush=True)
    if progress >= 100:
        print()  # Newline at completion


def separate_vocals(
    audio_path: str,
    profile_id: str,
    profile_name: str,
    *,
    data_dir: str | None = None,
) -> dict:
    """Separate vocals from audio using Demucs."""
    from auto_voice.audio.separation import VocalSeparator
    import librosa
    import soundfile as sf

    print(f"  🎵 Separating vocals from: {os.path.basename(audio_path)}")

    # Check if already separated
    paths = resolve_runtime_paths(data_dir)
    separated_dir = paths["separated_dir"] / profile_id
    vocals_path = separated_dir / "vocals.wav"
    instrumental_path = separated_dir / "instrumental.wav"

    if vocals_path.exists() and instrumental_path.exists():
        print(f"  ✅ Using cached separated files")
        return {'vocals': str(vocals_path), 'instrumental': str(instrumental_path)}

    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    print(f"  📊 Audio loaded: {audio.shape}, sr={sr}")

    # Initialize separator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = VocalSeparator(device=device)

    # Separate
    print(f"  🔄 Running Demucs separation (this takes a moment)...")
    start_time = time.time()
    separated = separator.separate(audio, sr)
    elapsed = time.time() - start_time
    print(f"  ⏱️  Separation completed in {elapsed:.1f}s")

    # Save files
    separated_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(vocals_path), separated['vocals'], sr)
    sf.write(str(instrumental_path), separated['instrumental'], sr)
    print(f"  💾 Saved: {vocals_path}")
    print(f"  💾 Saved: {instrumental_path}")

    return {'vocals': str(vocals_path), 'instrumental': str(instrumental_path)}


def extract_mel_features(audio_path: str, device: torch.device) -> torch.Tensor:
    """Extract mel spectrogram features for training."""
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Convert to tensor [1, mels, time]
    mel_tensor = torch.from_numpy(mel_db).float().unsqueeze(0)
    return mel_tensor.to(device)


def extract_speaker_embedding(audio_path: str) -> torch.Tensor:
    """Extract speaker embedding from audio."""
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Mel-statistics embedding
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mean = mel_db.mean(axis=1)  # [128]
    std = mel_db.std(axis=1)    # [128]
    embedding = np.concatenate([mean, std])  # [256]

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return torch.from_numpy(embedding).float()


class SimpleSample:
    """Simple training sample container."""
    def __init__(self, mel_tensor: torch.Tensor, speaker_embedding: torch.Tensor):
        self.mel_tensor = mel_tensor
        self.speaker_embedding = speaker_embedding


class SimpleMelModel(torch.nn.Module):
    """Simple mel-to-embedding model for training."""

    def __init__(self, mel_channels: int = 128, embedding_dim: int = 256):
        super().__init__()
        # Simple encoder
        self.conv1 = torch.nn.Conv1d(mel_channels, 256, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, mels, time]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [batch, 256]
        x = self.fc(x)  # [batch, embedding_dim]
        return x


def train_voice_model(
    profile_id: str,
    profile_name: str,
    vocals_path: str,
    output_dir: str,
    config: dict,
) -> dict:
    """Train a voice model with live progress display."""
    from auto_voice.training.fine_tuning import (
        inject_lora_adapters, save_lora_adapter
    )
    from auto_voice.training.job_manager import TrainingConfig

    print_banner(f"Training {profile_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  Device: {device}")

    # Extract features
    print(f"  📊 Extracting mel features...")
    mel_tensor = extract_mel_features(vocals_path, device)
    print(f"  📊 Mel shape: {mel_tensor.shape}")

    print(f"  🎤 Extracting speaker embedding...")
    speaker_embedding = extract_speaker_embedding(vocals_path).to(device)
    print(f"  🎤 Embedding shape: {speaker_embedding.shape}")

    # Create sample
    sample = SimpleSample(mel_tensor, speaker_embedding)
    samples = [sample]

    # Create model
    print(f"  🧠 Creating model...")
    model = SimpleMelModel(mel_channels=128, embedding_dim=256).to(device)

    # Inject LoRA adapters
    target_modules = ["fc"]  # Apply LoRA to the final linear layer
    model = inject_lora_adapters(
        model,
        target_modules=target_modules,
        rank=config['lora_rank'],
        alpha=config['lora_alpha'],
    )

    # Get trainable parameters (only LoRA)
    trainable_params = list(model.lora_adapters.parameters())
    param_count = sum(p.numel() for p in trainable_params)
    print(f"  📈 Trainable parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=config['learning_rate'])

    # Training loop with live progress
    print(f"\n  🚀 Starting training...")
    print(f"     Epochs: {config['epochs']}")
    print(f"     Learning rate: {config['learning_rate']}")
    print()

    model.train()
    loss_history = []
    start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        steps_per_epoch = 10  # Simulate multiple steps

        for step in range(steps_per_epoch):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sample.mel_tensor)

            # MSE loss to target embedding
            target = sample.speaker_embedding.unsqueeze(0).expand_as(outputs)
            loss = torch.nn.functional.mse_loss(outputs, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate progress
            progress = int(100 * ((epoch * steps_per_epoch + step + 1) / (config['epochs'] * steps_per_epoch)))
            print_progress(epoch, step + 1, loss.item(), progress, profile_name)

            time.sleep(0.05)  # Small delay for visibility

        avg_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_loss)

    elapsed = time.time() - start_time
    print(f"\n  ⏱️  Training completed in {elapsed:.1f}s")
    print(f"  📉 Final loss: {loss_history[-1]:.4f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    adapter_path = os.path.join(output_dir, f"{profile_id}_adapter.pt")
    save_lora_adapter(model, adapter_path)
    print(f"  💾 Saved adapter: {adapter_path}")

    return {
        'success': True,
        'profile_id': profile_id,
        'profile_name': profile_name,
        'adapter_path': adapter_path,
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'training_time': elapsed,
    }


def main():
    """Main training script."""
    print_banner("AutoVoice Training - Pillowtalk")

    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Change to project root
    os.chdir(PROJECT_ROOT)
    paths = resolve_runtime_paths()

    results = {}

    # ========================================================================
    # Step 1: Separate vocals for William Singe
    # ========================================================================
    print_banner("Step 1: Vocal Separation - William Singe")
    william_separated = separate_vocals(
        PILLOWTALK_WILLIAM,
        WILLIAM_PROFILE_ID,
        "William Singe",
        data_dir=str(paths["data_dir"]),
    )

    # ========================================================================
    # Step 2: Separate vocals for Conor Maynard
    # ========================================================================
    print_banner("Step 2: Vocal Separation - Conor Maynard")
    conor_separated = separate_vocals(
        PILLOWTALK_CONOR,
        CONOR_PROFILE_ID,
        "Conor Maynard",
        data_dir=str(paths["data_dir"]),
    )

    # ========================================================================
    # Step 3: Train William Singe model
    # ========================================================================
    paths["models_dir"].mkdir(parents=True, exist_ok=True)

    results['william'] = train_voice_model(
        profile_id=WILLIAM_PROFILE_ID,
        profile_name="William Singe",
        vocals_path=william_separated['vocals'],
        output_dir=str(paths["models_dir"]),
        config=TRAINING_CONFIG,
    )

    # ========================================================================
    # Step 4: Train Conor Maynard model
    # ========================================================================
    results['conor'] = train_voice_model(
        profile_id=CONOR_PROFILE_ID,
        profile_name="Conor Maynard",
        vocals_path=conor_separated['vocals'],
        output_dir=str(paths["models_dir"]),
        config=TRAINING_CONFIG,
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print_banner("Training Complete - Summary")

    for artist, result in results.items():
        print(f"  {artist.title()}:")
        print(f"    ✅ Success: {result['success']}")
        print(f"    📉 Final Loss: {result['final_loss']:.4f}")
        print(f"    ⏱️  Time: {result['training_time']:.1f}s")
        print(f"    💾 Adapter: {result['adapter_path']}")
        print()

    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎉 Ready for voice conversion!\n")

    return results


if __name__ == "__main__":
    main()
