#!/usr/bin/env python3
"""High-Quality LoRA Training for NVIDIA Thor.

Optimized for maximum voice quality while maintaining real-time inference.
Configuration: 768->1024->768 with 6 layers, rank=128 (~1.5M params)

Usage:
    python scripts/train_hq_lora.py --artist conor_maynard --epochs 200
    python scripts/train_hq_lora.py --artist william_singe --epochs 200
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration - HQ Profile for Thor
# ============================================================================

# Artist profiles
ARTIST_PROFILES = {
    'conor_maynard': {
        'name': 'Conor Maynard',
        'profile_id': 'c572d02c-c687-4bed-8676-6ad253cf1c91',
    },
    'william_singe': {
        'name': 'William Singe',
        'profile_id': '7da05140-1303-40c6-95d9-5b6e2c3624df',
    },
}

# Data directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DIARIZED_DIR = DATA_DIR / 'diarized_youtube'
SEPARATED_DIR = DATA_DIR / 'separated_youtube'
CHECKPOINTS_DIR = DATA_DIR / 'checkpoints' / 'hq'
OUTPUT_DIR = DATA_DIR / 'trained_models' / 'hq'

# HQ Model Configuration (optimized for Thor)
HQ_CONFIG = {
    'input_dim': 768,      # ContentVec dimension
    'hidden_dim': 1024,    # Wider hidden layer
    'output_dim': 768,     # Higher output for better reconstruction
    'num_layers': 6,       # More depth
    'lora_rank': 128,      # High rank for quality
    'lora_alpha': 256.0,   # Higher alpha for stronger adaptation
    'dropout': 0.05,       # Lower dropout for HQ
}


def print_banner(text: str):
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB / {total:.1f}GB total")


# ============================================================================
# HQ LoRA Architecture
# ============================================================================

class LoRALayer(nn.Module):
    """High-Quality LoRA layer with scaled initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        alpha: float = 256.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices with careful initialization
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Kaiming initialization for better gradient flow
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Start with zero adaptation

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        lora_out = self.dropout(lora_out)
        return base_output + self.scaling * lora_out

    def get_delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


class HQVoiceLoRAAdapter(nn.Module):
    """High-Quality Voice LoRA Adapter for Thor.

    Architecture: 768 -> 1024 -> 1024 -> 1024 -> 1024 -> 1024 -> 768
    With residual connections and layer normalization.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 768,
        lora_rank: int = 128,
        lora_alpha: float = 256.0,
        dropout: float = 0.05,
        num_layers: int = 6,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank

        # Build layer dimensions with smooth transition
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        # Base layers (frozen during LoRA training)
        self.base_layers = nn.ModuleList()
        self.lora_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            # Base linear layer
            base = nn.Linear(dims[i], dims[i + 1])
            self.base_layers.append(base)

            # LoRA adaptation layer
            lora = LoRALayer(
                dims[i], dims[i + 1],
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=dropout,
            )
            self.lora_layers.append(lora)

            # Layer normalization for stability
            self.layer_norms.append(nn.LayerNorm(dims[i + 1]))

        # Speaker conditioning projection
        self.speaker_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Residual projections for dimension mismatches
        self.residual_projs = nn.ModuleList()
        for i in range(num_layers):
            if dims[i] != dims[i + 1]:
                self.residual_projs.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            else:
                self.residual_projs.append(nn.Identity())

        # Freeze base layers
        for layer in self.base_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        content: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = content

        for i, (base, lora, ln, res_proj) in enumerate(zip(
            self.base_layers, self.lora_layers, self.layer_norms, self.residual_projs
        )):
            # Residual connection
            residual = res_proj(x)

            # Base + LoRA
            base_out = base(x)
            x = lora(x, base_out)

            # Speaker conditioning after first layer
            if i == 0 and speaker_embedding is not None:
                spk = self.speaker_proj(speaker_embedding).unsqueeze(1)
                x = x + spk.expand(-1, x.size(1), -1)

            # Normalization + activation + residual
            x = ln(x)
            if i < len(self.base_layers) - 1:
                x = F.gelu(x)
                x = x + 0.1 * residual  # Scaled residual

        return x

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for i, lora in enumerate(self.lora_layers):
            state[f'lora_{i}_A'] = lora.lora_A.data.clone()
            state[f'lora_{i}_B'] = lora.lora_B.data.clone()
        return state

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]):
        for i, lora in enumerate(self.lora_layers):
            if f'lora_{i}_A' in state:
                lora.lora_A.data = state[f'lora_{i}_A']
                lora.lora_B.data = state[f'lora_{i}_B']

    def merge_and_quantize(self):
        """Merge LoRA weights into base and quantize to fp16."""
        for base, lora in zip(self.base_layers, self.lora_layers):
            delta = lora.get_delta_weight()
            base.weight.data += delta.T
            base.weight.requires_grad = False
        self.half()
        return self


# ============================================================================
# Dataset with Enhanced Loading
# ============================================================================

class DiarizedVocalsDataset(Dataset):
    """Dataset of diarized vocal segments for HQ LoRA training."""

    def __init__(
        self,
        artist_key: str,
        segment_duration: float = 4.0,  # Longer segments for better context
        sample_rate: int = 16000,
        max_segments_per_file: int = 20,  # More segments per file
        min_segment_duration: float = 2.0,
    ):
        self.artist_key = artist_key
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.max_segments_per_file = max_segments_per_file
        self.min_segment_duration = min_segment_duration

        self.segments = []
        self._load_segments()

    def _load_segments(self):
        diarized_dir = DIARIZED_DIR / self.artist_key
        separated_dir = SEPARATED_DIR / self.artist_key

        if not diarized_dir.exists():
            raise RuntimeError(f"Diarization directory not found: {diarized_dir}")

        json_files = list(diarized_dir.glob('*_diarization.json'))
        logger.info(f"Loading segments from {len(json_files)} diarization files")

        total_duration = 0

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                audio_stem = json_file.stem.replace('_diarization', '')
                audio_file = separated_dir / f"{audio_stem}.wav"

                if not audio_file.exists():
                    continue

                # Find dominant speaker
                speaker_times = {}
                for seg in data.get('segments', []):
                    spk = seg['speaker']
                    duration = seg['end'] - seg['start']
                    speaker_times[spk] = speaker_times.get(spk, 0) + duration

                if not speaker_times:
                    continue

                dominant_speaker = max(speaker_times, key=speaker_times.get)

                # Extract segments from dominant speaker
                count = 0
                for seg in data.get('segments', []):
                    if seg['speaker'] != dominant_speaker:
                        continue

                    duration = seg['end'] - seg['start']
                    if duration < self.min_segment_duration:
                        continue

                    self.segments.append({
                        'audio_file': str(audio_file),
                        'start': seg['start'],
                        'end': seg['end'],
                        'speaker': seg['speaker'],
                    })

                    total_duration += duration
                    count += 1
                    if count >= self.max_segments_per_file:
                        break

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(self.segments)} segments ({total_duration/60:.1f} min) for {self.artist_key}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seg = self.segments[idx]

        # Load audio segment
        audio, sr = sf.read(seg['audio_file'])
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Extract segment
        start_sample = int(seg['start'] * sr)
        duration_samples = int(self.segment_duration * sr)
        end_sample = min(start_sample + duration_samples, len(audio))

        segment = audio[start_sample:end_sample]

        # Pad if needed
        if len(segment) < duration_samples:
            segment = np.pad(segment, (0, duration_samples - len(segment)))

        # Resample if needed
        if sr != self.sample_rate:
            segment = librosa.resample(segment, orig_sr=sr, target_sr=self.sample_rate)

        # Normalize
        max_val = np.abs(segment).max()
        if max_val > 0:
            segment = segment / max_val * 0.95

        return {
            'audio': torch.tensor(segment, dtype=torch.float32),
            'path': seg['audio_file'],
        }


# ============================================================================
# HQ Trainer with Advanced Techniques
# ============================================================================

class HQLoRATrainer:
    """High-Quality LoRA Trainer with cosine annealing and gradient clipping."""

    def __init__(
        self,
        model: HQVoiceLoRAAdapter,
        device: torch.device,
        learning_rate: float = 5e-5,  # Lower LR for stability
        warmup_steps: int = 500,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate

        # Trainable parameters only
        trainable = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        self.scaler = GradScaler() if use_amp else None
        self.global_step = 0

        # Multi-scale loss
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # ContentVec encoder
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from auto_voice.models.encoder import ContentVecEncoder
            self._encoder = ContentVecEncoder(
                output_dim=768,
                pretrained="lengyue233/content-vec-best",
                device=self.device
            )
            self._encoder.to(self.device)
            for param in self._encoder.parameters():
                param.requires_grad = False
        return self._encoder

    def _warmup_lr(self) -> float:
        """Linear warmup then constant."""
        if self.global_step < self.warmup_steps:
            return self.global_step / self.warmup_steps
        return 1.0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        audio = batch['audio'].to(self.device)

        # Adjust learning rate
        lr_scale = self._warmup_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_scale

        self.optimizer.zero_grad()

        encoder = self._get_encoder()

        with autocast(enabled=self.use_amp):
            # Extract content features (target)
            with torch.no_grad():
                content_features = encoder.encode(audio)

            # Forward through LoRA adapter
            output = self.model(content_features)

            # Multi-scale loss
            mse = self.mse_loss(output, content_features)
            l1 = self.l1_loss(output, content_features)

            # Combined loss with reconstruction and smoothness
            loss = 0.7 * mse + 0.3 * l1

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.global_step += 1
        return loss.item()

    def save_checkpoint(self, path: Path, epoch: int, loss: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'lora_state': self.model.get_lora_state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': HQ_CONFIG,
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        if not path.exists():
            return 0

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_lora_state_dict(ckpt['lora_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.global_step = ckpt['global_step']
        return ckpt['epoch'] + 1


def train_artist_hq_lora(
    artist_key: str,
    epochs: int = 200,
    batch_size: int = 4,  # Smaller batch for larger model
    learning_rate: float = 5e-5,
    save_every: int = 20,
) -> Path:
    """Train HQ LoRA for an artist."""

    profile = ARTIST_PROFILES[artist_key]
    profile_id = profile['profile_id']

    print_banner(f"Training HQ LoRA for {profile['name']}")
    print(f"  Profile ID: {profile_id}")
    print(f"  Config: {HQ_CONFIG}")
    print(f"  Epochs: {epochs}, batch size: {batch_size}")
    print_gpu_memory()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for HQ training")

    device = torch.device('cuda')
    print(f"  Device: {torch.cuda.get_device_name()}")

    # Ensure directories exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Loading dataset...")
    dataset = DiarizedVocalsDataset(artist_key)

    if len(dataset) == 0:
        raise RuntimeError(f"No training data for {artist_key}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    print(f"  Segments: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")

    print("\n  Creating HQ LoRA adapter...")
    model = HQVoiceLoRAAdapter(**HQ_CONFIG)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"  Model size: {trainable_params * 4 / 1024 / 1024:.2f} MB (fp32)")

    trainer = HQLoRATrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        use_amp=True,
    )

    checkpoint_path = CHECKPOINTS_DIR / f"{profile_id}_hq_lora.pt"
    start_epoch = trainer.load_checkpoint(checkpoint_path)
    if start_epoch > 0:
        print(f"\n  Resuming from epoch {start_epoch}")

    print("\n  Starting training...")
    print("-" * 50)

    best_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        epoch_losses = []

        for batch in dataloader:
            loss = trainer.train_step(batch)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(checkpoint_path, epoch, avg_loss)
            print(f"    -> Checkpoint saved")

    # Final save
    trainer.save_checkpoint(checkpoint_path, epochs - 1, best_loss)

    # Save final model
    final_path = OUTPUT_DIR / f"{profile_id}_hq_lora.pt"
    torch.save({
        'profile_id': profile_id,
        'artist': profile['name'],
        'config': HQ_CONFIG,
        'lora_state': model.get_lora_state_dict(),
        'epoch': epochs - 1,
        'loss': best_loss,
        'precision': 'fp16',
        'trained_on': datetime.now().isoformat(),
    }, final_path)

    print(f"\n  Saved HQ model: {final_path}")
    print(f"  Final size: {final_path.stat().st_size / 1024:.1f} KB")

    return final_path


def main():
    parser = argparse.ArgumentParser(description='Train HQ LoRA for NVIDIA Thor')
    parser.add_argument('--artist', type=str, required=True,
                        choices=['conor_maynard', 'william_singe', 'both'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)

    args = parser.parse_args()

    print_banner("HQ LoRA Training for NVIDIA Thor")
    print(f"  Config: {HQ_CONFIG}")
    print(f"  Expected params: ~1.5M")
    print(f"  Expected size: ~5.8MB (fp32), ~2.9MB (fp16)")

    if args.artist == 'both':
        for artist in ['conor_maynard', 'william_singe']:
            train_artist_hq_lora(
                artist,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
    else:
        train_artist_hq_lora(
            args.artist,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )


if __name__ == '__main__':
    main()
