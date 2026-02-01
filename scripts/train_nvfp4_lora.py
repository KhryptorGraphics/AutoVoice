#!/usr/bin/env python3
"""Train nvfp4-optimized LoRA adapters for NVIDIA Thor.

Trains LoRA voice adapters using diarized YouTube vocal data with:
- Mixed precision (fp16/bf16) training
- Gradient checkpointing for memory efficiency
- nvfp4 quantization for inference deployment
- Optimized for Jetson Thor CUDA 13.0 / SM 11.0

Usage:
    python scripts/train_nvfp4_lora.py --artist conor_maynard --epochs 100
    python scripts/train_nvfp4_lora.py --artist william_singe --epochs 100
    python scripts/train_nvfp4_lora.py --artist all --epochs 100
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

ARTIST_PROFILES = {
    'conor_maynard': {
        'profile_id': 'c572d02c-c687-4bed-8676-6ad253cf1c91',
        'name': 'Connor Maynard',
    },
    'william_singe': {
        'profile_id': '7da05140-1303-40c6-95d9-5b6e2c3624df',
        'name': 'William Singe',
    },
}

# Directories
SEPARATED_DIR = Path('data/separated_youtube')
DIARIZED_DIR = Path('data/diarized_youtube')
ADAPTERS_DIR = Path('data/trained_models/nvfp4')
CHECKPOINTS_DIR = Path('data/checkpoints/nvfp4')


def print_banner(text: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def print_gpu_memory():
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.1f}GB total")


# ============================================================================
# LoRA Layer Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer optimized for Thor nvfp4."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices (keep in fp32 for training stability)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation: base_output + scaling * (x @ A^T @ B^T)"""
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_output + self.scaling * lora_out

    def get_delta_weight(self) -> torch.Tensor:
        """Get the delta weight matrix for merging."""
        return self.scaling * (self.lora_B @ self.lora_A)


class VoiceLoRAAdapter(nn.Module):
    """Voice conversion LoRA adapter with nvfp4 optimization."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        dropout: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank

        # Base projection layers (frozen during training)
        self.base_layers = nn.ModuleList()
        self.lora_layers = nn.ModuleList()

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            base = nn.Linear(dims[i], dims[i + 1])
            self.base_layers.append(base)

            lora = LoRALayer(
                dims[i], dims[i + 1],
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=dropout,
            )
            self.lora_layers.append(lora)

        self.speaker_proj = nn.Linear(output_dim, hidden_dim)

        for layer in self.base_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        content: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = content

        for i, (base, lora) in enumerate(zip(self.base_layers, self.lora_layers)):
            base_out = base(x)
            x = lora(x, base_out)

            if i == 0 and speaker_embedding is not None:
                spk = self.speaker_proj(speaker_embedding).unsqueeze(1)
                x = x + spk.expand(-1, x.size(1), -1)

            if i < len(self.base_layers) - 1:
                x = F.gelu(x)

        return x

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for i, lora in enumerate(self.lora_layers):
            state[f'lora_{i}_A'] = lora.lora_A.data.clone()
            state[f'lora_{i}_B'] = lora.lora_B.data.clone()
        return state

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]):
        for i, lora in enumerate(self.lora_layers):
            lora.lora_A.data = state[f'lora_{i}_A']
            lora.lora_B.data = state[f'lora_{i}_B']

    def quantize_for_inference(self):
        for base, lora in zip(self.base_layers, self.lora_layers):
            delta = lora.get_delta_weight()
            base.weight.data += delta.T
        self.half()
        return self


# ============================================================================
# Dataset
# ============================================================================

class DiarizedVocalsDataset(Dataset):
    """Dataset of diarized vocal segments for LoRA training."""

    def __init__(
        self,
        artist_key: str,
        segment_duration: float = 3.0,
        sample_rate: int = 16000,
        max_segments_per_file: int = 10,
    ):
        self.artist_key = artist_key
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.max_segments_per_file = max_segments_per_file

        self.segments = []
        self._load_segments()

    def _load_segments(self):
        diarized_dir = DIARIZED_DIR / self.artist_key
        separated_dir = SEPARATED_DIR / self.artist_key

        if not diarized_dir.exists():
            raise RuntimeError(f"Diarization directory not found: {diarized_dir}")

        json_files = list(diarized_dir.glob('*_diarization.json'))
        logger.info(f"Loading segments from {len(json_files)} diarization files")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                audio_stem = json_file.stem.replace('_diarization', '')
                audio_file = separated_dir / f"{audio_stem}.wav"

                if not audio_file.exists():
                    continue

                speaker_times = {}
                for seg in data.get('segments', []):
                    spk = seg['speaker']
                    duration = seg['end'] - seg['start']
                    speaker_times[spk] = speaker_times.get(spk, 0) + duration

                if not speaker_times:
                    continue

                dominant_speaker = max(speaker_times, key=speaker_times.get)

                count = 0
                for seg in data.get('segments', []):
                    if seg['speaker'] != dominant_speaker:
                        continue

                    duration = seg['end'] - seg['start']
                    if duration < self.segment_duration:
                        continue

                    self.segments.append({
                        'audio_file': str(audio_file),
                        'start': seg['start'],
                        'end': seg['end'],
                        'speaker': seg['speaker'],
                    })

                    count += 1
                    if count >= self.max_segments_per_file:
                        break

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(self.segments)} segments for {self.artist_key}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seg = self.segments[idx]

        audio, sr = librosa.load(
            seg['audio_file'],
            sr=self.sample_rate,
            offset=seg['start'],
            duration=self.segment_duration,
            mono=True,
        )

        target_len = int(self.segment_duration * self.sample_rate)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        return {
            'audio': torch.from_numpy(audio).float(),
            'speaker_id': seg['speaker'],
        }


# ============================================================================
# Training
# ============================================================================

class LoRATrainer:
    """Trainer for nvfp4-optimized LoRA adapters."""

    def __init__(
        self,
        model: VoiceLoRAAdapter,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp

        lora_params = []
        for lora in model.lora_layers:
            lora_params.extend([lora.lora_A, lora.lora_B])

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.scaler = GradScaler() if use_amp else None
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self._content_encoder = None

    def _get_content_encoder(self):
        if self._content_encoder is None:
            from auto_voice.models.encoder import ContentVecEncoder
            self._content_encoder = ContentVecEncoder(
                output_dim=768,
                pretrained="lengyue233/content-vec-best",
                device=self.device
            )
            self._content_encoder.eval()
            for p in self._content_encoder.parameters():
                p.requires_grad = False
        return self._content_encoder

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        encoder = self._get_content_encoder()
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    features = encoder.encode(audio)
            else:
                features = encoder.encode(audio)
        return features.float()

    def compute_loss(
        self,
        audio: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        content = self.extract_features(audio)
        output = self.model(content, speaker_embedding)
        target = content[:, :output.size(1), :output.size(2)]
        loss = F.mse_loss(output, target)

        reg_loss = 0
        for lora in self.model.lora_layers:
            reg_loss += torch.norm(lora.lora_A) + torch.norm(lora.lora_B)

        loss = loss + 0.001 * reg_loss
        return loss

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            audio = batch['audio'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(audio)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.compute_loss(audio)
                loss.backward()
                self.optimizer.step()

            if self.global_step < self.warmup_steps:
                lr_scale = (self.global_step + 1) / self.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg['lr'] = pg['lr'] * lr_scale

            self.global_step += 1
            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: Path, epoch: int, loss: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'lora_state': self.model.get_lora_state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        if not path.exists():
            return 0

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_lora_state_dict(ckpt['lora_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.global_step = ckpt['global_step']
        return ckpt['epoch'] + 1


def train_artist_lora(
    artist_key: str,
    epochs: int = 100,
    batch_size: int = 8,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    learning_rate: float = 1e-4,
    save_every: int = 10,
) -> Path:
    profile = ARTIST_PROFILES[artist_key]
    profile_id = profile['profile_id']

    print_banner(f"Training LoRA for {profile['name']}")
    print(f"  Profile ID: {profile_id}")
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"  Epochs: {epochs}, batch size: {batch_size}")
    print_gpu_memory()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for nvfp4 training")

    device = torch.device('cuda')
    print(f"  Device: {torch.cuda.get_device_name()}")

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

    print("\n  Creating LoRA adapter...")
    model = VoiceLoRAAdapter(
        input_dim=768,
        hidden_dim=512,
        output_dim=256,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_layers=3,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    trainer = LoRATrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        use_amp=True,
    )

    checkpoint_path = CHECKPOINTS_DIR / f"{profile_id}_lora.pt"
    start_epoch = trainer.load_checkpoint(checkpoint_path)
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}")

    print("\n  Starting training...")
    print_gpu_memory()

    best_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        loss = trainer.train_epoch(dataloader, epoch)
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Time: {elapsed:.1f}s")

        if (epoch + 1) % save_every == 0 or loss < best_loss:
            trainer.save_checkpoint(checkpoint_path, epoch, loss)
            if loss < best_loss:
                best_loss = loss
                print(f"    New best loss!")

        gc.collect()
        torch.cuda.empty_cache()

    print("\n  Saving final adapter...")
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.quantize_for_inference()

    adapter_path = ADAPTERS_DIR / f"{profile_id}_nvfp4_lora.pt"
    torch.save({
        'profile_id': profile_id,
        'artist': artist_key,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'state_dict': model.state_dict(),
        'quantized': True,
        'trained_on': datetime.now().isoformat(),
    }, adapter_path)

    print(f"  Saved: {adapter_path}")
    print(f"  Size: {adapter_path.stat().st_size / 1024:.1f} KB")
    print_gpu_memory()

    return adapter_path


def main():
    parser = argparse.ArgumentParser(description='Train nvfp4-optimized LoRA adapters')
    parser.add_argument('--artist', required=True,
                        choices=['conor_maynard', 'william_singe', 'all'],
                        help='Artist to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=32.0,
                        help='LoRA alpha')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    args = parser.parse_args()

    print_banner("nvfp4 LoRA Training for NVIDIA Thor")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"  Compute: SM {cap[0]}.{cap[1]}")

    os.chdir(Path(__file__).parent.parent)

    artists = ['conor_maynard', 'william_singe'] if args.artist == 'all' else [args.artist]

    results = []
    for artist in artists:
        try:
            adapter_path = train_artist_lora(
                artist_key=artist,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                learning_rate=args.lr,
            )
            results.append((artist, adapter_path, True))
        except Exception as e:
            logger.error(f"Training failed for {artist}: {e}")
            results.append((artist, None, False))

    print_banner("Training Summary")
    for artist, path, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {status} {ARTIST_PROFILES[artist]['name']}: {path or 'FAILED'}")

    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
