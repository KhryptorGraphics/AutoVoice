#!/usr/bin/env python3
"""Optimized High-Quality LoRA Training for NVIDIA Thor.

Key optimizations for full GPU utilization:
1. Pre-extract all ContentVec features before training
2. Large batch sizes (32) for better GPU saturation
3. Multiple data loading workers with prefetching
4. Pin memory for faster CPU to GPU transfer
5. Gradient accumulation for effective larger batches

Usage:
    python scripts/train_hq_lora_optimized.py --artist conor_maynard --epochs 200
    python scripts/train_hq_lora_optimized.py --artist william_singe --epochs 200
    python scripts/train_hq_lora_optimized.py --artist both --epochs 200
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Enable line-buffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration - HQ Profile for Thor
# ============================================================================

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

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DIARIZED_DIR = DATA_DIR / 'diarized_youtube'
SEPARATED_DIR = DATA_DIR / 'separated_youtube'
FEATURES_DIR = DATA_DIR / 'features_cache'
CHECKPOINTS_DIR = DATA_DIR / 'checkpoints' / 'hq'
OUTPUT_DIR = DATA_DIR / 'trained_models' / 'hq'

# HQ Model Configuration
HQ_CONFIG = {
    'input_dim': 768,
    'hidden_dim': 1024,
    'output_dim': 768,
    'num_layers': 6,
    'lora_rank': 128,
    'lora_alpha': 256.0,
    'dropout': 0.05,
}

# Training Configuration (optimized for Thor)
TRAIN_CONFIG = {
    'batch_size': 32,           # Large batch for GPU saturation
    'gradient_accumulation': 2,  # Effective batch = 64
    'num_workers': 8,           # Parallel data loading
    'learning_rate': 1e-4,      # Higher LR for larger effective batch
    'warmup_epochs': 5,
    'prefetch_factor': 4,       # Prefetch batches
}


def print_banner(text: str):
    width = 70
    print("\n" + "=" * width)
    print("  " + text)
    print("=" * width + "\n")


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("  GPU Memory: {:.2f}GB / {:.1f}GB total".format(allocated, total))


# ============================================================================
# LoRA Architecture (same as before)
# ============================================================================

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 128,
                 alpha: float = 256.0, dropout: float = 0.05):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        lora_out = self.dropout(lora_out)
        return base_output + self.scaling * lora_out

    def get_delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


class HQVoiceLoRAAdapter(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024,
                 output_dim: int = 768, lora_rank: int = 128,
                 lora_alpha: float = 256.0, dropout: float = 0.05,
                 num_layers: int = 6):
        super().__init__()
        self.lora_rank = lora_rank
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        self.base_layers = nn.ModuleList()
        self.lora_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            self.base_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.lora_layers.append(LoRALayer(dims[i], dims[i + 1], lora_rank, lora_alpha, dropout))
            self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
            if dims[i] != dims[i + 1]:
                self.residual_projs.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            else:
                self.residual_projs.append(nn.Identity())

        self.speaker_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

        for layer in self.base_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, content: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = content
        for i, (base, lora, ln, res) in enumerate(zip(
                self.base_layers, self.lora_layers, self.layer_norms, self.residual_projs)):
            residual = res(x)
            x = lora(x, base(x))
            if i == 0 and speaker_embedding is not None:
                x = x + self.speaker_proj(speaker_embedding).unsqueeze(1).expand(-1, x.size(1), -1)
            x = ln(x)
            if i < len(self.base_layers) - 1:
                x = F.gelu(x) + 0.1 * residual
        return x

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for i, lora in enumerate(self.lora_layers):
            state['lora_{}_A'.format(i)] = lora.lora_A.data.clone()
            state['lora_{}_B'.format(i)] = lora.lora_B.data.clone()
        return state

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]):
        for i, lora in enumerate(self.lora_layers):
            key_a = 'lora_{}_A'.format(i)
            key_b = 'lora_{}_B'.format(i)
            if key_a in state:
                lora.lora_A.data = state[key_a]
                lora.lora_B.data = state[key_b]


# ============================================================================
# Feature Pre-extraction
# ============================================================================

def extract_and_cache_features(artist_key: str, device: torch.device) -> Path:
    """Pre-extract all ContentVec features and cache to disk."""

    cache_path = FEATURES_DIR / "{}_features.pt".format(artist_key)

    if cache_path.exists():
        logger.info("Loading cached features from {}".format(cache_path))
        return cache_path

    logger.info("Extracting features for {}...".format(artist_key))
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load ContentVec encoder once
    from auto_voice.models.encoder import ContentVecEncoder
    encoder = ContentVecEncoder(output_dim=768, pretrained="lengyue233/content-vec-best", device=device)
    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    diarized_dir = DIARIZED_DIR / artist_key
    separated_dir = SEPARATED_DIR / artist_key

    all_features = []
    segment_duration = 4.0
    sample_rate = 16000
    min_duration = 2.0
    max_per_file = 20

    json_files = list(diarized_dir.glob('*_diarization.json'))
    logger.info("Processing {} diarization files...".format(len(json_files)))

    for json_file in tqdm(json_files, desc="Extracting features"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            audio_stem = json_file.stem.replace('_diarization', '')
            audio_file = separated_dir / "{}.wav".format(audio_stem)

            if not audio_file.exists():
                continue

            # Load full audio
            audio, sr = sf.read(str(audio_file))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Find dominant speaker
            speaker_times = {}
            for seg in data.get('segments', []):
                spk = seg['speaker']
                dur = seg['end'] - seg['start']
                speaker_times[spk] = speaker_times.get(spk, 0) + dur

            if not speaker_times:
                continue

            dominant = max(speaker_times, key=speaker_times.get)

            # Extract segments from dominant speaker
            count = 0
            for seg in data.get('segments', []):
                if seg['speaker'] != dominant:
                    continue

                dur = seg['end'] - seg['start']
                if dur < min_duration:
                    continue

                # Extract segment
                start_sample = int(seg['start'] * sr)
                dur_samples = int(segment_duration * sr)
                end_sample = min(start_sample + dur_samples, len(audio))
                segment = audio[start_sample:end_sample]

                # Pad if needed
                if len(segment) < dur_samples:
                    segment = np.pad(segment, (0, dur_samples - len(segment)))

                # Resample if needed
                if sr != sample_rate:
                    segment = librosa.resample(segment, orig_sr=sr, target_sr=sample_rate)

                # Normalize
                max_val = np.abs(segment).max()
                if max_val > 0:
                    segment = segment / max_val * 0.95

                # Extract features
                audio_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = encoder.encode(audio_tensor)
                all_features.append(features.cpu())

                count += 1
                if count >= max_per_file:
                    break

        except Exception as e:
            logger.warning("Failed to process {}: {}".format(json_file, e))

    # Stack all features
    if not all_features:
        raise RuntimeError("No features extracted for {}".format(artist_key))

    all_features = torch.cat(all_features, dim=0)
    logger.info("Extracted {} feature sequences, shape: {}".format(all_features.shape[0], all_features.shape))

    # Save to cache
    torch.save(all_features, cache_path)
    logger.info("Cached features to {}".format(cache_path))

    # Free encoder memory
    del encoder
    torch.cuda.empty_cache()

    return cache_path


# ============================================================================
# Optimized Training
# ============================================================================

def train_artist_optimized(
    artist_key: str,
    epochs: int = 200,
    save_every: int = 20,
) -> Path:
    """Train HQ LoRA with full GPU utilization."""

    profile = ARTIST_PROFILES[artist_key]
    profile_id = profile['profile_id']

    print_banner("Optimized HQ LoRA Training: {}".format(profile['name']))
    print("  Profile ID: {}".format(profile_id))
    print("  Model config: {}".format(HQ_CONFIG))
    print("  Train config: {}".format(TRAIN_CONFIG))

    device = torch.device('cuda')
    print("  Device: {}".format(torch.cuda.get_device_name()))
    print_gpu_memory()

    # Phase 1: Extract/load features
    print("\n  Phase 1: Loading features...")
    cache_path = extract_and_cache_features(artist_key, device)
    features = torch.load(cache_path, map_location='cpu', weights_only=True)
    print("  Loaded {} feature sequences".format(features.shape[0]))

    # Create dataset from cached features
    dataset = TensorDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True,
        prefetch_factor=TRAIN_CONFIG['prefetch_factor'],
        persistent_workers=True,
    )

    print("  Batches per epoch: {}".format(len(dataloader)))
    print("  Effective batch size: {}".format(TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['gradient_accumulation']))

    # Phase 2: Create model
    print("\n  Phase 2: Creating model...")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = HQVoiceLoRAAdapter(**HQ_CONFIG).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("  Trainable params: {:,} / {:,}".format(trainable, total))

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TRAIN_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    scaler = GradScaler('cuda')

    # Loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # Load checkpoint if exists
    checkpoint_path = CHECKPOINTS_DIR / "{}_hq_lora.pt".format(profile_id)
    start_epoch = 0
    global_step = 0

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_lora_state_dict(ckpt['lora_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        print("  Resumed from epoch {}".format(start_epoch))

    # Phase 3: Training
    print("\n  Phase 3: Training...")
    print("-" * 60)

    best_loss = float('inf')
    grad_accum = TRAIN_CONFIG['gradient_accumulation']
    warmup_steps = TRAIN_CONFIG['warmup_epochs'] * len(dataloader)
    base_lr = TRAIN_CONFIG['learning_rate']

    # Cosine annealing with warm restarts parameters (TUNED FOR PLATEAU ESCAPE)
    restart_period = 500   # More frequent restarts (was 1000)
    min_lr = 1e-5          # Higher floor to maintain learning (was 1e-6)
    plateau_count = 0
    plateau_threshold = 100  # Trigger LR boost after 100 epochs without improvement
    last_best_loss = float('inf')
    gradient_noise_scale = 0.01  # Add noise to escape local minima

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()

        # Cosine annealing with warm restarts
        cycle_epoch = epoch % restart_period
        if cycle_epoch == 0 and epoch > 0:
            # Warm restart: reset to base LR (decayed slower)
            cycle_num = epoch // restart_period
            current_base_lr = base_lr * (0.8 ** (cycle_num // 10))  # Slower decay: 0.8 every 10 cycles (was 0.5 every 5)
            current_base_lr = max(current_base_lr, min_lr * 5)
            logger.info("Warm restart at epoch {} - new base LR: {:.6f}".format(epoch, current_base_lr))
        else:
            cycle_num = epoch // restart_period
            current_base_lr = base_lr * (0.8 ** (cycle_num // 10))
            current_base_lr = max(current_base_lr, min_lr * 5)

        # Plateau escape: boost LR if stuck
        if plateau_count >= plateau_threshold:
            current_base_lr = base_lr * 0.5  # Reset to half base LR
            plateau_count = 0
            logger.info("Plateau detected at epoch {} - boosting LR to {:.6f}".format(epoch, current_base_lr))

        # Cosine annealing within cycle
        progress = cycle_epoch / restart_period
        current_lr = min_lr + (current_base_lr - min_lr) * (1 + np.cos(np.pi * progress)) / 2

        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        for batch_idx, (batch_features,) in enumerate(dataloader):
            batch_features = batch_features.to(device, non_blocking=True)

            # Warmup learning rate (only at very beginning)
            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr * lr_scale

            with autocast('cuda'):
                output = model(batch_features)

                # Multi-scale loss
                mse = mse_loss(output, batch_features)
                l1 = l1_loss(output, batch_features)
                loss = (0.7 * mse + 0.3 * l1) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                # Add gradient noise to escape local minima (decays over time)
                noise_scale = gradient_noise_scale / (1 + epoch * 0.001)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.add_(torch.randn_like(param.grad) * noise_scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_losses.append(loss.item() * grad_accum)
            global_step += 1

        avg_loss = np.mean(epoch_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            plateau_count = 0
        else:
            plateau_count += 1

        # Show LR info periodically
        lr_info = " | LR: {:.2e}".format(current_lr) if (epoch + 1) % 100 == 0 else ""
        print("  Epoch {:5d}/{} | Loss: {:.4f} | Best: {:.4f}{}".format(epoch+1, epochs, avg_loss, best_loss, lr_info))

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'loss': avg_loss,
                'lora_state': model.get_lora_state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'global_step': global_step,
                'config': HQ_CONFIG,
            }, checkpoint_path)
            print("    -> Checkpoint saved")

    # Final save
    final_path = OUTPUT_DIR / "{}_hq_lora.pt".format(profile_id)
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

    size_kb = final_path.stat().st_size / 1024
    print("\n  Saved: {}".format(final_path))
    print("  Size: {:.1f} KB".format(size_kb))

    return final_path


def main():
    parser = argparse.ArgumentParser(description='Optimized HQ LoRA Training for Thor')
    parser.add_argument('--artist', type=str, required=True,
                        choices=['conor_maynard', 'william_singe', 'both'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    if args.batch_size != 32:
        TRAIN_CONFIG['batch_size'] = args.batch_size

    print_banner("Optimized HQ LoRA Training for NVIDIA Thor")
    print("  Optimizations: Pre-extracted features, batch={}, workers=8".format(TRAIN_CONFIG['batch_size']))
    print("  Expected GPU utilization: 80-95%")

    if args.artist == 'both':
        for artist in ['conor_maynard', 'william_singe']:
            train_artist_optimized(artist, epochs=args.epochs)
    else:
        train_artist_optimized(args.artist, epochs=args.epochs)


if __name__ == '__main__':
    main()
