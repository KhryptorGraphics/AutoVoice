"""Training pipeline for So-VITS-SVC model.

Supports dataset loading, distributed training, loss computation,
checkpointing, and quality assessment.
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class VoiceDataset(Dataset):
    """Dataset for voice conversion training."""

    def __init__(self, data_dir: str, sample_rate: int = 22050,
                 segment_length: int = 32768, speaker_id: Optional[str] = None,
                 augment: bool = False):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.speaker_id = speaker_id
        self.augment = augment
        self._augmentation = None
        if augment:
            from ..audio.augmentation import AugmentationPipeline
            self._augmentation = AugmentationPipeline()
        self.audio_files = self._scan_files()
        logger.info(f"VoiceDataset: {len(self.audio_files)} files from {data_dir}"
                    f"{' (augment=True)' if augment else ''}")

    def _scan_files(self) -> List[Path]:
        extensions = {'.wav', '.flac', '.mp3', '.ogg'}
        files = []
        for ext in extensions:
            files.extend(self.data_dir.rglob(f'*{ext}'))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import librosa
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        if len(audio) > self.segment_length:
            start = np.random.randint(0, len(audio) - self.segment_length)
            audio = audio[start:start + self.segment_length]
        else:
            audio = np.pad(audio, (0, self.segment_length - len(audio)))

        # Apply augmentation if enabled
        if self._augmentation is not None:
            audio = self._augmentation(audio, self.sample_rate)

        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=2048,
            hop_length=512, n_mels=80,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db + 80.0) / 80.0

        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=self.sample_rate)
        f0 = np.nan_to_num(f0, nan=0.0)

        return {
            'audio': torch.from_numpy(audio).float(),
            'mel': torch.from_numpy(mel_db).float(),
            'f0': torch.from_numpy(f0).float(),
            'path': str(audio_path),
        }


class Trainer:
    """Training loop for So-VITS-SVC model."""

    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None,
                 device=None):
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.lr = self.config.get('learning_rate', 1e-4)
        self.batch_size = self.config.get('batch_size', 16)
        self.epochs = self.config.get('epochs', 100)
        self.save_every = self.config.get('save_every', 10)
        self.log_every = self.config.get('log_every', 100)
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.gradient_clip = self.config.get('gradient_clip', 1.0)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            betas=(0.8, 0.99), weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.999,
        )

        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses: List[float] = []

        # Shared encoders for feature extraction (frozen during training)
        from ..models.encoder import ContentEncoder, PitchEncoder
        self.content_encoder = ContentEncoder(device=self.device).to(self.device)
        self.pitch_encoder = PitchEncoder().to(self.device)
        self.speaker_embedding: Optional[torch.Tensor] = None
        self.sample_rate = self.config.get('sample_rate', 22050)

    def train(self, train_dir: str, val_dir: Optional[str] = None,
              resume_from: Optional[str] = None):
        """Run training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        train_dataset = VoiceDataset(train_dir, segment_length=32768)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )

        val_loader = None
        if val_dir and Path(val_dir).exists():
            val_dataset = VoiceDataset(val_dir, segment_length=32768)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=False, num_workers=2)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting training: {self.epochs} epochs, "
                    f"{len(train_dataset)} samples, device={self.device}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(epoch_loss)

            if val_loader and (epoch + 1) % self.save_every == 0:
                val_loss = self.assess(val_loader)
                logger.info(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, "
                            f"val_loss={val_loss:.4f}")
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(self.checkpoint_dir / 'best.pth')
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}")

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

            self.scheduler.step()

        self.save_checkpoint(self.checkpoint_dir / 'final.pth')
        logger.info("Training complete")

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            audio = batch['audio'].to(self.device)   # [B, segment_length]
            mel = batch['mel'].to(self.device)        # [B, 80, T_mel]
            f0 = batch['f0'].to(self.device)          # [B, T_f0]

            n_mel_frames = mel.shape[2]

            # Extract content features from training audio (no grad - frozen encoder)
            with torch.no_grad():
                content = self.content_encoder.extract_features(
                    audio, sr=self.sample_rate
                )  # [B, N, 256]
                pitch = self.pitch_encoder(f0)  # [B, T, 256]

            # Align to mel frame count
            content = F.interpolate(
                content.transpose(1, 2), size=n_mel_frames, mode='linear',
                align_corners=False
            ).transpose(1, 2)
            pitch = F.interpolate(
                pitch.transpose(1, 2), size=n_mel_frames, mode='linear',
                align_corners=False
            ).transpose(1, 2)

            # Speaker embedding (same for all batches of this speaker)
            if self.speaker_embedding is None:
                raise RuntimeError("Speaker embedding not set. Call set_speaker_embedding() first.")
            speaker = self.speaker_embedding.unsqueeze(0).expand(audio.shape[0], -1)

            # Compute spectrogram for posterior encoder
            spec = self._compute_spec(audio, n_mel_frames)

            # Forward + loss
            self.optimizer.zero_grad()
            outputs = self.model(content, pitch, speaker, spec=spec)
            losses = self.model.compute_loss(outputs, mel)
            loss = losses['total_loss']

            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                logger.info(f"Step {self.global_step}: loss={loss.item():.4f}, "
                            f"recon={losses['reconstruction_loss'].item():.4f}")

        return total_loss / max(n_batches, 1)

    def assess(self, loader: DataLoader) -> float:
        """Run assessment on given data loader."""
        self.model.train(False)
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                audio = batch['audio'].to(self.device)
                mel = batch['mel'].to(self.device)
                f0 = batch['f0'].to(self.device)
                n_mel_frames = mel.shape[2]

                content = self.content_encoder.extract_features(
                    audio, sr=self.sample_rate
                )
                pitch = self.pitch_encoder(f0)

                content = F.interpolate(
                    content.transpose(1, 2), size=n_mel_frames, mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                pitch = F.interpolate(
                    pitch.transpose(1, 2), size=n_mel_frames, mode='linear',
                    align_corners=False
                ).transpose(1, 2)

                if self.speaker_embedding is None:
                    raise RuntimeError("Speaker embedding not set.")
                speaker = self.speaker_embedding.unsqueeze(0).expand(audio.shape[0], -1)
                spec = self._compute_spec(audio, n_mel_frames)

                outputs = self.model(content, pitch, speaker, spec=spec)
                losses = self.model.compute_loss(outputs, mel)
                total_loss += losses['total_loss'].item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        if path is None:
            path = self.checkpoint_dir / 'latest.pth'
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': self.config,
        }
        torch.save(checkpoint, str(path))
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Resumed from {path} (epoch {self.current_epoch}, "
                    f"step {self.global_step})")

    def _compute_spec(self, audio: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Compute linear spectrogram for posterior encoder input."""
        n_fft = 1024
        hop_length = 512
        window = torch.hann_window(n_fft, device=audio.device)
        specs = []
        for i in range(audio.shape[0]):
            stft = torch.stft(audio[i], n_fft=n_fft, hop_length=hop_length,
                              window=window, return_complex=True)
            spec = stft.abs()  # [513, T_frames]
            # Align to target_frames
            spec = F.interpolate(spec.unsqueeze(0), size=target_frames,
                                 mode='linear', align_corners=False).squeeze(0)
            specs.append(spec)
        return torch.stack(specs)  # [B, 513, target_frames]

    def set_speaker_embedding(self, audio_dir: str):
        """Compute and set fixed speaker embedding from training audio."""
        from ..inference.voice_cloner import VoiceCloner
        cloner = VoiceCloner(device=self.device)
        audio_files = sorted(
            list(Path(audio_dir).rglob('*.wav')) +
            list(Path(audio_dir).rglob('*.flac')) +
            list(Path(audio_dir).rglob('*.mp3'))
        )
        if not audio_files:
            raise RuntimeError(f"No audio files found in {audio_dir}")
        embedding = cloner.create_speaker_embedding([str(f) for f in audio_files])
        self.speaker_embedding = torch.from_numpy(embedding).float().to(self.device)
        logger.info(f"Speaker embedding set from {len(audio_files)} files")
