"""Training pipeline for So-VITS-SVC model.

Supports dataset loading, distributed training, loss computation,
checkpointing, and quality assessment.
"""
import logging
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)


class TrainingCancelledError(RuntimeError):
    """Raised when an in-flight training run is cancelled."""


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
        self.min_source_duration_seconds = 0.5
        self.min_active_ratio = 0.08
        self.min_voiced_ratio = 0.02
        self.min_peak_amplitude = 1e-4
        self.min_speaker_purity = 0.75
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

    def _load_quality_sidecar(self, audio_path: Path) -> Dict[str, Any]:
        sidecar = audio_path.with_suffix('.json')
        if not sidecar.exists():
            return {}
        try:
            return json.loads(sidecar.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid quality metadata for {audio_path}: {exc}") from exc

    def _validate_quality_gates(
        self,
        *,
        audio_path: Path,
        raw_audio: np.ndarray,
        f0: np.ndarray,
        voiced: Optional[np.ndarray],
    ) -> None:
        if raw_audio.size < int(self.sample_rate * self.min_source_duration_seconds):
            raise ValueError(f"Training sample too short for stable features: {audio_path}")

        peak = float(np.max(np.abs(raw_audio))) if raw_audio.size else 0.0
        if peak < self.min_peak_amplitude:
            raise ValueError(f"Training sample is effectively silent: {audio_path}")

        active_ratio = float(np.mean(np.abs(raw_audio) >= self.min_peak_amplitude))
        if active_ratio < self.min_active_ratio:
            raise ValueError(f"Training sample is silence-heavy: {audio_path}")

        voiced_mask = np.asarray(voiced, dtype=bool) if voiced is not None else (f0 > 0)
        voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0
        if voiced_ratio < self.min_voiced_ratio:
            raise ValueError(f"Training sample has insufficient pitch coverage: {audio_path}")

        metadata = self._load_quality_sidecar(audio_path)
        quality_metadata = metadata.get('quality_metadata') if isinstance(metadata, dict) else None
        if isinstance(quality_metadata, dict):
            speaker_purity = quality_metadata.get('speaker_purity')
            diarization_ok = quality_metadata.get('diarization_ok')
        else:
            speaker_purity = metadata.get('speaker_purity') if isinstance(metadata, dict) else None
            diarization_ok = metadata.get('diarization_ok') if isinstance(metadata, dict) else None

        if speaker_purity is not None and float(speaker_purity) < self.min_speaker_purity:
            raise ValueError(f"Training sample failed speaker purity gate: {audio_path}")
        if diarization_ok is False:
            raise ValueError(f"Training sample failed diarization gate: {audio_path}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import librosa
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        raw_audio = np.asarray(audio, dtype=np.float32)

        if len(raw_audio) > self.segment_length:
            start = np.random.randint(0, len(raw_audio) - self.segment_length)
            audio = raw_audio[start:start + self.segment_length]
        else:
            audio = np.pad(raw_audio, (0, self.segment_length - len(raw_audio)))

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
        self._validate_quality_gates(
            audio_path=audio_path,
            raw_audio=raw_audio,
            f0=f0,
            voiced=voiced,
        )

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

        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses: List[float] = []
        self.resume_event = None
        self.cancel_event = None
        self.on_batch_end = None
        self.checkpoint_interval_steps = int(self.config.get('checkpoint_interval_steps', 0))
        self.validation_split = float(self.config.get('validation_split', 0.0) or 0.0)
        self.early_stopping_patience = int(self.config.get('early_stopping_patience', 0) or 0)
        self.early_stopping_min_delta = float(self.config.get('early_stopping_min_delta', 0.0) or 0.0)
        self._early_stopping_wait = 0

        # Shared encoders for feature extraction (frozen during training)
        from ..models.encoder import ContentEncoder, PitchEncoder
        from ..models.feature_contract import feature_contract_from_model
        self.feature_contract = feature_contract_from_model(self.model, self.config)
        self.content_encoder = ContentEncoder(
            output_size=self.feature_contract.content_dim,
            encoder_backend='contentvec',
            device=self.device
        ).to(self.device)
        self.pitch_encoder = PitchEncoder(output_size=self.feature_contract.pitch_dim).to(self.device)
        self.speaker_embedding: Optional[torch.Tensor] = None
        self.sample_rate = self.config.get('sample_rate', 22050)

        betas = (
            float(self.config.get('adam_beta1', 0.8)),
            float(self.config.get('adam_beta2', 0.99)),
        )
        weight_decay = float(self.config.get('weight_decay', 0.01))
        optimizer_name = str(self.config.get('optimizer', 'adamw')).lower()
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay,
            )

        scheduler_name = str(self.config.get('scheduler', 'exponential')).lower()
        if scheduler_name == 'none':
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=float(self.config.get('scheduler_gamma', 0.999)),
            )

    def train(self, train_dir: str, val_dir: Optional[str] = None,
              resume_from: Optional[str] = None):
        """Run training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        train_dataset = VoiceDataset(train_dir, segment_length=32768)
        val_dataset = None
        if not val_dir and 0.0 < self.validation_split < 1.0 and len(train_dataset) > 1:
            val_size = max(1, int(round(len(train_dataset) * self.validation_split)))
            train_size = len(train_dataset) - val_size
            if train_size > 0:
                train_dataset, val_dataset = random_split(
                    train_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42),
                )
        # drop_last=False when dataset < batch_size to ensure training can proceed
        actual_batch_size = min(self.batch_size, len(train_dataset))
        train_loader = DataLoader(
            train_dataset, batch_size=actual_batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
            drop_last=(len(train_dataset) >= self.batch_size),
        )

        val_loader = None
        if val_dir and Path(val_dir).exists():
            val_dataset = VoiceDataset(val_dir, segment_length=32768)

        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=False, num_workers=2)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting training: {self.epochs} epochs, "
                    f"{len(train_dataset)} samples, device={self.device}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(epoch_loss)

            val_loss = None
            if val_loader:
                val_loss = self.assess(val_loader)
                logger.info(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, "
                            f"val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}")

            monitored_loss = val_loss if val_loss is not None else epoch_loss
            previous_best_loss = self.best_loss
            should_stop = self._should_stop_early(monitored_loss)
            if val_loader and self.best_loss < previous_best_loss:
                self.save_checkpoint(self.checkpoint_dir / 'best.pth')

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

            if self.scheduler is not None:
                self.scheduler.step()

            if should_stop:
                logger.info(
                    "Early stopping after epoch %s: best_loss=%.4f, current_loss=%.4f",
                    epoch + 1,
                    self.best_loss,
                    monitored_loss,
                )
                break

        self.save_checkpoint(self.checkpoint_dir / 'final.pth')
        logger.info("Training complete")

    def _should_stop_early(self, monitored_loss: float) -> bool:
        """Return True when configured early stopping has run out of patience."""
        improvement = self.best_loss - float(monitored_loss)
        if improvement > self.early_stopping_min_delta:
            self.best_loss = float(monitored_loss)
            self._early_stopping_wait = 0
            return False

        if self.early_stopping_patience <= 0:
            return False

        self._early_stopping_wait += 1
        return self._early_stopping_wait >= self.early_stopping_patience

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise TrainingCancelledError("Training cancelled by user")

            if self.resume_event is not None:
                self.resume_event.wait()
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise TrainingCancelledError("Training cancelled by user")

            audio = batch['audio'].to(self.device)   # [B, segment_length]
            mel = batch['mel'].to(self.device)        # [B, 80, T_mel]
            f0 = batch['f0'].to(self.device)          # [B, T_f0]

            n_mel_frames = mel.shape[2]

            # Extract content features from training audio (no grad - frozen encoder)
            with torch.no_grad():
                content = self.content_encoder.extract_features(
                    audio, sr=self.sample_rate
                )  # [B, N, 768]
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

            # Content is now 768-dim natively for best quality
            # [B, N, 768] from ContentVec, [B, N, 768] from PitchEncoder

            # Speaker embedding (same for all batches of this speaker)
            if self.speaker_embedding is None:
                raise RuntimeError("Speaker embedding not set. Call set_speaker_embedding() first.")
            speaker = self.speaker_embedding.unsqueeze(0).expand(audio.shape[0], -1)
            self._validate_model_feature_shapes(content, pitch, speaker)

            # Compute spectrogram for posterior encoder
            spec = self._compute_spec(audio, n_mel_frames)

            # Forward + loss
            self.optimizer.zero_grad()
            outputs = self.model(content, pitch, speaker, spec=spec)
            losses = self.model.compute_loss(outputs, mel)
            loss = losses['total_loss']

            # Skip NaN/Inf losses to prevent gradient explosion
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Step {self.global_step}: Skipping NaN/Inf loss")
                continue

            # Clamp large losses to prevent gradient explosion
            loss = torch.clamp(loss, max=1e6)

            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.checkpoint_interval_steps > 0 and self.global_step % self.checkpoint_interval_steps == 0:
                self.save_checkpoint(self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pth')

            if callable(self.on_batch_end):
                try:
                    self.on_batch_end({
                        'epoch': epoch + 1,
                        'total_epochs': self.epochs,
                        'step': batch_idx,
                        'total_steps': total_batches,
                        'global_step': self.global_step,
                        'loss': float(loss.item()),
                        'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                    })
                except Exception as exc:
                    logger.debug("on_batch_end callback failed: %s", exc)

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

                # Content is 768-dim natively for best quality

                if self.speaker_embedding is None:
                    raise RuntimeError("Speaker embedding not set.")
                speaker = self.speaker_embedding.unsqueeze(0).expand(audio.shape[0], -1)
                self._validate_model_feature_shapes(content, pitch, speaker)
                spec = self._compute_spec(audio, n_mel_frames)

                outputs = self.model(content, pitch, speaker, spec=spec)
                losses = self.model.compute_loss(outputs, mel)
                total_loss += losses['total_loss'].item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint.

        If model has LoRA injected, saves only LoRA weights (~2MB).
        Otherwise saves full model weights.
        """
        if path is None:
            path = self.checkpoint_dir / 'latest.pth'
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check if model has LoRA injected (CoMoSVCDecoder)
        has_lora = getattr(self.model, '_lora_injected', False)

        if has_lora:
            # Save only LoRA adapter weights (small file ~2MB)
            lora_state = self.model.get_lora_state_dict()
            checkpoint = {
                'lora_state': lora_state,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'config': self.config,
                'is_lora': True,
            }
            torch.save(checkpoint, str(path))
            size_kb = path.stat().st_size / 1024
            logger.info(f"LoRA checkpoint saved: {path} ({size_kb:.1f} KB)")
        else:
            # Save full model checkpoint
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'config': self.config,
                'is_lora': False,
            }
            torch.save(checkpoint, str(path))
            logger.info(f"Full checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Handles both LoRA-only checkpoints and full model checkpoints.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)

        if checkpoint.get('is_lora', False):
            # Load LoRA weights into model (model must have LoRA injected)
            if not getattr(self.model, '_lora_injected', False):
                raise RuntimeError("Checkpoint is LoRA but model doesn't have LoRA injected")
            self.model.load_lora_state_dict(checkpoint['lora_state'])
            logger.info(f"Loaded LoRA checkpoint from {path}")
        else:
            # Load full model state
            self.model.load_state_dict(checkpoint['model'])
            # Legacy checkpoints may have content_proj - ignore it (now using 768-dim natively)
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None and checkpoint.get('scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f"Loaded full checkpoint from {path}")

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

    def _validate_model_feature_shapes(
        self,
        content: torch.Tensor,
        pitch: torch.Tensor,
        speaker: torch.Tensor,
    ) -> None:
        """Validate encoder feature dimensions against the decoder contract."""
        expected_content_dim = getattr(self.model, "content_dim", None)
        expected_pitch_dim = getattr(self.model, "pitch_dim", None)
        expected_speaker_dim = getattr(self.model, "speaker_dim", None)

        mismatches = []
        if expected_content_dim is not None and content.shape[-1] != expected_content_dim:
            mismatches.append(
                f"content_dim={content.shape[-1]} expected {expected_content_dim}"
            )
        if expected_pitch_dim is not None and pitch.shape[-1] != expected_pitch_dim:
            mismatches.append(f"pitch_dim={pitch.shape[-1]} expected {expected_pitch_dim}")
        if expected_speaker_dim is not None and speaker.shape[-1] != expected_speaker_dim:
            mismatches.append(
                f"speaker_dim={speaker.shape[-1]} expected {expected_speaker_dim}"
            )

        if mismatches:
            raise RuntimeError(
                "Training feature shape mismatch: "
                + ", ".join(mismatches)
                + ". Check encoder and decoder dimension configuration."
            )

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


# Alias for backwards compatibility
VoiceTrainer = Trainer
