"""Training pipeline for So-VITS-SVC model.

Supports dataset loading, distributed training, loss computation,
checkpointing, and quality assessment.
"""
import logging
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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
        # Decoded-audio cache: librosa.load+resample of a full source file costs
        # ~1.5s and __getitem__ repeats it every epoch, starving the GPU.
        # Voice-profile datasets are minutes of mono float32, so unbounded is fine.
        self._audio_cache: Dict[Path, np.ndarray] = {}
        # Full-file F0 cache: librosa.pyin is single-threaded CPU and costs
        # ~0.2s per second of audio; computing it per random window per epoch
        # dominates training wall-time. Computed once per file and sliced per
        # window (valid only without augmentation, which changes the audio).
        # Long-window pyin also matches how serving extracts pitch.
        self._f0_cache: Dict[Path, tuple] = {}
        self._f0_hop = 512
        # Full-file ContentVec features, precomputed to a .content.npy sidecar
        # (on GPU in the main process — the CPU dataloader workers can't run
        # ContentVec) and sliced per window here, so the GPU never waits on
        # per-batch content extraction (the jumpy-utilisation bottleneck).
        self._content_cache: Dict[Path, np.ndarray] = {}
        # Full-file log-mel cache: librosa.feature.melspectrogram is FFT work
        # recomputed from scratch on every __getitem__ call otherwise -- with
        # content/F0 already cached above, this becomes the dominant per-item
        # cost and starves the GPU exactly like the F0 case did. Same
        # once-per-file, sliced-per-window pattern as _full_file_f0 (valid
        # only without augmentation, which changes the audio).
        self._mel_cache: Dict[Path, np.ndarray] = {}
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

    def _full_file_f0(self, audio_path: Path, raw_audio: np.ndarray):
        """Full-file pyin F0/voiced contour, cached in memory and on disk.

        librosa.pyin is single-threaded and costs ~20-35s on a 2-5 minute
        vocal track — the dominant per-epoch stall that starves the GPU.
        The contour only depends on the file (at a fixed sample rate/hop),
        so compute it at most once ever: an on-disk .f0.npz sidecar is
        shared across the 8 dataloader workers and across training runs;
        the in-memory dict avoids re-reading disk within a worker.
        """
        import librosa

        cached = self._f0_cache.get(audio_path)
        if cached is not None:
            return cached

        sidecar = audio_path.with_suffix('.f0.npz')
        if sidecar.exists():
            try:
                data = np.load(sidecar)
                if (int(data['sample_rate']) == int(self.sample_rate)
                        and int(data['hop']) == int(self._f0_hop)):
                    cached = (data['f0'].astype(np.float32),
                              data['voiced'].astype(bool))
                    self._f0_cache[audio_path] = cached
                    return cached
            except Exception:
                pass  # corrupt/old sidecar — recompute below

        f0_full, voiced_full, _ = librosa.pyin(
            raw_audio, fmin=50, fmax=1100, sr=self.sample_rate,
        )
        f0_full = np.nan_to_num(np.asarray(f0_full, dtype=np.float32), nan=0.0)
        voiced_full = np.asarray(voiced_full, dtype=bool)
        try:
            np.savez(sidecar, f0=f0_full, voiced=voiced_full,
                     sample_rate=self.sample_rate, hop=self._f0_hop)
        except Exception:
            pass  # cache is an optimization; never fail training on it
        cached = (f0_full, voiced_full)
        self._f0_cache[audio_path] = cached
        return cached

    def _window_f0(self, audio_path: Path, raw_audio: np.ndarray,
                   start: int, window_len: int):
        """Slice a window's F0 from the cached full-file pyin contour."""
        f0_full, voiced_full = self._full_file_f0(audio_path, raw_audio)
        expected = 1 + window_len // self._f0_hop
        frame_start = start // self._f0_hop
        f0 = f0_full[frame_start:frame_start + expected]
        voiced = voiced_full[frame_start:frame_start + expected]
        if len(f0) < expected:
            f0 = np.pad(f0, (0, expected - len(f0)))
            voiced = np.pad(voiced, (0, expected - len(voiced)))
        return f0.copy(), voiced.copy()

    # Mel contract fixed across this class: 80 mels, n_fft=1024, hop=256,
    # win_length=1024, fmin=0, fmax=8000, power=1.0 -- matches the vocoder
    # (see __getitem__ below and sota_pipeline._load_universal_vocoder).
    _MEL_HOP = 256
    _MEL_N_MELS = 80

    def _full_file_mel(self, audio_path: Path, raw_audio: np.ndarray) -> np.ndarray:
        """Full-file log-mel, cached in memory and on disk (mirrors _full_file_f0)."""
        import librosa

        cached = self._mel_cache.get(audio_path)
        if cached is not None:
            return cached

        sidecar = audio_path.with_suffix('.mel.npy')
        if sidecar.exists():
            try:
                mel_full = np.load(sidecar)
                if mel_full.shape[0] == self._MEL_N_MELS:
                    self._mel_cache[audio_path] = mel_full
                    return mel_full
            except Exception:
                pass  # corrupt/old sidecar -- recompute below

        mel_full = librosa.feature.melspectrogram(
            y=raw_audio, sr=self.sample_rate, n_fft=1024, hop_length=self._MEL_HOP,
            win_length=1024, n_mels=self._MEL_N_MELS, fmin=0, fmax=8000, power=1.0,
        ).astype(np.float32)
        try:
            np.save(sidecar, mel_full)
        except Exception:
            pass  # cache is an optimization; never fail training on it
        self._mel_cache[audio_path] = mel_full
        return mel_full

    def _window_mel(self, audio_path: Path, raw_audio: np.ndarray,
                     start: int, window_len: int) -> np.ndarray:
        """Slice a window's mel from the cached full-file log-mel (mirrors _window_f0)."""
        mel_full = self._full_file_mel(audio_path, raw_audio)
        expected = 1 + window_len // self._MEL_HOP
        frame_start = start // self._MEL_HOP
        mel = mel_full[:, frame_start:frame_start + expected]
        if mel.shape[1] < expected:
            mel = np.pad(mel, ((0, 0), (0, expected - mel.shape[1])))
        return mel.copy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # A specific file can fail every quality-gate retry below (e.g. it's
        # an all-pause/all-breath clip) without the speaker's corpus itself
        # being bad -- with hundreds of files per speaker over a multi-hour,
        # multi-thousand-epoch run, that WILL happen eventually. Substitute a
        # different file from this same dataset instead of raising, so one
        # bad recording can't crash an otherwise-healthy run. Bounded to the
        # file count so a genuinely all-bad dataset still fails loudly rather
        # than recursing forever.
        tried: set = set()
        while True:
            item = self._try_load_item(idx)
            if item is not None:
                return item
            tried.add(idx)
            logger.warning(
                "VoiceDataset: %s failed every quality-gate retry, "
                "substituting a different file", self.audio_files[idx],
            )
            remaining = [i for i in range(len(self.audio_files)) if i not in tried]
            if not remaining:
                raise ValueError(
                    f"VoiceDataset: no file in {self.data_dir} passed quality "
                    f"gates (tried all {len(self.audio_files)})"
                )
            idx = int(np.random.choice(remaining))

    def _try_load_item(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load one training item, or None if every quality-gate retry failed."""
        import librosa
        audio_path = self.audio_files[idx]
        raw_audio = self._audio_cache.get(audio_path)
        if raw_audio is None:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            raw_audio = np.asarray(audio, dtype=np.float32)
            self._audio_cache[audio_path] = raw_audio

        # Gates run on a random window each epoch, so any rare quiet stretch
        # in an otherwise-valid sample would eventually be drawn and crash a
        # long run. Redraw failing windows instead; a sample only fails when
        # every attempt fails (i.e. the sample itself is bad, not the draw).
        last_gate_error: Optional[ValueError] = None
        for _ in range(10):
            start = 0
            if len(raw_audio) > self.segment_length:
                start = np.random.randint(0, len(raw_audio) - self.segment_length)
                audio = raw_audio[start:start + self.segment_length]
            else:
                audio = np.pad(raw_audio, (0, self.segment_length - len(raw_audio)))

            # Apply augmentation if enabled
            if self._augmentation is not None:
                audio = self._augmentation(audio, self.sample_rate)

            if self._augmentation is None:
                f0, voiced = self._window_f0(audio_path, raw_audio, start, len(audio))
            else:
                f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=self.sample_rate)
            f0 = np.nan_to_num(f0, nan=0.0)
            try:
                self._validate_quality_gates(
                    audio_path=audio_path,
                    raw_audio=raw_audio,
                    f0=f0,
                    voiced=voiced,
                )
                last_gate_error = None
                break
            except ValueError as exc:
                last_gate_error = exc
                if len(raw_audio) <= self.segment_length:
                    break  # only one possible window; redrawing cannot help
        if last_gate_error is not None:
            return None

        # Mel MUST match the vocoder's expected format or the decoder learns a
        # representation no vocoder can render (the cause of the whine).
        # Verified by copy-synthesis (pitch corr 0.988): the universal HiFiGAN
        # wants magnitude log-mel at n_fft=1024, hop=256, fmin=0, fmax=8000.
        # Cached full-file (see _window_mel) when there's no augmentation to
        # invalidate it -- recomputing this FFT on every __getitem__ call is
        # the dominant per-item cost once content/F0 are already cached.
        if self._augmentation is None:
            mel = self._window_mel(audio_path, raw_audio, start, len(audio))
        else:
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_fft=1024, hop_length=256,
                win_length=1024, n_mels=80, fmin=0, fmax=8000, power=1.0,
            )
        log_mel = np.log(np.clip(mel, 1e-5, None)).astype(np.float32)
        # Normalize to [0,1] so the decoder can actually fit it at lr 1e-4;
        # serving denormalizes back to log-mel for the vocoder.
        from ..models.vocoder import normalize_log_mel
        mel_db = np.clip(normalize_log_mel(log_mel), 0.0, 1.0).astype(np.float32)

        item = {
            'audio': torch.from_numpy(audio).float(),
            'mel': torch.from_numpy(mel_db).float(),
            'f0': torch.from_numpy(f0).float(),
            'path': str(audio_path),
        }

        # Cached content: slice the full-file features for this window and align
        # to the mel frame count (fixed -> batch-collatable). Absent sidecar ->
        # the trainer extracts content itself (fallback).
        content_full = self._content_cache.get(audio_path)
        if content_full is None:
            sidecar = audio_path.with_suffix('.content.npy')
            if sidecar.exists():
                try:
                    content_full = np.load(sidecar)
                    self._content_cache[audio_path] = content_full
                except Exception:
                    content_full = None
        if content_full is not None and content_full.shape[0] > 0:
            tc, span = content_full.shape[0], max(len(raw_audio), 1)
            cs = min(int(start / span * tc), tc - 1)
            ce = max(min(int((start + len(audio)) / span * tc), tc), cs + 1)
            cw = torch.from_numpy(content_full[cs:ce]).float().t().unsqueeze(0)  # [1,768,t]
            n_frames = mel_db.shape[1]
            cw = torch.nn.functional.interpolate(cw, size=n_frames, mode='linear', align_corners=False)
            item['content'] = cw.squeeze(0).t().contiguous()  # [n_frames, 768]

        return item


def resolve_precision(precision: str, capability: Optional[tuple]) -> str:
    """Downgrade bf16 to fp16 on pre-Ampere GPUs.

    bf16 autocast needs Ampere (compute capability sm80+). On older CUDA GPUs
    (e.g. V100 sm70) it is unsupported/unstable, so a bf16 request is downgraded
    to fp16. ``capability`` is a ``(major, minor)`` tuple or ``None`` (unknown ->
    leave unchanged).
    """
    if precision == 'bf16' and capability is not None and capability[0] < 8:
        return 'fp16'
    return precision
class MultiSpeakerVoiceDataset(Dataset):
    """Compose per-speaker ``VoiceDataset``s and tag every item with that
    speaker's embedding, so one training run learns a universal, speaker-
    conditioned decoder.

    Each speaker's embedding is computed with the SAME method used at serving
    (``VoiceCloner.create_speaker_embedding`` = mel-statistics), so the
    conditioning the decoder learns matches what the profile store hands the
    pipeline at inference. Items keep ``VoiceDataset``'s fixed shapes, so they
    still collate into batches; the added ``speaker`` is a fixed 256-dim vector.
    """

    def __init__(self, speaker_dirs: Dict[str, str], sample_rate: int = 22050,
                 segment_length: int = 32768, device: str = 'cpu',
                 augment: bool = False):
        from ..inference.voice_cloner import VoiceCloner
        self.datasets: List[VoiceDataset] = []
        self.embeddings: List[torch.Tensor] = []
        self.speaker_names: List[str] = []
        self._index: List[Tuple[int, int]] = []  # (dataset_idx, local_idx)
        excluded: List[str] = []
        cloner = VoiceCloner(device=device)
        for name, audio_dir in sorted(speaker_dirs.items()):
            ds = VoiceDataset(audio_dir, sample_rate=sample_rate,
                              segment_length=segment_length, augment=augment)
            if len(ds) == 0:
                logger.warning("MultiSpeakerVoiceDataset: no audio for speaker "
                               "'%s' in %s (skipped)", name, audio_dir)
                excluded.append(name)
                continue
            # create_speaker_embedding requires every file to clear a 3s floor;
            # raw corpora routinely have short tail clips (e.g. the last chunk
            # of a fixed-length split) that VoiceDataset itself pads and trains
            # on fine, so filter here rather than let one short file abort the
            # whole speaker.
            import soundfile as sf
            files = []
            for p in ds.audio_files:
                try:
                    info = sf.info(str(p))
                    if info.frames / info.samplerate >= 3.0:
                        files.append(str(p))
                except Exception:
                    continue
            if not files:
                logger.warning("MultiSpeakerVoiceDataset: speaker '%s' has no "
                               "file >=3s for embedding (skipped)", name)
                excluded.append(name)
                continue
            # A file can pass the sf.info duration check yet still fail inside
            # create_speaker_embedding (its own re-measurement can disagree by
            # a few ms at the boundary) -- don't let one speaker's embedding
            # failure abort every other speaker's dataset build.
            try:
                emb = torch.from_numpy(cloner.create_speaker_embedding(files)).float()
            except Exception as exc:
                logger.warning("MultiSpeakerVoiceDataset: embedding failed for "
                               "speaker '%s' (skipped): %s", name, exc)
                excluded.append(name)
                continue
            di = len(self.datasets)
            self.datasets.append(ds)
            self.embeddings.append(emb)
            self.speaker_names.append(name)
            self._index.extend((di, li) for li in range(len(ds)))
            logger.info("MultiSpeakerVoiceDataset: speaker '%s' -> %d windows",
                        name, len(ds))
        if not self._index:
            raise RuntimeError("MultiSpeakerVoiceDataset: no training data found "
                               "across any speaker")
        if excluded:
            logger.warning(
                "MultiSpeakerVoiceDataset: %d of %d requested speakers EXCLUDED "
                "entirely (no usable audio): %s -- the universal decoder will "
                "train WITHOUT these voices. Included: %s",
                len(excluded), len(speaker_dirs), excluded, self.speaker_names,
            )
        logger.info("MultiSpeakerVoiceDataset: %d speakers, %d total windows",
                    len(self.datasets), len(self._index))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        di, li = self._index[idx]
        item = self.datasets[di][li]
        item['speaker'] = self.embeddings[di]  # [256], same for this speaker
        return item


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
        self.early_stopped = False
        self.epochs_ran = 0
        self.monitored_metric = 'train'
        self.warmup_steps = int(self.config.get('warmup_steps', 0) or 0)
        self.precision = str(self.config.get('precision', 'fp32')).lower()
        # Set when bf16 is downgraded to fp16 on pre-Ampere GPUs so the job
        # manager can surface the coercion to the user (live log + results).
        self.precision_coerced_from: Optional[str] = None
        use_cuda = str(self.device) != 'cpu' and torch.cuda.is_available()
        # bf16 needs Ampere (sm80+). On pre-Ampere CUDA GPUs (e.g. V100 sm70)
        # autocast bf16 is unsupported/unstable, so downgrade to fp16.
        if self.precision == 'bf16' and use_cuda:
            try:
                cap = torch.cuda.get_device_capability(self.device)
            except Exception:
                # Fail safe: unknown capability due to an error -> treat as
                # pre-Ampere so bf16 downgrades to (universally safe) fp16.
                cap = (0, 0)
            resolved = resolve_precision(self.precision, cap)
            if resolved != self.precision:
                logger.warning(
                    "bf16 precision requested but device capability %s < (8,0); "
                    "using %s instead.", cap, resolved,
                )
                self.precision_coerced_from = self.precision
                self.precision = resolved
        self._autocast_dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }.get(self.precision)
        self._autocast_enabled = bool(self._autocast_dtype is not None and use_cuda)
        self._grad_scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.precision == 'fp16' and use_cuda)
        )

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

        # Opt-in adversarial (mel-GAN) training: a mel discriminator pushes the
        # generator off the L1 conditional mean so mels keep their sharp
        # harmonic structure. The discriminator is training-only; the generator
        # (self.model) serves unchanged.
        self.adversarial = bool(self.config.get('adversarial', False))
        self.discriminator = None
        self.optimizer_d = None
        if self.adversarial:
            from ..models.mel_discriminator import MultiScaleMelDiscriminator
            self.discriminator = MultiScaleMelDiscriminator().to(self.device)
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay,
            )
            self.adv_lambda_l1 = float(self.config.get('adv_lambda_l1', 45.0))
            self.adv_lambda_fm = float(self.config.get('adv_lambda_fm', 2.0))

    def _precompute_content_cache(self, dataset) -> None:
        """Extract full-file ContentVec features to .content.npy sidecars once
        (GPU, main process). Chunked to bound memory; workers load + slice them.

        Handles both a plain ``VoiceDataset`` (has ``audio_files``) and a
        ``MultiSpeakerVoiceDataset`` (has ``datasets``, one ``VoiceDataset`` per
        speaker) -- callers that pass the latter get real precompute instead of
        a silent no-op just because the top-level object has no ``audio_files``.
        """
        sub_datasets = getattr(dataset, 'datasets', None)
        if sub_datasets:
            files = [p for ds in sub_datasets for p in getattr(ds, 'audio_files', [])]
        else:
            files = getattr(dataset, 'audio_files', None)
        if not files:
            return
        import librosa
        chunk = int(self.sample_rate * 30)  # 30s chunks
        made = 0
        for path in files:
            path = Path(path)
            sidecar = path.with_suffix('.content.npy')
            if sidecar.exists():
                continue
            try:
                audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
                parts = []
                with torch.no_grad():
                    for i in range(0, len(audio), chunk):
                        seg = audio[i:i + chunk]
                        if len(seg) < 512:
                            continue
                        feat = self.content_encoder.extract_features(
                            torch.from_numpy(seg).float().unsqueeze(0).to(self.device),
                            sr=self.sample_rate,
                        )
                        parts.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
                if parts:
                    np.save(sidecar, np.concatenate(parts, axis=0))
                    made += 1
            except Exception as exc:
                logger.warning("Content precompute failed for %s: %s", path, exc)
        if made:
            logger.info("Precomputed %d ContentVec sidecars", made)

    def train(self, train_dir: str, val_dir: Optional[str] = None,
              resume_from: Optional[str] = None,
              train_dataset: Optional[Dataset] = None,
              precompute_content: bool = True):
        """Run training loop.

        ``train_dataset`` may be supplied pre-built (e.g. a
        ``MultiSpeakerVoiceDataset`` for universal-decoder training); when given,
        ``train_dir`` is ignored for dataset construction. ``precompute_content``
        can be turned off when the ``.content.npy`` sidecars were already
        extracted out-of-band (keeps the main process from touching the GPU
        before the dataloader forks its workers).
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        if train_dataset is None:
            train_dataset = VoiceDataset(train_dir, segment_length=32768)
        # Precompute full-file ContentVec sidecars once (GPU, main process) so
        # the dataloader workers never wait on content extraction.
        if precompute_content:
            self._precompute_content_cache(train_dataset)
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
        # persistent_workers keeps each worker's decoded-audio cache alive
        # across epochs; without it the cache is rebuilt every epoch.
        # Per-item cost is dominated by single-threaded librosa.pyin, so
        # feed the GPU with more parallel workers and deeper prefetch.
        loader_workers = min(int(self.config.get('num_workers', 12)), os.cpu_count() or 4)
        train_loader_kwargs = dict(
            batch_size=actual_batch_size, shuffle=True,
            num_workers=loader_workers, pin_memory=True,
            drop_last=(len(train_dataset) >= self.batch_size),
        )
        # persistent_workers/prefetch_factor are only valid with num_workers>0.
        # multiprocessing_context='spawn' is required whenever num_workers>0:
        # by the time this runs the caller has already put the model on CUDA,
        # and Linux's default 'fork' start method forking a process with an
        # active CUDA context deadlocks/crashes worker startup. spawn re-execs
        # cleanly instead (slower per-worker startup, but persistent_workers
        # amortizes that across the whole run).
        if loader_workers > 0:
            train_loader_kwargs.update(
                persistent_workers=True, prefetch_factor=4,
                multiprocessing_context='spawn',
            )
        train_loader = DataLoader(train_dataset, **train_loader_kwargs)

        val_loader = None
        if val_dir and Path(val_dir).exists():
            val_dataset = VoiceDataset(val_dir, segment_length=32768)

        if val_dataset is not None:
            val_workers = min(2, loader_workers)
            val_loader_kwargs = dict(batch_size=self.batch_size, shuffle=False,
                                     num_workers=val_workers)
            if val_workers > 0:
                val_loader_kwargs.update(
                    persistent_workers=True, multiprocessing_context='spawn',
                )
            val_loader = DataLoader(val_dataset, **val_loader_kwargs)
            self.monitored_metric = 'val'

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

            self.epochs_ran = epoch + 1
            if should_stop:
                logger.info(
                    "Early stopping after epoch %s: best_loss=%.4f, current_loss=%.4f",
                    epoch + 1,
                    self.best_loss,
                    monitored_loss,
                )
                self.early_stopped = True
                break

        # Ship the best-validation weights, not whatever epoch training
        # stopped on: with early stopping the final epoch is `patience`
        # epochs past the best point by construction.
        best_path = self.checkpoint_dir / 'best.pth'
        if val_loader is not None and best_path.exists():
            epochs_ran = self.epochs_ran
            self.load_checkpoint(best_path)
            self.epochs_ran = epochs_ran
            logger.info(
                "Restored best-validation weights from %s (best epoch %s, val_loss=%.4f)",
                best_path,
                self.current_epoch + 1,
                self.best_loss,
            )

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

            # Content features: use the precomputed cache (already aligned to
            # mel frames) when present, else extract on the fly. Caching keeps
            # the GPU from stalling on per-batch ContentVec extraction.
            with torch.no_grad():
                cached_content = batch.get('content')
                if cached_content is not None and cached_content.numel() > 0:
                    content = cached_content.to(self.device)  # [B, n_mel_frames, 768]
                else:
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

            # Speaker embedding: per-item from the batch when the dataset
            # provides it (multi-speaker / universal-decoder training), else the
            # single fixed embedding broadcast to the batch (single-speaker).
            batch_speaker = batch.get('speaker')
            if batch_speaker is not None and batch_speaker.numel() > 0:
                speaker = batch_speaker.to(self.device)
            elif self.speaker_embedding is not None:
                speaker = self.speaker_embedding.unsqueeze(0).expand(audio.shape[0], -1)
            else:
                raise RuntimeError(
                    "Speaker embedding not set. Call set_speaker_embedding() or "
                    "provide per-item embeddings via a multi-speaker dataset."
                )
            self._validate_model_feature_shapes(content, pitch, speaker)

            # Compute spectrogram for posterior encoder
            spec = self._compute_spec(audio, n_mel_frames)

            # Linear LR warmup over the first warmup_steps global steps.
            # MUST scale from the scheduler's IMMUTABLE base_lrs, never
            # get_last_lr(): ExponentialLR.get_lr() chains off the *current*
            # group lr, so feeding the already-warmup-scaled lr back multiplies
            # `scale` (<1) in every epoch and collapses the lr toward zero
            # (catastrophic when steps/epoch is small -- e.g. 1.3e-35 on a
            # single-file dataset). base_lrs is stable, so warmup is a clean
            # ramp and the scheduler resumes normal decay from base afterward.
            if self.warmup_steps > 0 and self.global_step <= self.warmup_steps:
                scale = min(1.0, (self.global_step + 1) / float(self.warmup_steps))
                base_lrs = (
                    list(self.scheduler.base_lrs)
                    if self.scheduler is not None
                    else [self.lr] * len(self.optimizer.param_groups)
                )
                for group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                    group['lr'] = base_lr * scale

            if self.adversarial:
                # mel-GAN: alternate a discriminator step and a generator step.
                # fp32 (GANs are unstable in fp16); generator serves unchanged.
                from ..models.mel_discriminator import (
                    discriminator_loss, generator_adv_loss, feature_matching_loss,
                )
                gen_mel = self.model(content, pitch, speaker, spec=spec)
                if gen_mel.shape[-1] != mel.shape[-1]:
                    gen_mel = F.interpolate(gen_mel, size=mel.shape[-1],
                                            mode='linear', align_corners=False)
                # D step
                self.optimizer_d.zero_grad()
                real_outs, real_feats = self.discriminator(mel)
                fake_outs, _ = self.discriminator(gen_mel.detach())
                d_loss = discriminator_loss(real_outs, fake_outs)
                d_loss.backward()
                self.optimizer_d.step()
                # G step: L1 anchor + adversarial + feature matching
                self.optimizer.zero_grad()
                fake_outs2, fake_feats2 = self.discriminator(gen_mel)
                g_l1 = F.l1_loss(gen_mel, mel)
                g_adv = generator_adv_loss(fake_outs2)
                g_fm = feature_matching_loss(real_feats, fake_feats2)
                loss = self.adv_lambda_l1 * g_l1 + g_adv + self.adv_lambda_fm * g_fm
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Step {self.global_step}: Skipping NaN/Inf loss")
                    continue
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                losses = {'total_loss': loss, 'reconstruction_loss': g_l1.detach()}
                total_loss += loss.item()
                n_batches += 1
                self.global_step += 1
                if callable(self.on_batch_end):
                    try:
                        self.on_batch_end({
                            'epoch': epoch + 1, 'total_epochs': self.epochs,
                            'step': batch_idx, 'total_steps': total_batches,
                            'global_step': self.global_step, 'loss': float(loss.item()),
                            'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                        })
                    except Exception as exc:
                        logger.debug("on_batch_end callback failed: %s", exc)
                continue

            # Forward + loss (optionally under autocast for fp16/bf16)
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type='cuda',
                dtype=self._autocast_dtype or torch.float16,
                enabled=self._autocast_enabled,
            ):
                outputs = self.model(content, pitch, speaker, spec=spec)
                losses = self.model.compute_loss(outputs, mel)
                loss = losses['total_loss']

            # Skip NaN/Inf losses to prevent gradient explosion
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Step {self.global_step}: Skipping NaN/Inf loss")
                continue

            # Clamp large losses to prevent gradient explosion
            loss = torch.clamp(loss, max=1e6)

            if self._grad_scaler.is_enabled():
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.unscale_(self.optimizer)
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self._grad_scaler.step(self.optimizer)
                self._grad_scaler.update()
            else:
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

                batch_speaker = batch.get('speaker')
                if batch_speaker is not None and batch_speaker.numel() > 0:
                    speaker = batch_speaker.to(self.device)
                elif self.speaker_embedding is not None:
                    speaker = self.speaker_embedding.unsqueeze(0).expand(audio.shape[0], -1)
                else:
                    raise RuntimeError("Speaker embedding not set.")
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
