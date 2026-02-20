"""Karaoke session and separation job manager.

Manages asynchronous vocal separation jobs and karaoke session state.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import torch
import torchaudio
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio file using soundfile (workaround for torchaudio torchcodec issue).

    Args:
        path: Path to audio file

    Returns:
        Tuple of (audio tensor [channels, samples], sample rate)
    """
    try:
        # Try soundfile first (works for wav, flac, ogg)
        data, sr = sf.read(path, dtype='float32')
        # Convert to torch tensor with channels first
        if data.ndim == 1:
            audio = torch.from_numpy(data).unsqueeze(0)
        else:
            audio = torch.from_numpy(data.T)
        return audio, sr
    except Exception as e:
        logger.warning(f"soundfile failed, trying librosa: {e}")
        # Fallback to librosa for mp3/m4a
        import librosa
        data, sr = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return torch.from_numpy(data), sr


def save_audio(path: str, audio: torch.Tensor, sample_rate: int):
    """Save audio file using soundfile (workaround for torchaudio torchcodec issue).

    Args:
        path: Output path
        audio: Audio tensor [channels, samples]
        sample_rate: Sample rate
    """
    # Convert to numpy with channels last for soundfile
    data = audio.numpy()
    if data.ndim == 2:
        data = data.T  # [samples, channels]
    sf.write(path, data, sample_rate)


class KaraokeManager:
    """Manages karaoke separation jobs and sessions.

    Handles:
    - Async vocal/instrumental separation using MelBandRoFormer
    - Job progress tracking and callbacks
    - Output file management
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        output_dir: str = '/tmp/autovoice_karaoke',
        max_workers: int = 2,
        progress_callback: Optional[Callable[[str, int, str], None]] = None
    ):
        """Initialize KaraokeManager.

        Args:
            device: Torch device for separation model
            output_dir: Directory to store separated audio files
            max_workers: Max concurrent separation jobs
            progress_callback: Callback(job_id, progress, status) for updates
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.progress_callback = progress_callback

        # Thread pool for async separation
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Lazy-load separator
        self._separator = None

    def _get_separator(self):
        """Lazy-load the vocal separator."""
        if self._separator is None:
            try:
                from ..audio.separator import MelBandRoFormer
                self._separator = MelBandRoFormer(device=self.device)
                self._separator.to(self.device)
                self._separator.eval()
                logger.info(f"MelBandRoFormer loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load separator: {e}")
                raise RuntimeError(f"Separator initialization failed: {e}")
        return self._separator

    def start_separation(self, job_id: str, audio_path: str) -> bool:
        """Start async vocal separation job.

        Args:
            job_id: Unique job identifier
            audio_path: Path to input audio file

        Returns:
            True if job started successfully
        """
        with self._lock:
            if job_id in self._jobs:
                logger.warning(f"Job {job_id} already exists")
                return False

            self._jobs[job_id] = {
                'status': 'queued',
                'progress': 0,
                'input_path': audio_path,
                'vocals_path': None,
                'instrumental_path': None,
                'error': None,
                'started_at': time.time()
            }

        # Submit to thread pool
        self._executor.submit(self._run_separation, job_id, audio_path)
        logger.info(f"Separation job {job_id} queued")
        return True

    def _run_separation(self, job_id: str, audio_path: str):
        """Execute separation in background thread.

        Args:
            job_id: Job identifier
            audio_path: Input audio path
        """
        try:
            self._update_job(job_id, status='processing', progress=10)

            # Load audio (using soundfile to avoid torchaudio torchcodec issue)
            logger.info(f"Job {job_id}: Loading audio from {audio_path}")
            audio, sr = load_audio(audio_path)
            self._update_job(job_id, progress=20)

            # Resample to 44.1kHz if needed (separator expects 44.1kHz)
            target_sr = 44100
            if sr != target_sr:
                logger.info(f"Job {job_id}: Resampling {sr}Hz -> {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                audio = resampler(audio)
            self._update_job(job_id, progress=30)

            # Convert to mono if stereo for processing
            if audio.shape[0] > 1:
                audio_mono = audio.mean(dim=0, keepdim=True)
            else:
                audio_mono = audio

            # Move to device
            audio_mono = audio_mono.to(self.device)
            self._update_job(job_id, progress=40)

            # Run separation
            logger.info(f"Job {job_id}: Running vocal separation")
            separator = self._get_separator()

            with torch.no_grad():
                vocals, instrumental = separator.separate(audio_mono)

            self._update_job(job_id, progress=80)

            # Save outputs
            vocals_path = self.output_dir / f"{job_id}_vocals.wav"
            instrumental_path = self.output_dir / f"{job_id}_instrumental.wav"

            # Move back to CPU for saving
            vocals_cpu = vocals.cpu()
            instrumental_cpu = instrumental.cpu()

            save_audio(str(vocals_path), vocals_cpu, target_sr)
            save_audio(str(instrumental_path), instrumental_cpu, target_sr)

            self._update_job(
                job_id,
                status='completed',
                progress=100,
                vocals_path=str(vocals_path),
                instrumental_path=str(instrumental_path)
            )
            logger.info(f"Job {job_id}: Separation complete")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            self._update_job(job_id, status='failed', error=str(e))

    def _update_job(self, job_id: str, **kwargs):
        """Update job state and notify callback.

        Args:
            job_id: Job identifier
            **kwargs: Fields to update
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
                job = self._jobs[job_id]

        # Notify callback
        if self.progress_callback:
            try:
                self.progress_callback(
                    job_id,
                    kwargs.get('progress', job.get('progress', 0)),
                    kwargs.get('status', job.get('status', 'unknown'))
                )
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status.

        Args:
            job_id: Job identifier

        Returns:
            Job status dict or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id, {}).copy() if job_id in self._jobs else None

    def get_separated_paths(self, job_id: str) -> Optional[tuple]:
        """Get paths to separated audio files.

        Args:
            job_id: Job identifier

        Returns:
            (vocals_path, instrumental_path) or None
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.get('status') == 'completed':
                return job.get('vocals_path'), job.get('instrumental_path')
        return None

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
        logger.info("KaraokeManager shutdown")
