"""Benchmark dataset loading for voice conversion evaluation.

Supports two dataset layouts:

1. Canonical manifest layout rooted at ``data_dir/metadata.json`` with explicit
   sample definitions and split membership.
2. Legacy recursive directory scanning for existing tests and ad hoc fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import librosa
import numpy as np
import torch
import torchaudio


class BenchmarkDataset:
    """Dataset for voice conversion benchmarking."""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 24000,
        split: Optional[str] = None,
        manifest_name: str = "metadata.json",
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.split = split
        self.manifest_path = self.data_dir / manifest_name
        self.dataset_metadata: Dict[str, Any] = {}
        self.samples: List[Dict[str, Any]] = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Load samples from the manifest when present, else fall back."""
        if not self.data_dir.exists():
            raise RuntimeError(f"Benchmark data directory not found: {self.data_dir}")

        if self._has_root_manifest():
            self._load_manifest_samples()
            return

        self._load_legacy_samples()

    def _has_root_manifest(self) -> bool:
        if not self.manifest_path.exists():
            return False

        try:
            manifest = json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid benchmark manifest: {self.manifest_path}") from exc

        return isinstance(manifest, dict) and isinstance(manifest.get("samples"), list)

    def _load_manifest_samples(self) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        self.dataset_metadata = {
            key: value for key, value in manifest.items() if key != "samples"
        }

        for raw_entry in manifest.get("samples", []):
            sample = self._load_manifest_sample(raw_entry)
            if sample is not None:
                self.samples.append(sample)

    def _load_manifest_sample(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sample_id = str(entry.get("sample_id") or entry.get("id") or "")
        source_rel = entry.get("source_audio") or entry.get("source_path")
        if not source_rel:
            return None

        sample_split = entry.get("split")
        if self.split and sample_split and sample_split != self.split:
            return None

        source_path = self._resolve_path(source_rel)
        if not source_path.exists():
            return None

        try:
            source_audio = self._load_audio(source_path)
        except Exception:
            return None

        reference_path = self._first_existing_path(
            entry.get("reference_audio"),
            entry.get("target_reference_audio"),
        )
        reference_audio = self._safe_load_audio(reference_path)

        target_speaker_path = self._first_existing_path(
            entry.get("target_speaker"),
            entry.get("target_speaker_embedding"),
            entry.get("speaker_embedding"),
        )
        target_speaker = self._load_target_speaker(
            target_speaker_path=target_speaker_path,
            reference_audio=reference_audio,
            source_audio=source_audio,
        )

        instrumental_path = self._first_existing_path(entry.get("instrumental_audio"))
        instrumental_audio = self._safe_load_audio(instrumental_path)

        sample_metadata = dict(self.dataset_metadata.get("defaults", {}))
        sample_metadata.update(entry.get("metadata", {}))
        sample_metadata.setdefault("sample_rate", self.sample_rate)
        if sample_split:
            sample_metadata.setdefault("split", sample_split)

        sample = {
            "sample_id": sample_id or source_path.stem,
            "split": sample_split,
            "source_audio": source_audio,
            "source_path": str(source_path),
            "target_speaker": target_speaker,
            "metadata": sample_metadata,
        }

        if reference_audio is not None and reference_path is not None:
            sample["reference_audio"] = reference_audio
            sample["reference_path"] = str(reference_path)

        if instrumental_audio is not None and instrumental_path is not None:
            sample["instrumental_audio"] = instrumental_audio
            sample["instrumental_path"] = str(instrumental_path)

        if target_speaker_path is not None:
            sample["target_speaker_path"] = str(target_speaker_path)

        return sample

    def _load_legacy_samples(self) -> None:
        audio_files = sorted(self.data_dir.glob("**/*.wav")) + sorted(
            self.data_dir.glob("**/*.mp3")
        )

        for audio_path in audio_files:
            sample = self._load_legacy_sample(audio_path)
            if sample is not None:
                self.samples.append(sample)

    def _load_legacy_sample(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        sample_dir = audio_path.parent

        try:
            source_audio = self._load_audio(audio_path)
        except Exception:
            return None

        reference_path = sample_dir / "reference.wav"
        reference_audio = self._safe_load_audio(reference_path if reference_path.exists() else None)

        target_speaker_path = sample_dir / "speaker.pt"
        target_speaker = self._load_target_speaker(
            target_speaker_path if target_speaker_path.exists() else None,
            reference_audio=reference_audio,
            source_audio=source_audio,
        )

        return {
            "sample_id": audio_path.stem,
            "source_audio": source_audio,
            "source_path": str(audio_path),
            "target_speaker": target_speaker,
            "metadata": {"sample_rate": self.sample_rate},
            **(
                {
                    "reference_audio": reference_audio,
                    "reference_path": str(reference_path),
                }
                if reference_audio is not None
                else {}
            ),
        }

    def _resolve_path(self, relative_or_absolute: str) -> Path:
        candidate = Path(relative_or_absolute)
        if candidate.is_absolute():
            return candidate
        return (self.data_dir / candidate).resolve()

    def _first_existing_path(self, *values: Optional[str]) -> Optional[Path]:
        for value in values:
            if not value:
                continue
            path = self._resolve_path(value)
            if path.exists():
                return path
        return None

    def _safe_load_audio(self, path: Optional[Path]) -> Optional[torch.Tensor]:
        if path is None:
            return None
        try:
            return self._load_audio(path)
        except Exception:
            return None

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            import soundfile as sf

            audio, sr = sf.read(str(audio_path), dtype="float32")
            waveform = torch.tensor(audio, dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.squeeze().float()

    def _load_target_speaker(
        self,
        target_speaker_path: Optional[Path],
        reference_audio: Optional[torch.Tensor],
        source_audio: torch.Tensor,
    ) -> torch.Tensor:
        if target_speaker_path is not None:
            loaded = self._load_embedding_file(target_speaker_path)
            if loaded is not None:
                return loaded

        if reference_audio is not None:
            return self._proxy_embedding(reference_audio)

        return self._proxy_embedding(source_audio)

    def _load_embedding_file(self, path: Path) -> Optional[torch.Tensor]:
        try:
            suffix = path.suffix.lower()

            if suffix == ".npy":
                return torch.from_numpy(np.load(path)).float().flatten()

            if suffix in {".pt", ".pth"}:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(payload, torch.Tensor):
                    return payload.float().flatten()
                if isinstance(payload, np.ndarray):
                    return torch.from_numpy(payload).float().flatten()
                if isinstance(payload, dict):
                    for key in ("embedding", "speaker_embedding", "target_speaker"):
                        value = payload.get(key)
                        if isinstance(value, torch.Tensor):
                            return value.float().flatten()
                        if isinstance(value, np.ndarray):
                            return torch.from_numpy(value).float().flatten()
                return None
        except Exception:
            return None

        return None

    def _proxy_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        mel = librosa.feature.melspectrogram(
            y=audio_np,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
        )
        mel_db = librosa.power_to_db(mel + 1e-6, ref=np.max)
        embedding = np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)]).astype(np.float32)
        norm = np.linalg.norm(embedding) + 1e-6
        return torch.from_numpy(embedding / norm)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.samples)
