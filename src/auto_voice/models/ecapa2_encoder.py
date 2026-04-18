"""Optional ECAPA2-style speaker encoder wrapper with portable fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ECAPA2EmbeddingResult:
    embedding: np.ndarray
    backend: str


class ECAPA2SpeakerEncoder:
    """Extract speaker embeddings with SpeechBrain when available."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._classifier = None
        self._backend = None

    @property
    def backend(self) -> str:
        return self._backend or "mel-statistics-fallback"

    def _load_classifier(self):
        if self._classifier is not None:
            return self._classifier

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            self._backend = "mel-statistics-fallback"
            self._classifier = None
            return None

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=None,
            run_opts={"device": self.device},
        )
        self._backend = "speechbrain-ecapa"
        self._classifier = classifier
        return classifier

    def _fallback_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
        )
        mel_db = librosa.power_to_db(mel + 1e-6, ref=np.max)
        base = np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)]).astype(np.float32)
        norm = np.linalg.norm(base) + 1e-6
        return base / norm

    def extract_embedding(self, audio: np.ndarray, sample_rate: int) -> ECAPA2EmbeddingResult:
        audio_np = np.asarray(audio, dtype=np.float32)
        classifier = self._load_classifier()
        if classifier is None:
            embedding = self._fallback_embedding(audio_np, sample_rate)
            return ECAPA2EmbeddingResult(embedding=embedding, backend=self.backend)

        import torch
        import librosa

        if sample_rate != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        waveform = torch.tensor(audio_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embedding = classifier.encode_batch(waveform).squeeze().detach().cpu().numpy()
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding) + 1e-6
        return ECAPA2EmbeddingResult(embedding=embedding / norm, backend=self.backend)
