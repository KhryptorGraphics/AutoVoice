"""Shared offline helpers for file-based realtime conversion."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def run_offline_realtime_conversion(
    audio_path: str,
    speaker_embedding: np.ndarray,
    *,
    pitch_shift: float = 0.0,
    chunk_duration_seconds: float = 1.0,
    pipeline=None,
) -> Dict[str, Any]:
    """Run the live realtime pipeline over a full audio file offline.

    This keeps the product contract for offline `realtime` requests without
    pretending they are quality-pipeline jobs. Audio is downmixed, resampled
    to the realtime pipeline's expected input rate, processed chunk-by-chunk,
    and reassembled into a single output waveform.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError("librosa is required for offline realtime conversion") from exc

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError("soundfile is required for offline realtime conversion") from exc

    if pipeline is None:
        from ..inference.pipeline_factory import PipelineFactory

        pipeline = PipelineFactory.get_instance().get_pipeline("realtime")

    embedding = np.asarray(speaker_embedding, dtype=np.float32)
    if embedding.ndim > 1:
        embedding = embedding.flatten()
    if embedding.size == 0:
        raise ValueError("Profile missing speaker embedding for realtime conversion")

    audio, sample_rate = sf.read(audio_path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    target_sample_rate = int(getattr(pipeline, "sample_rate", 16000))
    output_sample_rate = int(getattr(pipeline, "output_sample_rate", target_sample_rate))
    if sample_rate != target_sample_rate and audio.size:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        audio = np.asarray(audio, dtype=np.float32)

    chunk_size = max(int(target_sample_rate * float(chunk_duration_seconds)), 1)
    outputs = []

    pipeline.set_speaker_embedding(embedding)
    try:
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if chunk.size == 0:
                continue
            converted = np.asarray(pipeline.process_chunk(chunk), dtype=np.float32)
            if converted.ndim > 1:
                converted = converted.reshape(-1)
            outputs.append(converted)
    finally:
        clear_speaker = getattr(pipeline, "clear_speaker", None)
        if callable(clear_speaker):
            clear_speaker()

    output_audio = (
        np.concatenate(outputs).astype(np.float32, copy=False)
        if outputs
        else np.zeros(0, dtype=np.float32)
    )

    if output_audio.size and abs(float(pitch_shift)) > 1e-6:
        output_audio = librosa.effects.pitch_shift(
            output_audio,
            sr=output_sample_rate,
            n_steps=float(pitch_shift),
        ).astype(np.float32, copy=False)

    metadata: Dict[str, Any] = {"pipeline": "realtime"}
    get_metrics = getattr(pipeline, "get_latency_metrics", None)
    if callable(get_metrics):
        metadata["latency_metrics"] = get_metrics()

    return {
        "mixed_audio": output_audio,
        "sample_rate": output_sample_rate,
        "duration": len(output_audio) / max(output_sample_rate, 1),
        "metadata": metadata,
        "stems": {},
    }
