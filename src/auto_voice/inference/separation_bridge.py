"""Bridge to the UVR-family separation tools (isolated ``uvr`` conda env).

Lead-vs-backing vocal separation via Mel-Band RoFormer "karaoke" models
(purpose-trained to split the lead singer from backing-vocal harmonies —
the practical SOTA for the MedleyVox "main vs. rest" task), plus polyphonic
note tracking via Spotify basic-pitch for decomposing harmony stacks into
per-voice lines.

Same isolation pattern as ``svc_fork_bridge``: the tools live in a separate
conda env (their dependency set must not disturb the pinned serving env) and
are invoked via subprocess. Every caller treats a bridge failure as
"unavailable" and falls back to the existing behavior.

Model checkpoints auto-download on first use into
``<data_dir>/uvr_models`` (community UVR checkpoints are typically
non-commercial — same licensing posture as the so-vits-svc-fork engine).
"""
import csv
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

MODELS_DIRNAME = "uvr_models"
DEFAULT_SEPARATOR_BIN = os.environ.get(
    "AUTOVOICE_UVR_BIN", "/home/kp/anaconda3/envs/uvr/bin/audio-separator"
)
# basic-pitch lives in its own py3.10 env: on Linux/py>=3.11 it hard-requires
# tensorflow<2.15.1 (unavailable on Jetson aarch64); py3.10 uses tflite-runtime.
DEFAULT_BASIC_PITCH_BIN = os.environ.get(
    "AUTOVOICE_BASIC_PITCH_BIN", "/home/kp/anaconda3/envs/basicpitch/bin/basic-pitch"
)
# Default karaoke model; Stage-1 calibration may override via env/config.
DEFAULT_KARAOKE_MODEL = os.environ.get(
    "AUTOVOICE_UVR_KARAOKE_MODEL",
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
)
_SEPARATE_TIMEOUT_S = int(os.environ.get("AUTOVOICE_UVR_TIMEOUT", "900"))


def _clean_env() -> Dict[str, str]:
    """Subprocess env without the serving env's Python leakage."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE"):
        env.pop(key, None)
    return env


def is_available() -> bool:
    """Whether the uvr env's tools are installed on this box."""
    return os.path.exists(DEFAULT_SEPARATOR_BIN)


def separate_lead_backing(
    audio: np.ndarray,
    sr: int,
    data_dir: str = "data",
    model: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a vocal stem into (lead, backing) via a karaoke separation model.

    Unlike diarization-span masking, this splits SIMULTANEOUS voices: harmony
    stacked on top of the lead ends up in the backing stem instead of
    contaminating the lead.

    Args:
        audio: Mono vocal stem (float32).
        sr: Sample rate of ``audio``.
        data_dir: Runtime data dir; model checkpoints live in
            ``<data_dir>/uvr_models`` (auto-downloaded on first use).
        model: Separator model filename (default: the calibrated karaoke
            roformer).

    Returns:
        ``(lead, backing)`` float32 mono arrays, length-matched to ``audio``.

    Raises:
        RuntimeError: if the bridge is unavailable or separation fails.
    """
    if not is_available():
        raise RuntimeError(f"audio-separator not found at {DEFAULT_SEPARATOR_BIN}")

    model = model or DEFAULT_KARAOKE_MODEL
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise RuntimeError("Empty audio passed to separate_lead_backing")

    model_dir = Path(data_dir) / MODELS_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        sf.write(in_wav, audio, sr)
        cmd = [
            DEFAULT_SEPARATOR_BIN, in_wav,
            "--model_filename", model,
            "--output_dir", td,
            "--model_file_dir", str(model_dir),
            "--output_format", "WAV",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_SEPARATE_TIMEOUT_S, env=_clean_env(),
        )
        outputs = sorted(Path(td).glob("in_*.wav"))
        if proc.returncode != 0 or not outputs:
            raise RuntimeError(
                f"audio-separator failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '')[-600:]}"
            )

        # Karaoke models emit the lead as "(Vocals)" and everything else —
        # which for a vocal-stem input is the backing vocals — as
        # "(Instrumental)".
        lead_path = backing_path = None
        for p in outputs:
            name = p.name.lower()
            if "vocals" in name and "instrumental" not in name:
                lead_path = p
            elif "instrumental" in name:
                backing_path = p
        if lead_path is None or backing_path is None:
            raise RuntimeError(
                f"Unexpected separator outputs: {[p.name for p in outputs]}")

        lead = _read_mono_at(lead_path, sr)
        backing = _read_mono_at(backing_path, sr)

    n = len(audio)
    return _fit_length(lead, n), _fit_length(backing, n)


def polyphonic_notes(
    audio: np.ndarray,
    sr: int,
) -> List[Dict[str, float]]:
    """Polyphonic note events for a harmony stack, via basic-pitch.

    Returns a list of ``{'start': s, 'end': s, 'pitch_midi': float,
    'amplitude': 0..1}`` sorted by start time. Raises RuntimeError when
    basic-pitch is unavailable or fails (callers fall back to the iterative
    pyin decomposer).
    """
    if not os.path.exists(DEFAULT_BASIC_PITCH_BIN):
        raise RuntimeError(f"basic-pitch not found at {DEFAULT_BASIC_PITCH_BIN}")

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "stack.wav")
        sf.write(in_wav, audio, sr)
        # Force the ONNX serialization: the default tflite model fails to
        # load with the aarch64 tflite-runtime wheel on this box.
        proc = subprocess.run(
            [DEFAULT_BASIC_PITCH_BIN, td, in_wav, "--save-note-events",
             "--model-serialization", "onnx"],
            capture_output=True, text=True,
            timeout=_SEPARATE_TIMEOUT_S, env=_clean_env(),
        )
        csv_files = list(Path(td).glob("*_basic_pitch.csv"))
        if proc.returncode != 0 or not csv_files:
            raise RuntimeError(
                f"basic-pitch failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '')[-600:]}"
            )
        notes: List[Dict[str, float]] = []
        with open(csv_files[0], newline="") as f:
            # Header: start_time_s,end_time_s,pitch_midi,velocity,pitch_bend
            # (pitch_bend spills unquoted commas; DictReader shunts the excess
            # into the None key, which we ignore).
            for row in csv.DictReader(f):
                notes.append({
                    "start": float(row["start_time_s"]),
                    "end": float(row["end_time_s"]),
                    "pitch_midi": float(row["pitch_midi"]),
                    "amplitude": float(row.get("velocity") or 127.0) / 127.0,
                })
    notes.sort(key=lambda n: (n["start"], n["pitch_midi"]))
    return notes


def _read_mono_at(path: Path, sr: int) -> np.ndarray:
    data, out_sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if out_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=out_sr, target_sr=sr)
    return np.asarray(data, dtype=np.float32)


def _fit_length(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) >= n:
        return a[:n]
    return np.pad(a, (0, n - len(a)))
