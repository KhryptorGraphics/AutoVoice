"""Bridge to the so-vits-svc-fork SVC engine (isolated ``svcfork`` conda env).

The in-repo from-scratch decoder cannot render F0 into voiced harmonics on the
available per-speaker data (measured: 79-86%-voiced input -> 17-23%-voiced
output, ~0 melody correlation). A pretrained so-vits-svc-fork base fine-tuned on
the same speaker fixes this (melody correlation ~0.98, 82-90% voiced) because it
carries a strong acoustic prior plus explicit NSF F0 conditioning.

The fork lives in a separate conda env (its dependency set conflicts with the
serving env). We invoke it via subprocess so the two environments stay isolated.

Registry: a profile is served by the fork iff ``<data_dir>/fork_models/<id>.json``
exists and points at a real model + config. Every other profile falls through to
the in-repo decoder unchanged, so this is additive and opt-in per profile.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf

REGISTRY_DIRNAME = "fork_models"
DEFAULT_SVC_BIN = os.environ.get(
    "AUTOVOICE_SVCFORK_BIN", "/home/kp/anaconda3/envs/svcfork/bin/svc"
)
_INFER_TIMEOUT_S = int(os.environ.get("AUTOVOICE_SVCFORK_TIMEOUT", "900"))

# Parsed-entry cache keyed by (data_dir, profile_id) -> entry|None. Cleared by
# tests via clear_cache(); production entries are static per server run.
_CACHE: Dict[Tuple[str, str], Optional[dict]] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _registry_path(profile_id: str, data_dir: str) -> Path:
    return Path(data_dir) / REGISTRY_DIRNAME / f"{profile_id}.json"


def get_fork_model(profile_id: str, data_dir: str = "data") -> Optional[dict]:
    """Return a validated fork-model registry entry for ``profile_id``, else None.

    Validates that the referenced model and config actually exist on disk so a
    stale/partial registry entry silently falls back to the in-repo decoder
    rather than failing a conversion.
    """
    key = (str(data_dir), str(profile_id))
    if key in _CACHE:
        return _CACHE[key]
    entry: Optional[dict] = None
    f = _registry_path(profile_id, data_dir)
    if f.exists():
        try:
            candidate = json.loads(f.read_text())
        except (ValueError, OSError):
            candidate = None
        if (
            isinstance(candidate, dict)
            and candidate.get("model_path")
            and candidate.get("config_path")
            and candidate.get("speaker")
            and os.path.exists(candidate["model_path"])
            and os.path.exists(candidate["config_path"])
        ):
            entry = candidate
    _CACHE[key] = entry
    return entry


def is_available(profile_id: str, data_dir: str = "data") -> bool:
    """True iff ``profile_id`` has a usable fork model registered."""
    return get_fork_model(profile_id, data_dir) is not None


def _clean_env() -> Dict[str, str]:
    """Env for the fork subprocess: drop the serving env's PYTHONPATH so the
    fork imports only its own packages, and pin PYTHONNOUSERSITE."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def convert(audio: np.ndarray, sr: int, profile_id: str,
            data_dir: str = "data") -> np.ndarray:
    """Convert source vocals to the target voice via the fork engine.

    Mirrors ``ModelManager.infer``'s contract: returns float32 audio at the input
    sample rate ``sr``. The fork extracts its own F0 (crepe) and content, so pass
    the separated source vocals exactly as the in-repo path would receive them.

    Raises:
        RuntimeError: if no fork model is registered or the fork invocation fails.
    """
    entry = get_fork_model(profile_id, data_dir)
    if entry is None:
        raise RuntimeError(f"No fork model registered for profile '{profile_id}'")

    svc_bin = entry.get("svc_bin", DEFAULT_SVC_BIN)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio

    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        out_wav = os.path.join(td, "out.wav")
        sf.write(in_wav, audio, sr)
        cmd = [
            svc_bin, "infer", in_wav,
            "-s", str(entry["speaker"]),
            "-c", str(entry["config_path"]),
            "-m", str(entry["model_path"]),
            "-fm", str(entry.get("f0_method", "crepe")),
            "-na",  # preserve the source melody (no auto-predict F0)
            "-t", str(int(entry.get("transpose", 0))),
            "-o", out_wav,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_INFER_TIMEOUT_S, env=_clean_env(),
        )
        if not os.path.exists(out_wav):
            # some fork versions append a suffix; take newest wav in the dir
            wavs = sorted(Path(td).glob("*.wav"), key=os.path.getmtime)
            wavs = [w for w in wavs if w.name != "in.wav"]
            if wavs:
                out_wav = str(wavs[-1])
        if not os.path.exists(out_wav):
            raise RuntimeError(
                f"fork infer produced no output (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '')[-600:]}"
            )
        out, out_sr = sf.read(out_wav, dtype="float32")

    if out.ndim > 1:
        out = out.mean(axis=1)
    if out_sr != sr:
        import librosa
        out = librosa.resample(out, orig_sr=out_sr, target_sr=sr)
    return np.asarray(out, dtype=np.float32)
