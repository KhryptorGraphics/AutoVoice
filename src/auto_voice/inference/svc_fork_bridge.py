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

# Kept identical to svc_fork_trainer._ALLOC_CONF on purpose: training and
# inference share one GPU with no guard between them, so they must not run it
# under conflicting allocator policies.
_ALLOC_CONF = "max_split_size_mb:512"

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
    # Cache hits only: a miss may be transient (registry/model files can
    # appear after startup), and a cached None would demand a server restart.
    if entry is not None:
        _CACHE[key] = entry
    return entry


def is_available(profile_id: str, data_dir: str = "data") -> bool:
    """True iff ``profile_id`` has a usable fork model registered."""
    return get_fork_model(profile_id, data_dir) is not None


def _clean_env(entry: Optional[dict] = None) -> Dict[str, str]:
    """Env for the fork subprocess: drop the serving env's PYTHONPATH so the
    fork imports only its own packages, and pin PYTHONNOUSERSITE.

    ``requires_uv_contract`` opts a single registry entry into the
    SVCFORK_UV_CONTRACT patch (site-packages, see
    patches/svcfork_uv_contract.patch) that masks the decoder's f0 input by
    the real voiced/unvoiced flag. Per-model, not global: a checkpoint
    trained without the fix (unmasked f0 throughout) would face a fresh
    train/serve mismatch if served WITH it on, and a checkpoint trained WITH
    it needs it on to match. Unset/false reproduces the patch's own default
    (a no-op) exactly.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    # Pin the allocator, matching svc_fork_trainer._clean_env. The systemd unit
    # exports PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which infer
    # subprocesses were inheriting: expandable_segments never returns the
    # segments it grows, and on Jetson GPU memory IS system RAM, so a long
    # serving session drifts the whole box toward starvation. The trainer
    # already overrode this; the inference side was left inheriting it, so the
    # two subprocesses ran the same GPU under opposite policies - and only the
    # one that never gives memory back was the one running continuously.
    env["PYTORCH_CUDA_ALLOC_CONF"] = _ALLOC_CONF
    if entry and entry.get("requires_uv_contract"):
        env["SVCFORK_UV_CONTRACT"] = "1"
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
            "-n", str(float(entry.get("noise_scale", 0.4))),  # lower = steadier voice
            "-o", out_wav,
        ]

        # Chunking controls the audible seams. The fork splits input on silence
        # (``db_thresh``) then caps each piece at ``chunk_seconds``, converting
        # every piece independently - at the fork's 0.5s default that is a seam
        # roughly twice a second, which reads as a "jumpy" voice on sustained
        # notes. ``pad_seconds`` gives each piece context that is trimmed after.
        #
        # Left unset these keys reproduce the fork's own defaults exactly, so
        # the realtime path (which reaches this same bridge via
        # ModelManager.infer) keeps its low-latency behaviour. Offline profiles
        # opt in per-model by setting them in the fork registry entry.
        for flag, key, cast in (
            ("-ch", "chunk_seconds", float),
            ("-mc", "max_chunk_seconds", float),
            ("-p", "pad_seconds", float),
            ("-db", "db_thresh", int),
        ):
            value = entry.get(key)
            if value is not None:
                cmd += [flag, str(cast(value))]
        if entry.get("absolute_thresh") is not None:
            cmd.append("-ab" if entry["absolute_thresh"] else "-nab")

        # Cluster model pulls out-of-distribution content vectors toward the
        # training speaker's distribution before the flow inverts them - a
        # treatment for the VITS prior/posterior gap (the flow only ever saw
        # this speaker's own content at training time, so it generalizes
        # poorly to a different singer's content vectors). Unset reproduces
        # the fork's own default (no cluster blending) exactly.
        cluster_path = entry.get("cluster_model_path")
        if cluster_path is not None:
            cmd += ["-k", str(cluster_path),
                    "-r", str(float(entry.get("cluster_infer_ratio", 0.0)))]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_INFER_TIMEOUT_S, env=_clean_env(entry),
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
