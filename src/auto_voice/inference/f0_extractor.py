"""F0 extraction with quality-ordered fallback chain: rmvpe -> torchcrepe -> pyin.

Why this module exists
----------------------
The in-repo ``RMVPEPitchExtractor`` (198 keys, ``model.*`` layout) shares ZERO
state-dict keys with the actual checkpoint at ``models/pretrained/rmvpe.pt``
(741 keys, ``unet.*`` / E2E layout from the RVC ecosystem). Its
``strict=False`` load silently matched nothing, so it always ran on random
weights. The vendored copy of the original implementation at
``models/seed-vc/modules/rmvpe.py`` *does* match that checkpoint exactly
(``E2E(4, 1, (2, 2))``, strict load, 16 kHz mel, hop 160 -> 100 fps frames),
so we import that file directly and wrap it here.

All extractors return f0 on the librosa frame grid of the *caller's*
``(sr, hop_length)`` — ``1 + len(audio) // hop_length`` frames, unvoiced
frames hard-zeroed — so callers are agnostic to which backend served the
request. Unvoiced zeros are preserved via nearest-neighbor sampling from the
backend's native 100 fps grid (linear interpolation would smear 0 Hz into
audible glides at voicing boundaries).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_RMVPE_SOURCE_RELPATH = Path("models") / "seed-vc" / "modules" / "rmvpe.py"
_RMVPE_CKPT_RELPATH = Path("models") / "pretrained" / "rmvpe.pt"
_RMVPE_SR = 16000
_RMVPE_HOP = 160  # 10 ms frames -> 100 fps

# Module-level singleton: the E2E net is ~180 MB and load takes seconds.
_rmvpe_instance = None
_rmvpe_failed = False


def _repo_root() -> Path:
    # src/auto_voice/inference/f0_extractor.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _resolve(rel: Path, override: Optional[str]) -> Optional[Path]:
    if override:
        p = Path(override)
        return p if p.exists() else None
    for base in (_repo_root(), Path.cwd()):
        p = base / rel
        if p.exists():
            return p
    return None


def _load_rmvpe(device, model_path: Optional[str] = None, is_half: bool = False):
    """Import the vendored seed-vc RMVPE implementation and load the checkpoint."""
    global _rmvpe_instance, _rmvpe_failed
    if _rmvpe_instance is not None:
        return _rmvpe_instance
    if _rmvpe_failed:
        return None

    src = _resolve(_RMVPE_SOURCE_RELPATH, None)
    ckpt = _resolve(_RMVPE_CKPT_RELPATH, model_path)
    if src is None or ckpt is None:
        logger.warning(
            "RMVPE unavailable (source=%s, checkpoint=%s); falling back", src, ckpt
        )
        _rmvpe_failed = True
        return None

    try:
        spec = importlib.util.spec_from_file_location("_vendored_seedvc_rmvpe", src)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_vendored_seedvc_rmvpe"] = mod
        spec.loader.exec_module(mod)
        # Vendored class does a STRICT load_state_dict, so an architecture
        # mismatch raises here instead of silently no-op'ing.
        _rmvpe_instance = mod.RMVPE(str(ckpt), is_half=is_half, device=device)
        logger.info("RMVPE loaded from %s (device=%s, half=%s)", ckpt, device, is_half)
        return _rmvpe_instance
    except Exception:
        logger.exception("RMVPE load failed; falling back to torchcrepe/pyin")
        _rmvpe_failed = True
        return None


def _to_frame_grid(
    f0_10ms: np.ndarray, n_samples: int, sr: int, hop_length: int
) -> np.ndarray:
    """Nearest-neighbor sample a 100 fps f0 track onto the librosa frame grid.

    Nearest (not linear) so hard 0 Hz unvoiced frames stay 0 instead of
    becoming pitch glides at voicing boundaries.
    """
    n_frames = 1 + n_samples // hop_length
    times = np.arange(n_frames) * (hop_length / sr)
    idx = np.clip(np.round(times * (_RMVPE_SR / _RMVPE_HOP)).astype(int), 0,
                  max(len(f0_10ms) - 1, 0))
    if len(f0_10ms) == 0:
        return np.zeros(n_frames, dtype=np.float32)
    return f0_10ms[idx].astype(np.float32)


def _extract_rmvpe(audio, sr, hop_length, device, model_path, is_half):
    rmvpe = _load_rmvpe(device, model_path, is_half)
    if rmvpe is None:
        return None
    import librosa

    audio16 = audio if sr == _RMVPE_SR else librosa.resample(
        audio.astype(np.float32), orig_sr=sr, target_sr=_RMVPE_SR
    )
    f0 = rmvpe.infer_from_audio(audio16, thred=0.03)  # 100 fps, unvoiced -> 0
    return _to_frame_grid(np.asarray(f0, dtype=np.float32), len(audio), sr, hop_length)


def _extract_torchcrepe(audio, sr, hop_length, device):
    try:
        import torch
        import torchcrepe
        import librosa
    except ImportError:
        return None
    try:
        audio16 = audio if sr == _RMVPE_SR else librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=_RMVPE_SR
        )
        x = torch.from_numpy(audio16.astype(np.float32)).unsqueeze(0)
        f0, periodicity = torchcrepe.predict(
            x, _RMVPE_SR, hop_length=_RMVPE_HOP, fmin=50.0, fmax=1100.0,
            model="full", batch_size=512, device=str(device),
            return_periodicity=True,
        )
        periodicity = torchcrepe.filter.median(periodicity, 3)
        f0 = f0.squeeze(0).cpu().numpy()
        voiced = periodicity.squeeze(0).cpu().numpy() > 0.21
        f0 = np.where(voiced, f0, 0.0).astype(np.float32)
        return _to_frame_grid(f0, len(audio), sr, hop_length)
    except Exception:
        logger.exception("torchcrepe F0 extraction failed; falling back to pyin")
        return None


def _extract_pyin(audio, sr, hop_length):
    import librosa

    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=sr, hop_length=hop_length)
    return np.nan_to_num(f0, nan=0.0).astype(np.float32)


def extract_f0(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    method: str = "rmvpe",
    device="cpu",
    rmvpe_model_path: Optional[str] = None,
    rmvpe_is_half: bool = False,
) -> Tuple[np.ndarray, str]:
    """Extract f0 in Hz on the librosa frame grid for (sr, hop_length).

    Returns ``(f0, method_used)`` where f0 has ``1 + len(audio)//hop_length``
    frames and unvoiced frames are exactly 0. ``method`` picks the preferred
    backend; anything above it in the chain is skipped, anything below serves
    as automatic fallback (rmvpe -> torchcrepe -> pyin).
    """
    order = ["rmvpe", "torchcrepe", "pyin"]
    start = order.index(method) if method in order else 0
    for name in order[start:]:
        if name == "rmvpe":
            f0 = _extract_rmvpe(audio, sr, hop_length, device,
                                rmvpe_model_path, rmvpe_is_half)
        elif name == "torchcrepe":
            f0 = _extract_torchcrepe(audio, sr, hop_length, device)
        else:
            f0 = _extract_pyin(audio, sr, hop_length)
        if f0 is not None:
            if name != method:
                logger.warning("F0 method '%s' unavailable; used '%s'", method, name)
            return f0, name
    # Unreachable (pyin always returns), but keep a hard floor anyway.
    return np.zeros(1 + len(audio) // hop_length, dtype=np.float32), "zeros"
