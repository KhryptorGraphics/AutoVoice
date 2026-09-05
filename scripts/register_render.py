"""Register a rendered wav as a GUI conversion record so it appears in History.

The GUI's History tab lists records from
<DATA_DIR>/app_state/conversion_history.json (via
AppStateStore.list_conversion_history); downloads resolve by convention from
<DATA_DIR>/conversions/<job_id>/mix.wav (JobManager.get_job_asset_path). A
render produced outside the job pipeline needs both halves, which this does.

Idempotent on --key (a stable identifier, not the display title): the job id is
uuid5(key), so re-running replaces rather than duplicates.

  python scripts/register_render.py --wav out.wav \
      --title "hero20 - seed v3 ep135" --note "..." --key seed/hero20 \
      [--profile <id>] [--tag brandy] [--data-dir data]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

NS = uuid.UUID("6f6c1e2a-1f7a-4a1e-9c3d-5b8e7a2d4c11")
DEFAULT_PROFILE = "fb17af66-8415-4ffe-81b3-600efe75b6d7"


def build(job_id: str, title: str, note: str, profile: str,
          tags: list[str], duration: float, created: float) -> dict:
    url = f"/api/v1/convert/download/{job_id}"
    return {
        "id": job_id, "status": "completed",
        "created_at": created, "started_at": created,
        "completed_at": created, "timestamp": created,
        "input_file": title, "originalFileName": title,
        "profile_id": profile, "targetVoice": profile,
        "preset": "studio", "quality": "studio",
        "pipeline_type": "quality", "requested_pipeline": "quality",
        "resolved_pipeline": "quality", "runtime_backend": "pytorch",
        "adapter_type": None, "active_model_type": "svc_fork",
        "conversion_metadata": {"note": note, "registered_by":
                                "scripts/register_render.py"},
        "original_audio_asset_id": None, "original_audio_url": None,
        "duration": duration, "audio_duration_seconds": duration,
        "processing_time_seconds": None, "rtf": None, "error": None,
        "output_url": url, "download_url": url, "resultUrl": url,
        "stem_urls": None, "reassemble_url": None,
        "notes": note, "isFavorite": False, "tags": tags,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--key", required=True,
                    help="stable idempotency key, e.g. 'seed/hero20'")
    ap.add_argument("--note", default="")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--data-dir",
                    default=os.environ.get("DATA_DIR", "data"))
    a = ap.parse_args()

    src = Path(a.wav).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"no such wav: {src}")
    data = Path(a.data_dir).expanduser().resolve()
    hist_path = data / "app_state/conversion_history.json"
    if not hist_path.is_file():
        raise SystemExit(f"no conversion history at {hist_path}")

    try:
        import soundfile as sf
        duration = float(sf.info(str(src)).duration)
    except Exception:
        duration = 0.0

    job_id = str(uuid.uuid5(NS, a.key))
    shutil.copy2(hist_path, hist_path.with_suffix(".json.bak"))
    hist = json.loads(hist_path.read_text())
    dest = data / "conversions" / job_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / "mix.wav")
    hist[job_id] = build(job_id, a.title, a.note, a.profile,
                         a.tag, duration, time.time())
    tmp = hist_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(hist, indent=2, sort_keys=True))
    tmp.replace(hist_path)
    print(f"{job_id}  {a.title}  ({duration:.1f}s)")


if __name__ == "__main__":
    main()
