"""Register the edge-fix A/B renders in the GUI history.

Pairs the user can compare directly for the two reported artifacts:
  - quiet-edge cutoff  -> base (-db -40) vs FIX (-db -60)
  - high-register duck -> same pair, on the highest-register 20s of Hero
"""
import json
import shutil
import time
import uuid
from pathlib import Path

REPO = Path("/home/kp/thordrive/autovoice")
SP = Path("/tmp/claude-2002/-home-kp-thordrive-autovoice/"
          "8456d362-0cc9-4b51-97e6-7964b4a5ebde/scratchpad")
PROFILE = "fb17af66-8415-4ffe-81b3-600efe75b6d7"

ITEMS = [
    ("fix_out/out_fix_base_hero20.wav",
     "EDGE A/B 1of2 — hero20 — BEFORE "
     "(db -40, edges cut: 26 frames zeroed)",
     "Baseline recipe. 26 breath/tail frames became digital silence."),
    ("fix_out/out_fix_db60_hero20.wav",
     "EDGE A/B 2of2 — hero20 — AFTER FIX (db -60, 0 frames zeroed)",
     "db_thresh -60. Zeroed edge frames 26 -> 0; silence bleed -49.8 dBFS."),
    ("fix_out/out_fix_base_clip_high.wav",
     "HIGH-REG A/B 1of2 — Hero 168-188s (highest register) — BEFORE",
     "Highest-register 20s, F0 to 1031 Hz. 21 frames zeroed."),
    ("fix_out/out_fix_db60_clip_high.wav",
     "HIGH-REG A/B 2of2 — Hero 168-188s — AFTER FIX",
     "Edges restored. Residual ~1.3 dB high-register duck is the "
     "training-range gap (corpus has 0% frames above 700 Hz), "
     "not fixable by render flags."),
    ("fix_out/out_fix_gate_hero20.wav",
     "EDGE reference — hero20 — gate fully OFF (db -200, rejected)",
     "Rejected: removes zeroing but bleeds model noise into silences."),
    ("fix_out/out_fix_uv01_hero20.wav",
     "HIGH-REG probe — hero20 — UV threshold 0.1 (no improvement)",
     "Rules out the crepe voicing threshold as the high-register cause: "
     "-4.9 dB duck vs -4.2 dB for gate alone."),
]


def rec(job_id, title, note, created):
    url = f"/api/v1/convert/download/{job_id}"
    return {
        "id": job_id, "status": "completed",
        "created_at": created, "started_at": created,
        "completed_at": created, "timestamp": created,
        "input_file": title, "originalFileName": title,
        "profile_id": PROFILE, "targetVoice": PROFILE,
        "preset": "studio", "quality": "studio",
        "pipeline_type": "quality", "requested_pipeline": "quality",
        "resolved_pipeline": "quality", "runtime_backend": "pytorch",
        "adapter_type": None, "active_model_type": "svc_fork",
        "conversion_metadata": {"origin": "edge-artifact diagnosis 2026-09-05",
                                "note": note},
        "original_audio_asset_id": None, "original_audio_url": None,
        "duration": 20.0, "audio_duration_seconds": 20.0,
        "processing_time_seconds": None, "rtf": None, "error": None,
        "output_url": url, "download_url": url, "resultUrl": url,
        "stem_urls": None, "reassemble_url": None,
        "notes": note, "isFavorite": False,
        "tags": ["edge-fix", "2026-09-05"],
    }


def main():
    hist_path = REPO / "data/app_state/conversion_history.json"
    shutil.copy2(hist_path, hist_path.with_suffix(".json.bak_edgefix"))
    hist = json.loads(hist_path.read_text())
    now = time.time()
    added = 0
    for rel, title, note in ITEMS:
        src = SP / rel
        if not src.exists():
            print("MISSING", src)
            continue
        if any(r.get("input_file") == title for r in hist.values()):
            continue
        job_id = str(uuid.uuid4())
        dest = REPO / "data/conversions" / job_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / "mix.wav")
        hist[job_id] = rec(job_id, title, note, now)
        added += 1
    tmp = hist_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(hist, indent=2, sort_keys=True))
    tmp.replace(hist_path)
    print(f"registered={added} total={len(hist)}")


main()
