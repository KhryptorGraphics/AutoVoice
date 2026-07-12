#!/usr/bin/env python3
"""Ingest a local video/audio file into a trainable voice profile.

Extracts the audio track (via ffmpeg for video containers), separates
vocals from any instrumental/background with a *pretrained* Demucs model
(``audio.separation.VocalSeparator`` -- HTDemucs, real weights, built-in
segmented/overlap-add inference so arbitrary-length input is memory-safe),
creates or reuses a voice profile, and attaches the separated vocals as a
real training sample (run through the same QA analysis the web API uses,
not a stubbed "unknown" status).

Deliberately does NOT use ``web.karaoke_manager.KaraokeManager`` /
``audio.separator.MelBandRoFormer`` for this: that separator is
constructed with no ``pretrained`` path (random weights) and has no
built-in chunking, so it is unsuitable for real dataset ingestion.

Usage:
    python scripts/ingest_local_audio_profile.py \\
        --source /home/kp/Videos/brandy.mp4 \\
        --name Brandy
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import soundfile as sf
import torch

from auto_voice.storage.voice_profiles import PROFILE_ROLE_TARGET_USER, VoiceProfileStore
from auto_voice.training.sample_quality import MIN_TRAINING_SAMPLE_SECONDS

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def extract_audio(source: Path, workdir: Path) -> Path:
    """Extract/normalize the source to a WAV file ffmpeg + soundfile can both read.

    Video containers (mp4, mkv, ...) and compressed audio (mp3, m4a, aac ...)
    go through ffmpeg. Already-WAV/FLAC/OGG/AIFF sources pass through untouched.
    """
    if source.suffix.lower() in AUDIO_SUFFIXES:
        return source

    out_path = workdir / f"{source.stem}_extracted.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"ffmpeg audio extraction failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '')[-800:]}"
        )
    return out_path


def separate_vocals(audio_path: Path, device: torch.device,
                    segment: Optional[float] = None,
                    model_name: str = "htdemucs_ft"):
    """Run the source WAV through the pretrained Demucs separator.

    Returns (vocals, instrumental, sr, duration_seconds); vocals/instrumental
    are mono float32 numpy arrays at the source sample rate.
    """
    from auto_voice.audio.separation import VocalSeparator

    data, sr = sf.read(str(audio_path), dtype="float32")
    mono = data if data.ndim == 1 else data.mean(axis=1)
    duration_seconds = len(mono) / sr

    print(f"Separating {duration_seconds:.1f}s of audio on {device} "
          f"(model={model_name}, segment={segment}, chunked internally)...")
    separator = VocalSeparator(device=device, model_name=model_name, segment=segment)
    result = separator.separate(mono, sr, mono=True)
    return result["vocals"], result["instrumental"], sr, duration_seconds


def ensure_profile(store: VoiceProfileStore, name: str, profile_id: Optional[str]) -> str:
    """Create the profile if it doesn't already exist; return its id."""
    if profile_id and store.exists(profile_id):
        print(f"Reusing existing profile {profile_id} ({name})")
        return profile_id

    payload = {
        "name": name,
        "user_id": "operator",
        "created_from": "local_video_ingest",
        "profile_role": PROFILE_ROLE_TARGET_USER,
    }
    if profile_id:
        payload["profile_id"] = profile_id
    saved_id = store.save(payload)
    print(f"Created profile {saved_id} ({name})")
    return saved_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local video/audio file")
    parser.add_argument("--name", required=True, help="Voice profile name")
    parser.add_argument("--profile-id", default=None, help="Reuse an existing profile id")
    parser.add_argument("--segment", type=float, default=None,
                        help="Demucs segment seconds (default: model native ~7.8s; "
                             "values >7.8 break htdemucs)")
    parser.add_argument("--model", default="htdemucs_ft",
                        help="Demucs model (default htdemucs_ft, the fork HQ SOTA model)")
    parser.add_argument("--data-dir", default="data", help="AutoVoice DATA_DIR")
    parser.add_argument("--slice-seconds", type=float, default=None,
                        help="If set, slice separated vocals into clips of this "
                             "length and add each as a separate training sample "
                             "(gives len(dataset)>1 so batching/multi-worker and "
                             "warmup behave correctly). Default: one whole sample.")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir).resolve()
    store = VoiceProfileStore(
        profiles_dir=str(data_dir / "voice_profiles"),
        samples_dir=str(data_dir / "samples"),
        trained_models_dir=str(data_dir / "trained_models"),
    )

    with tempfile.TemporaryDirectory(prefix="ingest_") as tmp:
        workdir = Path(tmp)
        wav_path = extract_audio(source, workdir)
        vocals, instrumental, sr, duration_seconds = separate_vocals(
            wav_path, device, segment=args.segment, model_name=args.model
        )

        profile_id = ensure_profile(store, args.name, args.profile_id)

        if args.slice_seconds and args.slice_seconds > 0:
            clip_len = int(args.slice_seconds * sr)
            min_len = int(MIN_TRAINING_SAMPLE_SECONDS * sr)
            clips = []
            idx = 0
            for start in range(0, len(vocals), clip_len):
                seg_v = vocals[start:start + clip_len]
                if len(seg_v) < min_len:
                    continue  # drop the sub-minimum tail
                idx += 1
                seg_i = instrumental[start:start + clip_len]
                clips.append((seg_v, seg_i, f"{source.stem}_clip{idx:03d}", len(seg_v) / sr))
            print(f"Sliced into {len(clips)} clips of ~{args.slice_seconds}s")
        else:
            clips = [(vocals, instrumental, source.name, duration_seconds)]

        samples = []
        for seg_v, seg_i, label, dur in clips:
            vpath = workdir / f"{label}_vocals.wav"
            ipath = workdir / f"{label}_instrumental.wav"
            sf.write(str(vpath), seg_v, sr)
            sf.write(str(ipath), seg_i, sr)
            samples.append(store.add_training_sample(
                profile_id=profile_id,
                vocals_path=str(vpath),
                instrumental_path=str(ipath),
                source_file=label,
                duration=dur,
                extra_metadata={"provenance": f"local_video_ingest:{source}"},
            ))

    passed = sum(1 for s in samples if (s.quality_metadata or {}).get("qa_status") == "pass")
    print("---")
    print(f"profile_id={profile_id}")
    print(f"samples_added={len(samples)}")
    print(f"qa_pass={passed}/{len(samples)}")
    for s in samples[:5]:
        qa = s.quality_metadata or {}
        print(f"  {s.sample_id} dur={s.duration:.1f}s qa={qa.get('qa_status')} "
              f"issues={qa.get('issues')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
