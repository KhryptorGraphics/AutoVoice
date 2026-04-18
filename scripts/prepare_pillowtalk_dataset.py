#!/usr/bin/env python3
"""Prepare the canonical Pillowtalk dataset layout and manifest."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aligned_conversion import align_with_dtw
from auto_voice.models.ecapa2_encoder import ECAPA2SpeakerEncoder
from auto_voice.storage.voice_profiles import VoiceProfileStore


@dataclass(frozen=True)
class ArtistSpec:
    key: str
    name: str
    profile_id: str
    raw_audio: Path

    @property
    def separated_vocals(self) -> Path:
        return PROJECT_ROOT / "data" / "separated" / self.profile_id / "vocals.wav"

    @property
    def separated_instrumental(self) -> Path:
        return PROJECT_ROOT / "data" / "separated" / self.profile_id / "instrumental.wav"


ARTISTS: Dict[str, ArtistSpec] = {
    "william_singe": ArtistSpec(
        key="william_singe",
        name="William Singe",
        profile_id="7da05140-1303-40c6-95d9-5b6e2c3624df",
        raw_audio=PROJECT_ROOT / "tests/quality_samples/william_singe_pillowtalk.wav",
    ),
    "conor_maynard": ArtistSpec(
        key="conor_maynard",
        name="Conor Maynard",
        profile_id="9679a6ec-e6e2-43c4-b64e-1f004fed34f9",
        raw_audio=PROJECT_ROOT / "tests/quality_samples/conor_maynard_pillowtalk.wav",
    ),
}


def ensure_relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    relative_source = os.path.relpath(source.resolve(), start=destination.parent.resolve())
    destination.symlink_to(relative_source, target_is_directory=False)


def compute_embedding(audio_path: Path, device: str) -> Dict[str, object]:
    encoder = ECAPA2SpeakerEncoder(device=device)
    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    result = encoder.extract_embedding(audio, sample_rate)
    return {
        "embedding": result.embedding.astype(np.float32),
        "backend": result.backend,
    }


def audio_info(audio_path: Path) -> Dict[str, Any]:
    info = sf.info(str(audio_path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": round(info.frames / info.samplerate, 4) if info.samplerate else 0.0,
        "format": info.format,
        "subtype": info.subtype,
    }


def relative_to_dataset(dataset_root: Path, path: Path) -> str:
    return os.path.relpath(path, start=dataset_root)


def ensure_profile(
    store: VoiceProfileStore,
    artist: ArtistSpec,
    embedding: np.ndarray,
    duration_seconds: float,
) -> Dict[str, Any]:
    profile_payload: Dict[str, Any] = {
        "profile_id": artist.profile_id,
        "name": artist.name,
        "artist_name": artist.name,
        "audio_duration": duration_seconds,
        "vocals_extracted": True,
        "created_from": "prepare_pillowtalk_dataset.py",
        "profile_role": "source_artist",
        "clean_vocal_seconds": duration_seconds,
        "total_training_duration": duration_seconds,
        "embedding": embedding,
    }
    if store.exists(artist.profile_id):
        existing = store.load(artist.profile_id)
        existing.update({k: v for k, v in profile_payload.items() if k != "embedding"})
        existing["embedding"] = embedding
        store.save(existing)
    else:
        store.save(profile_payload)
    return store.load(artist.profile_id)


def ensure_training_sample(
    store: VoiceProfileStore,
    artist: ArtistSpec,
    vocals_path: Path,
    instrumental_path: Path,
    duration_seconds: float,
) -> Dict[str, Any]:
    source_name = f"{artist.key}_pillowtalk"
    existing_samples = store.list_training_samples(artist.profile_id)
    for sample in existing_samples:
        if sample.source_file == source_name:
            return sample.to_dict()

    sample = store.add_training_sample(
        artist.profile_id,
        vocals_path=str(vocals_path),
        instrumental_path=str(instrumental_path),
        source_file=source_name,
        duration=duration_seconds,
    )
    return sample.to_dict()


def write_aligned_pair(
    pair_id: str,
    source_vocals: Path,
    target_vocals: Path,
    target_instrumental: Path,
    destination_dir: Path,
) -> Dict[str, Any]:
    source_audio, source_sr = librosa.load(source_vocals, sr=None, mono=True)
    target_audio, target_sr = librosa.load(target_vocals, sr=None, mono=True)
    aligned_audio, aligned_sr = align_with_dtw(
        source_audio=source_audio,
        source_sr=source_sr,
        target_audio=target_audio,
        target_sr=target_sr,
    )

    destination_dir.mkdir(parents=True, exist_ok=True)
    aligned_source_path = destination_dir / "source.wav"
    sf.write(str(aligned_source_path), aligned_audio, aligned_sr)
    ensure_relative_symlink(target_vocals, destination_dir / "reference.wav")
    ensure_relative_symlink(target_instrumental, destination_dir / "instrumental.wav")

    alignment_metadata = {
        "pair_id": pair_id,
        "method": "librosa_dtw",
        "source_duration_seconds": round(len(source_audio) / source_sr, 4),
        "target_duration_seconds": round(len(target_audio) / target_sr, 4),
        "aligned_duration_seconds": round(len(aligned_audio) / aligned_sr, 4),
        "sample_rate": int(aligned_sr),
    }
    (destination_dir / "alignment.json").write_text(json.dumps(alignment_metadata, indent=2))

    return {
        "source_audio": aligned_source_path,
        "reference_audio": destination_dir / "reference.wav",
        "instrumental_audio": destination_dir / "instrumental.wav",
        "alignment_metadata": destination_dir / "alignment.json",
        "stats": alignment_metadata,
    }


def build_manifest(
    dataset_root: Path,
    embeddings: Dict[str, Dict[str, object]],
    training_samples: Dict[str, Dict[str, Any]],
    alignment_artifacts: Dict[str, Dict[str, Any]],
) -> Dict[str, object]:
    william = ARTISTS["william_singe"]
    conor = ARTISTS["conor_maynard"]

    track_specs = {}
    for artist in ARTISTS.values():
        track_specs[artist.key] = {
            "profile_id": artist.profile_id,
            "display_name": artist.name,
            "raw_audio": relative_to_dataset(dataset_root, dataset_root / "raw" / artist.key / "pillowtalk.wav"),
            "vocals_audio": relative_to_dataset(dataset_root, dataset_root / "vocals" / artist.key / "pillowtalk.wav"),
            "instrumental_audio": relative_to_dataset(
                dataset_root,
                dataset_root / "instrumentals" / artist.key / "pillowtalk.wav",
            ),
            "speaker_embedding": relative_to_dataset(dataset_root, dataset_root / "speakers" / f"{artist.key}.npy"),
            "training_sample_id": training_samples[artist.key]["sample_id"],
            "training_sample_vocals": str(Path(training_samples[artist.key]["vocals_path"]).resolve()),
            "training_sample_instrumental": str(
                Path(training_samples[artist.key]["instrumental_path"]).resolve()
            ),
            "provenance": "tests/quality_samples fixtures + data/separated cache",
            "audio_info": audio_info(artist.separated_vocals),
        }

    samples = [
        {
            "sample_id": "william_to_conor_train",
            "split": "train",
            "source_audio": relative_to_dataset(dataset_root, alignment_artifacts["william_to_conor"]["source_audio"]),
            "reference_audio": relative_to_dataset(
                dataset_root, alignment_artifacts["william_to_conor"]["reference_audio"]
            ),
            "instrumental_audio": relative_to_dataset(
                dataset_root, alignment_artifacts["william_to_conor"]["instrumental_audio"]
            ),
            "target_speaker_embedding": "speakers/conor_maynard.npy",
            "metadata": {
                "source_artist": william.key,
                "target_artist": conor.key,
                "song": "Pillowtalk",
                "provenance": "DTW-aligned separated vocals",
            },
        },
        {
            "sample_id": "conor_to_william_train",
            "split": "train",
            "source_audio": relative_to_dataset(dataset_root, alignment_artifacts["conor_to_william"]["source_audio"]),
            "reference_audio": relative_to_dataset(
                dataset_root, alignment_artifacts["conor_to_william"]["reference_audio"]
            ),
            "instrumental_audio": relative_to_dataset(
                dataset_root, alignment_artifacts["conor_to_william"]["instrumental_audio"]
            ),
            "target_speaker_embedding": "speakers/william_singe.npy",
            "metadata": {
                "source_artist": conor.key,
                "target_artist": william.key,
                "song": "Pillowtalk",
                "provenance": "DTW-aligned separated vocals",
            },
        },
        {
            "sample_id": "william_to_conor_holdout",
            "split": "holdout",
            "source_audio": "holdout/william_to_conor/source.wav",
            "reference_audio": "holdout/william_to_conor/reference.wav",
            "target_speaker_embedding": "speakers/conor_maynard.npy",
            "metadata": {
                "source_artist": william.key,
                "target_artist": conor.key,
                "song": "Pillowtalk",
                "provenance": "tests/quality_samples fixtures",
            },
        },
        {
            "sample_id": "conor_to_william_holdout",
            "split": "holdout",
            "source_audio": "holdout/conor_to_william/source.wav",
            "reference_audio": "holdout/conor_to_william/reference.wav",
            "target_speaker_embedding": "speakers/william_singe.npy",
            "metadata": {
                "source_artist": conor.key,
                "target_artist": william.key,
                "song": "Pillowtalk",
                "provenance": "tests/quality_samples fixtures",
            },
        },
    ]

    return {
        "dataset_name": "pillowtalk",
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": 24000,
        "defaults": {
            "sample_rate": 24000,
            "evaluation_mode": "offline_quality",
        },
        "profiles": {
            artist.key: {
                "profile_id": artist.profile_id,
                "display_name": artist.name,
            }
            for artist in ARTISTS.values()
        },
        "tracks": track_specs,
        "alignments": {
            pair_id: {
                "source_audio": relative_to_dataset(dataset_root, payload["source_audio"]),
                "reference_audio": relative_to_dataset(dataset_root, payload["reference_audio"]),
                "instrumental_audio": relative_to_dataset(dataset_root, payload["instrumental_audio"]),
                "alignment_metadata": relative_to_dataset(dataset_root, payload["alignment_metadata"]),
                "stats": payload["stats"],
            }
            for pair_id, payload in alignment_artifacts.items()
        },
        "samples": samples,
        "speaker_backends": {
            artist: embedding_data["backend"] for artist, embedding_data in embeddings.items()
        },
    }


def prepare_dataset(
    output_dir: Path,
    device: str,
    profiles_dir: Path,
    samples_dir: Path,
) -> Path:
    dataset_root = output_dir.resolve()
    raw_dir = dataset_root / "raw"
    holdout_dir = dataset_root / "holdout"
    aligned_dir = dataset_root / "aligned"
    vocals_dir = dataset_root / "vocals"
    instrumentals_dir = dataset_root / "instrumentals"
    speakers_dir = dataset_root / "speakers"

    for directory in (raw_dir, holdout_dir, aligned_dir, vocals_dir, instrumentals_dir, speakers_dir):
        directory.mkdir(parents=True, exist_ok=True)

    embeddings: Dict[str, Dict[str, object]] = {}
    training_samples: Dict[str, Dict[str, Any]] = {}
    store = VoiceProfileStore(
        profiles_dir=str(profiles_dir),
        samples_dir=str(samples_dir),
    )

    for artist in ARTISTS.values():
        ensure_relative_symlink(artist.raw_audio, raw_dir / artist.key / "pillowtalk.wav")
        ensure_relative_symlink(artist.separated_vocals, vocals_dir / artist.key / "pillowtalk.wav")
        ensure_relative_symlink(
            artist.separated_instrumental,
            instrumentals_dir / artist.key / "pillowtalk.wav",
        )

        embeddings[artist.key] = compute_embedding(artist.raw_audio, device=device)
        np.save(speakers_dir / f"{artist.key}.npy", embeddings[artist.key]["embedding"])

        track_info = audio_info(artist.separated_vocals)
        ensure_profile(
            store=store,
            artist=artist,
            embedding=embeddings[artist.key]["embedding"],
            duration_seconds=track_info["duration_seconds"],
        )
        training_samples[artist.key] = ensure_training_sample(
            store=store,
            artist=artist,
            vocals_path=artist.separated_vocals,
            instrumental_path=artist.separated_instrumental,
            duration_seconds=track_info["duration_seconds"],
        )

    william = ARTISTS["william_singe"]
    conor = ARTISTS["conor_maynard"]

    ensure_relative_symlink(william.raw_audio, holdout_dir / "william_to_conor" / "source.wav")
    ensure_relative_symlink(conor.raw_audio, holdout_dir / "william_to_conor" / "reference.wav")
    ensure_relative_symlink(conor.raw_audio, holdout_dir / "conor_to_william" / "source.wav")
    ensure_relative_symlink(william.raw_audio, holdout_dir / "conor_to_william" / "reference.wav")

    alignment_artifacts = {
        "william_to_conor": write_aligned_pair(
            pair_id="william_to_conor",
            source_vocals=william.separated_vocals,
            target_vocals=conor.separated_vocals,
            target_instrumental=conor.separated_instrumental,
            destination_dir=aligned_dir / "william_to_conor",
        ),
        "conor_to_william": write_aligned_pair(
            pair_id="conor_to_william",
            source_vocals=conor.separated_vocals,
            target_vocals=william.separated_vocals,
            target_instrumental=william.separated_instrumental,
            destination_dir=aligned_dir / "conor_to_william",
        ),
    }

    manifest = build_manifest(
        dataset_root=dataset_root,
        embeddings=embeddings,
        training_samples=training_samples,
        alignment_artifacts=alignment_artifacts,
    )

    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2))
    return metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data/training/pillowtalk"),
        help="Canonical dataset output directory.",
    )
    parser.add_argument(
        "--profiles-dir",
        default=str(PROJECT_ROOT / "data/voice_profiles"),
        help="VoiceProfileStore profiles directory.",
    )
    parser.add_argument(
        "--samples-dir",
        default=str(PROJECT_ROOT / "data/samples"),
        help="VoiceProfileStore samples directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding extraction device for ECAPA2SpeakerEncoder.",
    )
    args = parser.parse_args()

    metadata_path = prepare_dataset(
        output_dir=Path(args.output_dir),
        device=args.device,
        profiles_dir=Path(args.profiles_dir),
        samples_dir=Path(args.samples_dir),
    )
    print(f"Wrote manifest: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
