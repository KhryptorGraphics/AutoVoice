#!/usr/bin/env python3
"""Voice conversion: Swap vocals between William Singe and Conor Maynard.

This script:
1. Converts William's vocals to sound like Conor on Conor's instrumental
2. Converts Conor's vocals to sound like William on William's instrumental
3. Runs quality metrics and saves outputs for listening test
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import librosa
import soundfile as sf
from auto_voice.storage.paths import (
    resolve_data_dir,
    resolve_profiles_dir,
    resolve_trained_models_dir,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WILLIAM_PROFILE_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
CONOR_PROFILE_ID = "9679a6ec-e6e2-43c4-b64e-1f004fed34f9"


def resolve_runtime_paths(data_dir: str | None = None) -> dict[str, Path]:
    """Resolve runtime paths without assuming the current working directory."""

    resolved_data_dir = resolve_data_dir(data_dir)
    return {
        "data_dir": resolved_data_dir,
        "profiles_dir": resolve_profiles_dir(data_dir=str(resolved_data_dir)),
        "models_dir": resolve_trained_models_dir(data_dir=str(resolved_data_dir)),
        "separated_dir": resolved_data_dir / "separated",
        "output_dir": resolved_data_dir / "conversions",
    }


def print_banner(text: str):
    """Print a prominent banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def load_speaker_embedding(profile_id: str, data_dir: str | None = None) -> np.ndarray:
    """Load speaker embedding from profile."""
    embedding_path = resolve_runtime_paths(data_dir)["profiles_dir"] / f"{profile_id}.npy"
    return np.load(embedding_path)


def simple_voice_conversion(
    source_vocals: np.ndarray,
    source_sr: int,
    target_embedding: np.ndarray,
    shift_semitones: float = 0.0,
) -> np.ndarray:
    """Simple voice conversion using pitch shifting and spectral matching.

    This is a basic conversion for demonstration. Production would use
    the full SOTAConversionPipeline with CoMoSVC.
    """
    # Resample to 16kHz for processing
    if source_sr != 16000:
        source_vocals = librosa.resample(source_vocals, orig_sr=source_sr, target_sr=16000)
        sr = 16000
    else:
        sr = source_sr

    # Apply pitch shift if needed
    if abs(shift_semitones) > 0.1:
        converted = librosa.effects.pitch_shift(
            source_vocals, sr=sr, n_steps=shift_semitones
        )
    else:
        converted = source_vocals

    # Apply subtle timbre transfer via spectral envelope adjustment
    # (Simplified version - full pipeline would use neural vocoder)

    # Get source spectral envelope
    source_mel = librosa.feature.melspectrogram(y=source_vocals, sr=sr, n_mels=128)
    source_env = np.mean(librosa.power_to_db(source_mel), axis=1)

    # Create target envelope from embedding
    target_env = target_embedding[:128] * 10  # Scale embedding

    # Compute envelope ratio
    ratio = np.clip(target_env / (source_env + 1e-6), 0.5, 2.0)

    # Apply to converted audio via simple EQ
    stft = librosa.stft(converted)
    n_bins = stft.shape[0]

    # Interpolate ratio to match STFT bins
    ratio_interp = np.interp(
        np.linspace(0, 127, n_bins),
        np.arange(128),
        ratio
    )

    # Apply gentle spectral shaping
    stft_modified = stft * np.sqrt(ratio_interp[:, np.newaxis])

    converted = librosa.istft(stft_modified, length=len(converted))

    return converted


def mix_vocals_with_instrumental(
    vocals: np.ndarray,
    instrumental: np.ndarray,
    vocals_sr: int,
    inst_sr: int,
    vocal_level: float = 0.8,
    inst_level: float = 1.0,
) -> tuple:
    """Mix converted vocals with instrumental track."""
    # Resample to common rate
    target_sr = max(vocals_sr, inst_sr)

    if vocals_sr != target_sr:
        vocals = librosa.resample(vocals, orig_sr=vocals_sr, target_sr=target_sr)
    if inst_sr != target_sr:
        instrumental = librosa.resample(instrumental, orig_sr=inst_sr, target_sr=target_sr)

    # Match lengths
    min_len = min(len(vocals), len(instrumental))
    vocals = vocals[:min_len]
    instrumental = instrumental[:min_len]

    # Mix
    mixed = vocal_level * vocals + inst_level * instrumental

    # Normalize
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * 0.95 / peak

    return mixed, target_sr


def compute_quality_metrics(
    converted: np.ndarray,
    reference: np.ndarray,
    sr: int,
) -> dict:
    """Compute quality metrics for converted audio."""
    # Ensure same length
    min_len = min(len(converted), len(reference))
    converted = converted[:min_len]
    reference = reference[:min_len]

    # 1. Mel Cepstral Distortion (MCD)
    converted_mfcc = librosa.feature.mfcc(y=converted, sr=sr, n_mfcc=13)
    reference_mfcc = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=13)

    # Align lengths
    min_frames = min(converted_mfcc.shape[1], reference_mfcc.shape[1])
    converted_mfcc = converted_mfcc[:, :min_frames]
    reference_mfcc = reference_mfcc[:, :min_frames]

    mcd = np.mean(np.sqrt(2 * np.sum((converted_mfcc - reference_mfcc) ** 2, axis=0)))

    # 2. F0 RMSE (pitch accuracy)
    f0_converted = librosa.pyin(converted, fmin=50, fmax=800, sr=sr)[0]
    f0_reference = librosa.pyin(reference, fmin=50, fmax=800, sr=sr)[0]

    # Filter valid F0 values
    valid = ~np.isnan(f0_converted) & ~np.isnan(f0_reference)
    if np.sum(valid) > 0:
        f0_rmse = np.sqrt(np.mean((f0_converted[valid] - f0_reference[valid]) ** 2))
    else:
        f0_rmse = float('nan')

    # 3. Speaker similarity (embedding cosine similarity)
    converted_mel = librosa.feature.melspectrogram(y=converted, sr=sr, n_mels=128)
    reference_mel = librosa.feature.melspectrogram(y=reference, sr=sr, n_mels=128)

    conv_embedding = np.concatenate([
        librosa.power_to_db(converted_mel).mean(axis=1),
        librosa.power_to_db(converted_mel).std(axis=1)
    ])
    ref_embedding = np.concatenate([
        librosa.power_to_db(reference_mel).mean(axis=1),
        librosa.power_to_db(reference_mel).std(axis=1)
    ])

    similarity = np.dot(conv_embedding, ref_embedding) / (
        np.linalg.norm(conv_embedding) * np.linalg.norm(ref_embedding)
    )

    return {
        'mcd': float(mcd),
        'f0_rmse': float(f0_rmse) if not np.isnan(f0_rmse) else None,
        'speaker_similarity': float(similarity),
    }


def run_conversion(
    source_name: str,
    target_name: str,
    source_profile_id: str,
    target_profile_id: str,
    *,
    data_dir: str | None = None,
) -> dict:
    """Run a single voice conversion."""
    print(f"\n  🎤 Converting: {source_name} → {target_name} voice")
    paths = resolve_runtime_paths(data_dir)

    # Load source vocals
    source_vocals_path = paths["separated_dir"] / source_profile_id / "vocals.wav"
    source_vocals, source_sr = librosa.load(str(source_vocals_path), sr=None, mono=True)
    print(f"  📊 Source vocals: {len(source_vocals)/source_sr:.1f}s @ {source_sr}Hz")

    # Load target instrumental
    target_inst_path = paths["separated_dir"] / target_profile_id / "instrumental.wav"
    target_inst, inst_sr = librosa.load(str(target_inst_path), sr=None, mono=True)
    print(f"  🎸 Target instrumental: {len(target_inst)/inst_sr:.1f}s @ {inst_sr}Hz")

    # Load target speaker embedding
    target_embedding = load_speaker_embedding(target_profile_id, data_dir=data_dir)
    print(f"  🧬 Target embedding: {target_embedding.shape}")

    # Load target reference vocals (for quality comparison)
    target_vocals_path = paths["separated_dir"] / target_profile_id / "vocals.wav"
    target_vocals, target_sr = librosa.load(str(target_vocals_path), sr=None, mono=True)

    # Estimate pitch shift from embeddings
    source_embedding = load_speaker_embedding(source_profile_id, data_dir=data_dir)
    # Use mean frequency from vocal range stored in embedding
    source_mean_freq = abs(source_embedding[64])  # Middle of embedding
    target_mean_freq = abs(target_embedding[64])

    # Calculate semitone shift (rough approximation)
    if source_mean_freq > 0 and target_mean_freq > 0:
        ratio = target_mean_freq / source_mean_freq
        shift = 12 * np.log2(ratio) if ratio > 0 else 0
        shift = np.clip(shift, -12, 12)  # Limit shift range
    else:
        shift = 0

    print(f"  🎵 Estimated pitch shift: {shift:.1f} semitones")

    # Convert vocals
    print(f"  🔄 Converting vocals...")
    start_time = time.time()

    converted_vocals = simple_voice_conversion(
        source_vocals=source_vocals,
        source_sr=source_sr,
        target_embedding=target_embedding,
        shift_semitones=shift,
    )

    conversion_time = time.time() - start_time
    print(f"  ⏱️  Conversion time: {conversion_time:.2f}s")

    # Mix with instrumental
    print(f"  🎚️  Mixing with instrumental...")
    mixed, output_sr = mix_vocals_with_instrumental(
        vocals=converted_vocals,
        instrumental=target_inst,
        vocals_sr=16000,  # Conversion outputs 16kHz
        inst_sr=inst_sr,
        vocal_level=0.8,
        inst_level=1.0,
    )

    # Save output
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    output_filename = f"{source_name.lower().replace(' ', '_')}_as_{target_name.lower().replace(' ', '_')}.wav"
    output_path = paths["output_dir"] / output_filename
    sf.write(str(output_path), mixed, output_sr)
    print(f"  💾 Saved: {output_path}")

    # Compute quality metrics
    print(f"  📏 Computing quality metrics...")
    metrics = compute_quality_metrics(
        converted=librosa.resample(converted_vocals, orig_sr=16000, target_sr=target_sr),
        reference=target_vocals,
        sr=target_sr,
    )

    print(f"     MCD (lower=better): {metrics['mcd']:.2f}")
    if metrics['f0_rmse']:
        print(f"     F0 RMSE: {metrics['f0_rmse']:.1f} Hz")
    print(f"     Speaker similarity: {metrics['speaker_similarity']:.3f}")

    return {
        'source': source_name,
        'target': target_name,
        'output_path': str(output_path),
        'duration': len(mixed) / output_sr,
        'conversion_time': conversion_time,
        'metrics': metrics,
    }


def main():
    """Main conversion script."""
    print_banner("AutoVoice - Voice Conversion")

    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    print()

    # Change to project root
    os.chdir(PROJECT_ROOT)
    paths = resolve_runtime_paths()

    results = []

    # ========================================================================
    # Conversion 1: William's vocals → Conor's voice on Conor's instrumental
    # ========================================================================
    print_banner("Conversion 1: William → Conor")
    result1 = run_conversion(
        source_name="William Singe",
        target_name="Conor Maynard",
        source_profile_id=WILLIAM_PROFILE_ID,
        target_profile_id=CONOR_PROFILE_ID,
        data_dir=str(paths["data_dir"]),
    )
    results.append(result1)

    # ========================================================================
    # Conversion 2: Conor's vocals → William's voice on William's instrumental
    # ========================================================================
    print_banner("Conversion 2: Conor → William")
    result2 = run_conversion(
        source_name="Conor Maynard",
        target_name="William Singe",
        source_profile_id=CONOR_PROFILE_ID,
        target_profile_id=WILLIAM_PROFILE_ID,
        data_dir=str(paths["data_dir"]),
    )
    results.append(result2)

    # ========================================================================
    # Summary
    # ========================================================================
    print_banner("Conversion Complete - Summary")

    for result in results:
        print(f"  {result['source']} → {result['target']}:")
        print(f"    📁 Output: {result['output_path']}")
        print(f"    ⏱️  Duration: {result['duration']:.1f}s")
        print(f"    📏 MCD: {result['metrics']['mcd']:.2f}")
        print(f"    🎤 Speaker Similarity: {result['metrics']['speaker_similarity']:.3f}")
        print()

    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📂 Output files are in: {paths['output_dir']}/")
    print("🎧 Please listen and provide feedback on quality!")
    print()

    # Quality assessment
    print_banner("Quality Assessment")

    avg_mcd = np.mean([r['metrics']['mcd'] for r in results])
    avg_similarity = np.mean([r['metrics']['speaker_similarity'] for r in results])

    print(f"  Average MCD: {avg_mcd:.2f}")
    print(f"    (Lower is better. <6 is good, <4 is excellent)")
    print()
    print(f"  Average Speaker Similarity: {avg_similarity:.3f}")
    print(f"    (Higher is better. >0.8 is good, >0.9 is excellent)")
    print()

    # Quality verdict
    if avg_mcd < 6 and avg_similarity > 0.8:
        print("  ✅ Quality: GOOD - Ready for user listening test")
    elif avg_mcd < 8 and avg_similarity > 0.6:
        print("  ⚠️  Quality: ACCEPTABLE - May benefit from more training")
    else:
        print("  ❌ Quality: NEEDS IMPROVEMENT - Consider adding more training samples")

    return results


if __name__ == "__main__":
    main()
