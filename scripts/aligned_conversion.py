#!/usr/bin/env python3
"""
Aligned Voice Conversion Pipeline

This pipeline:
1. Aligns source vocals to match target timing using DTW
2. Adjusts pitch to match target pitch contour (optional)
3. Converts the voice timbre using Seed-VC
4. Mixes with target instrumental

This ensures the converted vocals match the target's timing perfectly.
"""

import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCCs for DTW alignment."""
    # Resample to 22050 for consistent feature extraction
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sr = 22050

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)

    # Add delta features for better alignment
    mfcc_delta = librosa.feature.delta(mfcc)
    features = np.vstack([mfcc, mfcc_delta])

    return features.T  # [T, features]


def align_with_dtw(
    source_audio: np.ndarray,
    source_sr: int,
    target_audio: np.ndarray,
    target_sr: int,
) -> Tuple[np.ndarray, int]:
    """
    Align source audio to match target timing using DTW.

    Returns:
        Aligned audio at target sample rate
    """
    logger.info("Extracting features for DTW alignment...")

    # Extract features
    source_features = extract_features(source_audio, source_sr)
    target_features = extract_features(target_audio, target_sr)

    logger.info(f"Source features: {source_features.shape}, Target features: {target_features.shape}")

    # Run DTW
    logger.info("Computing DTW alignment...")
    D, wp = librosa.sequence.dtw(
        X=source_features.T,
        Y=target_features.T,
        metric='cosine',
        step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]),
        weights_mul=np.array([1, 1, 1]),
        subseq=False,
        backtrack=True
    )

    # wp is [N, 2] where each row is (source_frame, target_frame)
    # We need to map source to target
    wp = wp[::-1]  # Reverse to get chronological order

    logger.info(f"Alignment path length: {len(wp)}")

    # Resample source to target sample rate for manipulation
    if source_sr != target_sr:
        source_audio = librosa.resample(source_audio, orig_sr=source_sr, target_sr=target_sr)

    # Convert frame indices to sample indices
    hop_length = 512
    target_len = len(target_audio)

    # Create aligned audio by time-stretching/warping
    logger.info("Warping audio to match target timing...")

    # Simple approach: linear interpolation between alignment points
    source_samples = wp[:, 0] * hop_length
    target_samples = wp[:, 1] * hop_length

    # Clamp to valid ranges
    source_samples = np.clip(source_samples, 0, len(source_audio) - 1)
    target_samples = np.clip(target_samples, 0, target_len - 1)

    # Create output by interpolating
    aligned = np.zeros(target_len, dtype=np.float32)

    # Use the warping path to stretch/compress source
    # For each target sample, find corresponding source sample
    target_indices = np.arange(target_len)
    source_indices = np.interp(target_indices, target_samples, source_samples)

    # Resample source audio according to warping
    aligned = np.interp(source_indices, np.arange(len(source_audio)), source_audio)

    logger.info(f"Aligned audio: {len(aligned)/target_sr:.1f}s @ {target_sr}Hz")

    return aligned.astype(np.float32), target_sr


def align_pitch(
    aligned_audio: np.ndarray,
    sr: int,
    target_audio: np.ndarray,
    target_sr: int,
) -> np.ndarray:
    """
    Adjust pitch of aligned audio to match target pitch contour.

    Uses PSOLA-like pitch shifting.
    """
    logger.info("Extracting pitch contours...")

    # Resample if needed
    if sr != target_sr:
        aligned_audio = librosa.resample(aligned_audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Extract pitch using pyin (more robust than yin)
    f0_source, voiced_flag_source, _ = librosa.pyin(
        aligned_audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr
    )

    f0_target, voiced_flag_target, _ = librosa.pyin(
        target_audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr
    )

    # Fill NaN with zeros
    f0_source = np.nan_to_num(f0_source, nan=0.0)
    f0_target = np.nan_to_num(f0_target, nan=0.0)

    # Calculate pitch shift ratio per frame
    # Only shift where both are voiced
    voiced_mask = (f0_source > 0) & (f0_target > 0)

    # Calculate semitone shift
    shift_semitones = np.zeros_like(f0_source)
    shift_semitones[voiced_mask] = 12 * np.log2(f0_target[voiced_mask] / f0_source[voiced_mask])

    # Smooth the shift
    from scipy.ndimage import median_filter
    shift_semitones = median_filter(shift_semitones, size=5)

    # Apply pitch shift frame by frame
    # This is computationally expensive but gives best results
    logger.info("Applying pitch correction...")

    # For simplicity, use librosa's pitch_shift with average shift
    # (Frame-by-frame PSOLA would be ideal but is much more complex)
    avg_shift = np.mean(shift_semitones[voiced_mask]) if np.any(voiced_mask) else 0
    logger.info(f"Average pitch shift: {avg_shift:.1f} semitones")

    if abs(avg_shift) > 0.5:  # Only shift if significant
        corrected = librosa.effects.pitch_shift(
            aligned_audio, sr=sr, n_steps=avg_shift
        )
    else:
        corrected = aligned_audio

    return corrected


def run_voice_conversion(
    source_audio: np.ndarray,
    source_sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
) -> Tuple[np.ndarray, int]:
    """Run Seed-VC voice conversion."""
    logger.info("Running Seed-VC voice conversion...")

    # Import quality pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from quality_pipeline import QualityVoiceConverter, QualityConfig

    config = QualityConfig(
        diffusion_steps=30,
        auto_f0_adjust=True,
        fp16=True,
    )

    converter = QualityVoiceConverter(config)
    converted, out_sr = converter.convert(
        source_audio, source_sr,
        reference_audio, reference_sr,
        pitch_shift=0
    )

    return converted, out_sr


def mix_with_instrumental(
    vocals: np.ndarray,
    vocals_sr: int,
    instrumental: np.ndarray,
    instrumental_sr: int,
    vocal_gain: float = 1.0,
    inst_gain: float = 0.8,
) -> Tuple[np.ndarray, int]:
    """Mix vocals with instrumental."""
    logger.info("Mixing with instrumental...")

    # Resample to match
    if vocals_sr != instrumental_sr:
        vocals = librosa.resample(vocals, orig_sr=vocals_sr, target_sr=instrumental_sr)

    # Match lengths
    min_len = min(len(vocals), len(instrumental))
    vocals = vocals[:min_len]
    instrumental = instrumental[:min_len]

    # Apply gains and mix
    mixed = vocals * vocal_gain + instrumental * inst_gain

    # Normalize to prevent clipping
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed = mixed * 0.95 / peak

    return mixed, instrumental_sr


def main():
    parser = argparse.ArgumentParser(description='Aligned Voice Conversion')
    parser.add_argument('--source', required=True, help='Source vocals (to convert)')
    parser.add_argument('--target', required=True, help='Target vocals (for timing reference)')
    parser.add_argument('--reference', help='Reference for voice style (defaults to target)')
    parser.add_argument('--instrumental', help='Instrumental to mix with')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--skip-alignment', action='store_true', help='Skip DTW alignment')
    parser.add_argument('--skip-pitch-correction', action='store_true', help='Skip pitch correction')
    parser.add_argument('--vocal-gain', type=float, default=1.0, help='Vocal volume')
    parser.add_argument('--inst-gain', type=float, default=0.8, help='Instrumental volume')
    args = parser.parse_args()

    # Load audio files
    logger.info(f"Loading source: {args.source}")
    source, source_sr = librosa.load(args.source, sr=None, mono=True)

    logger.info(f"Loading target: {args.target}")
    target, target_sr = librosa.load(args.target, sr=None, mono=True)

    reference = target
    reference_sr = target_sr
    if args.reference:
        logger.info(f"Loading reference: {args.reference}")
        reference, reference_sr = librosa.load(args.reference, sr=None, mono=True)

    logger.info(f"Source: {len(source)/source_sr:.1f}s, Target: {len(target)/target_sr:.1f}s")

    # Step 1: Align source to target timing
    if not args.skip_alignment:
        aligned, aligned_sr = align_with_dtw(source, source_sr, target, target_sr)
    else:
        aligned = source
        aligned_sr = source_sr
        logger.info("Skipping alignment")

    # Step 2: Optional pitch correction
    if not args.skip_pitch_correction:
        aligned = align_pitch(aligned, aligned_sr, target, target_sr)
    else:
        logger.info("Skipping pitch correction")

    # Save aligned audio for debugging
    aligned_path = Path(args.output).parent / f"{Path(args.output).stem}_aligned.wav"
    sf.write(aligned_path, aligned, aligned_sr)
    logger.info(f"Saved aligned audio: {aligned_path}")

    # Step 3: Voice conversion
    converted, converted_sr = run_voice_conversion(
        aligned, aligned_sr,
        reference, reference_sr
    )

    # Save converted vocals
    converted_path = Path(args.output).parent / f"{Path(args.output).stem}_converted.wav"
    sf.write(converted_path, converted, converted_sr)
    logger.info(f"Saved converted vocals: {converted_path}")

    # Step 4: Mix with instrumental (if provided)
    if args.instrumental:
        logger.info(f"Loading instrumental: {args.instrumental}")
        instrumental, inst_sr = librosa.load(args.instrumental, sr=None, mono=True)

        final, final_sr = mix_with_instrumental(
            converted, converted_sr,
            instrumental, inst_sr,
            args.vocal_gain, args.inst_gain
        )
    else:
        final = converted
        final_sr = converted_sr

    # Save final output
    sf.write(args.output, final, final_sr)
    logger.info(f"Saved final output: {args.output}")

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Source duration: {len(source)/source_sr:.1f}s")
    logger.info(f"Target duration: {len(target)/target_sr:.1f}s")
    logger.info(f"Aligned duration: {len(aligned)/aligned_sr:.1f}s")
    logger.info(f"Converted duration: {len(converted)/converted_sr:.1f}s")
    logger.info(f"Final duration: {len(final)/final_sr:.1f}s")


if __name__ == '__main__':
    main()
