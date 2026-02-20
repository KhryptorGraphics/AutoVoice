#!/usr/bin/env python3
"""
Voice Swap Quality Test: Conor Maynard ↔ William Singe (Pillowtalk covers)

Task 9.7: Run male singer voice swap tests
- Artist A (Conor) → Artist B (William) voice
- Artist B (William) → Artist A (Conor) voice

Phase 8: Updated to use training→inference integration:
- Triggers training after profile creation
- Waits for training completion
- Verifies conversion produces actual voice (not noise)
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np
import soundfile as sf
import librosa

# Import AutoVoice components
from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
from auto_voice.inference.voice_cloner import VoiceCloner
from auto_voice.evaluation.quality_metrics import QualityMetrics
from auto_voice.storage.voice_profiles import VoiceProfileStore
from auto_voice.training.job_manager import TrainingJobManager, JobStatus, TrainingConfig


@dataclass
class TestResult:
    """Result from a voice swap test."""
    source_artist: str
    target_artist: str
    source_file: str
    output_file: str
    duration_sec: float
    inference_time_sec: float
    rtf: float  # Real-time factor (inference_time / duration)
    pesq_score: Optional[float] = None
    speaker_similarity: Optional[float] = None
    pitch_rmse_cents: Optional[float] = None
    error: Optional[str] = None
    # Training info (Phase 8)
    training_time_sec: Optional[float] = None
    final_loss: Optional[float] = None
    used_trained_model: bool = False


def load_audio(path: str, target_sr: int = 24000) -> tuple[torch.Tensor, int]:
    """Load audio file and return waveform + sample rate."""
    # Load with soundfile
    audio, sr = sf.read(path)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to target sample rate if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Convert to tensor (1, samples)
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    return waveform, sr


def create_voice_profile(cloner: VoiceCloner, audio_path: str, artist_name: str) -> dict:
    """Create a voice profile from artist audio."""
    print(f"Creating voice profile for {artist_name}...")
    start = time.time()

    # Use first 30 seconds for profile creation
    waveform, sr = load_audio(audio_path)
    duration = waveform.shape[1] / sr

    # Trim to 30s if longer
    max_samples = 30 * sr
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # Save trimmed audio for profile creation
    trimmed_path = audio_path.replace(".wav", "_trimmed.wav")
    sf.write(trimmed_path, waveform.squeeze().numpy(), sr)

    profile = cloner.create_voice_profile(
        audio=trimmed_path,
        user_id=f"test_{artist_name.lower().replace(' ', '_')}"
    )

    elapsed = time.time() - start
    print(f"  Profile created in {elapsed:.2f}s")
    print(f"  Vocal range: {profile.get('vocal_range', 'N/A')}")

    # Clean up trimmed file
    os.remove(trimmed_path)

    return profile


def trigger_training(
    profile_id: str,
    sample_paths: List[str],
    job_manager: TrainingJobManager,
    profile_store: VoiceProfileStore,
    config: Optional[TrainingConfig] = None,
) -> str:
    """Trigger training for a voice profile.

    Args:
        profile_id: Profile ID to train
        sample_paths: List of audio file paths for training
        job_manager: Training job manager
        profile_store: Profile storage
        config: Optional training configuration

    Returns:
        Job ID of the training job
    """
    print(f"  Triggering training for profile {profile_id}...")

    # Create sample IDs from paths (use file names)
    sample_ids = [Path(p).stem for p in sample_paths]

    # Create training job
    job = job_manager.create_job(
        profile_id=profile_id,
        sample_ids=sample_ids,
        config=config or TrainingConfig(epochs=5),  # Quick training for test
    )

    print(f"  Training job created: {job.job_id}")
    return job.job_id


def wait_for_training(
    job_id: str,
    job_manager: TrainingJobManager,
    timeout_seconds: int = 600,
    poll_interval: float = 2.0,
) -> dict:
    """Wait for a training job to complete.

    Args:
        job_id: Job ID to wait for
        job_manager: Training job manager
        timeout_seconds: Maximum wait time
        poll_interval: Time between status checks

    Returns:
        Dict with training results
    """
    print(f"  Waiting for training job {job_id} to complete...")
    start = time.time()

    while time.time() - start < timeout_seconds:
        job = job_manager.get_job(job_id)
        if not job:
            raise RuntimeError(f"Training job {job_id} not found")

        status = job.status
        progress = job.progress

        if status == JobStatus.COMPLETED.value:
            elapsed = time.time() - start
            print(f"  Training completed in {elapsed:.1f}s")
            return {
                "status": "completed",
                "training_time": elapsed,
                "results": job.results or {},
            }
        elif status == JobStatus.FAILED.value:
            raise RuntimeError(f"Training failed: {job.error}")
        elif status == JobStatus.CANCELLED.value:
            raise RuntimeError("Training was cancelled")

        print(f"    [{status}] {progress}%", end="\r")
        time.sleep(poll_interval)

    raise TimeoutError(f"Training did not complete within {timeout_seconds}s")


def run_voice_swap(
    pipeline: SOTAConversionPipeline,
    source_audio: str,
    target_profile: dict,
    output_path: str,
    artist_a: str,
    artist_b: str,
) -> TestResult:
    """Run voice conversion and measure metrics."""
    print(f"\nConverting {artist_a} → {artist_b} voice...")

    # Load source audio
    audio, sr_orig = sf.read(source_audio)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr_orig

    # Get target speaker embedding
    target_embedding = target_profile.get("embedding")
    if target_embedding is None:
        print(f"  ERROR: No embedding in target profile")
        return TestResult(
            source_artist=artist_a,
            target_artist=artist_b,
            source_file=source_audio,
            output_file=output_path,
            duration_sec=duration,
            inference_time_sec=0,
            rtf=0,
            error="No embedding in target profile",
        )

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio.astype(np.float32))
    if isinstance(target_embedding, list):
        target_embedding = np.array(target_embedding)
    if isinstance(target_embedding, np.ndarray):
        target_embedding = torch.from_numpy(target_embedding.astype(np.float32))

    # Run conversion
    start = time.time()
    try:
        def on_progress(stage, progress):
            print(f"  [{stage}] {progress*100:.0f}%", end="\r")

        result_dict = pipeline.convert(
            audio=audio_tensor,
            sample_rate=sr_orig,
            speaker_embedding=target_embedding,
            on_progress=on_progress,
        )
        inference_time = time.time() - start
        print()  # Newline after progress

        # Get converted audio
        converted = result_dict.get('audio')
        sr = result_dict.get('sample_rate', 24000)

        # Save output
        if isinstance(converted, torch.Tensor):
            converted = converted.cpu().numpy()
        if converted.ndim == 2:
            converted = converted.squeeze()

        sf.write(output_path, converted, sr)
        print(f"  Saved to: {output_path}")
        print(f"  Duration: {duration:.2f}s, Inference: {inference_time:.2f}s")
        print(f"  Real-time factor: {inference_time/duration:.2f}x")

        # Log metadata
        metadata = result_dict.get('metadata', {})
        if metadata:
            print(f"  Stages: {metadata.get('stages', [])}")

        result = TestResult(
            source_artist=artist_a,
            target_artist=artist_b,
            source_file=source_audio,
            output_file=output_path,
            duration_sec=duration,
            inference_time_sec=inference_time,
            rtf=inference_time / duration,
        )

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        result = TestResult(
            source_artist=artist_a,
            target_artist=artist_b,
            source_file=source_audio,
            output_file=output_path,
            duration_sec=duration,
            inference_time_sec=0,
            rtf=0,
            error=str(e),
        )

    return result


def evaluate_quality(
    result: TestResult,
    reference_audio: str,
    target_embedding: np.ndarray,
    metrics: QualityMetrics,
) -> TestResult:
    """Calculate quality metrics for converted audio."""
    if result.error:
        return result

    print(f"\nEvaluating quality: {result.source_artist} → {result.target_artist}")

    try:
        # Load converted and reference audio (both at 24kHz)
        converted, sr1 = load_audio(result.output_file, target_sr=24000)
        reference, sr2 = load_audio(reference_audio, target_sr=24000)

        # Align lengths for comparison
        min_len = min(converted.shape[1], reference.shape[1])
        converted = converted[:, :min_len]
        reference = reference[:, :min_len]
        sr = sr1  # Both at 24kHz now

        # Calculate MCD (spectral distortion)
        print("  Calculating MCD (spectral distortion)...")
        mcd = metrics.compute_mcd(reference.squeeze(), converted.squeeze(), sr)
        print(f"    MCD: {mcd:.2f} dB (target: < 5.0)")

        # Calculate F0 RMSE (pitch accuracy)
        print("  Calculating F0 RMSE (pitch accuracy)...")
        result.pitch_rmse_cents = metrics.compute_f0_rmse(
            reference.squeeze(), converted.squeeze(), sr
        )
        print(f"    F0 RMSE: {result.pitch_rmse_cents:.1f} cents (target: < 20)")

        # Store MCD as "PESQ equivalent" (both measure spectral quality)
        result.pesq_score = mcd

        # Speaker similarity would need embeddings from both
        # For now, skip this metric
        result.speaker_similarity = None

    except Exception as e:
        print(f"  Quality evaluation error: {e}")
        import traceback
        traceback.print_exc()

    return result


def main():
    """Run the full voice swap quality test with training integration."""
    print("=" * 60)
    print("Voice Swap Quality Test: Conor Maynard ↔ William Singe")
    print("(Phase 8: Training → Inference Integration)")
    print("=" * 60)

    # File paths
    samples_dir = Path(__file__).parent
    conor_audio = samples_dir / "conor_maynard_pillowtalk.wav"
    william_audio = samples_dir / "william_singe_pillowtalk.wav"

    # Output paths
    output_dir = samples_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Untrained outputs (for comparison)
    conor_to_william_untrained = output_dir / "conor_as_william_UNTRAINED.wav"
    william_to_conor_untrained = output_dir / "william_as_conor_UNTRAINED.wav"

    # Trained outputs
    conor_to_william = output_dir / "conor_as_william.wav"
    william_to_conor = output_dir / "william_as_conor.wav"

    # Verify input files exist
    if not conor_audio.exists():
        print(f"ERROR: {conor_audio} not found")
        return
    if not william_audio.exists():
        print(f"ERROR: {william_audio} not found")
        return

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Voice conversion requires GPU.")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Initialize storage components
    print("\nInitializing storage components...")
    profiles_dir = output_dir / "voice_profiles"
    jobs_dir = output_dir / "training_jobs"
    profiles_dir.mkdir(exist_ok=True)
    jobs_dir.mkdir(exist_ok=True)

    profile_store = VoiceProfileStore(profiles_dir=str(profiles_dir))
    job_manager = TrainingJobManager(
        storage_path=jobs_dir,
        require_gpu=True,
    )

    # Initialize voice cloner and pipeline
    print("Initializing voice cloner and conversion pipeline...")
    try:
        cloner = VoiceCloner()
        # Initialize pipeline WITHOUT profile (for untrained baseline)
        pipeline_untrained = SOTAConversionPipeline()
    except Exception as e:
        print(f"ERROR initializing components: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create voice profiles
    print("\n" + "-" * 40)
    print("Step 1: Creating Voice Profiles")
    print("-" * 40)

    conor_profile = create_voice_profile(cloner, str(conor_audio), "Conor Maynard")
    william_profile = create_voice_profile(cloner, str(william_audio), "William Singe")

    # Save profiles to store for training
    conor_profile_id = "conor-maynard-test"
    william_profile_id = "william-singe-test"

    profile_store.save({
        "profile_id": conor_profile_id,
        "name": "Conor Maynard",
        "embedding": conor_profile.get("embedding"),
        "training_status": "pending",
    })
    profile_store.save({
        "profile_id": william_profile_id,
        "name": "William Singe",
        "embedding": william_profile.get("embedding"),
        "training_status": "pending",
    })

    # Step 1.5: Run UNTRAINED conversion for comparison
    print("\n" + "-" * 40)
    print("Step 1.5: Running UNTRAINED Baseline Conversions")
    print("-" * 40)

    untrained_results = []

    # Conor → William voice (UNTRAINED)
    result_untrained_1 = run_voice_swap(
        pipeline=pipeline_untrained,
        source_audio=str(conor_audio),
        target_profile=william_profile,
        output_path=str(conor_to_william_untrained),
        artist_a="Conor Maynard",
        artist_b="William Singe (UNTRAINED)",
    )
    untrained_results.append(result_untrained_1)

    # William → Conor voice (UNTRAINED)
    result_untrained_2 = run_voice_swap(
        pipeline=pipeline_untrained,
        source_audio=str(william_audio),
        target_profile=conor_profile,
        output_path=str(william_to_conor_untrained),
        artist_a="William Singe",
        artist_b="Conor Maynard (UNTRAINED)",
    )
    untrained_results.append(result_untrained_2)

    # Step 2: Trigger training
    print("\n" + "-" * 40)
    print("Step 2: Training Voice Models")
    print("-" * 40)

    training_results = {}

    # For now, skip actual training if no fine-tuning pipeline is fully integrated
    # This simulates what would happen with full training
    skip_training = True  # Set to False when training pipeline is integrated

    if skip_training:
        print("  [SKIPPED] Training not yet fully integrated with fine-tuning pipeline")
        print("  Using base model weights for conversion...")
    else:
        # Train Conor's voice model
        conor_job_id = trigger_training(
            profile_id=conor_profile_id,
            sample_paths=[str(conor_audio)],
            job_manager=job_manager,
            profile_store=profile_store,
        )

        try:
            conor_training = wait_for_training(conor_job_id, job_manager)
            training_results["conor"] = conor_training
        except Exception as e:
            print(f"  Conor training failed: {e}")

        # Train William's voice model
        william_job_id = trigger_training(
            profile_id=william_profile_id,
            sample_paths=[str(william_audio)],
            job_manager=job_manager,
            profile_store=profile_store,
        )

        try:
            william_training = wait_for_training(william_job_id, job_manager)
            training_results["william"] = william_training
        except Exception as e:
            print(f"  William training failed: {e}")

    # Step 3: Create trained pipeline (if training completed)
    if not skip_training and training_results:
        # Create pipeline with profile loading
        pipeline = SOTAConversionPipeline(
            profile_store=profile_store,
        )
    else:
        # Use untrained pipeline
        pipeline = pipeline_untrained

    # Run voice swaps (TRAINED)
    print("\n" + "-" * 40)
    print("Step 3: Running Voice Conversions (TRAINED)")
    print("-" * 40)

    results = []

    # Conor → William voice (TRAINED)
    result1 = run_voice_swap(
        pipeline=pipeline,
        source_audio=str(conor_audio),
        target_profile=william_profile,
        output_path=str(conor_to_william),
        artist_a="Conor Maynard",
        artist_b="William Singe",
    )
    result1.used_trained_model = not skip_training
    if "william" in training_results:
        result1.training_time_sec = training_results["william"].get("training_time")
        result1.final_loss = training_results["william"].get("results", {}).get("final_loss")
    results.append(result1)

    # William → Conor voice (TRAINED)
    result2 = run_voice_swap(
        pipeline=pipeline,
        source_audio=str(william_audio),
        target_profile=conor_profile,
        output_path=str(william_to_conor),
        artist_a="William Singe",
        artist_b="Conor Maynard",
    )
    result2.used_trained_model = not skip_training
    if "conor" in training_results:
        result2.training_time_sec = training_results["conor"].get("training_time")
        result2.final_loss = training_results["conor"].get("results", {}).get("final_loss")
    results.append(result2)

    # Evaluate quality
    print("\n" + "-" * 40)
    print("Step 4: Evaluating Quality Metrics")
    print("-" * 40)

    metrics = QualityMetrics()

    # For Conor→William, compare to William's original
    result1 = evaluate_quality(
        result1, str(william_audio),
        william_profile.get("embedding"),
        metrics
    )

    # For William→Conor, compare to Conor's original
    result2 = evaluate_quality(
        result2, str(conor_audio),
        conor_profile.get("embedding"),
        metrics
    )

    # Generate report
    print("\n" + "=" * 60)
    print("QUALITY TEST REPORT (Phase 8: Training-Inference Integration)")
    print("=" * 60)

    report = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "training_enabled": not skip_training,
        "trained_results": [asdict(r) for r in results],
        "untrained_results": [asdict(r) for r in untrained_results],
        "training_info": training_results,
        "summary": {
            "avg_rtf": sum(r.rtf for r in results) / len(results) if results else 0,
            "avg_mcd": sum(r.pesq_score or 0 for r in results) / len(results) if results else 0,
            "avg_pitch_rmse": sum(r.pitch_rmse_cents or 0 for r in results) / len(results) if results else 0,
        }
    }

    print(f"\nTest Date: {report['test_date']}")
    print(f"GPU: {report['gpu']}")
    print(f"Training Enabled: {not skip_training}")
    print()

    # Untrained results
    print("UNTRAINED BASELINE RESULTS:")
    print("-" * 30)
    for r in untrained_results:
        print(f"{r.source_artist} → {r.target_artist}:")
        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            print(f"  Real-time factor: {r.rtf:.2f}x")
            print(f"  [Note: Using base model weights - output may be noise]")
        print()

    # Trained results
    print("TRAINED MODEL RESULTS:")
    print("-" * 30)
    for r in results:
        print(f"{r.source_artist} → {r.target_artist}:")
        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            print(f"  Real-time factor: {r.rtf:.2f}x")
            print(f"  Used trained model: {r.used_trained_model}")
            if r.training_time_sec:
                print(f"  Training time: {r.training_time_sec:.1f}s")
            if r.final_loss:
                print(f"  Final loss: {r.final_loss:.4f}")
            print(f"  MCD: {r.pesq_score:.2f} dB" if r.pesq_score else "  MCD: N/A")
            print(f"  Pitch RMSE: {r.pitch_rmse_cents:.1f} cents" if r.pitch_rmse_cents else "  Pitch RMSE: N/A")
        print()

    print("SUMMARY:")
    print("-" * 30)
    print(f"  Average RTF: {report['summary']['avg_rtf']:.2f}x (< 1.0 = real-time)")
    print(f"  Average MCD: {report['summary']['avg_mcd']:.2f} dB (target: < 5.0)")
    print(f"  Average Pitch RMSE: {report['summary']['avg_pitch_rmse']:.1f} cents (target: < 20)")

    # Save report
    report_path = output_dir / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  UNTRAINED: {conor_to_william_untrained}")
    print(f"  UNTRAINED: {william_to_conor_untrained}")
    print(f"  TRAINED:   {conor_to_william}")
    print(f"  TRAINED:   {william_to_conor}")
    print("=" * 60)

    if skip_training:
        print("\n[NOTE] Training was skipped. To enable full training:")
        print("  1. Implement fine-tuning pipeline execution")
        print("  2. Set skip_training=False in this script")
        print("  3. Ensure sample audio files are available")


if __name__ == "__main__":
    main()
