#!/usr/bin/env python3
"""Realistic demo of AutoVoice singing voice conversion.

This demo shows the complete workflow:
1. Load pre-trained models
2. Create voice profile from reference audio
3. Convert a song to the target voice
4. Save and evaluate results

Usage:
    python examples/demo_voice_conversion.py --song <path> --reference <path>
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import soundfile as sf

from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
from auto_voice.inference.voice_cloner import VoiceCloner


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def progress_callback(percent: float, stage: str):
    """Progress callback for conversion."""
    stages = {
        'source_separation': '🎵 Separating vocals from instrumental',
        'pitch_extraction': '🎼 Extracting pitch contour',
        'voice_conversion': '🎤 Converting voice',
        'audio_mixing': '🎹 Mixing final audio'
    }
    stage_name = stages.get(stage, stage)
    print(f"\r  [{percent:5.1f}%] {stage_name}...", end='', flush=True)
    if percent >= 100:
        print("  ✓")


def main():
    parser = argparse.ArgumentParser(description='AutoVoice Singing Voice Conversion Demo')
    parser.add_argument(
        '--song',
        type=str,
        help='Path to song file (MP3/WAV/FLAC)'
    )
    parser.add_argument(
        '--reference',
        type=str,
        help='Path to reference voice audio (30-60s recommended)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/demo_converted.wav',
        help='Output file path (default: output/demo_converted.wav)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/pretrained',
        help='Directory containing pre-trained models'
    )
    parser.add_argument(
        '--pitch-shift',
        type=float,
        default=0.0,
        help='Pitch shift in semitones (default: 0)'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='balanced',
        choices=['draft', 'fast', 'balanced', 'high', 'studio'],
        help='Quality preset (default: balanced)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.song or not args.reference:
        print("❌ Error: Both --song and --reference are required")
        print("\nExample usage:")
        print("  python examples/demo_voice_conversion.py \\")
        print("    --song data/test_song.mp3 \\")
        print("    --reference data/my_voice.wav \\")
        print("    --preset balanced")
        return 1
    
    if not os.path.exists(args.song):
        print(f"❌ Error: Song file not found: {args.song}")
        return 1
    
    if not os.path.exists(args.reference):
        print(f"❌ Error: Reference audio not found: {args.reference}")
        return 1
    
    # Check for pre-trained models
    models_dir = Path(args.models_dir)
    required_models = [
        'sovits5.0_main_1500.pth',
        'hifigan_ljspeech.ckpt',
        'hubert-soft-0d54a1f4.pt'
    ]
    
    missing_models = [m for m in required_models if not (models_dir / m).exists()]
    if missing_models:
        print("❌ Error: Required pre-trained models not found:")
        for model in missing_models:
            print(f"  - {model}")
        print(f"\nPlease run: python scripts/download_pretrained_models.py")
        return 1
    
    print_header("🎤 AutoVoice Singing Voice Conversion Demo")
    
    # System info
    print("\n📋 Configuration:")
    print(f"  Song: {args.song}")
    print(f"  Reference voice: {args.reference}")
    print(f"  Output: {args.output}")
    print(f"  Pitch shift: {args.pitch_shift:+.1f} semitones")
    print(f"  Quality preset: {args.preset}")
    print(f"  Device: {args.device}")
    
    try:
        # Check PyTorch
        import torch
        print(f"\n🔧 System:")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\n⚠️  Warning: PyTorch not available, using CPU fallback")
        args.device = 'cpu'
    
    # Initialize components
    print_header("Step 1: Initialize Components")
    
    try:
        print("\n🔧 Initializing voice cloner...")
        voice_cloner = VoiceCloner(device=args.device)
        print("  ✓ Voice cloner ready")
        
        print("\n🔧 Initializing conversion pipeline...")
        # Create config pointing to pre-trained models
        config = {
            'model_path': str(models_dir / 'sovits5.0_main_1500.pth'),
            'vocoder_path': str(models_dir / 'hifigan_ljspeech.ckpt'),
            'hubert_path': str(models_dir / 'hubert-soft-0d54a1f4.pt'),
            'device': args.device,
            'preset': args.preset
        }
        
        pipeline = SingingConversionPipeline(
            config=config,
            device=args.device,
            voice_cloner=voice_cloner,
            preset=args.preset
        )
        print("  ✓ Pipeline ready")
        
    except Exception as e:
        print(f"\n❌ Error initializing components: {e}")
        print("\nThis might be due to:")
        print("  - Missing dependencies (run: pip install -r requirements.txt)")
        print("  - PyTorch environment issues (run: ./scripts/setup_pytorch_env.sh)")
        return 1
    
    # Create voice profile
    print_header("Step 2: Create Voice Profile")
    
    try:
        print(f"\n📁 Loading reference audio: {args.reference}")
        
        profile = voice_cloner.create_voice_profile(
            audio=args.reference,
            user_id='demo_user',
            profile_name='Demo Target Voice',
            metadata={'source': 'demo', 'file': args.reference}
        )
        
        profile_id = profile['profile_id']
        print(f"  ✓ Voice profile created: {profile_id}")
        
        if 'vocal_range' in profile:
            vr = profile['vocal_range']
            print(f"  Vocal range: {vr.get('min_note', 'N/A')} - {vr.get('max_note', 'N/A')}")
        
        if 'quality_score' in profile:
            print(f"  Quality score: {profile['quality_score']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error creating voice profile: {e}")
        return 1
    
    # Convert song
    print_header("Step 3: Convert Song")
    
    try:
        print(f"\n🎵 Converting song: {args.song}")
        print("\nThis may take 1-5 minutes depending on song length and quality preset...\n")
        
        start_time = time.time()
        
        result = pipeline.convert_song(
            song_path=args.song,
            target_profile_id=profile_id,
            vocal_volume=1.0,
            instrumental_volume=0.9,
            pitch_shift=args.pitch_shift,
            preset=args.preset,
            progress_callback=progress_callback,
            return_stems=True
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n  ✓ Conversion complete in {elapsed:.1f}s")
        print(f"  Duration: {result['duration']:.1f}s")
        print(f"  Sample rate: {result['sample_rate']} Hz")
        
        # Show quality metrics if available
        if 'metadata' in result and 'f0_stats' in result['metadata']:
            f0_stats = result['metadata']['f0_stats']
            if f0_stats:
                print(f"\n  📊 Pitch Statistics:")
                print(f"    Mean F0: {f0_stats.get('mean_f0', 0):.1f} Hz")
                print(f"    Range: {f0_stats.get('min_f0', 0):.1f} - {f0_stats.get('max_f0', 0):.1f} Hz")
                print(f"    Voiced: {f0_stats.get('voiced_fraction', 0)*100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save output
    print_header("Step 4: Save Results")
    
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mixed audio
        print(f"\n💾 Saving converted audio: {output_path}")
        sf.write(output_path, result['mixed_audio'], result['sample_rate'])
        print("  ✓ Saved")
        
        # Save stems if available
        if result.get('vocals') is not None:
            vocals_path = output_path.parent / f"{output_path.stem}_vocals{output_path.suffix}"
            print(f"\n💾 Saving converted vocals: {vocals_path}")
            sf.write(vocals_path, result['vocals'], result['sample_rate'])
            print("  ✓ Saved")
        
        if result.get('instrumental') is not None:
            inst_path = output_path.parent / f"{output_path.stem}_instrumental{output_path.suffix}"
            print(f"\n💾 Saving instrumental: {inst_path}")
            sf.write(inst_path, result['instrumental'], result['sample_rate'])
            print("  ✓ Saved")
        
    except Exception as e:
        print(f"\n❌ Error saving output: {e}")
        return 1
    
    # Success summary
    print_header("✅ Demo Complete!")
    print(f"\nConverted audio saved to: {output_path.absolute()}")
    print(f"\n🎧 To listen, run:")
    print(f"  - On Linux: aplay {output_path}")
    print(f"  - On macOS: afplay {output_path}")
    print(f"  - Or open with your media player")
    
    print(f"\n📊 Performance:")
    print(f"  Processing time: {elapsed:.1f}s")
    print(f"  Audio duration: {result['duration']:.1f}s")
    print(f"  Real-time factor: {elapsed/result['duration']:.2f}x")
    
    print("\n💡 Tips:")
    print("  - Use --preset studio for best quality (slower)")
    print("  - Use --preset fast for quick testing")
    print("  - Try --pitch-shift ±2 to adjust key")
    print("  - Provide 30-60s reference audio for best voice cloning")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
