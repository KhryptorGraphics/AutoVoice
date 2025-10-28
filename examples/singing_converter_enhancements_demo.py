#!/usr/bin/env python3
"""
Demo script for SingingVoiceConverter enhancements

This script demonstrates the new features:
1. Temperature API
2. Pitch shifting
3. Quality presets
4. Advanced features (denoise, enhance, preserve_dynamics)
"""

import sys
import argparse
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from auto_voice.models.singing_voice_converter import SingingVoiceConverter


def load_config(config_path='config/model_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_test_audio(duration=1.0, sample_rate=16000, frequency=440.0):
    """Generate synthetic test audio (sine wave)."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Add some vibrato
    vibrato_rate = 5.0  # Hz
    vibrato_depth = 10.0  # Hz
    freq_modulation = frequency + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    audio = np.sin(2 * np.pi * freq_modulation * t) * 0.5
    return audio.astype(np.float32)


def generate_test_embedding():
    """Generate random speaker embedding for testing."""
    return np.random.randn(256).astype(np.float32)


def demo_temperature_api(model):
    """Demo 1: Temperature API."""
    print("\n" + "="*60)
    print("DEMO 1: Temperature API")
    print("="*60)

    audio = generate_test_audio()
    embedding = generate_test_embedding()

    # Test different temperatures
    temperatures = [0.5, 1.0, 1.5]
    print("\nTesting different temperatures:")
    for temp in temperatures:
        model.set_temperature(temp)
        print(f"  Temperature {temp:.1f}: Set successfully")

    # Auto-tune temperature
    print("\nAuto-tuning temperature based on audio characteristics:")
    optimal = model.auto_tune_temperature(audio, embedding, 16000)
    print(f"  Optimal temperature: {optimal:.3f}")


def demo_pitch_shifting(model):
    """Demo 2: Pitch Shifting."""
    print("\n" + "="*60)
    print("DEMO 2: Pitch Shifting")
    print("="*60)

    audio = generate_test_audio(frequency=440.0)  # A4
    embedding = generate_test_embedding()

    shifts = [
        (2.0, "linear", "Up 2 semitones (whole step)"),
        (-3.0, "linear", "Down 3 semitones (minor third)"),
        (7.0, "formant_preserving", "Up 7 semitones (perfect fifth, formant-preserving)")
    ]

    print("\nTesting pitch shifts:")
    for semitones, method, description in shifts:
        try:
            result = model.convert(
                audio,
                embedding,
                pitch_shift_semitones=semitones,
                pitch_shift_method=method
            )
            print(f"  ✓ {description}")
            print(f"    Method: {method}, Output length: {len(result)} samples")
        except Exception as e:
            print(f"  ✗ {description}: {e}")


def demo_quality_presets(model):
    """Demo 3: Quality Presets."""
    print("\n" + "="*60)
    print("DEMO 3: Quality Presets")
    print("="*60)

    presets = ['draft', 'fast', 'balanced', 'high', 'studio']

    print("\nAvailable quality presets:")
    for preset in presets:
        info = model.get_quality_preset_info(preset)
        print(f"\n  {preset.upper()}:")
        print(f"    Description: {info['description']}")
        print(f"    Decoder steps: {info['decoder_steps']}")
        print(f"    Relative quality: {info['relative_quality']:.1f}x")
        print(f"    Relative speed: {info['relative_speed']:.1f}x")

    print("\nEstimated conversion times for 30-second audio:")
    audio_duration = 30.0
    for preset in presets:
        est_time = model.estimate_conversion_time(audio_duration, preset)
        print(f"  {preset:>10}: {est_time:>6.2f} seconds")

    print("\nSetting preset to 'high':")
    model.set_quality_preset('high')
    print(f"  Current preset: {model.quality_preset}")
    print(f"  Decoder steps: {model.decoder_steps}")


def demo_advanced_features(model):
    """Demo 4: Advanced Features."""
    print("\n" + "="*60)
    print("DEMO 4: Advanced Features")
    print("="*60)

    # Generate noisy audio
    clean_audio = generate_test_audio()
    noise = np.random.randn(len(clean_audio)) * 0.05
    noisy_audio = clean_audio + noise

    embedding = generate_test_embedding()

    features = [
        ({"denoise_input": True}, "Denoise input"),
        ({"enhance_output": True}, "Enhance output"),
        ({"preserve_dynamics": True}, "Preserve dynamics"),
        ({"denoise_input": True, "enhance_output": True, "preserve_dynamics": True}, "All features"),
    ]

    print("\nTesting advanced features:")
    for kwargs, description in features:
        try:
            result = model.convert(
                noisy_audio,
                embedding,
                **kwargs
            )
            print(f"  ✓ {description}: Success (output: {len(result)} samples)")
        except Exception as e:
            print(f"  ✗ {description}: {e}")


def demo_combined_example(model):
    """Demo 5: Combined Example."""
    print("\n" + "="*60)
    print("DEMO 5: Combined Example (All Features)")
    print("="*60)

    audio = generate_test_audio()
    embedding = generate_test_embedding()

    # Configure model
    print("\nConfiguration:")
    print("  - Quality preset: 'high'")
    print("  - Temperature: Auto-tuned")
    print("  - Pitch shift: +2 semitones")
    print("  - Features: denoise + enhance + preserve_dynamics")

    model.set_quality_preset('high')
    optimal_temp = model.auto_tune_temperature(audio, embedding, 16000)

    try:
        result = model.convert(
            audio,
            embedding,
            pitch_shift_semitones=2.0,
            pitch_shift_method='linear',
            denoise_input=True,
            enhance_output=True,
            preserve_dynamics=True
        )
        print(f"\n✓ Conversion successful!")
        print(f"  Output length: {len(result)} samples")
        print(f"  Temperature used: {optimal_temp:.3f}")
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")


def main():
    """Main demo runner."""
    parser = argparse.ArgumentParser(
        description='Demo SingingVoiceConverter enhancements'
    )
    parser.add_argument(
        '--config',
        default='config/model_config.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--demo',
        choices=['all', 'temperature', 'pitch', 'quality', 'advanced', 'combined'],
        default='all',
        help='Which demo to run'
    )
    args = parser.parse_args()

    print("="*60)
    print("SingingVoiceConverter Enhancements Demo")
    print("="*60)

    # Load configuration and create model
    print("\nLoading model...")
    try:
        config = load_config(args.config)
        model = SingingVoiceConverter(config)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return 1

    # Run demos
    demos = {
        'temperature': demo_temperature_api,
        'pitch': demo_pitch_shifting,
        'quality': demo_quality_presets,
        'advanced': demo_advanced_features,
        'combined': demo_combined_example
    }

    if args.demo == 'all':
        for demo_func in demos.values():
            demo_func(model)
    else:
        demos[args.demo](model)

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
