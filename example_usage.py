#!/usr/bin/env python3
"""
Example usage of the AutoVoice system.
Demonstrates voice synthesis, voice cloning, and voice conversion.
"""

import sys
import os
import torch
import numpy as np

# Add src directory to path
sys.path.insert(0, 'src')

from auto_voice import (
    GPUManager,
    AudioProcessor,
    VoiceAnalyzer,
    VoiceTransformer,
    Vocoder,
    VoiceSynthesizer,
    Config
)

def example_basic_synthesis():
    """Example of basic voice synthesis."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Voice Synthesis")
    print("="*60)

    # Initialize GPU manager
    gpu_manager = GPUManager()
    device = gpu_manager.get_device()
    print(f"Using device: {device}")

    # Create models
    print("\nCreating models...")
    acoustic_model = VoiceTransformer(
        input_dim=80,
        hidden_dim=256,
        num_layers=4,
        num_heads=4
    ).to(device)

    vocoder = Vocoder(
        input_dim=80,
        hidden_dim=128,
        num_layers=10
    ).to(device)

    print(f"Acoustic model parameters: {sum(p.numel() for p in acoustic_model.parameters()):,}")
    print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")

    # Generate random mel spectrogram (as placeholder for text-to-mel)
    print("\nGenerating sample features...")
    batch_size = 1
    seq_length = 100
    feature_dim = 80

    input_features = torch.randn(batch_size, seq_length, feature_dim).to(device)

    # Generate acoustic features
    print("Running acoustic model...")
    with torch.no_grad():
        acoustic_output = acoustic_model(input_features)

    # Generate waveform
    print("Running vocoder...")
    with torch.no_grad():
        # Transpose for vocoder input
        vocoder_input = acoustic_output.transpose(1, 2)
        waveform = vocoder(vocoder_input)

    print(f"\nGenerated waveform shape: {waveform.shape}")
    print(f"Duration: {waveform.shape[-1] / 22050:.2f} seconds (at 22.05 kHz)")

    return waveform

def example_voice_analysis():
    """Example of voice analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Voice Analysis")
    print("="*60)

    # Initialize components
    audio_processor = AudioProcessor(sample_rate=44100)
    voice_analyzer = VoiceAnalyzer(sample_rate=44100)

    # Generate synthetic audio for demonstration
    print("\nGenerating synthetic audio for analysis...")
    duration = 2.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Create a complex waveform with harmonics
    fundamental_freq = 220  # A3 note
    audio = np.sin(2 * np.pi * fundamental_freq * t)
    audio += 0.5 * np.sin(2 * np.pi * fundamental_freq * 2 * t)  # 2nd harmonic
    audio += 0.3 * np.sin(2 * np.pi * fundamental_freq * 3 * t)  # 3rd harmonic

    # Add some vibrato
    vibrato_freq = 5  # Hz
    vibrato_depth = 10  # Hz
    frequency_modulation = fundamental_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
    audio = np.sin(2 * np.pi * np.cumsum(frequency_modulation) / sample_rate)

    # Convert to tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    # Analyze voice characteristics
    print("\nAnalyzing voice characteristics...")
    characteristics = voice_analyzer.analyze_voice_characteristics(audio_tensor)

    print("\nVoice Analysis Results:")
    print("-" * 40)
    print(f"Pitch Statistics:")
    print(f"  Mean pitch: {characteristics['pitch_statistics']['mean']:.2f} Hz")
    print(f"  Pitch range: {characteristics['pitch_statistics']['range']:.2f} Hz")

    print(f"\nTimbre:")
    print(f"  Brightness: {characteristics['timbre']['brightness']:.3f}")
    print(f"  Spectral contrast: {characteristics['timbre']['spectral_contrast']:.3f}")

    print(f"\nRhythm:")
    print(f"  Tempo: {characteristics['rhythm']['tempo_bpm']:.1f} BPM")
    print(f"  Regularity: {characteristics['rhythm']['rhythm_regularity']:.3f}")

    print(f"\nVoice Quality:")
    print(f"  SNR: {characteristics['voice_quality']['snr_db']:.1f} dB")
    print(f"  Clarity score: {characteristics['voice_quality']['clarity_score']:.3f}")

def example_configuration():
    """Example of using configuration system."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Configuration Management")
    print("="*60)

    # Create configuration
    config = Config()

    print("\nDefault Configuration:")
    print("-" * 40)
    print(f"Model Configuration:")
    print(f"  Model type: {config.model.model_type}")
    print(f"  Hidden dimension: {config.model.hidden_dim}")
    print(f"  Number of layers: {config.model.num_layers}")
    print(f"  Number of attention heads: {config.model.num_heads}")

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Number of epochs: {config.training.num_epochs}")

    print(f"\nData Configuration:")
    print(f"  Sample rate: {config.data.sample_rate} Hz")
    print(f"  Segment length: {config.data.segment_length} samples")
    print(f"  Number of mel bands: {config.data.n_mels}")

    # Modify configuration
    config.model.hidden_dim = 1024
    config.training.batch_size = 64
    config.training.learning_rate = 5e-4

    # Save configuration
    config_path = "example_config.json"
    config.save(config_path)
    print(f"\n✓ Configuration saved to {config_path}")

    # Load configuration
    loaded_config = Config.from_file(config_path)
    print(f"✓ Configuration loaded from {config_path}")

    # Clean up
    os.remove(config_path)

def example_gpu_memory_management():
    """Example of GPU memory management."""
    print("\n" + "="*60)
    print("EXAMPLE 4: GPU Memory Management")
    print("="*60)

    # Initialize GPU manager
    gpu_manager = GPUManager()

    if not gpu_manager.is_available():
        print("GPU not available. Skipping GPU memory example.")
        return

    print(f"\nGPU Device: {gpu_manager.device}")

    # Get compute capability
    major, minor = gpu_manager.get_compute_capability()
    print(f"Compute Capability: {major}.{minor}")

    # Get memory info
    memory_info = gpu_manager.get_memory_info()
    print(f"\nMemory Information:")
    print(f"  Total: {memory_info['total'] / 1e9:.2f} GB")
    print(f"  Allocated: {memory_info['allocated'] / 1e9:.2f} GB")
    print(f"  Free: {memory_info['free'] / 1e9:.2f} GB")
    print(f"  Cached: {memory_info['cached'] / 1e9:.2f} GB")

    # Calculate optimal batch size
    model_size = 100 * 1024 * 1024  # 100 MB model
    sample_size = 4 * 80 * 1000  # 80 features, 1000 time steps, float32

    optimal_batch_size = gpu_manager.optimize_batch_size(model_size, sample_size)
    print(f"\nOptimal batch size: {optimal_batch_size}")

    # Enable mixed precision
    gpu_manager.enable_mixed_precision()
    print("✓ Mixed precision training enabled")

    # Clear cache
    gpu_manager.clear_cache()
    print("✓ GPU cache cleared")

def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# AutoVoice System Examples")
    print("#"*60)

    try:
        # Run examples
        example_basic_synthesis()
        example_voice_analysis()
        example_configuration()
        example_gpu_memory_management()

        print("\n" + "#"*60)
        print("# All examples completed successfully!")
        print("#"*60)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()