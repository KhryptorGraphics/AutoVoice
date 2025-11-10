#!/usr/bin/env python3
"""
Example demonstrating model integration with the Auto Voice Cloning system.

This example shows:
1. Using mock models for development
2. Loading real models
3. Integrating with VoiceConversionPipeline
4. Model warmup and caching
5. Custom model configurations
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from auto_voice.models import (
    ModelRegistry,
    ModelConfig,
    ModelType,
    HuBERTModel,
    HiFiGANModel,
)


def example_1_mock_models():
    """Example 1: Using mock models for development."""
    print("\n" + "="*60)
    print("Example 1: Mock Models for Development")
    print("="*60)

    # Initialize registry with mock models
    registry = ModelRegistry(use_mock=True)

    # Load models (instantaneous, no downloads)
    hubert = registry.load_hubert()
    hifigan = registry.load_hifigan()
    speaker_encoder = registry.load_speaker_encoder()

    print(f"✓ Loaded HuBERT: {type(hubert).__name__}")
    print(f"✓ Loaded HiFi-GAN: {type(hifigan).__name__}")
    print(f"✓ Loaded Speaker Encoder: {type(speaker_encoder).__name__}")

    # Test models with mock data
    audio = np.random.randn(16000).astype(np.float32)

    print("\nTesting models with mock data...")

    # HuBERT feature extraction
    features = hubert.extract_features(audio)
    print(f"✓ HuBERT features shape: {features.shape}")

    # Speaker embedding
    embedding = speaker_encoder.encode(audio)
    print(f"✓ Speaker embedding shape: {embedding.shape}")

    # HiFi-GAN synthesis
    mel = np.random.randn(80, 100).astype(np.float32)
    synthesized = hifigan.synthesize(mel)
    print(f"✓ Synthesized audio shape: {synthesized.shape}")


def example_2_real_models():
    """Example 2: Loading real models (requires download)."""
    print("\n" + "="*60)
    print("Example 2: Real Models (Download Required)")
    print("="*60)

    try:
        # Initialize registry for real models
        registry = ModelRegistry(
            model_dir='models/',
            use_mock=False
        )

        print("Models will be downloaded automatically if not present...")

        # Load models (downloads if needed)
        hubert = registry.load_hubert()
        print(f"✓ Loaded real HuBERT model")

        # Test with real model
        audio = np.random.randn(16000).astype(np.float32)
        features = hubert.extract_features(audio)
        print(f"✓ Extracted features: {features.shape}")

    except Exception as e:
        print(f"⚠ Real model loading failed: {e}")
        print("  This is expected if models aren't downloaded yet.")
        print("  Run: python scripts/download_models.py")


def example_3_pipeline_integration():
    """Example 3: Integration with VoiceConversionPipeline."""
    print("\n" + "="*60)
    print("Example 3: Pipeline Integration")
    print("="*60)

    try:
        from auto_voice.inference import (
            VoiceConversionPipeline,
            PipelineConfig
        )

        # Method 1: Pipeline creates its own registry
        config = PipelineConfig(
            use_mock_models=True,  # Use mock for demo
            enable_model_warmup=False
        )
        pipeline1 = VoiceConversionPipeline(config=config)
        print("✓ Pipeline 1: Auto-initialized model registry")

        # Method 2: Provide custom registry
        registry = ModelRegistry(use_mock=True)
        pipeline2 = VoiceConversionPipeline(
            config=config,
            model_registry=registry
        )
        print("✓ Pipeline 2: Custom model registry")

        # Access models through pipeline
        if pipeline1.model_registry:
            hubert = pipeline1.hubert_model
            print(f"✓ Accessed HuBERT through pipeline: {type(hubert).__name__}")

    except ImportError as e:
        print(f"⚠ Pipeline import failed: {e}")
        print("  Pipeline integration requires full installation")


def example_4_model_warmup():
    """Example 4: Model warmup and caching."""
    print("\n" + "="*60)
    print("Example 4: Model Warmup and Caching")
    print("="*60)

    # Initialize registry
    registry = ModelRegistry(use_mock=True)

    # Warmup all models
    print("Warming up all models...")
    registry.warmup_models()
    print(f"✓ Warmed up {len(registry._models)} models")

    # Demonstrate caching
    print("\nDemonstrating model caching:")
    hubert1 = registry.load_hubert()
    hubert2 = registry.load_hubert()

    if hubert1 is hubert2:
        print("✓ Models are cached (same instance returned)")
    else:
        print("✗ Caching issue detected")

    # Clear cache
    registry.clear_cache()
    print("✓ Cache cleared")

    hubert3 = registry.load_hubert()
    if hubert3 is not hubert1:
        print("✓ New instance loaded after cache clear")


def example_5_custom_config():
    """Example 5: Custom model configurations."""
    print("\n" + "="*60)
    print("Example 5: Custom Model Configurations")
    print("="*60)

    # Create custom model config
    custom_config = ModelConfig(
        name='custom_hubert',
        model_type=ModelType.HUBERT,
        version='2.0.0',
        local_path='/path/to/custom/model.pt',
        requires_gpu=True,
        min_memory_gb=8.0,
        metadata={
            'description': 'Custom fine-tuned HuBERT',
            'training_data': 'custom_dataset',
            'sample_rate': 16000
        }
    )

    print(f"Custom config created:")
    print(f"  Name: {custom_config.name}")
    print(f"  Type: {custom_config.model_type.value}")
    print(f"  Version: {custom_config.version}")
    print(f"  GPU Required: {custom_config.requires_gpu}")
    print(f"  Metadata: {custom_config.metadata}")

    # Serialize config
    config_dict = custom_config.to_dict()
    print(f"\n✓ Config serialized to dict: {len(config_dict)} keys")

    # Deserialize
    restored = ModelConfig.from_dict(config_dict)
    print(f"✓ Config restored: {restored.name}")


def example_6_model_info():
    """Example 6: Inspecting model registry."""
    print("\n" + "="*60)
    print("Example 6: Model Registry Information")
    print("="*60)

    registry = ModelRegistry(use_mock=True)

    # List all models
    models = registry.list_models()
    print(f"Available models ({len(models)}):")
    for model_name in models:
        config = registry.get_config(model_name)
        downloaded = "✓" if registry.is_model_downloaded(model_name) else "✗"

        print(f"\n  [{downloaded}] {model_name}")
        print(f"      Type: {config.model_type.value}")
        print(f"      Version: {config.version}")
        print(f"      GPU: {'Yes' if config.requires_gpu else 'No'}")
        print(f"      Memory: {config.min_memory_gb} GB")

        if 'description' in config.metadata:
            print(f"      Description: {config.metadata['description']}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Auto Voice Cloning - Model Integration Examples")
    print("="*60)

    examples = [
        ("Mock Models", example_1_mock_models),
        ("Real Models", example_2_real_models),
        ("Pipeline Integration", example_3_pipeline_integration),
        ("Model Warmup", example_4_model_warmup),
        ("Custom Config", example_5_custom_config),
        ("Model Info", example_6_model_info),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠ Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == '__main__':
    main()
