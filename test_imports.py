#!/usr/bin/env python3
"""Test script to verify all AutoVoice modules can be imported."""

import sys
import importlib

def test_imports():
    """Test importing all AutoVoice modules."""
    modules_to_test = [
        'auto_voice.gpu',
        'auto_voice.gpu.gpu_manager',
        'auto_voice.gpu.memory_pool',
        'auto_voice.gpu.kernel_launcher',
        'auto_voice.audio',
        'auto_voice.audio.processor',
        'auto_voice.audio.features',
        'auto_voice.audio.voice_analyzer',
        'auto_voice.models',
        'auto_voice.models.voice_transformer',
        'auto_voice.models.vocoder',
        'auto_voice.models.encoder',
        'auto_voice.training',
        'auto_voice.training.trainer',
        'auto_voice.training.dataset',
        'auto_voice.training.losses',
        'auto_voice.inference',
        'auto_voice.inference.inference_engine',
        'auto_voice.inference.voice_synthesizer',
        'auto_voice.web',
        'auto_voice.web.app',
        'auto_voice.web.api',
        'auto_voice.utils',
        'auto_voice.utils.config',
        'auto_voice.utils.metrics',
        'auto_voice.utils.data_utils',
    ]

    print("Testing AutoVoice module imports...")
    print("-" * 50)

    failed_imports = []
    successful_imports = []

    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            successful_imports.append(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
            print(f"✗ {module_name}: {e}")
        except Exception as e:
            failed_imports.append((module_name, str(e)))
            print(f"✗ {module_name}: Unexpected error: {e}")

    print("-" * 50)
    print(f"\nResults:")
    print(f"  Successful: {len(successful_imports)}/{len(modules_to_test)}")
    print(f"  Failed: {len(failed_imports)}/{len(modules_to_test)}")

    if failed_imports:
        print("\nFailed imports:")
        for module_name, error in failed_imports:
            print(f"  - {module_name}: {error}")
        return False

    print("\n✓ All modules imported successfully!")
    return True

def test_core_functionality():
    """Test basic functionality of core components."""
    print("\nTesting core functionality...")
    print("-" * 50)

    try:
        # Test GPU manager
        from auto_voice.gpu import GPUManager
        gpu_manager = GPUManager()
        print(f"✓ GPU Manager initialized (device: {gpu_manager.device})")

        # Test audio processor
        from auto_voice.audio import AudioProcessor
        audio_processor = AudioProcessor()
        print("✓ Audio Processor initialized")

        # Test config
        from auto_voice.utils import Config
        config = Config()
        print("✓ Config initialized")

        # Test model creation
        from auto_voice.models import VoiceTransformer
        model = VoiceTransformer()
        print(f"✓ VoiceTransformer model created ({sum(p.numel() for p in model.parameters())} parameters)")

        print("\n✓ Core functionality test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Core functionality test failed: {e}")
        return False

if __name__ == "__main__":
    # Add src directory to path
    sys.path.insert(0, 'src')

    # Run tests
    import_success = test_imports()

    if import_success:
        functionality_success = test_core_functionality()
        if functionality_success:
            print("\n✅ All tests passed successfully!")
            sys.exit(0)

    print("\n❌ Some tests failed. Please check the errors above.")
    sys.exit(1)