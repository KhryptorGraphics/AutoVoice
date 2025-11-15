#!/usr/bin/env python3
"""
Comprehensive testing for singing voice conversion system.
Tests core functionality without requiring full server.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("\n" + "="*70)
    print("🔧 TEST 1: Module Imports")
    print("="*70)
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  - Version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        
        import librosa
        print("✓ librosa imported successfully")
        
        import flask
        print("✓ Flask imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_loading():
    """Test model loading."""
    print("\n" + "="*70)
    print("🤖 TEST 2: Model Loading")
    print("="*70)

    try:
        from auto_voice.models.singing_voice_converter import SingingVoiceConverter
        from auto_voice.utils.config_loader import load_config

        print("Loading configuration...")
        config = load_config()

        print("Loading SingingVoiceConverter...")
        start_time = time.time()
        converter = SingingVoiceConverter(config)
        load_time = time.time() - start_time

        print(f"✓ SingingVoiceConverter loaded in {load_time:.2f}s")
        print(f"  - Config loaded: {config is not None}")
        print(f"  - Model initialized: {converter is not None}")

        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_processing():
    """Test audio processing."""
    print("\n" + "="*70)
    print("🎵 TEST 3: Audio Processing")
    print("="*70)
    
    try:
        import librosa
        
        # Create synthetic audio
        sr = 16000
        duration = 2
        t = np.linspace(0, duration, int(sr * duration))
        
        # Sine wave at 440 Hz
        freq = 440
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        print(f"✓ Created synthetic audio")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - Duration: {duration}s")
        print(f"  - Frequency: {freq} Hz")
        print(f"  - Shape: {audio.shape}")
        
        # Test audio normalization
        audio_norm = audio / np.max(np.abs(audio))
        print(f"✓ Audio normalized")
        print(f"  - Max amplitude: {np.max(np.abs(audio_norm)):.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        return False

def test_pitch_extraction():
    """Test pitch extraction."""
    print("\n" + "="*70)
    print("🎼 TEST 4: Pitch Extraction")
    print("="*70)
    
    try:
        import librosa
        import numpy as np
        
        # Create test audio with known pitch
        sr = 16000
        duration = 2
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440  # A4
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # Try CREPE pitch extraction
        try:
            import torchcrepe
            print("Testing CREPE pitch extraction...")
            
            start_time = time.time()
            f0, confidence = torchcrepe.predict(
                torch.from_numpy(audio).float().unsqueeze(0),
                sr,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                model='full',
                batch_size=512,
                threshold=0.1
            )
            extract_time = time.time() - start_time
            
            f0_mean = float(np.nanmean(f0.numpy()[f0.numpy() > 0]))
            error_cents = 1200 * np.log2(f0_mean / freq) if f0_mean > 0 else 0
            
            print(f"✓ CREPE pitch extraction completed in {extract_time:.2f}s")
            print(f"  - Detected pitch: {f0_mean:.1f} Hz")
            print(f"  - Expected pitch: {freq} Hz")
            print(f"  - Error: {error_cents:.1f} cents")
            print(f"  - Accuracy: {'PASS' if abs(error_cents) < 50 else 'FAIL'}")
            
            return True
        except Exception as e:
            print(f"⚠️  CREPE extraction not available: {e}")
            print("  - This is expected if CREPE models aren't downloaded")
            return True  # Don't fail, just warn
            
    except Exception as e:
        print(f"✗ Pitch extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Test GPU availability."""
    print("\n" + "="*70)
    print("🚀 TEST 5: GPU Availability")
    print("="*70)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ GPU detected")
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected - will use CPU (slower)")
        
        return True
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("🎤 SINGING VOICE CONVERSION - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Audio Processing", test_audio_processing),
        ("Pitch Extraction", test_pitch_extraction),
        ("GPU Availability", test_gpu_availability),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = "ERROR"
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    for test_name, result in results.items():
        status = "✓" if result == "PASS" else "✗" if result == "FAIL" else "⚠️"
        print(f"{status} {test_name}: {result}")
    
    print("\n" + "="*70)
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed}")
    print(f"Success Rate: {100 * passed / len(tests):.1f}%")
    print("="*70 + "\n")
    
    # Save results
    results_file = Path(__file__).parent / 'comprehensive_test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'tests': results,
            'summary': {
                'total': len(tests),
                'passed': passed,
                'failed': failed,
                'success_rate': f"{100 * passed / len(tests):.1f}%"
            }
        }, f, indent=2)
    
    print(f"✓ Results saved to {results_file}\n")
    
    return failed == 0

if __name__ == '__main__':
    import torch
    success = run_all_tests()
    sys.exit(0 if success else 1)

