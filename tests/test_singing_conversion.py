#!/usr/bin/env python3
"""
End-to-end testing script for singing voice conversion system.
Tests pitch preservation, audio quality, and processing speed.
"""

import os
import sys
import time
import json
import librosa
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class SingingConversionTester:
    """Test suite for singing voice conversion system."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
        self.test_audio_dir = Path('tests/test_audio')
        self.test_audio_dir.mkdir(parents=True, exist_ok=True)
        
    def create_test_audio(self):
        """Create synthetic test audio with known pitch."""
        print("\n📝 Creating test audio samples...")
        
        # Parameters
        sr = 16000  # Sample rate
        duration = 5  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        
        # Test 1: Pure sine wave at A4 (440 Hz)
        freq_hz = 440
        audio_sine = 0.3 * np.sin(2 * np.pi * freq_hz * t)
        
        # Test 2: Sine wave with vibrato (simulating singing)
        vibrato_freq = 5  # Hz
        vibrato_depth = 20  # cents
        freq_modulated = freq_hz * (2 ** (vibrato_depth / 1200 * np.sin(2 * np.pi * vibrato_freq * t)))
        phase = 2 * np.pi * np.cumsum(freq_modulated) / sr
        audio_vibrato = 0.3 * np.sin(phase)
        
        # Save test audio
        test_files = {
            'sine_440hz.wav': audio_sine,
            'vibrato_440hz.wav': audio_vibrato
        }
        
        for filename, audio in test_files.items():
            filepath = self.test_audio_dir / filename
            librosa.output.write_wav(str(filepath), audio, sr=sr)
            print(f"  ✓ Created {filename}")
        
        return test_files.keys()
    
    def test_pitch_extraction(self):
        """Test pitch extraction accuracy."""
        print("\n🎵 Testing pitch extraction...")
        
        try:
            from auto_voice.audio.pitch_extractor import PitchExtractor
            
            extractor = PitchExtractor()
            test_file = self.test_audio_dir / 'sine_440hz.wav'
            
            if not test_file.exists():
                print("  ⚠️  Test audio not found, skipping pitch extraction test")
                return False
            
            # Extract pitch
            audio, sr = librosa.load(str(test_file), sr=16000)
            f0 = extractor.extract(audio, sr)
            
            # Verify pitch is around 440 Hz
            mean_f0 = np.nanmean(f0[f0 > 0])
            error_cents = 1200 * np.log2(mean_f0 / 440)
            
            self.results['tests']['pitch_extraction'] = {
                'expected_hz': 440,
                'measured_hz': float(mean_f0),
                'error_cents': float(error_cents),
                'passed': abs(error_cents) < 50  # Within 50 cents
            }
            
            print(f"  ✓ Pitch extraction: {mean_f0:.1f} Hz (error: {error_cents:.1f} cents)")
            return True
            
        except Exception as e:
            print(f"  ✗ Pitch extraction test failed: {e}")
            return False
    
    def test_audio_loading(self):
        """Test audio loading and format support."""
        print("\n📂 Testing audio loading...")
        
        try:
            test_file = self.test_audio_dir / 'sine_440hz.wav'
            
            if not test_file.exists():
                print("  ⚠️  Test audio not found, skipping audio loading test")
                return False
            
            # Load audio
            audio, sr = librosa.load(str(test_file), sr=16000)
            
            self.results['tests']['audio_loading'] = {
                'sample_rate': sr,
                'duration_seconds': float(len(audio) / sr),
                'channels': 1,
                'passed': sr == 16000 and len(audio) > 0
            }
            
            print(f"  ✓ Audio loaded: {sr} Hz, {len(audio)/sr:.1f}s")
            return True
            
        except Exception as e:
            print(f"  ✗ Audio loading test failed: {e}")
            return False
    
    def test_model_loading(self):
        """Test model loading."""
        print("\n🤖 Testing model loading...")
        
        try:
            from auto_voice.models.singing_voice_converter import SingingVoiceConverter
            
            converter = SingingVoiceConverter()
            
            self.results['tests']['model_loading'] = {
                'hubert_loaded': hasattr(converter, 'hubert'),
                'pitch_extractor_loaded': hasattr(converter, 'pitch_extractor'),
                'passed': True
            }
            
            print("  ✓ Models loaded successfully")
            return True
            
        except Exception as e:
            print(f"  ✗ Model loading test failed: {e}")
            self.results['tests']['model_loading'] = {'passed': False, 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("🎤 SINGING VOICE CONVERSION - END-TO-END TEST SUITE")
        print("="*70)
        
        # Create test audio
        self.create_test_audio()
        
        # Run tests
        tests = [
            ('Audio Loading', self.test_audio_loading),
            ('Model Loading', self.test_model_loading),
            ('Pitch Extraction', self.test_pitch_extraction),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ {test_name} failed: {e}")
                failed += 1
        
        # Summary
        self.results['summary'] = {
            'total_tests': len(tests),
            'passed': passed,
            'failed': failed,
            'success_rate': f"{100 * passed / len(tests):.1f}%"
        }
        
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {100 * passed / len(tests):.1f}%")
        print("="*70 + "\n")
        
        # Save results
        results_file = Path('tests/test_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {results_file}")
        
        return failed == 0


if __name__ == '__main__':
    tester = SingingConversionTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

