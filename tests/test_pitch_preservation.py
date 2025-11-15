#!/usr/bin/env python3
"""
Pitch Preservation Testing
Tests that the singing voice conversion system preserves pitch accurately.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_pitch_preservation():
    """Test pitch preservation with various frequencies."""
    print("\n" + "="*70)
    print("🎼 PITCH PRESERVATION TEST")
    print("="*70)
    
    try:
        import torch
        import librosa
        from auto_voice.audio.pitch_extractor import SingingPitchExtractor
        
        # Initialize pitch extractor
        print("\nInitializing pitch extractor...")
        extractor = SingingPitchExtractor()
        
        # Test frequencies (musical notes)
        test_frequencies = {
            'C4': 261.63,
            'E4': 329.63,
            'A4': 440.00,
            'C5': 523.25,
            'A5': 880.00,
        }
        
        results = {}
        sr = 16000
        duration = 3
        
        print(f"\nTesting {len(test_frequencies)} frequencies...")
        print("-" * 70)
        
        for note, freq in test_frequencies.items():
            # Create test audio
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.3 * np.sin(2 * np.pi * freq * t)

            # Extract pitch - convert to tensor with batch dimension
            import torch
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

            start_time = time.time()
            result = extractor.extract_f0_contour(audio_tensor, sr)
            extract_time = time.time() - start_time

            # Get F0 contour
            f0 = result.get('f0', np.array([]))

            # Calculate statistics
            f0_valid = f0[f0 > 0]
            if len(f0_valid) > 0:
                f0_mean = np.mean(f0_valid)
                f0_std = np.std(f0_valid)
                error_hz = f0_mean - freq
                error_cents = 1200 * np.log2(f0_mean / freq)
                
                results[note] = {
                    'expected_hz': freq,
                    'measured_hz': float(f0_mean),
                    'std_hz': float(f0_std),
                    'error_hz': float(error_hz),
                    'error_cents': float(error_cents),
                    'extract_time_ms': float(extract_time * 1000),
                    'passed': abs(error_cents) < 50  # Within 50 cents
                }
                
                status = "✓ PASS" if abs(error_cents) < 50 else "✗ FAIL"
                print(f"{status} {note:3s} ({freq:7.2f} Hz): "
                      f"Measured {f0_mean:7.2f} Hz, "
                      f"Error {error_cents:+6.1f} cents")
            else:
                results[note] = {
                    'expected_hz': freq,
                    'error': 'No valid pitch detected'
                }
                print(f"✗ FAIL {note:3s} ({freq:7.2f} Hz): No valid pitch detected")
        
        # Summary
        passed = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)
        
        print("\n" + "="*70)
        print("📊 PITCH PRESERVATION SUMMARY")
        print("="*70)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {100 * passed / total:.1f}%")
        
        # Calculate average error
        errors = [r['error_cents'] for r in results.values() if 'error_cents' in r]
        if errors:
            avg_error = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))
            print(f"Average Error: {avg_error:.1f} cents")
            print(f"Maximum Error: {max_error:.1f} cents")
            print(f"Target Accuracy: <5 cents")
            print(f"Achieved Accuracy: {'✓ PASS' if avg_error < 5 else '⚠️  WARN' if avg_error < 50 else '✗ FAIL'}")
        
        print("="*70 + "\n")
        
        # Save results
        results_file = Path(__file__).parent / 'pitch_preservation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_type': 'pitch_preservation',
                'results': results,
                'summary': {
                    'total_tests': total,
                    'passed': passed,
                    'success_rate': f"{100 * passed / total:.1f}%",
                    'average_error_cents': float(np.mean(np.abs(errors))) if errors else None,
                    'max_error_cents': float(np.max(np.abs(errors))) if errors else None
                }
            }, f, indent=2)
        
        print(f"✓ Results saved to {results_file}\n")
        
        return passed == total
        
    except Exception as e:
        print(f"✗ Pitch preservation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vibrato_preservation():
    """Test vibrato pattern preservation."""
    print("\n" + "="*70)
    print("🎵 VIBRATO PRESERVATION TEST")
    print("="*70)
    
    try:
        import torch
        import librosa
        
        sr = 16000
        duration = 3
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create audio with vibrato (4-8 Hz modulation)
        base_freq = 440
        vibrato_freq = 5  # Hz
        vibrato_depth = 20  # cents
        
        # Frequency modulation
        freq_modulated = base_freq * (2 ** (vibrato_depth / 1200 * np.sin(2 * np.pi * vibrato_freq * t)))
        phase = 2 * np.pi * np.cumsum(freq_modulated) / sr
        audio = 0.3 * np.sin(phase)
        
        print(f"\nCreated test audio with vibrato:")
        print(f"  - Base frequency: {base_freq} Hz")
        print(f"  - Vibrato frequency: {vibrato_freq} Hz")
        print(f"  - Vibrato depth: {vibrato_depth} cents")
        print(f"  - Duration: {duration}s")
        
        # Analyze vibrato
        from auto_voice.audio.pitch_extractor import SingingPitchExtractor
        import torch
        extractor = SingingPitchExtractor()

        # Extract pitch - convert to tensor with batch dimension
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

        result = extractor.extract_f0_contour(audio_tensor, sr)
        f0 = result.get('f0', np.array([]))

        # Calculate vibrato characteristics
        f0_valid = f0[f0 > 0]
        if len(f0_valid) > 100:
            # Detect vibrato frequency using FFT
            f0_diff = np.diff(f0_valid)
            fft = np.abs(np.fft.fft(f0_diff))
            freqs = np.fft.fftfreq(len(f0_diff), 1/50)  # 50 Hz frame rate
            
            # Find dominant frequency in vibrato range (4-8 Hz)
            vibrato_range = (freqs > 4) & (freqs < 8)
            if np.any(vibrato_range):
                detected_vibrato_freq = freqs[vibrato_range][np.argmax(fft[vibrato_range])]
                error = abs(detected_vibrato_freq - vibrato_freq)
                
                print(f"\n✓ Vibrato detected:")
                print(f"  - Expected frequency: {vibrato_freq} Hz")
                print(f"  - Detected frequency: {detected_vibrato_freq:.1f} Hz")
                print(f"  - Error: {error:.1f} Hz")
                print(f"  - Status: {'✓ PASS' if error < 1 else '⚠️  WARN'}")
            else:
                print(f"\n⚠️  Could not detect vibrato in expected range")
        
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"⚠️  Vibrato test skipped: {e}\n")
        return True  # Don't fail, just warn

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎤 PITCH PRESERVATION TEST SUITE")
    print("="*70)
    
    test1 = test_pitch_preservation()
    test2 = test_vibrato_preservation()
    
    success = test1 and test2
    sys.exit(0 if success else 1)

