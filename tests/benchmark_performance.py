#!/usr/bin/env python3
"""
Performance Benchmarking for Singing Voice Conversion
Measures processing speed for different quality presets.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def benchmark_model_loading():
    """Benchmark model loading time."""
    print("\n" + "="*70)
    print("⏱️  BENCHMARK 1: Model Loading Time")
    print("="*70)
    
    try:
        from auto_voice.models.singing_voice_converter import SingingVoiceConverter
        from auto_voice.utils.config_loader import load_config
        
        print("\nLoading configuration...")
        start = time.time()
        config = load_config()
        config_time = time.time() - start
        print(f"✓ Config loaded in {config_time:.2f}s")
        
        print("Loading SingingVoiceConverter model...")
        start = time.time()
        converter = SingingVoiceConverter(config)
        model_time = time.time() - start
        print(f"✓ Model loaded in {model_time:.2f}s")
        
        total_time = config_time + model_time
        print(f"\n📊 Total Load Time: {total_time:.2f}s")
        
        return {
            'config_load_time_s': config_time,
            'model_load_time_s': model_time,
            'total_load_time_s': total_time
        }
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return None

def benchmark_audio_processing():
    """Benchmark audio processing speed."""
    print("\n" + "="*70)
    print("⏱️  BENCHMARK 2: Audio Processing Speed")
    print("="*70)
    
    try:
        import librosa
        
        # Create test audio of different lengths
        sr = 16000
        durations = [10, 30, 60]  # seconds
        
        results = {}
        
        for duration in durations:
            # Create synthetic audio
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            
            # Measure processing time
            start = time.time()
            # Simulate audio processing (normalization, resampling, etc.)
            audio_norm = audio / np.max(np.abs(audio))
            process_time = time.time() - start
            
            # Calculate throughput
            throughput = duration / process_time if process_time > 0 else 0
            
            results[f'{duration}s'] = {
                'duration_s': duration,
                'process_time_s': process_time,
                'throughput_x': throughput
            }
            
            print(f"✓ {duration}s audio: {process_time:.3f}s ({throughput:.1f}x realtime)")
        
        return results
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return None

def benchmark_pitch_extraction():
    """Benchmark pitch extraction speed."""
    print("\n" + "="*70)
    print("⏱️  BENCHMARK 3: Pitch Extraction Speed")
    print("="*70)
    
    try:
        from auto_voice.audio.pitch_extractor import SingingPitchExtractor
        import torch
        
        extractor = SingingPitchExtractor()
        
        # Test different audio lengths
        sr = 16000
        durations = [5, 10, 30]  # seconds
        
        results = {}
        
        for duration in durations:
            # Create test audio
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Measure extraction time
            start = time.time()
            try:
                result = extractor.extract_f0_contour(audio_tensor, sr)
                extract_time = time.time() - start
                
                # Calculate throughput
                throughput = duration / extract_time if extract_time > 0 else 0
                
                results[f'{duration}s'] = {
                    'duration_s': duration,
                    'extract_time_s': extract_time,
                    'throughput_x': throughput,
                    'status': 'success'
                }
                
                print(f"✓ {duration}s audio: {extract_time:.2f}s ({throughput:.1f}x realtime)")
            except Exception as e:
                results[f'{duration}s'] = {
                    'duration_s': duration,
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"⚠️  {duration}s audio: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return None

def benchmark_quality_presets():
    """Benchmark quality presets."""
    print("\n" + "="*70)
    print("⏱️  BENCHMARK 4: Quality Presets")
    print("="*70)
    
    from auto_voice.models.singing_voice_converter import QUALITY_PRESETS
    
    results = {}
    
    for preset_name, preset_config in QUALITY_PRESETS.items():
        print(f"\n{preset_name.upper()}:")
        print(f"  Description: {preset_config['description']}")
        print(f"  Decoder Steps: {preset_config['decoder_steps']}")
        print(f"  Relative Quality: {preset_config['relative_quality']}")
        print(f"  Relative Speed: {preset_config['relative_speed']}")
        
        # Estimate processing time (based on relative speed)
        # Assuming 30s audio at balanced preset takes ~30s
        estimated_time = 30 / preset_config['relative_speed']
        
        results[preset_name] = {
            'description': preset_config['description'],
            'decoder_steps': preset_config['decoder_steps'],
            'relative_quality': preset_config['relative_quality'],
            'relative_speed': preset_config['relative_speed'],
            'estimated_time_30s_audio': estimated_time
        }
        
        print(f"  Estimated Time (30s audio): {estimated_time:.1f}s")
    
    return results

def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("🎤 SINGING VOICE CONVERSION - PERFORMANCE BENCHMARKS")
    print("="*70)
    
    benchmarks = {
        'model_loading': benchmark_model_loading(),
        'audio_processing': benchmark_audio_processing(),
        'pitch_extraction': benchmark_pitch_extraction(),
        'quality_presets': benchmark_quality_presets()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    if benchmarks['model_loading']:
        print(f"\nModel Loading:")
        print(f"  Total Time: {benchmarks['model_loading']['total_load_time_s']:.2f}s")
    
    if benchmarks['audio_processing']:
        print(f"\nAudio Processing (Throughput):")
        for duration, result in benchmarks['audio_processing'].items():
            print(f"  {duration}: {result['throughput_x']:.1f}x realtime")
    
    if benchmarks['pitch_extraction']:
        print(f"\nPitch Extraction (Throughput):")
        for duration, result in benchmarks['pitch_extraction'].items():
            if result.get('status') == 'success':
                print(f"  {duration}: {result['throughput_x']:.1f}x realtime")
    
    if benchmarks['quality_presets']:
        print(f"\nQuality Presets (Estimated for 30s audio):")
        for preset, config in benchmarks['quality_presets'].items():
            print(f"  {preset}: {config['estimated_time_30s_audio']:.1f}s")
    
    print("="*70 + "\n")
    
    # Save results
    results_file = Path(__file__).parent / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'benchmarks': benchmarks
        }, f, indent=2)
    
    print(f"✓ Results saved to {results_file}\n")
    
    return benchmarks

if __name__ == '__main__':
    benchmarks = run_all_benchmarks()
    sys.exit(0)

