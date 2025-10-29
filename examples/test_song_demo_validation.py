#!/usr/bin/env python3
"""
Validation script for song_conversion_demo.ipynb

This script validates that the notebook functions correctly by testing each component
in the expected execution order, simulating notebook cell execution.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path as the notebook does
sys.path.append('../src')

def test_notebook_simulation():
    """Simulate notebook execution to validate functionality"""
    
    print("=" * 60)
    print("Testing Song Conversion Demo Notebook")
    print("=" * 60)
    
    # Test 1: Check directory structure
    print("\n1. Testing directory structure...")
    data_dir = Path('./data/songs')
    if data_dir.exists():
        print("✓ Data directory exists")
    else:
        print("✗ Data directory missing")
        return False
    
    # Test 2: Test imports (simulate notebook imports)
    print("\n2. Testing imports...")
    try:
        import librosa
        print("✓ librosa imported")
    except ImportError as e:
        print(f"✗ librosa import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
        
    try:
        import soundfile as sf
        print("✓ soundfile imported")
    except ImportError as e:
        print(f"✗ soundfile import failed: {e}")
        return False
    
    # Test 3: Test AutoVoice imports
    print("\n3. Testing AutoVoice imports...")
    try:
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        print("✓ SingingConversionPipeline imported")
    except ImportError as e:
        print(f"✗ SingingConversionPipeline import failed: {e}")
        return False
    
    try:
        from auto_voice.inference.voice_cloner import VoiceCloner
        print("✓ VoiceCloner imported")
    except ImportError as e:
        print(f"✗ VoiceCloner import failed: {e}")
        return False
    
    # Test 4: Test widget imports
    print("\n4. Testing IPython widgets...")
    try:
        import ipywidgets as widgets
        print("✓ ipywidgets imported")
    except ImportError as e:
        print(f"✗ ipywidgets import failed: {e}")
        print("  Note: This is expected if not running in Jupyter")
    
    # Test 5: Create sample song file for testing
    print("\n5. Creating test audio file...")
    try:
        # Create a simple test audio file
        sample_rate = 22050
        duration = 3.0  # 3 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple tone (440 Hz sine wave)
        test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        # Save test file
        test_file = data_dir / "test_song.wav"
        sf.write(test_file, test_audio, sample_rate)
        
        print(f"✓ Test audio file created: {test_file}")
        
        # Test loading the audio
        loaded_audio, loaded_sr = librosa.load(test_file, sr=None)
        if len(loaded_audio) > 0 and loaded_sr == sample_rate:
            print(f"✓ Audio validation successful: {len(loaded_audio)/loaded_sr:.1f}s at {loaded_sr}Hz")
        else:
            print("✗ Audio validation failed")
            return False
            
    except Exception as e:
        print(f"✗ Test audio creation failed: {e}")
        return False
    
    # Test 6: Test path handling (simulate notebook behavior)
    print("\n6. Testing path handling...")
    song_path = str(test_file)
    
    if song_path and os.path.exists(song_path):
        print(f"✓ Song path is valid: {song_path}")
    else:
        print("✗ Song path validation failed")
        return False
    
    # Test 7: Test device detection
    print("\n7. Testing device detection...")
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Device detected: {device}")
    except ImportError:
        device = 'cpu'
        print("✓ Using CPU (torch not available)")
    
    # Test 8: Test component initialization (simplified)
    print("\n8. Testing component initialization...")
    try:
        # Test VoiceCloner initialization
        voice_cloner = VoiceCloner(device=device)
        print("✓ VoiceCloner initialized")
    except Exception as e:
        print(f"✗ VoiceCloner initialization failed: {e}")
        return False
    
    # Test 9: Test pipeline initialization
    try:
        pipeline = SingingConversionPipeline(
            device=device,
            voice_cloner=voice_cloner
        )
        print("✓ SingingConversionPipeline initialized")
    except Exception as e:
        print(f"✗ SingingConversionPipeline initialization failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! The notebook is ready for use.")
    print("=" * 60)
    
    print("\nUsage Instructions:")
    print("1. Open the notebook: examples/song_conversion_demo.ipynb")
    print("2. Run cells in order")
    print("3. In the 'Song Input Setup' cell, either:")
    print("   - Upload a song file using the widget")
    print("   - Click 'Download Sample Song' to get a test file")
    print("4. Continue with the remaining cells")
    
    return True

if __name__ == "__main__":
    success = test_notebook_simulation()
    sys.exit(0 if success else 1)