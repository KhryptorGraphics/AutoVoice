"""Test basic imports for AutoVoice modules"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        # Test main module imports
        from auto_voice import (
            create_app,
            load_config,
            GPUManager,
            AudioProcessor,
            VoiceModel,
            VoiceTrainer,
            VoiceSynthesizer
        )
        print("✓ All main modules imported successfully")
        
        # Test individual module imports
        from auto_voice.web import create_app, run_server
        print("✓ Web module imported successfully")
        
        from auto_voice.utils import load_config
        print("✓ Utils module imported successfully")
        
        from auto_voice.gpu import GPUManager
        print("✓ GPU module imported successfully")
        
        from auto_voice.audio import AudioProcessor
        print("✓ Audio module imported successfully")
        
        from auto_voice.models import VoiceModel
        print("✓ Models module imported successfully")
        
        from auto_voice.training import VoiceTrainer
        print("✓ Training module imported successfully")
        
        from auto_voice.inference import VoiceSynthesizer
        print("✓ Inference module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
