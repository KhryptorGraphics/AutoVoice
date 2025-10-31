#!/usr/bin/env python3
"""Validate AutoVoice installation and dependencies.

Checks:
1. Python version compatibility
2. PyTorch installation and CUDA availability
3. Core dependencies
4. Pre-trained models
5. Import tests

Usage:
    python scripts/validate_installation.py
"""

import sys
import os
from pathlib import Path


def print_section(title: str):
    print("\n" + "="*70)
    print(title)
    print("="*70)


def check_python_version():
    """Check Python version compatibility."""
    print_section("1. Python Version")
    
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 13):
        print("⚠️  Python 3.13: Requires PyTorch 2.7+ (experimental)")
        return True
    elif version >= (3, 12):
        print("✅ Python 3.12: Recommended (stable with PyTorch 2.5+)")
        return True
    elif version >= (3, 8):
        print("✅ Python 3.8-3.11: Supported")
        return True
    else:
        print("❌ Python < 3.8: Not supported")
        return False


def check_pytorch():
    """Check PyTorch installation."""
    print_section("2. PyTorch")
    
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
            print("   For GPU support, install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        # Version compatibility
        major, minor = torch.__version__.split('.')[:2]
        version_tuple = (int(major), int(minor))
        
        if version_tuple >= (2, 7):
            print("✅ PyTorch 2.7+: Full Python 3.13 support")
        elif version_tuple >= (2, 5):
            print("✅ PyTorch 2.5+: Recommended for Python 3.12")
        else:
            print("⚠️  PyTorch < 2.5: Consider upgrading")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("\nInstall with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_dependencies():
    """Check core dependencies."""
    print_section("3. Core Dependencies")
    
    required = {
        'numpy': 'Numerical computing',
        'librosa': 'Audio processing',
        'soundfile': 'Audio I/O',
        'scipy': 'Scientific computing',
        'flask': 'Web framework',
        'demucs': 'Vocal separation',
        'resemblyzer': 'Speaker encoding',
    }
    
    optional = {
        'torchcrepe': 'Pitch extraction (GPU)',
        'crepe': 'Pitch extraction (CPU fallback)',
        'tensorrt': 'Optimized inference',
        'onnx': 'Model export',
    }
    
    all_passed = True
    
    # Check required
    print("\nRequired packages:")
    for package, description in required.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ❌ {package}: {description} - MISSING")
            all_passed = False
    
    # Check optional
    print("\nOptional packages:")
    for package, description in optional.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ⚠️  {package}: {description} - not installed")
    
    return all_passed


def check_models():
    """Check pre-trained models."""
    print_section("4. Pre-trained Models")
    
    models_dir = Path('models/pretrained')
    
    required_models = {
        'sovits5.0_main_1500.pth': 'So-VITS-SVC 5.0 (300 MB)',
        'hifigan_ljspeech.ckpt': 'HiFi-GAN vocoder (40 MB)',
        'hubert-soft-0d54a1f4.pt': 'HuBERT-Soft encoder (95 MB)',
    }
    
    optional_models = {
        'G_0.pth': 'So-VITS-SVC 4.0 Generator (fallback)',
        'D_0.pth': 'So-VITS-SVC 4.0 Discriminator (fallback)',
    }
    
    all_required = True
    
    print(f"\nModels directory: {models_dir.absolute()}")
    
    if not models_dir.exists():
        print(f"\n❌ Models directory not found")
        print("\nDownload models with:")
        print("  python scripts/download_pretrained_models.py")
        return False
    
    # Check required
    print("\nRequired models:")
    for filename, description in required_models.items():
        path = models_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename}: {description} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {filename}: {description} - MISSING")
            all_required = False
    
    # Check optional
    print("\nOptional models:")
    for filename, description in optional_models.items():
        path = models_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename}: {description} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  {filename}: {description} - not downloaded")
    
    if not all_required:
        print("\nDownload missing models with:")
        print("  python scripts/download_pretrained_models.py")
    
    return all_required


def check_imports():
    """Test importing AutoVoice modules."""
    print_section("5. AutoVoice Modules")
    
    sys.path.insert(0, 'src')
    
    modules = [
        'auto_voice.models.singing_voice_converter',
        'auto_voice.inference.singing_conversion_pipeline',
        'auto_voice.inference.voice_cloner',
        'auto_voice.audio.pitch_extractor',
        'auto_voice.audio.source_separator',
    ]
    
    all_passed = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {str(e)[:50]}")
            all_passed = False
    
    return all_passed


def main():
    print("="*70)
    print("AutoVoice Installation Validator")
    print("="*70)
    
    results = {
        'Python Version': check_python_version(),
        'PyTorch': check_pytorch(),
        'Dependencies': check_dependencies(),
        'Pre-trained Models': check_models(),
        'Module Imports': check_imports(),
    }
    
    print_section("Summary")
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 All checks passed! Installation is complete.")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run demo: python examples/demo_voice_conversion.py --help")
        print("  2. Start web UI: python main.py")
        print("  3. Read guide: docs/QUICK_START_GUIDE.md")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  Installation incomplete. Please fix the issues above.")
        print("="*70)
        print("\nCommon fixes:")
        print("  - Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Download models: python scripts/download_pretrained_models.py")
        return 1


if __name__ == '__main__':
    sys.exit(main())
