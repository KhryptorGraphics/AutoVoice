#!/usr/bin/env python3
"""Verify AutoVoice module bindings and CUDA extension."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def check_module(name, import_path):
    """Check if a module imports successfully."""
    try:
        __import__(import_path)
        print(f"  [OK] {name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main():
    print("=== AutoVoice Binding Verification ===\n")

    modules = [
        ("Package root", "auto_voice"),
        ("Audio processor", "auto_voice.audio.processor"),
        ("Audio effects", "auto_voice.audio.effects"),
        ("Audio separation", "auto_voice.audio.separation"),
        ("Voice cloner", "auto_voice.inference.voice_cloner"),
        ("Singing pipeline", "auto_voice.inference.singing_conversion_pipeline"),
        ("Realtime pipeline", "auto_voice.inference.realtime_voice_conversion_pipeline"),
        ("Models - encoder", "auto_voice.models.encoder"),
        ("Models - vocoder", "auto_voice.models.vocoder"),
        ("Models - So-VITS-SVC", "auto_voice.models.so_vits_svc"),
        ("Evaluation metrics", "auto_voice.evaluation.metrics"),
        ("Monitoring", "auto_voice.monitoring.prometheus"),
        ("Training", "auto_voice.training.trainer"),
        ("Storage", "auto_voice.storage.voice_profiles"),
        ("GPU kernels", "auto_voice.gpu.cuda_kernels"),
        ("GPU memory", "auto_voice.gpu.memory_manager"),
        ("Web app", "auto_voice.web.app"),
        ("Web API", "auto_voice.web.api"),
        ("Web utils", "auto_voice.web.utils"),
        ("Job manager", "auto_voice.web.job_manager"),
    ]

    passed = 0
    failed = 0

    print("Module imports:")
    for name, path in modules:
        if check_module(name, path):
            passed += 1
        else:
            failed += 1

    # Check CUDA
    print("\nCUDA status:")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            print(f"  Compute capability: {cap[0]}.{cap[1]}")
    except Exception as e:
        print(f"  PyTorch error: {e}")

    # Check CUDA kernels
    print("\nCUDA kernels:")
    try:
        from auto_voice.gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE
        print(f"  Native kernels: {'available' if CUDA_KERNELS_AVAILABLE else 'fallback (PyTorch)'}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n=== Results: {passed}/{passed+failed} modules OK ===")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
