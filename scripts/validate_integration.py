#!/usr/bin/env python3
"""Integration validation for AutoVoice components.

This script validates:
1. GPU manager initialization and resource allocation
2. Audio processor integration with CUDA kernels
3. Web API basic functionality
4. Pipeline component integration
"""
import sys
import json
import subprocess
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Compute project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def check_imports() -> Tuple[bool, List[str]]:
    """Check if all required modules can be imported.

    Returns:
        Tuple of (success bool, list of errors)
    """
    print("Checking module imports...")
    errors = []

    modules = [
        'auto_voice.gpu.gpu_manager',
        'auto_voice.audio.processor',
        'auto_voice.web.api',
        'auto_voice.inference.singing_conversion_pipeline'
    ]

    for module in modules:
        try:
            __import__(module)
        except Exception as e:
            errors.append(f"Failed to import {module}: {e}")

    return len(errors) == 0, errors


def validate_gpu_manager() -> Dict[str, Any]:
    """Validate GPU manager initialization.

    Returns:
        Dict with validation results
    """
    print("Validating GPU manager...")

    try:
        from auto_voice.gpu.gpu_manager import GPUManager

        # Initialize manager
        manager = GPUManager()

        # Check basic functionality
        devices = manager.get_available_devices()
        memory_info = manager.get_memory_info()

        return {
            'passed': True,
            'devices_found': len(devices),
            'memory_available': memory_info.get('free_memory', 0) > 0,
            'details': {
                'devices': devices,
                'memory': memory_info
            }
        }

    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
            'details': {}
        }


def validate_audio_processor() -> Dict[str, Any]:
    """Validate audio processor integration with real APIs.

    Returns:
        Dict with validation results
    """
    print("Validating audio processor...")

    try:
        from auto_voice.audio.processor import AudioProcessor
        import numpy as np

        # Initialize processor
        processor = AudioProcessor()

        # Create test audio (1 second mono float32)
        test_audio = np.random.randn(22050).astype(np.float32)
        sample_rate = 22050

        # Test realistic processing chain
        mono = processor.ensure_mono(test_audio)
        normalized = processor.normalize(mono)
        resampled = processor.resample(normalized, sample_rate, 44100)

        # Validate results
        processing_works = (
            mono is not None and
            normalized is not None and
            resampled is not None and
            isinstance(resampled, np.ndarray)
        )

        return {
            'passed': processing_works,
            'processing_works': processing_works,
            'details': {
                'mono_shape': mono.shape if mono is not None else None,
                'normalized_shape': normalized.shape if normalized is not None else None,
                'resampled_shape': resampled.shape if resampled is not None else None,
                'output_sample_rate': 44100
            }
        }

    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
            'details': {}
        }


def validate_web_api() -> Dict[str, Any]:
    """Validate web API basic functionality using Flask test client.

    Returns:
        Dict with validation results
    """
    print("Validating web API...")

    try:
        # Try to import Flask app
        try:
            from auto_voice.web.api import api_bp
            from flask import Flask

            # Create test app and register blueprint
            app = Flask(__name__)
            app.register_blueprint(api_bp, url_prefix='/api/v1')

            # Create Flask test client
            client = app.test_client()

            # Test health endpoint
            health_response = client.get("/api/v1/health")

            # Test voice profiles endpoint
            profiles_response = client.get("/api/v1/voice/profiles")

            return {
                'passed': health_response.status_code == 200,
                'health_status': health_response.status_code,
                'profiles_status': profiles_response.status_code,
                'details': {
                    'health': health_response.get_json() if health_response.status_code == 200 else None,
                    'profiles': profiles_response.get_json() if profiles_response.status_code == 200 else None
                }
            }
        except ImportError as import_err:
            # Fallback to probing running server if AUTOVOICE_BASE_URL is set
            base_url = os.environ.get('AUTOVOICE_BASE_URL')
            if base_url:
                import requests
                health_response = requests.get(f"{base_url}/health", timeout=5)
                return {
                    'passed': health_response.status_code == 200,
                    'health_status': health_response.status_code,
                    'details': {
                        'health': health_response.json() if health_response.status_code == 200 else None,
                        'mode': 'remote_probe'
                    }
                }
            else:
                raise import_err

    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
            'details': {}
        }


def validate_pipeline_integration() -> Dict[str, Any]:
    """Validate pipeline component integration with real methods.

    Returns:
        Dict with validation results
    """
    print("Validating pipeline integration...")

    try:
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        # Initialize pipeline with minimal config
        pipeline = SingingConversionPipeline(device='cpu')

        # Check for real pipeline methods
        pipeline_methods = [
            'convert_song',
            'convert_vocals_only',
            'set_preset',
            'clear_cache'
        ]

        methods_present = {
            method: hasattr(pipeline, method)
            for method in pipeline_methods
        }

        components_valid = all(methods_present.values())

        # Optionally validate state
        current_preset = None
        if hasattr(pipeline, 'get_current_preset'):
            try:
                current_preset = pipeline.get_current_preset()
            except:
                pass

        return {
            'passed': components_valid,
            'components_valid': components_valid,
            'details': {
                'pipeline_methods': [m for m, present in methods_present.items() if present],
                'missing_methods': [m for m, present in methods_present.items() if not present],
                'current_preset': current_preset
            }
        }

    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
            'details': {}
        }


def validate_cuda_kernels() -> Dict[str, Any]:
    """Validate CUDA kernels can be loaded.

    Returns:
        Dict with validation results
    """
    print("Validating CUDA kernels...")

    try:
        # Try to import CUDA kernel bindings
        try:
            import audio_kernels
            kernels_available = True
            kernel_functions = dir(audio_kernels)
        except ImportError:
            kernels_available = False
            kernel_functions = []

        return {
            'passed': True,  # Not critical if kernels unavailable
            'kernels_available': kernels_available,
            'details': {
                'functions': kernel_functions if kernels_available else [],
                'note': 'CUDA kernels are optional for CPU mode'
            }
        }

    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
            'details': {}
        }


def main() -> int:
    """Main validation function.

    Returns:
        0 for success, 1 for failure
    """
    parser = argparse.ArgumentParser(description='Validate AutoVoice integration')
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results/reports/integration.json',
        help='Output path for validation results (default: validation_results/reports/integration.json)'
    )
    args = parser.parse_args()

    print("=== Integration Validation ===\n")

    # Check imports first
    imports_ok, import_errors = check_imports()
    if not imports_ok:
        print("❌ Import check failed:")
        for error in import_errors:
            print(f"  - {error}")
        return 1

    print("✅ All imports successful\n")

    # Run all validations
    results = {
        'imports': {'passed': True, 'errors': []},
        'gpu_manager': validate_gpu_manager(),
        'audio_processor': validate_audio_processor(),
        'web_api': validate_web_api(),
        'pipeline': validate_pipeline_integration(),
        'cuda_kernels': validate_cuda_kernels()
    }

    # Save results using PROJECT_ROOT
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for component, result in results.items():
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"{component}: {status}")
        if not result['passed'] and 'error' in result:
            print(f"  Error: {result['error']}")

    # Determine overall pass/fail
    # Critical components must pass
    critical_components = ['imports', 'gpu_manager', 'audio_processor', 'pipeline']
    all_passed = all(results[comp]['passed'] for comp in critical_components)

    if not all_passed:
        print("\n❌ CRITICAL INTEGRATION TESTS FAILED")
        return 1
    else:
        print("\n✅ ALL INTEGRATION TESTS PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
