#!/usr/bin/env python3
"""Test script to verify AutoVoice startup and enhanced functionality with fixed edge cases."""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")

    try:
        from auto_voice.utils.config_loader import load_config, load_config_with_defaults
        print("✓ config_loader import successful")
    except Exception as e:
        print(f"✗ config_loader import failed: {e}")
        return False

    try:
        from auto_voice.web.app import create_app
        print("✓ web app import successful")
    except Exception as e:
        print(f"✗ web app import failed: {e}")
        return False

    try:
        from auto_voice.web.api import api_bp
        print("✓ API blueprint import successful")
    except Exception as e:
        print(f"✗ API blueprint import failed: {e}")
        return False

    try:
        from auto_voice.web.websocket_handler import WebSocketHandler
        print("✓ WebSocket handler import successful")
    except Exception as e:
        print(f"✗ WebSocket handler import failed: {e}")
        return False

    try:
        from auto_voice.gpu.cuda_manager import CUDAManager
        print("✓ CUDA manager import successful")
    except Exception as e:
        print(f"✗ CUDA manager import failed: {e}")
        return False

    return True

def test_config():
    """Test enhanced config loading with defaults and edge cases."""
    print("\nTesting enhanced config loading...")

    try:
        from auto_voice.utils.config_loader import (
            load_config,
            load_config_with_defaults,
            validate_config,
            merge_configs,
            load_config_from_env
        )

        # Test loading with defaults (no file)
        config = load_config_with_defaults()
        print(f"✓ Default config loaded with keys: {list(config.keys())}")

        # Verify expected sections exist
        expected_sections = ['audio', 'model', 'gpu', 'web', 'logging']
        for section in expected_sections:
            if section not in config:
                print(f"✗ Missing expected section: {section}")
                return False
        print(f"✓ All expected sections present: {expected_sections}")

        # Test config validation
        validate_config(config)
        print("✓ Config validation passed")

        # Test config merging
        base = {'a': {'b': 1}, 'c': 2}
        update = {'a': {'b': 2, 'd': 3}, 'e': 4}
        merged = merge_configs(base, update)
        if (merged['a']['b'] != 2 or merged['a']['d'] != 3 or
            merged['c'] != 2 or merged['e'] != 4):
            print("✗ Config merging failed: incorrect merge results")
            return False
        print("✓ Config merging works correctly")

        # Test environment variable override (double underscore format)
        os.environ['AUTOVOICE_WEB__PORT'] = '8080'
        config_with_env = load_config_from_env(config.copy())  # Use copy to avoid modifying original
        expected_port = 8080
        actual_port = config_with_env.get('web', {}).get('port')
        if actual_port == expected_port:
            print("✓ Environment variable override works (double underscore)")
        else:
            print(f"✗ Environment variable override failed: expected {expected_port}, got {actual_port}")
            return False
        del os.environ['AUTOVOICE_WEB__PORT']

        # Test legacy single underscore format support
        os.environ['AUTOVOICE_WEB_PORT'] = '9000'
        config_with_env_legacy = load_config_from_env(config.copy())
        expected_port_legacy = 9000
        actual_port_legacy = config_with_env_legacy.get('web', {}).get('port')
        if actual_port_legacy == expected_port_legacy:
            print("✓ Environment variable override works (single underscore legacy)")
        else:
            print(f"✗ Legacy environment variable override failed: expected {expected_port_legacy}, got {actual_port_legacy}")
            return False
        del os.environ['AUTOVOICE_WEB_PORT']

        # Test edge cases
        # Empty config validation
        try:
            validate_config({})
            print("✗ Empty config validation should have failed")
            return False
        except ValueError as e:
            print(f"✓ Empty config validation properly failed: {str(e)[:50]}...")

        # Invalid section type validation
        try:
            invalid_config = config.copy()
            invalid_config['audio'] = "not_a_dict"
            validate_config(invalid_config)
            print("✗ Invalid section type validation should have failed")
            return False
        except ValueError as e:
            print(f"✓ Invalid section type validation properly failed: {str(e)[:50]}...")

        # Test loading with missing file (should use defaults)
        config_missing = load_config('non_existent_file.yaml', use_defaults=True)
        if not config_missing:
            print("✗ Loading with missing file should return default config")
            return False
        print("✓ Loading with missing file uses defaults")

        return True
    except Exception as e:
        print(f"✗ Config testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_app():
    """Test enhanced web app creation."""
    print("\nTesting enhanced web app creation...")

    try:
        from auto_voice.web.app import create_app
        from auto_voice.web.api import api_bp

        # Create app with default config
        app, socketio = create_app()
        print("✓ Web app created successfully")

        # Blueprint is already registered in create_app()
        print("✓ API blueprint registered")

        # Test basic routes with test client
        with app.test_client() as client:
            # Test root endpoint - request JSON explicitly
            response = client.get('/', headers={'Accept': 'application/json'})
            if response.status_code == 200:
                data = response.get_json()
                if data and 'message' in data and 'status' in data and 'components' in data:
                    print(f"✓ Root endpoint working: {data['message']}")
                else:
                    print(f"✗ Root endpoint missing expected fields. Data: {data}")
                    return False
            else:
                print(f"✗ Root endpoint failed with status {response.status_code}")
                return False

            # Test health check endpoint
            response = client.get('/health')
            if response.status_code == 200:
                data = response.get_json()
                if data and 'status' in data and 'components' in data:
                    print(f"✓ Health check endpoint working: status={data['status']}")
                else:
                    print(f"✗ Health check missing expected fields. Data: {data}")
                    return False
            else:
                print(f"✗ Health check failed with status {response.status_code}")
                return False

            # Test API health endpoint
            response = client.get('/api/health')
            if response.status_code == 200:
                data = response.get_json()
                if data:
                    print(f"✓ API health endpoint working")
                else:
                    print("✗ API health endpoint returned no data")
                    return False
            else:
                print(f"✗ API health endpoint failed with status {response.status_code}")
                return False

            # Test configuration endpoint
            response = client.get('/api/config')
            if response.status_code == 200:
                data = response.get_json()
                if data and 'audio' in data and 'limits' in data and 'processing' in data:
                    print(f"✓ Config endpoint working")
                else:
                    print(f"✗ Config endpoint missing expected fields. Keys: {list(data.keys()) if data else 'None'}")
                    return False
            else:
                print(f"✗ Config endpoint failed with status {response.status_code}")
                return False

            # Test models info endpoint
            response = client.get('/api/models/info')
            if response.status_code == 200:
                data = response.get_json()
                if data and 'status' in data and 'models' in data:
                    print(f"✓ Models info endpoint working: status={data['status']}")
                else:
                    print(f"✗ Models info endpoint missing expected fields. Keys: {list(data.keys()) if data else 'None'}")
                    return False
            else:
                print(f"✗ Models info endpoint failed with status {response.status_code}")
                return False

        return True

    except Exception as e:
        print(f"✗ Web app testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoint functionality."""
    print("\nTesting API endpoints...")

    try:
        from auto_voice.web.app import create_app
        from auto_voice.web.api import api_bp

        app, socketio = create_app()
        # Blueprint is already registered in create_app()

        with app.test_client() as client:
            # Test synthesize endpoint with missing data
            response = client.post('/api/synthesize',
                                  json={},
                                  content_type='application/json')
            if response.status_code == 400:
                data = response.get_json()
                if data and 'error' in data:
                    print("✓ Synthesize endpoint validates required fields")
                else:
                    print(f"✗ Synthesize endpoint validation response issue. Data: {data}")
                    return False
            else:
                print(f"✗ Expected 400 for missing fields, got {response.status_code}")
                return False

            # Test synthesize endpoint with valid data (will fail without model)
            response = client.post('/api/synthesize',
                                  json={'text': 'Hello world'},
                                  content_type='application/json')
            if response.status_code in [503, 500]:  # Service unavailable or internal error
                print("✓ Synthesize endpoint responds (model not available)")
            elif response.status_code == 200:
                print("✓ Synthesize endpoint working with model")
            else:
                print(f"✗ Unexpected synthesize response: {response.status_code}")
                return False

            # Test analyze endpoint
            response = client.post('/api/analyze',
                                  json={},
                                  content_type='application/json')
            if response.status_code in [400, 503]:
                print("✓ Analyze endpoint validates input")
            else:
                print(f"✗ Unexpected analyze response: {response.status_code}")
                return False

            # Test config update endpoint
            response = client.post('/api/config',
                                  json={'temperature': 0.8},
                                  content_type='application/json')
            if response.status_code == 200:
                data = response.get_json()
                if data and 'status' in data and data['status'] == 'success':
                    print("✓ Config update endpoint working")
                else:
                    print(f"✗ Config update response issue. Data: {data}")
                    return False
            else:
                print(f"✗ Config update failed with status {response.status_code}")
                return False

        return True

    except Exception as e:
        print(f"✗ API endpoint testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket():
    """Test WebSocket handler functionality."""
    print("\nTesting WebSocket handler...")

    try:
        from auto_voice.web.websocket_handler import WebSocketHandler
        from flask_socketio import SocketIO
        from flask import Flask

        # Create minimal app for testing
        app = Flask(__name__)
        socketio = SocketIO(app)
        handler = WebSocketHandler(socketio)

        print("✓ WebSocket handler created successfully")

        # Verify handler methods exist
        required_methods = [
            'process_audio_chunk',
            'broadcast_to_room',
            'cleanup_session',
            '_get_capabilities',
            '_get_performance_metrics'
        ]

        for method in required_methods:
            if not hasattr(handler, method):
                print(f"✗ Missing required method: {method}")
                return False

        print(f"✓ All required methods present")

        # Add app context for WebSocket handler methods
        with app.app_context():
            # Test capabilities method
            capabilities = handler._get_capabilities()
            if isinstance(capabilities, dict) and 'audio_processing' in capabilities:
                print(f"✓ Capabilities method works: {list(capabilities.keys())}")
            else:
                print(f"✗ Capabilities method failed. Result: {capabilities}")
                return False

            # Test performance metrics
            metrics = handler._get_performance_metrics()
            if isinstance(metrics, dict) and 'active_sessions' in metrics:
                print(f"✓ Performance metrics method works")
            else:
                print(f"✗ Performance metrics method failed. Result: {metrics}")
                return False

        return True

    except Exception as e:
        print(f"✗ WebSocket testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_availability():
    """Test availability of core components."""
    print("\nTesting component availability...")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✓ PyTorch available, CUDA: {cuda_available}")

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"  GPU: {device_name} (Count: {device_count})")
    except ImportError:
        print("✗ PyTorch not available")
        return False

    try:
        import torchaudio
        print("✓ Torchaudio available")
    except ImportError:
        print("✗ Torchaudio not available")
        return False

    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        print(f"✓ System monitoring available (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)")
    except ImportError:
        print("✗ psutil not available")
        return False

    try:
        from flask_cors import CORS
        print("✓ Flask-CORS available")
    except ImportError:
        print("✗ Flask-CORS not available")
        return False

    return True

def main():
    """Run all tests."""
    print("AutoVoice Enhanced Startup Test (Fixed Edge Cases)")
    print("=" * 55)

    tests = [
        test_imports,
        test_config,
        test_web_app,
        test_api_endpoints,
        test_websocket,
        test_component_availability,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 55)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✓ All tests passed! AutoVoice is ready to start.")
        print("\nTo start the server, run:")
        print("  python -m auto_voice.web.app")
        print("\nAPI endpoints available at:")
        print("  - GET  /health              - System health check")
        print("  - GET  /api/config          - Get configuration")
        print("  - POST /api/synthesize      - Synthesize speech from text")
        print("  - POST /api/process_audio   - Process audio file")
        print("  - POST /api/analyze         - Analyze audio data")
        print("  - GET  /api/models/info     - Get model information")
        print("\nWebSocket events:")
        print("  - audio_stream      - Stream audio for processing")
        print("  - synthesize_stream - Stream text for synthesis")
        print("  - audio_analysis    - Real-time audio analysis")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install flask flask-socketio flask-cors psutil")
        print("  - CUDA not available: Check CUDA installation and PyTorch CUDA support")
        print("  - Import errors: Check Python path and module structure")
        return 1

if __name__ == '__main__':
    sys.exit(main())