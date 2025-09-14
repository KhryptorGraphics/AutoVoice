#!/usr/bin/env python3
"""Test script to verify AutoVoice startup without building CUDA extensions."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")

    try:
        from auto_voice.utils.config_loader import load_config
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
        from auto_voice.gpu.cuda_manager import CUDAManager
        print("✓ CUDA manager import successful")
    except Exception as e:
        print(f"✗ CUDA manager import failed: {e}")
        return False

    return True

def test_config():
    """Test config loading."""
    print("\nTesting config loading...")

    try:
        from auto_voice.utils.config_loader import load_config
        config = load_config()
        print(f"✓ Config loaded with keys: {list(config.keys())}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_web_app():
    """Test web app creation."""
    print("\nTesting web app creation...")

    try:
        from auto_voice.web.app import create_app
        app, socketio = create_app()
        print("✓ Web app created successfully")

        # Test basic route
        with app.test_client() as client:
            response = client.get('/healthz')
            if response.status_code == 200:
                print("✓ Health check endpoint working")
                return True
            else:
                print(f"✗ Health check failed with status {response.status_code}")
                return False

    except Exception as e:
        print(f"✗ Web app creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("AutoVoice Startup Test")
    print("=" * 30)

    tests = [
        test_imports,
        test_config,
        test_web_app,
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
            failed += 1

    print("\n" + "=" * 30)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✓ All tests passed! AutoVoice is ready to start.")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())