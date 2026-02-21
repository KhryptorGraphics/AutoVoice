#!/usr/bin/env python3
"""End-to-end verification of secret key security with different configurations.

This script tests the Flask app's SECRET_KEY behavior in different configurations:
1. With SECRET_KEY env var (direct override)
2. With AUTOVOICE_SECRET_FLASK_SECRET_KEY env var (SecretsManager)
3. With no env vars in testing mode (random key generation)
4. Verifies hardcoded 'autovoice-dev-key' is never used
"""

import os
import sys
import subprocess
import tempfile
import json
import time
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_test(name):
    """Print test name."""
    print(f"\n{BOLD}🧪 Test: {name}{RESET}")


def print_success(message):
    """Print success message."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message):
    """Print error message."""
    print(f"{RED}✗ {message}{RESET}")


def print_warning(message):
    """Print warning message."""
    print(f"{YELLOW}⚠ {message}{RESET}")


def run_python_test(code, env=None, expect_error=False):
    """Run Python code and return stdout, stderr, returncode.

    Args:
        code: Python code to execute
        env: Environment variables dict (merged with current env)
        expect_error: If True, don't fail on non-zero exit

    Returns:
        tuple: (stdout, stderr, returncode)
    """
    # Merge environment
    test_env = os.environ.copy()

    # Clear any existing secret env vars for clean test
    keys_to_remove = [k for k in test_env.keys() if k.startswith('AUTOVOICE_SECRET_')]
    for key in keys_to_remove:
        del test_env[key]

    if env:
        test_env.update(env)

    # Add required Python environment
    test_env['PYTHONNOUSERSITE'] = '1'
    test_env['PYTHONPATH'] = 'src'

    # Run the test
    result = subprocess.run(
        [sys.executable, '-c', code],
        env=test_env,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    if not expect_error and result.returncode != 0:
        print_error(f"Python test failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

    return result.stdout, result.stderr, result.returncode


def test_secret_key_from_config():
    """Test 1: App uses SECRET_KEY from config when provided."""
    print_test("App uses SECRET_KEY from config when provided")

    test_key = "my_custom_secret_key_12345678901234567890"

    code = f"""
from auto_voice.web.app import create_app

# Pass SECRET_KEY via config dict
app, _ = create_app(config={{'SECRET_KEY': '{test_key}', 'TESTING': True}}, testing=True)

secret_key = app.config['SECRET_KEY']

# Verify it uses our provided key
assert secret_key == '{test_key}', f"Expected '{test_key}', got '{{secret_key}}'"
assert secret_key != 'autovoice-dev-key', "Should not use hardcoded key"

print(f"SUCCESS: App uses SECRET_KEY from config")
print(f"Key length: {{len(secret_key)}} chars")
"""

    stdout, stderr, returncode = run_python_test(code)

    if returncode == 0 and "SUCCESS" in stdout:
        print_success("App correctly uses SECRET_KEY from config")
        return True
    else:
        print_error("App did not use SECRET_KEY from config")
        return False


def test_secret_key_from_secrets_manager():
    """Test 2: App uses AUTOVOICE_SECRET_FLASK_SECRET_KEY via SecretsManager."""
    print_test("App uses AUTOVOICE_SECRET_FLASK_SECRET_KEY via SecretsManager")

    test_key = "secrets_manager_key_12345678901234567890"

    code = f"""
from auto_voice.web.app import create_app

# Production mode (testing=False) to trigger SecretsManager
# But use TESTING=True in config to avoid SocketIO eventlet
app, _ = create_app(config={{'TESTING': True}}, testing=False)

secret_key = app.config['SECRET_KEY']

# Verify it loaded from SecretsManager
assert secret_key == '{test_key}', f"Expected '{test_key}', got '{{secret_key}}'"
assert secret_key != 'autovoice-dev-key', "Should not use hardcoded key"

print(f"SUCCESS: App uses secret from SecretsManager")
print(f"Key length: {{len(secret_key)}} chars")
"""

    stdout, stderr, returncode = run_python_test(
        code,
        env={'AUTOVOICE_SECRET_FLASK_SECRET_KEY': test_key}
    )

    if returncode == 0 and "SUCCESS" in stdout:
        print_success("App correctly loads secret from SecretsManager (AUTOVOICE_SECRET_FLASK_SECRET_KEY)")
        return True
    else:
        print_error("App did not load secret from SecretsManager")
        return False


def test_random_key_in_testing_mode():
    """Test 3: App generates random key in testing mode when no env var."""
    print_test("App generates random key in testing mode")

    code = """
from auto_voice.web.app import create_app

# Testing mode should auto-generate secure random key
app, _ = create_app(testing=True)

secret_key = app.config['SECRET_KEY']

# Verify it's a random key (not hardcoded)
assert secret_key != 'autovoice-dev-key', "Should not use hardcoded key"
assert len(secret_key) >= 32, f"Key too short: {len(secret_key)} chars"
assert len(secret_key) == 64, f"Expected 64 hex chars, got {len(secret_key)}"

# Verify it's hex (from secrets.token_hex(32))
assert all(c in '0123456789abcdef' for c in secret_key), "Should be hex characters"

# Generate another app and verify keys are unique
app2, _ = create_app(testing=True)
secret_key2 = app2.config['SECRET_KEY']

assert secret_key != secret_key2, "Each app should get unique random key"

print(f"SUCCESS: Testing mode generates unique random keys")
print(f"Key 1: {secret_key[:16]}... (length: {len(secret_key)})")
print(f"Key 2: {secret_key2[:16]}... (length: {len(secret_key2)})")
"""

    stdout, stderr, returncode = run_python_test(code)

    if returncode == 0 and "SUCCESS" in stdout:
        print_success("Testing mode correctly generates unique random keys")
        return True
    else:
        print_error("Testing mode did not generate random keys correctly")
        return False


def test_production_requires_env_var():
    """Test 4: Production mode requires env var and fails without it."""
    print_test("Production mode requires AUTOVOICE_SECRET_FLASK_SECRET_KEY")

    code = """
from auto_voice.web.app import create_app
from auto_voice.config.secrets import SecretError
import sys

try:
    # Production mode (testing=False) without env var should fail
    # Use TESTING=True to avoid SocketIO eventlet
    app, _ = create_app(config={'TESTING': True}, testing=False)
    print("ERROR: Should have raised SecretError")
    sys.exit(1)
except SecretError as e:
    # Expected behavior
    assert "Required secret 'flask_secret_key' not found" in str(e)
    print(f"SUCCESS: Production mode correctly requires env var")
    print(f"Error message: {e}")
    sys.exit(0)
"""

    stdout, stderr, returncode = run_python_test(code, expect_error=False)

    if returncode == 0 and "SUCCESS" in stdout:
        print_success("Production mode correctly requires AUTOVOICE_SECRET_FLASK_SECRET_KEY")
        return True
    else:
        print_error("Production mode did not properly enforce env var requirement")
        return False


def test_hardcoded_key_never_used():
    """Test 5: Verify hardcoded 'autovoice-dev-key' is never used."""
    print_test("Hardcoded 'autovoice-dev-key' is never used")

    # Test in multiple scenarios
    test_scenarios = [
        ("Testing mode", "testing=True", {}),
        ("Production with env var", "testing=False", {'AUTOVOICE_SECRET_FLASK_SECRET_KEY': 'prod_key_123'}),
        ("Config override", "testing=False", {}, "{'SECRET_KEY': 'override_key', 'TESTING': True}"),
    ]

    all_passed = True

    for scenario_name, mode, env, *config in test_scenarios:
        config_str = config[0] if config else "{'TESTING': True}"

        code = f"""
from auto_voice.web.app import create_app

app, _ = create_app(config={config_str}, {mode})

secret_key = app.config['SECRET_KEY']

# MUST NOT be the hardcoded key
assert secret_key != 'autovoice-dev-key', f"FAIL: Using hardcoded key in {{'{scenario_name}'}}"

print(f"✓ {{'{scenario_name}'}}: Not using hardcoded key")
"""

        stdout, stderr, returncode = run_python_test(code, env=env)

        if returncode == 0:
            print(f"  {GREEN}✓{RESET} {scenario_name}: Passed")
        else:
            print(f"  {RED}✗{RESET} {scenario_name}: Failed")
            all_passed = False

    if all_passed:
        print_success("Hardcoded 'autovoice-dev-key' is never used in any scenario")
        return True
    else:
        print_error("Hardcoded key was used in at least one scenario")
        return False


def test_source_code_scan():
    """Test 6: Scan source code to verify no hardcoded 'autovoice-dev-key'."""
    print_test("Source code scan for hardcoded 'autovoice-dev-key'")

    # Search in source files (excluding tests which may reference it)
    src_dir = Path(__file__).parent / 'src'

    found_files = []

    for py_file in src_dir.rglob('*.py'):
        content = py_file.read_text()
        if 'autovoice-dev-key' in content:
            found_files.append(str(py_file))

    if found_files:
        print_error(f"Found hardcoded 'autovoice-dev-key' in source files:")
        for f in found_files:
            print(f"  - {f}")
        return False
    else:
        print_success("No hardcoded 'autovoice-dev-key' found in source code")
        return True


def test_secrets_manager_from_file():
    """Test 7: SecretsManager can load from JSON file."""
    print_test("SecretsManager loads secret from JSON file")

    # Create temp secrets file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        secrets_data = {
            'flask_secret_key': 'file_based_secret_key_12345678901234567890'
        }
        json.dump(secrets_data, f)
        secrets_file = f.name

    try:
        code = f"""
from auto_voice.config.secrets import SecretsManager

manager = SecretsManager(secrets_file='{secrets_file}')

secret_key = manager.get('flask_secret_key')

assert secret_key == 'file_based_secret_key_12345678901234567890'
assert manager.has('flask_secret_key')

print(f"SUCCESS: SecretsManager loaded from file")
print(f"Key length: {{len(secret_key)}} chars")
"""

        stdout, stderr, returncode = run_python_test(code)

        if returncode == 0 and "SUCCESS" in stdout:
            print_success("SecretsManager correctly loads secrets from JSON file")
            return True
        else:
            print_error("SecretsManager did not load from file correctly")
            return False
    finally:
        # Clean up temp file
        Path(secrets_file).unlink(missing_ok=True)


def main():
    """Run all verification tests."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Secret Key Security End-to-End Verification{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    tests = [
        test_secret_key_from_config,
        test_secret_key_from_secrets_manager,
        test_random_key_in_testing_mode,
        test_production_requires_env_var,
        test_hardcoded_key_never_used,
        test_source_code_scan,
        test_secrets_manager_from_file,
    ]

    results = []

    for test_func in tests:
        try:
            passed = test_func()
            results.append((test_func.__name__, passed))
        except Exception as e:
            print_error(f"Test raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Print summary
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Test Summary{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{status} - {test_name}")

    print(f"\n{BOLD}Results: {passed_count}/{total_count} tests passed{RESET}")

    if passed_count == total_count:
        print(f"\n{GREEN}{BOLD}✓ All verification tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ Some tests failed{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
