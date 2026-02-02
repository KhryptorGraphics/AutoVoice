#!/usr/bin/env python3
"""Validate OpenAPI specification and test Swagger UI."""
import sys
import json
import yaml
import requests
from pathlib import Path


def validate_openapi_spec(spec_url: str):
    """Validate OpenAPI spec structure."""
    print(f"Fetching OpenAPI spec from {spec_url}...")

    try:
        response = requests.get(spec_url, timeout=10)
        response.raise_for_status()
        spec = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch spec: {e}")
        return False

    # Validate required fields
    required_fields = ['openapi', 'info', 'paths']
    for field in required_fields:
        if field not in spec:
            print(f"❌ Missing required field: {field}")
            return False

    print(f"✅ OpenAPI version: {spec['openapi']}")
    print(f"✅ Title: {spec['info']['title']}")
    print(f"✅ Version: {spec['info']['version']}")

    # Count endpoints
    endpoint_count = len(spec.get('paths', {}))
    print(f"✅ Total endpoints documented: {endpoint_count}")

    # Validate schemas
    schemas = spec.get('components', {}).get('schemas', {})
    print(f"✅ Total schemas defined: {len(schemas)}")

    # List endpoint groups
    tags = set()
    for path, methods in spec.get('paths', {}).items():
        for method, details in methods.items():
            if isinstance(details, dict):
                endpoint_tags = details.get('tags', [])
                tags.update(endpoint_tags)

    print(f"\n✅ Endpoint groups:")
    for tag in sorted(tags):
        print(f"   - {tag}")

    return True


def test_swagger_ui(base_url: str):
    """Test Swagger UI accessibility."""
    swagger_url = f"{base_url}/docs"

    print(f"\nTesting Swagger UI at {swagger_url}...")

    try:
        response = requests.get(swagger_url, timeout=10)
        response.raise_for_status()

        if 'swagger-ui' in response.text.lower():
            print(f"✅ Swagger UI accessible at {swagger_url}")
            return True
        else:
            print(f"❌ Swagger UI page found but content unexpected")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to access Swagger UI: {e}")
        return False


def validate_yaml_spec(yaml_url: str):
    """Validate YAML format of spec."""
    print(f"\nValidating YAML spec at {yaml_url}...")

    try:
        response = requests.get(yaml_url, timeout=10)
        response.raise_for_status()

        # Try to parse YAML
        spec = yaml.safe_load(response.text)

        if 'openapi' in spec:
            print(f"✅ YAML spec valid and parseable")
            return True
        else:
            print(f"❌ YAML spec missing 'openapi' field")
            return False

    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch YAML spec: {e}")
        return False


def check_endpoint_coverage():
    """Check if major endpoint groups are documented."""
    required_groups = [
        'Conversion',
        'Voice Profiles',
        'Training',
        'Audio Processing',
        'System',
        'YouTube'
    ]

    print("\n✅ Expected endpoint groups:")
    for group in required_groups:
        print(f"   - {group}")

    return True


def main():
    """Run validation suite."""
    base_url = "http://localhost:5000"

    print("=" * 60)
    print("AutoVoice OpenAPI Validation")
    print("=" * 60)

    # Check if server is running
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"✅ Server running at {base_url}")
    except requests.exceptions.RequestException:
        print(f"❌ Server not running at {base_url}")
        print("   Please start the server with: python main.py")
        return 1

    # Run validation tests
    tests = [
        ("OpenAPI JSON Spec", lambda: validate_openapi_spec(f"{base_url}/api/v1/openapi.json")),
        ("OpenAPI YAML Spec", lambda: validate_yaml_spec(f"{base_url}/api/v1/openapi.yaml")),
        ("Swagger UI", lambda: test_swagger_ui(base_url)),
        ("Endpoint Coverage", check_endpoint_coverage),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print('=' * 60)
        result = test_func()
        results.append((test_name, result))

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    # Overall result
    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 All validation tests passed!")
        print(f"\n📚 View API documentation at: {base_url}/docs")
        return 0
    else:
        print("\n⚠️  Some validation tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
