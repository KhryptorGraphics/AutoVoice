#!/usr/bin/env python3
"""Manual verification script for preset system.

This script verifies that the preset system is correctly implemented
without requiring full module installation.
"""

import os
import sys
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def verify_preset_file():
    """Verify preset file structure."""
    print("=" * 80)
    print("VERIFYING PRESET FILE")
    print("=" * 80)

    preset_path = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'voice_conversion_presets.yaml'
    )

    # Check file exists
    if not os.path.exists(preset_path):
        print(f"❌ FAIL: Preset file not found: {preset_path}")
        return False

    print(f"✓ Preset file exists: {preset_path}")

    # Load YAML
    try:
        with open(preset_path, 'r') as f:
            data = yaml.safe_load(f)
        print("✓ YAML is valid")
    except Exception as e:
        print(f"❌ FAIL: Invalid YAML: {e}")
        return False

    # Check structure
    if 'presets' not in data:
        print("❌ FAIL: Missing 'presets' key")
        return False
    print("✓ 'presets' key found")

    if 'default_preset' not in data:
        print("❌ FAIL: Missing 'default_preset' key")
        return False
    print(f"✓ 'default_preset' = '{data['default_preset']}'")

    # Check required presets
    required_presets = ['fast', 'balanced', 'quality', 'custom']
    presets = data['presets']

    for preset_name in required_presets:
        if preset_name not in presets:
            print(f"❌ FAIL: Missing preset '{preset_name}'")
            return False
        print(f"✓ Preset '{preset_name}' found")

    # Check preset structure
    required_components = ['vocal_separator', 'pitch_extractor', 'voice_converter', 'audio_mixer']

    for preset_name, preset_config in presets.items():
        print(f"\n  Checking preset '{preset_name}':")

        if 'description' not in preset_config:
            print(f"    ❌ FAIL: Missing 'description'")
            return False
        print(f"    ✓ Description: {preset_config['description']}")

        for component in required_components:
            if component not in preset_config:
                print(f"    ❌ FAIL: Missing '{component}' config")
                return False
            print(f"    ✓ Component '{component}' configured")

    print("\n✅ Preset file structure is valid\n")
    return True


def verify_preset_settings():
    """Verify preset settings are reasonable."""
    print("=" * 80)
    print("VERIFYING PRESET SETTINGS")
    print("=" * 80)

    preset_path = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'voice_conversion_presets.yaml'
    )

    with open(preset_path, 'r') as f:
        data = yaml.safe_load(f)

    presets = data['presets']

    # Verify fast preset
    print("\nFast Preset:")
    fast = presets['fast']
    print(f"  VocalSeparator model: {fast['vocal_separator'].get('model')}")
    print(f"  VocalSeparator shifts: {fast['vocal_separator'].get('shifts')}")
    print(f"  PitchExtractor model: {fast['pitch_extractor'].get('model')}")
    print(f"  PitchExtractor hop_length_ms: {fast['pitch_extractor'].get('hop_length_ms')}")
    print(f"  VoiceConverter num_flows: {fast['voice_converter']['flow_decoder'].get('num_flows')}")

    # Verify quality preset
    print("\nQuality Preset:")
    quality = presets['quality']
    print(f"  VocalSeparator model: {quality['vocal_separator'].get('model')}")
    print(f"  VocalSeparator shifts: {quality['vocal_separator'].get('shifts')}")
    print(f"  PitchExtractor model: {quality['pitch_extractor'].get('model')}")
    print(f"  PitchExtractor hop_length_ms: {quality['pitch_extractor'].get('hop_length_ms')}")
    print(f"  VoiceConverter num_flows: {quality['voice_converter']['flow_decoder'].get('num_flows')}")

    # Verify tradeoffs
    print("\n" + "=" * 80)
    print("VERIFYING QUALITY/SPEED TRADEOFFS")
    print("=" * 80)

    checks = []

    # Fast should use tiny model
    if fast['pitch_extractor'].get('model') == 'tiny':
        print("✓ Fast preset uses tiny CREPE model")
        checks.append(True)
    else:
        print("❌ Fast preset should use tiny CREPE model")
        checks.append(False)

    # Quality should use full model
    if quality['pitch_extractor'].get('model') == 'full':
        print("✓ Quality preset uses full CREPE model")
        checks.append(True)
    else:
        print("❌ Quality preset should use full CREPE model")
        checks.append(False)

    # Quality should have more shifts
    if quality['vocal_separator'].get('shifts', 0) > fast['vocal_separator'].get('shifts', 0):
        print(f"✓ Quality preset has more shifts ({quality['vocal_separator'].get('shifts')} vs {fast['vocal_separator'].get('shifts')})")
        checks.append(True)
    else:
        print("❌ Quality preset should have more shifts than fast")
        checks.append(False)

    # Quality should have higher time resolution (smaller hop)
    if quality['pitch_extractor'].get('hop_length_ms', 10) < fast['pitch_extractor'].get('hop_length_ms', 10):
        print(f"✓ Quality preset has higher time resolution ({quality['pitch_extractor'].get('hop_length_ms')}ms vs {fast['pitch_extractor'].get('hop_length_ms')}ms)")
        checks.append(True)
    else:
        print("❌ Quality preset should have smaller hop_length_ms than fast")
        checks.append(False)

    # Quality should have more flow layers
    if quality['voice_converter']['flow_decoder'].get('num_flows', 0) > fast['voice_converter']['flow_decoder'].get('num_flows', 0):
        print(f"✓ Quality preset has more flow layers ({quality['voice_converter']['flow_decoder'].get('num_flows')} vs {fast['voice_converter']['flow_decoder'].get('num_flows')})")
        checks.append(True)
    else:
        print("❌ Quality preset should have more flow layers than fast")
        checks.append(False)

    if all(checks):
        print("\n✅ All quality/speed tradeoffs are correctly configured\n")
        return True
    else:
        print("\n❌ Some quality/speed tradeoffs are incorrect\n")
        return False


def verify_code_integration():
    """Verify code integration (without running full pipeline)."""
    print("=" * 80)
    print("VERIFYING CODE INTEGRATION")
    print("=" * 80)

    # Check if pipeline file has been modified
    pipeline_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'auto_voice', 'inference', 'singing_conversion_pipeline.py'
    )

    if not os.path.exists(pipeline_path):
        print(f"❌ FAIL: Pipeline file not found: {pipeline_path}")
        return False

    with open(pipeline_path, 'r') as f:
        code = f.read()

    # Check for key methods and features
    checks = {
        'preset parameter in __init__': 'preset: Optional[str] = None' in code,
        '_load_preset_config method': 'def _load_preset_config(self' in code,
        'set_preset method': 'def set_preset(self, preset_name: str' in code,
        'get_current_preset method': 'def get_current_preset(self)' in code,
        'current_preset attribute': 'self.current_preset' in code,
        'preset config loading': '_load_preset_config(preset' in code or '_load_preset_config(preset_name' in code,
        'voice_conversion_presets.yaml path': 'voice_conversion_presets.yaml' in code,
    }

    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            print(f"✓ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False

    if all_passed:
        print("\n✅ Code integration verified\n")
    else:
        print("\n❌ Some code integration checks failed\n")

    return all_passed


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("PRESET SYSTEM VERIFICATION")
    print("=" * 80 + "\n")

    results = []

    # Run all checks
    results.append(("Preset File", verify_preset_file()))
    results.append(("Preset Settings", verify_preset_settings()))
    results.append(("Code Integration", verify_code_integration()))

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "=" * 80)
        print("✅ ALL CHECKS PASSED - PRESET SYSTEM IS READY")
        print("=" * 80 + "\n")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ SOME CHECKS FAILED - PLEASE REVIEW")
        print("=" * 80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
