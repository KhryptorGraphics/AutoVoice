#!/usr/bin/env python3
"""
Comprehensive Validation Suite Runner

Orchestrates all validation activities:
1. Code quality validation (pylint, flake8, mypy, radon, bandit)
2. Integration validation (GPU manager, audio processor, web API, pipeline)
3. Documentation validation (docstrings, code examples, README links, API docs)
4. Performance profiling (stage timing, GPU utilization)
5. Final validation report generation

Exit codes:
0 - All validations passed
1 - Critical validation failures
2 - Non-critical warnings (passes with warnings)
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


class ValidationRunner:
    """Orchestrates all validation scripts."""

    def __init__(self, scripts_dir: Path):
        self.scripts_dir = scripts_dir
        self.results: Dict[str, Dict[str, any]] = {}

    def run_script(self, script_name: str, description: str, critical: bool = True) -> Tuple[bool, float]:
        """Run a validation script and capture results."""
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            print(f"⚠️  Script not found: {script_name}")
            return False, 0.0

        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Script: {script_name}")
        print(f"{'='*60}\n")

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            elapsed = time.time() - start_time

            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr, file=sys.stderr)

            success = result.returncode == 0

            if success:
                print(f"\n✅ {description} completed successfully ({elapsed:.1f}s)")
            else:
                if critical:
                    print(f"\n❌ {description} FAILED ({elapsed:.1f}s)")
                else:
                    print(f"\n⚠️  {description} completed with warnings ({elapsed:.1f}s)")

            return success, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"\n❌ {description} TIMEOUT after {elapsed:.1f}s")
            return False, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ {description} ERROR: {e}")
            return False, elapsed

    def run_validation_suite(self) -> int:
        """Run complete validation suite."""
        print("="*60)
        print("AUTOVOICE COMPREHENSIVE VALIDATION SUITE")
        print("="*60)

        validation_steps = [
            # (script_name, description, critical)
            ('validate_code_quality.py', 'Code Quality Validation', True),
            ('validate_integration.py', 'Integration Validation', True),
            ('validate_documentation.py', 'Documentation Validation', False),
            ('profile_performance.py', 'Performance Profiling', False),
        ]

        results = []
        total_time = 0.0
        critical_failures = []
        warnings = []

        for script_name, description, critical in validation_steps:
            success, elapsed = self.run_script(script_name, description, critical)
            total_time += elapsed

            results.append({
                'script': script_name,
                'description': description,
                'success': success,
                'elapsed': elapsed,
                'critical': critical
            })

            if not success:
                if critical:
                    critical_failures.append(description)
                else:
                    warnings.append(description)

        # Generate final validation report
        print(f"\n{'='*60}")
        print("Generating Final Validation Report")
        print(f"{'='*60}\n")

        report_success, report_time = self.run_script(
            'generate_validation_report.py',
            'Validation Report Generation',
            critical=False
        )
        total_time += report_time

        # Print summary
        print(f"\n{'='*60}")
        print("VALIDATION SUITE SUMMARY")
        print(f"{'='*60}\n")

        print(f"Total Time: {total_time:.1f}s\n")

        for result in results:
            status_icon = "✅" if result['success'] else ("❌" if result['critical'] else "⚠️ ")
            print(f"{status_icon} {result['description']:<40} {result['elapsed']:>6.1f}s")

        if report_success:
            print(f"✅ {'Validation Report Generation':<40} {report_time:>6.1f}s")
        else:
            print(f"⚠️  {'Validation Report Generation':<40} {report_time:>6.1f}s")

        # Determine exit code
        print(f"\n{'='*60}")

        if critical_failures:
            print("❌ CRITICAL VALIDATION FAILURES:")
            for failure in critical_failures:
                print(f"  - {failure}")
            print(f"{'='*60}\n")
            return 1

        elif warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
            print(f"{'='*60}\n")
            print("Non-critical validations had issues but core functionality validated.")
            return 2

        else:
            print("✅ ALL VALIDATIONS PASSED")
            print(f"{'='*60}\n")
            return 0


def main() -> int:
    """Main entry point."""
    # Determine scripts directory
    scripts_dir = Path(__file__).parent

    # Check if we're in the right directory
    if not scripts_dir.name == 'scripts':
        print("Error: Script must be run from scripts directory")
        return 1

    # Create runner
    runner = ValidationRunner(scripts_dir)

    # Run validation suite
    exit_code = runner.run_validation_suite()

    # Print final message
    print("\nValidation Results:")
    print(f"  Report: {scripts_dir.parent / 'FINAL_VALIDATION_REPORT.md'}")
    print(f"  Raw data: {scripts_dir.parent / 'validation_results' / '*.json'}")
    print()

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
