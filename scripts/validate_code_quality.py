#!/usr/bin/env python3
"""Code quality validation using pylint, flake8, mypy, radon, bandit.

This script runs multiple code quality tools and generates a comprehensive report:
- pylint: Code analysis and style checking
- flake8: Style guide enforcement
- mypy: Static type checking
- radon: Complexity analysis
- bandit: Security vulnerability scanning
"""
import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Compute project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pylint() -> Tuple[List[Dict[str, Any]], int]:
    """Run pylint on src/ directory.

    Returns:
        Tuple of (issues list, error count)
    """
    print("Running pylint...")
    result = subprocess.run(
        ['pylint', 'src/auto_voice', '--output-format=json', '--exit-zero'],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    try:
        issues = json.loads(result.stdout) if result.stdout else []
        error_count = sum(1 for issue in issues if issue.get('type') == 'error')
        return issues, error_count
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse pylint output: {result.stdout[:200]}")
        return [], 0


def run_flake8() -> Tuple[bool, str]:
    """Run flake8 for style checks.

    Returns:
        Tuple of (success bool, output string)
    """
    print("Running flake8...")
    result = subprocess.run(
        ['flake8', 'src/auto_voice', '--max-line-length=100', '--exclude=__pycache__,.git'],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return result.returncode == 0, result.stdout


def run_mypy() -> Tuple[bool, str]:
    """Run mypy for type checking.

    Returns:
        Tuple of (success bool, output string)
    """
    print("Running mypy...")
    result = subprocess.run(
        ['mypy', 'src/auto_voice', '--ignore-missing-imports', '--no-error-summary'],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return result.returncode == 0, result.stdout


def run_radon() -> Tuple[Dict[str, Any], int]:
    """Run radon for complexity analysis.

    Returns:
        Tuple of (complexity dict, high complexity count)
    """
    print("Running radon...")
    result = subprocess.run(
        ['radon', 'cc', 'src/auto_voice', '--min', 'C', '--json'],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    try:
        complexity = json.loads(result.stdout) if result.stdout else {}
        # Count functions with complexity >= 10 (high complexity)
        high_count = 0
        for file_data in complexity.values():
            high_count += sum(1 for func in file_data if func.get('complexity', 0) >= 10)
        return complexity, high_count
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse radon output")
        return {}, 0


def run_bandit() -> Tuple[Dict[str, Any], int]:
    """Run bandit for security checks.

    Returns:
        Tuple of (bandit results dict, high severity count)
    """
    print("Running bandit...")
    result = subprocess.run(
        ['bandit', '-r', 'src/auto_voice', '-f', 'json', '--exit-zero'],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    try:
        bandit_results = json.loads(result.stdout) if result.stdout else {}
        high_severity = sum(
            1 for issue in bandit_results.get('results', [])
            if issue.get('issue_severity') == 'HIGH'
        )
        return bandit_results, high_severity
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse bandit output")
        return {}, 0


def check_dependencies() -> bool:
    """Check if all required tools are installed.

    Returns:
        True if all tools available, False otherwise
    """
    tools = ['pylint', 'flake8', 'mypy', 'radon', 'bandit']
    missing = []

    for tool in tools:
        result = subprocess.run(['which', tool], capture_output=True)
        if result.returncode != 0:
            missing.append(tool)

    if missing:
        print(f"ERROR: Missing required tools: {', '.join(missing)}")
        print("Install with: pip install pylint flake8 mypy radon bandit")
        return False

    return True


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run code quality validation tools'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results/code_quality.json',
        help='Output file path (default: validation_results/code_quality.json)'
    )
    return parser.parse_args()


def main() -> int:
    """Main validation function.

    Returns:
        0 for success, 1 for failure
    """
    print("=== Code Quality Validation ===\n")

    # Parse CLI arguments
    args = parse_args()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Resolve output path relative to project root
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run all checks
    pylint_issues, pylint_errors = run_pylint()
    flake8_passed, flake8_output = run_flake8()
    mypy_passed, mypy_output = run_mypy()
    radon_complexity, high_complexity_count = run_radon()
    bandit_results, high_severity_count = run_bandit()

    # Aggregate results
    results = {
        'pylint': {
            'issues': pylint_issues,
            'error_count': pylint_errors,
            'passed': pylint_errors == 0
        },
        'flake8': {
            'passed': flake8_passed,
            'output': flake8_output
        },
        'mypy': {
            'passed': mypy_passed,
            'output': mypy_output
        },
        'radon': {
            'complexity': radon_complexity,
            'high_complexity_count': high_complexity_count,
            'passed': high_complexity_count == 0
        },
        'bandit': {
            'results': bandit_results,
            'high_severity_count': high_severity_count,
            'passed': high_severity_count == 0
        }
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Pylint errors: {pylint_errors}")
    print(f"Flake8: {'PASSED' if flake8_passed else 'FAILED'}")
    print(f"Mypy: {'PASSED' if mypy_passed else 'FAILED'}")
    print(f"Radon high complexity functions: {high_complexity_count}")
    print(f"Bandit high severity issues: {high_severity_count}")

    # Determine overall pass/fail
    # Critical: flake8 and mypy must pass
    # Warnings: pylint errors, high complexity, security issues
    critical_passed = flake8_passed and mypy_passed
    warnings_exist = (pylint_errors > 0 or high_complexity_count > 0 or high_severity_count > 0)

    if not critical_passed:
        print("\n❌ CRITICAL CHECKS FAILED")
        return 1
    elif warnings_exist:
        print("\n⚠️  PASSED WITH WARNINGS")
        return 0
    else:
        print("\n✅ ALL CHECKS PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
