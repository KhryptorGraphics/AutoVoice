"""
Comprehensive test runner for AutoVoice.

Supports pytest with markers, coverage reporting, and flexible test execution strategies.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional


def run_pytest(
    markers: Optional[List[str]] = None,
    coverage: bool = True,
    verbose: bool = False,
    fail_fast: bool = False,
    parallel: bool = False,
    test_files: Optional[List[str]] = None
) -> int:
    """
    Run pytest with specified options.

    Args:
        markers: List of pytest markers to select tests (e.g., ['unit', 'cuda'])
        coverage: Enable coverage reporting
        verbose: Enable verbose output
        fail_fast: Stop on first failure
        parallel: Run tests in parallel (requires pytest-xdist)
        test_files: Specific test files to run

    Returns:
        Exit code (0 for success)
    """
    cmd = [sys.executable, '-m', 'pytest']

    # Add test paths or use default
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append('tests/')

    # Add marker filters
    if markers:
        marker_expr = ' or '.join(markers)
        cmd.extend(['-m', marker_expr])

    # Add coverage
    if coverage:
        cmd.extend([
            '--cov=src/auto_voice',
            '--cov-report=html',
            '--cov-report=term',
            '--cov-report=xml'
        ])

    # Add verbosity
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')

    # Add fail-fast
    if fail_fast:
        cmd.append('-x')

    # Add parallel execution
    if parallel:
        cmd.extend(['-n', 'auto'])

    print(f"Running command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_unit_tests(coverage: bool = True, verbose: bool = False) -> int:
    """Run only unit tests."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    return run_pytest(markers=['unit'], coverage=coverage, verbose=verbose)


def run_integration_tests(coverage: bool = True, verbose: bool = False) -> int:
    """Run only integration tests."""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    return run_pytest(markers=['integration'], coverage=coverage, verbose=verbose)


def run_e2e_tests(coverage: bool = True, verbose: bool = False) -> int:
    """Run only end-to-end tests."""
    print("=" * 60)
    print("RUNNING END-TO-END TESTS")
    print("=" * 60)
    return run_pytest(markers=['e2e'], coverage=coverage, verbose=verbose)


def run_cuda_tests(coverage: bool = True, verbose: bool = False) -> int:
    """Run only CUDA/GPU tests."""
    print("=" * 60)
    print("RUNNING CUDA TESTS")
    print("=" * 60)
    return run_pytest(markers=['cuda'], coverage=coverage, verbose=verbose)


def run_performance_tests(coverage: bool = False, verbose: bool = True) -> int:
    """Run only performance benchmarks."""
    print("=" * 60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 60)
    return run_pytest(markers=['performance'], coverage=coverage, verbose=verbose)


def run_all_tests(coverage: bool = True, verbose: bool = False, parallel: bool = False) -> int:
    """Run all tests."""
    print("=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)
    return run_pytest(coverage=coverage, verbose=verbose, parallel=parallel)


def run_quick_tests(coverage: bool = False) -> int:
    """Run quick tests (unit tests only, no coverage)."""
    print("=" * 60)
    print("RUNNING QUICK TESTS (Unit only, no coverage)")
    print("=" * 60)
    return run_pytest(markers=['unit'], coverage=coverage, verbose=False)


def run_ci_tests(fail_fast: bool = True, parallel: bool = True) -> int:
    """Run tests suitable for CI/CD pipeline."""
    print("=" * 60)
    print("RUNNING CI TESTS")
    print("=" * 60)
    return run_pytest(
        markers=['unit', 'integration'],
        coverage=True,
        verbose=True,
        fail_fast=fail_fast,
        parallel=parallel
    )


def main():
    """Main test runner with command-line interface."""
    parser = argparse.ArgumentParser(
        description='AutoVoice Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Run all tests
  %(prog)s --unit              # Run only unit tests
  %(prog)s --cuda              # Run only CUDA tests
  %(prog)s --quick             # Quick test run (no coverage)
  %(prog)s --ci                # CI/CD test run
  %(prog)s --markers unit cuda # Run unit and CUDA tests
  %(prog)s --no-coverage       # Run without coverage
  %(prog)s --parallel          # Run tests in parallel
  %(prog)s tests/test_models.py  # Run specific test file
        """
    )

    # Test selection arguments
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests only')
    parser.add_argument('--cuda', action='store_true', help='Run CUDA tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--quick', action='store_true', help='Quick test run (unit only, no coverage)')
    parser.add_argument('--ci', action='store_true', help='Run CI/CD tests')

    # Test execution options
    parser.add_argument('--markers', nargs='+', help='Custom pytest markers (e.g., unit cuda)')
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-fast', '-x', action='store_true', help='Stop on first failure')
    parser.add_argument('--parallel', '-n', action='store_true', help='Run tests in parallel')

    # Specific test files
    parser.add_argument('test_files', nargs='*', help='Specific test files to run')

    args = parser.parse_args()

    coverage = not args.no_coverage
    verbose = args.verbose

    # Run specific test suite
    if args.unit:
        return run_unit_tests(coverage=coverage, verbose=verbose)
    elif args.integration:
        return run_integration_tests(coverage=coverage, verbose=verbose)
    elif args.e2e:
        return run_e2e_tests(coverage=coverage, verbose=verbose)
    elif args.cuda:
        return run_cuda_tests(coverage=coverage, verbose=verbose)
    elif args.performance:
        return run_performance_tests(coverage=coverage, verbose=verbose)
    elif args.quick:
        return run_quick_tests(coverage=False)
    elif args.ci:
        return run_ci_tests(fail_fast=args.fail_fast, parallel=args.parallel)
    elif args.markers:
        return run_pytest(
            markers=args.markers,
            coverage=coverage,
            verbose=verbose,
            fail_fast=args.fail_fast,
            parallel=args.parallel
        )
    elif args.test_files:
        return run_pytest(
            test_files=args.test_files,
            coverage=coverage,
            verbose=verbose,
            fail_fast=args.fail_fast,
            parallel=args.parallel
        )
    else:
        return run_all_tests(coverage=coverage, verbose=verbose, parallel=args.parallel)


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
