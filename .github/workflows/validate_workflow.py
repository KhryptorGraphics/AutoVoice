#!/usr/bin/env python3
"""
GitHub Actions Workflow Validator

Validates workflow YAML files for common issues and best practices.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


class WorkflowValidator:
    """Validates GitHub Actions workflow files."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_file(self, filepath: Path) -> bool:
        """Validate a single workflow file."""
        print(f"\n🔍 Validating: {filepath.name}")
        print("=" * 60)

        try:
            with open(filepath) as f:
                workflow = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"File reading error: {e}")
            return False

        # Run validation checks
        self._validate_structure(workflow)
        self._validate_jobs(workflow)
        self._validate_steps(workflow)
        self._validate_security(workflow)
        self._validate_performance(workflow)
        self._validate_best_practices(workflow)

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _validate_structure(self, workflow: Dict):
        """Validate basic workflow structure."""
        # Check required fields
        if 'name' not in workflow:
            self.warnings.append("Missing workflow name")

        if 'on' not in workflow:
            self.errors.append("Missing 'on' trigger configuration")

        if 'jobs' not in workflow:
            self.errors.append("Missing 'jobs' section")

        # Check triggers
        triggers = workflow.get('on', {})
        if isinstance(triggers, dict):
            if 'push' in triggers or 'pull_request' in triggers:
                self.info.append("✓ Basic CI triggers configured")

            if 'schedule' in triggers:
                self.info.append("✓ Scheduled runs configured")

            if 'workflow_dispatch' in triggers:
                self.info.append("✓ Manual trigger enabled")

    def _validate_jobs(self, workflow: Dict):
        """Validate job configurations."""
        jobs = workflow.get('jobs', {})

        if not jobs:
            self.errors.append("No jobs defined")
            return

        for job_name, job_config in jobs.items():
            # Check runs-on
            if 'runs-on' not in job_config:
                self.errors.append(f"Job '{job_name}' missing 'runs-on'")

            # Check timeout
            if 'timeout-minutes' not in job_config:
                self.warnings.append(
                    f"Job '{job_name}' has no timeout (default: 360 min)"
                )

            # Check steps
            if 'steps' not in job_config:
                self.errors.append(f"Job '{job_name}' has no steps")

            # Check if condition
            if 'if' in job_config:
                self.info.append(f"✓ Job '{job_name}' has conditional execution")

    def _validate_steps(self, workflow: Dict):
        """Validate step configurations."""
        jobs = workflow.get('jobs', {})

        for job_name, job_config in jobs.items():
            steps = job_config.get('steps', [])

            for i, step in enumerate(steps):
                step_id = step.get('name', f'step-{i}')

                # Check step has action or run
                if 'uses' not in step and 'run' not in step:
                    self.errors.append(
                        f"Step '{step_id}' in job '{job_name}' "
                        "has neither 'uses' nor 'run'"
                    )

                # Check for deprecated actions
                if 'uses' in step:
                    action = step['uses']
                    if '@v1' in action or '@v2' in action:
                        self.warnings.append(
                            f"Step '{step_id}' uses old action version: {action}"
                        )

                # Check checkout depth
                if 'actions/checkout' in step.get('uses', ''):
                    if 'fetch-depth' not in step.get('with', {}):
                        self.info.append(
                            f"Step '{step_id}' uses default checkout depth (1)"
                        )

                # Check caching
                if 'actions/cache' in step.get('uses', ''):
                    self.info.append(f"✓ Caching configured in '{step_id}'")

    def _validate_security(self, workflow: Dict):
        """Validate security best practices."""
        jobs = workflow.get('jobs', {})

        for job_name, job_config in jobs.items():
            steps = job_config.get('steps', [])

            for step in steps:
                # Check for secret usage
                run_cmd = step.get('run', '')
                if '${{ secrets.' in run_cmd:
                    self.warnings.append(
                        f"Direct secret usage in run command in job '{job_name}'. "
                        "Consider using environment variables."
                    )

                # Check for write permissions
                if 'permissions' in job_config:
                    perms = job_config['permissions']
                    if isinstance(perms, dict):
                        write_perms = [
                            k for k, v in perms.items()
                            if v == 'write'
                        ]
                        if write_perms:
                            self.info.append(
                                f"Job '{job_name}' has write permissions: "
                                f"{', '.join(write_perms)}"
                            )

    def _validate_performance(self, workflow: Dict):
        """Validate performance optimizations."""
        jobs = workflow.get('jobs', {})

        has_caching = False
        has_parallel_jobs = len(jobs) > 1
        has_matrix = False

        for job_config in jobs.values():
            steps = job_config.get('steps', [])

            # Check for caching
            for step in steps:
                if 'cache' in step.get('uses', '').lower():
                    has_caching = True

            # Check for matrix strategy
            if 'strategy' in job_config:
                if 'matrix' in job_config['strategy']:
                    has_matrix = True

        if has_caching:
            self.info.append("✓ Caching strategy implemented")
        else:
            self.warnings.append(
                "No caching detected. Consider adding cache steps."
            )

        if has_parallel_jobs:
            self.info.append(f"✓ Multiple jobs for parallel execution ({len(jobs)})")

        if has_matrix:
            self.info.append("✓ Matrix strategy used for test parallelization")

    def _validate_best_practices(self, workflow: Dict):
        """Validate general best practices."""
        jobs = workflow.get('jobs', {})

        # Check for artifact retention
        has_artifacts = False
        for job_config in jobs.values():
            steps = job_config.get('steps', [])
            for step in steps:
                if 'upload-artifact' in step.get('uses', ''):
                    has_artifacts = True
                    if 'retention-days' in step.get('with', {}):
                        self.info.append(
                            "✓ Artifact retention configured"
                        )

        if not has_artifacts:
            self.info.append("No artifacts uploaded")

        # Check for continue-on-error
        for job_name, job_config in jobs.items():
            steps = job_config.get('steps', [])
            for step in steps:
                if step.get('continue-on-error'):
                    step_name = step.get('name', 'unknown')
                    self.warnings.append(
                        f"Step '{step_name}' in job '{job_name}' "
                        "continues on error"
                    )

        # Check for if: always()
        for job_name, job_config in jobs.items():
            steps = job_config.get('steps', [])
            for step in steps:
                if step.get('if') == 'always()':
                    step_name = step.get('name', 'unknown')
                    self.info.append(
                        f"✓ Step '{step_name}' always runs (cleanup/reporting)"
                    )

    def _print_results(self):
        """Print validation results."""
        print()

        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")
            print()

        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()

        if self.info:
            print("ℹ️  INFO:")
            for info in self.info:
                print(f"  • {info}")
            print()

        if not self.errors and not self.warnings:
            print("✅ No issues found!")
        elif not self.errors:
            print("✅ Validation passed (with warnings)")
        else:
            print("❌ Validation failed")


def main():
    parser = argparse.ArgumentParser(
        description='Validate GitHub Actions workflow files'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Workflow files to validate (default: all in .github/workflows/)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )

    args = parser.parse_args()

    # Find workflow files
    if args.files:
        workflow_files = [Path(f) for f in args.files]
    else:
        workflows_dir = Path('.github/workflows')
        if not workflows_dir.exists():
            print(f"Error: {workflows_dir} not found")
            return 1

        workflow_files = list(workflows_dir.glob('*.yml')) + \
                        list(workflows_dir.glob('*.yaml'))

    if not workflow_files:
        print("No workflow files found")
        return 1

    # Validate each file
    all_passed = True
    for filepath in workflow_files:
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            all_passed = False
            continue

        validator = WorkflowValidator()
        passed = validator.validate_file(filepath)

        if not passed:
            all_passed = False
        elif args.strict and validator.warnings:
            print(f"\n❌ Strict mode: Treating warnings as errors")
            all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All workflow files validated successfully")
        return 0
    else:
        print("❌ Some workflow files have issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())
