#!/usr/bin/env python3
"""Generate comprehensive validation report from all validation results.

Aggregates results from:
- Code quality validation
- Integration testing
- Documentation validation
- System validation tests
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Compute project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_result_file(
    explicit_path: Optional[str],
    default_paths: list,
    key: str
) -> Optional[Dict[str, Any]]:
    """Load a single result file with fallback logic.

    Args:
        explicit_path: Explicit path provided via CLI
        default_paths: List of default paths to try (relative to PROJECT_ROOT)
        key: Result key name

    Returns:
        Loaded JSON data or None
    """
    paths_to_try = []

    if explicit_path:
        paths_to_try.append(PROJECT_ROOT / explicit_path)

    # Add default paths
    for default_path in default_paths:
        paths_to_try.append(PROJECT_ROOT / default_path)

    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {'error': f'Invalid JSON in {path}'}

    return None


def load_results(args) -> Dict[str, Any]:
    """Load all validation results with CLI argument support.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dict containing all results
    """
    results = {}

    # Define file mappings with fallback paths
    file_mappings = {
        'code_quality': {
            'arg': args.code_quality,
            'defaults': [
                'validation_results/reports/code_quality.json',
                'validation_results/code_quality.json'
            ]
        },
        'integration': {
            'arg': args.integration,
            'defaults': [
                'validation_results/reports/integration.json',
                'validation_results/integration.json',
                'validation_results/integration_validation.json'
            ]
        },
        'documentation': {
            'arg': args.documentation,
            'defaults': [
                'validation_results/reports/documentation.json',
                'validation_results/documentation.json'
            ]
        },
        'system': {
            'arg': args.system_validation,
            'defaults': [
                'validation_results/reports/system_validation.json',
                'validation_results/system_validation.json'
            ]
        },
        'test_results': {
            'arg': args.e2e_tests,
            'defaults': [
                'validation_results/reports/test_results.json',
                'validation_results/test_results.json'
            ]
        },
        'performance_breakdown': {
            'arg': args.performance,
            'defaults': [
                'validation_results/reports/performance_breakdown.json',
                'validation_results/performance_breakdown.json'
            ]
        },
        'latency_tensorrt': {
            'arg': args.performance,  # Can use same arg or separate
            'defaults': [
                'validation_results/reports/latency_tensorrt.json',
                'validation_results/latency_tensorrt.json'
            ]
        },
        'security': {
            'arg': args.security,
            'defaults': [
                'validation_results/reports/security.json',
                'validation_results/security.json'
            ]
        }
    }

    for key, config in file_mappings.items():
        result = load_result_file(config['arg'], config['defaults'], key)
        if result is not None:
            results[key] = result

    # Load Docker validation log if exists
    docker_log_paths = [
        PROJECT_ROOT / 'validation_results/reports/docker_validation.log',
        PROJECT_ROOT / 'validation_results/docker_validation.log'
    ]
    for docker_log_path in docker_log_paths:
        if docker_log_path.exists():
            with open(docker_log_path, 'r') as f:
                results['docker_log'] = f.read()
            break

    # Load quality evaluation results if exist
    quality_eval_dirs = [
        PROJECT_ROOT / 'validation_results/reports/quality_evaluation',
        PROJECT_ROOT / 'validation_results/quality_evaluation'
    ]
    for quality_eval_dir in quality_eval_dirs:
        if quality_eval_dir.exists():
            results['quality_eval'] = {}
            for eval_file in quality_eval_dir.glob('*.json'):
                try:
                    with open(eval_file, 'r') as f:
                        results['quality_eval'][eval_file.stem] = json.load(f)
                except Exception:
                    pass
            break

    return results


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate markdown report from results.

    Args:
        results: Aggregated validation results

    Returns:
        Markdown formatted report string
    """
    report = []
    report.append("# AutoVoice Final Validation Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Executive Summary
    report.append("## Executive Summary\n")

    overall_status = "✅ PASSED"
    critical_failures = []
    warnings = []

    # Code Quality
    if 'code_quality' in results:
        cq = results['code_quality']
        if not cq.get('flake8', {}).get('passed') or not cq.get('mypy', {}).get('passed'):
            critical_failures.append("Code quality checks failed")
            overall_status = "❌ FAILED"

        if cq.get('pylint', {}).get('error_count', 0) > 0:
            warnings.append(f"Pylint: {cq['pylint']['error_count']} errors")
        if cq.get('radon', {}).get('high_complexity_count', 0) > 0:
            warnings.append(f"High complexity: {cq['radon']['high_complexity_count']} functions")
        if cq.get('bandit', {}).get('high_severity_count', 0) > 0:
            warnings.append(f"Security: {cq['bandit']['high_severity_count']} high severity issues")

    # Integration
    if 'integration' in results:
        integ = results['integration']
        critical_components = ['gpu_manager', 'audio_processor', 'pipeline']
        for comp in critical_components:
            if not integ.get(comp, {}).get('passed'):
                critical_failures.append(f"Integration: {comp} failed")
                overall_status = "❌ FAILED"

    # Documentation
    if 'documentation' in results:
        doc = results['documentation']
        if not doc.get('doc_files', {}).get('passed'):
            warnings.append("Missing required documentation files")

    report.append(f"**Overall Status:** {overall_status}\n")

    if critical_failures:
        report.append("**Critical Failures:**")
        for failure in critical_failures:
            report.append(f"- {failure}")
        report.append("")

    if warnings:
        report.append("**Warnings:**")
        for warning in warnings:
            report.append(f"- {warning}")
        report.append("")

    # Code Quality Details
    report.append("\n## Code Quality\n")

    if 'code_quality' in results and 'error' not in results['code_quality']:
        cq = results['code_quality']

        report.append("### Linting & Style")
        report.append(f"- **Flake8:** {'✅ PASSED' if cq.get('flake8', {}).get('passed') else '❌ FAILED'}")
        report.append(f"- **Pylint Errors:** {cq.get('pylint', {}).get('error_count', 'N/A')}")
        report.append(f"- **Mypy:** {'✅ PASSED' if cq.get('mypy', {}).get('passed') else '❌ FAILED'}")

        report.append("\n### Complexity Analysis")
        report.append(f"- **High Complexity Functions:** {cq.get('radon', {}).get('high_complexity_count', 'N/A')}")

        report.append("\n### Security Analysis")
        report.append(f"- **High Severity Issues:** {cq.get('bandit', {}).get('high_severity_count', 'N/A')}")
    else:
        report.append("⚠️  Code quality validation not run or failed to load")

    # Integration Details
    report.append("\n## Integration Testing\n")

    if 'integration' in results and 'error' not in results['integration']:
        integ = results['integration']

        components = [
            ('GPU Manager', 'gpu_manager'),
            ('Audio Processor', 'audio_processor'),
            ('Web API', 'web_api'),
            ('Pipeline', 'pipeline'),
            ('CUDA Kernels', 'cuda_kernels')
        ]

        for name, key in components:
            if key in integ:
                status = '✅ PASSED' if integ[key].get('passed') else '❌ FAILED'
                report.append(f"- **{name}:** {status}")

                if not integ[key].get('passed') and 'error' in integ[key]:
                    report.append(f"  - Error: {integ[key]['error']}")
    else:
        report.append("⚠️  Integration validation not run or failed to load")

    # Documentation Details
    report.append("\n## Documentation\n")

    if 'documentation' in results and 'error' not in results['documentation']:
        doc = results['documentation']

        checks = [
            ('Required Files', 'doc_files'),
            ('Module Docstrings', 'docstrings'),
            ('Code Examples', 'code_examples'),
            ('README Links', 'readme_links'),
            ('API Documentation', 'api_docs')
        ]

        for name, key in checks:
            if key in doc:
                status = '✅ PASSED' if doc[key].get('passed') else '⚠️  ISSUES'
                report.append(f"- **{name}:** {status}")

                if 'missing_count' in doc[key] and doc[key]['missing_count'] > 0:
                    report.append(f"  - Missing: {doc[key]['missing_count']}")
    else:
        report.append("⚠️  Documentation validation not run or failed to load")

    # System Capabilities
    report.append("\n## System Capabilities\n")

    if 'system' in results and 'error' not in results['system']:
        sys_info = results['system']
        report.append(f"- **CUDA Available:** {'✅ Yes' if sys_info.get('cuda_available') else '❌ No'}")
        report.append(f"- **CUDA Version:** {sys_info.get('cuda_version', 'N/A')}")
        report.append(f"- **GPU Count:** {sys_info.get('gpu_count', 0)}")

        if sys_info.get('gpu_devices'):
            report.append("\n### GPU Devices:")
            for gpu in sys_info['gpu_devices']:
                report.append(f"- {gpu.get('name', 'Unknown')} ({gpu.get('memory', 'Unknown')})")
    else:
        report.append("⚠️  System validation not run or failed to load")

    # Performance Benchmarks
    report.append("\n## Performance Benchmarks\n")

    if 'performance_breakdown' in results and 'error' not in results['performance_breakdown']:
        perf = results['performance_breakdown']
        report.append(f"- **Device:** {perf.get('device', 'N/A')}")
        report.append(f"- **Audio Duration:** {perf.get('audio_duration_seconds', 0):.1f}s")
        report.append(f"- **Total Time:** {perf.get('total_time_seconds', 0):.3f}s")
        report.append(f"- **RTF (Real-Time Factor):** {perf.get('rtf', 0):.2f}x")

        if perf.get('gpu_utilization') is not None:
            report.append(f"- **Average GPU Utilization:** {perf.get('gpu_utilization'):.1f}%")

        # Stage breakdown table
        if perf.get('stage_timings_ms'):
            report.append("\n### Stage Breakdown:\n")
            report.append("| Stage | Time (ms) | Percentage |")
            report.append("|-------|-----------|------------|")

            stage_timings = perf['stage_timings_ms']
            stage_percentages = perf.get('stage_percentages', {})

            # Display stages in order
            stage_order = ['separation', 'f0_extraction', 'conversion', 'mixing', 'total']
            for stage in stage_order:
                if stage in stage_timings:
                    time_ms = stage_timings[stage]
                    pct = stage_percentages.get(stage, 0)
                    report.append(f"| {stage.replace('_', ' ').title()} | {time_ms:.2f} | {pct:.1f}% |")
    else:
        report.append("⚠️  Performance profiling not run or failed to load")

    # TensorRT Latency Results
    if 'latency_tensorrt' in results and 'error' not in results['latency_tensorrt']:
        report.append("\n## TensorRT Latency Test\n")
        latency = results['latency_tensorrt']

        report.append(f"- **Audio Duration:** {latency.get('duration_seconds', 0):.1f}s")
        report.append(f"- **Conversion Time:** {latency.get('elapsed_seconds', 0):.3f}s")
        report.append(f"- **RTF:** {latency.get('rtf', 0):.2f}x")
        report.append(f"- **Preset:** {latency.get('preset', 'N/A')}")
        report.append(f"- **TensorRT Requested:** {'✅ Yes' if latency.get('tensorrt_requested') else '❌ No'}")
        report.append(f"- **TensorRT Enabled:** {'✅ Yes' if latency.get('tensorrt_enabled') else '❌ No'}")
        report.append(f"- **TensorRT Precision:** {latency.get('tensorrt_precision', 'N/A')}")

        target_met = latency.get('target_met', False)
        target_status = '✅ PASSED' if target_met else '❌ FAILED'
        report.append(f"- **Target (<5s):** {target_status}")

    # Quality Evaluation Results
    if 'quality_eval' in results and results['quality_eval']:
        report.append("\n## Quality Evaluation\n")

        for eval_name, eval_data in results['quality_eval'].items():
            if 'metrics' in eval_data:
                report.append(f"\n### {eval_name}")
                metrics = eval_data['metrics']
                for metric_name, value in metrics.items():
                    report.append(f"- **{metric_name}:** {value:.3f}")

    # Test Results
    report.append("\n## System Validation Tests\n")

    if 'test_results' in results and 'error' not in results['test_results']:
        tests = results['test_results']
        report.append(f"- **Total Tests:** {tests.get('total', 'N/A')}")
        report.append(f"- **Passed:** {tests.get('passed', 'N/A')}")
        report.append(f"- **Failed:** {tests.get('failed', 'N/A')}")
        report.append(f"- **Duration:** {tests.get('duration', 'N/A')}s")
    else:
        report.append("⚠️  System validation tests not run or failed to load")

    # Docker Deployment
    if 'docker_log' in results:
        report.append("\n## Docker Deployment\n")
        report.append("Docker validation log available (see validation_results/docker_validation.log)")

    # Recommendations
    report.append("\n## Recommendations\n")

    if overall_status == "✅ PASSED":
        report.append("- All critical validation checks passed")
        if warnings:
            report.append("- Address warnings to improve code quality")
    else:
        report.append("- **CRITICAL:** Fix all failed checks before deployment")
        report.append("- Review error details in individual validation reports")

    report.append("\n## Next Steps\n")
    report.append("1. Review detailed results in `validation_results/` directory")
    report.append("2. Address any critical failures")
    report.append("3. Consider addressing warnings for improved quality")
    report.append("4. Re-run validation after fixes")

    return "\n".join(report)


def main() -> int:
    """Main report generation function.

    Returns:
        0 for success, 1 for failure
    """
    parser = argparse.ArgumentParser(
        description='Generate comprehensive validation report from all validation results'
    )

    # Input file arguments
    parser.add_argument(
        '--code-quality',
        type=str,
        help='Path to code quality validation results (default: auto-search)'
    )
    parser.add_argument(
        '--integration',
        type=str,
        help='Path to integration validation results (default: auto-search)'
    )
    parser.add_argument(
        '--documentation',
        type=str,
        help='Path to documentation validation results (default: auto-search)'
    )
    parser.add_argument(
        '--system-validation',
        type=str,
        help='Path to system validation results (default: auto-search)'
    )
    parser.add_argument(
        '--e2e-tests',
        type=str,
        help='Path to end-to-end test results (default: auto-search)'
    )
    parser.add_argument(
        '--performance',
        type=str,
        help='Path to performance profiling results (default: auto-search)'
    )
    parser.add_argument(
        '--security',
        type=str,
        help='Path to security validation results (default: auto-search)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default='FINAL_VALIDATION_REPORT.md',
        help='Output path for markdown report (default: FINAL_VALIDATION_REPORT.md)'
    )
    parser.add_argument(
        '--summary',
        type=str,
        default='validation_results/reports/summary.json',
        help='Output path for JSON summary (default: validation_results/reports/summary.json)'
    )

    args = parser.parse_args()

    print("=== Generating Validation Report ===\n")

    # Load all results with CLI argument support
    results = load_results(args)

    # Generate report
    report = generate_markdown_report(results)

    # Save report using PROJECT_ROOT
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report generated: {output_path}\n")

    # Also save JSON summary
    summary_path = PROJECT_ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Summary saved: {summary_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
