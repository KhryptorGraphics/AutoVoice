#!/usr/bin/env python3
"""
Coverage Analysis Script
Parses coverage.json and generates detailed analysis report with component breakdown
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def load_coverage_data(coverage_file: str = "coverage.json") -> Dict:
    """Load coverage data from JSON file."""
    if not os.path.exists(coverage_file):
        print(f"❌ Coverage file not found: {coverage_file}")
        print("   Run: pytest --cov=src/auto_voice --cov-report=json")
        sys.exit(1)

    with open(coverage_file, 'r') as f:
        return json.load(f)


def analyze_module_coverage(coverage_data: Dict) -> List[Tuple[str, float, Dict]]:
    """Analyze coverage by module and return sorted results."""
    modules = []

    files = coverage_data.get('files', {})

    for file_path, file_data in files.items():
        summary = file_data.get('summary', {})

        # Extract metrics
        covered = summary.get('covered_lines', 0)
        total = summary.get('num_statements', 0)
        missing = summary.get('missing_lines', 0)
        excluded = summary.get('excluded_lines', 0)

        # Calculate percentage
        if total > 0:
            percentage = (covered / total) * 100
        else:
            percentage = 100.0 if total == 0 else 0.0

        # Get missing line numbers
        missing_lines = file_data.get('missing_lines', [])
        excluded_lines = file_data.get('excluded_lines', [])

        modules.append((
            file_path,
            percentage,
            {
                'covered': covered,
                'total': total,
                'missing': missing,
                'excluded': excluded,
                'missing_lines': missing_lines,
                'excluded_lines': excluded_lines,
            }
        ))

    # Sort by coverage percentage (lowest first)
    modules.sort(key=lambda x: x[1])

    return modules


def format_line_ranges(lines: List[int]) -> str:
    """Format line numbers into readable ranges."""
    if not lines:
        return "None"

    lines = sorted(lines)
    ranges = []
    start = lines[0]
    prev = lines[0]

    for line in lines[1:]:
        if line != prev + 1:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = line
        prev = line

    # Add final range
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    # Limit display
    if len(ranges) > 10:
        return ", ".join(ranges[:10]) + f"... (+{len(ranges)-10} more)"

    return ", ".join(ranges)


def categorize_by_component(modules: List[Tuple[str, float, Dict]]) -> Dict[str, List]:
    """Group files into components by path prefix."""
    components = {
        'audio': [],
        'models': [],
        'inference': [],
        'cuda_kernels': [],
        'training': [],
        'web': [],
        'core': []
    }

    for file_path, percent, details in modules:
        # Determine component from path
        path_lower = file_path.lower()

        if 'audio' in path_lower:
            components['audio'].append((file_path, percent, details))
        elif 'model' in path_lower:
            components['models'].append((file_path, percent, details))
        elif 'inference' in path_lower:
            components['inference'].append((file_path, percent, details))
        elif 'cuda' in path_lower or 'kernel' in path_lower:
            components['cuda_kernels'].append((file_path, percent, details))
        elif 'train' in path_lower:
            components['training'].append((file_path, percent, details))
        elif 'web' in path_lower or 'api' in path_lower or 'websocket' in path_lower:
            components['web'].append((file_path, percent, details))
        else:
            components['core'].append((file_path, percent, details))

    return components


def calculate_component_coverage(component_files: List[Tuple[str, float, Dict]]) -> Dict:
    """Calculate weighted coverage for a component."""
    if not component_files:
        return {'coverage': 0.0, 'covered': 0, 'total': 0, 'files': 0}

    total_covered = sum(details['covered'] for _, _, details in component_files)
    total_statements = sum(details['total'] for _, _, details in component_files)

    coverage = (total_covered / total_statements * 100) if total_statements > 0 else 0.0

    return {
        'coverage': coverage,
        'covered': total_covered,
        'total': total_statements,
        'files': len(component_files)
    }


def identify_coverage_gaps(modules: List[Tuple[str, float, Dict]]) -> Dict[str, List]:
    """Identify and categorize coverage gaps by priority."""
    gaps = {
        'P0': [],  # 0% coverage files
        'P1': [],  # Public API files with <50% coverage
        'P2': []   # All files with <80% coverage
    }

    for file_path, percent, details in modules:
        file_name = Path(file_path).name

        # P0: 0% coverage
        if percent == 0:
            gaps['P0'].append({
                'file': file_path,
                'coverage': percent,
                'covered': details['covered'],
                'total': details['total']
            })

        # P1: Public API files with <50% coverage
        if percent < 50 and (file_name == '__init__.py' or file_name == 'api.py'):
            gaps['P1'].append({
                'file': file_path,
                'coverage': percent,
                'covered': details['covered'],
                'total': details['total']
            })

        # P2: All files with <80% coverage
        if percent < 80:
            gaps['P2'].append({
                'file': file_path,
                'coverage': percent,
                'covered': details['covered'],
                'total': details['total']
            })

    return gaps


def generate_coverage_gaps_json(coverage_data: Dict, modules: List[Tuple[str, float, Dict]],
                                  components: Dict, gaps: Dict) -> Dict:
    """Generate machine-readable JSON output with coverage gaps."""
    totals = coverage_data.get('totals', {})
    overall_coverage = totals.get('percent_covered', 0)

    # Calculate component coverage
    component_coverage = {}
    for comp_name, comp_files in components.items():
        if comp_files:
            comp_stats = calculate_component_coverage(comp_files)
            component_coverage[comp_name] = comp_stats

    # Generate recommendations based on gaps
    recommendations = []

    if gaps['P0']:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Zero Coverage Files',
            'count': len(gaps['P0']),
            'action': 'Add basic test coverage for files with 0% coverage',
            'files': [g['file'] for g in gaps['P0'][:5]]
        })

    if gaps['P1']:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Public API Coverage',
            'count': len(gaps['P1']),
            'action': 'Improve coverage for public API files (__init__.py, api.py)',
            'files': [g['file'] for g in gaps['P1'][:5]]
        })

    if len(gaps['P2']) > 10:
        # Focus on lowest coverage files
        lowest = sorted(gaps['P2'], key=lambda x: x['coverage'])[:10]
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Low Coverage Modules',
            'count': len(gaps['P2']),
            'action': 'Increase coverage for modules below 80% threshold',
            'files': [g['file'] for g in lowest]
        })

    return {
        'timestamp': datetime.now().isoformat(),
        'overall_coverage': round(overall_coverage, 2),
        'target_coverage': 80.0,
        'coverage_gap': round(max(0, 80.0 - overall_coverage), 2),
        'total_files': len(modules),
        'component_coverage': component_coverage,
        'gaps': {
            'P0_zero_coverage': {
                'count': len(gaps['P0']),
                'files': gaps['P0']
            },
            'P1_public_api_low': {
                'count': len(gaps['P1']),
                'files': gaps['P1']
            },
            'P2_below_threshold': {
                'count': len(gaps['P2']),
                'files': gaps['P2'][:20]  # Limit to first 20 for readability
            }
        },
        'recommendations': recommendations
    }


def generate_report(coverage_data: Dict, modules: List[Tuple[str, float, Dict]]) -> str:
    """Generate detailed coverage analysis report."""

    # Get totals
    totals = coverage_data.get('totals', {})
    total_covered = totals.get('covered_lines', 0)
    total_statements = totals.get('num_statements', 0)
    total_missing = totals.get('missing_lines', 0)
    total_excluded = totals.get('excluded_lines', 0)
    overall_percent = totals.get('percent_covered', 0)

    # Build report
    report = []
    report.append("# Coverage Analysis Report")
    report.append("")
    report.append("## Overall Coverage")
    report.append("")
    report.append(f"- **Total Coverage:** {overall_percent:.1f}%")
    report.append(f"- **Covered Lines:** {total_covered:,}")
    report.append(f"- **Total Statements:** {total_statements:,}")
    report.append(f"- **Missing Lines:** {total_missing:,}")
    report.append(f"- **Excluded Lines:** {total_excluded:,}")
    report.append("")

    # Status assessment
    if overall_percent >= 80:
        status = "✅ **Status:** Target met (≥80%)"
    elif overall_percent >= 70:
        status = "⚠️  **Status:** Close to target (70-80%)"
    else:
        status = "❌ **Status:** Below target (<70%)"

    report.append(status)
    report.append("")

    # Modules below threshold
    report.append("## Modules Requiring Attention")
    report.append("")
    report.append("Modules with coverage below 80%:")
    report.append("")

    below_threshold = [m for m in modules if m[1] < 80]

    if below_threshold:
        report.append("| Module | Coverage | Covered/Total | Missing Lines |")
        report.append("|--------|----------|---------------|---------------|")

        for file_path, percent, details in below_threshold:
            module_name = Path(file_path).name
            covered = details['covered']
            total = details['total']
            missing_lines = format_line_ranges(details['missing_lines'][:5])

            report.append(f"| `{module_name}` | {percent:.1f}% | {covered}/{total} | {missing_lines} |")

        report.append("")
    else:
        report.append("✅ All modules have ≥80% coverage!")
        report.append("")

    # Top priority improvements
    report.append("## Priority Improvements")
    report.append("")

    if below_threshold:
        report.append("### Critical Gaps (Lowest Coverage)")
        report.append("")

        critical = below_threshold[:5]
        for i, (file_path, percent, details) in enumerate(critical, 1):
            module_name = Path(file_path).name
            missing_lines = details['missing_lines']

            report.append(f"#### {i}. `{module_name}` ({percent:.1f}% coverage)")
            report.append("")
            report.append(f"- **Path:** `{file_path}`")
            report.append(f"- **Coverage:** {details['covered']}/{details['total']} lines")
            report.append(f"- **Missing:** {details['missing']} lines")

            if missing_lines:
                report.append(f"- **Missing Line Ranges:** {format_line_ranges(missing_lines)}")

            report.append("")
            report.append("**Recommended Actions:**")

            if percent < 50:
                report.append("- 🔴 **Critical:** Add comprehensive test suite for this module")
                report.append("- Create unit tests covering main functionality")
                report.append("- Add integration tests for workflows")
            elif percent < 70:
                report.append("- 🟡 **Important:** Expand existing test coverage")
                report.append("- Add tests for uncovered code paths")
                report.append("- Focus on edge cases and error handling")
            else:
                report.append("- 🟢 **Minor:** Add tests for remaining edge cases")
                report.append("- Cover error handling scenarios")
                report.append("- Add boundary condition tests")

            report.append("")
    else:
        report.append("🎉 No critical coverage gaps! All modules meet the 80% threshold.")
        report.append("")

    # Well-covered modules
    report.append("## Well-Covered Modules")
    report.append("")
    report.append("Modules with excellent coverage (≥90%):")
    report.append("")

    excellent = [m for m in modules if m[1] >= 90]

    if excellent:
        report.append("| Module | Coverage | Covered/Total |")
        report.append("|--------|----------|---------------|")

        for file_path, percent, details in excellent[-10:]:  # Show top 10
            module_name = Path(file_path).name
            covered = details['covered']
            total = details['total']

            report.append(f"| `{module_name}` | {percent:.1f}% | {covered}/{total} |")

        report.append("")
    else:
        report.append("No modules with ≥90% coverage yet.")
        report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    if overall_percent < 80:
        gap = 80 - overall_percent
        report.append(f"To reach 80% coverage, increase overall coverage by {gap:.1f} percentage points.")
        report.append("")
        report.append("**Action Plan:**")
        report.append("")
        report.append("1. **Prioritize modules with <50% coverage** (highest impact)")
        report.append("2. **Add tests for uncovered critical paths**")
        report.append("3. **Focus on modules with high complexity** (more bugs likely)")
        report.append("4. **Expand integration tests** (catch interaction bugs)")
        report.append("5. **Re-run coverage analysis** after each improvement")
        report.append("")

        # Estimate lines to test
        lines_needed = int((80 - overall_percent) / 100 * total_statements)
        report.append(f"**Estimated effort:** Add tests covering ~{lines_needed:,} additional lines")
        report.append("")
    else:
        report.append("✅ Coverage target met! Consider these stretch goals:")
        report.append("")
        report.append("1. **Aim for 90% coverage** for production-critical modules")
        report.append("2. **Add more integration tests** to catch interaction bugs")
        report.append("3. **Test edge cases and error paths** more thoroughly")
        report.append("4. **Add performance regression tests**")
        report.append("")

    # Testing strategy
    report.append("## Testing Strategy")
    report.append("")
    report.append("### Quick Wins (High Impact, Low Effort)")
    report.append("")

    quick_wins = [m for m in modules if 60 <= m[1] < 80 and m[2]['total'] < 100]

    if quick_wins[:3]:
        for file_path, percent, details in quick_wins[:3]:
            module_name = Path(file_path).name
            missing = details['missing']
            report.append(f"- `{module_name}`: Add {missing} lines of tests to reach 80%")
        report.append("")
    else:
        report.append("- Focus on modules listed in Priority Improvements")
        report.append("")

    report.append("### Long-term Improvements")
    report.append("")
    report.append("1. **Maintain coverage during development** (run tests before commits)")
    report.append("2. **Set up pre-commit hooks** to enforce minimum coverage")
    report.append("3. **Add coverage gates to CI/CD** (fail if coverage drops)")
    report.append("4. **Regular coverage reviews** (weekly/monthly)")
    report.append("5. **Document testing patterns** for team consistency")
    report.append("")

    # Appendix
    report.append("## Appendix: All Modules")
    report.append("")
    report.append("Complete coverage breakdown by module:")
    report.append("")
    report.append("| Module | Coverage | Covered/Total | Status |")
    report.append("|--------|----------|---------------|--------|")

    for file_path, percent, details in sorted(modules, key=lambda x: x[1]):
        module_name = Path(file_path).name
        covered = details['covered']
        total = details['total']

        if percent >= 90:
            status = "🟢 Excellent"
        elif percent >= 80:
            status = "✅ Good"
        elif percent >= 70:
            status = "⚠️  Fair"
        else:
            status = "❌ Poor"

        report.append(f"| `{module_name}` | {percent:.1f}% | {covered}/{total} | {status} |")

    report.append("")
    report.append("---")
    report.append("")
    report.append("*Generated by `scripts/analyze_coverage.py`*")

    return "\n".join(report)


def main():
    """Main execution function."""
    print("🔍 Analyzing coverage data...")

    # Load coverage data
    try:
        coverage_data = load_coverage_data()
    except Exception as e:
        print(f"❌ Error loading coverage data: {e}")
        sys.exit(1)

    # Analyze modules
    modules = analyze_module_coverage(coverage_data)

    # Categorize by component
    components = categorize_by_component(modules)

    # Identify coverage gaps
    gaps = identify_coverage_gaps(modules)

    # Generate markdown report
    report = generate_report(coverage_data, modules)

    # Write markdown report
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "coverage_analysis_report.md"

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✅ Coverage analysis complete!")
    print(f"   Markdown Report: {output_file}")

    # Generate and write JSON gaps output
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    json_output = generate_coverage_gaps_json(coverage_data, modules, components, gaps)
    json_file = logs_dir / "coverage_gaps.json"

    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"   JSON Gaps: {json_file}")

    # Display summary
    totals = coverage_data.get('totals', {})
    overall_percent = totals.get('percent_covered', 0)

    print(f"\n📊 Summary:")
    print(f"   Overall Coverage: {overall_percent:.1f}%")
    print(f"   Target: 80%")

    # Display component breakdown
    print(f"\n📦 Component Coverage:")
    for comp_name, comp_files in components.items():
        if comp_files:
            comp_stats = calculate_component_coverage(comp_files)
            print(f"   {comp_name:15s}: {comp_stats['coverage']:5.1f}% ({comp_stats['files']} files)")

    # Display gap summary
    print(f"\n🔍 Coverage Gaps:")
    print(f"   P0 (Zero coverage):     {len(gaps['P0'])} files")
    print(f"   P1 (Public API <50%):   {len(gaps['P1'])} files")
    print(f"   P2 (Below 80%):         {len(gaps['P2'])} files")

    if overall_percent >= 80:
        print(f"\n   Status: ✅ Target Met")
        return 0
    elif overall_percent >= 70:
        print(f"\n   Status: ⚠️  Close to Target")
        return 1
    else:
        print(f"\n   Status: ❌ Below Target")
        return 2


if __name__ == "__main__":
    sys.exit(main())
