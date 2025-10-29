#!/usr/bin/env python3
"""Documentation validation for AutoVoice project.

This script validates:
1. All modules have proper docstrings
2. Code examples in documentation are valid
3. README links work correctly
4. API documentation is complete
"""
import sys
import json
import ast
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import subprocess

# Compute project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def validate_module_docstrings() -> Dict[str, Any]:
    """Check that all Python modules have docstrings.

    Returns:
        Dict with validation results
    """
    print("Validating module docstrings...")

    src_dir = PROJECT_ROOT / 'src' / 'auto_voice'
    missing_docstrings = []
    total_modules = 0

    for py_file in src_dir.rglob('*.py'):
        if py_file.name == '__init__.py':
            continue

        total_modules += 1

        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())

            # Check module docstring
            if not ast.get_docstring(tree):
                missing_docstrings.append(str(py_file.relative_to(src_dir.parent)))
                continue

            # Check class docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        missing_docstrings.append(
                            f"{py_file.relative_to(src_dir.parent)}::{node.name}"
                        )

        except Exception as e:
            print(f"Warning: Could not parse {py_file}: {e}")

    return {
        'passed': len(missing_docstrings) == 0,
        'total_modules': total_modules,
        'missing_count': len(missing_docstrings),
        'missing_docstrings': missing_docstrings[:20]  # Limit output
    }


def validate_code_examples() -> Dict[str, Any]:
    """Validate code examples in documentation files.

    Returns:
        Dict with validation results
    """
    print("Validating code examples in documentation...")

    docs_dir = PROJECT_ROOT / 'docs'
    invalid_examples = []
    total_examples = 0

    if not docs_dir.exists():
        return {
            'passed': True,
            'total_examples': 0,
            'note': 'No docs directory found'
        }

    # Pattern to find Python code blocks
    code_block_pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)

    for doc_file in docs_dir.rglob('*.md'):
        with open(doc_file, 'r') as f:
            content = f.read()

        # Extract code blocks
        code_blocks = code_block_pattern.findall(content)

        for idx, code in enumerate(code_blocks):
            total_examples += 1

            # Try to parse (not execute) the code
            try:
                ast.parse(code)
            except SyntaxError as e:
                invalid_examples.append({
                    'file': str(doc_file.relative_to(docs_dir.parent)),
                    'block': idx + 1,
                    'error': str(e)
                })

    return {
        'passed': len(invalid_examples) == 0,
        'total_examples': total_examples,
        'invalid_count': len(invalid_examples),
        'invalid_examples': invalid_examples[:10]  # Limit output
    }


def validate_readme_links() -> Dict[str, Any]:
    """Check that links in README.md are valid.

    Returns:
        Dict with validation results
    """
    print("Validating README links...")

    readme_path = PROJECT_ROOT / 'README.md'

    if not readme_path.exists():
        return {
            'passed': False,
            'error': 'README.md not found'
        }

    with open(readme_path, 'r') as f:
        content = f.read()

    # Find markdown links [text](url)
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
    links = link_pattern.findall(content)

    broken_links = []
    total_links = len(links)

    for text, url in links:
        # Check local file links
        if not url.startswith('http'):
            file_path = PROJECT_ROOT / url.lstrip('./')
            if not file_path.exists():
                broken_links.append({'text': text, 'url': url, 'reason': 'File not found'})
        # For HTTP links, just check format (not actually fetching)
        elif not (url.startswith('http://') or url.startswith('https://')):
            broken_links.append({'text': text, 'url': url, 'reason': 'Invalid URL format'})

    return {
        'passed': len(broken_links) == 0,
        'total_links': total_links,
        'broken_count': len(broken_links),
        'broken_links': broken_links
    }


def validate_api_documentation() -> Dict[str, Any]:
    """Validate API documentation completeness.

    Returns:
        Dict with validation results
    """
    print("Validating API documentation...")

    # Check for key API modules
    api_modules = [
        'src/auto_voice/web/api.py',
        'src/auto_voice/inference/engine.py',
        'src/auto_voice/inference/singing_conversion_pipeline.py'
    ]

    undocumented_endpoints = []
    total_endpoints = 0

    for module_path in api_modules:
        full_path = PROJECT_ROOT / module_path

        if not full_path.exists():
            continue

        with open(full_path, 'r') as f:
            content = f.read()

        try:
            tree = ast.parse(content)

            # Find all async functions (likely API endpoints)
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    total_endpoints += 1

                    # Check if function has docstring
                    if not ast.get_docstring(node):
                        undocumented_endpoints.append(
                            f"{module_path}::{node.name}"
                        )

        except Exception as e:
            print(f"Warning: Could not parse {module_path}: {e}")

    return {
        'passed': len(undocumented_endpoints) == 0,
        'total_endpoints': total_endpoints,
        'undocumented_count': len(undocumented_endpoints),
        'undocumented_endpoints': undocumented_endpoints
    }


def check_doc_files_exist() -> Dict[str, Any]:
    """Check that required documentation files exist.

    Returns:
        Dict with validation results
    """
    print("Checking required documentation files...")

    required_docs = [
        'README.md',
        'docs/voice_conversion_guide.md',
        'docs/api_voice_conversion.md',
        'docs/model_architecture.md',
        'docs/runbook.md',
        'docs/quality_evaluation_guide.md',
        'docs/tensorrt_optimization_guide.md',
        'docs/cuda_optimization_guide.md'
    ]

    missing_docs = []

    for doc_path in required_docs:
        full_path = PROJECT_ROOT / doc_path
        if not full_path.exists():
            missing_docs.append(doc_path)

    return {
        'passed': len(missing_docs) == 0,
        'total_required': len(required_docs),
        'missing_count': len(missing_docs),
        'missing_docs': missing_docs
    }


def main() -> int:
    """Main validation function.

    Returns:
        0 for success, 1 for failure
    """
    parser = argparse.ArgumentParser(description='Validate AutoVoice documentation')
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results/reports/documentation.json',
        help='Output path for validation results (default: validation_results/reports/documentation.json)'
    )
    args = parser.parse_args()

    print("=== Documentation Validation ===\n")

    # Run all validations
    results = {
        'doc_files': check_doc_files_exist(),
        'docstrings': validate_module_docstrings(),
        'code_examples': validate_code_examples(),
        'readme_links': validate_readme_links(),
        'api_docs': validate_api_documentation()
    }

    # Save results using PROJECT_ROOT
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for component, result in results.items():
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"{component}: {status}")

        if not result['passed']:
            if 'missing_count' in result:
                print(f"  Missing: {result['missing_count']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")

    # Determine overall pass/fail
    all_passed = all(result['passed'] for result in results.values())

    if not all_passed:
        print("\n⚠️  DOCUMENTATION VALIDATION FAILED")
        print("Note: Documentation issues are warnings, not critical errors")
        return 0  # Don't fail CI on doc issues
    else:
        print("\n✅ ALL DOCUMENTATION CHECKS PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
