# Spec: Coverage Report Generation

**Track ID:** coverage-report-generation_20260201
**Priority:** P0 (CRITICAL)
**Created:** 2026-02-01

## Problem

Phase 6 of comprehensive-testing-coverage_20260201 is incomplete. We have 1,562 tests but no coverage report to verify the 80% target.

## Goal

Generate pytest-cov HTML report, identify modules <70% coverage, and fill gaps to reach 80% overall coverage.

## Acceptance Criteria

1. `pytest --cov=src/auto_voice --cov-report=html` executes successfully
2. HTML report generated in `htmlcov/` directory
3. Overall coverage ≥80% for `src/auto_voice/`
4. Inference coverage ≥85%
5. No critical modules (<70%) in inference/, web/, audio/
6. Coverage report committed to git
7. CLAUDE.md updated with coverage patterns

## Context

**Upstream Dependencies:**
- comprehensive-testing-coverage_20260201 Phases 1-5 (complete)

**Downstream Impact:**
- Blocks production deployment confidence
- Required for CI/CD integration
- Enables confident refactoring

## Out of Scope

- New feature tests (only gap filling)
- Performance benchmarks (separate track)
- E2E UI tests (separate track)

## Technical Constraints

- Test suite must complete in <20 minutes
- Use fixtures from `tests/fixtures/` (no new audio downloads)
- Mark slow tests with `@pytest.mark.slow`
