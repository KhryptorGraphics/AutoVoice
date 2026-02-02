# Spec: Database and Storage Tests

**Track ID:** database-storage-tests_20260201
**Priority:** P1 (HIGH)
**Created:** 2026-02-01

## Problem

Phase 3 of comprehensive-testing-coverage_20260201 (Database and Storage Tests) is incomplete. No tests exist for CRUD operations, schema validation, or file storage.

## Goal

Create comprehensive tests for database operations, schema validation, and voice profile file storage. Target 70% coverage for db/ and storage/ modules.

## Acceptance Criteria

1. All CRUD operations tested (create, read, update, delete)
2. Schema validation tested (foreign keys, constraints, defaults)
3. Session lifecycle tested (connection pooling, cleanup, error recovery)
4. Voice profile storage tested (directory creation, file storage, cleanup)
5. Sample collection tested (validation, duplicates, organization)
6. All tests use in-memory SQLite (no external dependencies)
7. Coverage ≥70% for `src/auto_voice/db/` and storage modules
8. Tests complete in <2 minutes

## Context

**Modules to Test:**
- `db/operations.py` - CRUD operations
- `db/schema.py` - Data model validation
- `db/session.py` - Connection lifecycle
- `storage/voice_profiles.py` - File storage
- `profiles/sample_collector.py` - Sample collection

**Upstream Dependencies:**
- None (can start immediately)

**Downstream Impact:**
- Contributes to overall 80% coverage target
- Validates data integrity
- Enables confident database refactoring

## Out of Scope

- Production database migration scripts
- Database performance optimization
- Distributed database support

## Technical Constraints

- Use in-memory SQLite for all tests (fast, no side effects)
- No file system writes (use tempfile or mock)
- Tests must be deterministic (no random data without seeding)
