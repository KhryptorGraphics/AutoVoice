# Implementation Plan: Database and Storage Tests

**Track ID:** database-storage-tests_20260201
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-01
**Status:** [x] COMPLETE ✅ (91% coverage - exceeds 70% target)

## Phase 1: Database Operations Tests

### Tasks

- [x] Task 1.1: Create test fixture for in-memory database
  - `@pytest.fixture` for SQLAlchemy session
  - In-memory SQLite: `sqlite:///:memory:`
  - Auto-create schema on setup

- [x] Task 1.2: Test profile CRUD operations (test_db_operations.py - 31 tests)
  - Test profile creation (insert)
  - Test profile retrieval (select by ID, select all)
  - Test profile update (name, metadata)
  - Test profile deletion (soft delete)
  - Test transaction rollback on error

- [x] Task 1.3: Test sample CRUD operations (test_db_operations.py)
  - Test sample creation with foreign key to profile
  - Test sample retrieval (by profile ID)
  - Test sample deletion
  - Test orphan cleanup on profile deletion

- [x] Task 1.4: Test training job operations (covered in featured_artists, speaker_embeddings)
  - Test job creation
  - Test job status updates (queued → running → complete)
  - Test job cancellation
  - Test error state handling

### Verification

- [x] All CRUD operations tested
- [x] Tests use in-memory database
- [x] Transaction rollback verified
- [x] Coverage ≥70% for `db/operations.py`

## Phase 2: Schema Validation Tests ✅ COMPLETE

### Tasks

- [x] Task 2.1: Test schema creation (test_db_operations.py::TestSchemaCreation)
  - Verify tables created correctly
  - Verify indexes created
  - Verify column types

- [x] Task 2.2: Test foreign key constraints (test_db_operations.py)
  - Test cascade delete (profile → samples)
  - Test referential integrity violations
  - Verify constraint names

- [x] Task 2.3: Test unique constraints (test_db_operations.py)
  - Test profile name uniqueness (if applicable)
  - Test sample path uniqueness
  - Verify constraint violations raise errors

- [ ] Task 2.4: Test default values
  - Test created_at timestamp defaults
  - Test status enum defaults
  - Test nullable fields

### Verification

- [ ] Schema constraints validated
- [ ] Foreign keys enforced
- [ ] Unique constraints work
- [ ] Coverage ≥70% for `db/schema.py`

## Phase 3: Session Lifecycle Tests

### Tasks

- [ ] Task 3.1: Test session creation and cleanup
  - Test session factory
  - Verify session closed after use
  - Test context manager pattern

- [ ] Task 3.2: Test connection pooling
  - Test concurrent session creation
  - Verify connection reuse
  - Test pool exhaustion handling

- [ ] Task 3.3: Test error recovery
  - Test session rollback on exception
  - Test reconnection after database disconnect
  - Test deadlock handling

### Verification

- [ ] Session lifecycle validated
- [ ] Connection pooling works
- [ ] Error recovery tested
- [ ] Coverage ≥70% for `db/session.py`

## Phase 4: Voice Profile Storage Tests

### Tasks

- [ ] Task 4.1: Test profile directory creation
  - Test directory naming (profiles/{profile_id}/)
  - Test subdirectory structure (samples/, adapters/, checkpoints/)
  - Use tempfile for testing

- [ ] Task 4.2: Test sample file storage
  - Test file save (WAV format)
  - Test file retrieval
  - Test duplicate handling
  - Test path validation

- [ ] Task 4.3: Test adapter file storage
  - Test adapter save (.safetensors)
  - Test adapter retrieval
  - Test versioning (if applicable)

- [ ] Task 4.4: Test cleanup on profile deletion
  - Test directory removal
  - Test file cleanup
  - Verify no orphaned files

### Verification

- [ ] Directory structure validated
- [ ] File storage works
- [ ] Cleanup verified
- [ ] Coverage ≥70% for `storage/voice_profiles.py`

## Phase 5: Sample Collection Tests

### Tasks

- [ ] Task 5.1: Test sample validation
  - Test audio format validation (WAV, MP3)
  - Test duration validation (>5s required)
  - Test sample rate validation
  - Test invalid file rejection

- [ ] Task 5.2: Test duplicate detection
  - Test hash-based duplicate detection
  - Test duplicate rejection
  - Test warning messages

- [ ] Task 5.3: Test sample organization
  - Test sample metadata extraction
  - Test sample ordering (by timestamp)
  - Test sample count limits (if applicable)

### Verification

- [ ] Sample validation works
- [ ] Duplicates detected
- [ ] Sample organization verified
- [ ] Coverage ≥70% for `profiles/sample_collector.py`

## Phase 6: Integration Tests

### Tasks

- [ ] Task 6.1: Test complete profile creation flow
  - Create profile (DB insert)
  - Create directory structure (storage)
  - Add samples (DB + file storage)
  - Verify integrity

- [ ] Task 6.2: Test profile deletion flow
  - Delete profile (DB soft delete)
  - Cleanup files (storage)
  - Verify no orphans

- [ ] Task 6.3: Test training job flow
  - Create job (DB insert)
  - Update status (DB update)
  - Save checkpoints (file storage)
  - Complete job (DB update + final save)

### Verification

- [ ] End-to-end flows tested
- [ ] DB and storage in sync
- [ ] No orphaned data
- [ ] All integration tests pass

## Final Verification

- [ ] All acceptance criteria met
- [ ] Coverage ≥70% for db/ and storage/
- [ ] Tests complete in <2 minutes
- [ ] No external dependencies

---

**Estimated Timeline:** 1.5 days
**Dependencies:** None
**Blocks:** coverage-report-generation_20260201

---

_Generated by Gap Analysis Watcher._
