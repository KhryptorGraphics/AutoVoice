# Cross-Context Action Items - Post Session 2026-02-02

**Generated:** 2026-02-02 after multi-agent testing orchestration
**Session:** Master Orchestrator - 6 parallel agents

## Immediate Actions (P0 - Start Next Session)

### 1. Install Missing Dependencies ⚠️ BLOCKING
**Priority:** P0 - Blocks 10+ tests
**Impact:** +3% coverage
**Effort:** 0.5 days

```bash
# Missing packages causing test failures
pip install demucs  # For vocal separation tests
pip install local-attention  # For transformer models
```

**Tests Blocked:**
- test_audio_separation.py (10 tests)
- test_audio_speaker_diarization.py (some tests)

---

### 2. Fix Validation Issues ⚠️
**Priority:** P0
**Impact:** +2% test pass rate
**Effort:** 0.5 days

**Issues:**
- Mock complexity with random seeds (8 tests)
- Database import mocking issues (4 tests)
- Minor edge cases (9 tests)

**Files:**
- tests/test_audio_speaker_diarization.py
- tests/test_database_storage.py

---

### 3. Fill Critical Coverage Gaps (0% modules)
**Priority:** P0 - Critical functionality untested
**Impact:** +4.7% coverage
**Effort:** 2 days

#### Module: voice_identifier.py (0% → 70%)
**Effort:** 0.5 days
**Tests Needed:** Voice matching, similarity threshold, profile ID, error handling

#### Module: mean_flow_decoder.py (0% → 70%)
**Effort:** 0.5 days
**Tests Needed:** Decoder init, forward pass, KV cache, streaming

#### Module: conversion_quality_analyzer.py (0% → 70%)
**Effort:** 0.5 days
**Tests Needed:** Quality metrics, MCD, similarity, reports

---

## Timeline: 21 days to production ready (80% coverage + enhancements)

**Next Session Focus:** Coverage gap-filling (63% → 80%)
**Agents:** 4-5 parallel agents for P0/P1 gaps
