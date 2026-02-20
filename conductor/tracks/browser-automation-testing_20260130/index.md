# Track: Browser Automation Testing

**ID:** browser-automation-testing_20260130
**Status:** In Progress

## Documents

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)

## Progress

- Phases: 2/7 complete
- Tasks: 14/34 complete
- Bug fixes: 1 (frontend field name mismatch)

## Current Phase

**Phase 3: Voice Profiles CRUD Testing**
- BUG FIXED: `audio` → `reference_audio` in api.ts:462
- Frontend rebuilt
- Ready to retry profile creation with Connor audio file
- Collaborative testing: user clicks, I type/monitor

## Pending Requirements

- Voice/instrumental separation needed before profile training
- Backend voice_cloner.py doesn't currently separate vocals

## Quick Links

- [Back to Tracks](../../tracks.md)
- [Product Context](../../product.md)

## Notes

- RDP session for real-time testing (DISPLAY=:10)
- Collaborative testing: user drives browser, Claude monitors backend
- Three-layer memory: Cipher + claude-mem + Conductor

## Cross-Context Memory

- **Cipher**: Global persistent state, project decisions
- **Claude-mem**: Session tags for quick retrieval
- **Conductor**: This track - phase/task progress

---

## Session 2 Progress (2026-01-30 ~10:00 AM)

### Completed
- Phase 3: Voice Profiles CRUD Testing ✅
- Fixed profile name storage (was showing UUIDs)
- Implemented automatic vocal separation with Demucs

### Next Session Tasks
1. Create Conor Maynard profile WITH vocal separation
2. Train both artist models via web UI
3. Voice swap: Connor↔William instrumentals

See RESUME.md for full details.
