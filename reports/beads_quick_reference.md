# Beads Quick Reference
**Updated**: 2026-02-01

## Current Status
- **Total**: 16 tasks
- **Ready to Work**: 8 (4×P0, 4×P1)
- **Blocked**: 6
- **Completed**: 5

## Ready to Launch (No Blockers)

### Priority 0 (Critical)
1. **AV-hx0**: SOTA Dual-Pipeline Swarm (10 agents)
   - Unblocks: AV-cvj, AV-95n, AV-l2e
   - Impact: Critical path blocker

2. **AV-4tg**: Generate pytest coverage report (3-5 agents)
   - Unblocks: AV-eeu, AV-mcf (partial)
   - Impact: Deployment prerequisite

3. **AV-39t**: E2E Voice Profile Training Workflows
   - Independent validation task

4. **AV-eey**: Coverage Report Generation (Phase 6)
   - Part of comprehensive testing coverage

### Priority 1 (High)
1. **AV-n0s**: Performance Validation Suite (2-3 agents)
   - Unblocks: AV-mcf (partial)
   - Benchmarks: 4 pipelines, 5 metrics

2. **AV-4kt**: Performance Benchmark Infrastructure
   - Foundation for performance testing

3. **AV-hjk**: Database and Storage Tests
   - Phase 3 of comprehensive coverage

4. **AV-cut**: Audio Processing Tests
   - Phase 2 of comprehensive coverage

## Blocked Tasks (Wait for Dependencies)

1. **AV-cvj**: Training-Inference Swarm (5 agents)
   - Blocked by: AV-hx0

2. **AV-95n**: Voice Profile Training E2E Validation
   - Blocked by: AV-hx0, AV-cvj

3. **AV-l2e**: Frontend Complete Integration
   - Blocked by: AV-hx0, AV-cvj

4. **AV-eeu**: API Documentation Suite
   - Blocked by: AV-4tg

5. **AV-mcf**: Production Deployment Preparation
   - Blocked by: AV-4tg, AV-n0s

6. **AV-maw**: Benchmark all 4 pipelines
   - Blocked by: AV-4kt

## Recommended Launch Sequence

### Wave 1 (Launch Now)
```bash
# Primary blocker - unblocks 3 tasks
Launch: AV-hx0 (10 agents)

# Parallel critical work
Launch: AV-4tg (3-5 agents)
Launch: AV-n0s (2-3 agents)
```

### Wave 2 (After AV-hx0)
```bash
Launch: AV-cvj (5 agents)
```

### Wave 3 (After AV-cvj)
```bash
Launch: AV-95n (2-3 agents)
Launch: AV-l2e (2-3 agents)
```

### Wave 4 (After AV-4tg)
```bash
Launch: AV-eeu (1-2 agents)
```

### Wave 5 (After AV-4tg + AV-n0s)
```bash
Launch: AV-mcf (3-5 agents)
```

## Quick Commands

```bash
# Check ready tasks
bd ready

# Check blocked tasks
bd blocked

# Update task status
bd update {issue-id} --status in_progress

# Close completed task
bd close {issue-id} --force --reason "Track completed"

# Sync after changes
bd sync --flush-only

# View task details
bd show {issue-id}
```

## Swarm Capacity Planning

**Total Agent Budget**: ~30 agents across all waves

### Wave 1 (15-18 agents)
- AV-hx0: 10 agents
- AV-4tg: 3-5 agents
- AV-n0s: 2-3 agents

### Wave 2 (5 agents)
- AV-cvj: 5 agents

### Wave 3 (4-6 agents)
- AV-95n: 2-3 agents
- AV-l2e: 2-3 agents

### Wave 4 (1-2 agents)
- AV-eeu: 1-2 agents

### Wave 5 (3-5 agents)
- AV-mcf: 3-5 agents

## Critical Path Timeline

```
Day 1-2: AV-hx0 (SOTA Dual-Pipeline)
         AV-4tg (Testing Coverage) [parallel]
         AV-n0s (Performance Validation) [parallel]

Day 3-4: AV-cvj (Training-Inference)

Day 5-6: AV-95n (Voice Profile E2E)
         AV-l2e (Frontend Integration) [parallel]
         AV-eeu (API Docs) [parallel]

Day 7-8: AV-mcf (Production Deployment)
```

**Estimated Total**: 7-8 days for full critical path completion
