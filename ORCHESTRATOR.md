# AutoVoice Master Orchestrator Stack

## Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     RALPH ORCHESTRATOR                       │
│  (Top-level workflow coordination via PROMPT.md)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐         ┌─────────────────────────┐    │
│  │     CIPHER      │◄───────►│       SERENA            │    │
│  │ Context Memory  │         │  Code-level Memory      │    │
│  └────────┬────────┘         └───────────┬─────────────┘    │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    BEADS                             │    │
│  │  Task Management: Epics, Tasks, Dependencies         │    │
│  │  Commands: bd list, bd ready, bd update, bd close    │    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  CONDUCTOR                           │    │
│  │  Track Planning: Phases, Tasks, Verification         │    │
│  │  Files: conductor/tracks/{track_id}/plan.md          │    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              CLAUDE-FLOW SWARMS                      │    │
│  │  Parallel Execution: Agents work on tasks            │    │
│  │  Strategies: development, testing, research          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### 1. RALPH (Top Level)
- Reads PROMPT.md for sprint objectives
- Coordinates overall development workflow
- Triggers Beads task operations
- Persists context to Cipher/Serena

### 2. CIPHER (Context Memory)
- Stores development context across compactions
- Query: `mcp__cipher__ask_cipher "AutoVoice status"`
- Key memories:
  - "AutoVoice Development Stack"
  - "SOTA Dual-Pipeline Status"
  - "Training-Inference Integration"

### 3. SERENA (Code Memory)
- Stores code-level context
- Activate: `mcp__serena__activate_project project="/home/kp/repo2/autovoice"`
- Memory file: `.serena/memories/sota-dual-pipeline-2026-01-30.md`

### 4. BEADS (Task Management)
- Manages epics and tasks
- Tracks dependencies
- Commands:
  ```bash
  bd list           # All tasks
  bd ready          # Unblocked tasks
  bd update ID --status in_progress  # Claim
  bd close ID --force --reason "..."  # Complete
  ```

### 5. CONDUCTOR (Track Planning)
- Detailed phase-by-phase plans
- Files: `conductor/tracks/{track_id}/plan.md`
- Commands:
  ```bash
  /conductor:status           # View progress
  /conductor:implement ID     # Execute track
  ```

### 6. CLAUDE-FLOW SWARMS (Parallel Execution)
- Launch agents for parallel work
- Strategies: development, testing, research
- Command:
  ```bash
  claude-flow swarm "task" --strategy development --parallel --max-agents 8
  ```

## Orchestration Flow

```
1. Ralph reads PROMPT.md
   │
2. Query Cipher for context
   │
3. Check Beads: bd ready
   │
4. For each ready task:
   │
   ├──► Check Conductor plan for task details
   │
   ├──► Launch claude-flow swarm for implementation
   │
   ├──► On completion:
   │    ├──► Update Beads: bd close ID --force
   │    ├──► Update Conductor: mark task [x]
   │    ├──► Store to Cipher: progress update
   │    └──► Store to Serena: code context
   │
   └──► Loop until all tasks complete
```

## Current State

### Active Tasks (Beads)
- AV-5k7 [in_progress]: REALTIME_PIPELINE
- AV-u6e [open]: QUALITY_PIPELINE
- AV-v7p [open]: AdapterManager
- AV-d11 [open]: Web UI selector
- AV-508 [blocked]: HQ-SVC enhancement
- AV-8k8 [blocked]: SmoothSinger concepts

### Active Tracks (Conductor)
- sota-dual-pipeline_20260130
- training-inference-integration_20260130

## Resume Commands

```bash
cd /home/kp/repo2/autovoice

# 1. Query Cipher for context
mcp__cipher__ask_cipher "What is the AutoVoice development status?"

# 2. Check ready tasks
bd ready

# 3. View Conductor tracks
cat conductor/tracks.md

# 4. Read orchestrator config
cat ORCHESTRATOR.md

# 5. Read sprint objectives
cat PROMPT.md

# 6. Start swarm for next task
claude-flow swarm "Implement AV-XXX: <task description>" \
  --strategy development \
  --background \
  --monitor \
  --testing \
  --parallel \
  --max-agents 6
```

---
Created: 2026-01-30 11:45 CST
