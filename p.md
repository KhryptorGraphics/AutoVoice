# Ralph Full-Stack Orchestrator Prompt

## Usage

```bash
# From any project directory:
cat ~/repos/autovoice/p.md | claude --dangerously-skip-permissions

# Or copy p.md to your project first:
cp ~/repos/autovoice/p.md /path/to/project/
cd /path/to/project && cat p.md | claude --dangerously-skip-permissions

# Or just paste the contents of this file into Claude Code directly.
```

---

## Instructions

You are initializing the Ralph autonomous development orchestrator with full-stack memory, self-healing, and self-upgrading capabilities. Follow ALL phases below in order. Do not skip phases. Each phase builds on the previous.

---

## Phase 1: Deep Project Analysis

### 1.1 Codebase Discovery
- Use serena tools (`get_symbols_overview`, `find_symbol`, `list_dir`) to map the project structure
- Identify: language, framework, package manager, build system, test framework
- Find: entry points, main modules, configuration files
- Map: class hierarchy, module dependencies, data flow

### 1.2 Environment Detection
- Find conda env / virtualenv / nvm / etc from config files or shell history
- Detect CUDA/GPU requirements (check for .cu files, torch imports, GPU configs)
- Identify platform constraints (aarch64, x86, docker, etc.)
- Record exact test command (pytest, jest, cargo test, go test, etc.)
- Record exact build command if applicable

### 1.3 Architecture Analysis
- Identify core abstractions (key classes/interfaces/types)
- Map data flow through the system (input -> processing -> output)
- Find critical invariants (error handling patterns, type constraints, no-fallback rules)
- Document public API surface
- Note any existing test coverage patterns

### 1.4 Existing State Assessment
- Run the test suite and record pass/fail counts
- Identify broken tests vs working tests
- Check git log for recent work direction
- Look for TODO/FIXME/HACK comments indicating known issues
- Check for any existing task/issue tracking (PROMPT.md, TODO.md, etc.)

### 1.5 Academic Research (if SOTA/ML project)
- Search arxiv for relevant papers: `mcp__research-arxiv__search_arxiv`
- Search semantic scholar for citation graphs: `mcp__research-semantic-scholar__search_semantic_scholar`
- Check for reference implementations in nearby directories
- Create `academic-research/bibliography.md` with papers, arxiv IDs, relevance
- Create `academic-research/<topic>-analysis.md` with architecture analysis

### 1.6 Cross-Session Context Recovery
Before starting fresh analysis, check if previous sessions already did work:
```
# Claude-mem: search for project observations
mcp__plugin_claude-mem_mcp-search__search: query="<project-name>"

# Serena: check for existing memories
mcp__serena__list_memories

# Cipher: check for state snapshots
mcp__cipher__ask_cipher: "What do you know about the <project-name> project?"
```
If prior context exists, build on it instead of re-analyzing from scratch.

---

## Phase 2: Task Decomposition

### 2.1 Define the Goal
- What is the project trying to achieve? (from README, PROMPT.md, or user input)
- What is the current gap between existing state and goal?
- What are the priority areas (bugs first, then features, then optimizations)?

### 2.2 Create User Stories
Break the goal into atomic stories that each:
- Can be completed in ONE context window (the critical constraint)
- Have verifiable acceptance criteria (specific test commands, output shapes, etc.)
- Are ordered by dependency (no story depends on a later story)
- Include the primary file being changed and any research references

### 2.3 Initialize Beads
```bash
# For local filesystems:
mcp__beads__init

# For network mounts (CIFS/NFS/SMB - SQLite WAL mode fails):
mkdir -p /home/kp/.beads-local/<project-name>/
bd --no-daemon --db /home/kp/.beads-local/<project-name>/beads.db init
```

Create beads tasks for each user story:
- One epic for the overall feature
- Child tasks for each story with `dep_type: "parent-child"`

---

## Phase 3: Ralph File Creation

### 3.1 Create `scripts/ralph/` directory

### 3.2 Create `scripts/ralph/ralph.sh`
```bash
#!/bin/bash
set -e

TOOL="claude"
MAX_ITERATIONS=50
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    DATE=$(date +%Y-%m-%d)
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"
    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  [ -n "$CURRENT_BRANCH" ] && echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
fi

# Initialize progress file if missing
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tool) TOOL="$2"; shift 2 ;;
    --tool=*) TOOL="${1#*=}"; shift ;;
    *) [[ "$1" =~ ^[0-9]+$ ]] && MAX_ITERATIONS="$1"; shift ;;
  esac
done

echo "Starting Ralph - Tool: $TOOL - Max iterations: $MAX_ITERATIONS"

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "==============================================================="

  OUTPUT=$(claude --dangerously-skip-permissions --print < "$SCRIPT_DIR/CLAUDE.md" 2>&1 | tee /dev/stderr) || true

  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "Ralph completed all tasks!"
    exit 0
  fi

  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo "Ralph reached max iterations ($MAX_ITERATIONS)."
exit 1
```

Make executable: `chmod +x scripts/ralph/ralph.sh`

### 3.3 Create `scripts/ralph/prd.json`
Use the user stories from Phase 2. Format:
```json
{
  "project": "<ProjectName>",
  "branchName": "ralph/<feature-name>",
  "description": "<Feature description>",
  "userStories": [
    {
      "id": "<PRJ-NNN>",
      "title": "<Story title>",
      "description": "<What to implement>",
      "acceptanceCriteria": [
        "<Specific verifiable criterion>",
        "<Test command: exact command to run>"
      ],
      "priority": 1,
      "passes": false,
      "notes": "beads: <ISSUE-ID>. File: <primary-file>. Research: <arxiv-id if applicable>"
    }
  ]
}
```

### 3.4 Create `scripts/ralph/progress.txt`
```
# Ralph Progress Log
Started: <date>

## Codebase Patterns
- Test command: <exact test command with env vars>
- Environment: <conda env / venv / node version>
- Build: <build command if applicable>
- Architecture: <key classes and roles, one per line>
- Critical rules: <project-specific constraints>
- Beads: <exact bd command with flags>

## Pre-existing State
- <test pass count> tests passing
- <what already works>
- <known issues or gaps>
---

```

### 3.5 Create `scripts/ralph/CLAUDE.md`
This is the most critical file. It's what each iteration reads. Include ALL of the following:

```markdown
# Ralph Agent Instructions - <Project>

You are an autonomous coding agent. Each iteration: implement ONE story, test, commit.
You are self-healing (fix broken state) and self-upgrading (learn and improve globally).

## Environment
<exact commands to run tests, build, etc with full paths>

## Beads Task Tracking
<exact bd commands with flags for this project>

---

## Cross-Compaction Context Recovery (Step 0)

Before doing ANY work, recover context from previous iterations:

### 0.1 Read Local State
- Read `scripts/ralph/progress.txt` - patterns and learnings
- Read `scripts/ralph/prd.json` - which stories done vs pending

### 0.2 Query Memory Systems
- claude-mem: `mcp__plugin_claude-mem_mcp-search__search: query="<project>"`
- serena: `mcp__serena__read_memory: name="<project>-architecture"`
- cipher: `mcp__cipher__ask_cipher: "Current state of <project>?"`

### 0.3 Reconstruct Context
- What iteration we're on (count passes: true)
- What was last implemented
- Known failures or blockers
- Patterns that apply to current work

---

## Self-Healing (Step 0.5)

Before picking a new story, check for broken state:

### Detect
- `git status --short` (uncommitted changes?)
- Run quick smoke test (failing tests?)
- `git diff --check` (merge conflicts?)

### Fix
1. Uncommitted changes: complete partial work or reset
2. Failing tests: fix if from last story, note if pre-existing
3. Merge conflicts: resolve and commit
4. Import errors: fix dependencies

After healing, append to progress.txt:
- What was broken, root cause, prevention pattern

---

## 14-Step Iteration Workflow

1. Context Recovery - Query claude-mem, serena, cipher
2. Self-Heal - Fix broken state from prior iterations
3. Read progress.txt - Patterns and learnings
4. Read prd.json - Find highest-priority `passes: false`
5. Update beads - Mark task as `in_progress`
6. Research - If story.notes has arxiv refs
7. Implement - Code changes for ONE story
8. Test - Run acceptance criteria
9. Commit - Atomic commit for the story
10. Update prd.json - Set `passes: true`
11. Update progress.txt - Append learnings
12. Self-Upgrade - Store in memory systems (see below)
13. Close beads - Mark task closed
14. Signal - `<promise>COMPLETE</promise>` if all done

---

## Self-Upgrading (Step 12)

After each successful story:

### 12.1 Update progress.txt Codebase Patterns
Add new patterns to the TOP section (future iterations read this first)

### 12.2 Store in Serena Memory
`mcp__serena__write_memory: name="<project>-iteration-<N>"`
Content: story completed, files changed, key insight, reusable pattern

### 12.3 Store in Cipher (Cross-Project)
`mcp__cipher__ask_cipher: "Update <project> state: completed X, tests Y/Z, next is W"`

### 12.4 Update This CLAUDE.md
If new architecture or rules discovered, append them.
Future iterations read this updated version.

### 12.5 Global Pattern Extraction
If pattern applies to ALL projects:
`mcp__serena__write_memory: name="global-pattern-<name>"`

---

## Architecture
<key classes, data flow, invariants>

## Critical Rules
<project-specific constraints - no fallback, type safety, etc>

## Source Structure
<directory layout with annotations>

## Research References
<academic-research/ folder contents, reference implementations>

## Quality Gates
<what must pass before marking story complete>
```

---

## Phase 4: Memory Persistence

### 4.1 Cipher Snapshot
Store full project state:
```
mcp__cipher__ask_cipher: "Store project state: <project-name> at <path>.
Architecture: <summary>. Test status: <pass/fail>.
Ralph PRD: <story count> stories. Next: <title>.
Beads: <command>. Environment: <details>."
```

### 4.2 Serena Memory
```
mcp__serena__write_memory: name="<project>-architecture", content="<details>"
mcp__serena__write_memory: name="<project>-patterns", content="<patterns>"
```

### 4.3 Claude-mem
Hooks fire automatically. Verify:
```
mcp__plugin_claude-mem_mcp-search__search: query="<project-name>"
```

---

## Phase 5: Start the Orchestrator

### 5.1 Verify Setup
- [ ] `scripts/ralph/ralph.sh` exists and is executable
- [ ] `scripts/ralph/prd.json` has user stories with `passes: false`
- [ ] `scripts/ralph/progress.txt` has codebase patterns
- [ ] `scripts/ralph/CLAUDE.md` has full agent instructions with self-heal + self-upgrade
- [ ] Beads tasks created and accessible
- [ ] Tests can run (verify with one test command)
- [ ] Memory systems store project state (cipher, serena)

### 5.2 Launch
```bash
cd <project-root>
./scripts/ralph/ralph.sh --tool claude 50
```

The orchestrator will:
- Spawn fresh Claude instances per iteration
- Each instance recovers context from memory systems (survives compaction)
- Self-heals broken state from prior iterations
- Implements one story, tests, commits
- Self-upgrades by storing learnings in memory systems
- Gets smarter each iteration (progress.txt + serena + cipher accumulate knowledge)
- Loop ends when all stories `passes: true` or max iterations reached

---

## How It Gets Smarter Over Time

```
Iteration 1: Reads CLAUDE.md + empty progress.txt
             Learns: "tests need PYTHONNOUSERSITE=1"
             Stores in: progress.txt patterns, serena memory

Iteration 2: Reads CLAUDE.md + progress.txt with pattern
             Skips the mistake iteration 1 made
             Learns: "frame alignment needs transpose before interpolate"
             Stores in: progress.txt, serena, updates CLAUDE.md architecture

Iteration 3: Reads updated CLAUDE.md + richer progress.txt
             Has full architectural knowledge from iterations 1-2
             Learns: "CUDA kernels need sm_110 flag"
             Stores in: progress.txt, serena, cipher (global pattern)

...

Iteration N: Has accumulated N-1 iterations of knowledge
             Runs faster, makes fewer mistakes
             Global patterns available to OTHER projects via serena/cipher
```

---

## Stack Components

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **serena** | Semantic code analysis + memory | Phase 1 + each iteration (upgrade) |
| **beads** | Task tracking | Phase 2 + each iteration (status) |
| **claude-mem** | Auto-observations | Always (hooks fire automatically) |
| **cipher** | Cross-project state | Phase 4 + each iteration (upgrade) |
| **arxiv** | Paper search | Phase 1.5 (ML research) |
| **semantic-scholar** | Citation graphs | Phase 1.5 (related work) |
| **context7** | Library docs | Implementation (API reference) |
| **perplexity** | Web research | When docs insufficient |

---

## Platform Notes

### Network Mounts (CIFS/NFS/SMB)
SQLite WAL mode fails on network mounts. Use:
```bash
bd --no-daemon --db /home/kp/.beads-local/<project>/beads.db <command>
```

### Jetson/aarch64/CUDA
- `PYTHONNOUSERSITE=1` to avoid system package conflicts
- `TORCH_CUDA_ARCH_LIST=<sm>` for compute capability
- Build CUDA extensions from source
- `--index-url https://download.pytorch.org/whl/cu130` for PyTorch wheels

### Conda Environments
- Each project in its own env
- Always prefix: `PYTHONNOUSERSITE=1 PYTHONPATH=src`
- Never mix system and conda packages
