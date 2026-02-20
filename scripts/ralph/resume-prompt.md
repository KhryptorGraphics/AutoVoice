# Ralph v3 - Story-Focused Autonomous Agent Orchestrator

## How It Works

Ralph v3 generates a **focused per-story prompt** for each remaining story and executes
`claude -p "prompt" --dangerously-skip-permissions` from the project root.

The project's `CLAUDE.md` is auto-loaded as system context (build commands, architecture).
The focused prompt provides: acceptance criteria, research context, test commands, and
explicit step-by-step execution instructions.

## Current State

| Metric | Value |
|--------|-------|
| Stories Complete | 7/10 (AV-001–005, AV-007, AV-009) |
| Stories Remaining | 3 (AV-006, AV-008, AV-010) |
| Tests Passing | 246 (100%) |
| Platform | Jetson Thor, aarch64, CUDA 13.0 |

## Remaining Stories

| ID | Title | Test Pattern | Complexity |
|----|-------|-------------|------------|
| AV-006 | BigVGAN vocoder | `tests/test_models.py -k bigvgan` | Medium |
| AV-008 | Demucs vocal separation | `tests/test_vocal_separator.py` | Medium |
| AV-010 | Consistency distillation | `tests/test_consistency.py` | High |

## Usage

```bash
cd /home/kp/repo2/autovoice

# Run with default model
./scripts/ralph/ralph.sh

# Run with specific model
./scripts/ralph/ralph.sh --model opus

# Skip environment checks (faster restart)
./scripts/ralph/ralph.sh --skip-checks

# More retries per story
./scripts/ralph/ralph.sh --max-retries 5
```

## Debugging

- Logs: `scripts/ralph/logs/AV-XXX_attemptN_HHMMSS.log`
- Prompt: `scripts/ralph/logs/AV-XXX_attemptN_HHMMSS.log.prompt`
- Progress: `scripts/ralph/progress.txt`
- PRD: `scripts/ralph/prd.json`

## v3 vs v2 Differences

| Feature | v2 (broken) | v3 (fixed) |
|---------|-------------|------------|
| Prompt delivery | Pipe 400-line CLAUDE.md via stdin | Focused per-story prompt via `-p` |
| Agent instructions | In CLAUDE.md (was deleted) | Project CLAUDE.md auto-loaded |
| Story selection | Agent decides | Orchestrator decides from prd.json |
| Verification | Trust agent's signal | Run tests independently |
| On failure | No retry context | Inject error output into next attempt |
| Iteration limit | 10 (arbitrary) | Until all pass or max-retries exhausted |
