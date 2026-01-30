#!/bin/bash
# Ralph v3 - Story-Focused Autonomous Agent Orchestrator
# Usage: ./ralph.sh [--model MODEL] [--max-retries N] [--skip-checks]
#
# Fixes from v2:
#   - Per-story focused prompts (not piping 400-line instruction manual)
#   - Proper claude invocation (project CLAUDE.md auto-loads as system context)
#   - Post-iteration test verification (don't trust agent to self-report)
#   - Retry logic with error context injection on failure
#   - Runs until ALL stories pass (no arbitrary max iterations)

set -euo pipefail

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────

MODEL=""
MAX_RETRIES=3          # Retries per story before moving on
SKIP_CHECKS=false
COOLDOWN=5             # Seconds between iterations
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)       MODEL="$2"; shift 2 ;;
    --model=*)     MODEL="${1#*=}"; shift ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --skip-checks) SKIP_CHECKS=true; shift ;;
    --verbose)     VERBOSE=true; shift ;;
    -h|--help)
      echo "Usage: ./ralph.sh [--model MODEL] [--max-retries N] [--skip-checks] [--verbose]"
      echo ""
      echo "Options:"
      echo "  --model MODEL      Claude model (opus, sonnet, haiku)"
      echo "  --max-retries N    Max retries per story (default: 3)"
      echo "  --skip-checks      Skip environment pre-checks"
      echo "  --verbose          Show detailed output (tests, prompts, selection)"
      exit 0
      ;;
    *) shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
LOG_DIR="$SCRIPT_DIR/logs"

PYTHON="/home/kp/anaconda3/envs/autovoice-thor/bin/python"
PYTEST_CMD="PYTHONNOUSERSITE=1 PYTHONPATH=$PROJECT_DIR/src $PYTHON -m pytest"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_info()  { echo -e "${BLUE}[ralph]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[ralph]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[ralph]${NC} $1"; }
log_err()   { echo -e "${RED}[ralph]${NC} $1"; }
log_story() { echo -e "${CYAN}[ralph]${NC} ${BOLD}$1${NC}"; }
log_verbose() { [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}[verbose]${NC} $1" || true; }

# ─────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────

get_remaining_stories() {
  jq -r '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE" 2>/dev/null || echo "0"
}

get_next_story_id() {
  jq -r '[.userStories[] | select(.passes == false)] | sort_by(.priority) | .[0].id // empty' "$PRD_FILE" 2>/dev/null
}

get_story_field() {
  local story_id="$1" field="$2"
  jq -r --arg id "$story_id" '.userStories[] | select(.id == $id) | .'"$field" "$PRD_FILE" 2>/dev/null
}

get_story_criteria() {
  local story_id="$1"
  jq -r --arg id "$story_id" '.userStories[] | select(.id == $id) | .acceptanceCriteria[]' "$PRD_FILE" 2>/dev/null
}

mark_story_complete() {
  local story_id="$1"
  local tmp=$(mktemp)
  jq --arg id "$story_id" '(.userStories[] | select(.id == $id)).passes = true' "$PRD_FILE" > "$tmp" && mv "$tmp" "$PRD_FILE"
}

# Map story IDs to their relevant research files and test patterns
get_story_research_files() {
  local story_id="$1"
  case "$story_id" in
    AV-006) echo "deepwiki-bigvgan.md gitmcp-reference-code.md vocoder-separation-research.md linkup-latest-research.md" ;;
    AV-008) echo "deepwiki-demucs.md hf-demucs-models.md vocoder-separation-research.md" ;;
    AV-010) echo "consistency-research.md hf-consistency-papers.md gitmcp-reference-code.md perplexity-sota-svc.md" ;;
    AV-011) echo "gitmcp-reference-code.md deepwiki-bigvgan.md" ;;
    AV-012) echo "gitmcp-reference-code.md deepwiki-bigvgan.md consistency-research.md" ;;
    AV-013) echo "" ;;
    AV-014) echo "" ;;
    AV-015) echo "perplexity-sota-svc.md" ;;
    AV-016) echo "gitmcp-reference-code.md deepwiki-bigvgan.md" ;;
    AV-017) echo "perplexity-sota-svc.md" ;;
    AV-018) echo "" ;;
    AV-019) echo "perplexity-sota-svc.md" ;;
    *)      echo "" ;;
  esac
}

get_story_test_pattern() {
  local story_id="$1"
  case "$story_id" in
    AV-006) echo "tests/test_models.py -k bigvgan" ;;
    AV-008) echo "tests/test_vocal_separator.py" ;;
    AV-010) echo "tests/test_consistency.py" ;;
    AV-011) echo "tests/test_model_manager.py" ;;
    AV-012) echo "tests/test_e2e_pipeline.py" ;;
    AV-013) echo "tests/test_model_manager.py -k validation" ;;
    AV-014) echo "tests/test_vocal_separator.py" ;;
    AV-015) echo "tests/test_training_convergence.py" ;;
    AV-016) echo "tests/test_onnx_export.py" ;;
    AV-017) echo "tests/test_tensorrt.py" ;;
    AV-018) echo "tests/test_web_api.py" ;;
    AV-019) echo "tests/test_gpu_memory.py" ;;
    *)      echo "tests/" ;;
  esac
}

# ─────────────────────────────────────────────────
# Build Per-Story Prompt
# ─────────────────────────────────────────────────

build_story_prompt() {
  local story_id="$1"
  local retry_context="${2:-}"

  local title=$(get_story_field "$story_id" "title")
  local description=$(get_story_field "$story_id" "description")
  local criteria=$(get_story_criteria "$story_id")
  local research_files=$(get_story_research_files "$story_id")
  local test_pattern=$(get_story_test_pattern "$story_id")

  # Build research context (truncated to key sections)
  local research_context=""
  for f in $research_files; do
    local fpath="$PROJECT_DIR/academic-research/$f"
    if [[ -f "$fpath" ]]; then
      # Take first 100 lines of each research file (enough for key architecture info)
      research_context+="
--- Research: $f ---
$(head -100 "$fpath")
...
"
    fi
  done

  # Build the focused prompt
  cat <<PROMPT
You are an autonomous coding agent. Your ONE task is to implement story $story_id.
Do NOT plan or discuss. IMPLEMENT the code, write tests, run them, commit, and update prd.json.

# Story: $story_id - $title

## Description
$description

## Acceptance Criteria
$criteria

## Environment
- Python: $PYTHON
- Run tests: PYTHONNOUSERSITE=1 PYTHONPATH=$PROJECT_DIR/src $PYTHON -m pytest $PROJECT_DIR/$test_pattern -v --tb=short
- Run full suite: PYTHONNOUSERSITE=1 PYTHONPATH=$PROJECT_DIR/src $PYTHON -m pytest $PROJECT_DIR/tests/ --tb=no -q
- Working dir: $PROJECT_DIR

## Architecture (Current Files)
- src/auto_voice/models/vocoder.py - HiFiGAN + BigVGAN vocoders
- src/auto_voice/models/encoder.py - ContentEncoder (HuBERT + ContentVec + Conformer)
- src/auto_voice/models/conformer.py - ConformerEncoder (6-layer)
- src/auto_voice/models/consistency.py - DiffusionDecoder + ConsistencyStudent
- src/auto_voice/audio/separation.py - VocalSeparator (Demucs HTDemucs)
- src/auto_voice/audio/augmentation.py - AugmentationPipeline
- src/auto_voice/inference/model_manager.py - ModelManager orchestrator
- src/auto_voice/inference/singing_conversion_pipeline.py - Main pipeline
- src/auto_voice/inference/realtime_voice_conversion_pipeline.py - Realtime pipeline
- src/auto_voice/export/ - ONNX/TensorRT export (new for AV-016/017)
- src/auto_voice/web/api.py - Flask REST endpoints
- src/auto_voice/training/trainer.py - Training loop + VoiceDataset
- tests/ - All test files (403 tests currently passing)

## Research Context
$research_context

## Critical Rules
1. No fallback behavior - raise RuntimeError, never pass silently
2. Use real model architectures (no placeholder/random outputs)
3. Tests must actually verify behavior (shapes, non-NaN, correct types)
4. Frame alignment: F.interpolate(transpose(1,2), size=target)
5. PYTHONNOUSERSITE=1 always set for python commands

## Steps (execute ALL of these):
1. Read the relevant source files to understand current architecture
2. Implement the required classes/functions
3. Write comprehensive tests (test file: $PROJECT_DIR/$test_pattern)
4. Run the specific tests and fix any failures
5. Run the full test suite to verify no regressions
6. Git commit: git add <files> && git commit -m "feat(autovoice): $title"
7. Update PRD: In $PRD_FILE, set passes=true for story $story_id
8. After ALL steps complete successfully, output exactly: STORY_COMPLETE
PROMPT

  # Add retry context if this is a retry
  if [[ -n "$retry_context" ]]; then
    cat <<RETRY

## PREVIOUS ATTEMPT FAILED
The previous attempt to implement this story failed. Here's what went wrong:
$retry_context

Fix the issues above and complete the implementation.
RETRY
  fi
}

# ─────────────────────────────────────────────────
# Environment Pre-Checks
# ─────────────────────────────────────────────────

if [[ "$SKIP_CHECKS" == "false" ]]; then
  log_info "Running environment checks..."

  # Python environment
  if [[ ! -f "$PYTHON" ]]; then
    log_err "Python not found: $PYTHON"
    exit 1
  fi
  log_ok "Python: $PYTHON"

  # Core dependencies
  for pkg in torch torchaudio transformers; do
    if ! PYTHONNOUSERSITE=1 "$PYTHON" -c "import $pkg" 2>/dev/null; then
      log_err "Missing: $pkg"
      exit 1
    fi
  done
  log_ok "Core deps: torch, torchaudio, transformers"

  # Claude CLI
  if ! command -v claude &>/dev/null; then
    log_err "'claude' not found in PATH"
    exit 1
  fi
  log_ok "Claude: $(claude --version 2>/dev/null || echo 'unknown')"

  # PRD file
  if [[ ! -f "$PRD_FILE" ]]; then
    log_err "PRD not found: $PRD_FILE"
    exit 1
  fi

  REMAINING=$(get_remaining_stories)
  if [[ "$REMAINING" == "0" ]]; then
    log_ok "All stories already pass! Nothing to do."
    exit 0
  fi
  log_ok "PRD: $REMAINING stories remaining"

  # Test baseline
  log_info "Verifying test baseline (this takes ~80s)..."
  if [[ "$VERBOSE" == "true" ]]; then
    BASELINE_OUTPUT=$(eval "$PYTEST_CMD $PROJECT_DIR/tests/ --tb=short -q" 2>&1 | tee /dev/stderr | tail -1)
  else
    BASELINE_OUTPUT=$(eval "$PYTEST_CMD $PROJECT_DIR/tests/ --tb=no -q" 2>&1)
  fi
  BASELINE=$(echo "$BASELINE_OUTPUT" | tail -1)
  if echo "$BASELINE" | grep -qE "failed|error"; then
    log_err "Tests broken: $BASELINE"
    exit 1
  fi
  log_ok "Tests: $BASELINE"
  echo ""
fi

# Create log directory
mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────
# Main Orchestration Loop
# ─────────────────────────────────────────────────

MODEL_FLAG=""
if [[ -n "$MODEL" ]]; then
  MODEL_FLAG="--model $MODEL"
fi

TOTAL_START=$(date +%s)
STORIES_COMPLETED=0
TOTAL_ITERATIONS=0

log_info "Starting Ralph v3 - Autonomous Story Executor"
log_info "Model: ${MODEL:-default} | Max retries/story: $MAX_RETRIES | Verbose: $VERBOSE"
echo ""

while true; do
  # Find next story
  STORY_ID=$(get_next_story_id)
  if [[ -z "$STORY_ID" ]]; then
    break  # All done
  fi

  STORY_TITLE=$(get_story_field "$STORY_ID" "title")
  TEST_PATTERN=$(get_story_test_pattern "$STORY_ID")

  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  log_story "  Story: $STORY_ID - $STORY_TITLE"
  echo "╚══════════════════════════════════════════════════════════════╝"
  log_verbose "Priority: $(get_story_field "$STORY_ID" "priority")"
  log_verbose "Test pattern: $TEST_PATTERN"
  log_verbose "Research files: $(get_story_research_files "$STORY_ID")"

  RETRY_CONTEXT=""
  STORY_DONE=false

  for attempt in $(seq 1 $MAX_RETRIES); do
    TOTAL_ITERATIONS=$((TOTAL_ITERATIONS + 1))
    ITER_START=$(date +%s)

    log_info "Attempt $attempt/$MAX_RETRIES for $STORY_ID (iteration #$TOTAL_ITERATIONS)"

    # Build the focused prompt
    PROMPT=$(build_story_prompt "$STORY_ID" "$RETRY_CONTEXT")

    # Write prompt to log for debugging
    ITER_LOG="$LOG_DIR/${STORY_ID}_attempt${attempt}_$(date +%H%M%S).log"
    echo "$PROMPT" > "$ITER_LOG.prompt"

    if [[ "$VERBOSE" == "true" ]]; then
      log_verbose "Prompt (first 20 lines):"
      echo "$PROMPT" | head -20 | sed 's/^/  /'
      echo "  ..."
    fi

    # Execute claude with proper invocation:
    # - Working directory = project root (auto-loads CLAUDE.md)
    # - -p = print mode with tool use
    # - --dangerously-skip-permissions = autonomous execution
    # - Prompt written to temp file, piped via stdin (handles special chars/long prompts)
    log_info "Executing claude (autonomous mode)..."
    PROMPT_FILE=$(mktemp)
    echo "$PROMPT" > "$PROMPT_FILE"
    set +e
    OUTPUT=$(cd "$PROJECT_DIR" && claude \
      --dangerously-skip-permissions \
      --print \
      $MODEL_FLAG \
      < "$PROMPT_FILE" \
      2>&1 | tee "$ITER_LOG")
    CLAUDE_EXIT=$?
    set -e
    rm -f "$PROMPT_FILE"

    ITER_END=$(date +%s)
    ITER_DURATION=$((ITER_END - ITER_START))
    log_info "Claude finished in ${ITER_DURATION}s (exit: $CLAUDE_EXIT)"

    # Log iteration to progress
    echo "" >> "$PROGRESS_FILE"
    echo "## $STORY_ID attempt $attempt ($(date '+%Y-%m-%d %H:%M')) - ${ITER_DURATION}s" >> "$PROGRESS_FILE"

    # Check if story was completed by verifying prd.json
    if [[ "$(get_story_field "$STORY_ID" "passes")" == "true" ]]; then
      log_ok "Story $STORY_ID marked complete in PRD!"
      STORY_DONE=true
      break
    fi

    # Also check if the agent output STORY_COMPLETE signal
    if echo "$OUTPUT" | grep -q "STORY_COMPLETE"; then
      # Agent says it's done - verify with tests
      log_info "Agent signals completion. Verifying tests..."
      set +e
      if [[ "$VERBOSE" == "true" ]]; then
        TEST_OUTPUT=$(eval "$PYTEST_CMD $PROJECT_DIR/$TEST_PATTERN -v --tb=short" 2>&1 | tee /dev/stderr)
      else
        TEST_OUTPUT=$(eval "$PYTEST_CMD $PROJECT_DIR/$TEST_PATTERN -v --tb=short" 2>&1)
      fi
      TEST_EXIT=$?
      set -e

      if [[ $TEST_EXIT -eq 0 ]]; then
        log_ok "Tests pass! Marking $STORY_ID complete."
        mark_story_complete "$STORY_ID"
        STORY_DONE=true
        break
      else
        log_warn "Agent claimed completion but tests FAILED:"
        echo "$TEST_OUTPUT" | tail -20
        RETRY_CONTEXT="Tests failed after your previous implementation attempt:
$(echo "$TEST_OUTPUT" | tail -30)"
      fi
    else
      # Agent didn't signal completion - check if tests pass anyway
      # Extract just the file/dir path (before any pytest args like -k)
      TEST_FILE=$(echo "$TEST_PATTERN" | awk '{print $1}')
      if [[ -f "$PROJECT_DIR/$TEST_FILE" ]] || [[ -d "$PROJECT_DIR/$TEST_FILE" ]]; then
        set +e
        if [[ "$VERBOSE" == "true" ]]; then
          TEST_OUTPUT=$(eval "$PYTEST_CMD $PROJECT_DIR/$TEST_PATTERN --tb=short -q" 2>&1 | tee /dev/stderr)
        else
          TEST_OUTPUT=$(eval "$PYTEST_CMD $PROJECT_DIR/$TEST_PATTERN --tb=short -q" 2>&1)
        fi
        TEST_EXIT=$?
        set -e

        if [[ $TEST_EXIT -eq 0 ]] && echo "$TEST_OUTPUT" | grep -qE "[0-9]+ passed"; then
          log_ok "Tests pass (agent forgot to signal). Marking $STORY_ID complete."
          mark_story_complete "$STORY_ID"
          STORY_DONE=true
          break
        fi
      fi

      # Collect error context for retry
      RETRY_CONTEXT="The previous attempt did not complete the story. Duration: ${ITER_DURATION}s.
Exit code: $CLAUDE_EXIT
Last 50 lines of output:
$(echo "$OUTPUT" | tail -50)"
    fi

    if [[ $attempt -lt $MAX_RETRIES ]]; then
      log_warn "Retrying in ${COOLDOWN}s..."
      sleep $COOLDOWN
    fi
  done

  if [[ "$STORY_DONE" == "true" ]]; then
    STORIES_COMPLETED=$((STORIES_COMPLETED + 1))
    log_ok "Completed: $STORY_ID ($STORIES_COMPLETED stories done this session)"

    # Update progress file
    echo "- Completed: $STORY_ID ($STORY_TITLE)" >> "$PROGRESS_FILE"
  else
    log_err "FAILED: $STORY_ID after $MAX_RETRIES attempts. Skipping."
    echo "- FAILED: $STORY_ID after $MAX_RETRIES attempts" >> "$PROGRESS_FILE"
  fi

  # Check if all done
  REMAINING=$(get_remaining_stories)
  if [[ "$REMAINING" == "0" ]]; then
    break
  fi
  log_info "$REMAINING stories remaining..."
  sleep $COOLDOWN
done

# ─────────────────────────────────────────────────
# Final Report
# ─────────────────────────────────────────────────

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
REMAINING=$(get_remaining_stories)

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [[ "$REMAINING" == "0" ]]; then
  log_ok "ALL STORIES COMPLETE!"
  log_ok "Stories completed this session: $STORIES_COMPLETED"
  log_ok "Total iterations: $TOTAL_ITERATIONS"
  log_ok "Total time: $((TOTAL_DURATION/60))m $((TOTAL_DURATION%60))s"
  echo "---" >> "$PROGRESS_FILE"
  echo "## ALL COMPLETE - $(date '+%Y-%m-%d %H:%M') (${TOTAL_DURATION}s, $TOTAL_ITERATIONS iterations)" >> "$PROGRESS_FILE"
  exit 0
else
  log_err "NOT ALL STORIES COMPLETED"
  log_err "Remaining: $REMAINING stories"
  log_err "Completed this session: $STORIES_COMPLETED"
  log_err "Total iterations: $TOTAL_ITERATIONS"
  log_err "Total time: $((TOTAL_DURATION/60))m $((TOTAL_DURATION%60))s"
  log_info "Logs: $LOG_DIR/"
  log_info "Re-run: ./ralph.sh --skip-checks"
  exit 1
fi
