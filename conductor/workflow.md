# Workflow

## TDD Policy

**Strict** - Tests are required before implementation.

- Write failing test first (red)
- Implement minimum code to pass (green)
- Refactor while keeping tests green (refactor)
- No code merges without passing test coverage
- Tests must verify real behavior (shapes, non-NaN, correct types, actual outputs)

## Commit Strategy

**Descriptive messages** - No strict format required, but messages should clearly describe what changed and why. The existing conventional commit format (feat:, fix:) is acceptable but not enforced.

## Code Review

**Required for all changes** - Every PR must be reviewed before merge. For AI-assisted development, this means running the full test suite and verifying integration before committing.

## Verification Checkpoints

**After each task completion** - Verify every individual task before moving on. Deep integration verification is required to ensure the code is written and integrated correctly for the project. This includes:

1. Unit tests pass for the new code
2. Integration tests pass (existing tests don't break)
3. The feature actually works in context (not just in isolation)
4. Code follows project conventions (CLAUDE.md rules)
5. No silent fallback behavior introduced

## Task Lifecycle

1. **Define** - Clearly specify what the task accomplishes
2. **Research** - Use academic MCP servers to find SOTA approaches:
   - `paper-search` — Search across arXiv, PubMed, Semantic Scholar, Google Scholar
   - `arxiv-advanced` — Deep-dive into specific arXiv papers, download and read full text
   - `semantic-scholar-citations` — Trace citation graphs to find foundational and recent work
   - Focus: find current best architectures, loss functions, training recipes, and benchmarks
   - Output: document chosen approach with paper references before implementing
3. **Test** - Write failing tests that define success criteria
4. **Implement** - Write minimum code to pass tests, using SOTA techniques
5. **Verify** - Run full test suite, check integration
6. **Review** - Deep verification of correctness and integration
7. **Commit** - Only after all checks pass

## Critical Rules (from CLAUDE.md)

- No fallback behavior: Always raise RuntimeError, never pass through silently
- Speaker embedding: mel-statistics (mean+std of 128 mels = 256-dim, L2-normalized)
- Frame alignment: F.interpolate(transpose(1,2), size=target) for content/pitch
- PYTHONNOUSERSITE=1 always set for python commands
- Atomic commits: one feature per commit, always run full test suite first
