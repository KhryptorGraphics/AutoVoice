# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoVoice is a GPU-accelerated singing voice conversion and TTS system built with PyTorch, CUDA kernels, and Flask. It converts songs to a target voice while preserving pitch and timing, using the So-VITS-SVC architecture.

## Build & Development Commands

### Platform (Jetson Thor)
```bash
# Active conda environment (ALWAYS use this)
PYTHON=/home/kp/anaconda3/envs/autovoice-thor/bin/python

# Run any python command (PYTHONNOUSERSITE prevents system package contamination)
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON <script.py>

# Run tests
PYTHONNOUSERSITE=1 PYTHONPATH=src $PYTHON -m pytest tests/ -x --tb=short -q
```

### Environment Setup
```bash
# Create conda environment
conda create -n autovoice python=3.12 -y && conda activate autovoice

# Install PyTorch with CUDA (REQUIRED FIRST)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
pip install -e .
```

### Running the Application
```bash
# Start server (Flask + SocketIO)
python main.py --host 127.0.0.1 --port 10600

# With Docker
docker-compose up
```

### Testing
```bash
# Run complete test suite
./run_tests.sh all

# Quick smoke tests (< 30s)
./run_tests.sh smoke

# Fast tests (excludes slow)
./run_tests.sh fast

# With coverage report
./run_tests.sh coverage

# Run specific test file
pytest tests/test_voice_conversion.py -v

# Run by marker
pytest tests/ -m cuda -v
pytest tests/ -m "not slow" -v
```

### Code Quality
```bash
black src/ tests/
isort src/ tests/
mypy src/auto_voice
```

### Key Scripts
- `scripts/setup_pytorch_env.sh` - Automated PyTorch/CUDA setup
- `scripts/build_and_test.sh` - Build and verify
- `scripts/verify_bindings.py` - Verify CUDA extension build
- `scripts/download_pretrained_models.py` - Download model weights

## Architecture

### Source Structure (`src/`)
```
auto_voice/
  inference/           # Voice conversion pipeline (So-VITS-SVC)
    singing_conversion_pipeline.py  # Main conversion entry point
    voice_cloner.py                 # Speaker encoder/profile creation
    realtime_voice_conversion_pipeline.py
  models/              # Neural network architectures
    encoder.py         # Content/pitch encoders
    vocoder.py         # HiFiGAN vocoder
  audio/               # Audio processing utilities
  evaluation/          # Quality metrics (pitch RMSE, speaker similarity)
  web/                 # Flask API + SocketIO handlers
    app.py             # create_app() factory
    api.py             # REST endpoints
  training/            # Training pipeline
  gpu/                 # GPU memory management
  monitoring/          # Prometheus metrics
cuda_kernels/          # Custom CUDA kernels (.cu files)
```

### Key Classes
- `SingingConversionPipeline` - Main voice conversion class (`inference/singing_conversion_pipeline.py`)
- `VoiceCloner` - Creates voice profiles from audio samples (`inference/voice_cloner.py`)
- `create_app()` - Flask app factory (`web/app.py`)

### Frontend (`frontend/`)
React + TypeScript + Vite + Tailwind CSS
```bash
cd frontend && npm install && npm run dev
```

### API Endpoints
- `GET /health` - Health check with component status
- `POST /api/v1/voice/clone` - Create voice profile
- `POST /api/v1/convert/song` - Convert song to target voice
- `GET /api/v1/convert/status/{id}` - Check conversion status
- WebSocket events for real-time progress

## Test Markers
Tests use pytest markers defined in `pytest.ini`:
- `smoke` - Fast validation
- `cuda` - Requires GPU
- `slow` / `very_slow` - Long-running tests
- `integration` - Component interaction tests
- `performance` - Benchmarks
- `tensorrt` - TensorRT-specific tests
- `browser` - Browser automation tests (VNC display required)

## Test Coverage

**Canonical readiness signal:** the completion matrix
(`reports/completion/latest/completion_matrix.json`) and
[docs/current-truth.md](docs/current-truth.md) — not the snapshot figures below.

**Last full pytest run (2026-07-02, sha `0a4dc379`):** 4,613 tests, **4,540
passed, 2 failed, 57 skipped, 1 xfailed, 13 errors** (2h18m,
`reports/full_pytest_cov_20260702.log`).
- All 13 errors are the `test_docker_deployment.py` suite (no local Docker).
- Both failures are **fixed at HEAD**: the `test_platform_scripts` pair was the
  `requirements.lock` hash going stale after adding `requests` (lock updated),
  and the flaky `test_e2e_pipeline` speaker-sensitivity test is now
  seed-pinned. Effective supported-lane pass rate at HEAD: 100%.

**Coverage (measured 2026-07-02 with the run above):**
- Overall **88%** (target 80% — met), inference **92%** (target 85% — met)
- Models/Training 94%, Audio 93%, Evaluation/Monitoring/Profiles 91%,
  Web 84%, Storage 97%, DB 100%; below-target outliers: swarm 27%, utils 66%
  (operator tooling, not on the product path)

> The 2026-04-29 snapshot (4,490/12/53/13 at 63% stale coverage) is superseded
> by the run above. Re-measure at the candidate commit before release claims.

### Test Patterns and Best Practices

#### Fixtures (tests/conftest.py)
```python
# Audio fixtures - use generated samples, not real files
@pytest.fixture
def sample_audio():
    """Generate 5-second test audio at 44.1kHz"""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz tone
    return audio, sr

# Database fixtures - use in-memory SQLite
@pytest.fixture
def test_db():
    """In-memory database for fast, isolated tests"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

# Mock expensive ML models
@pytest.fixture
def mock_model():
    """Mock model for fast tests without GPU"""
    with patch('auto_voice.models.load_model') as mock:
        mock.return_value = MagicMock()
        yield mock
```

#### Test Organization
```python
# Group related tests in classes
class TestPipelineInitialization:
    """Tests for pipeline creation and configuration"""

    def test_default_config(self, test_config):
        """Test pipeline with default settings"""
        pipeline = Pipeline(test_config)
        assert pipeline.sample_rate == 44100

    def test_gpu_device_selection(self, test_config):
        """Test automatic GPU selection"""
        pipeline = Pipeline(test_config)
        assert pipeline.device.type == 'cuda'

# Use parametrize for multiple inputs
@pytest.mark.parametrize("input_len,expected", [
    (100, 100),
    (1000, 1000),
    (10000, 10000)
])
def test_audio_processing(input_len, expected):
    """Test with various audio lengths"""
    audio = np.random.randn(input_len)
    result = process(audio)
    assert len(result) == expected
```

#### Mocking External Dependencies
```python
# Mock network calls
@patch('requests.get')
def test_youtube_download(mock_get):
    """Test YouTube download without network"""
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b'fake audio data'

    result = download_youtube('fake_url')
    assert result is not None

# Mock expensive ML operations
@patch('auto_voice.inference.model_manager.load_model')
def test_conversion_without_model(mock_load):
    """Test conversion logic without loading real model"""
    mock_load.return_value = MagicMock()

    pipeline = ConversionPipeline()
    result = pipeline.convert(audio, profile_id='test')
    assert result.shape[0] > 0
```

#### Testing Strategies by Module

**Inference Tests** (Target: 85% coverage):
- Test all pipeline types (quality, realtime, streaming, TRT)
- Mock model loading for fast tests, use `@pytest.mark.cuda` for real GPU tests
- Test error paths (missing adapter, GPU OOM, invalid audio)
- Verify output shapes, non-NaN values, and correct device placement

**Audio Tests** (Target: 70% coverage):
- Use generated synthetic audio (sine waves, white noise)
- Mock external tools (demucs, pyannote) if not installed
- Test edge cases (empty audio, single sample, very long audio)
- Verify audio properties (sample rate, channels, duration)

**Database Tests** (Target: 70% coverage):
- Always use in-memory SQLite for speed and isolation
- Test CRUD operations, constraints, cascades
- Test transaction rollback on errors
- Verify no file system side effects

**Web API Tests** (Target: 80% coverage):
- Use Flask test client (no server needed)
- Test all endpoints (success, error responses, validation)
- Test WebSocket events with SocketIO test client
- Mock ML operations for fast execution

**Integration Tests**:
- Test complete workflows (upload → train → convert)
- Use mocks for expensive operations
- Mark as `@pytest.mark.slow` or `@pytest.mark.integration`
- Verify end-to-end data flow

#### Common Pitfalls to Avoid

❌ **DON'T:**
- Use real audio files from disk (slow, depends on file system)
- Make network calls in tests (slow, unreliable)
- Load real ML models in unit tests (slow, requires GPU)
- Use time.sleep() for async tests (flaky, slow)
- Test implementation details (brittle)

✅ **DO:**
- Generate synthetic test data in fixtures
- Mock external dependencies (network, file system, ML models)
- Use in-memory databases
- Use async test utilities (pytest-asyncio)
- Test behavior and contracts

#### Coverage-Driven Development

When adding new features:
1. Write failing tests first (TDD)
2. Implement minimum code to pass tests
3. Refactor with tests as safety net
4. Verify coverage with `pytest --cov`
5. Aim for 80%+ coverage on new code

When fixing bugs:
1. Write test that reproduces bug
2. Verify test fails
3. Fix bug
4. Verify test passes
5. Check coverage didn't decrease

#### Running Tests Efficiently

```bash
# Run tests in parallel (faster)
pytest -n auto

# Run only changed tests
pytest --lf  # last failed
pytest --ff  # failed first

# Run with coverage
pytest --cov=src/auto_voice --cov-report=html --cov-report=term

# Skip slow tests for quick feedback
pytest -m "not slow"

# Run specific module tests
pytest tests/test_inference*.py -v

# Debug failing test
pytest tests/test_foo.py::test_bar -vv --pdb
```

#### Test Performance Targets

- Unit tests: <100ms per test
- Integration tests: <1s per test
- E2E tests: <5s per test
- Full suite: <30 minutes
- Smoke suite: <30 seconds

## Configuration
- `config/gpu_config.yaml` - GPU and server settings
- `config/logging_config.yaml` - Logging configuration
- Environment variables: `LOG_LEVEL`, `LOG_FORMAT`, `CUDA_VISIBLE_DEVICES`

## Pre-trained Models
Located in `models/pretrained/`:
- `sovits5.0_main_1500.pth` - Main So-VITS model
- `hifigan_ljspeech.ckpt` - HiFiGAN vocoder
- `hubert-soft-0d54a1f4.pt` - HuBERT feature extractor

## Served models and tuned serving settings

**Read [docs/served-models.md](docs/served-models.md) before changing which
checkpoint a profile serves, or before "correcting" anything in
`data/fork_models/*.json`.** `data/` is gitignored, so the live checkpoint
choice, `f0_method`, and the by-ear multi-speaker settings do not appear in any
diff — a registry entry that differs from `HEAD` is usually a deliberate,
by-ear decision rather than drift, and that file records which. It also lists
the rollback backups and the thresholds known to be fragile.

## File Organization Rules
- Source code: `/src`
- Tests: `/tests`
- Documentation: `/docs`
- Scripts: `/scripts`
- Configuration: `/config`
- Examples: `/examples`

## Critical Coding Rules
- No fallback behavior: Always raise RuntimeError, never pass through silently
- Speaker embedding: mel-statistics (mean+std of 128 mels = 256-dim, L2-normalized)
- Frame alignment: F.interpolate(transpose(1,2), size=target) for content/pitch
- PYTHONNOUSERSITE=1 always set for python commands
- Tests must verify real behavior (shapes, non-NaN, correct types)
- Atomic commits: one feature per commit, always run full test suite first

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **AutoVoice** (23039 symbols, 37457 relationships, 627 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).
> On query/context/impact/cypher object results, read staleness.status and branch/lastCommit. Re-analyze only for behind or diverged — current is clone HEAD, not main.

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/AutoVoice/context` | Codebase overview, check index freshness |
| `gitnexus://repo/AutoVoice/clusters` | All functional areas |
| `gitnexus://repo/AutoVoice/processes` | All execution flows |
| `gitnexus://repo/AutoVoice/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |
| Listen to a conversion, judge a render by measurement | `.claude/skills/earshot-conversion-ears/SKILL.md` |

<!-- gitnexus:end -->
