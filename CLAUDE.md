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
python main.py --host 0.0.0.0 --port 5000

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

**Current Status (2026-02-02):**
- Overall Coverage: **63%** (15,063 lines, 9,467 covered)
- Target: **80%** overall, **85%** for inference modules
- Test Suite: 1,984 tests (1,791 passing, 147 failing, 39 skipped, 47 errors)
- Runtime: ~27 minutes

**Module Coverage:**
- Database: **87%** ✅ (exceeds 70% target)
- Inference Core: **68%** (adapter_bridge 97%, pipeline_factory 94%, meanvc 91%)
- Audio Processing: **55%** (diarization 50%, separation 40%, youtube 38%)
- Web API: **60%** (training API 70%, karaoke 30%)
- Storage: **78%** ✅ (exceeds 70% target)

**Coverage Report:** `reports/coverage_summary_20260202.md`
**HTML Report:** `htmlcov/index.html`

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
# GitNexus MCP

This project is indexed by GitNexus as **autovoice** (20214 symbols, 52143 relationships, 300 execution flows).

## Always Start Here

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `npx gitnexus analyze` in the terminal first.

## Skills

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
