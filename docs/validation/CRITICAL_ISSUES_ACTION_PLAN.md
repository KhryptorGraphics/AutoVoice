# Critical Issues & Action Plan

**Generated:** November 9, 2025
**Status:** PRODUCTION BLOCKED
**Priority:** P0 - IMMEDIATE ACTION REQUIRED

---

## 🚨 PRODUCTION BLOCKERS (P0)

### 1. GLIBCXX Dependency Conflict

**Issue:** `GLIBCXX_3.4.30` not found in Anaconda's libstdc++.so.6

**Impact:**
- 10 test modules cannot import
- Scipy library unavailable
- Nearly all tests fail to run
- 0% test coverage achieved

**Solutions (Choose One):**

**Option A: Update Conda Package (RECOMMENDED)**
```bash
conda install -c conda-forge libstdcxx-ng
conda update --all
```

**Option B: Use System Library**
```bash
# Temporarily override
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Permanently (add to .bashrc or environment.yml)
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Option C: Fresh Environment with Compatible Versions**
```bash
# Create new environment with Python 3.11 (better compatibility)
conda create -n autovoice-prod python=3.11
conda activate autovoice-prod
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

**Verification:**
```bash
python -c "import scipy; print('Scipy OK')"
python -c "from src.auto_voice.utils.metrics import compute_metrics; print('Metrics OK')"
```

**Estimated Time:** 1-2 hours
**Owner:** DevOps / Infrastructure

---

### 2. Syntax Error in websocket_handler.py

**Issue:** Missing indented block after 'else' statement at line 737

**File:** `/home/kp/autovoice/src/auto_voice/web/websocket_handler.py:737`

**Impact:**
- WebSocket functionality broken
- Code will not execute
- Production deployment impossible

**Fix Required:**
```python
# Current (BROKEN):
else:
# Line 737 - needs indentation

# Fix Option 1 (add pass):
else:
    pass

# Fix Option 2 (add actual logic):
else:
    logger.debug("No action required")
```

**Verification:**
```bash
python -m py_compile src/auto_voice/web/websocket_handler.py
pylint src/auto_voice/web/websocket_handler.py --errors-only
```

**Estimated Time:** 10 minutes
**Owner:** Backend Developer

---

### 3. Zero Test Coverage

**Issue:** 0.00% coverage (target: ≥80%)

**Root Causes:**
- GLIBCXX dependency prevents imports
- Missing core components cause test skips
- Broken test fixtures

**Impact:**
- Cannot validate functionality
- Unknown bugs likely present
- Quality gates failed
- Production deployment blocked

**Action Plan:**

**Step 1:** Fix dependency issue (see #1)

**Step 2:** Fix missing test fixture
```python
# Add to tests/conftest.py
@pytest.fixture
def memory_monitor():
    """Fixture for GPU memory monitoring"""
    from auto_voice.gpu.memory_manager import GPUMemoryMonitor
    return GPUMemoryMonitor()
```

**Step 3:** Re-run tests
```bash
pytest tests/ --cov=. --cov-report=term --cov-report=html
```

**Step 4:** Implement missing components (see P1 issues)

**Verification Target:**
- Coverage ≥ 80%
- Pass rate ≥ 95%
- All critical paths tested

**Estimated Time:** 1-2 weeks (after P1 fixes)
**Owner:** QA Team + Developers

---

## ⚠️ HIGH PRIORITY ISSUES (P1)

### 4. Missing Core Components

**Components Not Available:**

**a) VoiceProfileStorage** (affects 22 tests)
- Required for: Voice profile management, speaker encoding
- Location: Should be in `src/auto_voice/storage/voice_profiles.py`
- Status: Module exists but import fails

**Fix:**
```bash
# Verify module
python -c "from src.auto_voice.storage.voice_profiles import VoiceProfileStorage; print('OK')"

# If import fails, check __init__.py exports
```

**b) VocalSeparator** (affects 1 test)
- Required for: Audio source separation
- Location: Should be in `src/auto_voice/audio/source_separator.py`
- Status: Module exists but class not exported

**Fix:**
```python
# Verify __init__.py includes:
from .source_separator import VocalSeparator
```

**c) SingingPitchExtractor** (affects 2 tests)
- Required for: Pitch extraction from singing voice
- Location: `src/auto_voice/audio/pitch_extractor.py`
- Status: Module exists but class not exported

**d) SingingVoiceConverter** (affects 1 test)
- Required for: Singing voice conversion
- Location: `src/auto_voice/models/singing_voice_converter.py`
- Status: Module exists but import fails

**General Fix Strategy:**
1. Check if classes exist in modules
2. Verify __init__.py exports
3. Fix import paths
4. Add missing implementations if needed

**Verification:**
```bash
python -c "from src.auto_voice.storage.voice_profiles import VoiceProfileStorage; print('Storage OK')"
python -c "from src.auto_voice.audio.source_separator import VocalSeparator; print('Separator OK')"
python -c "from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor; print('Pitch OK')"
python -c "from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter; print('Converter OK')"
```

**Estimated Time:** 3-5 days
**Owner:** Core Development Team

---

### 5. Missing CUDA Kernel: launch_pitch_detection

**Issue:** `cuda_kernels.launch_pitch_detection` not found

**Impact:**
- CUDA kernel profiling fails
- Performance benchmarks incomplete
- GPU acceleration not validated
- Falls back to slower PyTorch implementation

**File:** `src/auto_voice/gpu/cuda_kernels.py`

**Current Status:**
```
Custom CUDA kernels not available, using PyTorch fallbacks
```

**Required Implementation:**
```python
# In cuda_kernels.py, add:
def launch_pitch_detection(
    audio: torch.Tensor,
    sample_rate: int,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 1100.0
) -> torch.Tensor:
    """
    CUDA-accelerated pitch detection kernel

    Args:
        audio: Input audio tensor [batch, samples]
        sample_rate: Audio sample rate
        hop_length: Hop length for frame analysis
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        pitch: Pitch contour [batch, frames]
    """
    # Implement CUDA kernel or PyTorch CUDA-optimized version
    pass
```

**Verification:**
```bash
python scripts/profile_cuda_kernels.py --kernel pitch_detection --iterations 30
```

**Estimated Time:** 1-2 weeks (CUDA expertise required)
**Owner:** GPU Optimization Team

---

## 📋 MEDIUM PRIORITY (P2)

### 6. Mock Implementations in Benchmarks

**Issue:** TTS and quality metrics use mock implementations

**Impact:**
- Cannot validate real performance
- Unknown if targets are achievable
- Production metrics uncertain

**Action:**
- Replace mock TTSPipeline with real implementation
- Use actual audio for quality evaluation
- Re-run benchmarks with real workloads

**Estimated Time:** 3-5 days
**Owner:** Performance Team

---

### 7. Code Quality Issues

**Minor Linting Issues:**
- Missing final newlines (2 files)
- Missing module docstrings
- Wrong import order

**Fix:**
```bash
# Auto-fix with tools
black src/ tests/
isort src/ tests/
pylint src/ --disable=C0114,C0304 --fix
```

**Estimated Time:** 2-3 hours
**Owner:** Any Developer

---

## 📊 Success Criteria

**Before marking as Production Ready:**

- [ ] All P0 issues resolved (dependency, syntax, coverage)
- [ ] All P1 issues resolved (components, CUDA kernels)
- [ ] Test coverage ≥ 80%
- [ ] Test pass rate ≥ 95%
- [ ] No critical linting errors
- [ ] Real (not mock) benchmarks passing targets
- [ ] End-to-end integration tests passing
- [ ] Performance validation on production hardware
- [ ] Security audit completed
- [ ] Load testing completed
- [ ] Documentation updated with known issues

---

## 🗓️ Estimated Timeline

**Optimistic (assuming no blockers):**
```
Week 1:
- Day 1-2: Fix GLIBCXX dependency (P0)
- Day 2: Fix syntax error (P0)
- Day 3-5: Implement missing components (P1)

Week 2:
- Day 1-5: Implement CUDA kernels (P1)

Week 3:
- Day 1-3: Fix test infrastructure
- Day 4-5: Achieve 80% coverage

Total: 3 weeks
```

**Realistic (with typical blockers):**
```
Week 1-2:
- Dependency resolution and environment setup
- Component implementation and integration

Week 3-4:
- CUDA kernel development and optimization
- Test infrastructure improvements

Week 5-6:
- Test coverage achievement
- Performance validation
- Bug fixes

Total: 6 weeks
```

**Conservative (with significant challenges):**
```
Week 1-3: Infrastructure and dependencies
Week 4-6: Core component implementation
Week 7-8: CUDA optimization
Week 9-10: Testing and validation

Total: 10 weeks
```

---

## 🔍 Quick Verification Checklist

Run these commands to verify fixes:

```bash
# 1. Dependency check
python -c "import scipy; print('✅ Scipy OK')" || echo "❌ Fix GLIBCXX"

# 2. Syntax check
python -m py_compile src/auto_voice/web/websocket_handler.py && echo "✅ Syntax OK" || echo "❌ Fix syntax"

# 3. Component imports
python -c "from src.auto_voice.storage.voice_profiles import VoiceProfileStorage; print('✅ Storage OK')" 2>/dev/null || echo "❌ Fix VoiceProfileStorage"

# 4. CUDA kernel
python -c "from src.auto_voice.gpu.cuda_kernels import launch_pitch_detection; print('✅ CUDA OK')" 2>/dev/null || echo "❌ Fix CUDA kernel"

# 5. Test execution
pytest tests/test_performance.py -x && echo "✅ Tests OK" || echo "❌ Fix tests"

# 6. Coverage
pytest tests/ --cov=. --cov-fail-under=80 && echo "✅ Coverage OK" || echo "❌ Improve coverage"
```

---

## 📞 Escalation Path

**P0 Issues (Immediate):**
- Contact: Tech Lead / DevOps Manager
- SLA: Same day resolution required

**P1 Issues (High Priority):**
- Contact: Engineering Manager
- SLA: Resolution within 1 week

**P2 Issues (Medium Priority):**
- Contact: Team Lead
- SLA: Resolution within 2 weeks

---

## 📝 Notes

- This action plan is based on validation results from November 9, 2025
- All file paths are relative to `/home/kp/autovoice`
- Estimated times assume full-time dedicated effort
- Re-validation required after each major fix
- Update this document as issues are resolved

---

**Generated by:** QA Tester Agent #2
**Validation Report:** `/home/kp/autovoice/docs/validation/production_readiness_report.md`
**Status:** ACTIVE - BLOCKERS PRESENT
