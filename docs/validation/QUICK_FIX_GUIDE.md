# Quick Fix Guide - AutoVoice Critical Issues

**For Developers: Step-by-step fixes for production blockers**

---

## 🚀 5-Minute Quick Fixes

### Fix #1: GLIBCXX Dependency (2 minutes) ✅ FIXED

**Status:** RESOLVED (2025-11-09)

```bash
# SOLUTION APPLIED: Symlink to system libstdc++ (PERMANENT FIX)
mv $CONDA_PREFIX/lib/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6

# Verify fix
python -c "import scipy; import librosa; import sklearn; print('✅ All imports OK!')"
```

**Result:**
- ✅ scipy (v1.13.1) - Working
- ✅ librosa (v0.10.2.post1) - Working
- ✅ sklearn (v1.6.1) - Working
- ✅ torch (v2.9.0+cu128) - Working
- ✅ torchaudio - Working

**Details:** See `/home/kp/autovoice/docs/validation/GLIBCXX_FIX_APPLIED.md`

### Fix #2: Syntax Error (1 minute)

**File:** `src/auto_voice/web/websocket_handler.py`

**Line 737:** Add indentation or remove empty else block

```bash
# Quick edit
nano +737 src/auto_voice/web/websocket_handler.py

# Or use sed
sed -i '737i\    pass' src/auto_voice/web/websocket_handler.py

# Verify
python -m py_compile src/auto_voice/web/websocket_handler.py && echo "✅ Fixed!"
```

### Fix #3: Missing memory_monitor Fixture (2 minutes)

**File:** `tests/conftest.py`

Add this fixture:

```python
@pytest.fixture
def memory_monitor():
    """GPU memory monitoring fixture"""
    from src.auto_voice.gpu.memory_manager import GPUMemoryMonitor
    return GPUMemoryMonitor()
```

```bash
# Verify
pytest tests/test_performance.py::TestQualityVsSpeedTradeoffs::test_memory_usage_vs_quality_tradeoff --collect-only
```

---

## ⚙️ 30-Minute Fixes

### Fix #4: Component Import Issues

**Problem:** Components exist but aren't importable

**Solution:** Check and fix `__init__.py` exports

```bash
# Check what's missing
python -c "from src.auto_voice.storage.voice_profiles import VoiceProfileStorage" 2>&1
python -c "from src.auto_voice.audio.source_separator import VocalSeparator" 2>&1
python -c "from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor" 2>&1
python -c "from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter" 2>&1
```

**Fix storage/__init__.py:**
```python
from .voice_profiles import VoiceProfileStorage
```

**Fix audio/__init__.py:**
```python
from .source_separator import VocalSeparator
from .pitch_extractor import SingingPitchExtractor
```

**Fix models/__init__.py:**
```python
from .singing_voice_converter import SingingVoiceConverter
```

**Verify all:**
```bash
python -c "
from src.auto_voice.storage.voice_profiles import VoiceProfileStorage
from src.auto_voice.audio.source_separator import VocalSeparator
from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor
from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter
print('✅ All components importable!')
"
```

---

## 🔧 Complete Environment Setup

### Clean Environment (Recommended for GLIBCXX issues)

```bash
# 1. Create fresh environment
conda create -n autovoice-fixed python=3.11 -y
conda activate autovoice-fixed

# 2. Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "
import torch
import scipy
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print('✅ Environment ready!')
"
```

---

## 🧪 Verification Suite

Run after each fix to verify progress:

```bash
#!/bin/bash
# save as verify_fixes.sh

echo "=== AutoVoice Fix Verification ==="
echo ""

# Test 1: Dependencies
echo "1. Testing dependencies..."
python -c "import scipy; print('✅ Scipy OK')" 2>/dev/null || echo "❌ Fix GLIBCXX"

# Test 2: Syntax
echo "2. Testing syntax..."
python -m py_compile src/auto_voice/web/websocket_handler.py 2>/dev/null && echo "✅ Syntax OK" || echo "❌ Fix syntax error"

# Test 3: Component imports
echo "3. Testing component imports..."
python -c "from src.auto_voice.storage.voice_profiles import VoiceProfileStorage" 2>/dev/null && echo "✅ VoiceProfileStorage OK" || echo "❌ Fix storage imports"
python -c "from src.auto_voice.audio.source_separator import VocalSeparator" 2>/dev/null && echo "✅ VocalSeparator OK" || echo "❌ Fix audio imports"
python -c "from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor" 2>/dev/null && echo "✅ SingingPitchExtractor OK" || echo "❌ Fix pitch imports"
python -c "from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter" 2>/dev/null && echo "✅ SingingVoiceConverter OK" || echo "❌ Fix model imports"

# Test 4: Run basic tests
echo "4. Testing test execution..."
pytest tests/test_performance.py::TestQualityRegressionDetection -v 2>/dev/null && echo "✅ Tests running" || echo "❌ Fix test infrastructure"

# Test 5: Check coverage capability
echo "5. Testing coverage..."
pytest tests/test_performance.py::TestQualityRegressionDetection --cov=src.auto_voice 2>/dev/null && echo "✅ Coverage working" || echo "❌ Fix coverage setup"

echo ""
echo "=== Verification Complete ==="
```

```bash
chmod +x verify_fixes.sh
./verify_fixes.sh
```

---

## 📊 Progress Tracking

Check off fixes as you complete them:

```markdown
- [ ] Fix #1: GLIBCXX dependency
- [ ] Fix #2: Syntax error (line 737)
- [ ] Fix #3: memory_monitor fixture
- [ ] Fix #4a: VoiceProfileStorage import
- [ ] Fix #4b: VocalSeparator import
- [ ] Fix #4c: SingingPitchExtractor import
- [ ] Fix #4d: SingingVoiceConverter import
- [ ] Run verification suite (all pass)
- [ ] Run full test suite
- [ ] Check coverage ≥80%
- [ ] Re-run benchmarks
```

---

## 🎯 Success Criteria

After fixes, you should see:

```bash
# 1. Dependency check
$ python -c "import scipy; print('OK')"
OK

# 2. Syntax check
$ python -m py_compile src/auto_voice/web/websocket_handler.py && echo "OK"
OK

# 3. Import check
$ python -c "from src.auto_voice.storage.voice_profiles import VoiceProfileStorage; print('OK')"
OK

# 4. Test execution
$ pytest tests/test_performance.py -x
===== 30 passed in 15.2s =====

# 5. Coverage
$ pytest tests/ --cov=src.auto_voice --cov-report=term
TOTAL    14949   2989    80%
```

---

## 🆘 Troubleshooting

### Still getting GLIBCXX errors?

```bash
# Check which libstdc++ is being used
ldd $(python -c "import scipy.fft._pocketfft.pypocketfft as p; print(p.__file__)") | grep libstdc++

# Force use of system library
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Or install gcc-11
conda install gcc_linux-64=11.2.0 -y
```

### Imports still failing?

```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Add project root
export PYTHONPATH=/home/kp/autovoice:$PYTHONPATH

# Verify sys.path
python -c "import sys; print('\n'.join(sys.path))"
```

### Tests still skipping?

```bash
# Check why tests are skipping
pytest tests/test_performance.py -v -rs

# Run with verbose output
pytest tests/test_performance.py -vv --tb=short
```

---

## 📞 Need Help?

1. **Check full report:** `/home/kp/autovoice/docs/validation/production_readiness_report.md`
2. **Check action plan:** `/home/kp/autovoice/docs/validation/CRITICAL_ISSUES_ACTION_PLAN.md`
3. **Check validation summary:** `/home/kp/autovoice/docs/validation/VALIDATION_SUMMARY.md`

---

## 🔄 After Fixes - Re-validate

```bash
# Run comprehensive validation
python scripts/run_comprehensive_benchmarks.py --quick

# Expected output:
# ✓ pytest
# ✓ pipeline
# ✓ cuda_kernels
# ✓ tts
# ✓ quality

# Check results
cat validation_results/benchmarks/nvidia_geforce_rtx_3080_ti/benchmark_summary.json
```

---

**Last Updated:** November 9, 2025
**Validation Report:** See `production_readiness_report.md`
