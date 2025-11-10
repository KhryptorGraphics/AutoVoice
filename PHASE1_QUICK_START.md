# Phase 1 Quick Start Guide

## TL;DR - Just Run This

```bash
# 1. Activate environment
conda activate autovoice_py312

# 2. Run Phase 1 execution
./scripts/phase1_execute.sh

# 3. Review results
cat PHASE1_EXECUTION_SUMMARY.txt
```

That's it! The script handles everything automatically.

---

## What Phase 1 Actually Does

**Important**: The user's request mentions "Python 3.12 downgrade" but **this is already done**. 

Phase 1 actually does:

1. ✅ **Already Complete**: Python 3.12.12 environment with PyTorch 2.5.1+cu121
2. 🔧 **Needs Action**: Install system CUDA toolkit with complete headers
3. 🔧 **Needs Action**: Build CUDA extensions
4. 🔧 **Needs Action**: Verify bindings work correctly

---

## Current Status

Run this to see what's already done:

```bash
./scripts/phase1_preflight_check.sh
```

**Expected Output**:
- ✅ Python 3.12.12 installed
- ✅ PyTorch 2.5.1+cu121 installed
- ✅ CUDA available in PyTorch
- ✅ GPU detected (RTX 3080 Ti)
- ⚠️ CUDA headers missing (this is what we'll fix)

---

## Step-by-Step Manual Execution

If you prefer to run steps manually instead of using the automated script:

### Step 1: Activate Environment

```bash
conda activate autovoice_py312
python --version  # Should show 3.12.12
```

### Step 2: Install CUDA Toolkit

```bash
./scripts/install_cuda_toolkit.sh
```

This installs system CUDA toolkit 12.1 with all headers (requires sudo).

### Step 3: Build CUDA Extensions

```bash
pip install -e .
```

This compiles the CUDA kernels and creates the `cuda_kernels` extension.

### Step 4: Verify Bindings

```bash
./scripts/verify_bindings.py
```

This checks that all functions are exposed and callable.

### Step 5: Test PyTorch CUDA

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Should print: `CUDA available: True`

---

## Troubleshooting

### Error: "nv/target: No such file or directory"

**Cause**: CUDA toolkit headers not installed

**Fix**:
```bash
./scripts/install_cuda_toolkit.sh
```

### Error: "Module 'cuda_kernels' not found"

**Cause**: Extensions not built

**Fix**:
```bash
pip install -e .
```

### Error: "nvcc not found"

**Cause**: CUDA toolkit not in PATH

**Fix**:
```bash
source ~/.bashrc
# Or manually set:
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
```

### Error: "torch.cuda.is_available() returns False"

**Cause**: PyTorch not built with CUDA support

**Fix**:
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## Verification Commands

After Phase 1, verify everything works:

```bash
# 1. Check Python version
python --version

# 2. Check PyTorch
python -c "import torch; print(torch.__version__)"

# 3. Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# 4. Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 5. Check CUDA extensions
python -c "from auto_voice import cuda_kernels; print('Success!')"

# 6. Check functions
python -c "from auto_voice import cuda_kernels; print(dir(cuda_kernels))"
```

**Expected Results**:
- Python 3.12.12
- PyTorch 2.5.1+cu121
- CUDA available: True
- GPU: NVIDIA GeForce RTX 3080 Ti
- cuda_kernels imports successfully
- Functions include: `launch_pitch_detection`, `launch_vibrato_analysis`

---

## Files to Review

After execution, check these files:

1. **PHASE1_EXECUTION_SUMMARY.txt** - Quick summary of what was done
2. **build.log** - Full build output (if build failed)
3. **cuda_check.log** - CUDA toolkit validation results

---

## What's Next (Phase 2)

After Phase 1 completes successfully:

1. Run comprehensive tests on CUDA kernels
2. Validate audio processing functionality
3. Benchmark performance (CPU vs GPU)
4. Test memory management
5. Run integration tests with real audio data

---

## Time Estimates

- **Pre-flight check**: < 1 minute
- **CUDA toolkit installation**: 5-10 minutes (requires download)
- **Extension building**: 2-5 minutes
- **Verification**: < 1 minute
- **Total**: ~10-20 minutes

---

## Requirements

- **Sudo access**: Required for CUDA toolkit installation
- **Internet connection**: Required to download CUDA toolkit
- **Disk space**: ~3 GB for CUDA toolkit
- **GPU**: NVIDIA GPU with CUDA support (you have RTX 3080 Ti ✅)

---

## Support

If you encounter issues:

1. Check error messages - they include specific fix commands
2. Review `PHASE1_EXECUTION_PLAN.md` for detailed troubleshooting
3. Run `./scripts/check_cuda_toolkit.sh` for diagnostics
4. Check `build.log` for build errors

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./scripts/phase1_preflight_check.sh` | Check current status |
| `./scripts/phase1_execute.sh` | Run full Phase 1 |
| `./scripts/install_cuda_toolkit.sh` | Install CUDA toolkit only |
| `./scripts/build_and_test.sh` | Build and test extensions |
| `./scripts/verify_bindings.py` | Verify bindings only |
| `./scripts/check_cuda_toolkit.sh` | Diagnose CUDA issues |

---

**Ready to start?**

```bash
conda activate autovoice_py312
./scripts/phase1_execute.sh
```

Good luck! 🚀

