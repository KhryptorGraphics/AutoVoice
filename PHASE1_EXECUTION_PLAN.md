# Phase 1 Execution Plan: Fix PyTorch Environment and Build CUDA Extensions

## Executive Summary

**Important Clarification**: The user's request mentions "Phase 1: Run setup_pytorch_env.sh Option 2 (Python 3.12 downgrade)", but this step is **already complete**. The actual work needed for Phase 1 is:

1. ✅ **Already Done**: Python 3.12.12 environment created and PyTorch 2.5.1+cu121 installed
2. ⚠️ **Needs Action**: Install system CUDA toolkit with complete headers
3. ⚠️ **Needs Action**: Build CUDA extensions (currently blocked by missing headers)
4. ⚠️ **Needs Action**: Verify bindings and validate PyTorch CUDA functionality

## Current Environment Status

### ✅ What's Already Working

Based on `PYTORCH_ENVIRONMENT_FIX_REPORT.md`:

- **Python Environment**: `autovoice_py312` conda environment with Python 3.12.12
- **PyTorch Installation**: PyTorch 2.5.1+cu121 installed via pip
- **CUDA Runtime**: `libtorch_global_deps.so` present and functional
- **PyTorch CUDA**: `torch.cuda.is_available()` returns `True`
- **GPU Detection**: NVIDIA GeForce RTX 3080 Ti detected
- **Dependencies**: All 104 project packages installed

### ⚠️ Current Blocker

**CUDA Extension Build Failure**:
```
fatal error: nv/target: No such file or directory
   14 | #include <nv/target>
```

**Root Cause**: Conda's CUDA toolkit (12.1) is incomplete - missing critical headers required for building CUDA extensions.

**Solution Required**: Install system CUDA toolkit 12.1 from NVIDIA with complete header files.

## Pre-Execution Verification

Before proceeding, verify the current state:

```bash
# 1. Check Python version (should be 3.12.x)
python --version

# 2. Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 3. Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Check for libtorch_global_deps.so
find ~/miniconda3/envs/autovoice_py312 -name "libtorch_global_deps.so"

# 5. Check CUDA toolkit (expected to show missing headers)
./scripts/check_cuda_toolkit.sh

# 6. Run comprehensive pre-flight check
./scripts/phase1_preflight_check.sh
```

**Expected Results**:
- Python 3.12.12 ✅
- PyTorch 2.5.1+cu121 ✅
- CUDA available: True ✅
- libtorch_global_deps.so found ✅
- CUDA toolkit headers: Missing ❌

## Step-by-Step Execution Plan

### Step 1: Activate Environment

```bash
conda activate autovoice_py312
```

**Verification**:
```bash
which python  # Should show ~/miniconda3/envs/autovoice_py312/bin/python
python --version  # Should show Python 3.12.12
```

### Step 2: Install System CUDA Toolkit

**Option A: Using Automated Script (Recommended)**

```bash
./scripts/install_cuda_toolkit.sh
# Or for non-interactive mode:
./scripts/install_cuda_toolkit.sh --yes
```

This script will:
- Download CUDA Toolkit 12.1 from NVIDIA
- Install to `/usr/local/cuda-12.1`
- Set up environment variables (CUDA_HOME, PATH, LD_LIBRARY_PATH)
- Verify installation including `nv/target` header
- Use `--yes` or `-y` flag to skip interactive prompts

**Option B: Manual Installation**

```bash
# Download CUDA Toolkit 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Install (requires sudo)
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Set environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Verification**:
```bash
# Reload environment
source ~/.bashrc

# Check nvcc
nvcc --version  # Should show CUDA 12.1

# Check for critical header
ls -la $CUDA_HOME/include/nv/target  # Should exist

# Run comprehensive check
./scripts/check_cuda_toolkit.sh  # Should pass all checks
```

### Step 3: Build CUDA Extensions

```bash
# Clean previous build artifacts
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -type f -delete

# Build extensions
pip install -e .
```

**Expected Output**:
```
Building CUDA extensions...
Compiling audio_kernels.cu...
Compiling fft_kernels.cu...
Compiling training_kernels.cu...
Compiling memory_kernels.cu...
Compiling kernel_wrappers.cu...
Compiling bindings.cpp...
Linking cuda_kernels extension...
Successfully installed auto-voice
```

**Verification**:
```bash
# Check if extension was built
find . -name "cuda_kernels*.so"  # Should find the compiled extension

# Test import
python -c "from auto_voice import cuda_kernels; print('Import successful!')"
```

### Step 4: Verify Bindings

```bash
./scripts/verify_bindings.py
```

**Expected Output**:
```
✅ Module imported successfully
✅ Function 'launch_pitch_detection' is exposed
✅ Function 'launch_vibrato_analysis' is exposed
✅ Function is callable
✅ Memory stability test passed
```

### Step 5: Validate PyTorch CUDA

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test basic CUDA operation
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f'CUDA tensor operation: Success')
"
```

**Expected Output**:
```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3080 Ti
CUDA tensor operation: Success
```

### Step 6: Run Full Build and Test

```bash
./scripts/build_and_test.sh
```

This will:
- Verify CUDA toolkit
- Build extensions
- Run verification tests
- Generate build report

## Verification Checklist

- [ ] Python 3.12.12 environment active
- [ ] PyTorch 2.5.1+cu121 installed
- [ ] System CUDA toolkit 12.1 installed
- [ ] `nv/target` header exists at `$CUDA_HOME/include/nv/target`
- [ ] CUDA extensions built successfully
- [ ] `cuda_kernels.so` file exists
- [ ] `from auto_voice import cuda_kernels` works
- [ ] `launch_pitch_detection` function exposed
- [ ] `launch_vibrato_analysis` function exposed
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] CUDA tensor operations work
- [ ] No errors in `build.log`

## Troubleshooting Guide

### Issue: "nv/target: No such file or directory"

**Cause**: CUDA toolkit headers not installed or CUDA_HOME not set correctly.

**Solution**:
```bash
# Check CUDA_HOME
echo $CUDA_HOME  # Should be /usr/local/cuda-12.1

# Check if header exists
ls -la $CUDA_HOME/include/nv/target

# If missing, reinstall CUDA toolkit
./scripts/install_cuda_toolkit.sh
```

### Issue: "nvcc not found"

**Cause**: CUDA toolkit not in PATH.

**Solution**:
```bash
# Add to PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH

# Make permanent
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue: "ImportError: cannot import name 'cuda_kernels'"

**Cause**: Extensions not built or build failed.

**Solution**:
```bash
# Check build log
cat build.log

# Rebuild
pip install -e . --force-reinstall --no-cache-dir
```

### Issue: "torch.cuda.is_available() returns False"

**Cause**: PyTorch not built with CUDA support or CUDA runtime missing.

**Solution**:
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Should show +cu121 suffix
# If not, reinstall PyTorch
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Success Criteria

Phase 1 is complete when:

1. ✅ Python 3.12.12 environment is active
2. ✅ PyTorch 2.5.1+cu121 is installed and working
3. ✅ System CUDA toolkit 12.1 is installed with all headers
4. ✅ CUDA extensions build without errors
5. ✅ `cuda_kernels` module imports successfully
6. ✅ Required functions (`launch_pitch_detection`, `launch_vibrato_analysis`) are exposed
7. ✅ `torch.cuda.is_available()` returns `True`
8. ✅ Basic CUDA tensor operations work
9. ✅ `PHASE1_COMPLETION_REPORT.md` is generated with all checks passing

## Next Steps (Phase 2)

After Phase 1 completion:

1. Run comprehensive tests on CUDA kernels
2. Validate audio processing functionality
3. Benchmark performance (CPU vs GPU)
4. Test memory management
5. Verify all kernel functions work correctly
6. Run integration tests with real audio data

## Quick Start

For automated execution of all steps:

```bash
# Run pre-flight check
./scripts/phase1_preflight_check.sh

# Execute Phase 1 (interactive mode)
./scripts/phase1_execute.sh

# Or execute Phase 1 (non-interactive mode)
./scripts/phase1_execute.sh --yes

# Review completion report
cat PHASE1_COMPLETION_REPORT.md
```

**Non-Interactive Mode**: Use the `--yes` or `-y` flag with both `phase1_execute.sh` and `install_cuda_toolkit.sh` to skip all interactive prompts and proceed with default options. This is useful for automated CI/CD pipelines or scripted installations.

