# AutoVoice Scripts

This directory contains automated scripts for setting up, building, and testing the AutoVoice project.

## Overview

The scripts in this directory help automate common tasks and resolve environment issues, particularly the PyTorch library blocker that prevents CUDA extension compilation.

## Scripts

### 1. `setup_pytorch_env.sh` - Environment Setup

**Purpose**: Diagnose and resolve PyTorch installation issues, especially for Python 3.13 compatibility.

**Features**:
- Detects Python and PyTorch versions
- Checks for missing `libtorch_global_deps.so` file
- Verifies CUDA availability
- Provides three solution paths with interactive prompts

**Usage**:
```bash
./scripts/setup_pytorch_env.sh
```

**Solution Options**:

| Option | Description | Time | Success Rate | Recommended For |
|--------|-------------|------|--------------|-----------------|
| 1 | Quick nightly reinstall | 5-10 min | ~40% | Quick attempt before major changes |
| 2 | Python 3.12 downgrade | 30 min | ~95% | **Recommended** - Stable production setup |
| 3 | Build from source | 2+ hours | ~80% | Advanced users needing Python 3.13 |

**When to Run**:
- After cloning the repository
- When PyTorch import fails
- Before building CUDA extensions
- When `libtorch_global_deps.so` is missing

---

### 2. `build_and_test.sh` - Build and Test

**Purpose**: Comprehensive build and test workflow for CUDA extensions and Python code.

**Features**:
- Environment prerequisite checks
- CUDA extension compilation
- Binding verification
- Smoke tests
- Unit tests (with CPU/GPU detection)
- Integration tests (CUDA-specific)
- Automated test reporting

**Usage**:
```bash
./scripts/build_and_test.sh
```

**Test Stages**:
1. **Environment Check**: Python, PyTorch, CUDA availability
2. **Build**: Compile CUDA extensions with `pip install -e .`
3. **Verification**: Check Python bindings are exposed
4. **Smoke Tests**: Basic functionality tests
5. **Unit Tests**: Run pytest with appropriate markers
6. **Integration Tests**: CUDA-specific tests (if GPU available)
7. **Reporting**: Generate timestamped test report

**Exit Codes**:
- `0`: All tests passed
- `1`: Build failed or tests failed

**Logs Generated**:
- `build.log` - CUDA extension build output
- `verify.log` - Binding verification output
- `smoke_test.log` - Smoke test results
- `pytest.log` - Full pytest output
- `test_report_YYYYMMDD_HHMMSS.txt` - Summary report

**When to Run**:
- After fixing PyTorch environment
- After modifying CUDA kernels
- Before committing code changes
- In CI/CD pipeline

---

### 3. `verify_bindings.py` - Quick Binding Verification

**Purpose**: Fast verification that CUDA extension bindings are properly exposed.

**Features**:
- Module import test
- Function exposure verification
- PyTorch and CUDA availability checks
- Tensor creation tests
- Function signature validation
- Colorful output with status indicators

**Usage**:
```bash
python ./scripts/verify_bindings.py
# or
./scripts/verify_bindings.py
```

**Checks Performed**:
1. PyTorch is installed and importable
2. CUDA is available (with graceful CPU fallback)
3. `cuda_kernels` module can be imported
4. Required functions are exposed:
   - `launch_pitch_detection`
   - `launch_vibrato_analysis`
5. Tensors can be created on appropriate device
6. Functions are callable with correct signatures

**Output Example**:
```
╔════════════════════════════════════════════════════════╗
║     CUDA Bindings Verification                         ║
╚════════════════════════════════════════════════════════╝

[ℹ] Checking PyTorch availability...
[✓] PyTorch 2.5.1 available
[✓] CUDA available: NVIDIA GeForce RTX 3090

[ℹ] Testing module import...
[✓] Module 'cuda_kernels' imported successfully

[ℹ] Checking exposed functions...
[✓] Function 'launch_pitch_detection' is exposed
[✓] Function 'launch_vibrato_analysis' is exposed

[ℹ] Testing tensor creation...
[✓] CPU tensor creation successful
[✓] GPU tensor creation successful

╔════════════════════════════════════════════════════════╗
║  ALL CHECKS PASSED!                                     ║
║  CUDA bindings are properly exposed and functional      ║
╚════════════════════════════════════════════════════════╝
```

**When to Run**:
- After building CUDA extensions
- Before running full tests
- After environment changes
- For quick sanity checks

---

## Workflow Recommendations

### Initial Setup
```bash
# 1. Fix PyTorch environment
./scripts/setup_pytorch_env.sh

# 2. Follow recommended solution (typically Option 2 for Python 3.12)

# 3. Build and test everything
./scripts/build_and_test.sh
```

### Development Workflow
```bash
# After code changes:
./scripts/verify_bindings.py  # Quick check
./scripts/build_and_test.sh   # Full validation
```

### CI/CD Pipeline
```bash
# In your CI pipeline:
./scripts/setup_pytorch_env.sh  # Ensure environment is correct
./scripts/build_and_test.sh     # Build and validate

# Check exit code
if [ $? -eq 0 ]; then
    echo "Build and tests passed"
else
    echo "Build or tests failed"
    exit 1
fi
```

---

## Troubleshooting

### PyTorch Import Fails

**Problem**: `OSError: libtorch_global_deps.so: cannot open shared object file`

**Solution**:
```bash
./scripts/setup_pytorch_env.sh
# Choose Option 1 (quick fix) or Option 2 (recommended)
```

### Build Fails

**Problem**: CUDA extension compilation errors

**Check**:
1. CUDA Toolkit is installed: `nvcc --version`
2. PyTorch with CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Correct CUDA version matches PyTorch: Check `torch.__version__` and `nvcc --version`

**Solution**:
```bash
# Clean build artifacts
rm -rf build/
pip uninstall auto-voice -y

# Rebuild
./scripts/build_and_test.sh
```

### Tests Fail

**Problem**: Unit or integration tests failing

**Debug Steps**:
1. Check test logs: `pytest.log`, `cuda_test_*.log`
2. Run specific test: `pytest tests/test_file.py::TestClass::test_method -v`
3. Verify bindings: `./scripts/verify_bindings.py`

**Common Issues**:
- CUDA out of memory: Reduce batch sizes in tests
- Missing test dependencies: `pip install pytest pytest-cov`
- GPU not available: Tests should skip automatically with markers

### CUDA Not Available

**Problem**: Tests skip CUDA tests even with GPU

**Check**:
```bash
nvidia-smi                                          # Driver
nvcc --version                                      # Toolkit
python -c "import torch; print(torch.cuda.is_available())"  # PyTorch
```

**Solution**:
- Install NVIDIA drivers
- Install CUDA Toolkit
- Reinstall PyTorch with CUDA support:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

---

## Environment Requirements

### Python Versions
- **Recommended**: Python 3.12 (stable PyTorch support)
- **Experimental**: Python 3.13 (limited PyTorch support)
- **Minimum**: Python 3.8

### PyTorch Versions
- **Recommended**: PyTorch 2.5.1+ with CUDA 12.1
- **Minimum**: PyTorch 2.0+

### CUDA Requirements
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- **CUDA Toolkit**: 11.8 or later (12.1 recommended)
- **Driver**: Recent NVIDIA driver supporting your CUDA version

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), WSL2, macOS (CPU only)
- **Disk Space**: 5GB for environment, 20GB for source builds
- **RAM**: 8GB minimum, 16GB recommended
- **GPU Memory**: 4GB minimum for inference, 8GB+ for training

---

## Script Permissions

All scripts should be executable:

```bash
chmod +x scripts/*.sh
chmod +x scripts/verify_bindings.py
```

---

## Additional Helper Scripts

### `setup_python312_helper.sh` (Auto-generated)

Created by `setup_pytorch_env.sh` when choosing Option 2.

**Purpose**: Automated Python 3.12 environment creation

**Usage**:
```bash
./scripts/setup_python312_helper.sh
conda activate autovoice_py312
./scripts/setup_pytorch_env.sh  # Re-run to install PyTorch
```

---

## Reference Documentation

For more details on the issues these scripts resolve, see:

- `/home/kp/autovoice/docs/pytorch_library_issue.md` - Complete PyTorch issue analysis
- `/home/kp/autovoice/docs/cuda_bindings_fix_summary.md` - CUDA binding fix details
- `/home/kp/autovoice/README.md` - Project overview and usage

---

## Contributing

When adding new scripts:

1. Follow the naming convention: `<action>_<target>.sh`
2. Include a header comment describing the script
3. Use colored output for better UX
4. Provide clear error messages
5. Return appropriate exit codes
6. Update this README with documentation

---

## Support

For issues with these scripts:

1. Check the troubleshooting section above
2. Review log files generated by scripts
3. Consult the reference documentation
4. Open an issue on GitHub with:
   - Script output
   - Environment details (`python --version`, `nvcc --version`)
   - Full error messages

---

**Last Updated**: 2025-10-27
