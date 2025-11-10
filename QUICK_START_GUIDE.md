# AutoVoice - Quick Start Guide

**Last Updated:** November 1, 2025  
**Current Status:** PyTorch ✅ Working | CUDA Toolkit ⚠️ Required

---

## 🎯 What You Need to Do

The PyTorch environment issue is **RESOLVED**. You now need to install the CUDA Toolkit to build GPU-accelerated extensions.

---

## 🚀 Quick Installation (Choose One Method)

### Method 1: Conda Installation (Easiest) ⭐ RECOMMENDED

```bash
# Install CUDA Toolkit 12.8 in current environment
conda install -c nvidia cuda-toolkit=12.8 -y

# Verify installation
nvcc --version

# Build CUDA extensions
cd /home/kp/autovoice
pip install -e . --force-reinstall --no-deps

# Verify CUDA kernels
python -c "from auto_voice import cuda_kernels; print('✓ Success!')"
```

**Time:** ~20 minutes  
**Pros:** Easy, environment-specific, no sudo required  
**Cons:** Larger download (~2-3 GB)

---

### Method 2: System Installation (Production)

```bash
# Download CUDA Toolkit 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_560.28.03_linux.run

# Install (requires sudo password)
sudo sh cuda_12.8.0_560.28.03_linux.run

# Add to environment (one-time setup)
cat >> ~/.bashrc << 'EOF'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

# Reload environment
source ~/.bashrc

# Verify installation
nvcc --version

# Build CUDA extensions
cd /home/kp/autovoice
pip install -e . --force-reinstall --no-deps

# Verify CUDA kernels
python -c "from auto_voice import cuda_kernels; print('✓ Success!')"
```

**Time:** ~25 minutes  
**Pros:** System-wide, persistent, production-ready  
**Cons:** Requires sudo, larger installation

---

## ✅ Verification Steps

After installation, run these commands to verify everything works:

```bash
# 1. Check CUDA Toolkit
nvcc --version
# Expected: CUDA compilation tools, release 12.8

# 2. Check PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch: 2.9.0+cu128, CUDA: True

# 3. Check CUDA Kernels
python -c "from auto_voice import cuda_kernels; print('CUDA kernels loaded!')"
# Expected: CUDA kernels loaded!

# 4. Run quick verification script
python scripts/verify_bindings.py
# Expected: All checks pass (2-5 seconds)

# 5. Run full test suite (optional, takes 30-180 min)
bash scripts/build_and_test.sh
# Expected: 151+ tests pass with GPU acceleration
```

---

## 📊 What's Already Working

✅ **Python Environment:** 3.13.5 (Anaconda)  
✅ **PyTorch:** 2.9.0+cu128 (fully functional)  
✅ **CUDA Runtime:** 12.8 (GPU operations working)  
✅ **GPU:** NVIDIA GeForce RTX 3080 Ti (detected)  
✅ **Dependencies:** All Python packages installed  
✅ **Critical Library:** libtorch_global_deps.so (present)

---

## ⚠️ What's Missing

❌ **CUDA Toolkit:** nvcc compiler not installed  
❌ **CUDA Extensions:** Cannot build without toolkit  
❌ **GPU Kernels:** 6 CUDA files not compiled  
❌ **Tests:** Cannot run GPU-accelerated tests

---

## 🔧 Troubleshooting

### Issue: "nvcc: command not found" after conda install
```bash
# Verify conda installed it
conda list | grep cuda-toolkit

# Check if nvcc is in conda bin
ls $CONDA_PREFIX/bin/nvcc

# If exists, add to PATH
export PATH=$CONDA_PREFIX/bin:$PATH
```

### Issue: "CUDA extensions build failed"
```bash
# Check CUDA_HOME is set
echo $CUDA_HOME

# If empty, set it
export CUDA_HOME=$CONDA_PREFIX  # for conda install
# OR
export CUDA_HOME=/usr/local/cuda-12.8  # for system install

# Retry build
cd /home/kp/autovoice
pip install -e . --force-reinstall --no-deps
```

### Issue: "ImportError: cannot import name 'cuda_kernels'"
```bash
# Check if extensions were built
ls build/lib*/auto_voice/cuda_kernels*.so

# If missing, rebuild
pip install -e . --force-reinstall --no-deps

# Check for build errors
pip install -e . 2>&1 | grep -i error
```

---

## 📚 Documentation

- **Detailed Report:** `PYTORCH_ENVIRONMENT_FIX_REPORT.md`
- **Current Status:** `docs/CURRENT_STATUS_SUMMARY.md`
- **Original Issue:** `docs/pytorch_library_issue.md`
- **Project Status:** `PROJECT_COMPLETION_REPORT.md`

---

## 🎯 Expected Outcome

After completing the installation:
- ✅ CUDA Toolkit installed and verified
- ✅ CUDA extensions built successfully
- ✅ All 151+ tests executable
- ✅ GPU-accelerated audio processing functional
- ✅ Project 100% complete

**Estimated Total Time:** 20-30 minutes for installation + 5-10 minutes for build + 2-5 minutes for verification = **~30-45 minutes**

---

## 💡 Quick Commands Reference

```bash
# Install CUDA Toolkit (conda method)
conda install -c nvidia cuda-toolkit=12.8 -y

# Build CUDA extensions
cd /home/kp/autovoice && pip install -e . --force-reinstall --no-deps

# Verify everything
nvcc --version && python -c "from auto_voice import cuda_kernels; print('✓ All working!')"

# Run tests
python scripts/verify_bindings.py
```

---

**Need Help?** Check `PYTORCH_ENVIRONMENT_FIX_REPORT.md` for detailed troubleshooting and alternative solutions.

