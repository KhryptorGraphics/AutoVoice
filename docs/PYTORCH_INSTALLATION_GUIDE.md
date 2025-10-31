# PyTorch Installation Guide for AutoVoice

**Last Updated:** October 30, 2025  
**Python Version:** 3.12.12 (recommended)  
**PyTorch Version:** 2.5.1+cu121 (stable)

---

## Quick Start

### Step 1: Install PyTorch (REQUIRED FIRST)

```bash
# Install PyTorch with CUDA 12.1 support (RECOMMENDED)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install AutoVoice Dependencies

```bash
cd /home/kp/autovoice
pip install -r requirements.txt
```

### Step 3: Install AutoVoice Package

```bash
pip install -e .
```

### Step 4: Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Expected Output:**
```
PyTorch: 2.5.1+cu121
CUDA: True
```

---

## Why Install PyTorch Separately?

### The Problem

Previously, `setup.py` included PyTorch in `install_requires`, which caused:

1. **Version Conflicts:** pip would install a different PyTorch version than intended
2. **Import Errors:** setup.py would fail on clean installs (ModuleNotFoundError)
3. **Inconsistent Guidance:** Different installation methods across documentation

### The Solution

**External Installation Contract:**
- PyTorch is a **prerequisite**, not a dependency
- Must be installed **before** running `pip install -e .`
- Installed from **official PyTorch index** to ensure correct version
- `requirements.txt` is the **single source of truth** for installation steps

---

## Installation Methods

### Method 1: pip (RECOMMENDED)

**Advantages:**
- ✅ Avoids Intel MKL symbol conflicts
- ✅ Guaranteed correct version
- ✅ Works with any Python environment (conda, venv, system)
- ✅ Proven successful in environment fix report

**CUDA 12.1 (GPU):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

### Method 2: conda (Alternative)

**Use if:**
- You prefer conda package management
- You're already using conda environments
- You need conda-specific features

**CUDA 12.1 (GPU):**
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**CPU Only:**
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  cpuonly -c pytorch -y
```

---

## Environment Setup Options

### Option A: Existing Environment (Quick)

If you already have Python 3.12:

```bash
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
cd /home/kp/autovoice
pip install -r requirements.txt

# Install package
pip install -e .
```

### Option B: New Conda Environment (Recommended)

Create a fresh Python 3.12 environment:

```bash
# Create environment
conda create -n autovoice_py312 python=3.12 -y
conda activate autovoice_py312

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
cd /home/kp/autovoice
pip install -r requirements.txt

# Install package
pip install -e .
```

### Option C: Python venv (No Conda Required)

If you don't have conda installed:

```bash
# Create virtual environment
python3.12 -m venv /home/kp/autovoice_py312_venv
source /home/kp/autovoice_py312_venv/bin/activate

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
cd /home/kp/autovoice
pip install -r requirements.txt

# Install package
pip install -e .
```

### Option D: Automated Helper Script

Use the automated setup script:

```bash
cd /home/kp/autovoice
./scripts/setup_pytorch_env.sh
# Select Option 2
# Answer 'Y' to create helper script
./scripts/setup_python312_helper.sh
```

**What it does:**
1. Backs up current environment
2. Creates Python 3.12 conda environment
3. Installs PyTorch 2.5.1 with CUDA 12.1
4. Installs all project dependencies
5. Verifies installation (including libtorch_global_deps.so check)

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch'"

**Cause:** Trying to run `pip install -e .` before installing PyTorch

**Solution:**
```bash
# Install PyTorch first
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Then install package
pip install -e .
```

### Error: "ImportError: undefined symbol: iJIT_NotifyEvent"

**Cause:** Intel MKL symbol conflict (conda-installed PyTorch)

**Solution:**
```bash
# Uninstall conda PyTorch
conda uninstall pytorch torchvision torchaudio -y

# Install via pip instead
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Error: "conda: command not found" (Option 2)

**Cause:** Conda is not installed or not in PATH

**Solution 1 - Install Miniconda:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

**Solution 2 - Use venv instead:**
```bash
python3.12 -m venv /home/kp/autovoice_py312_venv
source /home/kp/autovoice_py312_venv/bin/activate
# Continue with pip installation
```

### Error: "libtorch_global_deps.so: No such file or directory"

**Cause:** PyTorch nightly build missing critical library (Python 3.13 issue)

**Solution:** Downgrade to Python 3.12 with stable PyTorch 2.5.1
```bash
# Use automated helper script
./scripts/setup_pytorch_env.sh
# Select Option 2
```

---

## Verification Checklist

After installation, verify everything is working:

```bash
# 1. Check PyTorch version
python -c "import torch; print('PyTorch:', torch.__version__)"
# Expected: PyTorch: 2.5.1+cu121

# 2. Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# Expected: CUDA available: True

# 3. Check libtorch_global_deps.so
python -c "import torch, os; p=os.path.join(os.path.dirname(torch.__file__),'lib','libtorch_global_deps.so'); print('Library exists:', os.path.exists(p))"
# Expected: Library exists: True

# 4. Check GPU detection
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
# Expected: GPU: NVIDIA GeForce RTX 3080 Ti

# 5. Run tests
pytest tests/
```

---

## System Requirements

### Hardware
- **GPU:** NVIDIA GPU with compute capability >= 7.0 (RTX 20xx series or newer)
- **RAM:** 16 GB minimum, 32 GB recommended
- **Storage:** 10 GB free space for PyTorch and dependencies

### Software
- **OS:** Linux (Ubuntu 20.04+ recommended)
- **Python:** 3.12.12 (3.12.x recommended, 3.8+ minimum)
- **CUDA:** 12.1 (for GPU support)
- **cuDNN:** 8.9+ (optional, for optimized operations)

### Verified Configuration
- **Python:** 3.12.12
- **PyTorch:** 2.5.1+cu121
- **CUDA:** 12.1
- **GPU:** NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- **OS:** Linux

---

## Additional Resources

- **PyTorch Official Website:** https://pytorch.org/
- **PyTorch Installation Guide:** https://pytorch.org/get-started/locally/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **Miniconda:** https://docs.conda.io/en/latest/miniconda.html

### Project Documentation
- `requirements.txt` - Complete dependency list with PyTorch installation instructions
- `PYTORCH_ENVIRONMENT_FIX_REPORT.md` - Detailed environment fix report
- `docs/VERIFICATION_COMMENTS_2_IMPLEMENTATION.md` - Implementation details
- `docs/HELPER_SCRIPT_AUTOMATION_IMPLEMENTATION.md` - Helper script documentation

---

## Support

If you encounter issues not covered in this guide:

1. Check `PYTORCH_ENVIRONMENT_FIX_REPORT.md` for detailed troubleshooting
2. Run the automated setup script: `./scripts/setup_pytorch_env.sh`
3. Review the verification comments implementation docs
4. Check PyTorch official documentation

---

**Last Updated:** October 30, 2025  
**Maintainer:** AutoVoice Team

