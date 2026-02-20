# Troubleshooting Guide

This guide covers common errors and solutions when working with AutoVoice. Use this as a self-service resource for resolving GPU memory issues, model loading failures, audio processing errors, and dependency problems.

## Quick Start

If you encounter an error:

1. **Check the error category** in the table of contents below
2. **Find your specific error message** in that section
3. **Follow the solution steps** provided
4. **Run diagnostic commands** to verify the fix
5. If still stuck, see [Getting Help](#getting-help)

## Table of Contents

- [GPU and CUDA Errors](#gpu-and-cuda-errors)
  - CUDA out of memory
  - Insufficient GPU memory for model loading
  - GPU device not available
  - CUDA driver version mismatch

- [Model Loading Errors](#model-loading-errors)
  - Model file not found
  - Checkpoint missing or corrupted
  - Profile not found
  - No LoRA adapter for profile
  - Failed to initialize model

- [Audio Processing Errors](#audio-processing-errors)
  - Input audio contains NaN or Inf values
  - Insufficient audio quality
  - Speaker embedding contains NaN or Inf
  - Audio file format not supported
  - Sample rate mismatch

- [Database Errors](#database-errors)
  - SQLAlchemy session errors
  - Profile not found in database
  - Transaction rollback issues
  - Database connection failures

- [API and Validation Errors](#api-and-validation-errors)
  - Invalid adapter_type parameter
  - Type errors in request validation
  - Missing required parameters
  - Authentication failures

- [Dependency and Environment Errors](#dependency-and-environment-errors)
  - Missing sounddevice package
  - local-attention module not installed
  - PyWorld ARM64/Python 3.13 incompatibility
  - CUDA toolkit version mismatch
  - Python environment conflicts

- [Diagnostic Commands](#diagnostic-commands)
  - Verify GPU access
  - Check model files
  - Test audio processing
  - Validate database connection
  - Debug environment setup

- [Common Workflows](#common-workflows)
  - Fresh installation troubleshooting
  - GPU memory optimization
  - Model debugging workflow
  - Production deployment checklist

- [Getting Help](#getting-help)
  - Collecting diagnostic information
  - Reporting issues
  - Community resources

---

## GPU and CUDA Errors

### CUDA out of memory (OutOfMemoryError)

**Error Message**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause**: GPU memory exhausted during model loading, inference, or training.

**Solutions**:

1. **Immediate Recovery** (automatically handled by AutoVoice):
   ```python
   # AutoVoice automatically calls this on OOM:
   torch.cuda.empty_cache()
   torch.cuda.synchronize()
   ```

2. **Reduce Batch Size**:
   ```python
   # In training or inference configuration
   config.batch_size = 1  # Reduce from default
   config.chunk_size = 8192  # Smaller audio chunks
   ```

3. **Enable Mixed Precision** (reduces memory by 50%):
   ```python
   # For inference
   pipeline = RealtimePipeline(use_fp16=True)

   # For training
   config.use_amp = True  # Automatic Mixed Precision
   ```

4. **Clear Cache Between Operations**:
   ```bash
   # Check current memory usage
   nvidia-smi

   # In Python
   python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
   ```

5. **Monitor Memory Usage**:
   ```python
   from auto_voice.gpu.memory_manager import GPUMemoryManager

   manager = GPUMemoryManager(device='cuda:0', max_fraction=0.9)
   info = manager.get_memory_info()
   print(f"Allocated: {info['allocated_gb']:.2f} GB")
   print(f"Free: {info['free_gb']:.2f} GB")
   print(f"Utilization: {info['utilization']*100:.1f}%")
   ```

6. **Use Smaller Models**:
   - For realtime inference: Use `SimpleDecoder` instead of full So-VITS
   - For training: Reduce model hidden dimensions

**Diagnostic**:
```bash
# Check GPU memory capacity
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Monitor memory during operation
watch -n 1 nvidia-smi
```

**Prevention**:
- Set `max_fraction=0.8` in GPUMemoryManager to reserve headroom
- Enable continuous monitoring: `GPUMemoryMonitor(warning_threshold=0.8)`
- Use gradient checkpointing in training

---

### Insufficient GPU memory for ContentVec encoder

**Error Message**:
```
RuntimeError: Insufficient GPU memory for ContentVec encoder
```

**Cause**: ContentVec model (~768MB) cannot fit in available GPU memory.

**Solutions**:

1. **Clear Existing Allocations**:
   ```python
   # AutoVoice automatically attempts this
   torch.cuda.empty_cache()
   # Check available memory
   print(torch.cuda.memory_summary())
   ```

2. **Use CPU Fallback** (slower, but works):
   ```python
   pipeline = RealtimePipeline(device=torch.device('cpu'))
   ```

3. **Reduce Model Layer**:
   ```python
   # Use lower layer for content extraction (less memory)
   content_encoder = ContentVecEncoder(
       output_dim=768,
       layer=9,  # Default is 12, lower uses less memory
   )
   ```

4. **Sequential Model Loading** (load models one at a time):
   ```python
   # Load ContentVec first
   content_encoder = ContentVecEncoder(...)
   content_encoder.to('cuda')

   # Extract features
   features = content_encoder(audio)

   # Free ContentVec before loading vocoder
   del content_encoder
   torch.cuda.empty_cache()

   # Now load vocoder
   vocoder = HiFiGAN(...)
   ```

**Diagnostic**:
```bash
# Check if GPU has minimum required memory (4GB recommended)
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

**Requirements**:
- Minimum GPU memory: 4GB
- Recommended: 8GB for full pipeline
- For 2GB GPUs: Use CPU inference

---

### GPU device not available

**Error Message**:
```
AssertionError: CUDA is not available. Please check your GPU and CUDA installation.
RuntimeError: No CUDA GPUs are available
```

**Cause**: PyTorch cannot detect CUDA-capable GPU.

**Solutions**:

1. **Verify CUDA Installation**:
   ```bash
   # Check NVIDIA driver
   nvidia-smi

   # Check CUDA toolkit
   nvcc --version

   # Check PyTorch CUDA support
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
   ```

2. **Reinstall PyTorch with CUDA**:
   ```bash
   # Uninstall current PyTorch
   pip uninstall torch torchaudio torchvision

   # Install with CUDA 12.1 support
   pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

   # For CUDA 11.8
   pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check CUDA Device Visibility**:
   ```bash
   # Make GPU visible to CUDA
   export CUDA_VISIBLE_DEVICES=0

   # For multi-GPU, select specific GPU
   export CUDA_VISIBLE_DEVICES=1

   # Verify
   python -c "import torch; print(torch.cuda.device_count())"
   ```

4. **Use CPU as Fallback**:
   ```python
   # AutoVoice automatically falls back to CPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   pipeline = RealtimePipeline(device=device)
   ```

**Diagnostic**:
```bash
# Complete CUDA diagnostic
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('cuDNN version:', torch.backends.cudnn.version())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
    print('Device capability:', torch.cuda.get_device_capability(0))
"
```

**Common Causes**:
- PyTorch installed without CUDA support (CPU-only)
- CUDA driver version incompatible with PyTorch CUDA version
- GPU not CUDA-capable (e.g., Intel/AMD GPUs)
- CUDA_VISIBLE_DEVICES set incorrectly

---

### CUDA driver version mismatch

**Error Message**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
RuntimeError: The NVIDIA driver on your system is too old
```

**Cause**: CUDA driver version doesn't match PyTorch CUDA version.

**Solutions**:

1. **Check Version Compatibility**:
   ```bash
   # Check driver version
   nvidia-smi | grep "Driver Version"

   # Check PyTorch CUDA version
   python -c "import torch; print(torch.version.cuda)"

   # Check installed CUDA toolkit
   nvcc --version
   ```

2. **Compatibility Matrix**:
   - PyTorch CUDA 12.1 requires NVIDIA driver ≥ 525.60.13
   - PyTorch CUDA 11.8 requires NVIDIA driver ≥ 450.80.02
   - PyTorch CUDA 11.7 requires NVIDIA driver ≥ 450.80.02

3. **Update NVIDIA Driver** (if too old):
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-driver-535  # or latest version
   sudo reboot

   # Verify
   nvidia-smi
   ```

4. **Reinstall Matching PyTorch**:
   ```bash
   # For driver version 525+ (CUDA 12.1)
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # For driver version 450-524 (CUDA 11.8)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Use Automated Setup Script**:
   ```bash
   # AutoVoice provides automated detection and installation
   bash scripts/setup_pytorch_env.sh
   ```

**Diagnostic**:
```bash
# Run AutoVoice environment verification
python scripts/verify_bindings.py
```

---

### Failed to initialize model (general GPU error)

**Error Message**:
```
RuntimeError: Failed to initialize ContentVec: <error details>
RuntimeError: Failed to initialize HiFiGAN: model file missing
```

**Cause**: Model initialization failed due to missing files, corrupted checkpoints, or GPU errors.

**Solutions**:

1. **Verify Model Files Exist**:
   ```bash
   # Check pretrained models
   ls -lh models/pretrained/

   # Expected files:
   # - sovits5.0_main_1500.pth (So-VITS model)
   # - hifigan_ljspeech.ckpt (HiFiGAN vocoder)
   # - hubert-soft-0d54a1f4.pt (HuBERT/ContentVec)
   ```

2. **Download Missing Models**:
   ```bash
   # Use AutoVoice download script
   python scripts/download_pretrained_models.py

   # Or manually download and place in models/pretrained/
   ```

3. **Verify Checkpoint Integrity**:
   ```python
   # Check if checkpoint loads
   import torch
   checkpoint = torch.load('models/pretrained/sovits5.0_main_1500.pth')
   print(checkpoint.keys())
   ```

4. **Check File Permissions**:
   ```bash
   # Ensure files are readable
   chmod 644 models/pretrained/*.pth
   chmod 644 models/pretrained/*.pt
   chmod 644 models/pretrained/*.ckpt
   ```

5. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)

   # Now run pipeline initialization to see detailed error
   pipeline = RealtimePipeline()
   ```

**Diagnostic**:
```bash
# Verify all bindings and models
python scripts/verify_bindings.py

# Check GPU availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Test model loading
python -c "
from auto_voice.inference import RealtimePipeline
try:
    pipeline = RealtimePipeline()
    print('Pipeline initialized successfully')
except Exception as e:
    print(f'Failed: {e}')
"
```

**Recovery Steps**:
1. Clear GPU cache: `torch.cuda.empty_cache()`
2. Download models: `python scripts/download_pretrained_models.py`
3. Verify setup: `python scripts/verify_bindings.py`
4. Check logs: `tail -f logs/autovoice.log`

---

### GPU Memory Leak During Long Operations

**Symptoms**:
- Memory usage grows over time
- OOM errors after many iterations
- Memory not released after inference

**Solutions**:

1. **Enable Continuous Monitoring**:
   ```python
   from auto_voice.gpu.memory_manager import GPUMemoryMonitor

   monitor = GPUMemoryMonitor(
       device='cuda:0',
       interval_ms=1000,
       warning_threshold=0.8,
       on_warning=lambda snap: print(f"Warning: {snap.utilization*100:.1f}% used")
   )
   monitor.start()

   # Your long-running operation
   for i in range(1000):
       result = pipeline.process(audio)

   monitor.stop()
   ```

2. **Explicit Cleanup After Each Iteration**:
   ```python
   for audio_chunk in stream:
       output = pipeline.process(audio_chunk)

       # Explicit cleanup
       del output
       if i % 100 == 0:  # Every 100 iterations
           torch.cuda.empty_cache()
   ```

3. **Use Context Managers**:
   ```python
   with torch.no_grad():  # Disable gradient tracking
       for audio in batch:
           output = model(audio)
   ```

4. **Monitor Allocations**:
   ```python
   from auto_voice.gpu.memory_manager import GPUMemoryTracker

   tracker = GPUMemoryTracker(device='cuda:0')
   tracker.record_allocation('content_encoder', content_encoder)
   tracker.record_allocation('vocoder', vocoder)

   stats = tracker.get_stats()
   print(f"Total allocated: {stats['total_size_mb']:.2f} MB")
   ```

**Diagnostic**:
```bash
# Watch memory during operation
watch -n 0.5 nvidia-smi

# Profile memory usage
python -c "
import torch
torch.cuda.memory._record_memory_history(max_entries=100000)
# Run your operation
torch.cuda.memory._dump_snapshot('memory_snapshot.pickle')
"
```

## Model Loading Errors

*This section will be populated with model loading error solutions.*

## Audio Processing Errors

*This section will be populated with audio processing error solutions.*

## Database Errors

*This section will be populated with database error solutions.*

## API and Validation Errors

*This section will be populated with API and validation error solutions.*

## Dependency and Environment Errors

*This section will be populated with dependency error solutions.*

## Diagnostic Commands

*This section will be populated with diagnostic commands and utilities.*

## Common Workflows

*This section will be populated with troubleshooting workflows.*

## Getting Help

### Collecting Diagnostic Information

Before reporting an issue, collect the following information:

1. **System Information**:
   ```bash
   # OS and hardware
   uname -a

   # GPU information
   nvidia-smi

   # CUDA version
   nvcc --version
   ```

2. **Python Environment**:
   ```bash
   # Python version
   python --version

   # Installed packages
   pip list

   # PyTorch/CUDA status
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
   ```

3. **AutoVoice Configuration**:
   ```bash
   # Check configuration files
   cat config/gpu_config.yaml

   # Verify model files
   ls -lh models/pretrained/

   # Check logs
   tail -n 100 logs/autovoice.log
   ```

### Reporting Issues

When reporting an issue, include:

- Error message (full stack trace)
- System information (see above)
- Steps to reproduce
- Expected vs actual behavior
- Configuration files (remove sensitive data)

### Community Resources

- **Documentation**: See `docs/` directory for guides
- **Test Examples**: Check `tests/` for usage examples
- **Code Examples**: See `examples/` for sample code
- **Deployment Guide**: `docs/deployment-guide.md` for production setup

---

*Note: This guide is actively maintained. Sections are being populated with detailed error solutions and diagnostic procedures.*
