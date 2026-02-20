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

### Profile not found

**Error Message**:
```
ValueError: Profile not found: 7da05140-...
ValueError: Voice model not found: <model_id>
```

**Cause**: Voice profile does not exist in the profiles directory or database.

**Solutions**:

1. **List Available Profiles**:
   ```bash
   # Check existing profile files
   ls -lh data/voice_profiles/*.json

   # List profile IDs
   for f in data/voice_profiles/*.json; do
     echo "$(basename $f .json): $(jq -r '.name' $f)"
   done
   ```

2. **Verify Profile ID Format**:
   ```python
   # Profile IDs must be valid UUIDs
   import uuid

   profile_id = "7da05140-e5a7-4e89-b2c3-8f6d9a1c2b3e"
   try:
       uuid.UUID(profile_id)
       print("Valid UUID")
   except ValueError:
       print("Invalid UUID format")
   ```

3. **Create Missing Profile**:
   ```python
   from auto_voice.inference.voice_cloner import VoiceCloner

   cloner = VoiceCloner()
   profile = cloner.create_profile(
       name="Artist Name",
       audio_files=["path/to/audio1.wav", "path/to/audio2.wav"],
       min_duration=30.0  # Minimum 30 seconds recommended
   )
   print(f"Created profile: {profile.profile_id}")
   ```

4. **Check Database Consistency**:
   ```python
   from auto_voice.storage.voice_profiles import VoiceProfileManager

   manager = VoiceProfileManager()
   profiles = manager.list_profiles()
   for p in profiles:
       print(f"{p.profile_id}: {p.name} ({p.status})")
   ```

5. **Restore from Backup**:
   ```bash
   # If profile was accidentally deleted
   ls -lh data/backups/voice_profiles/
   cp data/backups/voice_profiles/<profile_id>.json data/voice_profiles/
   cp data/backups/voice_profiles/<profile_id>.npy data/voice_profiles/
   ```

**Diagnostic**:
```bash
# Verify profile directory structure
tree data/voice_profiles/ -L 1

# Check profile file integrity
python -c "
import json
from pathlib import Path

profile_dir = Path('data/voice_profiles')
for profile_file in profile_dir.glob('*.json'):
    try:
        with open(profile_file) as f:
            data = json.load(f)
        print(f'✓ {profile_file.stem}: {data.get(\"name\", \"Unknown\")}')
    except Exception as e:
        print(f'✗ {profile_file.stem}: {e}')
"
```

**Prevention**:
- Always validate profile IDs before using them
- Implement profile existence checks in API endpoints
- Use `VoiceProfileManager` for centralized profile access
- Enable automatic backups in production

---

### No LoRA adapter for profile

**Error Message**:
```
FileNotFoundError: No LoRA found for profile: 7da05140-...
RuntimeError: Failed to load LoRA weights: [Errno 2] No such file or directory
```

**Cause**: LoRA adapter has not been trained for the voice profile, or training failed.

**Solutions**:

1. **Check LoRA File Existence**:
   ```bash
   # Expected LoRA location
   ls -lh data/trained_models/hq/

   # Pattern: <profile_id>_hq_lora.pt
   ls -lh data/trained_models/hq/*_hq_lora.pt
   ```

2. **Train LoRA Adapter**:
   ```python
   from auto_voice.training.lora_trainer import LoRATrainer

   trainer = LoRATrainer(
       profile_id="7da05140-...",
       training_data_dir="data/separated_youtube/artist_name",
       output_dir="data/trained_models/hq",
       epochs=100,
       batch_size=4,
   )

   # Start training
   trainer.train()

   # Check output
   print(f"LoRA saved to: {trainer.output_path}")
   ```

3. **Verify Training Completion**:
   ```bash
   # Check training logs
   tail -f logs/training.log

   # Verify LoRA checkpoint
   python -c "
   import torch
   checkpoint = torch.load('data/trained_models/hq/<profile_id>_hq_lora.pt')
   print('Checkpoint keys:', checkpoint.keys())
   print('Epoch:', checkpoint.get('epoch', 'N/A'))
   print('Loss:', checkpoint.get('loss', 'N/A'))
   "
   ```

4. **Use Seed-VC Pipeline (No LoRA Required)**:
   ```python
   # Seed-VC uses reference audio instead of LoRA
   from auto_voice.inference import PipelineFactory

   pipeline = PipelineFactory.create_pipeline(
       pipeline_type='quality',  # Uses Seed-VC by default
       device='cuda'
   )

   # No LoRA needed - uses reference audio
   result = pipeline.convert_audio(
       audio_path="input.wav",
       profile_id="7da05140-..."  # Just needs voice profile
   )
   ```

5. **Copy LoRA from Another Environment**:
   ```bash
   # Transfer trained LoRA from training server
   scp user@training-server:~/autovoice/data/trained_models/hq/<profile_id>_hq_lora.pt \
       data/trained_models/hq/

   # Verify integrity
   python -c "import torch; torch.load('data/trained_models/hq/<profile_id>_hq_lora.pt'); print('Valid')"
   ```

**Diagnostic**:
```bash
# Check all trained LoRAs
find data/trained_models -name "*_lora.pt" -ls

# Verify LoRA structure
python -c "
import torch
from pathlib import Path

lora_dir = Path('data/trained_models/hq')
for lora_file in lora_dir.glob('*_hq_lora.pt'):
    try:
        checkpoint = torch.load(lora_file, map_location='cpu')
        profile_id = lora_file.stem.replace('_hq_lora', '')
        size_mb = lora_file.stat().st_size / (1024 * 1024)
        print(f'✓ {profile_id}: {size_mb:.1f}MB, keys={list(checkpoint.keys())}')
    except Exception as e:
        print(f'✗ {lora_file.stem}: {e}')
"
```

**When LoRA is Not Required**:
- Seed-VC pipeline (uses in-context learning with reference audio)
- MeanVC pipeline (uses statistical mean voice conversion)
- Voice identification/matching (only needs speaker embeddings)

---

### Model file not found (pretrained models)

**Error Message**:
```
RuntimeError: Failed to initialize HiFiGAN: checkpoint missing
FileNotFoundError: ContentVec model file not found
FileNotFoundError: RMVPE model file not found
RuntimeError: Model weights not found: models/pretrained/sovits5.0_main_1500.pth
```

**Cause**: Required pretrained model checkpoints are missing from `models/pretrained/`.

**Solutions**:

1. **Download All Required Models**:
   ```bash
   # Use AutoVoice download script
   python scripts/download_pretrained_models.py

   # This downloads:
   # - hubert-soft-35d9f29f.pt (361MB) - Content encoder
   # - generator_universal.pth.tar (55MB) - HiFiGAN vocoder
   ```

2. **Verify Model Files**:
   ```bash
   # Check expected models
   ls -lh models/pretrained/

   # Expected files:
   # - hubert-soft-*.pt (or hubert-soft-0d54a1f4.pt)
   # - generator_universal.pth.tar (or hifigan_ljspeech.ckpt)
   # - sovits5.0_main_1500.pth (optional, for original pipeline)
   ```

3. **Verify Model Integrity**:
   ```python
   import torch
   from pathlib import Path

   models_dir = Path('models/pretrained')

   models = {
       'hubert-soft': list(models_dir.glob('hubert-soft-*.pt')),
       'hifigan': list(models_dir.glob('generator_universal.pth.tar')) or
                  list(models_dir.glob('hifigan_*.ckpt')),
   }

   for name, paths in models.items():
       if not paths:
           print(f'✗ Missing: {name}')
           continue

       for path in paths:
           try:
               checkpoint = torch.load(path, map_location='cpu')
               size_mb = path.stat().st_size / (1024 * 1024)
               print(f'✓ {name}: {path.name} ({size_mb:.0f}MB)')
           except Exception as e:
               print(f'✗ {name}: {path.name} - {e}')
   ```

4. **Manual Download** (if script fails):
   ```bash
   # Create directory
   mkdir -p models/pretrained
   cd models/pretrained

   # Download HuBERT-Soft
   wget https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt

   # Download HiFiGAN (from Google Drive)
   # File ID: 1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW
   # Or use gdown: pip install gdown
   gdown 1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW -O generator_universal.pth.tar
   ```

5. **Check Configuration Paths**:
   ```python
   # Verify config points to correct paths
   from auto_voice.config import load_config

   config = load_config()
   print(f"Pretrained dir: {config['model']['pretrained_dir']}")
   print(f"HuBERT checkpoint: {config['model'].get('hubert_checkpoint', 'N/A')}")
   print(f"HiFiGAN checkpoint: {config['model'].get('hifigan_checkpoint', 'N/A')}")
   ```

**Diagnostic**:
```bash
# Run full verification
python scripts/verify_bindings.py

# Check specific models
python -c "
from pathlib import Path
import torch

pretrained = Path('models/pretrained')

required = {
    'HuBERT/ContentVec': ['hubert-soft-*.pt'],
    'HiFiGAN': ['generator_universal.pth.tar', 'hifigan_*.ckpt'],
}

for name, patterns in required.items():
    found = False
    for pattern in patterns:
        files = list(pretrained.glob(pattern))
        if files:
            print(f'✓ {name}: {files[0].name}')
            found = True
            break
    if not found:
        print(f'✗ {name}: MISSING')
"
```

**Expected File Sizes**:
- HuBERT-Soft: ~361MB
- HiFiGAN: ~55MB
- So-VITS: ~184MB (if used)

**Prevention**:
- Run `scripts/download_pretrained_models.py` during setup
- Add model verification to CI/CD pipeline
- Keep checksums to verify integrity
- Use volume mounts in Docker to persist models

---

### Checkpoint missing or corrupted

**Error Message**:
```
RuntimeError: Failed to initialize HiFiGAN: checkpoint missing
EOFError: Ran out of input
RuntimeError: PytorchStreamReader failed reading zip archive
```

**Cause**: Model checkpoint file is corrupted, incomplete, or in wrong format.

**Solutions**:

1. **Verify File Integrity**:
   ```bash
   # Check file size (should not be 0 or very small)
   ls -lh models/pretrained/*.pt
   ls -lh models/pretrained/*.pth
   ls -lh models/pretrained/*.tar

   # Files smaller than 1MB are likely corrupted
   find models/pretrained -type f -size -1M
   ```

2. **Test Loading**:
   ```python
   import torch

   checkpoint_path = "models/pretrained/hubert-soft-35d9f29f.pt"

   try:
       checkpoint = torch.load(checkpoint_path, map_location='cpu')
       print(f"✓ Checkpoint loaded successfully")
       print(f"  Keys: {list(checkpoint.keys())[:5]}")

       # Check for expected structure
       if isinstance(checkpoint, dict):
           print(f"  Type: State dict")
       else:
           print(f"  Type: {type(checkpoint)}")
   except EOFError:
       print("✗ File is incomplete or corrupted")
   except Exception as e:
       print(f"✗ Failed to load: {e}")
   ```

3. **Re-download Corrupted Files**:
   ```bash
   # Remove corrupted file
   rm models/pretrained/hubert-soft-*.pt

   # Re-download
   python scripts/download_pretrained_models.py

   # Or manual download
   cd models/pretrained
   wget https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt
   ```

4. **Verify Checksum** (if available):
   ```bash
   # Calculate MD5 checksum
   md5sum models/pretrained/hubert-soft-35d9f29f.pt

   # Compare with expected checksum (from model release)
   ```

5. **Check Disk Space**:
   ```bash
   # Ensure sufficient disk space
   df -h .

   # Check if download was interrupted
   du -sh models/pretrained/*
   ```

**Diagnostic**:
```bash
# Test all checkpoints
python -c "
import torch
from pathlib import Path

pretrained = Path('models/pretrained')

for checkpoint_file in pretrained.glob('*.pt'):
    try:
        ckpt = torch.load(checkpoint_file, map_location='cpu')
        size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        print(f'✓ {checkpoint_file.name}: {size_mb:.0f}MB')
    except EOFError:
        print(f'✗ {checkpoint_file.name}: Corrupted (incomplete)')
    except Exception as e:
        print(f'✗ {checkpoint_file.name}: {type(e).__name__}')

for checkpoint_file in pretrained.glob('*.pth'):
    try:
        ckpt = torch.load(checkpoint_file, map_location='cpu')
        size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        print(f'✓ {checkpoint_file.name}: {size_mb:.0f}MB')
    except Exception as e:
        print(f'✗ {checkpoint_file.name}: {type(e).__name__}')
"
```

**Common Causes**:
- Download interrupted (network failure)
- Disk full during download
- File system corruption
- Wrong file format (e.g., HTML error page saved as .pt)
- Incorrect git-lfs setup (if using git for models)

---

### Could not load embedding for model

**Error Message**:
```
ValueError: Could not load embedding for model: <model_id>
RuntimeError: Speaker embedding contains NaN or Inf
FileNotFoundError: [Errno 2] No such file or directory: 'data/voice_profiles/<profile_id>.npy'
```

**Cause**: Speaker embedding file is missing, corrupted, or contains invalid values.

**Solutions**:

1. **Check Embedding File**:
   ```bash
   # List embedding files
   ls -lh data/voice_profiles/*.npy

   # Embedding files should be paired with JSON profiles
   for json in data/voice_profiles/*.json; do
       npy="${json%.json}.npy"
       if [ -f "$npy" ]; then
           echo "✓ $(basename $json .json)"
       else
           echo "✗ $(basename $json .json) - missing .npy"
       fi
   done
   ```

2. **Regenerate Embedding**:
   ```python
   from auto_voice.inference.voice_cloner import VoiceCloner
   import numpy as np

   cloner = VoiceCloner()

   # Re-compute embedding from audio
   profile_id = "7da05140-..."
   audio_files = [
       "data/separated_youtube/artist_name/song1_vocals.wav",
       "data/separated_youtube/artist_name/song2_vocals.wav",
   ]

   # Regenerate
   embedding = cloner.compute_embedding(audio_files)

   # Save
   np.save(f"data/voice_profiles/{profile_id}.npy", embedding)
   print(f"Embedding shape: {embedding.shape}")
   print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
   ```

3. **Validate Embedding**:
   ```python
   import numpy as np

   profile_id = "7da05140-..."
   embedding_path = f"data/voice_profiles/{profile_id}.npy"

   try:
       embedding = np.load(embedding_path)

       # Check for NaN/Inf
       if np.isnan(embedding).any():
           print("✗ Embedding contains NaN")
       elif np.isinf(embedding).any():
           print("✗ Embedding contains Inf")
       else:
           print(f"✓ Valid embedding: shape={embedding.shape}, "
                 f"range=[{embedding.min():.3f}, {embedding.max():.3f}]")

       # Expected shape: (256,) for mel-statistics or (192,) for WavLM
       print(f"  Shape: {embedding.shape}")

   except FileNotFoundError:
       print(f"✗ Embedding file not found: {embedding_path}")
   except Exception as e:
       print(f"✗ Failed to load: {e}")
   ```

4. **Check Audio Quality**:
   ```python
   # Ensure reference audio is clean
   import torchaudio

   audio, sr = torchaudio.load("reference_audio.wav")

   # Check for silent or clipped audio
   if audio.abs().max() < 0.01:
       print("✗ Audio is too quiet (likely silent)")
   elif audio.abs().max() > 0.99:
       print("⚠ Audio may be clipped")
   else:
       print("✓ Audio level OK")

   # Check duration (minimum 10 seconds recommended)
   duration = audio.shape[-1] / sr
   if duration < 10:
       print(f"⚠ Audio too short: {duration:.1f}s (recommend ≥10s)")
   else:
       print(f"✓ Duration: {duration:.1f}s")
   ```

5. **Use Different Reference Audio**:
   ```python
   # Try with different/better quality audio
   from auto_voice.inference.voice_cloner import VoiceCloner

   cloner = VoiceCloner()

   # Use longer, cleaner samples
   better_audio = [
       "high_quality_vocal1.wav",  # Clear, isolated vocals
       "high_quality_vocal2.wav",  # No background noise
   ]

   embedding = cloner.compute_embedding(better_audio)
   ```

**Diagnostic**:
```bash
# Validate all embeddings
python -c "
import numpy as np
from pathlib import Path

profile_dir = Path('data/voice_profiles')

for npy_file in profile_dir.glob('*.npy'):
    try:
        embedding = np.load(npy_file)

        # Check validity
        has_nan = np.isnan(embedding).any()
        has_inf = np.isinf(embedding).any()

        if has_nan or has_inf:
            status = '✗ INVALID'
        else:
            status = '✓'

        print(f'{status} {npy_file.stem}: shape={embedding.shape}, '
              f'range=[{embedding.min():.3f}, {embedding.max():.3f}]')

    except Exception as e:
        print(f'✗ {npy_file.stem}: {e}')
"
```

**Expected Embedding Format**:
- **Mel-statistics**: Shape (256,), L2-normalized, range typically [-3, 3]
- **WavLM**: Shape (192,) or (512,), L2-normalized
- No NaN or Inf values
- Non-zero (not all zeros)

---

### Failed to load TRT engine

**Error Message**:
```
RuntimeError: Failed to load TRT engine: models/trt_engines/content_encoder.engine
FileNotFoundError: TRT engine not found
RuntimeError: TensorRT engine version mismatch
```

**Cause**: TensorRT engine files are missing, incompatible, or corrupted.

**Solutions**:

1. **Build TRT Engines**:
   ```bash
   # Build all required engines
   python scripts/build_trt_engines.py --output-dir models/trt_engines

   # This creates:
   # - content_encoder.engine
   # - pitch_estimator.engine
   # - decoder.engine
   # - vocoder.engine
   ```

2. **Check Engine Files**:
   ```bash
   # List existing engines
   ls -lh models/trt_engines/*.engine

   # Verify size (should be 50-500MB each)
   du -sh models/trt_engines/*.engine
   ```

3. **Verify TensorRT Version**:
   ```python
   try:
       import tensorrt as trt
       print(f"TensorRT version: {trt.__version__}")

       # Check CUDA compatibility
       print(f"CUDA version: {torch.version.cuda}")

   except ImportError:
       print("TensorRT not installed")
       print("Install with: pip install tensorrt")
   ```

4. **Rebuild for Current GPU**:
   ```bash
   # Engines are GPU-specific, rebuild if changing GPUs
   # Remove old engines
   rm -rf models/trt_engines/*.engine

   # Rebuild for current GPU
   python scripts/build_trt_engines.py --fp16 --output-dir models/trt_engines

   # Verify
   python -c "
   from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
   available = TRTStreamingPipeline.engines_available('models/trt_engines')
   print(f'TRT engines available: {available}')
   "
   ```

5. **Fall Back to Non-TRT Pipeline**:
   ```python
   # Use regular pipeline if TRT unavailable
   from auto_voice.inference import PipelineFactory

   # Explicitly disable TRT
   pipeline = PipelineFactory.create_pipeline(
       pipeline_type='streaming',
       use_trt=False,
       device='cuda'
   )
   ```

**Diagnostic**:
```bash
# Check TensorRT installation
python -c "
import subprocess
import sys

try:
    import tensorrt as trt
    print(f'✓ TensorRT installed: {trt.__version__}')
except ImportError:
    print('✗ TensorRT not installed')
    sys.exit(1)

# Check engines
from pathlib import Path
engine_dir = Path('models/trt_engines')

required_engines = ['content_encoder', 'decoder', 'vocoder']
for name in required_engines:
    engine_path = engine_dir / f'{name}.engine'
    if engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f'✓ {name}: {size_mb:.0f}MB')
    else:
        print(f'✗ {name}: Missing')
"
```

**Requirements**:
- TensorRT 8.6+ installed
- CUDA 11.8+ or 12.1+
- GPU with Compute Capability 7.5+ (Turing, Ampere, Ada, Hopper)
- Engines must be rebuilt when changing GPU models

**Performance Notes**:
- TRT engines reduce latency to <50ms
- Build time: 5-15 minutes per engine
- Engines are not portable between GPUs
- FP16 mode reduces memory and increases speed

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
