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

### Input audio contains NaN or Inf values

**Error Message**:
```
RuntimeError: Audio contains NaN or Inf values
ValueError: Invalid audio data - contains NaN or Inf
RuntimeError: Speaker embedding contains NaN or Inf
```

**Cause**: Audio data contains invalid numerical values (Not-a-Number or Infinity), typically from:
- Corrupted audio files
- Numerical instability during processing
- Division by zero in audio transformations
- Invalid resampling or filtering

**Solutions**:

1. **Validate Input Audio**:
   ```python
   import numpy as np
   import torchaudio

   # Load and check audio
   audio, sr = torchaudio.load("input.wav")

   # Check for NaN or Inf
   if torch.isnan(audio).any():
       print("✗ Audio contains NaN values")
   elif torch.isinf(audio).any():
       print("✗ Audio contains Inf values")
   else:
       print("✓ Audio is valid")

   # Auto-fix by replacing invalid values
   audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
   ```

2. **Check Audio File Integrity**:
   ```bash
   # Verify file is not corrupted
   ffmpeg -v error -i input.wav -f null - 2>&1

   # If corrupted, re-encode to fix
   ffmpeg -i input.wav -ar 44100 -ac 1 -c:a pcm_s16le output.wav
   ```

3. **Clean Audio Data**:
   ```python
   import numpy as np

   # Load audio as numpy
   audio = np.load("audio.npy")

   # Replace invalid values
   audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

   # Clip to valid range [-1.0, 1.0]
   audio = np.clip(audio, -1.0, 1.0)

   # Verify
   assert not np.isnan(audio).any(), "Still contains NaN"
   assert not np.isinf(audio).any(), "Still contains Inf"

   # Save cleaned audio
   np.save("audio_cleaned.npy", audio)
   ```

4. **Fix Speaker Embeddings**:
   ```python
   from auto_voice.inference.voice_cloner import VoiceCloner
   import numpy as np

   # Check embedding file
   embedding_path = "data/voice_profiles/<profile_id>.npy"
   embedding = np.load(embedding_path)

   if np.isnan(embedding).any() or np.isinf(embedding).any():
       print("✗ Embedding is corrupted - regenerating...")

       # Regenerate from clean audio
       cloner = VoiceCloner()
       new_embedding = cloner.create_speaker_embedding([
           "clean_audio1.wav",
           "clean_audio2.wav"
       ])

       # Save new embedding
       np.save(embedding_path, new_embedding)
       print("✓ Embedding regenerated")
   ```

5. **Prevent During Processing**:
   ```python
   import torch

   def safe_normalize(audio: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
       """Normalize audio safely without creating NaN/Inf."""
       # Clamp to prevent extreme values
       audio = torch.clamp(audio, -100.0, 100.0)

       # Normalize with epsilon to prevent division by zero
       max_val = audio.abs().max()
       if max_val > eps:
           audio = audio / max_val

       # Final safety check
       audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

       return audio

   # Use in pipeline
   audio = safe_normalize(audio)
   ```

**Diagnostic**:
```bash
# Check audio files for NaN/Inf
python -c "
import numpy as np
import torchaudio
from pathlib import Path

audio_dir = Path('data/audio')
for audio_file in audio_dir.glob('**/*.wav'):
    try:
        audio, sr = torchaudio.load(audio_file)

        has_nan = torch.isnan(audio).any().item()
        has_inf = torch.isinf(audio).any().item()

        if has_nan or has_inf:
            print(f'✗ {audio_file.name}: NaN={has_nan}, Inf={has_inf}')
        else:
            print(f'✓ {audio_file.name}')
    except Exception as e:
        print(f'✗ {audio_file.name}: {e}')
"
```

**Prevention**:
- Always validate audio after loading
- Use `torch.nan_to_num()` or `np.nan_to_num()` as safety net
- Add epsilon values to prevent division by zero
- Clip audio to valid ranges before processing
- Regenerate corrupted embeddings from source audio

---

### Insufficient audio quality

**Error Message**:
```
InsufficientQualityError: Audio quality too low for voice cloning
InvalidAudioError: Silent audio - cannot extract embedding
ValueError: Audio too short (2.3s). Minimum 3.0s required.
```

**Cause**: Audio quality does not meet minimum requirements for voice conversion:
- Too short duration (< 3 seconds)
- Silent or near-silent audio
- Excessive background noise
- Low sample rate
- Heavy compression artifacts

**Solutions**:

1. **Check Audio Duration**:
   ```python
   import librosa

   audio, sr = librosa.load("input.wav")
   duration = len(audio) / sr

   if duration < 3.0:
       print(f"✗ Audio too short: {duration:.1f}s (minimum 3.0s)")
   else:
       print(f"✓ Duration OK: {duration:.1f}s")
   ```

2. **Check Audio Level**:
   ```python
   import numpy as np
   import librosa

   audio, sr = librosa.load("input.wav")

   # Check RMS energy
   rms = np.sqrt(np.mean(audio**2))
   max_amplitude = np.abs(audio).max()

   if max_amplitude < 0.01:
       print("✗ Audio is too quiet (likely silent)")
       print("  Solution: Use audio with vocals present")
   elif max_amplitude > 0.99:
       print("⚠ Audio may be clipped")
       print("  Solution: Reduce input gain")
   else:
       print(f"✓ Audio level OK (max={max_amplitude:.3f}, rms={rms:.3f})")
   ```

3. **Improve Audio Quality**:
   ```bash
   # Normalize audio to -1dB peak
   ffmpeg -i input.wav -af "loudnorm=I=-1:TP=-1:LRA=7" normalized.wav

   # Remove silence from ends
   ffmpeg -i input.wav -af "silenceremove=start_periods=1:stop_periods=1:detection=peak" trimmed.wav

   # Denoise (reduce background noise)
   ffmpeg -i input.wav -af "afftdn=nf=-20" denoised.wav

   # Combine all improvements
   ffmpeg -i input.wav \
       -af "afftdn=nf=-20,silenceremove=start_periods=1:stop_periods=1,loudnorm=I=-1:TP=-1" \
       enhanced.wav
   ```

4. **Use Longer/Better Quality Audio**:
   ```python
   from auto_voice.inference.voice_cloner import VoiceCloner

   cloner = VoiceCloner()

   # Use multiple high-quality samples (10+ seconds each)
   audio_files = [
       "song1_vocals.wav",  # 30 seconds, clean vocals
       "song2_vocals.wav",  # 25 seconds, isolated vocals
       "song3_vocals.wav",  # 20 seconds, studio quality
   ]

   # Create profile with quality audio
   profile = cloner.create_profile(
       name="Artist Name",
       audio_files=audio_files,
       min_duration=10.0  # Increase minimum duration
   )
   ```

5. **Extract Vocals First**:
   ```python
   from auto_voice.audio.separation import VocalSeparator
   import torchaudio

   # Separate vocals from full mix
   separator = VocalSeparator()
   audio, sr = torchaudio.load("full_song.wav")

   separated = separator.separate(audio.numpy(), sr)
   vocals = separated['vocals']

   # Save isolated vocals
   torchaudio.save("vocals_only.wav",
                    torch.from_numpy(vocals).unsqueeze(0),
                    sr)

   # Use vocals for profile creation
   embedding = cloner.create_speaker_embedding(["vocals_only.wav"])
   ```

**Diagnostic**:
```bash
# Analyze audio quality
python -c "
import librosa
import numpy as np

audio, sr = librosa.load('input.wav')
duration = len(audio) / sr
rms = np.sqrt(np.mean(audio**2))
max_amp = np.abs(audio).max()
zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)

print(f'Duration: {duration:.1f}s')
print(f'Sample rate: {sr} Hz')
print(f'Max amplitude: {max_amp:.3f}')
print(f'RMS energy: {rms:.3f}')
print(f'Zero crossing rate: {zero_crossings:.3f}')

# Quality assessment
if duration < 3.0:
    print('✗ FAIL: Too short')
elif max_amp < 0.01:
    print('✗ FAIL: Too quiet/silent')
elif max_amp > 0.99:
    print('⚠ WARNING: May be clipped')
else:
    print('✓ PASS: Quality OK')
"
```

**Minimum Requirements**:
- **Duration**: ≥ 3 seconds (10+ seconds recommended)
- **Sample Rate**: ≥ 16kHz (44.1kHz recommended)
- **Level**: Peak amplitude 0.1 - 0.95 (avoid silence and clipping)
- **Format**: WAV, FLAC, or lossless (avoid low-bitrate MP3)
- **Content**: Clean vocals (use vocal separation if needed)

---

### Speaker embedding contains NaN or Inf

**Error Message**:
```
RuntimeError: Speaker embedding contains NaN or Inf
ValueError: Could not load embedding for model: <model_id>
FileNotFoundError: [Errno 2] No such file or directory: 'data/voice_profiles/<profile_id>.npy'
```

**Cause**: Speaker embedding file is missing, corrupted, or contains invalid values (see [Could not load embedding for model](#could-not-load-embedding-for-model) in Model Loading Errors section for file-related issues). This section focuses on NaN/Inf value issues.

**Solutions**:

1. **Validate Existing Embedding**:
   ```python
   import numpy as np

   profile_id = "7da05140-..."
   embedding_path = f"data/voice_profiles/{profile_id}.npy"

   try:
       embedding = np.load(embedding_path)

       # Check for invalid values
       if np.isnan(embedding).any():
           print("✗ Embedding contains NaN - regeneration required")
       elif np.isinf(embedding).any():
           print("✗ Embedding contains Inf - regeneration required")
       elif np.all(embedding == 0):
           print("✗ Embedding is all zeros - regeneration required")
       else:
           norm = np.linalg.norm(embedding)
           print(f"✓ Valid embedding: shape={embedding.shape}, norm={norm:.3f}")
           print(f"  Range: [{embedding.min():.3f}, {embedding.max():.3f}]")

   except FileNotFoundError:
       print(f"✗ Embedding file not found: {embedding_path}")
   ```

2. **Regenerate Corrupted Embedding**:
   ```python
   from auto_voice.inference.voice_cloner import VoiceCloner
   import numpy as np

   profile_id = "7da05140-..."
   cloner = VoiceCloner()

   # Use high-quality reference audio
   reference_audio = [
       "data/separated_youtube/artist/song1_vocals.wav",
       "data/separated_youtube/artist/song2_vocals.wav",
   ]

   # Regenerate embedding
   print("Regenerating speaker embedding...")
   embedding = cloner.create_speaker_embedding(reference_audio)

   # Validate before saving
   assert not np.isnan(embedding).any(), "New embedding contains NaN"
   assert not np.isinf(embedding).any(), "New embedding contains Inf"
   assert np.linalg.norm(embedding) > 0, "New embedding is zero"

   # Save
   embedding_path = f"data/voice_profiles/{profile_id}.npy"
   np.save(embedding_path, embedding)
   print(f"✓ Embedding saved: shape={embedding.shape}")
   ```

3. **Fix Silent Audio Issue**:
   ```python
   # If getting "Silent audio - cannot extract embedding"
   import librosa
   import numpy as np

   # Check reference audio is not silent
   for audio_path in reference_audio:
       audio, sr = librosa.load(audio_path)
       max_amp = np.abs(audio).max()

       if max_amp < 0.01:
           print(f"✗ {audio_path} is too quiet/silent")
           print("  → Use different audio or normalize:")

           # Normalize to -3dB
           normalized = audio / max_amp * 0.7
           output_path = audio_path.replace('.wav', '_normalized.wav')

           import soundfile as sf
           sf.write(output_path, normalized, sr)
           print(f"  → Saved normalized version: {output_path}")
   ```

4. **Use Better Reference Audio**:
   ```python
   from auto_voice.inference.voice_cloner import VoiceCloner

   cloner = VoiceCloner(auto_separate_vocals=True)

   # Use full songs - vocals will be auto-extracted
   full_songs = [
       "artist_song1.wav",
       "artist_song2.wav",
       "artist_song3.wav",
   ]

   # VoiceCloner will automatically:
   # 1. Separate vocals from instrumental
   # 2. Extract embeddings from clean vocals
   # 3. Average embeddings for robustness
   profile = cloner.create_profile(
       name="Artist Name",
       audio_files=full_songs
   )

   print(f"Profile created: {profile.profile_id}")
   ```

5. **Batch Validate All Embeddings**:
   ```python
   from pathlib import Path
   import numpy as np

   profile_dir = Path("data/voice_profiles")

   print("Validating all embeddings...")
   for npy_file in profile_dir.glob("*.npy"):
       try:
           embedding = np.load(npy_file)

           has_nan = np.isnan(embedding).any()
           has_inf = np.isinf(embedding).any()
           is_zero = np.all(embedding == 0)
           norm = np.linalg.norm(embedding)

           if has_nan or has_inf or is_zero:
               print(f"✗ {npy_file.stem}: INVALID (NaN={has_nan}, Inf={has_inf}, Zero={is_zero})")
           else:
               print(f"✓ {npy_file.stem}: OK (norm={norm:.3f})")

       except Exception as e:
           print(f"✗ {npy_file.stem}: ERROR - {e}")
   ```

**Diagnostic**:
```bash
# Quick validation of all profiles
python -c "
from pathlib import Path
import numpy as np
import json

profile_dir = Path('data/voice_profiles')

for json_file in profile_dir.glob('*.json'):
    profile_id = json_file.stem
    npy_file = profile_dir / f'{profile_id}.npy'

    # Check files exist
    if not npy_file.exists():
        print(f'✗ {profile_id}: Missing .npy file')
        continue

    # Load and validate
    try:
        with open(json_file) as f:
            profile = json.load(f)

        embedding = np.load(npy_file)

        # Validate
        valid = (
            not np.isnan(embedding).any() and
            not np.isinf(embedding).any() and
            np.linalg.norm(embedding) > 0
        )

        status = '✓' if valid else '✗'
        print(f'{status} {profile_id}: {profile.get(\"name\", \"Unknown\")} ({embedding.shape})')

    except Exception as e:
        print(f'✗ {profile_id}: {e}')
"
```

**Expected Embedding Properties**:
- **Shape**: (256,) for mel-statistics, (192,) or (512,) for WavLM
- **Normalization**: L2-normalized (norm ≈ 1.0)
- **Range**: Typically [-3, 3] for mel-statistics
- **No invalid values**: No NaN, Inf, or all-zeros
- **Paired files**: Both .json and .npy must exist

**Cross-Reference**: See [Could not load embedding for model](#could-not-load-embedding-for-model) for file-not-found issues.

---

### Audio file format not supported

**Error Message**:
```
RuntimeError: Unsupported audio format: .mp3
RuntimeError: Failed to load audio: unknown format
torchaudio.backend.common.BackendNotAvailableError
```

**Cause**: Audio file format is not supported by the audio backend, or required dependencies are missing.

**Solutions**:

1. **Convert to Supported Format**:
   ```bash
   # Convert to WAV (universally supported)
   ffmpeg -i input.mp3 -ar 44100 -ac 1 output.wav

   # Convert to FLAC (lossless)
   ffmpeg -i input.m4a -c:a flac output.flac

   # Batch convert all MP3s to WAV
   for f in *.mp3; do
       ffmpeg -i "$f" -ar 44100 -ac 1 "${f%.mp3}.wav"
   done
   ```

2. **Install Audio Backends**:
   ```bash
   # Install soundfile (recommended for WAV/FLAC)
   pip install soundfile

   # Install ffmpeg (for all formats)
   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg

   # Verify installation
   python -c "import soundfile as sf; print(f'soundfile version: {sf.__version__}')"
   ffmpeg -version
   ```

3. **Use torchaudio with ffmpeg Backend**:
   ```python
   import torchaudio

   # Set backend to ffmpeg (supports more formats)
   torchaudio.set_audio_backend("ffmpeg")

   # Now can load MP3, M4A, AAC, etc.
   audio, sr = torchaudio.load("input.mp3")

   # Convert to WAV for compatibility
   torchaudio.save("output.wav", audio, sr)
   ```

4. **Check Supported Formats**:
   ```python
   import soundfile as sf

   # List available formats
   print("Supported formats:", sf.available_formats())

   # Check if specific file is supported
   try:
       with sf.SoundFile("audio.opus") as f:
           print(f"✓ Format supported: {f.format}")
   except RuntimeError as e:
       print(f"✗ Format not supported: {e}")
   ```

5. **Pre-process Audio Files**:
   ```python
   from pathlib import Path
   import subprocess

   def convert_to_wav(input_path: str, output_dir: str = "converted") -> str:
       """Convert any audio format to WAV."""
       input_path = Path(input_path)
       output_dir = Path(output_dir)
       output_dir.mkdir(exist_ok=True)

       output_path = output_dir / f"{input_path.stem}.wav"

       cmd = [
           "ffmpeg", "-i", str(input_path),
           "-ar", "44100",  # 44.1kHz sample rate
           "-ac", "1",      # Mono
           "-y",            # Overwrite
           str(output_path)
       ]

       try:
           subprocess.run(cmd, check=True, capture_output=True)
           print(f"✓ Converted: {input_path.name} → {output_path.name}")
           return str(output_path)
       except subprocess.CalledProcessError as e:
           print(f"✗ Conversion failed: {e.stderr.decode()}")
           raise

   # Use in pipeline
   wav_path = convert_to_wav("input.m4a")
   # Now use wav_path for processing
   ```

**Supported Formats**:
- **Recommended**: WAV (PCM), FLAC (lossless)
- **With soundfile**: WAV, FLAC, OGG, AIFF
- **With ffmpeg**: MP3, M4A, AAC, OPUS, WMA, and more
- **Avoid**: Very low bitrate MP3 (<128kbps) due to quality loss

**Diagnostic**:
```bash
# Check audio file format
ffprobe -v error -show_entries format=format_name,bit_rate -show_entries stream=codec_name,sample_rate,channels input.mp3

# Test loading with Python
python -c "
import torchaudio
import sys

try:
    audio, sr = torchaudio.load('${1:-input.wav}')
    print(f'✓ Format supported: shape={audio.shape}, sr={sr}')
except Exception as e:
    print(f'✗ Format error: {e}')
    sys.exit(1)
"
```

---

### Sample rate mismatch

**Error Message**:
```
RuntimeError: Sample rate mismatch: expected 16000, got 44100
ValueError: Input audio sample rate (48000) != model sample rate (22050)
```

**Cause**: Audio sample rate doesn't match what the model expects. AutoVoice models use:
- **ContentVec**: 16kHz input
- **HiFiGAN vocoder**: 22.05kHz output
- **Demucs separation**: 44.1kHz

**Solutions**:

1. **Automatic Resampling** (handled by pipeline):
   ```python
   from auto_voice.inference import RealtimePipeline

   # Pipeline handles resampling automatically
   pipeline = RealtimePipeline()

   # Can input any sample rate - will be resampled to 16kHz internally
   import torchaudio
   audio, sr = torchaudio.load("audio_48khz.wav")  # 48kHz input

   # Pipeline resamples automatically
   output = pipeline.process_chunk(audio)  # Outputs at 22.05kHz
   ```

2. **Manual Resampling**:
   ```python
   import torchaudio
   import torchaudio.transforms as T

   # Load audio
   audio, sr = torchaudio.load("input.wav")
   print(f"Original: {sr} Hz")

   # Resample to target rate
   target_sr = 16000
   resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
   audio_resampled = resampler(audio)

   # Save resampled audio
   torchaudio.save("output_16k.wav", audio_resampled, target_sr)
   print(f"Resampled: {target_sr} Hz")
   ```

3. **Batch Resample with ffmpeg**:
   ```bash
   # Resample single file to 16kHz
   ffmpeg -i input.wav -ar 16000 output_16k.wav

   # Batch resample all WAVs to 16kHz
   for f in *.wav; do
       ffmpeg -i "$f" -ar 16000 "resampled/${f%.wav}_16k.wav"
   done

   # Resample to model-specific rates
   # For ContentVec (16kHz)
   ffmpeg -i input.wav -ar 16000 content_input.wav

   # For HiFiGAN (22.05kHz)
   ffmpeg -i input.wav -ar 22050 vocoder_output.wav

   # For Demucs (44.1kHz)
   ffmpeg -i input.wav -ar 44100 separation_input.wav
   ```

4. **Check Sample Rate Before Processing**:
   ```python
   import torchaudio

   def load_and_resample(audio_path: str, target_sr: int = 16000):
       """Load audio and resample if needed."""
       audio, sr = torchaudio.load(audio_path)

       if sr != target_sr:
           print(f"Resampling: {sr} Hz → {target_sr} Hz")
           resampler = torchaudio.transforms.Resample(sr, target_sr)
           audio = resampler(audio)

       return audio, target_sr

   # Use in pipeline
   audio, sr = load_and_resample("input.wav", target_sr=16000)
   assert sr == 16000, "Sample rate mismatch"
   ```

5. **Configure Pipeline Sample Rates**:
   ```python
   from auto_voice.inference import SingingConversionPipeline

   # Pipeline configuration with explicit sample rates
   pipeline = SingingConversionPipeline(
       device='cuda',
       input_sample_rate=44100,   # Your audio sample rate
       model_sample_rate=16000,   # ContentVec requirement
       output_sample_rate=22050,  # HiFiGAN output
   )
   ```

**Diagnostic**:
```bash
# Check sample rates of all audio files
python -c "
import torchaudio
from pathlib import Path

audio_dir = Path('data/audio')
sample_rates = {}

for audio_file in audio_dir.glob('**/*.wav'):
    try:
        info = torchaudio.info(audio_file)
        sr = info.sample_rate

        if sr not in sample_rates:
            sample_rates[sr] = []
        sample_rates[sr].append(audio_file.name)

    except Exception as e:
        print(f'✗ {audio_file.name}: {e}')

# Print summary
for sr, files in sorted(sample_rates.items()):
    print(f'{sr} Hz: {len(files)} files')
    if len(files) <= 5:
        for f in files:
            print(f'  - {f}')
"

# Quick check single file
ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 input.wav
```

**Model Sample Rate Requirements**:
- **ContentVec encoder**: 16kHz input
- **RMVPE pitch**: 16kHz input
- **HiFiGAN vocoder**: 22.05kHz output
- **Demucs separation**: 44.1kHz (model default)
- **So-VITS decoder**: 22.05kHz mel output

**Note**: AutoVoice pipelines handle resampling automatically - you typically don't need to manually resample unless working with raw model inference.

---

## Database Errors

### SQLAlchemy session errors

**Error Message**:
```
sqlalchemy.exc.InvalidRequestError: Object '<VoiceProfile>' is already attached to session
sqlalchemy.orm.exc.DetachedInstanceError: Instance is not bound to a Session
sqlalchemy.exc.PendingRollbackError: This Session's transaction has been rolled back
```

**Cause**: Session lifecycle management issues - improper session handling, detached objects, or uncommitted transactions.

**Solutions**:

1. **Use Context Managers for Sessions**:
   ```python
   from contextlib import contextmanager
   from sqlalchemy.orm import sessionmaker

   Session = sessionmaker(bind=engine)

   @contextmanager
   def get_db_session():
       """Context manager for database sessions."""
       session = Session()
       try:
           yield session
           session.commit()
       except Exception:
           session.rollback()
           raise
       finally:
           session.close()

   # Use in code
   with get_db_session() as session:
       profile = session.query(VoiceProfile).filter_by(id=profile_id).first()
       # Session auto-commits on success, auto-rolls back on error
   ```

2. **Fix Detached Instance Errors**:
   ```python
   from sqlalchemy.orm import Session

   # ✗ WRONG - object becomes detached after session closes
   def get_profile_wrong(profile_id: str):
       session = Session()
       profile = session.query(VoiceProfile).get(profile_id)
       session.close()  # Profile is now detached
       return profile

   # ✓ CORRECT - refresh or use within session scope
   def get_profile_correct(profile_id: str):
       with get_db_session() as session:
           profile = session.query(VoiceProfile).get(profile_id)
           # Access all needed attributes before session closes
           data = {
               'id': profile.id,
               'name': profile.name,
               'created_at': profile.created_at
           }
       return data
   ```

3. **Handle Rollback Errors**:
   ```python
   from sqlalchemy.exc import SQLAlchemyError

   def create_profile_safe(profile_data: dict):
       """Create profile with proper error handling."""
       with get_db_session() as session:
           try:
               profile = VoiceProfile(**profile_data)
               session.add(profile)
               session.flush()  # Check for errors before commit
               return profile.id
           except SQLAlchemyError as e:
               logger.error(f"Database error: {e}")
               # Session automatically rolls back via context manager
               raise
   ```

4. **Fix "Already Attached" Errors**:
   ```python
   # ✗ WRONG - trying to add object to multiple sessions
   profile = VoiceProfile(name="Artist")
   session1.add(profile)
   session2.add(profile)  # Error: already attached to session1

   # ✓ CORRECT - use merge for cross-session objects
   with get_db_session() as session1:
       profile = VoiceProfile(name="Artist")
       session1.add(profile)
       session1.flush()
       profile_id = profile.id

   # Later, in different session
   with get_db_session() as session2:
       # Use merge to re-attach to new session
       profile = session2.merge(profile)
       profile.name = "Updated Name"
   ```

5. **Enable Session Auto-Expire**:
   ```python
   # Configure session to prevent detached errors
   Session = sessionmaker(
       bind=engine,
       expire_on_commit=False,  # Keep objects usable after commit
   )

   # Alternative: explicitly refresh objects
   with get_db_session() as session:
       profile = session.query(VoiceProfile).get(profile_id)
       session.commit()
       session.refresh(profile)  # Re-load from database
       # Now safe to access profile attributes
   ```

**Diagnostic**:
```bash
# Test database connection
python -c "
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///data/autovoice.db')
Session = sessionmaker(bind=engine)

session = Session()
try:
    result = session.execute('SELECT 1')
    print('✓ Database connection OK')
finally:
    session.close()
"
```

**Prevention**:
- Always use context managers for session management
- Set `expire_on_commit=False` if accessing objects after commit
- Use `session.refresh()` to reload detached objects
- Never share session objects across threads
- Log all SQLAlchemy errors with full stack traces

---

### Profile not found in database

**Error Message**:
```
ValueError: Profile not found: 7da05140-e5a7-4e89-b2c3-8f6d9a1c2b3e
FileNotFoundError: Voice profile file not found
ProfileNotFoundError: No profile with ID: <profile_id>
```

**Cause**: Voice profile does not exist in database or file storage, or was deleted.

**Solutions**:

1. **Verify Profile Exists Before Access**:
   ```python
   from auto_voice.storage.voice_profiles import VoiceProfileStore, ProfileNotFoundError

   store = VoiceProfileStore()

   def get_profile_safe(profile_id: str):
       """Get profile with existence check."""
       try:
           profile = store.get(profile_id)
           if profile is None:
               raise ProfileNotFoundError(f"Profile not found: {profile_id}")
           return profile
       except FileNotFoundError:
           logger.error(f"Profile file missing: {profile_id}")
           raise ProfileNotFoundError(f"Profile not found: {profile_id}")

   # Use in API
   try:
       profile = get_profile_safe(profile_id)
   except ProfileNotFoundError:
       return jsonify({'error': 'Profile not found'}), 404
   ```

2. **List Available Profiles**:
   ```python
   from pathlib import Path
   import json

   def list_all_profiles():
       """List all available voice profiles."""
       profiles_dir = Path('data/voice_profiles')
       profiles = []

       for profile_file in profiles_dir.glob('*.json'):
           try:
               with open(profile_file) as f:
                   profile_data = json.load(f)
               profiles.append({
                   'id': profile_data['profile_id'],
                   'name': profile_data['name'],
                   'created_at': profile_data.get('created_at', 'Unknown')
               })
           except Exception as e:
               logger.warning(f"Failed to load profile {profile_file}: {e}")

       return profiles

   # Check if profile exists
   all_profiles = list_all_profiles()
   profile_ids = [p['id'] for p in all_profiles]

   if profile_id not in profile_ids:
       print(f"✗ Profile {profile_id} not found")
       print(f"Available profiles: {len(profile_ids)}")
       for p in all_profiles:
           print(f"  - {p['id']}: {p['name']}")
   ```

3. **Create Profile if Missing**:
   ```python
   from auto_voice.inference.voice_cloner import VoiceCloner

   def get_or_create_profile(profile_id: str, audio_files: list = None):
       """Get existing profile or create new one."""
       store = VoiceProfileStore()

       try:
           profile = store.get(profile_id)
           logger.info(f"Profile found: {profile_id}")
           return profile
       except (ProfileNotFoundError, FileNotFoundError):
           if audio_files is None:
               raise ValueError(f"Profile {profile_id} not found and no audio provided")

           # Create new profile
           logger.info(f"Creating new profile: {profile_id}")
           cloner = VoiceCloner()
           profile = cloner.create_profile(
               name=profile_id,
               audio_files=audio_files
           )
           return profile
   ```

4. **Check Database and File Storage Consistency**:
   ```python
   from pathlib import Path
   import json

   def validate_profile_storage():
       """Check for orphaned JSON or missing embeddings."""
       profiles_dir = Path('data/voice_profiles')

       json_files = set(f.stem for f in profiles_dir.glob('*.json'))
       npy_files = set(f.stem for f in profiles_dir.glob('*.npy'))

       # Find orphaned files
       orphaned_json = json_files - npy_files
       orphaned_npy = npy_files - json_files

       if orphaned_json:
           print(f"⚠ {len(orphaned_json)} profiles missing embeddings:")
           for profile_id in orphaned_json:
               print(f"  - {profile_id} (missing .npy)")

       if orphaned_npy:
           print(f"⚠ {len(orphaned_npy)} orphaned embeddings:")
           for profile_id in orphaned_npy:
               print(f"  - {profile_id} (missing .json)")

       # Validate each profile
       for profile_id in json_files & npy_files:
           json_path = profiles_dir / f"{profile_id}.json"
           npy_path = profiles_dir / f"{profile_id}.npy"

           try:
               with open(json_path) as f:
                   profile = json.load(f)
               import numpy as np
               embedding = np.load(npy_path)

               print(f"✓ {profile_id}: {profile['name']}")
           except Exception as e:
               print(f"✗ {profile_id}: {e}")

   # Run validation
   validate_profile_storage()
   ```

5. **Restore Deleted Profile from Backup**:
   ```bash
   # Check backups directory
   ls -lh data/backups/voice_profiles/

   # Restore specific profile
   PROFILE_ID="7da05140-e5a7-4e89-b2c3-8f6d9a1c2b3e"
   cp data/backups/voice_profiles/${PROFILE_ID}.json data/voice_profiles/
   cp data/backups/voice_profiles/${PROFILE_ID}.npy data/voice_profiles/

   # Verify restoration
   python -c "
   from auto_voice.storage.voice_profiles import VoiceProfileStore
   store = VoiceProfileStore()
   profile = store.get('${PROFILE_ID}')
   print(f'✓ Restored: {profile[\"name\"]}')
   "
   ```

**Diagnostic**:
```bash
# Check profile storage integrity
python -c "
from pathlib import Path
import json

profiles_dir = Path('data/voice_profiles')

json_count = len(list(profiles_dir.glob('*.json')))
npy_count = len(list(profiles_dir.glob('*.npy')))

print(f'JSON profiles: {json_count}')
print(f'NPY embeddings: {npy_count}')

if json_count != npy_count:
    print('⚠ WARNING: Profile count mismatch')
else:
    print('✓ Storage consistent')
"
```

**Prevention**:
- Always validate profile existence before operations
- Implement soft-delete instead of hard-delete
- Enable automatic backups for production
- Use database constraints for referential integrity
- Log all profile access attempts

---

### Transaction rollback issues

**Error Message**:
```
sqlalchemy.exc.PendingRollbackError: This Session's transaction has been rolled back due to a previous exception
sqlalchemy.exc.InvalidRequestError: This session is in 'inactive' state
RuntimeError: Transaction already rolled back
```

**Cause**: Previous database error caused transaction rollback, but session is still being used without proper cleanup.

**Solutions**:

1. **Proper Transaction Handling**:
   ```python
   from sqlalchemy.exc import SQLAlchemyError

   def update_profile_safe(profile_id: str, updates: dict):
       """Update profile with proper transaction handling."""
       session = Session()
       try:
           # Start explicit transaction
           with session.begin():
               profile = session.query(VoiceProfile).get(profile_id)
               if not profile:
                   raise ValueError(f"Profile not found: {profile_id}")

               # Apply updates
               for key, value in updates.items():
                   setattr(profile, key, value)

               # Commit happens automatically at end of 'with' block
           return True

       except SQLAlchemyError as e:
           # Rollback happens automatically
           logger.error(f"Failed to update profile: {e}")
           raise
       finally:
           session.close()
   ```

2. **Reset Session After Rollback**:
   ```python
   def execute_with_retry(session, operation, max_retries=3):
       """Execute database operation with automatic retry after rollback."""
       for attempt in range(max_retries):
           try:
               result = operation(session)
               session.commit()
               return result

           except SQLAlchemyError as e:
               session.rollback()
               logger.warning(f"Attempt {attempt + 1} failed: {e}")

               if attempt == max_retries - 1:
                   raise

               # Create new session for retry
               session.close()
               session = Session()

   # Usage
   def add_training_sample(session):
       sample = TrainingSample(profile_id=profile_id, audio_path=path)
       session.add(sample)
       return sample

   execute_with_retry(session, add_training_sample)
   ```

3. **Use Savepoints for Nested Transactions**:
   ```python
   from sqlalchemy.exc import IntegrityError

   def batch_insert_with_savepoints(records: list):
       """Insert records with savepoints to isolate failures."""
       session = Session()
       successful = 0
       failed = 0

       try:
           for record in records:
               # Create savepoint before each insert
               savepoint = session.begin_nested()

               try:
                   session.add(record)
                   session.flush()
                   savepoint.commit()
                   successful += 1

               except IntegrityError as e:
                   # Rollback to savepoint (not entire transaction)
                   savepoint.rollback()
                   logger.warning(f"Failed to insert {record}: {e}")
                   failed += 1

           # Commit all successful inserts
           session.commit()
           logger.info(f"Batch insert: {successful} success, {failed} failed")

       except Exception as e:
           session.rollback()
           raise
       finally:
           session.close()
   ```

4. **Check Session State Before Operations**:
   ```python
   def safe_query(session, model, filters):
       """Execute query with session state check."""
       # Check if session is in valid state
       if not session.is_active:
           logger.warning("Session inactive, creating new session")
           session.close()
           session = Session()

       try:
           result = session.query(model).filter_by(**filters).all()
           return result
       except Exception as e:
           logger.error(f"Query failed: {e}")
           session.rollback()
           raise
   ```

5. **Use Two-Phase Commit for Critical Operations**:
   ```python
   def update_profile_with_files(profile_id: str, updates: dict, files: list):
       """Update database and file system atomically."""
       session = Session()
       temp_files = []

       try:
           # Phase 1: Prepare database changes
           with session.begin():
               profile = session.query(VoiceProfile).get(profile_id)
               for key, value in updates.items():
                   setattr(profile, key, value)
               session.flush()  # Validate but don't commit

               # Phase 2: Write files
               for file_data in files:
                   temp_path = f"/tmp/{uuid.uuid4()}.tmp"
                   with open(temp_path, 'wb') as f:
                       f.write(file_data)
                   temp_files.append(temp_path)

               # Both phases successful - commit
               session.commit()

               # Move temp files to final location
               for temp_path in temp_files:
                   final_path = temp_path.replace('/tmp/', 'data/samples/')
                   os.rename(temp_path, final_path)

       except Exception as e:
           # Rollback database
           session.rollback()

           # Clean up temp files
           for temp_path in temp_files:
               if os.path.exists(temp_path):
                   os.remove(temp_path)

           raise
       finally:
           session.close()
   ```

**Diagnostic**:
```python
# Check session state
def diagnose_session(session):
    """Print session diagnostic information."""
    print(f"Session active: {session.is_active}")
    print(f"Session dirty: {len(session.dirty)}")
    print(f"Session new: {len(session.new)}")
    print(f"Session deleted: {len(session.deleted)}")
    print(f"Session in transaction: {session.in_transaction()}")

    if not session.is_active:
        print("⚠ WARNING: Session is inactive")
    if session.dirty:
        print(f"⚠ WARNING: {len(session.dirty)} uncommitted changes")
```

**Prevention**:
- Use context managers for automatic rollback
- Enable session pooling with proper timeout
- Set `autoflush=False` to control when flushes occur
- Add retry logic for transient database errors
- Monitor long-running transactions

---

### Database connection failures

**Error Message**:
```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file
sqlalchemy.exc.OperationalError: database is locked
OperationalError: no such table: voice_profiles
```

**Cause**: Database file missing, locked, or schema not initialized. Common with SQLite in multi-process environments.

**Solutions**:

1. **Initialize Database Schema**:
   ```python
   from sqlalchemy import create_engine
   from sqlalchemy.orm import declarative_base, sessionmaker

   Base = declarative_base()

   def init_database(db_url: str = 'sqlite:///data/autovoice.db'):
       """Initialize database schema."""
       # Create engine
       engine = create_engine(db_url, echo=False)

       # Create all tables
       Base.metadata.create_all(engine)

       # Verify tables created
       inspector = sqlalchemy.inspect(engine)
       tables = inspector.get_table_names()
       print(f"✓ Database initialized with {len(tables)} tables: {tables}")

       return engine

   # Run during setup
   engine = init_database()
   ```

2. **Fix Database Locked Errors** (SQLite):
   ```python
   from sqlalchemy import create_engine
   from sqlalchemy.pool import StaticPool

   # Use connection pooling to reduce lock contention
   engine = create_engine(
       'sqlite:///data/autovoice.db',
       connect_args={
           'timeout': 30,  # Wait up to 30s for lock
           'check_same_thread': False,  # Allow multi-threading
       },
       poolclass=StaticPool,  # Reuse connections
       echo=False
   )

   # Alternative: Use WAL mode for better concurrency
   with engine.connect() as conn:
       conn.execute("PRAGMA journal_mode=WAL")
       result = conn.execute("PRAGMA journal_mode").fetchone()
       print(f"Journal mode: {result[0]}")
   ```

3. **Verify Database File Permissions**:
   ```bash
   # Check if database file exists and is writable
   DB_FILE="data/autovoice.db"

   if [ -f "$DB_FILE" ]; then
       ls -lh "$DB_FILE"
       # Ensure write permissions
       chmod 644 "$DB_FILE"
       echo "✓ Database file permissions OK"
   else
       echo "✗ Database file not found: $DB_FILE"
       echo "Creating database directory..."
       mkdir -p data
       python -c "from auto_voice.database import init_database; init_database()"
   fi
   ```

4. **Handle Missing Tables**:
   ```python
   from sqlalchemy import inspect
   from sqlalchemy.exc import OperationalError

   def ensure_tables_exist(engine):
       """Check if required tables exist, create if missing."""
       inspector = inspect(engine)
       existing_tables = set(inspector.get_table_names())

       required_tables = {'voice_profiles', 'training_samples', 'conversion_jobs'}

       missing_tables = required_tables - existing_tables

       if missing_tables:
           logger.warning(f"Missing tables: {missing_tables}")
           logger.info("Creating missing tables...")
           Base.metadata.create_all(engine)
           logger.info("✓ Tables created")
       else:
           logger.info("✓ All required tables exist")

   # Use before operations
   ensure_tables_exist(engine)
   ```

5. **Use Connection Retry Logic**:
   ```python
   from sqlalchemy import create_engine, event
   from sqlalchemy.exc import OperationalError
   import time

   def create_engine_with_retry(db_url: str, max_retries: int = 5):
       """Create engine with connection retry."""
       for attempt in range(max_retries):
           try:
               engine = create_engine(db_url)
               # Test connection
               with engine.connect() as conn:
                   conn.execute("SELECT 1")
               logger.info("✓ Database connection successful")
               return engine

           except OperationalError as e:
               logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)  # Exponential backoff
               else:
                   logger.error("Failed to connect to database after retries")
                   raise

   engine = create_engine_with_retry('sqlite:///data/autovoice.db')
   ```

**Diagnostic**:
```bash
# Test database connection
python -c "
import os
from sqlalchemy import create_engine

db_file = 'data/autovoice.db'

# Check file exists
if not os.path.exists(db_file):
    print(f'✗ Database file not found: {db_file}')
    exit(1)

# Check permissions
if not os.access(db_file, os.R_OK | os.W_OK):
    print(f'✗ Database file not readable/writable')
    exit(1)

# Test connection
engine = create_engine(f'sqlite:///{db_file}')
try:
    with engine.connect() as conn:
        result = conn.execute('SELECT 1').fetchone()
    print('✓ Database connection OK')
except Exception as e:
    print(f'✗ Connection failed: {e}')
    exit(1)
"

# Check SQLite journal mode
sqlite3 data/autovoice.db "PRAGMA journal_mode;"

# Check for locks
lsof data/autovoice.db 2>/dev/null || echo "No processes holding database lock"
```

**Prevention**:
- Use PostgreSQL for production (better concurrency than SQLite)
- Enable WAL mode for SQLite: `PRAGMA journal_mode=WAL`
- Set appropriate connection timeouts
- Implement connection pooling
- Close sessions properly in `finally` blocks
- Run schema migrations during deployment

---

## API and Validation Errors

### Invalid adapter_type parameter

**Error Message**:
```
ValueError: Invalid adapter_type: <value>
ValueError: Invalid value for adapter_type
TypeError: adapter_type must be one of: ['lora', 'full', 'none']
```

**Cause**: API request contains invalid or unsupported adapter type value. AutoVoice supports specific adapter types for voice conversion.

**Solutions**:

1. **Use Valid Adapter Types**:
   ```python
   # Valid adapter types
   VALID_ADAPTER_TYPES = ['lora', 'full', 'none', 'auto']

   # ✓ CORRECT usage
   response = requests.post('/api/v1/convert/song', json={
       'audio_file': 'input.wav',
       'profile_id': '7da05140-...',
       'adapter_type': 'lora',  # Valid
   })

   # ✗ WRONG usage
   response = requests.post('/api/v1/convert/song', json={
       'adapter_type': 'LoRA',  # Case-sensitive, will fail
   })

   response = requests.post('/api/v1/convert/song', json={
       'adapter_type': 'custom',  # Not supported, will fail
   })
   ```

2. **Validate Input Before API Call**:
   ```python
   def validate_conversion_request(data: dict) -> dict:
       """Validate conversion request parameters."""
       valid_adapter_types = {'lora', 'full', 'none', 'auto'}

       adapter_type = data.get('adapter_type', 'auto')

       if adapter_type not in valid_adapter_types:
           raise ValueError(
               f"Invalid adapter_type: '{adapter_type}'. "
               f"Must be one of: {valid_adapter_types}"
           )

       # Normalize to lowercase
       data['adapter_type'] = adapter_type.lower()

       return data

   # Use before API call
   request_data = {
       'audio_file': 'song.wav',
       'profile_id': profile_id,
       'adapter_type': user_input_adapter_type
   }

   try:
       validated = validate_conversion_request(request_data)
       response = requests.post('/api/v1/convert/song', json=validated)
   except ValueError as e:
       print(f"Validation error: {e}")
   ```

3. **Check API Documentation**:
   ```python
   # Adapter type meanings:
   # - 'lora': Use LoRA adapter (fast, high quality, requires training)
   # - 'full': Use full model fine-tuning (best quality, slower)
   # - 'none': Use base model without adaptation (fastest, lower quality)
   # - 'auto': Automatically select best available (default)

   # Example: Use auto mode (recommended)
   response = requests.post('/api/v1/convert/song', json={
       'audio_file': 'input.wav',
       'profile_id': profile_id,
       'adapter_type': 'auto',  # Will use LoRA if available, else base model
   })
   ```

4. **Handle Missing LoRA Gracefully**:
   ```python
   def convert_with_fallback(audio_file: str, profile_id: str):
       """Convert audio with automatic fallback if LoRA not available."""
       # Try LoRA first
       try:
           response = requests.post('/api/v1/convert/song', json={
               'audio_file': audio_file,
               'profile_id': profile_id,
               'adapter_type': 'lora',
           })

           if response.status_code == 200:
               return response.json()
           elif response.status_code == 404:
               logger.warning("LoRA not found, falling back to base model")

       except Exception as e:
           logger.warning(f"LoRA conversion failed: {e}")

       # Fallback to base model
       response = requests.post('/api/v1/convert/song', json={
           'audio_file': audio_file,
           'profile_id': profile_id,
           'adapter_type': 'none',
       })

       return response.json()
   ```

5. **Use TypedDict for Request Validation**:
   ```python
   from typing import TypedDict, Literal

   class ConversionRequest(TypedDict):
       audio_file: str
       profile_id: str
       adapter_type: Literal['lora', 'full', 'none', 'auto']
       pitch_shift: int  # Optional
       formant_shift: float  # Optional

   def create_conversion_request(
       audio_file: str,
       profile_id: str,
       adapter_type: Literal['lora', 'full', 'none', 'auto'] = 'auto'
   ) -> ConversionRequest:
       """Type-safe conversion request builder."""
       return {
           'audio_file': audio_file,
           'profile_id': profile_id,
           'adapter_type': adapter_type,
       }

   # Type checker will catch invalid adapter types at development time
   request = create_conversion_request(
       'song.wav',
       profile_id,
       adapter_type='lora'  # ✓ Valid
       # adapter_type='invalid'  # ✗ Type error at development time
   )
   ```

**Diagnostic**:
```bash
# Test API parameter validation
python -c "
import requests

# Test valid adapter type
response = requests.post('http://localhost:5000/api/v1/convert/song', json={
    'audio_file': 'test.wav',
    'profile_id': 'test-id',
    'adapter_type': 'lora'
})

print(f'Status: {response.status_code}')
if response.status_code != 200:
    print(f'Error: {response.json()}')

# Test invalid adapter type
response = requests.post('http://localhost:5000/api/v1/convert/song', json={
    'adapter_type': 'invalid'
})

print(f'Invalid adapter response: {response.status_code}')
print(f'Error message: {response.json().get(\"error\", \"N/A\")}')
"
```

**Valid Adapter Types**:
- **lora**: LoRA adapter (recommended, requires training)
- **full**: Full model fine-tuning (highest quality, slowest)
- **none**: Base model without adaptation (fastest)
- **auto**: Automatic selection (default, uses LoRA if available)

**Prevention**:
- Use Pydantic models for request validation
- Add API parameter documentation
- Return clear error messages with valid options
- Implement client-side validation in frontend
- Use TypedDict or enums for type safety

---

### Type errors in request validation

**Error Message**:
```
TypeError: Expected str, got int
TypeError: 'NoneType' object is not subscriptable
ValueError: invalid literal for int() with base 10
KeyError: 'required_field'
```

**Cause**: API request parameters have incorrect types or missing required fields.

**Solutions**:

1. **Use Pydantic for Request Validation**:
   ```python
   from pydantic import BaseModel, Field, validator
   from typing import Optional, Literal

   class ConversionRequest(BaseModel):
       """Type-safe conversion request model."""
       audio_file: str = Field(..., min_length=1, description="Path to audio file")
       profile_id: str = Field(..., regex=r'^[a-f0-9-]{36}$', description="UUID of voice profile")
       adapter_type: Literal['lora', 'full', 'none', 'auto'] = 'auto'
       pitch_shift: Optional[int] = Field(0, ge=-12, le=12, description="Semitones")
       formant_shift: Optional[float] = Field(0.0, ge=-1.0, le=1.0)

       @validator('audio_file')
       def validate_audio_file(cls, v):
           if not v.endswith(('.wav', '.mp3', '.flac')):
               raise ValueError('Audio file must be WAV, MP3, or FLAC')
           return v

   # Use in Flask API
   from flask import request, jsonify

   @app.route('/api/v1/convert/song', methods=['POST'])
   def convert_song():
       try:
           # Validate request with Pydantic
           req_data = ConversionRequest(**request.json)

           # Now req_data is type-safe
           result = pipeline.convert(
               audio_file=req_data.audio_file,
               profile_id=req_data.profile_id,
               adapter_type=req_data.adapter_type,
               pitch_shift=req_data.pitch_shift
           )

           return jsonify(result), 200

       except ValueError as e:
           return jsonify({'error': str(e)}), 400
   ```

2. **Manual Type Validation**:
   ```python
   def validate_request_params(data: dict) -> dict:
       """Validate and coerce request parameters."""
       errors = []

       # Required string fields
       for field in ['audio_file', 'profile_id']:
           if field not in data:
               errors.append(f"Missing required field: {field}")
           elif not isinstance(data[field], str):
               errors.append(f"{field} must be a string, got {type(data[field]).__name__}")

       # Optional integer with range
       if 'pitch_shift' in data:
           try:
               pitch_shift = int(data['pitch_shift'])
               if not -12 <= pitch_shift <= 12:
                   errors.append("pitch_shift must be between -12 and 12")
               data['pitch_shift'] = pitch_shift
           except (ValueError, TypeError):
               errors.append(f"pitch_shift must be an integer, got {data['pitch_shift']}")

       # Optional float with range
       if 'formant_shift' in data:
           try:
               formant_shift = float(data['formant_shift'])
               if not -1.0 <= formant_shift <= 1.0:
                   errors.append("formant_shift must be between -1.0 and 1.0")
               data['formant_shift'] = formant_shift
           except (ValueError, TypeError):
               errors.append(f"formant_shift must be a float, got {data['formant_shift']}")

       if errors:
           raise ValueError("Validation errors: " + "; ".join(errors))

       return data

   # Use in API
   try:
       validated_data = validate_request_params(request.json or {})
   except ValueError as e:
       return jsonify({'error': str(e)}), 400
   ```

3. **Safe Dictionary Access**:
   ```python
   # ✗ WRONG - will raise KeyError if field missing
   audio_file = request_data['audio_file']

   # ✓ CORRECT - use .get() with default
   audio_file = request_data.get('audio_file')
   if not audio_file:
       return jsonify({'error': 'audio_file is required'}), 400

   # ✓ CORRECT - use .get() with type check
   pitch_shift = request_data.get('pitch_shift', 0)
   if not isinstance(pitch_shift, (int, float)):
       return jsonify({'error': 'pitch_shift must be a number'}), 400
   ```

4. **Type Coercion with Error Handling**:
   ```python
   def safe_int(value, default=None, min_val=None, max_val=None):
       """Safely convert value to int with range checking."""
       try:
           result = int(value)

           if min_val is not None and result < min_val:
               raise ValueError(f"Value {result} below minimum {min_val}")

           if max_val is not None and result > max_val:
               raise ValueError(f"Value {result} above maximum {max_val}")

           return result

       except (ValueError, TypeError) as e:
           if default is not None:
               return default
           raise ValueError(f"Invalid integer value: {value}")

   # Use in API
   try:
       pitch_shift = safe_int(
           request_data.get('pitch_shift'),
           default=0,
           min_val=-12,
           max_val=12
       )
   except ValueError as e:
       return jsonify({'error': str(e)}), 400
   ```

5. **Validate JSON Structure**:
   ```python
   from flask import request, jsonify

   @app.route('/api/v1/convert/song', methods=['POST'])
   def convert_song():
       # Check if request has JSON body
       if not request.is_json:
           return jsonify({'error': 'Content-Type must be application/json'}), 400

       # Check if JSON is valid dict
       data = request.json
       if not isinstance(data, dict):
           return jsonify({'error': 'Request body must be a JSON object'}), 400

       # Validate required fields
       required_fields = ['audio_file', 'profile_id']
       missing_fields = [f for f in required_fields if f not in data]

       if missing_fields:
           return jsonify({
               'error': f"Missing required fields: {', '.join(missing_fields)}"
           }), 400

       # Continue with validated data
       ...
   ```

**Diagnostic**:
```bash
# Test API with various invalid inputs
python -c "
import requests

base_url = 'http://localhost:5000/api/v1/convert/song'

# Test missing required field
response = requests.post(base_url, json={'audio_file': 'test.wav'})
print(f'Missing field: {response.status_code} - {response.json()}')

# Test wrong type
response = requests.post(base_url, json={
    'audio_file': 123,  # Should be string
    'profile_id': 'test'
})
print(f'Wrong type: {response.status_code} - {response.json()}')

# Test invalid range
response = requests.post(base_url, json={
    'audio_file': 'test.wav',
    'profile_id': 'test',
    'pitch_shift': 100  # Out of range
})
print(f'Invalid range: {response.status_code} - {response.json()}')

# Test valid request
response = requests.post(base_url, json={
    'audio_file': 'test.wav',
    'profile_id': '7da05140-e5a7-4e89-b2c3-8f6d9a1c2b3e',
    'adapter_type': 'lora',
    'pitch_shift': 2
})
print(f'Valid request: {response.status_code}')
"
```

**Prevention**:
- Use Pydantic, Marshmallow, or similar for schema validation
- Return 400 (Bad Request) with detailed error messages
- Document required fields and types in API docs
- Add request examples in documentation
- Use TypeScript on frontend for type safety
- Implement OpenAPI/Swagger schema validation

---

### Missing required parameters

**Error Message**:
```
KeyError: 'profile_id'
ValueError: Missing required parameter: audio_file
TypeError: Missing required argument: 'adapter_type'
```

**Cause**: API request missing required fields or arguments.

**Solutions**:

1. **Explicit Required Field Validation**:
   ```python
   from flask import request, jsonify

   REQUIRED_FIELDS = {
       '/api/v1/convert/song': ['audio_file', 'profile_id'],
       '/api/v1/voice/clone': ['name', 'audio_files'],
       '/api/v1/train/lora': ['profile_id', 'training_data'],
   }

   def validate_required_fields(endpoint: str, data: dict) -> tuple:
       """Validate required fields for endpoint."""
       required = REQUIRED_FIELDS.get(endpoint, [])
       missing = [field for field in required if field not in data or not data[field]]

       if missing:
           error_msg = f"Missing required fields: {', '.join(missing)}"
           return False, error_msg

       return True, None

   @app.route('/api/v1/convert/song', methods=['POST'])
   def convert_song():
       data = request.json or {}

       # Validate required fields
       valid, error = validate_required_fields('/api/v1/convert/song', data)
       if not valid:
           return jsonify({'error': error}), 400

       # Proceed with conversion
       ...
   ```

2. **Use Default Values for Optional Parameters**:
   ```python
   def convert_song_request(data: dict):
       """Extract and validate conversion request parameters."""
       # Required parameters (will raise KeyError if missing - caught below)
       try:
           audio_file = data['audio_file']
           profile_id = data['profile_id']
       except KeyError as e:
           raise ValueError(f"Missing required parameter: {e.args[0]}")

       # Optional parameters with defaults
       adapter_type = data.get('adapter_type', 'auto')
       pitch_shift = data.get('pitch_shift', 0)
       formant_shift = data.get('formant_shift', 0.0)
       output_format = data.get('output_format', 'wav')

       return {
           'audio_file': audio_file,
           'profile_id': profile_id,
           'adapter_type': adapter_type,
           'pitch_shift': pitch_shift,
           'formant_shift': formant_shift,
           'output_format': output_format,
       }

   # Use in API
   try:
       params = convert_song_request(request.json or {})
   except ValueError as e:
       return jsonify({'error': str(e)}), 400
   ```

3. **Provide Clear Error Messages with Examples**:
   ```python
   @app.route('/api/v1/convert/song', methods=['POST'])
   def convert_song():
       data = request.json or {}

       # Check for required fields with helpful error messages
       if 'audio_file' not in data:
           return jsonify({
               'error': 'Missing required parameter: audio_file',
               'example': {
                   'audio_file': 'path/to/song.wav',
                   'profile_id': '7da05140-e5a7-4e89-b2c3-8f6d9a1c2b3e',
                   'adapter_type': 'lora'  # optional
               }
           }), 400

       if 'profile_id' not in data:
           return jsonify({
               'error': 'Missing required parameter: profile_id',
               'hint': 'Use /api/v1/profiles to list available profiles'
           }), 400

       # Proceed with valid request
       ...
   ```

4. **Use Function Signatures with Type Hints**:
   ```python
   from typing import Optional

   def convert_audio(
       audio_file: str,
       profile_id: str,
       adapter_type: str = 'auto',
       pitch_shift: int = 0,
       formant_shift: float = 0.0
   ) -> dict:
       """
       Convert audio to target voice.

       Args:
           audio_file: Path to input audio file (required)
           profile_id: Voice profile UUID (required)
           adapter_type: Type of adapter ('lora', 'full', 'none', 'auto')
           pitch_shift: Pitch shift in semitones (-12 to +12)
           formant_shift: Formant shift (-1.0 to +1.0)

       Returns:
           Conversion result with output path

       Raises:
           ValueError: If required parameters missing or invalid
       """
       # Function signature enforces required parameters
       if not audio_file:
           raise ValueError("audio_file cannot be empty")

       if not profile_id:
           raise ValueError("profile_id cannot be empty")

       # Proceed with conversion
       ...

   # Use in API - Python will enforce required args
   try:
       result = convert_audio(
           audio_file=data['audio_file'],
           profile_id=data['profile_id'],
           # Optional params use defaults if not provided
       )
   except KeyError as e:
       return jsonify({'error': f'Missing required field: {e.args[0]}'}), 400
   ```

5. **Implement API Request Builder**:
   ```python
   class ConversionRequestBuilder:
       """Builder pattern for type-safe API requests."""

       def __init__(self):
           self._audio_file = None
           self._profile_id = None
           self._adapter_type = 'auto'
           self._pitch_shift = 0
           self._formant_shift = 0.0

       def with_audio_file(self, path: str):
           self._audio_file = path
           return self

       def with_profile(self, profile_id: str):
           self._profile_id = profile_id
           return self

       def with_adapter(self, adapter_type: str):
           self._adapter_type = adapter_type
           return self

       def with_pitch_shift(self, semitones: int):
           self._pitch_shift = semitones
           return self

       def build(self) -> dict:
           """Build request, validating required fields."""
           if not self._audio_file:
               raise ValueError("audio_file is required")

           if not self._profile_id:
               raise ValueError("profile_id is required")

           return {
               'audio_file': self._audio_file,
               'profile_id': self._profile_id,
               'adapter_type': self._adapter_type,
               'pitch_shift': self._pitch_shift,
               'formant_shift': self._formant_shift,
           }

   # Usage
   request_data = (ConversionRequestBuilder()
       .with_audio_file('song.wav')
       .with_profile(profile_id)
       .with_pitch_shift(2)
       .build())

   response = requests.post('/api/v1/convert/song', json=request_data)
   ```

**Diagnostic**:
```bash
# Test required parameter validation
python -c "
import requests

base_url = 'http://localhost:5000/api/v1/convert/song'

# Test completely empty request
response = requests.post(base_url, json={})
print(f'Empty request: {response.status_code}')
print(f'Error: {response.json().get(\"error\")}')

# Test with only one required field
response = requests.post(base_url, json={'audio_file': 'test.wav'})
print(f'Missing profile_id: {response.status_code}')
print(f'Error: {response.json().get(\"error\")}')

# Test with all required fields
response = requests.post(base_url, json={
    'audio_file': 'test.wav',
    'profile_id': 'test-id'
})
print(f'Valid request: {response.status_code}')
"
```

**Prevention**:
- Use schema validation libraries (Pydantic, Marshmallow)
- Return 400 (Bad Request) with specific field names
- Document required vs optional parameters in API docs
- Provide request examples in documentation
- Use OpenAPI/Swagger for automatic validation
- Add request logging to debug missing parameters

---

### Authentication failures

**Error Message**:
```
401 Unauthorized: Missing authentication token
403 Forbidden: Invalid API key
401 Unauthorized: Token expired
```

**Cause**: Missing, invalid, or expired authentication credentials.

**Solutions**:

1. **Provide Valid API Key**:
   ```python
   import requests

   # Set API key in headers
   headers = {
       'Authorization': 'Bearer YOUR_API_KEY_HERE',
       'Content-Type': 'application/json'
   }

   response = requests.post(
       'http://localhost:5000/api/v1/convert/song',
       headers=headers,
       json={'audio_file': 'song.wav', 'profile_id': profile_id}
   )

   if response.status_code == 401:
       print("Authentication failed - check API key")
   elif response.status_code == 200:
       print("Request successful")
   ```

2. **Configure Authentication**:
   ```python
   # config/api_config.yaml
   api:
     authentication:
       enabled: true
       type: bearer  # or 'api_key', 'oauth'
       api_keys:
         - key: "your-secret-key-here"
           name: "production"
         - key: "dev-key-12345"
           name: "development"

   # Load and use configuration
   import yaml

   with open('config/api_config.yaml') as f:
       config = yaml.safe_load(f)

   if config['api']['authentication']['enabled']:
       valid_keys = {k['key'] for k in config['api']['authentication']['api_keys']}
   ```

3. **Implement Token Refresh**:
   ```python
   import requests
   from datetime import datetime, timedelta

   class APIClient:
       """API client with automatic token refresh."""

       def __init__(self, base_url: str, api_key: str):
           self.base_url = base_url
           self.api_key = api_key
           self.token = None
           self.token_expires = None

       def get_token(self):
           """Get or refresh access token."""
           if self.token and self.token_expires > datetime.now():
               return self.token

           # Request new token
           response = requests.post(
               f'{self.base_url}/api/v1/auth/token',
               json={'api_key': self.api_key}
           )

           if response.status_code == 200:
               data = response.json()
               self.token = data['access_token']
               self.token_expires = datetime.now() + timedelta(seconds=data['expires_in'])
               return self.token
           else:
               raise ValueError("Failed to get access token")

       def convert_song(self, audio_file: str, profile_id: str):
           """Convert song with automatic token refresh."""
           token = self.get_token()

           response = requests.post(
               f'{self.base_url}/api/v1/convert/song',
               headers={'Authorization': f'Bearer {token}'},
               json={'audio_file': audio_file, 'profile_id': profile_id}
           )

           if response.status_code == 401:
               # Token might have expired, refresh and retry
               self.token = None
               token = self.get_token()

               response = requests.post(
                   f'{self.base_url}/api/v1/convert/song',
                   headers={'Authorization': f'Bearer {token}'},
                   json={'audio_file': audio_file, 'profile_id': profile_id}
               )

           return response.json()

   # Usage
   client = APIClient('http://localhost:5000', api_key='your-key')
   result = client.convert_song('song.wav', profile_id)
   ```

4. **Handle Auth Errors Gracefully**:
   ```python
   def api_request_with_auth(url: str, api_key: str, data: dict):
       """Make API request with error handling."""
       headers = {'Authorization': f'Bearer {api_key}'}

       response = requests.post(url, headers=headers, json=data)

       if response.status_code == 401:
           error_data = response.json()
           if 'Token expired' in error_data.get('error', ''):
               raise AuthenticationError("API token expired - please refresh")
           else:
               raise AuthenticationError("Invalid API key")

       elif response.status_code == 403:
           raise AuthenticationError("API key does not have permission for this operation")

       elif response.status_code == 200:
           return response.json()

       else:
           raise APIError(f"Request failed: {response.status_code}")

   # Usage
   try:
       result = api_request_with_auth(
           'http://localhost:5000/api/v1/convert/song',
           api_key='your-key',
           data={'audio_file': 'song.wav', 'profile_id': profile_id}
       )
   except AuthenticationError as e:
       print(f"Auth error: {e}")
       # Prompt user to re-authenticate
   ```

5. **For Local Development (Disable Auth)**:
   ```python
   # config/gpu_config.yaml (development)
   api:
     authentication:
       enabled: false  # Disable for local development

   # Or use environment variable
   import os

   AUTH_ENABLED = os.getenv('AUTOVOICE_AUTH_ENABLED', 'false').lower() == 'true'

   if not AUTH_ENABLED:
       # Skip authentication for local development
       print("⚠ WARNING: Authentication disabled (development mode)")
   ```

**Diagnostic**:
```bash
# Test authentication
python -c "
import requests

base_url = 'http://localhost:5000'

# Test without authentication
response = requests.get(f'{base_url}/api/v1/profiles')
print(f'No auth: {response.status_code}')

# Test with invalid API key
headers = {'Authorization': 'Bearer invalid-key'}
response = requests.get(f'{base_url}/api/v1/profiles', headers=headers)
print(f'Invalid key: {response.status_code} - {response.json()}')

# Test with valid API key (replace with your key)
headers = {'Authorization': 'Bearer YOUR_API_KEY'}
response = requests.get(f'{base_url}/api/v1/profiles', headers=headers)
print(f'Valid key: {response.status_code}')
"

# Check if authentication is enabled
grep -A 5 "authentication:" config/gpu_config.yaml
```

**Prevention**:
- Store API keys in environment variables, not code
- Use secure token storage (keyring, secrets manager)
- Implement token rotation policy
- Add rate limiting to prevent brute force
- Log authentication failures for security monitoring
- Use HTTPS in production to protect tokens

---

## Dependency and Environment Errors

### PyWorld installation fails or crashes (ARM64/Python 3.13)

**Error Message**:
```
ImportError: undefined symbol: __aarch64_ldadd4_relax
```
or
```
ModuleNotFoundError: No module named 'pyworld'
```

**Cause**: PyWorld binary incompatibility with Python 3.13 on ARM64 (Jetson Thor) or missing installation.

**Solutions**:

1. **Build PyWorld from Source** (recommended for ARM64):
   ```bash
   # Install build dependencies
   sudo apt-get install -y build-essential cmake

   # Clone and build PyWorld
   git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder pyworld
   cd pyworld
   pip install -e .
   ```

2. **Use Compatible Python Version**:
   ```bash
   # Python 3.10 or 3.11 recommended for ARM64
   conda create -n autovoice python=3.11 -y
   conda activate autovoice
   pip install pyworld
   ```

3. **Skip PyWorld-dependent Features** (temporary workaround):
   ```python
   # Use alternative pitch extraction
   from auto_voice.inference import RealtimePipeline

   pipeline = RealtimePipeline(
       pitch_extractor='harvest',  # Instead of 'dio' (PyWorld)
       use_hq_svc=False  # Disable HQ-SVC which requires PyWorld
   )
   ```

4. **Install Pre-built Wheel** (if available):
   ```bash
   # Check for ARM64 wheels
   pip install pyworld --find-links https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/releases
   ```

**Diagnostic**:
```bash
# Test PyWorld import
python -c "import pyworld; print(f'PyWorld version: {pyworld.__version__}')"

# Check binary compatibility
python -c "import pyworld; pyworld.dio(np.zeros(16000), 16000)"

# Verify Python version
python --version
```

**Prevention**:
- Use Python 3.10 or 3.11 on ARM64 platforms
- Build PyWorld from source during initial setup
- Add PyWorld check to test suite: `pytest tests/test_hq_svc_wrapper.py -v`

---

### Missing local-attention module

**Error Message**:
```
ModuleNotFoundError: No module named 'local_attention'
```

**Cause**: `local-attention` dependency not installed. Required for HQ-SVC and attention-based models.

**Solutions**:

1. **Install local-attention**:
   ```bash
   pip install local-attention==1.11.2
   ```

2. **Install with hyper-connections** (if also needed):
   ```bash
   pip install local-attention==1.11.2 hyper-connections==0.4.7
   ```

3. **Verify Installation**:
   ```python
   import local_attention
   print(f"local-attention version: {local_attention.__version__}")
   ```

4. **Install from requirements.txt**:
   ```bash
   # Full dependency installation
   pip install -r requirements.txt
   ```

**Diagnostic**:
```bash
# Check if module is installed
pip show local-attention

# Test import
python -c "import local_attention; print('OK')"

# Verify requirements
grep -i "local-attention" requirements.txt
```

**Prevention**:
- Run `pip install -r requirements.txt` during setup
- Add dependency check to startup scripts
- Enable test coverage for attention modules

---

### sounddevice not found (audio I/O errors)

**Error Message**:
```
ModuleNotFoundError: No module named 'sounddevice'
```
or
```
OSError: PortAudio library not found
```

**Cause**: Missing `sounddevice` module or underlying PortAudio library for real-time audio I/O.

**Solutions**:

1. **Install sounddevice with PortAudio**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y portaudio19-dev python3-pyaudio
   pip install sounddevice

   # macOS
   brew install portaudio
   pip install sounddevice
   ```

2. **Verify Audio Devices**:
   ```python
   import sounddevice as sd
   print(sd.query_devices())  # List available devices
   ```

3. **Set Default Device** (if multiple devices):
   ```python
   import sounddevice as sd
   sd.default.device = 'USB Audio Device'  # Set preferred device
   sd.default.samplerate = 44100
   sd.default.channels = 1
   ```

4. **Use Alternative Backend** (fallback):
   ```python
   # Use PyAudio instead
   import pyaudio
   # or use simpleaudio for playback only
   import simpleaudio as sa
   ```

**Diagnostic**:
```bash
# Test sounddevice
python -c "import sounddevice as sd; print(sd.query_devices())"

# Check PortAudio library
ldconfig -p | grep portaudio

# List audio devices
arecord -l  # Input devices
aplay -l    # Output devices
```

**Prevention**:
- Install PortAudio before pip install
- Test audio devices after environment setup
- Add device check to streaming pipeline initialization

---

### Demucs initialization failed (vocal separator)

**Error Message**:
```
RuntimeError: Demucs initialization failed
```
or
```
FileNotFoundError: Demucs model checkpoint not found
```

**Cause**: Missing Demucs model files or CUDA compatibility issues during vocal separator initialization.

**Solutions**:

1. **Download Demucs Models**:
   ```bash
   # Download pretrained Demucs models
   python -m demucs.download htdemucs
   python -m demucs.download htdemucs_ft
   ```

2. **Verify Model Path**:
   ```python
   from auto_voice.audio import VocalSeparator

   separator = VocalSeparator(
       model_name='htdemucs',
       device='cuda',
       cache_dir='./models/demucs'  # Explicit path
   )
   ```

3. **Use CPU if CUDA Fails**:
   ```python
   # Fallback to CPU
   separator = VocalSeparator(model_name='htdemucs', device='cpu')
   ```

4. **Clear Model Cache**:
   ```bash
   # Remove corrupted cache
   rm -rf ~/.cache/torch/hub/demucs

   # Re-download models
   python -m demucs.download htdemucs
   ```

5. **Manual Model Installation**:
   ```bash
   # Download specific model
   wget https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/htdemucs.th
   mkdir -p models/pretrained/demucs
   mv htdemucs.th models/pretrained/demucs/
   ```

**Diagnostic**:
```bash
# Test Demucs import
python -c "import demucs; print(f'Demucs installed: {demucs.__version__}')"

# List cached models
ls -lh ~/.cache/torch/hub/demucs/

# Test separator initialization
python -c "from auto_voice.audio import VocalSeparator; VocalSeparator(device='cpu')"
```

**Prevention**:
- Run `scripts/download_pretrained_models.py` during setup
- Verify model files exist before initialization
- Add Demucs tests: `pytest tests/test_vocal_separator.py -v`

---

### TensorRT not available (optional optimization)

**Error Message**:
```
ModuleNotFoundError: No module named 'tensorrt'
```
or
```
ImportError: cannot import name 'tensorrt_engine' from 'auto_voice.export'
```

**Cause**: TensorRT is an optional dependency for inference optimization. Safe to skip if not needed.

**Solutions**:

1. **Install TensorRT** (Jetson Thor has native support):
   ```bash
   # On Jetson platforms (pre-installed)
   export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

   # Verify TensorRT
   python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
   ```

2. **Install via pip** (other platforms):
   ```bash
   # Install TensorRT Python bindings
   pip install tensorrt

   # Or use NVIDIA NGC container
   docker pull nvcr.io/nvidia/tensorrt:23.11-py3
   ```

3. **Skip TensorRT Features** (optional):
   ```python
   # Use regular PyTorch inference
   from auto_voice.inference import RealtimePipeline

   pipeline = RealtimePipeline(
       use_tensorrt=False  # Skip TRT optimization
   )
   ```

4. **Conditional Import** (already handled):
   ```python
   # AutoVoice automatically handles missing TensorRT
   try:
       from auto_voice.export import TensorRTEngine
   except ImportError:
       print("TensorRT not available, using PyTorch backend")
   ```

**Diagnostic**:
```bash
# Check TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"

# Verify library path
echo $LD_LIBRARY_PATH | grep tensorrt

# Test TRT export (if installed)
python scripts/verify_bindings.py --tensorrt
```

**Prevention**:
- TensorRT is optional for most users
- Install only if targeting optimized inference (< 20ms latency)
- Tests automatically skip if TensorRT unavailable: `@pytest.mark.skipif(not has_tensorrt)`

---

### Python version incompatibility

**Error Message**:
```
RuntimeError: Python 3.13 not supported
```
or
```
SyntaxError: invalid syntax (f-strings, match statements)
```

**Cause**: AutoVoice requires Python 3.10, 3.11, or 3.12. Python 3.13 has compatibility issues with some dependencies.

**Solutions**:

1. **Use Recommended Python Version**:
   ```bash
   # Create conda environment with Python 3.11
   conda create -n autovoice python=3.11 -y
   conda activate autovoice

   # Verify version
   python --version  # Should show Python 3.11.x
   ```

2. **Check Current Version**:
   ```bash
   python --version
   which python
   ```

3. **Install PyTorch with Matching Python**:
   ```bash
   # CRITICAL: Install PyTorch FIRST with correct Python version
   pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

   # Then install AutoVoice dependencies
   pip install -r requirements.txt
   ```

4. **Downgrade if Necessary**:
   ```bash
   # Remove existing environment
   conda deactivate
   conda remove -n autovoice --all

   # Recreate with Python 3.11
   conda create -n autovoice python=3.11 -y
   conda activate autovoice
   ```

**Diagnostic**:
```bash
# Check Python version
python --version

# Verify conda environment
conda env list
conda list python

# Test PyTorch compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}, Python: {torch.__config__.show().split()[0]}')"
```

**Prevention**:
- Always use conda environment with pinned Python version
- Add version check to setup scripts: `scripts/setup_pytorch_env.sh`
- Document version requirements in README

---

### Missing CUDA toolkit (compilation errors)

**Error Message**:
```
RuntimeError: CUDA toolkit not found
```
or
```
error: command 'nvcc' failed: No such file or directory
```

**Cause**: CUDA toolkit not installed or not in PATH. Required for building CUDA extensions.

**Solutions**:

1. **Install CUDA Toolkit**:
   ```bash
   # Ubuntu 22.04
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
   sudo sh cuda_12.1.0_530.30.02_linux.run

   # Add to PATH
   echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Verify CUDA Installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Use Pre-built Wheels** (skip compilation):
   ```bash
   # Install PyTorch with pre-built CUDA binaries
   pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

   # Skip building CUDA extensions (slower but works)
   SKIP_CUDA_BUILD=1 pip install -e .
   ```

4. **Set CUDA Environment Variables**:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.1
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

**Diagnostic**:
```bash
# Check CUDA toolkit
nvcc --version
which nvcc

# Verify CUDA libraries
ldconfig -p | grep cuda

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test CUDA extension build
python scripts/verify_bindings.py
```

**Prevention**:
- Install CUDA toolkit before AutoVoice setup
- Use `scripts/setup_pytorch_env.sh` for automated setup
- Add CUDA path to shell profile permanently

---

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
