# VocalSeparator API Quick Reference

## New Features Summary

### 1. LRU Access Tracking
Automatic cache access time tracking for better cache eviction.

**Configuration**:
```python
config = {'lru_access_tracking': True}  # Default: True
```

**Behavior**:
- Writes `.timestamp` files on cache access
- Falls back to filesystem mtime if timestamp unavailable
- Transparent to users (automatic)

---

### 2. Batch Processing

#### Parallel Processing
```python
files = ['song1.mp3', 'song2.mp3', 'song3.mp3']

# With progress callback
def progress(completed, total, current_file):
    print(f"{completed}/{total}: {current_file}")

results = separator.separate_vocals_batch(
    audio_files=files,
    use_cache=True,
    max_workers=4,  # Default from config
    progress_callback=progress
)

# Results is list of tuples: [(vocals, instrumental, sr), ...]
for vocals, instrumental, sr in results:
    if vocals is not None:  # Check for failures
        # Process successful result
        pass
```

#### Sequential Processing
```python
results = separator.separate_vocals_batch_sequential(
    audio_files=files,
    use_cache=True,
    progress_callback=progress
)
```

**Configuration**:
```python
config = {'batch_max_workers': 4}  # Default: 4
```

---

### 3. Quality Presets

#### Set Preset
```python
# Fast mode (10x realtime on GPU)
separator.set_quality_preset('fast')

# Balanced mode (3.3x realtime) - default
separator.set_quality_preset('balanced')

# Quality mode (2x slower than realtime)
separator.set_quality_preset('quality')
```

#### Get Current Preset
```python
preset = separator.get_current_quality_preset()
# Returns: 'fast', 'balanced', 'quality', or 'custom'
```

#### Estimate Processing Time
```python
# For 3-minute audio with balanced preset
duration = 180.0  # seconds
estimated = separator.estimate_separation_time(duration, 'balanced')
print(f"Est: {estimated:.1f}s")  # ~54 seconds
```

**Configuration**:
```python
config = {'quality_preset': 'balanced'}  # Default: 'balanced'
```

---

## Preset Configurations

| Preset   | shifts | overlap | split | mixed_precision | Speed (GPU) | Use Case |
|----------|--------|---------|-------|-----------------|-------------|----------|
| fast     | 0      | 0.25    | False | True            | 10x RT      | Quick preview, testing |
| balanced | 1      | 0.25    | True  | True            | 3.3x RT     | Production default |
| quality  | 10     | 0.5     | True  | False           | 0.5x RT     | Final output, archival |

**RT = Realtime** (e.g., 10x RT means 10s audio processes in 1s)

---

## Complete Example

```python
from src.auto_voice.audio.source_separator import VocalSeparator

# Initialize with custom config
separator = VocalSeparator(
    device='cuda',
    config={
        'quality_preset': 'balanced',
        'batch_max_workers': 8,
        'lru_access_tracking': True,
        'cache_enabled': True,
        'cache_size_limit_gb': 10
    }
)

# Single file processing
vocals, instrumental, sr = separator.separate_vocals('song.mp3')

# Batch processing with progress
files = ['song1.mp3', 'song2.mp3', 'song3.mp3']

def show_progress(completed, total, current):
    pct = (completed / total) * 100
    print(f"[{pct:.0f}%] {current}")

results = separator.separate_vocals_batch(
    files,
    max_workers=8,
    progress_callback=show_progress
)

# Process results
for i, result in enumerate(results):
    if result is not None:
        vocals, instrumental, sr = result
        print(f"✓ {files[i]}: {sr} Hz")
    else:
        print(f"✗ {files[i]}: Failed")

# Switch preset for important file
separator.set_quality_preset('quality')
vocals, inst, sr = separator.separate_vocals('important.mp3')
```

---

## Error Handling

### Batch Processing Errors
```python
try:
    results = separator.separate_vocals_batch(files)
except ValueError as e:
    # Empty file list
    print(f"Invalid input: {e}")
except SeparationError as e:
    # All files failed
    print(f"Complete failure: {e}")

# Partial failures don't raise exceptions
# Check for None in results
for i, result in enumerate(results):
    if result is None:
        print(f"File {files[i]} failed to process")
```

### Quality Preset Errors
```python
try:
    separator.set_quality_preset('invalid_preset')
except ValueError as e:
    print(f"Invalid preset: {e}")
    # Suggests: 'fast', 'balanced', 'quality'
```

---

## Performance Tips

1. **Use batch processing** for multiple files (leverages parallelism)
2. **Enable caching** to avoid reprocessing (default: enabled)
3. **Choose appropriate preset**:
   - `fast`: Preview, testing, real-time applications
   - `balanced`: General production use
   - `quality`: Final deliverables, archival
4. **Adjust workers** based on GPU memory:
   - Small GPU (4-6GB): `max_workers=2`
   - Medium GPU (8-12GB): `max_workers=4`
   - Large GPU (16GB+): `max_workers=8`

---

## Configuration File Reference

```yaml
vocal_separation:
  # LRU cache tracking
  lru_access_tracking: true

  # Quality preset
  quality_preset: 'balanced'  # 'fast', 'balanced', 'quality'

  # Batch processing
  batch_max_workers: 4

  # Existing config (backward compatible)
  model: 'htdemucs'
  sample_rate: 44100
  cache_enabled: true
  cache_dir: '~/.cache/autovoice/separated/'
  cache_size_limit_gb: 10
  # ... other settings
```

---

## Migration Guide

### From Old API
```python
# Old way (still works)
vocals, inst, sr = separator.separate_vocals('song.mp3')

# New batch processing
files = ['song1.mp3', 'song2.mp3']
results = separator.separate_vocals_batch(files)

# New quality presets
separator.set_quality_preset('quality')
```

**No breaking changes** - all existing code continues to work!

---

## Advanced: Custom Presets

```python
# Create custom preset by modifying config
separator.config.update({
    'shifts': 5,
    'overlap': 0.4,
    'split': True,
    'mixed_precision': True
})

# Preset name will be 'custom'
print(separator.get_current_quality_preset())  # 'custom'

# Estimate uses custom parameters
time = separator.estimate_separation_time(180.0)
```

---

## Thread Safety

- All methods are thread-safe
- Batch processing uses `ThreadPoolExecutor`
- Cache operations protected by `self.lock`
- LRU tracking uses atomic file operations

---

## See Also

- **Full Documentation**: `/home/kp/autovoice/docs/vocal_separator_enhancements.md`
- **Source Code**: `/home/kp/autovoice/src/auto_voice/audio/source_separator.py`
- **Configuration**: `/home/kp/autovoice/config/audio_config.yaml`
