# VoiceCloner Extensions Implementation Summary

## Overview

Extended `VoiceCloner` class in `/home/kp/autovoice/src/auto_voice/inference/voice_cloner.py` with three major features:
1. **SNR Validation** - Signal-to-Noise Ratio computation and validation
2. **Multi-Sample Profile Support** - Create profiles from multiple audio samples
3. **Profile Versioning** - Track profile changes over time

---

## 1. SNR Validation

### Implementation

#### Private Method: `_compute_snr(audio: np.ndarray) -> Optional[float]`
```python
def _compute_snr(self, audio: np.ndarray) -> Optional[float]:
    """Compute Signal-to-Noise Ratio (SNR) in dB

    SNR = 10 * log10(signal_power / noise_power)
    - Signal power: RMS of entire audio
    - Noise floor: Quietest 10% of frames
    """
```

**Features:**
- Frame-based analysis (2048 samples, 512 hop)
- Percentile-based noise floor estimation (10th percentile)
- Zero-division protection
- Returns `None` for too-short audio or computation failures

#### Enhanced `validate_audio()` Method
```python
# Check SNR if threshold is configured
min_snr_db = self.config.get('min_snr_db', None)
if min_snr_db is not None:
    snr_db = self._compute_snr(audio)
    if snr_db is not None and snr_db < min_snr_db:
        return False, f"Audio SNR too low: {snr_db:.1f} dB", 'snr_too_low'
```

#### Public Method: `get_audio_quality_report(audio, sample_rate) -> Dict[str, Any]`
```python
def get_audio_quality_report(self, audio, sample_rate) -> Dict[str, Any]:
    """Generate detailed audio quality diagnostic report

    Returns:
        - duration: Audio duration in seconds
        - sample_rate: Sample rate in Hz
        - rms: Root Mean Square amplitude
        - peak: Peak amplitude
        - snr_db: Signal-to-Noise Ratio in dB
        - dynamic_range_db: Dynamic range in dB
        - is_valid: Whether audio passes validation
        - validation_errors: List of validation error messages
    """
```

### Configuration (audio_config.yaml)
```yaml
voice_cloning:
  min_snr_db: 10.0  # Minimum SNR threshold (set to null to disable)
```

---

## 2. Multi-Sample Profile Support

### Implementation

#### Method: `create_voice_profile_from_multiple_samples()`
```python
def create_voice_profile_from_multiple_samples(
    self,
    audio_samples: List[Union[np.ndarray, str]],
    user_id: Optional[str] = None,
    sample_rate: Optional[int] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create voice profile by averaging multiple audio samples

    Features:
    - Validates each sample individually
    - Computes SNR for quality weighting
    - Averages embeddings (quality-weighted or simple)
    - Merges vocal ranges (min of mins, max of maxs)
    - Averages timbre features
    - Stores per-sample metadata
    """
```

**Embedding Averaging:**
- **Quality-weighted** (default): Embeddings weighted by SNR
  ```python
  weights = 10 ** (snr_db / 10.0)  # Convert dB to linear power
  averaged_embedding = np.average(embeddings, axis=0, weights=weights)
  ```
- **Simple average**: Unweighted mean of embeddings

**Vocal Range Merging:**
```python
merged_vocal_range = {
    'min_f0': min(vr['min_f0'] for vr in vocal_ranges),
    'max_f0': max(vr['max_f0'] for vr in vocal_ranges),
    'mean_f0': np.mean([vr['mean_f0'] for vr in vocal_ranges]),
    'range_semitones': max(vr['range_semitones'] for vr in vocal_ranges)
}
```

**Profile Structure:**
```python
{
    'profile_id': 'uuid-...',
    'user_id': 'user123',
    'embedding': np.ndarray,  # Averaged embedding
    'multi_sample_info': {
        'num_samples': 3,
        'sample_metadata': [
            {
                'index': 0,
                'filename': 'voice1.wav',
                'snr_db': 18.5,
                'original_sample_rate': 44100
            },
            ...
        ],
        'quality_weighted': True,
        'average_snr_db': 17.2
    }
}
```

#### Method: `add_sample_to_profile()`
```python
def add_sample_to_profile(
    self,
    profile_id: str,
    audio: Union[np.ndarray, str],
    sample_rate: Optional[int] = None,
    weight: Optional[float] = None
) -> Dict[str, Any]:
    """Add a new sample to existing voice profile

    Features:
    - Incremental averaging of embeddings
    - Updates vocal range and timbre features
    - Supports manual weight override
    - Increments profile version
    - Adds version history entry
    """
```

**Incremental Averaging:**
```python
# Quality-weighted
old_weight = 10 ** (old_avg_snr / 10.0) * current_num_samples
new_weight = 10 ** (new_snr / 10.0)
averaged = (old_embedding * old_weight + new_embedding * new_weight) / (old_weight + new_weight)

# Or simple
averaged = (old_embedding * n + new_embedding) / (n + 1)
```

### Configuration (audio_config.yaml)
```yaml
voice_cloning:
  # Multi-sample support
  multi_sample_quality_weighting: true  # Weight by SNR
  multi_sample_min_samples: 1  # Minimum samples
  multi_sample_max_samples: 10  # Maximum samples per profile
```

---

## 3. Profile Versioning

### Implementation

**Version Metadata in Profiles:**
```python
profile = {
    'schema_version': '1.0.0',  # Profile schema version
    'profile_version': 1,  # Increments on updates
    'version_history': [
        {
            'version': 1,
            'timestamp': '2025-10-27T12:34:56Z',
            'change_description': 'Initial profile creation',
            'audio_duration': 32.5
        }
    ]
}
```

**Automatic Versioning:**
- `create_voice_profile()`: Sets `profile_version = 1`
- `create_voice_profile_from_multiple_samples()`: Sets `profile_version = 1`
- `add_sample_to_profile()`: Increments `profile_version`, adds history entry

**Version History Entry (Sample Addition):**
```python
{
    'version': 2,
    'timestamp': '2025-10-27T12:45:30Z',
    'change_description': 'Added sample: voice2.wav',
    'num_samples': 2,
    'sample_added': 'voice2.wav'
}
```

#### Method: `get_profile_version_history(profile_id: str) -> List[Dict[str, Any]]`
```python
def get_profile_version_history(self, profile_id: str) -> List[Dict[str, Any]]:
    """Get version history for a voice profile

    Returns:
        List of version history entries with timestamps and descriptions
    """
```

### Configuration (audio_config.yaml)
```yaml
voice_cloning:
  # Profile versioning
  versioning_enabled: true  # Enable version tracking
  version_history_max_entries: 50  # Max history entries
  schema_version: '1.0.0'  # Profile schema version
```

---

## Configuration Summary

### Complete voice_cloning Section
```yaml
voice_cloning:
  # Audio duration constraints
  min_duration: 30.0
  max_duration: 60.0

  # Validation thresholds
  min_sample_rate: 8000
  max_sample_rate: 48000
  silence_threshold: 0.001
  min_snr_db: 10.0  # NEW: SNR validation

  # Multi-sample support (NEW)
  multi_sample_quality_weighting: true
  multi_sample_min_samples: 1
  multi_sample_max_samples: 10

  # Profile versioning (NEW)
  versioning_enabled: true
  version_history_max_entries: 50
  schema_version: '1.0.0'

  # Existing settings
  embedding_dim: 256
  storage_dir: '~/.cache/autovoice/voice_profiles/'
  cache_enabled: true
  cache_size: 100
  extract_vocal_range: true
  extract_timbre_features: true
  similarity_threshold: 0.75
  gpu_acceleration: true
  device: 'cuda'
```

---

## API Examples

### 1. SNR Validation

```python
from src.auto_voice.inference.voice_cloner import VoiceCloner

cloner = VoiceCloner(config={'min_snr_db': 15.0})

# Get quality report
report = cloner.get_audio_quality_report('voice.wav')
print(f"SNR: {report['snr_db']:.1f} dB")
print(f"Valid: {report['is_valid']}")

# Validate with SNR threshold
is_valid, error_msg, error_code = cloner.validate_audio('voice.wav')
if not is_valid:
    print(f"Validation failed: {error_msg}")
```

### 2. Multi-Sample Profile Creation

```python
# Create profile from multiple samples
samples = ['voice1.wav', 'voice2.wav', 'voice3.wav']
profile = cloner.create_voice_profile_from_multiple_samples(
    audio_samples=samples,
    user_id='user123',
    metadata={'recording_session': 'studio-2025-10-27'}
)

print(f"Profile ID: {profile['profile_id']}")
print(f"Samples: {profile['multi_sample_info']['num_samples']}")
print(f"Average SNR: {profile['multi_sample_info']['average_snr_db']:.1f} dB")
```

### 3. Adding Samples to Existing Profile

```python
# Add new sample to profile
updated_profile = cloner.add_sample_to_profile(
    profile_id='uuid-1234',
    audio='voice4.wav'
)

print(f"Version: {updated_profile['profile_version']}")
print(f"Total samples: {updated_profile['multi_sample_info']['num_samples']}")
```

### 4. Version History

```python
# Get version history
history = cloner.get_profile_version_history('uuid-1234')
for entry in history:
    print(f"v{entry['version']}: {entry['change_description']}")
    print(f"  Timestamp: {entry['timestamp']}")
```

---

## Testing

### Test Suite
Created comprehensive test suite: `/home/kp/autovoice/tests/test_voice_cloner_extensions.py`

**Test Classes:**
1. `TestSNRValidation` - SNR computation and validation
2. `TestAudioQualityReport` - Quality report generation
3. `TestMultiSampleSupport` - Multi-sample profile creation
4. `TestProfileVersioning` - Version tracking
5. `TestIntegration` - Full workflow integration

**Run Tests:**
```bash
pytest tests/test_voice_cloner_extensions.py -v
```

---

## Implementation Notes

### SNR Computation Details
- **Frame-based Analysis**: 2048-sample frames, 512-hop overlap
- **Signal Power**: RMS of entire audio signal
- **Noise Floor**: 10th percentile of frame RMS values
- **Formula**: `SNR_dB = 10 * log10(signal_power / noise_power)`
- **Edge Cases**: Returns `None` for too-short audio or zero-energy signals

### Quality Weighting Algorithm
- Convert SNR from dB to linear power scale: `weight = 10^(SNR_dB / 10)`
- Normalize weights: `weights = weights / sum(weights)`
- Weighted average: `embedding = sum(emb_i * weight_i)`

### Vocal Range Merging
- **Min F0**: Minimum of all sample minimums (widest lower bound)
- **Max F0**: Maximum of all sample maximums (widest upper bound)
- **Mean F0**: Average of all sample means
- **Range**: Maximum range across all samples

### Version History Management
- Auto-trim history when exceeding `version_history_max_entries`
- Keep most recent entries (FIFO)
- Store timestamps in ISO 8601 format with timezone (UTC)

---

## Performance Considerations

1. **SNR Computation**: O(n) where n = audio length
   - Frame iteration: ~1-2ms for 1-minute audio
   - Negligible overhead for validation

2. **Multi-Sample Processing**: O(k * n) where k = number of samples
   - Each sample processed independently
   - Embedding extraction is the bottleneck (~100-200ms/sample)

3. **Memory Usage**:
   - Each embedding: 256 floats × 4 bytes = 1 KB
   - Version history: ~200 bytes per entry
   - Total overhead: ~10-50 KB per profile

---

## Future Enhancements (Optional)

1. **Profile Rollback**: Restore to previous version
2. **Differential Updates**: Store only changes in version history
3. **Sample Removal**: Remove specific samples from profile
4. **Adaptive SNR Threshold**: Auto-adjust based on recording environment
5. **Batch Processing**: Process multiple samples in parallel
6. **Export/Import**: Serialize profiles with version history

---

## Files Modified

1. `/home/kp/autovoice/src/auto_voice/inference/voice_cloner.py`
   - Added `_compute_snr()` method
   - Added `get_audio_quality_report()` method
   - Added `create_voice_profile_from_multiple_samples()` method
   - Added `add_sample_to_profile()` method
   - Added `get_profile_version_history()` method
   - Updated `validate_audio()` with SNR checking
   - Updated `create_voice_profile()` with versioning
   - Updated `_load_config()` with new config keys

2. `/home/kp/autovoice/config/audio_config.yaml`
   - Added `min_snr_db` config
   - Added multi-sample config keys
   - Added versioning config keys

3. `/home/kp/autovoice/tests/test_voice_cloner_extensions.py` (NEW)
   - Comprehensive test suite for all new features

---

## Summary

Successfully implemented all required features:

✅ **SNR Validation**
- `_compute_snr()` private method
- Enhanced `validate_audio()` with SNR checking
- `get_audio_quality_report()` for detailed diagnostics

✅ **Multi-Sample Profile Support**
- `create_voice_profile_from_multiple_samples()` with quality weighting
- `add_sample_to_profile()` for incremental updates
- Embedding averaging (quality-weighted or simple)
- Vocal range merging and timbre feature averaging

✅ **Profile Versioning**
- `schema_version`, `profile_version`, `version_history` metadata
- Auto-increment on updates
- `get_profile_version_history()` method
- History entry trimming

✅ **Configuration**
- All config keys added to `audio_config.yaml`
- Backward-compatible defaults
- Per-feature enable/disable flags

✅ **Testing**
- Comprehensive test suite
- Unit tests for each feature
- Integration test for full workflow
