# VoiceCloner Extensions - Quick Reference

## New Methods

### 1. SNR Computation
```python
# Private method - compute SNR in dB
snr_db = cloner._compute_snr(audio)  # Returns Optional[float]
```

### 2. Audio Quality Report
```python
# Get detailed audio quality diagnostics
report = cloner.get_audio_quality_report('voice.wav')
# report = {
#     'duration': 32.5,
#     'sample_rate': 22050,
#     'rms': 0.123,
#     'peak': 0.987,
#     'snr_db': 18.5,
#     'dynamic_range_db': 45.2,
#     'is_valid': True,
#     'validation_errors': []
# }
```

### 3. Multi-Sample Profile Creation
```python
# Create profile from multiple samples
samples = ['voice1.wav', 'voice2.wav', 'voice3.wav']
profile = cloner.create_voice_profile_from_multiple_samples(
    audio_samples=samples,
    user_id='user123',
    sample_rate=None,  # Auto-detect from files
    metadata={'session': 'recording-2025-10-27'}
)
# Returns profile without embedding (same as create_voice_profile)
```

### 4. Add Sample to Profile
```python
# Add new sample to existing profile
updated_profile = cloner.add_sample_to_profile(
    profile_id='uuid-1234',
    audio='new_voice.wav',
    sample_rate=None,  # Auto-detect
    weight=None  # Auto-weight by SNR
)
# Increments profile_version and updates features
```

### 5. Get Version History
```python
# Retrieve version history
history = cloner.get_profile_version_history('uuid-1234')
# Returns List[Dict] with version entries
```

---

## Configuration Quick Reference

```yaml
voice_cloning:
  # SNR Validation (NEW)
  min_snr_db: 10.0  # Set to null to disable

  # Multi-Sample Support (NEW)
  multi_sample_quality_weighting: true  # Weight by SNR
  multi_sample_min_samples: 1
  multi_sample_max_samples: 10

  # Versioning (NEW)
  versioning_enabled: true
  version_history_max_entries: 50
  schema_version: '1.0.0'
```

---

## Profile Structure Changes

### Single-Sample Profile (with versioning)
```python
{
    'profile_id': 'uuid-...',
    'user_id': 'user123',
    'created_at': '2025-10-27T12:34:56Z',
    'audio_duration': 32.5,
    'sample_rate': 22050,
    'embedding': np.ndarray,  # 256-dim
    'vocal_range': {...},
    'timbre_features': {...},
    'embedding_stats': {...},
    'metadata': {...},

    # NEW: Versioning
    'schema_version': '1.0.0',
    'profile_version': 1,
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

### Multi-Sample Profile
```python
{
    # ... all single-sample fields ...

    # NEW: Multi-sample info
    'multi_sample_info': {
        'num_samples': 3,
        'sample_metadata': [
            {
                'index': 0,
                'filename': 'voice1.wav',
                'original_sample_rate': 44100,
                'snr_db': 18.5
            },
            # ... more samples ...
        ],
        'quality_weighted': True,
        'average_snr_db': 17.2
    },

    # Versioning automatically included
    'profile_version': 1,
    'version_history': [...]
}
```

### Updated Profile (after add_sample_to_profile)
```python
{
    # ... all fields updated ...

    'profile_version': 2,  # Incremented
    'updated_at': '2025-10-27T12:45:30Z',  # NEW
    'multi_sample_info': {
        'num_samples': 4,  # Incremented
        'sample_metadata': [
            # ... existing samples ...
            {
                'index': 3,
                'filename': 'voice4.wav',
                'snr_db': 19.2,
                'added_at': '2025-10-27T12:45:30Z'  # NEW
            }
        ]
    },
    'version_history': [
        # ... existing entries ...
        {
            'version': 2,
            'timestamp': '2025-10-27T12:45:30Z',
            'change_description': 'Added sample: voice4.wav',
            'num_samples': 4,
            'sample_added': 'voice4.wav'
        }
    ]
}
```

---

## Common Use Cases

### Use Case 1: Validate Audio Quality Before Profile Creation
```python
# Check audio quality first
report = cloner.get_audio_quality_report('voice.wav')
if report['snr_db'] < 15.0:
    print("Warning: Low SNR, consider re-recording")

if report['is_valid']:
    profile = cloner.create_voice_profile('voice.wav', user_id='user123')
```

### Use Case 2: Create Robust Profile from Multiple Takes
```python
# Record multiple takes
takes = ['take1.wav', 'take2.wav', 'take3.wav']

# Create profile with quality weighting
profile = cloner.create_voice_profile_from_multiple_samples(
    audio_samples=takes,
    user_id='user123'
)

print(f"Average SNR: {profile['multi_sample_info']['average_snr_db']:.1f} dB")
```

### Use Case 3: Incrementally Improve Profile
```python
# Start with one sample
profile = cloner.create_voice_profile('voice1.wav', user_id='user123')
profile_id = profile['profile_id']

# Add more samples over time
for new_sample in ['voice2.wav', 'voice3.wav']:
    profile = cloner.add_sample_to_profile(profile_id, new_sample)
    print(f"Updated to version {profile['profile_version']}")
```

### Use Case 4: Track Profile Changes
```python
# Get version history
history = cloner.get_profile_version_history('uuid-1234')

for entry in history:
    print(f"Version {entry['version']}: {entry['change_description']}")
    print(f"  Timestamp: {entry['timestamp']}")
    if 'num_samples' in entry:
        print(f"  Total samples: {entry['num_samples']}")
```

---

## Error Codes

### New Error Codes
- `'snr_too_low'` - Audio SNR below `min_snr_db` threshold
- `'insufficient_samples'` - Not enough samples for multi-sample profile

### Existing Error Codes
- `'missing_sample_rate'` - Sample rate required for array input
- `'duration_too_short'` - Audio shorter than `min_duration`
- `'duration_too_long'` - Audio longer than `max_duration`
- `'invalid_sample_rate'` - Sample rate outside valid range
- `'audio_too_quiet'` - RMS below `silence_threshold`
- `'validation_error'` - Generic validation failure

---

## Performance Tips

1. **SNR Validation**: Adds ~1-2ms overhead per validation
   - Disable by setting `min_snr_db: null` in config

2. **Multi-Sample Processing**: Each sample processed sequentially
   - Embedding extraction: ~100-200ms per sample
   - Total time: O(num_samples × sample_duration)

3. **Quality Weighting**: Negligible overhead (~1ms)
   - Disable by setting `multi_sample_quality_weighting: false`

4. **Version History**: Minimal memory overhead (~200 bytes per entry)
   - Auto-trims to `version_history_max_entries`

---

## Backward Compatibility

All new features are **backward compatible**:

1. **Existing profiles**: Continue to work without versioning metadata
2. **Default config**: SNR validation disabled by default (`min_snr_db: 10.0` but not enforced if not in YAML)
3. **API**: All new methods are additions, no breaking changes

### Migration Path

**No migration required!** Existing profiles work as-is:
- `create_voice_profile()`: Now adds versioning metadata automatically
- Existing profiles: Can be updated with `add_sample_to_profile()`
- Version history: Created on first update if missing

---

## Testing

```bash
# Run extension tests
pytest tests/test_voice_cloner_extensions.py -v

# Run specific test class
pytest tests/test_voice_cloner_extensions.py::TestSNRValidation -v

# Run with coverage
pytest tests/test_voice_cloner_extensions.py --cov=src.auto_voice.inference.voice_cloner
```

---

## Troubleshooting

### Issue: SNR always returns None
**Cause**: Audio too short (< 10 frames)
**Solution**: Use longer audio samples (> 1 second)

### Issue: Multi-sample profile fails with "insufficient samples"
**Cause**: Less than `multi_sample_min_samples` provided
**Solution**: Provide more samples or adjust config

### Issue: Cannot add more samples to profile
**Cause**: Profile has `multi_sample_max_samples` samples
**Solution**: Increase `multi_sample_max_samples` or create new profile

### Issue: Version history missing
**Cause**: Profile created before versioning enabled
**Solution**: Add sample to trigger version history creation

---

## Key Files

- **Implementation**: `/home/kp/autovoice/src/auto_voice/inference/voice_cloner.py`
- **Configuration**: `/home/kp/autovoice/config/audio_config.yaml`
- **Tests**: `/home/kp/autovoice/tests/test_voice_cloner_extensions.py`
- **Documentation**: `/home/kp/autovoice/docs/voice_cloner_extensions_implementation.md`
