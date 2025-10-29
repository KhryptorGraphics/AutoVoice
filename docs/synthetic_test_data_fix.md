# Synthetic Test Data Generator Fix - Comment 9

## Implementation Summary

Fixed the synthetic data generator to create actual target profiles instead of using placeholder IDs.

## Changes Made

### 1. Updated `scripts/generate_test_data.py`

**Key Improvements:**
- Added VoiceCloner integration for profile creation
- Creates real voice profiles from reference audio
- Stores profiles in `{output_dir}/profiles/` directory
- Returns real profile IDs in metadata
- Includes fallback behavior when dependencies unavailable

**New Features:**
- `--no-profiles` flag: Skip profile creation for testing
- Graceful degradation when VoiceCloner unavailable
- Extended reference audio duration (30s minimum) for better profile quality
- Relaxed SNR validation (5.0 dB) for synthetic audio

**Profile Creation Flow:**
1. Generate synthetic reference audio (30s+ for profile quality)
2. Initialize VoiceCloner with relaxed validation config
3. Extract speaker embedding from reference audio
4. Store profile with metadata (base_freq, synthetic flag)
5. Return real profile UUID in test metadata

**Metadata Structure:**
```json
{
  "id": "test_001",
  "source_audio": "path/to/test_001_source.wav",
  "target_profile_id": "uuid-from-voice-cloner",
  "reference_audio": "path/to/test_001_reference.wav",
  "metadata": {
    "base_freq_hz": 220,
    "duration_sec": 3.0,
    "sample_rate": 44100,
    "synthetic": true,
    "has_real_profile": true
  }
}
```

## Usage Examples

### Generate test data with profiles:
```bash
python scripts/generate_test_data.py --output data/evaluation --num-samples 6
```

### Generate test data without profiles (fallback mode):
```bash
python scripts/generate_test_data.py --output data/evaluation --num-samples 6 --no-profiles
```

### Run evaluation with generated test data:
```bash
python examples/evaluate_voice_conversion.py --test-metadata data/evaluation/test_set.json --output-dir results/evaluation
```

## Configuration

**VoiceCloner Config (for synthetic data):**
```python
{
    'min_duration': 5.0,        # Relaxed for synthetic
    'max_duration': 300.0,
    'extract_vocal_range': True,
    'extract_timbre_features': True,
    'storage_dir': 'output_dir/profiles',
    'min_snr_db': 5.0           # Relaxed SNR threshold
}
```

## Benefits

1. **Real Profile Testing:** Evaluation now uses actual speaker embeddings
2. **Pipeline Validation:** Tests full VoiceCloner → Pipeline → Evaluation flow
3. **Graceful Fallback:** Works with or without dependencies
4. **CI Compatibility:** Can disable profile creation with `--no-profiles`
5. **Better Quality:** 30s reference audio creates better profiles

## Testing Status

- ✅ Script runs with fallback when dependencies missing
- ✅ Generates proper JSON metadata
- ✅ Creates audio files with correct structure
- ⏳ Full integration test pending (requires resemblyzer dependency)

## Dependencies

**Required for profile creation:**
- torch
- numpy
- resemblyzer (for speaker embedding extraction)
- soundfile

**Optional:**
- librosa (for enhanced timbre features)

## Future Enhancements

1. Mock SpeakerEncoder for CI testing without resemblyzer
2. Add profile validation after creation
3. Support multiple reference samples per profile
4. Add profile quality metrics to metadata

## Related Files

- `scripts/generate_test_data.py` - Main implementation
- `src/auto_voice/inference/voice_cloner.py` - Profile creation
- `examples/evaluate_voice_conversion.py` - Evaluation script
- `src/auto_voice/evaluation/evaluator.py` - Core evaluator
