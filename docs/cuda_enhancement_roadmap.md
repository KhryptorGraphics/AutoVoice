# CUDA Kernel Enhancement Implementation Roadmap

**Status:** Implementation Plan
**Created:** October 27, 2025
**Estimated Total Effort:** 5-7 phases over multiple sessions

---

## Overview

This roadmap breaks down the comprehensive CUDA kernel enhancement plan into manageable, testable phases. Each phase builds on the previous one and can be validated independently.

---

## Phase 1: Foundation - Helper Functions and Utilities (PRIORITY: CRITICAL)

**Estimated Lines:** ~500
**Files Modified:** 2
**Dependencies:** None
**Validation:** Unit tests for helper functions

### Scope

#### 1.1 Update kernel_utils.cuh
- Add LPC helper functions:
  - `levinson_durbin()` - Levinson-Durbin recursion for LPC coefficients
  - `compute_autocorrelation()` - Autocorrelation computation
  - `find_polynomial_roots()` - Laguerre's method for root finding
- Add STFT/ISTFT helpers:
  - `compute_window_sum()` - Window normalization for overlap-add
  - `complex_mul()` - Complex multiplication
  - `complex_conj()` - Complex conjugate
- Add perceptual weighting helpers:
  - `compute_a_weighting()` - A-weighting calculation
- Add optimization macros:
  - Block size constants
  - Shared memory size constants
  - Profiling macros (optional)

#### 1.2 Update fft_ops.cuh
- Add singing-specific constants
- Add kernel declarations for new kernels
- Add helper functions for singing voice processing
- Update shared memory configurations

### Success Criteria
- All helper functions compile without errors
- Basic unit tests pass for mathematical correctness
- Constants are properly defined

### Implementation Priority
**HIGH** - Required foundation for all subsequent phases

---

## Phase 2: Core Optimization - Pitch Detection Enhancement (PRIORITY: HIGH)

**Estimated Lines:** ~300
**Files Modified:** 1
**Dependencies:** Phase 1
**Validation:** Accuracy tests vs torchcrepe

### Scope

#### 2.1 Enhance pitch_detection_kernel
- Add harmonic weighting for octave error reduction
- Improve parabolic interpolation with bounds checking
- Optimize memory access patterns (use `__ldg()`)
- Add pitch history to shared memory (20 floats)
- Optimize block size based on profiling targets

#### 2.2 Enhance vibrato_analysis_kernel
- Replace zero-crossing with autocorrelation method
- Add Hilbert transform approximation for depth
- Optimize batch processing
- Use shared memory for window data

### Success Criteria
- Pitch detection accuracy: ±2 Hz for singing voice
- 5-10x speedup vs torchcrepe
- Vibrato detection accuracy: ±0.5 Hz rate, ±10 cents depth
- No octave errors on test cases

### Implementation Priority
**HIGH** - Core singing voice feature

---

## Phase 3: Advanced Audio - Formant Extraction (PRIORITY: HIGH)

**Estimated Lines:** ~400
**Files Modified:** 1
**Dependencies:** Phase 1 (LPC helpers)
**Validation:** Accuracy tests vs praat-parselmouth

### Scope

#### 3.1 Complete formant_extraction_kernel
- Implement full LPC analysis:
  - Compute autocorrelation (order 10-14)
  - Apply Levinson-Durbin recursion
  - Store LPC coefficients in shared memory
- Implement polynomial root finding:
  - Use Laguerre's method
  - Extract roots with positive imaginary parts
  - Convert to formant frequencies
- Validate and filter formants:
  - Check F1-F4 frequency ranges
  - Discard invalid formants
  - Return 4-5 valid formants

#### 3.2 Optimize performance
- Use shared memory for autocorrelation
- Parallelize across frames
- Use warp-level primitives

### Success Criteria
- Formant accuracy: <50 Hz RMSE vs praat
- 20-50x speedup vs praat-parselmouth
- Valid formant ranges: F1(200-1000), F2(600-3000), F3(1500-4000), F4(2500-5000)

### Implementation Priority
**HIGH** - Critical for voice timbre analysis

---

## Phase 4: Spectral Processing - STFT/ISTFT with Overlap-Add (PRIORITY: CRITICAL)

**Estimated Lines:** ~600
**Files Modified:** 1
**Dependencies:** Phase 1 (STFT helpers)
**Validation:** Perfect reconstruction tests

### Scope

#### 4.1 Fix ifft_reconstruction_kernel
- Remove simplified placeholder
- Implement proper overlap-add synthesis:
  - Two-pass approach (window sum + overlap-add)
  - Atomic operations optimization
  - Window normalization
- Ensure perfect reconstruction

#### 4.2 Add overlap_add_synthesis_kernel
- Dedicated kernel for overlap-add
- Support 50% overlap (hop_length = n_fft/2)
- Optimize atomic operations with shared memory

#### 4.3 Add normalize_istft_kernel
- Normalize by window sum
- Safe division with epsilon
- Vectorize with float4

#### 4.4 Optimize cuFFT usage
- Add plan caching mechanism
- Support batched FFT
- Add CUDA stream support

### Success Criteria
- Perfect reconstruction: error <1e-5
- 5-10x speedup vs torch.stft/istft
- Support window sizes: 512, 1024, 2048, 4096
- Support overlap ratios: 25%, 50%, 75%

### Implementation Priority
**CRITICAL** - Foundation for all spectral operations

---

## Phase 5: Advanced Kernels - Mel-Spectrogram and Real-time Conversion (PRIORITY: MEDIUM)

**Estimated Lines:** ~800
**Files Modified:** 1
**Dependencies:** Phases 1, 4
**Validation:** Accuracy vs librosa, latency tests

### Scope

#### 5.1 Add mel_spectrogram_singing_kernel
- Fused pipeline: windowing → FFT → magnitude → mel filterbank → log
- Singing-specific Hann window
- Optional perceptual weighting (A-weighting)
- Shared memory optimization

#### 5.2 Add realtime_voice_conversion_kernel
- Chunk-based processing (100ms chunks)
- Overlap buffer state management
- Fused feature extraction
- Low-latency optimization (<10ms)

#### 5.3 Add apply_perceptual_weighting_kernel
- Apply A-weighting to mel bins
- Shared memory for weighting curve
- Fuse with mel filterbank

#### 5.4 Add compute_log_mel_kernel
- Fused log computation
- Vectorize with float4

### Success Criteria
- Mel-spectrogram accuracy: <1e-3 vs librosa
- 10-20x speedup vs librosa
- Real-time conversion latency: <10ms per 100ms chunk
- Perceptual weighting improves quality

### Implementation Priority
**MEDIUM** - Advanced features, builds on core functionality

---

## Phase 6: Integration - Python Bindings and Host Functions (PRIORITY: HIGH)

**Estimated Lines:** ~400
**Files Modified:** 1
**Dependencies:** Phases 2-5
**Validation:** Python integration tests

### Scope

#### 6.1 Update bindings.cpp
- Add forward declarations for all new kernels
- Add Python bindings with default parameters:
  - `mel_spectrogram_singing`
  - `optimized_stft`
  - `optimized_istft`
  - `realtime_voice_conversion`
  - `apply_perceptual_weighting`
  - `formant_extraction` (enhanced)

#### 6.2 Add host functions
- `launch_mel_spectrogram_singing()`
- `launch_optimized_stft()`
- `launch_optimized_istft()`
- `launch_realtime_voice_conversion()`
- `launch_apply_perceptual_weighting()`
- Update `launch_pitch_detection()` with new parameters
- Update `launch_formant_extraction()` for LPC

### Success Criteria
- All kernels accessible from Python
- Proper error handling and validation
- Default parameters work correctly
- Integration with existing AudioProcessor

### Implementation Priority
**HIGH** - Required to use kernels from Python

---

## Phase 7: Validation - Comprehensive Test Suite (PRIORITY: HIGH)

**Estimated Lines:** ~1500
**Files Modified:** 2
**Dependencies:** Phases 1-6
**Validation:** All tests pass

### Scope

#### 7.1 Add TestVoiceConversionKernels class
- Pitch detection tests (accuracy, vibrato, harmonic weighting)
- Mel-spectrogram tests (accuracy, speedup, perceptual weighting)
- Formant extraction tests (accuracy, speedup, validation)
- STFT/ISTFT tests (accuracy, perfect reconstruction, speedup)
- Real-time conversion tests (latency, streaming, state management)

#### 7.2 Add TestKernelPerformance class
- Speedup benchmarks vs reference implementations
- Memory bandwidth tests
- Throughput tests
- End-to-end pipeline tests

#### 7.3 Add TestKernelAccuracy class
- Accuracy validation vs torchcrepe, librosa, praat
- Numerical precision tests
- Reconstruction error tests

#### 7.4 Add TestKernelStressTests class
- Long audio tests (10+ minutes)
- Large batch tests
- Sustained streaming tests
- Extreme parameter tests

#### 7.5 Add test fixtures
- `sample_audio_44khz()`
- `sample_singing_audio()`
- `mel_filterbank_44khz()`
- `benchmark_timer()`

### Success Criteria
- >90% code coverage
- All accuracy tests pass (tolerance met)
- All speedup tests pass (targets met)
- No memory leaks or crashes
- Stress tests complete successfully

### Implementation Priority
**HIGH** - Ensures correctness and performance

---

## Phase 8: Optimization - Profiling and Tuning (PRIORITY: MEDIUM)

**Estimated Lines:** ~800
**Files Modified:** 2 (new files)
**Dependencies:** Phases 1-7
**Validation:** Profiling reports

### Scope

#### 8.1 Create profile_cuda_kernels.py
- Profiling functions for each kernel
- Nsight Compute integration
- Nsight Systems integration
- Benchmark comparison with references
- Report generation
- Bottleneck analysis

#### 8.2 Profile and optimize
- Run Nsight profiling on all kernels
- Identify bottlenecks
- Optimize block/grid sizes
- Optimize shared memory usage
- Optimize memory access patterns
- Apply warp-level optimizations

#### 8.3 Create cuda_optimization_guide.md
- Document kernel architectures
- Document optimization methodology
- Document profiling results
- Document performance benchmarks
- Document accuracy validation
- Provide troubleshooting guide

### Success Criteria
- Profiling script works correctly
- All kernels meet speedup targets
- Occupancy >50% for all kernels
- Memory bandwidth >80% of peak
- Comprehensive documentation

### Implementation Priority
**MEDIUM** - Optimization and documentation

---

## Phase 9: Configuration - Runtime Tuning (PRIORITY: LOW)

**Estimated Lines:** ~100
**Files Modified:** 1
**Dependencies:** Phase 8 (optimization results)
**Validation:** Config loads correctly

### Scope

#### 9.1 Update audio_config.yaml
- Add cuda_kernels section
- Add pitch_detection config
- Add mel_spectrogram config
- Add formant_extraction config
- Add stft/istft config
- Add realtime_conversion config
- Add optimization flags

### Success Criteria
- Config file parses correctly
- Parameters can be tuned at runtime
- Optimal defaults from profiling

### Implementation Priority
**LOW** - Nice-to-have, optional tuning

---

## Implementation Schedule

### Session 1 (Current)
- **Phase 1:** Foundation (kernel_utils.cuh, fft_ops.cuh)
- **Phase 2:** Pitch detection enhancements

### Session 2
- **Phase 3:** Formant extraction
- **Phase 4:** STFT/ISTFT overlap-add (Part 1)

### Session 3
- **Phase 4:** STFT/ISTFT overlap-add (Part 2)
- **Phase 5:** Mel-spectrogram and real-time conversion

### Session 4
- **Phase 6:** Python bindings and host functions
- **Phase 7:** Test suite (Part 1)

### Session 5
- **Phase 7:** Test suite (Part 2)
- **Phase 8:** Profiling and optimization

### Session 6
- **Phase 8:** Documentation
- **Phase 9:** Configuration
- **Final validation and cleanup**

---

## Risk Mitigation

### Technical Risks
1. **LPC convergence issues** - Mitigation: Use established Levinson-Durbin algorithm, validate against praat
2. **Overlap-add artifacts** - Mitigation: Follow established STFT/ISTFT literature, test reconstruction
3. **Memory bandwidth bottlenecks** - Mitigation: Profile with Nsight, optimize access patterns
4. **Atomic operation contention** - Mitigation: Use shared memory accumulation before atomic writes

### Testing Risks
1. **Reference implementations unavailable** - Mitigation: Skip comparison tests, use synthetic validation
2. **Numerical precision differences** - Mitigation: Use appropriate tolerances, document differences
3. **Hardware dependency** - Mitigation: Test on multiple GPU architectures

---

## Success Metrics

### Performance Targets
- Pitch detection: 5-10x speedup ✓
- Mel-spectrogram: 10-20x speedup ✓
- Formant extraction: 20-50x speedup ✓
- STFT/ISTFT: 5-10x speedup ✓
- Real-time conversion: <10ms latency ✓

### Accuracy Targets
- Pitch detection: ±2 Hz ✓
- Mel-spectrogram: <1e-3 max error ✓
- Formant extraction: <50 Hz RMSE ✓
- STFT/ISTFT: <1e-5 reconstruction error ✓

### Code Quality Targets
- >90% test coverage ✓
- No memory leaks ✓
- No race conditions ✓
- Comprehensive documentation ✓

---

## Notes

- Each phase should be validated before proceeding to the next
- Profiling should be done incrementally after each major kernel addition
- Documentation should be updated as implementations are completed
- Test suite should grow with each phase

**Next Step:** Begin Phase 1 implementation
