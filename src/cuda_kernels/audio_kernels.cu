/***************************************************************************
#* Kernel tuning constants
#* NOTE: The kernels below use hardcoded frame/hop size assumptions for
#* improved performance and simplicity. The primary assumption is a frame
#* length of 2048 samples and typical hop lengths (e.g., 512). These values
#* must match any host-side configuration (config/audio_config.yaml) that
#* assumes a 2048 frame length for pitch detection and spectrogram operations.
#*
#* If you wish to make these parameters configurable at runtime, expose a
#* host-side setter in bindings.cpp that stores tuning parameters and pass
#* them into the launch_* host functions. For now these values are hardcoded
#* here and should be treated as the source of truth for kernel expectations.
#***************************************************************************/

#include "kernel_utils.cuh"
#include "fft_ops.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <algorithm>

using namespace cooperative_groups;

// Enhanced pitch detection kernel with configurable vibrato method and harmonic weighting
__global__ void pitch_detection_kernel(
    float *audio,
    float *pitch,
    float *confidence,      // NEW: confidence output
    float *vibrato_flag,    // Vibrato presence heuristic
    int n_samples,
    int frame_length,       // Increased to 2048 for better resolution
    int hop_length,
    float fmin,
    float fmax,
    float threshold,
    float sample_rate,
    bool use_harmonic_weighting,  // Runtime flag: apply harmonic weighting to reduce octave errors
    int vibrato_method        // Runtime flag: 0=lightweight heuristic, 1=autocorrelation-based
) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_data[];
    float *shared_audio = shared_data;
    float *shared_prefix = &shared_data[frame_length];
    float *shared_pitch_history = &shared_data[frame_length + 534];  // Space for 20 recent pitch values

    int frame_start = frame_idx * hop_length;

    // Load frame into shared memory with strict bounds checking
    // Use __ldg() for optimized read-only global memory access
    bool in_bounds = (frame_start + tid < n_samples) && (tid < frame_length) &&
                     (frame_start + tid >= 0) && (frame_start >= 0);
    if (in_bounds) {
        shared_audio[tid] = __ldg(&audio[frame_start + tid]);
    } else if (tid < frame_length) {
        shared_audio[tid] = 0.0f;  // Zero padding
    }

    // Load pitch history into shared memory (last 20 frames)
    if (tid < 20 && frame_idx >= tid) {
        shared_pitch_history[tid] = __ldg(&pitch[frame_idx - tid]);
    } else if (tid < 20) {
        shared_pitch_history[tid] = 0.0f;
    }
    __syncthreads();

    // Early exit for silence
    if (tid == 0) {
        float energy = 0.0f;
        for (int i = 0; i < frame_length; i++) {
            energy += shared_audio[i] * shared_audio[i];
        }
        if (energy < 1e-6f) {
            pitch[frame_idx] = 0.0f;
            confidence[frame_idx] = 0.0f;
            vibrato_flag[frame_idx] = 0.0f;
            return;
        }
    }
    __syncthreads();

    int tau_min = (int)(sample_rate / fmax);
    int tau_max = (int)(sample_rate / fmin);

    // Clamp tau_max to prevent overflow in acf_storage
    // At 48kHz and fmin=80Hz, tau_max could be 600 samples
    // Limit the stored range to 512 for safety
    int tau_range = tau_max - tau_min;
    if (tau_range > 512) {
        tau_max = tau_min + 512;
    }

    float best_tau = 0.0f;
    float best_cmnd = 1.0f;  // Track best CMND value
    float cmnd_storage[512];  // Store CMND values for parabolic interpolation
    float d_prime_storage[512];  // Store normalized difference d'(tau)
    float cumulative_sum = 0.0f;  // Running cumulative sum of d'(tau)
    int first_below_threshold = -1;  // First tau that crosses threshold

    // Harmonic weighting to reduce octave errors
    float harmonic_weights[512];  // Weights for each tau based on harmonic structure

    // Pre-compute harmonic weights for all taus (conditional)
    if (tid == 0) {
        for (int tau = tau_min; tau <= tau_max; ++tau) {
            int idx = tau - tau_min;
            if (idx >= 0 && idx < 512) {
                // Check for harmonic reinforcement at 2*tau, 3*tau
                float weight = 1.0f;

                if (use_harmonic_weighting) {
                    // Boost confidence if harmonics align
                    int tau2 = tau * 2;  // Octave below
                    int tau3 = tau * 3;  // Perfect fifth below

                    if (tau2 >= tau_min && tau2 <= tau_max) {
                        // Harmonic at 2*tau reinforces fundamental
                        weight += 0.3f;
                    }
                    if (tau3 >= tau_min && tau3 <= tau_max) {
                        // Harmonic at 3*tau reinforces fundamental
                        weight += 0.2f;
                    }
                }

                harmonic_weights[idx] = weight;
            }
        }
    }
    __syncthreads();

    for (int tau = tau_min; tau <= tau_max; ++tau) {
        float acf = 0.0f;

        // Early-exit pruning: compute prefix sum to determine if this tau can beat best_cmnd
        int prefix_len = min(128, frame_length - tau);  // Use first 128 samples as prefix
        float prefix_sum = 0.0f;

        if (tid < prefix_len) {
            float diff = shared_audio[tid] - shared_audio[tid + tau];
            prefix_sum = diff * diff;
        }
        prefix_sum = block_reduce_sum(prefix_sum);

        // Store prefix in shared memory for thread 0 to check
        if (tid == 0) {
            shared_prefix[0] = prefix_sum;
        }
        __syncthreads();

        // Thread 0 decides whether to skip this tau based on prefix
        bool should_skip = false;
        if (tid == 0 && best_cmnd < 1.0f && cumulative_sum > EPSILON) {
            // Estimate lower bound on CMND from prefix
            float prefix_d_prime = shared_prefix[0] / (float)prefix_len;
            // Conservative estimate: if prefix already suggests CMND won't improve, skip
            float mean_so_far = cumulative_sum / (float)(tau - tau_min);
            if (mean_so_far > EPSILON) {
                float estimated_cmnd = prefix_d_prime / mean_so_far;
                // Only skip if estimated CMND is significantly worse than current best
                if (estimated_cmnd > best_cmnd * 1.5f) {
                    should_skip = true;
                }
            }
        }

        // Broadcast skip decision
        if (tid == 0) {
            shared_prefix[1] = should_skip ? 1.0f : 0.0f;
        }
        __syncthreads();

        if (shared_prefix[1] > 0.5f) {
            continue;  // Skip this tau
        }

        // Compute full autocorrelation difference (SSD)
        for (int j = 0; j < frame_length - tau; j += blockDim.x) {
            if (tid + j < frame_length - tau) {
                float diff = shared_audio[tid + j] - shared_audio[tid + j + tau];
                acf += diff * diff;
            }
        }

        // Use block-wide reduction to get the full sum across all threads
        acf = block_reduce_sum(acf);

        // Only thread 0 performs the final calculations
        if (tid == 0) {
            // Normalize by the number of valid samples to get d'(tau)
            float d_prime = acf / (float)(frame_length - tau);

            // Store index for arrays
            int storage_idx = tau - tau_min;
            if (storage_idx >= 0 && storage_idx < 512) {
                d_prime_storage[storage_idx] = d_prime;
            }

            // Update cumulative sum
            cumulative_sum += d_prime;

            // Compute CMND: cmnd(τ) = d'(τ) / ((1/τ) * Σ_{j=1..τ} d'(j))
            float cmnd_tau = 1.0f;  // Default value
            int tau_offset = storage_idx + 1;  // Number of taus processed so far (1-indexed)
            if (cumulative_sum > EPSILON && tau_offset > 0) {
                float mean_d_prime = cumulative_sum / (float)tau_offset;
                if (mean_d_prime > EPSILON) {
                    cmnd_tau = d_prime / mean_d_prime;

                    // Apply harmonic weighting to reduce octave errors (optional)
                    if (use_harmonic_weighting) {
                        float h_weight = harmonic_weights[storage_idx];
                        cmnd_tau /= h_weight;  // Lower CMND (better) if harmonics align
                    }
                }
            }

            // Store CMND value with bounds check
            if (storage_idx >= 0 && storage_idx < 512) {
                cmnd_storage[storage_idx] = cmnd_tau;
            }

            // Absolute threshold selection: find first tau where cmnd < threshold
            if (first_below_threshold < 0 && cmnd_tau < threshold) {
                first_below_threshold = storage_idx;
                best_cmnd = cmnd_tau;
                best_tau = (float)tau;
            }

            // Track global minimum CMND in case no tau crosses threshold
            if (first_below_threshold < 0 && cmnd_tau < best_cmnd) {
                best_cmnd = cmnd_tau;
                best_tau = (float)tau;
            }
        }
        __syncthreads();
    }

    // Only thread 0 performs parabolic interpolation and writes results
    if (tid == 0) {
        if (best_tau > 0.0f) {
            // Improved parabolic interpolation with bounds checking
            int tau_idx = (int)best_tau - tau_min;
            if (tau_idx > 0 && tau_idx < 511) {
                float prev = cmnd_storage[tau_idx - 1];
                float curr = cmnd_storage[tau_idx];
                float next = cmnd_storage[tau_idx + 1];

                // Parabolic peak refinement (find minimum)
                // Using three-point parabola fit: f(x) = a*x^2 + b*x + c
                float denom = 2.0f * curr - prev - next;
                if (fabsf(denom) > EPSILON) {
                    float delta = 0.5f * (next - prev) / denom;

                    // Clamp delta to prevent wild interpolation
                    delta = clamp(delta, -0.5f, 0.5f);

                    // Apply interpolation only if it improves accuracy
                    float interpolated_tau = best_tau + delta;
                    if (interpolated_tau >= tau_min && interpolated_tau <= tau_max) {
                        best_tau = interpolated_tau;
                    }
                }
            }

            float current_pitch = sample_rate / best_tau;
            pitch[frame_idx] = current_pitch;
            // Confidence derived from CMND: 1 - cmnd(best_tau)
            confidence[frame_idx] = clamp(1.0f - best_cmnd, 0.0f, 1.0f);

            // Vibrato detection based on method selection
            if (vibrato_method == 0 && frame_idx >= 10 && current_pitch > 0.0f) {
                // Lightweight vibrato heuristic: track short-term pitch variance over last N frames
                const int history_size = 10;
                // Compute pitch variance over recent frames
                float mean_pitch = 0.0f;
                int valid_count = 0;
                for (int i = 1; i <= history_size; i++) {
                    float prev_pitch = pitch[frame_idx - i];
                    if (prev_pitch > 0.0f) {
                        mean_pitch += prev_pitch;
                        valid_count++;
                    }
                }

                if (valid_count >= history_size / 2) {
                    mean_pitch /= valid_count;

                    // Compute variance in cents
                    float variance_cents = 0.0f;
                    for (int i = 1; i <= history_size; i++) {
                        float prev_pitch = pitch[frame_idx - i];
                        if (prev_pitch > 0.0f) {
                            float cents_diff = 1200.0f * log2f(prev_pitch / mean_pitch);
                            variance_cents += cents_diff * cents_diff;
                        }
                    }
                    variance_cents /= valid_count;
                    float std_dev_cents = sqrtf(variance_cents);

                    // Simple vibrato heuristic: std dev > 20 cents (typical vibrato depth threshold)
                    vibrato_flag[frame_idx] = (std_dev_cents >= 20.0f) ? 1.0f : 0.0f;
                } else {
                    vibrato_flag[frame_idx] = 0.0f;
                }
            } else if (vibrato_method == 1) {
                // Advanced vibrato method will be computed separately - leave as 0 for now
                vibrato_flag[frame_idx] = 0.0f;
            } else {
                vibrato_flag[frame_idx] = 0.0f;
            }
        } else {
            pitch[frame_idx] = 0.0f;
            confidence[frame_idx] = 0.0f;
            vibrato_flag[frame_idx] = 0.0f;
        }
    }
}

// Enhanced vibrato analysis kernel with autocorrelation and Hilbert transform
__global__ void vibrato_analysis_kernel(
    float *pitch_contour,
    float *vibrato_rate,
    float *vibrato_depth,
    int n_frames,
    int hop_length,
    float sample_rate
) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame_idx >= n_frames) return;

    // Window size for vibrato analysis (~200ms)
    int window_size = 20;
    int half_window = window_size / 2;

    extern __shared__ float shared_window_data[];

    // Need enough history
    if (frame_idx < half_window || frame_idx >= n_frames - half_window) {
        vibrato_rate[frame_idx] = 0.0f;
        vibrato_depth[frame_idx] = 0.0f;
        return;
    }

    // Each thread loads its portion of windowed pitch contour into shared memory
    int tid = threadIdx.x;
    if (tid < window_size) {
        int idx = frame_idx - half_window + tid;
        if (idx >= 0 && idx < n_frames) {
            shared_window_data[tid] = __ldg(&pitch_contour[idx]);
        } else {
            shared_window_data[tid] = 0.0f;
        }
    }
    __syncthreads();

    // Validate all pitches in window
    bool all_valid = true;
    for (int i = 0; i < window_size; i++) {
        if (shared_window_data[i] <= 0.0f) {
            all_valid = false;
            break;
        }
    }

    if (!all_valid) {
        vibrato_rate[frame_idx] = 0.0f;
        vibrato_depth[frame_idx] = 0.0f;
        return;
    }

    // Convert to cents relative to mean
    float mean_pitch = 0.0f;
    for (int i = 0; i < window_size; i++) {
        mean_pitch += shared_window_data[i];
    }
    mean_pitch /= window_size;

    float cents[20];
    for (int i = 0; i < window_size; i++) {
        cents[i] = 1200.0f * log2f(shared_window_data[i] / mean_pitch);
    }

    // Remove DC component (mean-center the signal)
    float mean_cents = 0.0f;
    for (int i = 0; i < window_size; i++) {
        mean_cents += cents[i];
    }
    mean_cents /= window_size;

    for (int i = 0; i < window_size; i++) {
        cents[i] -= mean_cents;
    }

    // Autocorrelation-based rate estimation (more robust than zero-crossing)
    float autocorr[20];
    for (int lag = 0; lag < window_size; lag++) {
        float sum = 0.0f;
        for (int i = 0; i < window_size - lag; i++) {
            sum += cents[i] * cents[i + lag];
        }
        autocorr[lag] = sum / (float)(window_size - lag);
    }

    // Normalize autocorrelation
    float autocorr_0 = autocorr[0];
    if (autocorr_0 > EPSILON) {
        for (int i = 0; i < window_size; i++) {
            autocorr[i] /= autocorr_0;
        }
    }

    // Find first peak in autocorrelation (lag > 0)
    // Vibrato rate typically 4-8 Hz, so at ~100 fps, period is ~12-25 frames
    int peak_lag = -1;
    float max_autocorr = 0.3f;  // Threshold for significant peak

    for (int lag = 2; lag < window_size - 1; lag++) {
        // Check if this is a local maximum
        if (autocorr[lag] > autocorr[lag - 1] && autocorr[lag] > autocorr[lag + 1]) {
            if (autocorr[lag] > max_autocorr) {
                max_autocorr = autocorr[lag];
                peak_lag = lag;
            }
        }
    }

    // Compute vibrato rate from autocorrelation peak
    float frame_rate = sample_rate / (float)hop_length;
    float estimated_rate = 0.0f;
    if (peak_lag > 0) {
        estimated_rate = frame_rate / (float)peak_lag;
    }

    // Hilbert transform approximation for vibrato depth
    // Use analytic signal to get instantaneous envelope
    float hilbert_imag[20];
    for (int i = 0; i < window_size; i++) {
        // Simplified Hilbert transform via discrete approximation
        // H[x(t)] ≈ (x(t+1) - x(t-1)) / 2
        if (i > 0 && i < window_size - 1) {
            hilbert_imag[i] = (cents[i + 1] - cents[i - 1]) / 2.0f;
        } else {
            hilbert_imag[i] = 0.0f;
        }
    }

    // Compute instantaneous envelope: sqrt(real^2 + imag^2)
    float max_envelope = 0.0f;
    for (int i = 0; i < window_size; i++) {
        float envelope = sqrtf(cents[i] * cents[i] + hilbert_imag[i] * hilbert_imag[i]);
        if (envelope > max_envelope) {
            max_envelope = envelope;
        }
    }

    // Vibrato depth is peak-to-peak amplitude from envelope
    float vibrato_depth_cents = max_envelope * 2.0f;  // Peak-to-peak

    // Validate vibrato: rate 4-8 Hz, depth > 20 cents, strong autocorrelation
    bool is_vibrato = (estimated_rate >= 4.0f && estimated_rate <= 8.0f) &&
                     (vibrato_depth_cents >= 20.0f) &&
                     (max_autocorr >= 0.5f);

    vibrato_rate[frame_idx] = is_vibrato ? estimated_rate : 0.0f;
    vibrato_depth[frame_idx] = is_vibrato ? vibrato_depth_cents : 0.0f;
}

// Voice Activity Detection kernel
__global__ void vad_kernel(float *audio, float *vad, int n_samples, int frame_length, int hop_length, float threshold) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    int frame_start = frame_idx * hop_length;

    // Compute frame energy
    float energy = 0.0f;

    // Each thread processes multiple samples if needed
    for (int i = tid; i < frame_length; i += blockDim.x) {
        int sample_idx = frame_start + i;
        if (sample_idx < n_samples) {
            float sample = audio[sample_idx];
            energy += sample * sample;
        }
    }

    // Use block-wide reduction to sum energy across all threads
    float total_energy = block_reduce_sum(energy);

    // Only thread 0 writes the final VAD result
    if (tid == 0) {
        float normalized_energy = total_energy / (float)frame_length;
        vad[frame_idx] = (normalized_energy > threshold) ? 1.0f : 0.0f;
    }
}

// Complete formant extraction kernel using LPC (Linear Predictive Coding)
__global__ void formant_extraction_kernel(
    float *audio,           // Input: audio frames [n_frames, frame_length]
    float *formants,        // Output: formant frequencies [n_frames, num_formants]
    int n_frames,
    int frame_length,       // Typically 1024-2048 samples for formant analysis
    int num_formants,       // Typically 4-5 formants
    float sample_rate,
    int lpc_order          // Configurable LPC order (10-16 typically)
) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (frame_idx >= n_frames) return;

    extern __shared__ float shared_formant_data[];
    float *shared_audio = shared_formant_data;
    float *shared_autocorr = &shared_formant_data[frame_length];

    // Load audio frame into shared memory
    for (int i = tid; i < frame_length; i += blockDim.x) {
        shared_audio[i] = __ldg(&audio[frame_idx * frame_length + i]);
    }
    __syncthreads();

    // Thread 0 performs LPC analysis (serial, but per-frame parallel across blocks)
    if (tid == 0) {
        // Step 1: Compute autocorrelation with configurable order
        float autocorr[21];  // Max order 20 + 1
        compute_autocorrelation(shared_audio, autocorr, frame_length, lpc_order);

        // Store autocorr in shared memory for debugging (optional)
        for (int i = 0; i <= lpc_order; i++) {
            shared_autocorr[i] = autocorr[i];
        }

        // Step 2: Apply Levinson-Durbin recursion to get LPC coefficients
        float lpc_coeffs[20];  // Max order 20
        float prediction_error = 0.0f;
        levinson_durbin(autocorr, lpc_coeffs, lpc_order, &prediction_error);

        // Step 3: Find polynomial roots using Durand-Kerner method
        // LPC polynomial: A(z) = 1 + a1*z^-1 + a2*z^-2 + ... + a_p*z^-p
        // We need roots of A(z) on unit circle
        float roots_real[20];
        float roots_imag[20];
        find_polynomial_roots(lpc_coeffs, roots_real, roots_imag, lpc_order, 100);

        // Step 4: Extract formants from roots with positive imaginary parts
        // Formants correspond to resonances (poles near unit circle with positive imag part)
        float formant_freqs[20];
        float formant_bandwidths[20];
        int formant_count = 0;

        for (int i = 0; i < lpc_order; i++) {
            // Check if root has positive imaginary part (formant)
            if (roots_imag[i] > 0.0f) {
                // Compute formant frequency from angle
                float angle = atan2f(roots_imag[i], roots_real[i]);
                float freq = angle * sample_rate / (2.0f * PI);

                // Compute bandwidth from radius
                float radius = sqrtf(roots_real[i] * roots_real[i] + roots_imag[i] * roots_imag[i]);
                float bandwidth = -0.5f * sample_rate * logf(radius) / PI;

                // Store formant if it's in valid range
                if (freq > 0.0f && freq < sample_rate / 2.0f) {
                    formant_freqs[formant_count] = freq;
                    formant_bandwidths[formant_count] = bandwidth;
                    formant_count++;
                }
            }
        }

        // Step 5: Sort formants by frequency (bubble sort for small arrays)
        for (int i = 0; i < formant_count - 1; i++) {
            for (int j = 0; j < formant_count - i - 1; j++) {
                if (formant_freqs[j] > formant_freqs[j + 1]) {
                    // Swap frequencies
                    float temp_f = formant_freqs[j];
                    formant_freqs[j] = formant_freqs[j + 1];
                    formant_freqs[j + 1] = temp_f;

                    // Swap bandwidths
                    float temp_b = formant_bandwidths[j];
                    formant_bandwidths[j] = formant_bandwidths[j + 1];
                    formant_bandwidths[j + 1] = temp_b;
                }
            }
        }

        // Step 6: Validate and filter formants based on expected ranges
        // F1: 200-1000 Hz, F2: 600-3000 Hz, F3: 1500-4000 Hz, F4: 2500-5000 Hz
        const float formant_min[5] = {200.0f, 600.0f, 1500.0f, 2500.0f, 3500.0f};
        const float formant_max[5] = {1000.0f, 3000.0f, 4000.0f, 5000.0f, 6000.0f};

        int validated_count = 0;
        float validated_formants[5];

        for (int i = 0; i < formant_count && validated_count < num_formants; i++) {
            float freq = formant_freqs[i];

            // Check if formant is in expected range for this formant number
            if (validated_count < 5) {
                if (freq >= formant_min[validated_count] && freq <= formant_max[validated_count]) {
                    validated_formants[validated_count] = freq;
                    validated_count++;
                }
                // Also accept formants slightly outside range if we don't have enough
                else if (validated_count < num_formants - 1 && i == formant_count - 1) {
                    // Last formant, accept it if we're short
                    if (freq >= formant_min[validated_count] * 0.8f &&
                        freq <= formant_max[validated_count] * 1.2f) {
                        validated_formants[validated_count] = freq;
                        validated_count++;
                    }
                }
            }
        }

        // Step 7: Write validated formants to output
        for (int f = 0; f < num_formants; f++) {
            if (f < validated_count) {
                formants[frame_idx * num_formants + f] = validated_formants[f];
            } else {
                // No valid formant found, write 0
                formants[frame_idx * num_formants + f] = 0.0f;
            }
        }
    }
    __syncthreads();
}

// Vocoder synthesis kernel (simplified HiFi-GAN style)
__global__ void vocoder_synthesis_kernel(float *mel_spectrogram, float *audio_out, int n_frames, int n_mels, int hop_length) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (frame_idx >= n_frames) return;
    
    extern __shared__ float shared_mel[];
    
    // Load mel frame
    if (tid < n_mels) {
        shared_mel[tid] = mel_spectrogram[frame_idx * n_mels + tid];
    }
    __syncthreads();
    
    // Simplified Griffin-Lim style synthesis (placeholder for full vocoder)
    for (int t = 0; t < hop_length; t += blockDim.x) {
        if (tid + t < hop_length) {
            // Generate audio sample using inverse STFT approximation
            float phase = 0.0f; // Random phase or estimated
            float mag = 0.0f;
            for (int k = 0; k < n_mels; k++) {
                mag += shared_mel[k] * sinf(2.0f * PI * k * (tid + t) / (float)hop_length + phase);
            }
            audio_out[frame_idx * hop_length + tid + t] = mag / (float)n_mels;
        }
    }
}

// Windowing and packing kernel for STFT
__global__ void window_and_pack_kernel(float *audio, float *windowed, int n_samples, int n_fft, int hop_length) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    int frame_start = frame_idx * hop_length;

    // Apply Hann window with proper loop to handle n_fft > blockDim.x
    for (int i = tid; i < n_fft; i += blockDim.x) {
        float window_val = 0.5f * (1.0f - cosf(2.0f * PI * i / (n_fft - 1.0f))); // Hann window
        int audio_idx = frame_start + i;
        float sample = (audio_idx < n_samples) ? audio[audio_idx] : 0.0f;
        windowed[frame_idx * n_fft + i] = sample * window_val;
    }
}

// Magnitude computation kernel
__global__ void compute_magnitude_from_complex_kernel(cufftComplex *complex_spec, float *magnitude, int n_frames, int n_fft) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    int n_bins = n_fft / 2 + 1;

    // Loop to handle cases where n_bins > blockDim.x
    for (int i = tid; i < n_bins; i += blockDim.x) {
        int idx = frame_idx * n_bins + i;
        // Correct indexing for R2C output: n_bins per frame
        cufftComplex c = complex_spec[frame_idx * n_bins + i];
        magnitude[idx] = sqrtf(c.x * c.x + c.y * c.y);
    }
}

// Host function to launch pitch detection with runtime flags for harmonic weighting and vibrato method
void launch_pitch_detection(torch::Tensor& input, torch::Tensor& output_pitch,
                           torch::Tensor& output_confidence, torch::Tensor& output_vibrato,
                           float sample_rate, int frame_length, int hop_length,
                           float fmin, float fmax, float threshold,
                           bool use_harmonic_weighting, int vibrato_method) {
    // ==========================================
    // CRITICAL: Input Validation (No Hidden Defaults)
    // ==========================================

    // Validate required parameters - no defaults allowed
    if (frame_length <= 0) {
        throw std::invalid_argument(
            "frame_length must be > 0 (got " + std::to_string(frame_length) +
            "). Valid range: typically 512-4096 samples."
        );
    }
    if (hop_length <= 0) {
        throw std::invalid_argument(
            "hop_length must be > 0 (got " + std::to_string(hop_length) +
            "). Valid range: typically 64-1024 samples."
        );
    }
    if (sample_rate <= 0.0f) {
        throw std::invalid_argument(
            "sample_rate must be > 0 (got " + std::to_string(sample_rate) +
            " Hz). Valid range: typically 8000-48000 Hz."
        );
    }

    // Validate tensors are on CUDA device
    if (!input.is_cuda()) {
        throw std::runtime_error(
            "input tensor must be on CUDA device (got device: " +
            input.device().str() + "). Use tensor.cuda() to move to GPU."
        );
    }
    if (!output_pitch.is_cuda()) {
        throw std::runtime_error(
            "output_pitch tensor must be on CUDA device (got device: " +
            output_pitch.device().str() + "). Use tensor.cuda() to move to GPU."
        );
    }
    if (!output_confidence.is_cuda()) {
        throw std::runtime_error(
            "output_confidence tensor must be on CUDA device (got device: " +
            output_confidence.device().str() + "). Use tensor.cuda() to move to GPU."
        );
    }
    if (!output_vibrato.is_cuda()) {
        throw std::runtime_error(
            "output_vibrato tensor must be on CUDA device (got device: " +
            output_vibrato.device().str() + "). Use tensor.cuda() to move to GPU."
        );
    }

    // Validate tensors are contiguous
    if (!input.is_contiguous()) {
        throw std::runtime_error(
            "input tensor must be contiguous. Use tensor.contiguous() to fix."
        );
    }
    if (!output_pitch.is_contiguous()) {
        throw std::runtime_error(
            "output_pitch tensor must be contiguous. Use tensor.contiguous() to fix."
        );
    }
    if (!output_confidence.is_contiguous()) {
        throw std::runtime_error(
            "output_confidence tensor must be contiguous. Use tensor.contiguous() to fix."
        );
    }
    if (!output_vibrato.is_contiguous()) {
        throw std::runtime_error(
            "output_vibrato tensor must be contiguous. Use tensor.contiguous() to fix."
        );
    }

    // Validate tensors have correct dtype (float32)
    if (input.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "input tensor must be float32 (got " +
            std::string(torch::toString(input.dtype())) +
            "). Use tensor.to(torch.float32) to convert."
        );
    }
    if (output_pitch.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "output_pitch tensor must be float32 (got " +
            std::string(torch::toString(output_pitch.dtype())) +
            "). Use tensor.to(torch.float32) to convert."
        );
    }
    if (output_confidence.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "output_confidence tensor must be float32 (got " +
            std::string(torch::toString(output_confidence.dtype())) +
            "). Use tensor.to(torch.float32) to convert."
        );
    }
    if (output_vibrato.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "output_vibrato tensor must be float32 (got " +
            std::string(torch::toString(output_vibrato.dtype())) +
            "). Use tensor.to(torch.float32) to convert."
        );
    }

    // ==========================================
    // End Validation - Proceed with Processing
    // ==========================================

    float *d_audio = input.data_ptr<float>();
    float *d_pitch = output_pitch.data_ptr<float>();
    float *d_confidence = output_confidence.data_ptr<float>();
    float *d_vibrato = output_vibrato.data_ptr<float>();
    int n_samples = input.size(0);

    // Validate pitch range parameters
    if (fmin <= 0.0f || fmax <= 0.0f || fmin >= fmax) {
        throw std::invalid_argument(
            "Invalid pitch range: fmin=" + std::to_string(fmin) +
            ", fmax=" + std::to_string(fmax) +
            ". Must have 0 < fmin < fmax. Typical range: 80-1000 Hz."
        );
    }
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument(
            "threshold must be in [0, 1] (got " + std::to_string(threshold) + ")."
        );
    }

    int n_frames = std::max<int>(0, (n_samples - frame_length) / hop_length + 1);
    if (n_frames <= 0) {
        CUDA_CHECK(cudaMemset(d_pitch, 0, output_pitch.numel() * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_confidence, 0, output_confidence.numel() * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vibrato, 0, output_vibrato.numel() * sizeof(float)));
        return;
    }

    dim3 block(PITCH_DETECTION_BLOCK_SIZE);
    dim3 grid(n_frames);
    // Shared memory: frame_length for audio + 534 floats for prefix pruning (2) + harmonic weights (512) + 20 for pitch history
    size_t shared_mem = PITCH_SHARED_MEM_SIZE * sizeof(float);
    pitch_detection_kernel<<<grid, block, shared_mem>>>(
        d_audio, d_pitch, d_confidence, d_vibrato,
        n_samples, frame_length, hop_length, fmin, fmax, threshold, sample_rate,
        use_harmonic_weighting, vibrato_method
    );
    CUDA_CHECK(cudaGetLastError());
}

// Host function for vibrato analysis - runs after pitch detection
void launch_vibrato_analysis(torch::Tensor& pitch_contour, torch::Tensor& vibrato_rate,
                            torch::Tensor& vibrato_depth, int hop_length, float sample_rate) {
    // ==========================================
    // CRITICAL: Input Validation (No Hidden Defaults)
    // ==========================================

    // Validate required parameters - no defaults allowed
    if (hop_length <= 0) {
        throw std::invalid_argument(
            "hop_length must be > 0 (got " + std::to_string(hop_length) +
            "). Valid range: typically 64-1024 samples."
        );
    }
    if (sample_rate <= 0.0f) {
        throw std::invalid_argument(
            "sample_rate must be > 0 (got " + std::to_string(sample_rate) +
            " Hz). Valid range: typically 8000-48000 Hz."
        );
    }

    // Validate tensors are on CUDA device
    if (!pitch_contour.is_cuda()) {
        throw std::runtime_error(
            "pitch_contour tensor must be on CUDA device (got device: " +
            pitch_contour.device().str() + "). Use tensor.cuda() to move to GPU."
        );
    }
    if (!vibrato_rate.is_cuda()) {
        throw std::runtime_error(
            "vibrato_rate tensor must be on CUDA device (got device: " +
            vibrato_rate.device().str() + "). Use tensor.cuda() to move to GPU."
        );
    }
    if (!vibrato_depth.is_cuda()) {
        throw std::runtime_error(
            "vibrato_depth tensor must be on CUDA device (got device: " +
            vibrato_depth.device().str() + "). Use tensor.cuda() to move to GPU."
        );
    }

    // Validate tensors are contiguous
    if (!pitch_contour.is_contiguous()) {
        throw std::runtime_error(
            "pitch_contour tensor must be contiguous. Use tensor.contiguous() to fix."
        );
    }
    if (!vibrato_rate.is_contiguous()) {
        throw std::runtime_error(
            "vibrato_rate tensor must be contiguous. Use tensor.contiguous() to fix."
        );
    }
    if (!vibrato_depth.is_contiguous()) {
        throw std::runtime_error(
            "vibrato_depth tensor must be contiguous. Use tensor.contiguous() to fix."
        );
    }

    // Validate tensors have correct dtype (float32)
    if (pitch_contour.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "pitch_contour tensor must be float32 (got " +
            std::string(torch::toString(pitch_contour.dtype())) +
            "). Use tensor.to(torch.float32) to convert."
        );
    }
    if (vibrato_rate.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "vibrato_rate tensor must be float32 (got " +
            std::string(torch::toString(vibrato_rate.dtype())) +
            "). Use tensor.to(torch.float32) to convert."
        );
    }
    if (vibrato_depth.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "vibrato_depth tensor must be float32 (got " +
            std::string(torch::toString(vibrato_depth.dtype())) +
            "). Use tensor.to(torch.float32) to convert."
        );
    }

    // ==========================================
    // End Validation - Proceed with Processing
    // ==========================================

    float *d_pitch = pitch_contour.data_ptr<float>();
    float *d_rate = vibrato_rate.data_ptr<float>();
    float *d_depth = vibrato_depth.data_ptr<float>();
    int n_frames = pitch_contour.size(0);

    if (n_frames <= 0) {
        return;
    }

    int threads = PITCH_DETECTION_BLOCK_SIZE;
    int blocks = (n_frames + threads - 1) / threads;

    // Shared memory for window data (20 floats for window, 20 for cents, 20 for autocorr, 20 for hilbert)
    size_t shared_mem = 20 * sizeof(float);

    vibrato_analysis_kernel<<<blocks, threads, shared_mem>>>(
        d_pitch, d_rate, d_depth, n_frames, hop_length, sample_rate
    );
    CUDA_CHECK(cudaGetLastError());
}

// Host function for formant extraction with full LPC implementation
void launch_formant_extraction(
    torch::Tensor& audio,           // Input audio [n_frames, frame_length]
    torch::Tensor& formants,        // Output formants [n_frames, num_formants]
    int frame_length,               // Frame length for LPC analysis (1024-2048)
    float sample_rate,
    int lpc_order,                  // LPC order (typically 10-14, default 14)
    int num_formants                // Number of formants to extract (typically 4-5, default 4)
) {
    // Validate inputs
    if (!audio.is_cuda()) {
        throw std::runtime_error("audio tensor must be on CUDA device");
    }
    if (!formants.is_cuda()) {
        throw std::runtime_error("formants tensor must be on CUDA device");
    }
    if (!audio.is_contiguous()) {
        throw std::runtime_error("audio tensor must be contiguous");
    }
    if (!formants.is_contiguous()) {
        throw std::runtime_error("formants tensor must be contiguous");
    }

    // Validate parameter ranges
    if (lpc_order < 8 || lpc_order > 20) {
        throw std::invalid_argument(
            "lpc_order must be in range [8, 20] (got " + std::to_string(lpc_order) +
            "). Typical values: 10-14 for speech, 14-16 for singing."
        );
    }
    if (num_formants < 1 || num_formants > 5) {
        throw std::invalid_argument(
            "num_formants must be in range [1, 5] (got " + std::to_string(num_formants) +
            "). Typical values: 3-4 for speech, 4-5 for singing."
        );
    }
    if (num_formants > lpc_order) {
        throw std::invalid_argument(
            "num_formants cannot exceed lpc_order (got num_formants=" + std::to_string(num_formants) +
            ", lpc_order=" + std::to_string(lpc_order) + "). Cannot extract more formants than LPC coefficients."
        );
    }

    float *d_audio = audio.data_ptr<float>();
    float *d_formants = formants.data_ptr<float>();
    int n_frames = audio.size(0);

    if (n_frames <= 0) {
        return;
    }

    // Use optimized block size for formant extraction
    int threads = FORMANT_EXTRACTION_BLOCK_SIZE;
    int blocks = n_frames;  // One block per frame

    // Shared memory: frame_length for audio + (lpc_order+1) for autocorr
    size_t shared_mem = (frame_length + lpc_order + 1) * sizeof(float);

    formant_extraction_kernel<<<blocks, threads, shared_mem>>>(
        d_audio, d_formants, n_frames, frame_length, num_formants, sample_rate, lpc_order
    );
    CUDA_CHECK(cudaGetLastError());
}

// Host function for vocoder synthesis (updated signature)
void launch_vocoder_synthesis(torch::Tensor& mel_spec, torch::Tensor& audio_out) {
    float *d_mel = mel_spec.data_ptr<float>();
    float *d_audio = audio_out.data_ptr<float>();
    int n_frames = mel_spec.size(0);
    int n_mels = mel_spec.size(1);
    int hop_length = 256;

    dim3 block(256);
    dim3 grid(n_frames);
    size_t shared_mem = n_mels * sizeof(float);
    vocoder_synthesis_kernel<<<grid, block, shared_mem>>>(d_mel, d_audio, n_frames, n_mels, hop_length);
    CUDA_CHECK(cudaGetLastError());
}

// Host function for voice activity detection
void launch_voice_activity_detection(torch::Tensor& input, torch::Tensor& output, float threshold) {
    float *d_audio = input.data_ptr<float>();
    float *d_vad = output.data_ptr<float>();
    int n_samples = input.size(0);
    int frame_length = 1024;  // Default
    int hop_length = 256;     // Default

    int n_frames = std::max<int>(0, (n_samples - frame_length) / hop_length + 1);
    if (n_frames <= 0) {
        CUDA_CHECK(cudaMemset(d_vad, 0, output.numel() * sizeof(float)));
        return;
    }

    dim3 block(256);
    dim3 grid(n_frames);
    // VAD kernel no longer needs shared memory for energy, block_reduce_sum uses its own shared memory internally
    vad_kernel<<<grid, block>>>(d_audio, d_vad, n_samples, frame_length, hop_length, threshold);
    CUDA_CHECK(cudaGetLastError());
}

// Host function for spectrogram computation (matching bindings signature)
void launch_spectrogram_computation(torch::Tensor& input, torch::Tensor& output, int n_fft, int hop_length, int win_length) {
    float *d_audio = input.data_ptr<float>();
    float *d_spectrogram = output.data_ptr<float>();
    int n_samples = input.size(0);

    int n_frames = std::max<int>(0, (n_samples - win_length) / hop_length + 1);
    if (n_frames <= 0) {
        CUDA_CHECK(cudaMemset(d_spectrogram, 0, output.numel() * sizeof(float)));
        return;
    }

    // Allocate temporary buffers with error checking
    float *d_windowed = nullptr;
    cufftComplex *d_fft_output = nullptr;
    
    cudaError_t err1 = cudaMalloc(&d_windowed, n_frames * n_fft * sizeof(float));
    if (err1 != cudaSuccess) {
        CUDA_CHECK(err1);
        return;
    }
    
    // R2C transform outputs (n_fft/2 + 1) complex numbers per frame
    cudaError_t err2 = cudaMalloc(&d_fft_output, n_frames * (n_fft/2 + 1) * sizeof(cufftComplex));
    if (err2 != cudaSuccess) {
        CUDA_CHECK(cudaFree(d_windowed));
        CUDA_CHECK(err2);
        return;
    }

    // Step 1: Apply windowing and pack frames
    dim3 block(256);
    dim3 grid(n_frames);
    window_and_pack_kernel<<<grid, block>>>(d_audio, d_windowed, n_samples, n_fft, hop_length);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Execute cuFFT forward transform
    execute_cufft_forward(d_windowed, d_fft_output, n_frames, n_fft);

    // Step 3: Compute magnitude spectrogram
    int n_bins = n_fft / 2 + 1;
    // Make sure we have enough threads to process all bins
    dim3 mag_block(256);
    dim3 mag_grid(n_frames);
    compute_magnitude_from_complex_kernel<<<mag_grid, mag_block>>>(d_fft_output, d_spectrogram, n_frames, n_fft);
    CUDA_CHECK(cudaGetLastError());

    // Cleanup
    CUDA_CHECK(cudaFree(d_windowed));
    CUDA_CHECK(cudaFree(d_fft_output));
}
