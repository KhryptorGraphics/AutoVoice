#include "kernel_utils.cuh"
#include "fft_ops.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <algorithm>

using namespace cooperative_groups;

// Pitch detection kernel using autocorrelation (simplified YIN)
__global__ void pitch_detection_kernel(float *audio, float *pitch, int n_samples, int frame_length, int hop_length, float fmin, float fmax, float threshold, float sample_rate) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared_audio[];
    int frame_start = frame_idx * hop_length;

    // Load frame into shared memory with bounds check
    bool in_bounds = (frame_start + tid < n_samples) && (tid < frame_length);
    if (in_bounds) {
        shared_audio[tid] = audio[frame_start + tid];
    } else if (tid < frame_length) {
        shared_audio[tid] = 0.0f;  // Zero padding
    }
    __syncthreads();
    
    int tau_min = (int)(sample_rate / fmax);
    int tau_max = (int)(sample_rate / fmin);

    float diff_mean = 0.0f;
    float cum = 0.0f;
    float bottom = 0.0f;

    for (int tau = tau_min; tau <= tau_max; ++tau) {
        float acf = 0.0f;

        // Compute autocorrelation
        for (int j = 0; j < frame_length - tau; j += blockDim.x) {
            if (tid + j < frame_length - tau) {
                float diff = shared_audio[tid + j] - shared_audio[tid + j + tau];
                acf += diff * diff;
            }
        }
        acf = warp_reduce_sum(acf);

        if (frame_idx == 0 && tau == tau_min) {
            diff_mean = acf;
        }

        float diff = acf / frame_length;
        cum += diff;
        if (tau > tau_min) {
            bottom += diff * diff;
        }

        float running_measure = cum / (float)tau;
        float normalized = diff / diff_mean;

        if (running_measure < threshold) {
            pitch[frame_idx] = sample_rate / (float)tau;
            return;
        }
    }

    pitch[frame_idx] = 0.0f; // No pitch detected
}

// Voice Activity Detection kernel
__global__ void vad_kernel(float *audio, float *vad, int n_samples, int frame_length, int hop_length, float threshold) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    int frame_start = frame_idx * hop_length;

    // Compute frame energy
    extern __shared__ float shared_energy[];
    bool in_bounds = (frame_start + tid) < n_samples;
    float sample = in_bounds ? audio[frame_start + tid] : 0.0f;
    shared_energy[tid] = in_bounds ? sample * sample : 0.0f;

    __syncthreads();

    // Reduce energy
    float frame_energy = warp_reduce_sum(shared_energy[tid]);

    if (tid == 0) {
        frame_energy = frame_energy / frame_length;
        vad[frame_idx] = (frame_energy > threshold) ? 1.0f : 0.0f;
    }
}

// Formant extraction kernel using LPC (Linear Predictive Coding)
__global__ void formant_extraction_kernel(float *spectrogram, float *formants, int n_frames, int n_freqs, int num_formants, float sample_rate) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame_idx >= n_frames) return;
    
    // Simplified LPC on spectral data
    float lpc_coeffs[10]; // Assume order 10
    float energy = 0.0f;
    
    // Compute LPC coefficients (simplified)
    for (int i = 0; i < 10; i++) {
        lpc_coeffs[i] = 0.0f; // Placeholder for actual LPC computation
    }
    
    // Find formant peaks
    for (int f = 0; f < num_formants; f++) {
        float max_val = 0.0f;
        int max_idx = 0;
        for (int k = 100; k < n_freqs - 100; k += 50) { // Search for peaks
            if (spectrogram[frame_idx * n_freqs + k] > max_val) {
                max_val = spectrogram[frame_idx * n_freqs + k];
                max_idx = k;
            }
        }
        formants[frame_idx * num_formants + f] = (float)max_idx * sample_rate / (2.0f * n_freqs);
    }
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

    // Apply Hann window
    if (tid < n_fft) {
        float window_val = 0.5f * (1.0f - cosf(2.0f * PI * tid / (n_fft - 1.0f))); // Hann window
        int audio_idx = frame_start + tid;
        float sample = (audio_idx < n_samples) ? audio[audio_idx] : 0.0f;
        windowed[frame_idx * n_fft + tid] = sample * window_val;
    }
}

// Magnitude computation kernel
__global__ void compute_magnitude_from_complex_kernel(cufftComplex *complex_spec, float *magnitude, int n_frames, int n_fft) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    int n_bins = n_fft / 2 + 1;

    if (tid < n_bins) {
        int idx = frame_idx * n_bins + tid;
        // Correct indexing for R2C output: n_bins per frame
        cufftComplex c = complex_spec[frame_idx * n_bins + tid];
        magnitude[idx] = sqrtf(c.x * c.x + c.y * c.y);
    }
}

// Host function to launch pitch detection (updated signature to match bindings)
void launch_pitch_detection(torch::Tensor& input, torch::Tensor& output, float sample_rate) {
    float *d_audio = input.data_ptr<float>();
    float *d_pitch = output.data_ptr<float>();
    int n_samples = input.size(0);
    int frame_length = 1024;
    int hop_length = 256;
    float fmin = 80.0f;
    float fmax = 400.0f;
    float threshold = 0.1f;

    int n_frames = std::max<int>(0, (n_samples - frame_length) / hop_length + 1);
    if (n_frames <= 0) {
        CUDA_CHECK(cudaMemset(d_pitch, 0, output.numel() * sizeof(float)));
        return;
    }

    dim3 block(256);
    dim3 grid(n_frames);
    size_t shared_mem = frame_length * sizeof(float);
    pitch_detection_kernel<<<grid, block, shared_mem>>>(d_audio, d_pitch, n_samples, frame_length, hop_length, fmin, fmax, threshold, sample_rate);
    CUDA_CHECK(cudaGetLastError());
}

// Host function for formant extraction (updated signature)
void launch_formant_extraction(torch::Tensor& input, torch::Tensor& output, float sample_rate) {
    float *d_spectrogram = input.data_ptr<float>();
    float *d_formants = output.data_ptr<float>();
    int n_frames = input.size(0);
    int n_freqs = input.size(1);
    int num_formants = output.size(1);

    int threads = 256;
    int blocks = (n_frames + threads - 1) / threads;
    formant_extraction_kernel<<<blocks, threads>>>(d_spectrogram, d_formants, n_frames, n_freqs, num_formants, sample_rate);
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
    size_t shared_mem = 256 * sizeof(float);
    vad_kernel<<<grid, block, shared_mem>>>(d_audio, d_vad, n_samples, frame_length, hop_length, threshold);
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

    // Allocate temporary buffers
    float *d_windowed;
    cufftComplex *d_fft_output;
    CUDA_CHECK(cudaMalloc(&d_windowed, n_frames * n_fft * sizeof(float)));
    // R2C transform outputs (n_fft/2 + 1) complex numbers per frame
    CUDA_CHECK(cudaMalloc(&d_fft_output, n_frames * (n_fft/2 + 1) * sizeof(cufftComplex)));

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