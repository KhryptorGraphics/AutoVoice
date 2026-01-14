
#include "kernel_utils.cuh"
#include "fft_ops.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include <cstdint>   // GCC 14 compatibility
#include <cstddef>   // GCC 14 compatibility

// Prevent cublasLt.h extern "C" error during device code compilation (CUDA 13.0+)
#ifdef __CUDA_ARCH__
#define CUBLASLT_H_
#endif

#include <torch/extension.h>

using namespace cooperative_groups;

// Forward declarations (not in headers)
__global__ void normalize_kernel(float *data, int n_fft, int total_samples);
cufftHandle get_or_create_plan(const std::string& key, int n_fft, cufftType type, int batch);
void apply_perceptual_weighting(
    torch::Tensor mel_spectrogram,
    torch::Tensor mel_frequencies,
    int n_frames, int mel_bins, int batch_size);

// Windowing kernel using provided window tensor (batched version)
__global__ void apply_window_kernel(float *audio, float *window, float *windowed, int audio_length, int n_fft, int hop_length, int n_frames) {
    int frame_idx = blockIdx.x;   // Frame index
    int batch_idx = blockIdx.y;   // Batch index
    int tid = threadIdx.x;        // Thread within frame (sample index within window)

    if (frame_idx >= n_frames) return;

    // Compute frame start position and audio offset for this batch
    int frame_start = frame_idx * hop_length;
    int audio_offset = batch_idx * audio_length;

    if (tid < n_fft) {
        float w = window[tid];
        int audio_idx = audio_offset + frame_start + tid;
        float sample = (frame_start + tid < audio_length) ? audio[audio_idx] : 0.0f;

        // Write to windowed output: [batch_size, n_frames, n_fft]
        int windowed_idx = (batch_idx * n_frames + frame_idx) * n_fft + tid;
        windowed[windowed_idx] = sample * w;
    }
}

// Mel-scale filter bank application kernel (original complex version)
__global__ void apply_mel_filterbank_kernel(cufftComplex *spectrum, float *mel_spectrum, int n_freqs, int n_mels, float *mel_basis) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= n_mels) return;

    float sum = 0.0f;
    for (int k = 0; k < n_freqs; k++) {
        int bin_idx = frame_idx * n_freqs + k;
        float mag = sqrtf(spectrum[bin_idx].x * spectrum[bin_idx].x + spectrum[bin_idx].y * spectrum[bin_idx].y);
        sum += mag * mel_basis[tid * n_freqs + k];
    }
    mel_spectrum[frame_idx * n_mels + tid] = logf(safe_divide(sum, n_freqs));
}

// Magnitude-based mel filter bank kernel
__global__ void apply_mel_magnitude_kernel(const float *magnitude, float *mel_spectrum, int n_freqs, int n_mels, float *mel_basis) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= n_mels) return;

    float sum = 0.0f;
    for (int k = 0; k < n_freqs; k++) {
        int bin_idx = frame_idx * n_freqs + k;
        sum += magnitude[bin_idx] * mel_basis[tid * n_freqs + k];
    }
    mel_spectrum[frame_idx * n_mels + tid] = logf(safe_divide(sum, n_freqs));
}

// Parallel spectral analysis kernel for magnitude computation
__global__ void compute_magnitude_kernel(cufftComplex *spectrum, float *magnitude, int n_frames, int n_freqs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_frames * (n_freqs / 2 + 1)) return;
    
    int frame = idx / (n_freqs / 2 + 1);
    int bin = idx % (n_freqs / 2 + 1);
    
    int complex_idx = frame * n_freqs + bin;
    magnitude[idx] = sqrtf(spectrum[complex_idx].x * spectrum[complex_idx].x + spectrum[complex_idx].y * spectrum[complex_idx].y);
}

// Log magnitude computation kernel
__global__ void log_magnitude_kernel(float *magnitude, float *log_mag, int n_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bins) return;
    
    log_mag[idx] = logf(safe_divide(magnitude[idx], 1.0f));  // Add small epsilon if needed
}

// Optimized STFT kernel with windowing and batching
__global__ void optimized_stft_kernel(
    float *audio,               // Input audio [batch_size, audio_length]
    cufftComplex *stft_output,  // Output STFT [batch_size, n_frames, n_fft/2+1]
    float *window,              // Window function [n_fft]
    int audio_length,
    int n_fft,
    int hop_length,
    int batch_size
) {
    int frame_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    extern __shared__ float shared_windowed[];

    int frame_start = frame_idx * hop_length;
    int audio_offset = batch_idx * audio_length;

    // Apply window in shared memory
    for (int i = tid; i < n_fft; i += blockDim.x) {
        int audio_idx = frame_start + i;
        float sample = (audio_idx < audio_length) ? __ldg(&audio[audio_offset + audio_idx]) : 0.0f;
        float win_val = __ldg(&window[i]);
        shared_windowed[i] = sample * win_val;
    }
    __syncthreads();

    // FFT will be performed externally using cuFFT on shared_windowed data
    // This kernel prepares windowed frames for batched FFT execution
}

// Optimized ISTFT kernel with windowing
__global__ void optimized_istft_kernel(
    cufftComplex *stft_input,   // Input STFT [batch_size, n_frames, n_fft/2+1]
    float *audio_output,        // Output audio [batch_size, audio_length]
    float *window,              // Window function [n_fft]
    int audio_length,
    int n_fft,
    int hop_length,
    int batch_size
) {
    int frame_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    extern __shared__ float shared_ifft_frame[];

    // IFFT will be performed externally using cuFFT
    // This kernel applies windowing after IFFT

    int frame_start = frame_idx * hop_length;
    int audio_offset = batch_idx * audio_length;

    // Apply window to IFFT output
    for (int i = tid; i < n_fft; i += blockDim.x) {
        float win_val = __ldg(&window[i]);
        shared_ifft_frame[i] *= win_val;
    }
    __syncthreads();

    // Overlap-add will be performed by overlap_add_synthesis_kernel
}

// Dedicated overlap-add synthesis kernel for ISTFT
__global__ void overlap_add_synthesis_kernel(
    float *ifft_frames,         // IFFT output frames [n_frames, n_fft]
    float *audio_output,        // Output audio [audio_length]
    float *window,              // Window function [n_fft]
    int audio_length,
    int n_frames,
    int n_fft,
    int hop_length
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= audio_length) return;

    extern __shared__ float shared_ola_buffer[];

    // Compute which frames contribute to this audio sample
    int first_frame = max(0, (tid - n_fft + 1) / hop_length);
    int last_frame = min(n_frames - 1, tid / hop_length);

    float sum = 0.0f;

    // Accumulate contributions from overlapping frames
    for (int frame = first_frame; frame <= last_frame; frame++) {
        int frame_start = frame * hop_length;
        int offset_in_frame = tid - frame_start;

        if (offset_in_frame >= 0 && offset_in_frame < n_fft) {
            // Load windowed IFFT sample and accumulate
            float ifft_sample = ifft_frames[frame * n_fft + offset_in_frame];
            float win_val = __ldg(&window[offset_in_frame]);
            sum += ifft_sample * win_val;
        }
    }

    // Use atomic add to handle overlapping writes
    atomicAdd(&audio_output[tid], sum);
}

// Precompute window sum for perfect reconstruction (called once)
__global__ void precompute_window_sum_kernel(
    float *window,              // Window function [n_fft]
    float *window_sum,          // Output window sum [audio_length]
    int audio_length,
    int n_fft,
    int hop_length
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= audio_length) return;

    // Compute which frames contribute to this sample
    int n_frames = (audio_length - 1) / hop_length + 1;
    int first_frame = max(0, (sample_idx - n_fft + 1) / hop_length);
    int last_frame = min(n_frames - 1, sample_idx / hop_length);

    float sum = 0.0f;

    // Sum squared window values from overlapping frames
    for (int frame = first_frame; frame <= last_frame; frame++) {
        int frame_start = frame * hop_length;
        int offset_in_frame = sample_idx - frame_start;

        if (offset_in_frame >= 0 && offset_in_frame < n_fft) {
            float win_val = __ldg(&window[offset_in_frame]);
            sum += win_val * win_val;
        }
    }

    window_sum[sample_idx] = sum;
}

// Normalize ISTFT output by precomputed window sum for perfect reconstruction
__global__ void normalize_istft_kernel(
    float *audio,               // Audio to normalize [batch_size, audio_length]
    float *window_sum,          // Precomputed window sum [audio_length]
    int audio_length,
    int batch_size
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.y;

    if (tid >= audio_length || batch_idx >= batch_size) return;

    // Load precomputed window sum (O(1) lookup)
    float sum = __ldg(&window_sum[tid]);

    // Normalize by window sum for perfect reconstruction
    int audio_offset = batch_idx * audio_length;
    if (sum > EPSILON) {
        audio[audio_offset + tid] /= sum;
    }
}

// Host function to apply window
void launch_apply_window(torch::Tensor& input, torch::Tensor& window, torch::Tensor& output) {
    float *d_input = input.data_ptr<float>();
    float *d_window = window.data_ptr<float>();
    float *d_output = output.data_ptr<float>();

    int n_samples = input.size(0);
    int n_fft = window.size(0);
    int hop_length = 256;  // Default

    // Compute n_frames with proper bounds checking
    int n_frames = (n_samples >= n_fft) ? ((n_samples - n_fft) / hop_length + 1) : 0;

    // Early return if no frames to process
    if (n_frames == 0) {
        output.zero_();
        return;
    }

    // Use batched launch configuration: dim3(n_frames, batch_size)
    dim3 grid(n_frames, 1);  // batch_size=1 for single audio
    dim3 block(256);

    // Call kernel with all 7 parameters (including n_frames)
    apply_window_kernel<<<grid, block>>>(d_input, d_window, d_output, n_samples, n_fft, hop_length, n_frames);
    CUDA_CHECK(cudaGetLastError());
}

// Host function to compute magnitude from complex spectrum
void launch_compute_magnitude(torch::Tensor& complex_input, torch::Tensor& magnitude) {
    // Validate that input is complex type
    TORCH_CHECK(complex_input.is_complex(), "Input tensor must be complex type");
    TORCH_CHECK(complex_input.dtype() == torch::kComplexFloat, "Input must be complex float32");

    // Use c10::complex<float> which is compatible with cufftComplex
    cufftComplex *d_complex = reinterpret_cast<cufftComplex*>(complex_input.data_ptr<c10::complex<float>>());
    float *d_magnitude = magnitude.data_ptr<float>();

    int n_frames = complex_input.size(0);
    int n_freqs = complex_input.size(1);

    int total_bins = n_frames * (n_freqs / 2 + 1);
    dim3 block(256);
    dim3 grid((total_bins + block.x - 1) / block.x);

    compute_magnitude_kernel<<<grid, block>>>(d_complex, d_magnitude, n_frames, n_freqs);
    CUDA_CHECK(cudaGetLastError());
}

// Host function to apply mel filterbank
void launch_apply_mel_filters(torch::Tensor& magnitude, torch::Tensor& mel_filters, torch::Tensor& mel_spec) {
    float *d_magnitude = magnitude.data_ptr<float>();
    float *d_mel_filters = mel_filters.data_ptr<float>();
    float *d_mel_spec = mel_spec.data_ptr<float>();

    int n_frames = magnitude.size(0);
    int n_freqs = magnitude.size(1);
    int n_mels = mel_filters.size(0);

    dim3 block(256);
    dim3 grid(n_frames);

    // Use a magnitude-based mel filterbank kernel instead
    apply_mel_magnitude_kernel<<<grid, block>>>(d_magnitude, d_mel_spec, n_freqs, n_mels, d_mel_filters);
    CUDA_CHECK(cudaGetLastError());
}

// Host function to execute forward FFT (refactored to use plan cache)
void execute_cufft_forward(float *d_input, cufftComplex *d_output, int batch_size, int n_fft, cudaStream_t stream) {
    // Generate plan key
    std::string plan_key = "fft_forward_" + std::to_string(n_fft) + "_" + std::to_string(batch_size);

    // Get or create cached plan
    cufftHandle plan = get_or_create_plan(plan_key, n_fft, CUFFT_R2C, batch_size);

    // Set stream
    if (stream != 0) {
        CUFFT_CHECK(cufftSetStream(plan, stream));
    }

    // Execute FFT
    CUFFT_CHECK(cufftExecR2C(plan, d_input, d_output));
}

// Host function to execute inverse FFT (refactored to use plan cache)
void execute_cufft_inverse(cufftComplex *d_input, float *d_output, int batch_size, int n_fft, cudaStream_t stream) {
    // Generate plan key
    std::string plan_key = "fft_inverse_" + std::to_string(n_fft) + "_" + std::to_string(batch_size);

    // Get or create cached plan
    cufftHandle plan = get_or_create_plan(plan_key, n_fft, CUFFT_C2R, batch_size);

    // Set stream
    if (stream != 0) {
        CUFFT_CHECK(cufftSetStream(plan, stream));
    }

    // Execute IFFT
    CUFFT_CHECK(cufftExecC2R(plan, d_input, d_output));

    // Normalize
    int total_samples = batch_size * n_fft;
    dim3 block(256);
    dim3 grid((total_samples + block.x - 1) / block.x);

    normalize_kernel<<<grid, block, 0, stream>>>(d_output, n_fft, total_samples);
    CUDA_CHECK(cudaGetLastError());
}

// Normalization kernel
__global__ void normalize_kernel(float *data, int n_fft, int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;
    data[idx] /= (float)n_fft;
}

// cuFFT plan cache for optimization (global static cache)
static std::unordered_map<std::string, cufftHandle> g_cufft_plan_cache;
static std::mutex g_plan_cache_mutex;

// Helper function to get or create cached cuFFT plan
cufftHandle get_or_create_plan(const std::string& key, int n_fft, cufftType type, int batch) {
    std::lock_guard<std::mutex> lock(g_plan_cache_mutex);

    auto it = g_cufft_plan_cache.find(key);
    if (it != g_cufft_plan_cache.end()) {
        return it->second;
    }

    // Create new plan
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, type, batch));
    g_cufft_plan_cache[key] = plan;

    return plan;
}

// Host function to launch optimized STFT
void launch_optimized_stft(
    torch::Tensor& audio,           // Input audio [batch_size, audio_length]
    torch::Tensor& window,          // Window function [n_fft]
    torch::Tensor& stft_output,     // Output STFT [batch_size, n_frames, n_fft/2+1]
    int n_fft,
    int hop_length
) {
    // Validate inputs
    if (!audio.is_cuda() || !window.is_cuda() || !stft_output.is_cuda()) {
        throw std::runtime_error("All tensors must be on CUDA device");
    }

    float *d_audio = audio.data_ptr<float>();
    float *d_window = window.data_ptr<float>();
    cufftComplex *d_stft_output = reinterpret_cast<cufftComplex*>(stft_output.data_ptr<c10::complex<float>>());

    int batch_size = audio.size(0);
    int audio_length = audio.size(1);
    int n_frames = std::max<int>(0, (audio_length - n_fft) / hop_length + 1);
    if (n_frames == 0) {
        // Zero-fill outputs and return
        CUDA_CHECK(cudaMemset(d_stft_output, 0, stft_output.numel() * sizeof(cufftComplex)));
        return;
    }

    // Allocate global memory buffer for windowed frames [batch_size*n_frames, n_fft]
    float *d_windowed_frames;
    CUDA_CHECK(cudaMalloc(&d_windowed_frames, batch_size * n_frames * n_fft * sizeof(float)));

    // Step 1: Launch windowing kernel once with batched grid covering all frames and batches
    dim3 window_block(256);
    dim3 window_grid(n_frames, batch_size);

    // Single batched kernel launch for all frames and batches
    // blockIdx.x = frame index, blockIdx.y = batch index
    apply_window_kernel<<<window_grid, window_block>>>(
        d_audio,
        d_window,
        d_windowed_frames,
        audio_length,
        n_fft,
        hop_length,
        n_frames
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Execute batched FFT R2C using cached plan
    std::string plan_key = "stft_" + std::to_string(n_fft) + "_" + std::to_string(batch_size * n_frames);
    cufftHandle plan = get_or_create_plan(plan_key, n_fft, CUFFT_R2C, batch_size * n_frames);

    // Set stream for plan
    CUFFT_CHECK(cufftSetStream(plan, 0));

    // Execute FFT: windowed_frames -> stft_output
    CUFFT_CHECK(cufftExecR2C(plan, d_windowed_frames, d_stft_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary buffer
    CUDA_CHECK(cudaFree(d_windowed_frames));
}

// Host function to launch optimized ISTFT
void launch_optimized_istft(
    torch::Tensor& stft_input,      // Input STFT [batch_size, n_frames, n_fft/2+1]
    torch::Tensor& window,          // Window function [n_fft]
    torch::Tensor& audio_output,    // Output audio [batch_size, audio_length]
    int n_fft,
    int hop_length
) {
    // Validate inputs
    if (!stft_input.is_cuda() || !window.is_cuda() || !audio_output.is_cuda()) {
        throw std::runtime_error("All tensors must be on CUDA device");
    }

    cufftComplex *d_stft_input = reinterpret_cast<cufftComplex*>(stft_input.data_ptr<c10::complex<float>>());
    float *d_window = window.data_ptr<float>();
    float *d_audio_output = audio_output.data_ptr<float>();

    int batch_size = stft_input.size(0);
    int n_frames = stft_input.size(1);
    int audio_length = audio_output.size(1);

    // Allocate temporary buffers
    float *d_ifft_frames;
    float *d_window_sum;
    CUDA_CHECK(cudaMalloc(&d_ifft_frames, batch_size * n_frames * n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_window_sum, audio_length * sizeof(float)));

    // Step 1: Precompute window sum once for all batches
    int threads = 256;
    int blocks = (audio_length + threads - 1) / threads;

    precompute_window_sum_kernel<<<blocks, threads>>>(
        d_window, d_window_sum, audio_length, n_fft, hop_length
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Zero-initialize audio output before overlap-add
    CUDA_CHECK(cudaMemset(d_audio_output, 0, batch_size * audio_length * sizeof(float)));

    // Step 3: Execute batched IFFT C2R using cached plan
    std::string plan_key = "istft_" + std::to_string(n_fft) + "_" + std::to_string(batch_size * n_frames);
    cufftHandle plan = get_or_create_plan(plan_key, n_fft, CUFFT_C2R, batch_size * n_frames);

    // Set stream for plan
    CUFFT_CHECK(cufftSetStream(plan, 0));

    // Execute IFFT: stft_input -> ifft_frames
    CUFFT_CHECK(cufftExecC2R(plan, d_stft_input, d_ifft_frames));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Launch overlap-add synthesis kernel
    overlap_add_synthesis_kernel<<<blocks, ISTFT_BLOCK_SIZE>>>(
        d_ifft_frames, d_audio_output, d_window, audio_length, n_frames, n_fft, hop_length
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Launch optimized normalization kernel using precomputed window_sum
    dim3 norm_block(ISTFT_BLOCK_SIZE);
    dim3 norm_grid((audio_length + norm_block.x - 1) / norm_block.x, batch_size);

    normalize_istft_kernel<<<norm_grid, norm_block>>>(
        d_audio_output, d_window_sum, audio_length, batch_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_ifft_frames));
    CUDA_CHECK(cudaFree(d_window_sum));
}

// Helper function to clear cuFFT plan cache (call on cleanup)
void clear_cufft_plan_cache() {
    std::lock_guard<std::mutex> lock(g_plan_cache_mutex);

    for (auto& pair : g_cufft_plan_cache) {
        cufftDestroy(pair.second);
    }
    g_cufft_plan_cache.clear();
}

// Mel-spectrogram kernel optimized for singing voice (44.1kHz)
__global__ void mel_spectrogram_singing_kernel(
    float* audio,                // Input audio [batch_size, audio_length]
    cufftComplex* fft_output,   // FFT workspace [batch_size, n_frames, n_fft/2+1]
    float* mel_output,          // Output mel-spectrogram [batch_size, n_frames, mel_bins]
    float* filterbank,          // Mel filterbank [mel_bins, n_fft/2+1]
    float* window,              // Hann window [n_fft]
    int audio_length,
    int n_fft,
    int hop_length,
    int mel_bins,
    int batch_size
) {
    int frame_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    extern __shared__ float shared_mel_data[];
    float *shared_audio = shared_mel_data;
    float *shared_magnitude = &shared_mel_data[n_fft];

    int frame_start = frame_idx * hop_length;
    int audio_offset = batch_idx * audio_length;

    // Step 1: Windowing - load and apply Hann window
    for (int i = tid; i < n_fft; i += blockDim.x) {
        int audio_idx = frame_start + i;
        float sample = (audio_idx < audio_length) ? __ldg(&audio[audio_offset + audio_idx]) : 0.0f;
        float win_val = __ldg(&window[i]);
        shared_audio[i] = sample * win_val;
    }
    __syncthreads();

    // Step 2: FFT (performed externally via cuFFT on shared_audio)
    // Assume FFT output is in fft_output global memory

    // Step 3: Compute magnitude spectrum
    int n_bins = n_fft / 2 + 1;
    for (int i = tid; i < n_bins; i += blockDim.x) {
        int fft_idx = (batch_idx * ((audio_length - n_fft) / hop_length + 1) + frame_idx) * n_bins + i;
        cufftComplex c = fft_output[fft_idx];
        float mag = sqrtf(c.x * c.x + c.y * c.y);
        shared_magnitude[i] = mag;
    }
    __syncthreads();

    // Step 4: Apply mel filterbank (each thread computes one mel bin)
    if (tid < mel_bins) {
        float mel_sum = 0.0f;

        for (int k = 0; k < n_bins; k++) {
            float filter_val = __ldg(&filterbank[tid * n_bins + k]);
            mel_sum += shared_magnitude[k] * filter_val;
        }

        // Step 5: Perceptual weighting moved to apply_perceptual_weighting_kernel

        // Step 6: Log compression with safe epsilon addition
        float log_mel = logf(mel_sum + EPSILON);

        // Write to global memory
        int mel_idx = (batch_idx * ((audio_length - n_fft) / hop_length + 1) + frame_idx) * mel_bins + tid;
        mel_output[mel_idx] = log_mel;
    }
}

// Apply perceptual weighting (A-weighting) to mel-spectrogram
__global__ void apply_perceptual_weighting_kernel(
    float* mel_spectrogram,     // Input/output mel-spectrogram [batch_size, n_frames, mel_bins]
    float* mel_frequencies,     // Mel bin center frequencies [mel_bins]
    int n_frames,
    int mel_bins,
    int batch_size
) {
    int frame_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || frame_idx >= n_frames) return;

    extern __shared__ float shared_a_weights[];

    // Precompute A-weighting for all mel bins (thread 0)
    if (tid == 0) {
        for (int i = 0; i < mel_bins; i++) {
            float freq = __ldg(&mel_frequencies[i]);
            shared_a_weights[i] = a_weighting_db(freq);
        }
    }
    __syncthreads();

    // Apply A-weighting to mel bins
    if (tid < mel_bins) {
        int mel_idx = (batch_idx * n_frames + frame_idx) * mel_bins + tid;

        // Convert dB to linear scale
        float a_weight_linear = powf(10.0f, shared_a_weights[tid] / 20.0f);

        // Apply weighting (in linear scale, then convert back to log)
        float mel_val = mel_spectrogram[mel_idx];
        float linear_mel = expf(mel_val);  // Convert from log to linear
        linear_mel *= a_weight_linear;
        mel_spectrogram[mel_idx] = logf(linear_mel + EPSILON);
    }
}

// Compute log mel-spectrogram (fused log operation) - Fixed race conditions
__global__ void compute_log_mel_kernel(
    float* mel_spectrogram,     // Input mel-spectrogram [batch_size, n_frames, mel_bins]
    float* log_mel_output,      // Output log-mel [batch_size, n_frames, mel_bins]
    int n_frames,
    int mel_bins,
    int batch_size,
    float epsilon               // Small constant to avoid log(0)
) {
    int total_bins = batch_size * n_frames * mel_bins;

    // Option A: Vectorized implementation with clean partitioning
    // Each thread processes one float4 (4 elements)
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_vec4 = total_bins / 4;

    // Process vectorized portion
    if (vec_idx < num_vec4) {
        int base_idx = vec_idx * 4;

        // Check alignment before using vectorized load/store
        if (reinterpret_cast<uintptr_t>(&mel_spectrogram[base_idx]) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(&log_mel_output[base_idx]) % 16 == 0) {
            float4 mel_vals = *reinterpret_cast<float4*>(&mel_spectrogram[base_idx]);

            float4 log_vals;
            log_vals.x = logf(mel_vals.x + epsilon);
            log_vals.y = logf(mel_vals.y + epsilon);
            log_vals.z = logf(mel_vals.z + epsilon);
            log_vals.w = logf(mel_vals.w + epsilon);

            *reinterpret_cast<float4*>(&log_mel_output[base_idx]) = log_vals;
        } else {
            // Unaligned fallback - process 4 elements scalar
            for (int i = 0; i < 4; i++) {
                int idx = base_idx + i;
                if (idx < total_bins) {
                    log_mel_output[idx] = logf(mel_spectrogram[idx] + epsilon);
                }
            }
        }
    }

    // Tail loop for remainder elements (total_bins % 4)
    int tail_start = num_vec4 * 4;
    int tail_idx = tail_start + threadIdx.x;

    if (blockIdx.x == gridDim.x - 1 && tail_idx < total_bins) {
        log_mel_output[tail_idx] = logf(mel_spectrogram[tail_idx] + epsilon);
    }
}

// Real-time voice conversion kernel with chunk-based processing
__global__ void realtime_voice_conversion_kernel(
    float* audio_chunk,          // Input audio chunk [chunk_size]
    float* overlap_buffer,       // Overlap buffer state [overlap_size]
    cufftComplex* fft_workspace,  // FFT workspace [n_fft/2+1]
    float* features_output,      // Output converted audio or features [chunk_size or feature_dim]
    float* window,               // Window function [n_fft]
    int chunk_size,
    int overlap_size,
    int n_fft,
    int hop_length,
    int feature_dim,
    float* speaker_embedding,    // Optional: speaker embedding [embedding_dim] (nullptr if not provided)
    float* pitch_features,       // Optional: pitch features [feature_dim] (nullptr if not provided)
    int embedding_dim,
    bool perform_conversion      // If true, perform conversion; if false, extract features only
) {
    int tid = threadIdx.x;

    // Use shared memory sized at runtime: [overlap_size + chunk_size]
    extern __shared__ float shared_rt_buffer[];

    // Partition shared memory
    float *shared_overlap = shared_rt_buffer;  // [overlap_size]
    float *shared_chunk = &shared_rt_buffer[overlap_size];  // [chunk_size]
    float *shared_combined = shared_rt_buffer;  // Reuse for combined buffer [overlap_size + chunk_size]

    // Step 1: Load overlap buffer and current chunk into shared memory
    if (tid < overlap_size) {
        shared_overlap[tid] = __ldg(&overlap_buffer[tid]);
    }
    if (tid < chunk_size) {
        shared_chunk[tid] = __ldg(&audio_chunk[tid]);
    }
    __syncthreads();

    // Step 2: Create combined buffer in-place (overlap + chunk)
    // Already laid out correctly in shared memory: [overlap][chunk]
    int combined_size = overlap_size + chunk_size;

    // Guard against OOB access with runtime sizes
    if (combined_size > n_fft) {
        // Handle case where combined buffer exceeds n_fft
        combined_size = n_fft;
    }

    // Step 3: Apply window cooperatively using all threads
    for (int i = tid; i < combined_size && i < n_fft; i += blockDim.x) {
        float win_val = __ldg(&window[i]);
        shared_combined[i] *= win_val;
    }
    __syncthreads();

    // Step 4: FFT would be performed externally on shared_combined
    // If performing conversion, apply speaker embedding and pitch features
    if (perform_conversion && speaker_embedding != nullptr && pitch_features != nullptr) {
        // Simplified conversion: modulate features with speaker embedding
        // In a real implementation, this would be a more complex neural network operation
        if (tid < feature_dim) {
            // Compute content features (spectral envelope)
            float content_feature = 0.0f;
            for (int i = 0; i < n_fft && i < combined_size; i++) {
                content_feature += shared_combined[i];
            }
            content_feature /= (combined_size > 0) ? combined_size : 1.0f;

            // Load speaker embedding (using simple weighted combination)
            float speaker_weight = (tid < embedding_dim) ? speaker_embedding[tid] : 1.0f;

            // Load pitch feature
            float pitch_weight = pitch_features[tid];

            // Apply conversion: content * speaker_embedding * pitch_features
            features_output[tid] = content_feature * speaker_weight * pitch_weight;
        }
    } else {
        // Feature extraction only mode (original behavior)
        // Extract features from FFT output (mel-spectrogram, F0, etc.)
        // Write simple features to output (e.g., windowed frame energy, RMS)
        if (tid < feature_dim) {
            // Compute frame energy as a basic feature
            float energy = 0.0f;
            for (int i = 0; i < n_fft && i < combined_size; i++) {
                float val = shared_combined[i];
                energy += val * val;
            }
            // Guard combined_size > 0 and use combined_size as divisor instead of n_fft
            energy = (combined_size > 0) ? sqrtf(energy / combined_size) : 0.0f;  // RMS energy

            // Write to features output (tid maps to feature index)
            features_output[tid] = energy;
        }
    }
    __syncthreads();

    // Step 5: Update overlap buffer for next iteration
    // Copy last overlap_size samples from current chunk back to global overlap buffer
    if (tid < overlap_size) {
        int src_idx = chunk_size - overlap_size + tid;
        if (src_idx >= 0 && src_idx < chunk_size) {
            overlap_buffer[tid] = shared_chunk[src_idx];
        }
    }
}

// Host function to launch mel-spectrogram for singing
void launch_mel_spectrogram_singing(
    torch::Tensor& audio,
    torch::Tensor& window,
    torch::Tensor& mel_filterbank,
    torch::Tensor& mel_output,
    int n_fft,
    int hop_length,
    bool apply_a_weighting,
    c10::optional<torch::Tensor> mel_frequencies  // CHANGED: use c10::optional for safe null handling
) {
    // Validate inputs
    if (!audio.is_cuda() || !window.is_cuda() || !mel_filterbank.is_cuda() || !mel_output.is_cuda()) {
        throw std::runtime_error("All tensors must be on CUDA device");
    }

    // Validate mel_frequencies if A-weighting is requested
    if (apply_a_weighting && (!mel_frequencies.has_value() || !mel_frequencies->is_cuda())) {
        throw std::runtime_error("mel_frequencies tensor required for A-weighting and must be on CUDA device");
    }

    float *d_audio = audio.data_ptr<float>();
    float *d_window = window.data_ptr<float>();
    float *d_filterbank = mel_filterbank.data_ptr<float>();
    float *d_mel_output = mel_output.data_ptr<float>();

    int batch_size = audio.size(0);
    int audio_length = audio.size(1);
    int n_frames = std::max<int>(0, (audio_length - n_fft) / hop_length + 1);
    if (n_frames == 0) {
        // Zero-fill outputs and return
        CUDA_CHECK(cudaMemset(d_mel_output, 0, mel_output.numel() * sizeof(float)));
        return;
    }
    int mel_bins = mel_filterbank.size(0);

    // Allocate windowed frames buffer [batch_size * n_frames, n_fft]
    float *d_windowed_frames;
    CUDA_CHECK(cudaMalloc(&d_windowed_frames, batch_size * n_frames * n_fft * sizeof(float)));

    // Allocate FFT workspace [batch_size * n_frames, n_fft/2+1]
    cufftComplex *d_fft_output;
    CUDA_CHECK(cudaMalloc(&d_fft_output, batch_size * n_frames * (n_fft / 2 + 1) * sizeof(cufftComplex)));

    // Step 1: Launch windowing kernel once with batched grid covering all frames and batches
    dim3 window_block(256);
    dim3 window_grid(n_frames, batch_size);

    // Single batched kernel launch for all frames and batches
    // blockIdx.x = frame index, blockIdx.y = batch index
    apply_window_kernel<<<window_grid, window_block>>>(
        d_audio,
        d_window,
        d_windowed_frames,
        audio_length,
        n_fft,
        hop_length,
        n_frames
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Execute cuFFT R2C on windowed frames using cached plan
    std::string plan_key = "mel_stft_" + std::to_string(n_fft) + "_" + std::to_string(batch_size * n_frames);
    cufftHandle plan = get_or_create_plan(plan_key, n_fft, CUFFT_R2C, batch_size * n_frames);

    // Set plan to use default stream
    CUFFT_CHECK(cufftSetStream(plan, 0));

    // Execute FFT: windowed_frames -> fft_output
    CUFFT_CHECK(cufftExecR2C(plan, d_windowed_frames, d_fft_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Launch mel kernel to compute mel-spectrogram from FFT output
    dim3 block(MEL_SPECTROGRAM_BLOCK_SIZE);
    dim3 grid(n_frames, batch_size);
    size_t shared_mem = (n_fft + (n_fft / 2 + 1)) * sizeof(float);
    mel_spectrogram_singing_kernel<<<grid, block, shared_mem>>>(
        d_audio, d_fft_output, d_mel_output, d_filterbank, d_window,
        audio_length, n_fft, hop_length, mel_bins, batch_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Apply A-weighting if requested
    if (apply_a_weighting && mel_frequencies.has_value()) {
        apply_perceptual_weighting(mel_output, mel_frequencies.value(), n_frames, mel_bins, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_windowed_frames));
    CUDA_CHECK(cudaFree(d_fft_output));
}

// Host function to launch real-time voice conversion
void launch_realtime_voice_conversion(
    torch::Tensor& audio_chunk,
    torch::Tensor& overlap_buffer,
    torch::Tensor& features_output,
    torch::Tensor& window,
    int n_fft,
    int hop_length,
    c10::optional<torch::Tensor> speaker_embedding,
    c10::optional<torch::Tensor> pitch_features
) {
    // Validate inputs
    if (!audio_chunk.is_cuda() || !overlap_buffer.is_cuda() || !features_output.is_cuda() || !window.is_cuda()) {
        throw std::runtime_error("All tensors must be on CUDA device");
    }

    // Validate optional tensors if provided
    bool perform_conversion = speaker_embedding.has_value() && pitch_features.has_value();
    if (perform_conversion) {
        if (!speaker_embedding->is_cuda() || !pitch_features->is_cuda()) {
            throw std::runtime_error("speaker_embedding and pitch_features must be on CUDA device when provided");
        }
    }

    float *d_chunk = audio_chunk.data_ptr<float>();
    float *d_overlap = overlap_buffer.data_ptr<float>();
    float *d_features = features_output.data_ptr<float>();
    float *d_window = window.data_ptr<float>();

    int chunk_size = audio_chunk.size(0);
    int overlap_size = overlap_buffer.size(0);
    int feature_dim = features_output.size(0);

    // Get optional tensor pointers
    float *d_speaker_embedding = nullptr;
    float *d_pitch_features = nullptr;
    int embedding_dim = 0;

    if (perform_conversion) {
        d_speaker_embedding = speaker_embedding->data_ptr<float>();
        d_pitch_features = pitch_features->data_ptr<float>();
        embedding_dim = speaker_embedding->size(0);
    }

    // Allocate FFT workspace
    cufftComplex *d_fft_workspace;
    CUDA_CHECK(cudaMalloc(&d_fft_workspace, (n_fft / 2 + 1) * sizeof(cufftComplex)));

    // Compute shared memory dynamically based on runtime sizes
    size_t shared_mem = (chunk_size + overlap_size) * sizeof(float);

    // Launch kernel with single block (low-latency processing)
    dim3 block(REALTIME_CONVERSION_BLOCK_SIZE);
    dim3 grid(1);

    realtime_voice_conversion_kernel<<<grid, block, shared_mem>>>(
        d_chunk, d_overlap, d_fft_workspace, d_features, d_window,
        chunk_size, overlap_size, n_fft, hop_length, feature_dim,
        d_speaker_embedding, d_pitch_features, embedding_dim, perform_conversion
    );
    CUDA_CHECK(cudaGetLastError());

    // Free workspace
    CUDA_CHECK(cudaFree(d_fft_workspace));
}

// Host function to apply perceptual weighting (A-weighting)
void apply_perceptual_weighting(
    torch::Tensor& mel_spectrogram,
    torch::Tensor& mel_frequencies,
    int n_frames,
    int mel_bins,
    int batch_size
) {
    // Validate inputs
    if (!mel_spectrogram.is_cuda() || !mel_frequencies.is_cuda()) {
        throw std::runtime_error("All tensors must be on CUDA device");
    }

    float *d_mel = mel_spectrogram.data_ptr<float>();
    float *d_freqs = mel_frequencies.data_ptr<float>();

    // Launch perceptual weighting kernel
    dim3 block(256);
    dim3 grid(n_frames, batch_size);
    size_t shared_mem = mel_bins * sizeof(float);  // For A-weighting cache

    apply_perceptual_weighting_kernel<<<grid, block, shared_mem>>>(
        d_mel, d_freqs, n_frames, mel_bins, batch_size
    );
    CUDA_CHECK(cudaGetLastError());
}
