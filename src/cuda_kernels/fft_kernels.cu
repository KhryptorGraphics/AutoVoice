
#include "kernel_utils.cuh"
#include "fft_ops.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>

using namespace cooperative_groups;

// Windowing kernel using provided window tensor
__global__ void apply_window_kernel(float *audio, float *window, float *windowed, int n_samples, int n_fft, int hop_length) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    int frame_start = frame_idx * hop_length;

    if (tid < n_fft) {
        float w = window[tid];
        int audio_idx = frame_start + tid;
        float sample = (audio_idx < n_samples) ? audio[audio_idx] : 0.0f;
        windowed[frame_idx * n_fft + tid] = sample * w;
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

// Inverse FFT for audio reconstruction
__global__ void ifft_reconstruction_kernel(cufftComplex *spectrum, float *audio_out, int n_samples, int hop_length) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    int frame_start = frame_idx * hop_length;
    
    if (frame_start + tid >= n_samples) return;
    
    // Simplified overlap-add for ISTFT
    float real_part = spectrum[frame_idx * (n_samples / 2 + 1) + tid / 2].x; // Placeholder
    float imag_part = spectrum[frame_idx * (n_samples / 2 + 1) + tid / 2].y;
    audio_out[frame_start + tid] += real_part / (float)hop_length; // Simplified
}

#include <torch/extension.h>

// Host function to apply window
void launch_apply_window(torch::Tensor& input, torch::Tensor& window, torch::Tensor& output) {
    float *d_input = input.data_ptr<float>();
    float *d_window = window.data_ptr<float>();
    float *d_output = output.data_ptr<float>();

    int n_samples = input.size(0);
    int n_fft = window.size(0);
    int hop_length = 256;  // Default

    int n_frames = (n_samples - n_fft) / hop_length + 1;
    dim3 block(256);
    dim3 grid(n_frames);

    apply_window_kernel<<<grid, block>>>(d_input, d_window, d_output, n_samples, n_fft, hop_length);
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

// Host function to execute forward FFT
void execute_cufft_forward(float *d_input, cufftComplex *d_output, int batch_size, int n_fft) {
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, batch_size));
    CUFFT_CHECK(cufftExecR2C(plan, d_input, d_output));
    CUFFT_CHECK(cufftDestroy(plan));
}

// Host function to execute inverse FFT
void execute_cufft_inverse(cufftComplex *d_input, float *d_output, int batch_size, int n_fft) {
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_C2R, batch_size));
    CUFFT_CHECK(cufftExecC2R(plan, d_input, d_output));
    CUFFT_CHECK(cufftDestroy(plan));

    // Normalize
    int total_samples = batch_size * n_fft;
    dim3 block(256);
    dim3 grid((total_samples + block.x - 1) / block.x);

    // Use a proper normalization kernel instead of device lambda
    // Launch a separate kernel for normalization if needed
    CUDA_CHECK(cudaGetLastError());
}

// Normalization kernel
__global__ void normalize_kernel(float *data, int n_fft, int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;
    data[idx] /= (float)n_fft;
}