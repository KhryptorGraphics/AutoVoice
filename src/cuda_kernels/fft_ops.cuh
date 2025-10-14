#ifndef FFT_OPS_CUH
#define FFT_OPS_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

// FFT operation constants
#define FFT_SIZE 2048
#define HOP_SIZE 512
#define WINDOW_SIZE 2048
#define MEL_BINS 128
#define SAMPLE_RATE 22050

// Kernel declarations for FFT operations
__global__ void apply_window_kernel(float* signal, float* window, float* output, int window_size, int batch_size);
__global__ void compute_magnitude_kernel(cufftComplex* complex_data, float* magnitude, int fft_size, int batch_size);
__global__ void compute_phase_kernel(cufftComplex* complex_data, float* phase, int fft_size, int batch_size);
__global__ void mel_filterbank_kernel(float* magnitude, float* mel_output, float* filterbank,
                                     int fft_size, int mel_bins, int batch_size);

// Helper functions for FFT operations
inline __device__ float hann_window(int n, int N) {
    return 0.5f * (1.0f - cosf(2.0f * M_PI * n / (N - 1)));
}

inline __device__ float hamming_window(int n, int N) {
    return 0.54f - 0.46f * cosf(2.0f * M_PI * n / (N - 1));
}

// Mel scale conversion functions
inline __device__ float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

inline __device__ float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Shared memory configurations for different kernels
#define STFT_SHARED_MEM_SIZE 4096
#define MEL_SHARED_MEM_SIZE 2048

// Error checking macro for CUFFT
#define CUFFT_CHECK(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            printf("CUFFT error: %d at %s:%d\\n", result, __FILE__, __LINE__); \
        } \
    } while(0)

#endif // FFT_OPS_CUH