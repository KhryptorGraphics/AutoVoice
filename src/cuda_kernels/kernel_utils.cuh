#ifndef KERNEL_UTILS_CUH
#define KERNEL_UTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cufft.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdint.h>

// CUDA Error Checking Macro (PyTorch-compatible)
#ifdef __CUDACC__
  // In device code, just print error
  #define CUDA_CHECK(call) { \
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
          printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      } \
  }
#else
  // In host code, use PyTorch error handling
  #include <c10/util/Exception.h>
  #define CUDA_CHECK(call) { \
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
          TORCH_CHECK(false, "CUDA error at ", __FILE__, ":", __LINE__, ": ", cudaGetErrorString(err)); \
      } \
  }
#endif

// cublas Error Checking (PyTorch-compatible)
#ifdef __CUDACC__
  #define CUBLAS_CHECK(call) { \
      cublasStatus_t status = call; \
      if (status != CUBLAS_STATUS_SUCCESS) { \
          printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
      } \
  }
#else
  #include <c10/util/Exception.h>
  #define CUBLAS_CHECK(call) { \
      cublasStatus_t status = call; \
      if (status != CUBLAS_STATUS_SUCCESS) { \
          TORCH_CHECK(false, "cuBLAS error at ", __FILE__, ":", __LINE__, ": ", status); \
      } \
  }
#endif

// cuFFT Error Checking (PyTorch-compatible)
#ifdef __CUDACC__
  #define CUFFT_CHECK(call) { \
      cufftResult_t err = call; \
      if (err != CUFFT_SUCCESS) { \
          printf("cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, err); \
      } \
  }
#else
  #include <c10/util/Exception.h>
  #define CUFFT_CHECK(call) { \
      cufftResult_t err = call; \
      if (err != CUFFT_SUCCESS) { \
          TORCH_CHECK(false, "cuFFT error at ", __FILE__, ":", __LINE__, ": ", err); \
      } \
  }
#endif

// cuSPARSE Error Checking (PyTorch-compatible)
#ifdef __CUDACC__
  #define CUSPARSE_CHECK(call) { \
      cusparseStatus_t status = call; \
      if (status != CUSPARSE_STATUS_SUCCESS) { \
          printf("cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, status); \
      } \
  }
#else
  #include <c10/util/Exception.h>
  #define CUSPARSE_CHECK(call) { \
      cusparseStatus_t status = call; \
      if (status != CUSPARSE_STATUS_SUCCESS) { \
          TORCH_CHECK(false, "cuSPARSE error at ", __FILE__, ":", __LINE__, ": ", status); \
      } \
  }
#endif

// Common device functions
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float relu(float x) {
    return fmaxf(x, 0.0f);
}

__device__ __forceinline__ float gelu(float x) {
    float half = 0.5f * x;
    float cdff = 0.5f * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    return half * cdff;
}

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_log(float x) {
    return logf(x);
}

// Memory alignment macros
#define ALIGN_UP(value, alignment) (((value) + (alignment) - 1) & ~((alignment) - 1))
#define ALIGN_16(value) ALIGN_UP(value, 16)

// Cooperative groups utilities
namespace cg = cooperative_groups;

__device__ __forceinline__ void block_sync() {
    __syncthreads();
}

__device__ __forceinline__ void grid_sync() {
    cg::grid_group g = cg::this_grid();
    g.sync();
}

// Use native atomic add for float (available on modern GPUs)
__device__ __forceinline__ float atomic_add_float(float* address, float val) {
    return atomicAdd(address, val);
}

// Thread index utilities
__device__ __forceinline__ int thread_idx_x() { return threadIdx.x; }
__device__ __forceinline__ int block_idx_x() { return blockIdx.x; }
__device__ __forceinline__ int block_dim_x() { return blockDim.x; }
__device__ __forceinline__ int grid_dim_x() { return gridDim.x; }

// Global thread ID
__device__ __forceinline__ int global_thread_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Warp size
static const int WARP_SIZE = 32;

// Warp shuffle functions for fast reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-wide reduction using shared memory
__device__ __forceinline__ float block_reduce_sum(float val) {
    // Shared memory for reduction (one value per warp)
    __shared__ float warp_sums[32]; // Maximum 32 warps per block (1024 threads)

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // First reduce within each warp
    val = warp_reduce_sum(val);

    // Write reduced value to shared memory (only first thread in each warp)
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Only the first warp performs the final reduction
    if (warp_id == 0) {
        // Read from shared memory only if within valid range
        val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        // Final warp reduction
        val = warp_reduce_sum(val);
    }

    return val;
}

// Common constants
static const float PI = 3.141592653589793f;
static const float EPSILON = 1e-8f;

// Hann window device function
__device__ __forceinline__ void hann_window(float* window, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float arg = 2.0f * PI * i / (n - 1.0f);
        window[i] = 0.5f * (1.0f - cosf(arg));
    }
    __syncthreads();
}

// Safe division
__device__ __forceinline__ float safe_divide(float a, float b) {
    return b > EPSILON ? a / b : 0.0f;
}

// Clamp function
__device__ __forceinline__ float clamp(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

// ============================================================================
// PHASE 1: LPC Helper Functions for Formant Extraction
// ============================================================================

// Levinson-Durbin recursion for LPC coefficients
// Solves Toeplitz system: R * a = r
__device__ void levinson_durbin(
    float *autocorr,      // Input: autocorrelation [order+1]
    float *lpc_coeffs,    // Output: LPC coefficients [order]
    int order,
    float *error          // Output: prediction error
) {
    float k[32];  // Reflection coefficients (max order 32)
    float a_prev[32], a_curr[32];

    // Initialize
    float E = autocorr[0];
    if (E < EPSILON) {
        *error = 0.0f;
        for (int i = 0; i < order; i++) {
            lpc_coeffs[i] = 0.0f;
        }
        return;
    }

    // Levinson-Durbin recursion
    for (int i = 0; i < order; i++) {
        // Compute reflection coefficient k[i]
        float sum = autocorr[i + 1];
        for (int j = 0; j < i; j++) {
            sum += a_prev[j] * autocorr[i - j];
        }
        k[i] = -sum / E;

        // Update LPC coefficients
        a_curr[i] = k[i];
        for (int j = 0; j < i; j++) {
            a_curr[j] = a_prev[j] + k[i] * a_prev[i - 1 - j];
        }

        // Update prediction error
        E = E * (1.0f - k[i] * k[i]);

        // Copy current to previous for next iteration
        for (int j = 0; j <= i; j++) {
            a_prev[j] = a_curr[j];
        }
    }

    // Copy final coefficients to output
    for (int i = 0; i < order; i++) {
        lpc_coeffs[i] = a_curr[i];
    }
    *error = E;
}

// Compute autocorrelation for LPC
__device__ void compute_autocorrelation(
    float *signal,        // Input: audio frame [frame_length]
    float *autocorr,      // Output: autocorrelation [order+1]
    int frame_length,
    int order
) {
    for (int lag = 0; lag <= order; lag++) {
        float sum = 0.0f;
        for (int i = 0; i < frame_length - lag; i++) {
            sum += signal[i] * signal[i + lag];
        }
        autocorr[lag] = sum / (float)frame_length;
    }
}

// Find roots of LPC polynomial using Durand-Kerner method
// Constructs polynomial P(z) = z^order + Σ_{k=1..order} a[k] z^{order-k}
__device__ void find_polynomial_roots(
    float *lpc_coeffs,    // Input: LPC coefficients [order]
    float *roots_real,    // Output: real parts of roots [order]
    float *roots_imag,    // Output: imaginary parts of roots [order]
    int order,
    int max_iterations
) {
    // Simplified Durand-Kerner method for polynomial root finding
    // Initialize roots uniformly on unit circle
    for (int i = 0; i < order; i++) {
        float angle = 2.0f * PI * i / order;
        roots_real[i] = cosf(angle) * 0.9f;
        roots_imag[i] = sinf(angle) * 0.9f;
    }

    // Iterate to refine roots
    for (int iter = 0; iter < max_iterations; iter++) {
        float max_change = 0.0f;

        for (int i = 0; i < order; i++) {
            // Evaluate polynomial P(z) = z^order + Σ_{k=1..order} a[k] z^{order-k} at z_i
            // Using Horner's method for numerical stability
            float z_real = roots_real[i], z_imag = roots_imag[i];

            // Start with z^order (implicitly coefficient 1.0)
            float p_real = 1.0f, p_imag = 0.0f;

            // Horner evaluation: P(z) = (((...((1)*z + a[0])*z + a[1])*z + ...)*z + a[order-1])
            for (int k = 0; k < order; k++) {
                // Multiply by z: (p_real + i*p_imag) * (z_real + i*z_imag)
                float temp_real = p_real * z_real - p_imag * z_imag;
                float temp_imag = p_real * z_imag + p_imag * z_real;

                // Add coefficient a[k]
                p_real = temp_real + lpc_coeffs[k];
                p_imag = temp_imag;
            }

            // Compute product of differences with other roots: Π_{j≠i}(z_i - z_j)
            float prod_real = 1.0f, prod_imag = 0.0f;
            for (int j = 0; j < order; j++) {
                if (j != i) {
                    float diff_real = roots_real[i] - roots_real[j];
                    float diff_imag = roots_imag[i] - roots_imag[j];
                    float temp_real = prod_real * diff_real - prod_imag * diff_imag;
                    float temp_imag = prod_real * diff_imag + prod_imag * diff_real;
                    prod_real = temp_real;
                    prod_imag = temp_imag;
                }
            }

            // Durand-Kerner update: z_i := z_i - P(z_i) / Π_{j≠i}(z_i - z_j)
            float denom = prod_real * prod_real + prod_imag * prod_imag;
            if (denom > EPSILON) {
                float update_real = (p_real * prod_real + p_imag * prod_imag) / denom;
                float update_imag = (p_imag * prod_real - p_real * prod_imag) / denom;

                roots_real[i] -= update_real;
                roots_imag[i] -= update_imag;

                float change = sqrtf(update_real * update_real + update_imag * update_imag);
                max_change = fmaxf(max_change, change);
            }
        }

        // Check convergence
        if (max_change < 1e-6f) break;
    }
}

// ============================================================================
// PHASE 1: STFT/ISTFT Helper Functions
// ============================================================================

// Compute window normalization sum for ISTFT
__device__ float compute_window_sum(
    float *window,        // Input: synthesis window [n_fft]
    int n_fft,
    int hop_length,
    int sample_position,  // Current sample position in output
    int n_frames
) {
    float sum = 0.0f;

    for (int frame = 0; frame < n_frames; frame++) {
        int frame_start = frame * hop_length;
        int frame_end = frame_start + n_fft;

        if (sample_position >= frame_start && sample_position < frame_end) {
            int window_idx = sample_position - frame_start;
            float w = window[window_idx];
            sum += w * w;
        }
    }

    return sum;
}

// Complex multiplication for STFT operations
__device__ __forceinline__ cufftComplex complex_mul(
    cufftComplex a,
    cufftComplex b
) {
    cufftComplex result;
    result.x = a.x * b.x - a.y * b.y;
    result.y = a.x * b.y + a.y * b.x;
    return result;
}

// Complex conjugate
__device__ __forceinline__ cufftComplex complex_conj(cufftComplex a) {
    cufftComplex result;
    result.x = a.x;
    result.y = -a.y;
    return result;
}

// ============================================================================
// PHASE 1: Perceptual Weighting Helper Functions
// ============================================================================

// Compute A-weighting for frequency bin
__device__ float compute_a_weighting(float frequency_hz) {
    // A-weighting formula
    float f2 = frequency_hz * frequency_hz;
    float f4 = f2 * f2;

    float numerator = 12194.0f * 12194.0f * f4;
    float denominator = (f2 + 20.6f * 20.6f) *
                       sqrtf((f2 + 107.7f * 107.7f) * (f2 + 737.9f * 737.9f)) *
                       (f2 + 12194.0f * 12194.0f);

    float a_weight = sqrtf(numerator / (denominator + EPSILON));
    float a_weight_db = 20.0f * log10f(a_weight + EPSILON) + 2.0f;

    return powf(10.0f, a_weight_db / 20.0f);
}

// ============================================================================
// PHASE 1: Streaming Helper Functions
// ============================================================================

// Update overlap buffer for next chunk
__device__ void update_overlap_buffer(
    float *current_chunk,  // Input: current processed chunk
    float *overlap_buffer, // Output: overlap for next chunk
    int chunk_size,
    int overlap_size
) {
    int tid = threadIdx.x;

    // Copy last overlap_size samples to overlap buffer
    if (tid < overlap_size) {
        int src_idx = chunk_size - overlap_size + tid;
        overlap_buffer[tid] = current_chunk[src_idx];
    }
}

// ============================================================================
// PHASE 1: Optimization Macros
// ============================================================================

// Optimal block sizes for different kernels (determined by profiling)
#define PITCH_DETECTION_BLOCK_SIZE 256
#define MEL_SPECTROGRAM_BLOCK_SIZE 256
#define STFT_BLOCK_SIZE 256
#define ISTFT_BLOCK_SIZE 128
#define FORMANT_EXTRACTION_BLOCK_SIZE 128
#define REALTIME_CONVERSION_BLOCK_SIZE 256

// Shared memory sizes
#define PITCH_SHARED_MEM_SIZE (2048 + 534)  // frame + history + autocorr
#define MEL_SHARED_MEM_SIZE (2048 + 1024)   // frame + filterbank
#define STFT_SHARED_MEM_SIZE 4096
#define ISTFT_SHARED_MEM_SIZE 4096
#define FORMANT_SHARED_MEM_SIZE 2048

// Performance profiling (enabled with -DENABLE_PROFILING)
#ifdef ENABLE_PROFILING
  #define PROFILE_START(name) \
      cudaEvent_t start_##name, stop_##name; \
      cudaEventCreate(&start_##name); \
      cudaEventCreate(&stop_##name); \
      cudaEventRecord(start_##name);

  #define PROFILE_END(name) \
      cudaEventRecord(stop_##name); \
      cudaEventSynchronize(stop_##name); \
      float ms_##name = 0; \
      cudaEventElapsedTime(&ms_##name, start_##name, stop_##name); \
      printf(#name " took %.3f ms\n", ms_##name); \
      cudaEventDestroy(start_##name); \
      cudaEventDestroy(stop_##name);
#else
  #define PROFILE_START(name)
  #define PROFILE_END(name)
#endif

#endif // KERNEL_UTILS_CUH