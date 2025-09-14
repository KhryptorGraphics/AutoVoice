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

#endif // KERNEL_UTILS_CUH