# CUDA C++ Style Guide

## Target Platform

- **Device**: Jetson Thor (SM 11.0, sm_110)
- **CUDA**: 13.0
- **Architecture**: aarch64

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Kernel functions | snake_case with `_kernel` suffix | `mel_spectrogram_kernel` |
| Device functions | snake_case with `__device__` | `interpolate_frame` |
| Host wrappers | snake_case, no suffix | `compute_mel_spectrogram` |
| Constants | UPPER_SNAKE | `NUM_MEL_BANDS` |
| Template params | PascalCase | `typename FloatType` |

## Kernel Design

- One kernel per `.cu` file when possible
- Host wrapper in corresponding `.cuh` header
- Always specify launch bounds: `__launch_bounds__(BLOCK_SIZE)`
- Use shared memory for data reuse patterns
- Prefer warp-level primitives (`__shfl_sync`) over shared memory for small data

```cuda
__global__ void __launch_bounds__(256)
mel_spectrogram_kernel(
    const float* __restrict__ audio,
    float* __restrict__ mel_output,
    int n_frames,
    int n_mels
) {
    // ...
}
```

## Memory Management

- Use pinned memory (`cudaMallocHost`) for host-device transfers
- Prefer unified memory only for prototyping, explicit copies for production
- Always check allocation errors: `CUDA_CHECK(cudaMalloc(...))`
- Free resources in reverse allocation order

## Error Handling

- Use a `CUDA_CHECK` macro that throws on error:

```cuda
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            throw std::runtime_error(                             \
                std::string("CUDA error: ") +                     \
                cudaGetErrorString(err) +                         \
                " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                         \
    } while(0)
```

## Performance Guidelines

- Target SM 11.0 occupancy (check with `--ptxas-options=-v`)
- Avoid warp divergence in inner loops
- Coalesce global memory accesses (sequential threads access sequential addresses)
- Use `__restrict__` on all pointer parameters
- Prefer `float` over `double` for audio processing (FP16 for inference)
- Use `__ldg()` for read-only global memory on SM 11.0

## PyTorch Integration

- Use `torch/extension.h` for Python bindings
- Register kernels via `PYBIND11_MODULE`
- Accept `torch::Tensor` parameters, check device and dtype
- Use `AT_DISPATCH_FLOATING_TYPES_AND_HALF` for mixed precision support

## Build

- Compile with: `nvcc -arch=sm_110 -O3`
- Use `setup.py` with `torch.utils.cpp_extension.CUDAExtension`
- Always build from source (never use pre-built wheels on Jetson)
