#include "kernel_utils.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstring>

// Prevent cublasLt.h extern "C" error during device code compilation (CUDA 13.0+)
#ifdef __CUDA_ARCH__
#define CUBLASLT_H_
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using namespace cooperative_groups;

// Ring buffer enqueue kernel
__global__ void ring_buffer_enqueue_kernel(float *ring_buffer, float *input, int *head, int capacity, int size, int offset) {
    int tid = threadIdx.x;
    int buffer_head = atomicAdd(head, size);
    buffer_head %= capacity;
    
    if (tid < size) {
        int idx = (buffer_head + tid + offset) % capacity;
        ring_buffer[idx] = input[tid];
    }
}

// Ring buffer dequeue kernel
__global__ void ring_buffer_dequeue_kernel(float *ring_buffer, float *output, int *tail, int capacity, int size) {
    int tid = threadIdx.x;
    int buffer_tail = *tail;
    
    if (tid < size) {
        int idx = (buffer_tail + tid) % capacity;
        output[tid] = ring_buffer[idx];
    }
    
    if (tid == 0) {
        atomicAdd(tail, size);
        *tail %= capacity;
    }
}

// Device-to-device memory copy kernel
__global__ void async_memory_copy_kernel(float *dst, float *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];  // D2D copy via kernel
    }
}

// Memory pooling allocation kernel (placeholder for pool management)
__global__ void memory_pool_allocate_kernel(int *pool, int *pool_head, int pool_size, int request_size, int *allocated_idx) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int head = atomicAdd(pool_head, request_size);
        if (head + request_size <= pool_size) {
            *allocated_idx = head;
        } else {
            *allocated_idx = -1; // Out of memory
        }
    }
}

// Unified memory prefetch kernel
__global__ void unified_memory_prefetch_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx]; // Touch memory to prefetch
    }
}

// Memory garbage collection kernel (mark and sweep simulation)
__global__ void memory_garbage_collect_kernel(int *allocation_map, int *free_list, int *free_head, int num_allocs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_allocs) {
        if (allocation_map[idx] == 0) { // Free slot
            int free_slot = atomicAdd(free_head, 1);
            free_list[free_slot] = idx;
        }
    }
}

// Pinned memory buffer management kernel
__global__ void pinned_memory_buffer_kernel(float *pinned_buffer, float *device_buffer, int n, int direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (direction == 0) { // H2D
            device_buffer[idx] = pinned_buffer[idx];
        } else { // D2H
            pinned_buffer[idx] = device_buffer[idx];
        }
    }
}

// Simple memory copy kernel for CUDA graph capture
__global__ void memory_copy_kernel(float *dst, float *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Host function for stream synchronization
void launch_stream_synchronize(uintptr_t stream_id) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Host function for async memory copy with enhanced pinned memory checks
void launch_async_memory_copy(torch::Tensor& dst, torch::Tensor& src, uintptr_t stream_id) {
    // Add validation checks
    TORCH_CHECK(dst.dtype() == torch::kFloat32 && src.dtype() == torch::kFloat32, "Expected float32 tensors");
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(dst.numel() == src.numel(), "dst and src must have the same number of elements");

    float *d_dst = dst.data_ptr<float>();
    float *d_src = src.data_ptr<float>();
    int n = src.numel();

    // Get stream: use current stream if stream_id is 0, otherwise use provided stream
    cudaStream_t stream;
    if (stream_id == 0) {
        // Use current CUDA stream from PyTorch
        stream = at::cuda::getCurrentCUDAStream();
    } else {
        stream = reinterpret_cast<cudaStream_t>(stream_id);
    }

    // Check if both tensors are on same device type
    bool dst_is_cuda = dst.is_cuda();
    bool src_is_cuda = src.is_cuda();

    if (dst_is_cuda && src_is_cuda) {
        // Validate devices for cross-device copies
        int dst_device = dst.get_device();
        int src_device = src.get_device();
        if (dst_device != src_device) {
            fprintf(stderr, "Warning: Cross-device D2D copy (device %d -> device %d). Use peer-to-peer if available.\n",
                    src_device, dst_device);
        }
        // Device-to-device copy: use cudaMemcpyAsync for better performance
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else if (!src_is_cuda && dst_is_cuda) {
        // Host-to-device copy: check if host tensor is pinned
        if (!src.is_pinned()) {
            fprintf(stderr, "Warning: Host tensor is not pinned. H2D async copy may not overlap with computation.\n");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    } else if (src_is_cuda && !dst_is_cuda) {
        // Device-to-host copy: check if host tensor is pinned
        if (!dst.is_pinned()) {
            fprintf(stderr, "Warning: Host tensor is not pinned. D2H async copy may not overlap with computation.\n");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else {
        // Host-to-host copy: use std::memcpy (cudaMemcpyAsync not valid for H2H)
        std::memcpy(d_dst, d_src, n * sizeof(float));
    }
}

// Global CUDA graph handle (simplified - in real code, manage graph pool)
static cudaGraph_t cuda_graph = nullptr;
static cudaGraphExec_t cuda_graph_exec = nullptr;
// Static globals for reuse across graph captures
static float *d_dummy_input = nullptr;
static float *d_dummy_output = nullptr;

// Host function for creating CUDA graph (captures inference operations)
void launch_create_cuda_graph() {
    // Guard: if an existing graph exists, destroy it before recreating
    if (cuda_graph_exec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
        cuda_graph_exec = nullptr;
    }
    if (cuda_graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(cuda_graph));
        cuda_graph = nullptr;
    }

    // Allocate buffers BEFORE capture if not already allocated
    size_t dummy_size = 1024 * sizeof(float);
    if (d_dummy_input == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_dummy_input, dummy_size));
    }
    if (d_dummy_output == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_dummy_output, dummy_size));
    }

    // Create a dedicated stream for graph capture
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    // Begin graph capture on the stream
    cudaStreamCaptureStatus captureStatus;
    CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));

    // Capture only the kernel launch
    dim3 block(256);
    dim3 grid(4);
    memory_copy_kernel<<<grid, block, 0, capture_stream>>>(d_dummy_output, d_dummy_input, 1024);

    // End capture and create graph
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &cuda_graph));

    // Instantiate the graph for execution with error checking
    cudaGraphExecUpdateResult updateResult;
    cudaError_t err = cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to instantiate CUDA graph: %s\n", cudaGetErrorString(err));
        cuda_graph_exec = nullptr;
    }

    // Cleanup stream only (not buffers)
    cudaStreamDestroy(capture_stream);
}

// Host function for destroying CUDA graph and freeing resources
void launch_destroy_cuda_graph() {
    if (cuda_graph_exec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
        cuda_graph_exec = nullptr;
    }
    if (cuda_graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(cuda_graph));
        cuda_graph = nullptr;
    }
    if (d_dummy_input != nullptr) {
        CUDA_CHECK(cudaFree(d_dummy_input));
        d_dummy_input = nullptr;
    }
    if (d_dummy_output != nullptr) {
        CUDA_CHECK(cudaFree(d_dummy_output));
        d_dummy_output = nullptr;
    }
}

// Host function for executing CUDA graph
void launch_execute_cuda_graph() {
    if (cuda_graph_exec != nullptr) {
        cudaStream_t exec_stream;
        CUDA_CHECK(cudaStreamCreate(&exec_stream));
        CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, exec_stream));
        CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        cudaStreamDestroy(exec_stream);
    } else {
        // Graph not created yet - need to call launch_create_cuda_graph first
        fprintf(stderr, "Warning: CUDA graph not instantiated. Call launch_create_cuda_graph first.\n");
    }
}