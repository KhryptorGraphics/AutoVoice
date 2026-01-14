#include "kernel_utils.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <cstdint>   // GCC 14 compatibility
#include <cstddef>   // GCC 14 compatibility

// Prevent cublasLt.h extern "C" error during device code compilation (CUDA 13.0+)
#ifdef __CUDA_ARCH__
#define CUBLASLT_H_
#endif

#include <torch/extension.h>

// C libraries with extern "C" - only include for host code
#ifndef __CUDA_ARCH__
#include <cublas_v2.h>
#include <cusparse.h>
#endif

using namespace cooperative_groups;

// Fused multiply-add kernel for efficient computation
__global__ void fused_multiply_add_kernel(float *A, float *B, float *C, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float mul = A[idx] * B[idx];
    output[idx] = mul + C[idx];
}

// Fused layer normalization kernel with GELU activation
__global__ void fused_layer_norm_gelu_kernel(float *input, float *gamma, float *beta, float *mean, float *var, float *output, int n_features, int n_samples, float eps = 1e-5f) {
    int sample_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    
    if (sample_idx >= n_samples || feature_idx >= n_features) return;
    
    int idx = sample_idx * n_features + feature_idx;
    
    // Compute mean and variance (simplified, in practice use reduction)
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n_features; i++) {
        float x = input[sample_idx * n_features + i];
        sum += x;
        sum_sq += x * x;
    }
    __syncthreads();
    
    float m = sum / n_features;
    float v = sum_sq / n_features - m * m;
    float std = sqrtf(v + eps);  // Add epsilon for numerical stability
    
    mean[sample_idx] = m;
    var[sample_idx] = v;
    
    // Layer norm
    float normalized = (input[idx] - m) / std;
    float ln = gamma[feature_idx] * normalized + beta[feature_idx];
    
    // GELU activation
    output[idx] = gelu(ln);
}

// Attention mechanism kernel (simplified self-attention)
__global__ void self_attention_kernel(float *Q, float *K, float *V, float *output, float *attention_weights, int seq_len, int d_model, int n_heads) {
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (head_idx >= n_heads || seq_idx >= seq_len) return;
    
    int d_k = d_model / n_heads;
    
    // Compute attention scores
    float score = 0.0f;
    for (int k = 0; k < d_k; k++) {
        score += Q[seq_idx * d_model + head_idx * d_k + k] * K[seq_idx * d_model + head_idx * d_k + k];
    }
    score /= sqrtf((float)d_k);
    attention_weights[seq_idx * seq_len + seq_idx] = score; // Simplified, no softmax
    
    // Apply attention to V
    float attn_val = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        attn_val += attention_weights[seq_idx * seq_len + i] * V[i * d_model + head_idx * d_k + threadIdx.x % d_k];
    }
    output[seq_idx * d_model + head_idx * d_k + threadIdx.x % d_k] = attn_val;
}

// Fused gradient computation kernel for backprop
__global__ void fused_gradient_kernel(float *grad_output, float *input, float *grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Simplified gradient for ReLU
    grad_input[idx] = grad_output[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
}

// Grouped Query Attention kernel for efficiency
__global__ void grouped_query_attention_kernel(float *Q, float *K, float *V, float *output, int seq_len, int d_model, int n_heads, int n_kv_heads) {
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (head_idx >= n_heads || seq_idx >= seq_len) return;
    
    int d_k = d_model / n_heads;
    int kv_group_size = n_heads / n_kv_heads;
    
    // Compute scores within group
    float score = 0.0f;
    for (int k = 0; k < d_k; k++) {
        score += Q[seq_idx * d_model + head_idx * d_k + k] * K[seq_idx * d_model + (head_idx / kv_group_size) * d_k + k];
    }
    score /= sqrtf((float)d_k);
    
    // Softmax placeholder (simplified max)
    float max_score = score;
    for (int i = 0; i < seq_len; i++) {
        // Assume scores computed
        max_score = fmaxf(max_score, score);
    }
    
    // Weighted sum
    float attn_val = 0.0f;
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        float exp_score = fast_exp(score - max_score);
        sum_exp += exp_score;
        attn_val += exp_score * V[i * d_model + (head_idx / kv_group_size) * d_k + threadIdx.x % d_k];
    }
    attn_val /= sum_exp;
    output[seq_idx * d_model + head_idx * d_k + threadIdx.x % d_k] = attn_val;
}

// Fused operation for attention + add & norm
__global__ void fused_attention_add_norm_kernel(float *attention_out, float *residual, float *gamma, float *beta, float *norm_out, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * d_model) return;
    
    // Add & norm
    float added = attention_out[idx] + residual[idx];
    
    // Layer norm (simplified, mean/var precomputed or reduced)
    float mean = 0.0f; // Assume precomputed
    float var = 1.0f; // Assume precomputed
    float std = sqrtf(var + 1e-5f);
    float normalized = (added - mean) / std;
    norm_out[idx] = gamma[idx % d_model] * normalized + beta[idx % d_model];
}

// Tensor Core optimized matrix multiply for training
__global__ void tensor_core_matmul_kernel(float *A, float *B, float *C, int m, int n, int k) {
    // Use WMMA for Tensor Cores (Volta+)
    // Placeholder for wmma::fragment and load/store
    // This is a simplified version; actual implementation requires wmma.h
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Compute row and col
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
}

// Simple LayerNorm kernel for bindings
__global__ void layernorm_kernel(float *input, float *weight, float *bias, float *output, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    float mean = sum / n;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / n;
    float std = sqrtf(var + eps);

    // Apply normalization
    float normalized = (input[idx] - mean) / std;
    output[idx] = weight[idx] * normalized + bias[idx];
}

// Simple attention kernel for bindings
__global__ void attention_kernel(float *q, float *k, float *v, float *output, int seq_len, int d_k, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * d_k) return;

    int pos = idx / d_k;
    int dim = idx % d_k;

    // Compute attention score (simplified)
    float score = 0.0f;
    for (int i = 0; i < d_k; i++) {
        score += q[pos * d_k + i] * k[pos * d_k + i];
    }
    score *= scale;

    // Apply to V (simplified, no softmax)
    output[idx] = score * v[idx];
}

// Gated activation kernel
__global__ void gated_activation_kernel(float *input, float *gate, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = input[idx] * sigmoid(gate[idx]);
}

// Simple Conv1D kernel
__global__ void conv1d_kernel(float *input, float *weight, float *bias, float *output,
                               int batch_size, int in_channels, int out_channels,
                               int input_length, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    int total_output = batch_size * out_channels * output_length;

    if (idx >= total_output) return;

    int b = idx / (out_channels * output_length);
    int c = (idx / output_length) % out_channels;
    int o = idx % output_length;

    float sum = bias[c];
    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_pos = o * stride - padding + k;
            if (in_pos >= 0 && in_pos < input_length) {
                sum += input[b * in_channels * input_length + ic * input_length + in_pos] *
                       weight[c * in_channels * kernel_size + ic * kernel_size + k];
            }
        }
    }
    output[idx] = sum;
}

// Host functions matching bindings
void launch_layernorm_forward(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, torch::Tensor& output, float eps) {
    float *d_input = input.data_ptr<float>();
    float *d_weight = weight.data_ptr<float>();
    float *d_bias = bias.data_ptr<float>();
    float *d_output = output.data_ptr<float>();
    int n = input.numel();

    dim3 block(256);
    dim3 grid((n + 255) / 256);
    layernorm_kernel<<<grid, block>>>(d_input, d_weight, d_bias, d_output, n, eps);
    CUDA_CHECK(cudaGetLastError());
}

void launch_attention_forward(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, torch::Tensor& output, float scale) {
    float *d_q = q.data_ptr<float>();
    float *d_kptr = k.data_ptr<float>();
    float *d_v = v.data_ptr<float>();
    float *d_output = output.data_ptr<float>();

    int seq_len = q.size(0);
    int dk = q.size(1);

    dim3 block(256);
    dim3 grid((seq_len * dk + 255) / 256);
    attention_kernel<<<grid, block>>>(d_q, d_kptr, d_v, d_output, seq_len, dk, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_gated_activation(torch::Tensor& input, torch::Tensor& gate, torch::Tensor& output) {
    float *d_input = input.data_ptr<float>();
    float *d_gate = gate.data_ptr<float>();
    float *d_output = output.data_ptr<float>();
    int n = input.numel();

    dim3 block(256);
    dim3 grid((n + 255) / 256);
    gated_activation_kernel<<<grid, block>>>(d_input, d_gate, d_output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv1d_forward(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, torch::Tensor& output, int stride, int padding) {
    float *d_input = input.data_ptr<float>();
    float *d_weight = weight.data_ptr<float>();
    float *d_bias = bias.data_ptr<float>();
    float *d_output = output.data_ptr<float>();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    int total_output = batch_size * out_channels * output_length;

    dim3 block(256);
    dim3 grid((total_output + 255) / 256);
    conv1d_kernel<<<grid, block>>>(d_input, d_weight, d_bias, d_output,
                                    batch_size, in_channels, out_channels,
                                    input_length, kernel_size, stride, padding);
    CUDA_CHECK(cudaGetLastError());
}

