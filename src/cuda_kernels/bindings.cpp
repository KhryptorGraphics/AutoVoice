#include <torch/extension.h>

// Forward declarations of CUDA kernel launchers from audio_kernels.cu
void launch_pitch_detection(torch::Tensor& input, torch::Tensor& output, float sample_rate);
void launch_formant_extraction(torch::Tensor& input, torch::Tensor& output, float sample_rate);
void launch_vocoder_synthesis(torch::Tensor& mel_spec, torch::Tensor& audio_out);
void launch_voice_activity_detection(torch::Tensor& input, torch::Tensor& output, float threshold);
void launch_spectrogram_computation(torch::Tensor& input, torch::Tensor& output, int n_fft, int hop_length, int win_length);

// Forward declarations from fft_kernels.cu
void launch_apply_window(torch::Tensor& input, torch::Tensor& window, torch::Tensor& output);
void launch_compute_magnitude(torch::Tensor& complex_input, torch::Tensor& magnitude);
void launch_apply_mel_filters(torch::Tensor& magnitude, torch::Tensor& mel_filters, torch::Tensor& mel_spec);

// Forward declarations from training_kernels.cu
void launch_layernorm_forward(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, torch::Tensor& output, float eps);
void launch_attention_forward(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, torch::Tensor& output, float scale);
void launch_gated_activation(torch::Tensor& input, torch::Tensor& gate, torch::Tensor& output);
void launch_conv1d_forward(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, torch::Tensor& output, int stride, int padding);

// Forward declarations from memory_kernels.cu
void launch_async_memory_copy(torch::Tensor& dst, torch::Tensor& src, int64_t stream_id);
void launch_stream_synchronize(int64_t stream_id);
void launch_create_cuda_graph();
void launch_execute_cuda_graph();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Audio processing kernels
    m.def("launch_pitch_detection", &launch_pitch_detection, "Pitch detection using CUDA");
    m.def("launch_formant_extraction", &launch_formant_extraction, "Formant extraction using CUDA");
    m.def("launch_vocoder_synthesis", &launch_vocoder_synthesis, "Vocoder synthesis using CUDA");
    m.def("launch_voice_activity_detection", &launch_voice_activity_detection, "Voice activity detection using CUDA");
    m.def("launch_spectrogram_computation", &launch_spectrogram_computation, "Spectrogram computation using CUDA");

    // FFT kernels
    m.def("launch_apply_window", &launch_apply_window, "Apply window function using CUDA");
    m.def("launch_compute_magnitude", &launch_compute_magnitude, "Compute magnitude from complex spectrum");
    m.def("launch_apply_mel_filters", &launch_apply_mel_filters, "Apply mel filter banks");

    // Training kernels
    m.def("launch_layernorm_forward", &launch_layernorm_forward, "Layer normalization forward pass");
    m.def("launch_attention_forward", &launch_attention_forward, "Attention mechanism forward pass");
    m.def("launch_gated_activation", &launch_gated_activation, "Gated activation function");
    m.def("launch_conv1d_forward", &launch_conv1d_forward, "1D convolution forward pass");

    // Memory management kernels
    m.def("launch_async_memory_copy", &launch_async_memory_copy, "Asynchronous memory copy");
    m.def("launch_stream_synchronize", &launch_stream_synchronize, "Stream synchronization");
    m.def("launch_create_cuda_graph", &launch_create_cuda_graph, "Create CUDA graph");
    m.def("launch_execute_cuda_graph", &launch_execute_cuda_graph, "Execute CUDA graph");
}