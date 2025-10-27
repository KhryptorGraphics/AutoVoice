#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

// Forward declarations for CUDA kernel launchers
torch::Tensor voice_synthesis_cuda(torch::Tensor mel_spec, torch::Tensor speaker_embedding,
                                  torch::Tensor pitch_contour, int sample_rate);
torch::Tensor voice_conversion_cuda(torch::Tensor source_audio, torch::Tensor target_embedding,
                                   float pitch_shift, float formant_shift);
torch::Tensor pitch_shift_cuda(torch::Tensor audio, float shift_factor, int sample_rate);
torch::Tensor time_stretch_cuda(torch::Tensor audio, float stretch_factor);
torch::Tensor noise_reduction_cuda(torch::Tensor audio, float threshold);
torch::Tensor reverb_cuda(torch::Tensor audio, float room_size, float damping);
torch::Tensor stft_cuda(torch::Tensor signal, int n_fft, int hop_length, torch::Tensor window);
torch::Tensor istft_cuda(torch::Tensor stft_matrix, int hop_length, torch::Tensor window);
torch::Tensor mel_spectrogram_cuda(torch::Tensor audio, int n_mels, int n_fft, int hop_length);
torch::Tensor mfcc_cuda(torch::Tensor mel_spectrogram, int n_mfcc);
torch::Tensor griffin_lim_cuda(torch::Tensor magnitude, int n_iter);
torch::Tensor phase_vocoder_cuda(torch::Tensor stft_matrix, float rate);
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight,
                                 torch::Tensor bias, int stride, int padding);
torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight,
                             torch::Tensor bias, float epsilon);
torch::Tensor attention_cuda(torch::Tensor query, torch::Tensor key,
                           torch::Tensor value, float scale);
torch::Tensor gelu_activation_cuda(torch::Tensor input);
torch::Tensor adam_step_cuda(torch::Tensor param, torch::Tensor grad,
                            torch::Tensor m, torch::Tensor v,
                            float lr, float beta1, float beta2, float epsilon);
torch::Tensor allocate_pinned_memory(int size);
void transfer_to_device_async(torch::Tensor host_tensor, torch::Tensor device_tensor);
void transfer_to_host_async(torch::Tensor device_tensor, torch::Tensor host_tensor);
void synchronize_stream();

// Forward declarations for new launch_* functions from audio_kernels.cu and memory_kernels.cu
void launch_pitch_detection(torch::Tensor& input, torch::Tensor& output_pitch,
                           torch::Tensor& output_confidence, torch::Tensor& output_vibrato,
                           float sample_rate, int frame_length, int hop_length);
void launch_vibrato_analysis(torch::Tensor& pitch_contour, torch::Tensor& vibrato_rate,
                            torch::Tensor& vibrato_depth, int hop_length, float sample_rate);
void launch_voice_activity_detection(torch::Tensor& input, torch::Tensor& output, float threshold);
void launch_spectrogram_computation(torch::Tensor& input, torch::Tensor& output, int n_fft, int hop_length, int win_length);
void launch_formant_extraction(torch::Tensor& input, torch::Tensor& output, float sample_rate);
void launch_vocoder_synthesis(torch::Tensor& mel_spec, torch::Tensor& audio_out);
void launch_create_cuda_graph();
void launch_execute_cuda_graph();
void launch_destroy_cuda_graph();
void launch_stream_synchronize(uintptr_t stream_id);
void launch_async_memory_copy(torch::Tensor& dst, torch::Tensor& src, uintptr_t stream_id);

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voice_synthesis", &voice_synthesis_cuda, "Voice synthesis CUDA kernel",
          py::arg("mel_spec"), py::arg("speaker_embedding"),
          py::arg("pitch_contour"), py::arg("sample_rate") = 22050);

    m.def("voice_conversion", &voice_conversion_cuda, "Voice conversion CUDA kernel",
          py::arg("source_audio"), py::arg("target_embedding"),
          py::arg("pitch_shift") = 1.0f, py::arg("formant_shift") = 1.0f);

    m.def("pitch_shift", &pitch_shift_cuda, "Pitch shifting CUDA kernel",
          py::arg("audio"), py::arg("shift_factor"), py::arg("sample_rate") = 22050);

    m.def("time_stretch", &time_stretch_cuda, "Time stretching CUDA kernel",
          py::arg("audio"), py::arg("stretch_factor"));

    m.def("noise_reduction", &noise_reduction_cuda, "Noise reduction CUDA kernel",
          py::arg("audio"), py::arg("threshold") = 0.1f);

    m.def("reverb", &reverb_cuda, "Reverb effect CUDA kernel",
          py::arg("audio"), py::arg("room_size") = 0.5f, py::arg("damping") = 0.5f);

    m.def("stft", &stft_cuda, "STFT CUDA kernel",
          py::arg("signal"), py::arg("n_fft") = 2048,
          py::arg("hop_length") = 512, py::arg("window"));

    m.def("istft", &istft_cuda, "Inverse STFT CUDA kernel",
          py::arg("stft_matrix"), py::arg("hop_length") = 512, py::arg("window"));

    m.def("mel_spectrogram", &mel_spectrogram_cuda, "Mel spectrogram CUDA kernel",
          py::arg("audio"), py::arg("n_mels") = 128,
          py::arg("n_fft") = 2048, py::arg("hop_length") = 512);

    m.def("mfcc", &mfcc_cuda, "MFCC CUDA kernel",
          py::arg("mel_spectrogram"), py::arg("n_mfcc") = 13);

    m.def("griffin_lim", &griffin_lim_cuda, "Griffin-Lim CUDA kernel",
          py::arg("magnitude"), py::arg("n_iter") = 32);

    m.def("phase_vocoder", &phase_vocoder_cuda, "Phase vocoder CUDA kernel",
          py::arg("stft_matrix"), py::arg("rate"));

    m.def("matmul", &matmul_cuda, "Matrix multiplication CUDA kernel",
          py::arg("a"), py::arg("b"));

    m.def("conv2d_forward", &conv2d_forward_cuda, "2D convolution forward CUDA kernel",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 1, py::arg("padding") = 0);

    m.def("layer_norm", &layer_norm_cuda, "Layer normalization CUDA kernel",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("epsilon") = 1e-5f);

    m.def("attention", &attention_cuda, "Attention mechanism CUDA kernel",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("scale") = 1.0f);

    m.def("gelu_activation", &gelu_activation_cuda, "GELU activation CUDA kernel",
          py::arg("input"));

    m.def("adam_step", &adam_step_cuda, "Adam optimizer step CUDA kernel",
          py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"),
          py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f,
          py::arg("beta2") = 0.999f, py::arg("epsilon") = 1e-8f);

    m.def("allocate_pinned_memory", &allocate_pinned_memory, "Allocate pinned memory",
          py::arg("size"));

    m.def("transfer_to_device_async", &transfer_to_device_async,
          "Asynchronous transfer to device",
          py::arg("host_tensor"), py::arg("device_tensor"));

    m.def("transfer_to_host_async", &transfer_to_host_async,
          "Asynchronous transfer to host",
          py::arg("device_tensor"), py::arg("host_tensor"));

    m.def("synchronize_stream", &synchronize_stream, "Synchronize CUDA stream");

    // Enhanced pitch detection and vibrato analysis bindings
    m.def("launch_pitch_detection", &launch_pitch_detection,
          "Enhanced pitch detection (GPU)",
          py::arg("input"), py::arg("output_pitch"), py::arg("output_confidence"),
          py::arg("output_vibrato"), py::arg("sample_rate"),
          py::arg("frame_length"), py::arg("hop_length"));

    m.def("launch_vibrato_analysis", &launch_vibrato_analysis,
          "Vibrato analysis (GPU)",
          py::arg("pitch_contour"), py::arg("vibrato_rate"), py::arg("vibrato_depth"),
          py::arg("hop_length"), py::arg("sample_rate"));
}