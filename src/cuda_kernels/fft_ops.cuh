#ifndef FFT_OPS_CUH
#define FFT_OPS_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

// FFT operation constants (general speech/audio)
#define FFT_SIZE 2048
#define HOP_SIZE 512
#define WINDOW_SIZE 2048
#define MEL_BINS 128
#define SAMPLE_RATE 22050

// Singing-specific constants (higher quality, 44.1kHz)
#define SINGING_SAMPLE_RATE 44100
#define SINGING_FFT_SIZE 2048
#define SINGING_HOP_SIZE 512
#define SINGING_MEL_BINS 128
#define SINGING_FMIN 80.0f      // Lower bound for singing (typical: 80 Hz)
#define SINGING_FMAX 8000.0f    // Upper bound for singing (typical: 8000 Hz)

// Real-time conversion constants
#define REALTIME_CHUNK_SIZE 4410  // 100ms chunks at 44.1kHz
#define REALTIME_OVERLAP_SIZE 1102  // 25% overlap

// Perceptual weighting constants
#define A_WEIGHTING_20_6 20.6f
#define A_WEIGHTING_107_7 107.7f
#define A_WEIGHTING_737_9 737.9f
#define A_WEIGHTING_12194_0 12194.0f

// Kernel declarations for FFT operations (existing)
__global__ void apply_window_kernel(float* signal, float* window, float* output, int window_size, int batch_size);
__global__ void compute_magnitude_kernel(cufftComplex* complex_data, float* magnitude, int fft_size, int batch_size);
__global__ void compute_phase_kernel(cufftComplex* complex_data, float* phase, int fft_size, int batch_size);
__global__ void mel_filterbank_kernel(float* magnitude, float* mel_output, float* filterbank,
                                     int fft_size, int mel_bins, int batch_size);

// New kernel declarations for singing voice conversion
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
    int batch_size,
    bool apply_a_weighting      // Optional perceptual weighting
);

__global__ void optimized_stft_kernel(
    float* audio,               // Input audio [batch_size, audio_length]
    cufftComplex* stft_output,  // Output STFT [batch_size, n_frames, n_fft/2+1]
    float* window,              // Window function [n_fft]
    int audio_length,
    int n_fft,
    int hop_length,
    int batch_size
);

__global__ void optimized_istft_kernel(
    cufftComplex* stft_input,   // Input STFT [batch_size, n_frames, n_fft/2+1]
    float* audio_output,        // Output audio [batch_size, audio_length]
    float* window,              // Window function [n_fft]
    int audio_length,
    int n_fft,
    int hop_length,
    int batch_size
);

__global__ void overlap_add_synthesis_kernel(
    float* ifft_frames,         // IFFT output frames [n_frames, n_fft]
    float* audio_output,        // Output audio [audio_length]
    float* window,              // Window function [n_fft]
    int audio_length,
    int n_frames,
    int n_fft,
    int hop_length
);

__global__ void precompute_window_sum_kernel(
    float* window,              // Window function [n_fft]
    float* window_sum,          // Output window sum [audio_length]
    int audio_length,
    int n_fft,
    int hop_length
);

__global__ void normalize_istft_kernel(
    float* audio,               // Audio to normalize [batch_size, audio_length]
    float* window_sum,          // Precomputed window sum [audio_length]
    int audio_length,
    int batch_size
);

__global__ void apply_perceptual_weighting_kernel(
    float* mel_spectrogram,     // Input/output mel-spectrogram [batch_size, n_frames, mel_bins]
    float* mel_frequencies,     // Mel bin center frequencies [mel_bins]
    int n_frames,
    int mel_bins,
    int batch_size
);

__global__ void compute_log_mel_kernel(
    float* mel_spectrogram,     // Input mel-spectrogram [batch_size, n_frames, mel_bins]
    float* log_mel_output,      // Output log-mel [batch_size, n_frames, mel_bins]
    int n_frames,
    int mel_bins,
    int batch_size,
    float epsilon               // Small constant to avoid log(0)
);

__global__ void realtime_voice_conversion_kernel(
    float* audio_chunk,         // Input audio chunk [chunk_size]
    float* overlap_buffer,      // Overlap buffer state [overlap_size]
    cufftComplex* fft_workspace, // FFT workspace [n_fft/2+1]
    float* features_output,     // Output features [feature_dim]
    float* window,              // Window function [n_fft]
    int chunk_size,
    int overlap_size,
    int n_fft,
    int hop_length,
    int feature_dim
);

// Helper functions for FFT operations
inline __device__ float hann_window(int n, int N) {
    return 0.5f * (1.0f - cosf(2.0f * M_PI * n / (N - 1)));
}

inline __device__ float hamming_window(int n, int N) {
    return 0.54f - 0.46f * cosf(2.0f * M_PI * n / (N - 1));
}

// Mel scale conversion functions (standard)
inline __device__ float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

inline __device__ float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Singing-specific mel scale (higher resolution for singing voice)
inline __device__ float hz_to_mel_singing(float hz) {
    // Higher resolution mel scale for singing voice (80-8000 Hz)
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

inline __device__ float mel_to_hz_singing(float mel) {
    // Inverse of singing mel scale
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// A-weighting for perceptual weighting
inline __device__ float a_weighting_db(float frequency_hz) {
    /**
     * Compute A-weighting in dB for perceptual weighting.
     * Used to emphasize perceptually important frequencies.
     *
     * Reference: IEC 61672-1:2013
     */
    float f2 = frequency_hz * frequency_hz;
    float f4 = f2 * f2;

    // A-weighting transfer function
    float numerator = A_WEIGHTING_12194_0 * A_WEIGHTING_12194_0 * f4;
    float denom1 = f2 + A_WEIGHTING_20_6 * A_WEIGHTING_20_6;
    float denom2 = sqrtf((f2 + A_WEIGHTING_107_7 * A_WEIGHTING_107_7) *
                        (f2 + A_WEIGHTING_737_9 * A_WEIGHTING_737_9));
    float denom3 = f2 + A_WEIGHTING_12194_0 * A_WEIGHTING_12194_0;
    float denominator = denom1 * denom2 * denom3;

    // Convert to dB (reference: 1.0 linear gain)
    float ra = numerator / denominator;
    float a_weight_db = 20.0f * log10f(ra) + 2.0f;  // +2 dB normalization

    return a_weight_db;
}

// Window normalization factor for overlap-add
inline __device__ float window_normalization_factor(float* window, int n_fft, int hop_length) {
    /**
     * Compute normalization factor for overlap-add to ensure perfect reconstruction.
     * Sum of squared window values at each sample position should equal this factor.
     */
    float sum_sq = 0.0f;
    for (int i = 0; i < n_fft; i += hop_length) {
        sum_sq += window[i] * window[i];
    }
    return sum_sq;
}

// Shared memory configurations for different kernels
#define STFT_SHARED_MEM_SIZE 4096
#define MEL_SHARED_MEM_SIZE 2048

// Updated shared memory configurations for singing voice kernels
#define SINGING_MEL_SHARED_MEM_SIZE (SINGING_FFT_SIZE / 2 + SINGING_MEL_BINS)  // ~1152 floats
#define OPTIMIZED_STFT_SHARED_MEM_SIZE SINGING_FFT_SIZE  // 2048 floats
#define OPTIMIZED_ISTFT_SHARED_MEM_SIZE SINGING_FFT_SIZE  // 2048 floats
#define OVERLAP_ADD_SHARED_MEM_SIZE (SINGING_FFT_SIZE + 512)  // 2560 floats for window sum
#define REALTIME_CONVERSION_SHARED_MEM_SIZE (REALTIME_CHUNK_SIZE + REALTIME_OVERLAP_SIZE)  // ~5512 floats

// cuFFT plan cache size (for optimization)
#define MAX_CUFFT_PLANS 8

// Streaming state buffer sizes
#define MAX_OVERLAP_BUFFER_SIZE 4096
#define MAX_FEATURE_BUFFER_SIZE 2048

// Error checking macro for CUFFT
#define CUFFT_CHECK(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            printf("CUFFT error: %d at %s:%d\\n", result, __FILE__, __LINE__); \
        } \
    } while(0)

#endif // FFT_OPS_CUH