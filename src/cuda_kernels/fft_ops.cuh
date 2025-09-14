#pragma once
#include <cufft.h>

// Forward declarations for FFT operations
void execute_cufft_forward(float* d_input, cufftComplex* d_output, int batch_size, int n_fft);
void execute_cufft_inverse(cufftComplex* d_input, float* d_output, int batch_size, int n_fft);