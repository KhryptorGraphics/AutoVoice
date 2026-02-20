# Troubleshooting Guide

This guide covers common errors and solutions when working with AutoVoice. Use this as a self-service resource for resolving GPU memory issues, model loading failures, audio processing errors, and dependency problems.

## Quick Start

If you encounter an error:

1. **Check the error category** in the table of contents below
2. **Find your specific error message** in that section
3. **Follow the solution steps** provided
4. **Run diagnostic commands** to verify the fix
5. If still stuck, see [Getting Help](#getting-help)

## Table of Contents

- [GPU and CUDA Errors](#gpu-and-cuda-errors)
  - CUDA out of memory
  - Insufficient GPU memory for model loading
  - GPU device not available
  - CUDA driver version mismatch

- [Model Loading Errors](#model-loading-errors)
  - Model file not found
  - Checkpoint missing or corrupted
  - Profile not found
  - No LoRA adapter for profile
  - Failed to initialize model

- [Audio Processing Errors](#audio-processing-errors)
  - Input audio contains NaN or Inf values
  - Insufficient audio quality
  - Speaker embedding contains NaN or Inf
  - Audio file format not supported
  - Sample rate mismatch

- [Database Errors](#database-errors)
  - SQLAlchemy session errors
  - Profile not found in database
  - Transaction rollback issues
  - Database connection failures

- [API and Validation Errors](#api-and-validation-errors)
  - Invalid adapter_type parameter
  - Type errors in request validation
  - Missing required parameters
  - Authentication failures

- [Dependency and Environment Errors](#dependency-and-environment-errors)
  - Missing sounddevice package
  - local-attention module not installed
  - PyWorld ARM64/Python 3.13 incompatibility
  - CUDA toolkit version mismatch
  - Python environment conflicts

- [Diagnostic Commands](#diagnostic-commands)
  - Verify GPU access
  - Check model files
  - Test audio processing
  - Validate database connection
  - Debug environment setup

- [Common Workflows](#common-workflows)
  - Fresh installation troubleshooting
  - GPU memory optimization
  - Model debugging workflow
  - Production deployment checklist

- [Getting Help](#getting-help)
  - Collecting diagnostic information
  - Reporting issues
  - Community resources

---

## GPU and CUDA Errors

*This section will be populated with GPU and CUDA error solutions.*

## Model Loading Errors

*This section will be populated with model loading error solutions.*

## Audio Processing Errors

*This section will be populated with audio processing error solutions.*

## Database Errors

*This section will be populated with database error solutions.*

## API and Validation Errors

*This section will be populated with API and validation error solutions.*

## Dependency and Environment Errors

*This section will be populated with dependency error solutions.*

## Diagnostic Commands

*This section will be populated with diagnostic commands and utilities.*

## Common Workflows

*This section will be populated with troubleshooting workflows.*

## Getting Help

### Collecting Diagnostic Information

Before reporting an issue, collect the following information:

1. **System Information**:
   ```bash
   # OS and hardware
   uname -a

   # GPU information
   nvidia-smi

   # CUDA version
   nvcc --version
   ```

2. **Python Environment**:
   ```bash
   # Python version
   python --version

   # Installed packages
   pip list

   # PyTorch/CUDA status
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
   ```

3. **AutoVoice Configuration**:
   ```bash
   # Check configuration files
   cat config/gpu_config.yaml

   # Verify model files
   ls -lh models/pretrained/

   # Check logs
   tail -n 100 logs/autovoice.log
   ```

### Reporting Issues

When reporting an issue, include:

- Error message (full stack trace)
- System information (see above)
- Steps to reproduce
- Expected vs actual behavior
- Configuration files (remove sensitive data)

### Community Resources

- **Documentation**: See `docs/` directory for guides
- **Test Examples**: Check `tests/` for usage examples
- **Code Examples**: See `examples/` for sample code
- **Deployment Guide**: `docs/deployment-guide.md` for production setup

---

*Note: This guide is actively maintained. Sections are being populated with detailed error solutions and diagnostic procedures.*
