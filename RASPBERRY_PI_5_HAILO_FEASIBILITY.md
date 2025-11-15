# Raspberry Pi 5 + Hailo 8L NPU Feasibility Assessment

**Date:** 2025-11-15  
**System:** AutoVoice Singing Voice Conversion  
**Target Platform:** Raspberry Pi 5 (8GB RAM) + Hailo 8L NPU (13 TOPS)

---

## Executive Summary

**Verdict: ⚠️ PARTIALLY FEASIBLE with significant limitations**

Porting AutoVoice to Raspberry Pi 5 with Hailo 8L NPU is **technically possible but NOT recommended for production use**. The system can run in a degraded mode with:
- ✅ Basic inference capability (CPU-only or limited NPU acceleration)
- ⚠️ 10-50x slower performance than GPU
- ⚠️ Limited model support on NPU
- ❌ No real-time processing capability
- ❌ Significant development effort required

---

## 1. Hardware Analysis

### Raspberry Pi 5 Specifications
| Component | Specification | AutoVoice Requirement | Status |
|-----------|--------------|----------------------|--------|
| **CPU** | Quad-core ARM Cortex-A76 @ 2.4GHz | Multi-core CPU | ✅ Adequate |
| **RAM** | 8GB LPDDR4X-4267 | 8GB minimum | ✅ Meets minimum |
| **Storage** | microSD/NVMe | 10GB+ free space | ✅ Adequate |
| **GPU** | VideoCore VII (800MHz) | NVIDIA GPU with CUDA | ❌ No CUDA support |
| **NPU** | Hailo 8L (13 TOPS) | Optional accelerator | ⚠️ Limited support |

### Hailo 8L NPU Specifications
- **Performance:** 13 TOPS (Tera Operations Per Second)
- **Precision:** INT8 quantized models only
- **Interface:** M.2 2242 PCIe connection
- **Supported Frameworks:** TensorFlow, PyTorch (via ONNX), Keras
- **Model Compilation:** Requires Hailo Dataflow Compiler

**Comparison to Current GPU:**
- NVIDIA RTX 3080 Ti: ~34 TFLOPS FP32, ~273 TFLOPS Tensor Cores (FP16)
- Hailo 8L: 13 TOPS INT8 (roughly equivalent to 1-2 TFLOPS FP32)
- **Performance Gap: 15-30x slower than current GPU**

---

## 2. Software Compatibility Analysis

### 2.1 PyTorch on ARM64

**Status: ✅ SUPPORTED**

PyTorch officially supports ARM64 (aarch64) architecture:
- **Official Support:** PyTorch 1.9+ has ARM64 wheels
- **Installation:** `pip install torch torchvision torchaudio`
- **Performance:** CPU-only, no CUDA acceleration
- **Verified:** Raspberry Pi 5 can run PyTorch models

**Limitations:**
- ❌ No CUDA support (NVIDIA-specific)
- ❌ No TensorRT optimization
- ❌ No custom CUDA kernels
- ⚠️ Significantly slower inference (10-50x)

### 2.2 Core Dependencies

| Dependency | ARM64 Support | Notes |
|------------|--------------|-------|
| **torch** | ✅ Yes | CPU-only, no CUDA |
| **torchaudio** | ✅ Yes | Full support |
| **librosa** | ✅ Yes | Pure Python/NumPy |
| **soundfile** | ✅ Yes | libsndfile available |
| **scipy** | ✅ Yes | ARM64 wheels available |
| **numpy** | ✅ Yes | Optimized for ARM |
| **flask** | ✅ Yes | Platform-independent |
| **demucs** | ✅ Yes | PyTorch-based, CPU fallback |
| **torchcrepe** | ✅ Yes | PyTorch-based, CPU fallback |
| **transformers** | ✅ Yes | HuggingFace supports ARM |
| **fairseq** | ⚠️ Partial | May need source build |
| **faiss-cpu** | ✅ Yes | CPU version available |

**CUDA-Specific Dependencies (NOT AVAILABLE):**
- ❌ `pynvml` - NVIDIA GPU monitoring
- ❌ `nvitop` - NVIDIA GPU profiling
- ❌ `tensorrt` - NVIDIA inference optimization
- ❌ Custom CUDA kernels in `src/cuda_kernels/`

---

## 3. Model Compatibility with Hailo 8L NPU

### 3.1 Current AutoVoice Models

| Model | Size | Architecture | Hailo 8L Compatible? | Notes |
|-------|------|-------------|---------------------|-------|
| **HuBERT-Soft** | 361MB | Transformer (12 layers) | ⚠️ Partial | Requires quantization, may lose accuracy |
| **RMVPE** | 173MB | CNN-based pitch estimator | ✅ Likely | Simpler architecture |
| **Demucs (htdemucs)** | ~350MB | Hybrid Transformer | ⚠️ Difficult | Complex architecture |
| **So-VITS-SVC** | Variable | Flow-based + Transformer | ❌ Unlikely | Too complex for NPU |
| **HiFi-GAN** | 54MB | CNN vocoder | ✅ Possible | Convolutional architecture |

### 3.2 Hailo NPU Limitations

**Supported Operations:**
- ✅ Convolutions (2D, depthwise, grouped)
- ✅ Pooling (max, average)
- ✅ Activation functions (ReLU, Sigmoid, Tanh)
- ✅ Batch normalization
- ✅ Element-wise operations
- ⚠️ Limited attention mechanisms
- ❌ Complex flow-based models
- ❌ Dynamic shapes

**Model Requirements:**
1. **Quantization:** Must be INT8 quantized (accuracy loss)
2. **Compilation:** Requires Hailo Dataflow Compiler
3. **ONNX Export:** Models must be exportable to ONNX
4. **Static Shapes:** No dynamic batch sizes or sequence lengths
5. **Operator Support:** All operators must be supported by Hailo

**Estimated NPU Acceleration:**
- ✅ **RMVPE pitch extraction:** 3-5x speedup possible
- ⚠️ **HiFi-GAN vocoder:** 2-3x speedup possible
- ❌ **HuBERT-Soft:** Unlikely to fit or run efficiently
- ❌ **So-VITS-SVC:** Too complex for NPU
- ❌ **Demucs:** Too large and complex

---

## 4. Memory Requirements Analysis

### 4.1 Current System Memory Usage

**GPU VRAM (NVIDIA RTX 3080 Ti):**
- Model loading: 2-5 GB
- Inference: 3-8 GB peak
- Total: 4-12 GB depending on quality preset

**System RAM:**
- Python environment: ~500 MB
- Models in RAM: ~1 GB
- Audio buffers: ~200 MB
- Total: ~2 GB baseline

### 4.2 Raspberry Pi 5 Memory Constraints

**Available RAM: 8 GB**
- Operating system: ~1-2 GB
- Python environment: ~500 MB
- **Available for models: ~5-6 GB**

**Model Memory Requirements (CPU mode):**
- HuBERT-Soft: ~1.5 GB (loaded)
- RMVPE: ~500 MB (loaded)
- Demucs: ~1.2 GB (loaded)
- So-VITS-SVC: ~800 MB (loaded)
- HiFi-GAN: ~200 MB (loaded)
- **Total: ~4.2 GB for all models**

**Verdict: ✅ Memory is sufficient** (5-6 GB available vs 4.2 GB needed)

---

## 5. Performance Expectations

### 5.1 CPU-Only Performance (No NPU)

Based on ARM Cortex-A76 performance vs x86 CPUs:

| Operation | Current GPU | Raspberry Pi 5 CPU | Slowdown Factor |
|-----------|------------|-------------------|----------------|
| **Model Loading** | 2.07s | 10-15s | 5-7x |
| **Pitch Extraction (CREPE)** | 0.5s per 30s | 15-25s per 30s | 30-50x |
| **Voice Conversion** | 30s per 30s (Balanced) | 300-600s per 30s | 10-20x |
| **Vocal Separation (Demucs)** | 10-15s per 30s | 120-180s per 30s | 12-15x |
| **Audio Synthesis (HiFi-GAN)** | 2-3s per 30s | 20-40s per 30s | 10-15x |

**Total Processing Time (30s audio):**
- **Current GPU (Balanced preset):** ~30 seconds
- **Raspberry Pi 5 CPU-only:** ~450-850 seconds (7.5-14 minutes)
- **Performance: 15-28x slower**

### 5.2 With Hailo 8L NPU Acceleration (Optimistic)

Assuming successful model compilation and NPU offloading:

| Operation | CPU-Only | With NPU | Speedup |
|-----------|----------|----------|---------|
| **Pitch Extraction (RMVPE)** | 20s | 5-8s | 2.5-4x |
| **Audio Synthesis (HiFi-GAN)** | 30s | 10-15s | 2-3x |
| **Voice Conversion (partial)** | 400s | 300-350s | 1.1-1.3x |

**Total Processing Time (30s audio) with NPU:**
- **Optimistic estimate:** ~320-400 seconds (5-7 minutes)
- **Still 10-13x slower than current GPU**

### 5.3 Real-Time Processing

**Current System:**
- ✅ Real-time capable with GPU (Fast preset: 0.35x RT on RTX 4090)

**Raspberry Pi 5:**
- ❌ NOT real-time capable
- Processing time: 10-28x longer than audio duration
- 30-second song takes 5-14 minutes to process

---

## 6. Development Effort Required

### 6.1 Code Modifications Needed

**1. Remove CUDA Dependencies (HIGH EFFORT)**
- ❌ Remove all `src/cuda_kernels/` code
- ❌ Remove CUDA-specific imports and checks
- ❌ Disable TensorRT optimization
- ❌ Remove GPU monitoring (pynvml, nvitop)
- ⚠️ Estimated effort: 2-3 days

**2. Add CPU Fallbacks (MEDIUM EFFORT)**
- ✅ Most models already have CPU fallback
- ⚠️ Need to optimize for ARM architecture
- ⚠️ May need to reduce model complexity
- ⚠️ Estimated effort: 1-2 days

**3. Hailo NPU Integration (VERY HIGH EFFORT)**
- ❌ Export models to ONNX format
- ❌ Quantize models to INT8
- ❌ Compile models with Hailo Dataflow Compiler
- ❌ Integrate Hailo SDK and runtime
- ❌ Test and validate accuracy
- ❌ Handle unsupported operations
- ⚠️ Estimated effort: 2-4 weeks

**4. Memory Optimization (MEDIUM EFFORT)**
- ⚠️ Implement model unloading between stages
- ⚠️ Reduce batch sizes
- ⚠️ Optimize audio buffer management
- ⚠️ Estimated effort: 3-5 days

**5. Frontend Adjustments (LOW EFFORT)**
- ⚠️ Update progress indicators for longer processing
- ⚠️ Add warnings about performance
- ⚠️ Disable real-time features
- ⚠️ Estimated effort: 1 day

**Total Development Effort: 4-6 weeks**

### 6.2 Testing and Validation

- Model accuracy validation after quantization
- Performance benchmarking
- Memory usage profiling
- End-to-end integration testing
- **Additional effort: 1-2 weeks**

---

## 7. Recommended Approach

### Option 1: CPU-Only Mode (EASIEST)

**Pros:**
- ✅ Minimal code changes
- ✅ No NPU integration complexity
- ✅ Works out of the box with PyTorch ARM64
- ✅ 1-2 weeks development time

**Cons:**
- ❌ 15-28x slower than GPU
- ❌ 5-14 minutes per 30s song
- ❌ Not suitable for interactive use

**Use Cases:**
- Batch processing overnight
- Non-time-critical conversions
- Development/testing

### Option 2: Selective NPU Acceleration (MODERATE)

**Pros:**
- ⚠️ 2-4x speedup on specific models
- ⚠️ Better than CPU-only
- ⚠️ Leverages NPU hardware

**Cons:**
- ❌ Still 10-13x slower than GPU
- ❌ Complex integration (4-6 weeks)
- ❌ Model accuracy may degrade
- ❌ Limited operator support

**Recommended Models for NPU:**
1. **RMVPE** (pitch extraction) - Highest priority
2. **HiFi-GAN** (vocoder) - Good candidate
3. Skip HuBERT and Demucs (too complex)

### Option 3: Cloud Hybrid Architecture (RECOMMENDED)

**Pros:**
- ✅ Raspberry Pi 5 as edge device/interface
- ✅ Heavy processing on cloud GPU
- ✅ Best user experience
- ✅ Scalable

**Cons:**
- ⚠️ Requires internet connection
- ⚠️ Cloud infrastructure costs
- ⚠️ Latency for audio upload/download

**Architecture:**
```
Raspberry Pi 5 (Frontend + API)
        ↓ (Upload audio)
Cloud GPU Server (Processing)
        ↓ (Download result)
Raspberry Pi 5 (Playback)
```

---

## 8. Cost-Benefit Analysis

### Hardware Costs

| Component | Cost | Performance |
|-----------|------|-------------|
| **Raspberry Pi 5 8GB** | $80 | 15-28x slower |
| **Hailo 8L AI Kit** | $70 | 10-13x slower (with NPU) |
| **Power Supply + Storage** | $30 | - |
| **Total** | **$180** | **Still 10-28x slower** |

**vs. Current GPU:**
- NVIDIA RTX 3080 Ti: $800-1200 (used)
- Performance: Baseline (1x)

### Development Costs

- **CPU-only port:** 1-2 weeks ($2,000-4,000 developer time)
- **NPU integration:** 4-6 weeks ($8,000-12,000 developer time)
- **Testing/validation:** 1-2 weeks ($2,000-4,000 developer time)

**Total Investment: $12,000-20,000 for 10-28x slower performance**

---

## 9. Final Recommendations

### ❌ NOT RECOMMENDED for Production

**Reasons:**
1. **Performance:** 10-28x slower than current GPU
2. **User Experience:** 5-14 minutes per 30s song (unacceptable)
3. **Development Cost:** $12,000-20,000 investment
4. **ROI:** Negative - better to use cloud GPU or keep current hardware
5. **Complexity:** Significant engineering effort for poor results

### ✅ RECOMMENDED Alternatives

**1. Keep Current GPU Setup**
- Best performance
- Already working
- Production-ready

**2. Cloud GPU Deployment**
- Use Raspberry Pi 5 as thin client
- Process on AWS/GCP GPU instances
- Better scalability

**3. Optimize Current Code**
- Improve GPU utilization
- Reduce memory usage
- Faster than porting to Pi

**4. Wait for Better Hardware**
- Raspberry Pi 6 (future)
- Better ARM NPUs (Hailo-10, Hailo-15)
- Native ARM GPU acceleration

---

## 10. Conclusion

**Porting AutoVoice to Raspberry Pi 5 + Hailo 8L is technically feasible but NOT practical.**

The system would work but with severe performance degradation (10-28x slower), making it unsuitable for any real-time or interactive use. The development effort (4-6 weeks) and cost ($12,000-20,000) far outweigh the benefits.

**If you need edge deployment, consider:**
- NVIDIA Jetson Orin Nano (8GB, $499) - 40 TOPS, CUDA support, 5-10x faster than Pi
- NVIDIA Jetson AGX Orin (32GB, $1,999) - 275 TOPS, production-grade
- Cloud GPU instances (AWS g4dn, GCP T4) - Best performance/cost ratio

**Bottom Line:** Stick with your current GPU setup or move to cloud GPU. The Raspberry Pi 5 + Hailo 8L is not suitable for this workload.

---

## Appendix: Technical Details

### A. PyTorch ARM64 Installation

```bash
# On Raspberry Pi 5 (Debian Bookworm)
sudo apt update
sudo apt install python3-pip python3-venv

# Create virtual environment
python3 -m venv autovoice-env
source autovoice-env/bin/activate

# Install PyTorch (ARM64 wheels)
pip install torch torchvision torchaudio

# Install other dependencies
pip install librosa soundfile scipy numpy flask
```

### B. Hailo SDK Installation

```bash
# Install Hailo SDK (requires registration)
wget https://hailo.ai/developer/downloads/hailo-sdk-latest.deb
sudo dpkg -i hailo-sdk-latest.deb

# Install Hailo Python API
pip install hailort

# Verify NPU detection
hailortcli fw-control identify
```

### C. Model Export to ONNX

```python
import torch
import torch.onnx

# Export PyTorch model to ONNX
model = load_model()
dummy_input = torch.randn(1, 80, 100)  # Example input
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {2: 'time'}}
)
```

### D. Hailo Model Compilation

```bash
# Quantize and compile ONNX model for Hailo 8L
hailo compiler \
  --onnx model.onnx \
  --hw-arch hailo8l \
  --output model.hef \
  --quantization-config quantization.yaml
```

---

**Report Generated:** 2025-11-15
**Author:** Augment Agent
**Status:** Complete

