# Consistency Models for Singing Voice Conversion: Research Findings

## Table of Contents
1. [CoMoSVC: Consistency Model-based Singing Voice Conversion](#1-comosvc-consistency-model-based-singing-voice-conversion)
2. [Consistency Models: Mathematical Foundations](#2-consistency-models-mathematical-foundations)
3. [TensorRT Optimization for Real-Time Inference on Jetson Thor](#3-tensorrt-optimization-for-real-time-inference-on-jetson-thor)
4. [Related Work: Consistency Models in Audio/Speech](#4-related-work-consistency-models-in-audiospeech)
5. [Implementation Roadmap for AutoVoice](#5-implementation-roadmap-for-autovoice)

---

## 1. CoMoSVC: Consistency Model-based Singing Voice Conversion

**Paper:** Lu et al., "CoMoSVC: Consistency Model-based Singing Voice Conversion" (arXiv:2401.01792, January 2024)
**Published:** IEEE ISCSLP 2024
**Project Page:** https://comosvc.github.io/
**Code:** https://github.com/Grace9994/CoMoSVC
**Amphion Implementation:** https://github.com/open-mmlab/Amphion/blob/main/egs/svc/DiffComoSVC/README.md

### 1.1 System Overview

CoMoSVC achieves one-step singing voice conversion by distilling a diffusion-based teacher model into a consistency student model. The key contribution is maintaining comparable or superior conversion quality while reducing sampling from 50-1000 steps to a single step.

**Performance (NVIDIA GTX 4090):**
| Method | Steps | Similarity |
|--------|-------|-----------|
| DiffSVC | 100 | baseline |
| SoVITS-Diff | 1000 | baseline |
| Teacher Model | 50 | baseline |
| **CoMoSVC** | **1** | **+0.05 over all baselines** |

### 1.2 Architecture

The system is a two-stage architecture:

**Stage 1: Feature Encoding**
- Content features: 768-dimensional vectors from ContentVec (12th layer of pre-trained acoustic model)
- Pitch: F0 estimation via DIO estimator with voiced/unvoiced flags per frame
- Loudness: Squared magnitude of the audio signal
- Speaker embedding: Singer ID encoded as embedding vector
- All features projected to 256 dimensions and concatenated

**Stage 2: Conditional Mel-Spectrogram Generation**
- Conformer encoder generates coarse spectrogram
- Diffusion decoder (BiDilConv) refines the spectrogram
- Pre-trained HiFi-GAN vocoder renders audio from mel-spectrogram

### 1.3 BiDilConv Architecture (Bidirectional Non-Causal Dilated CNN)

The BiDilConv decoder is based on the non-causal WaveNet architecture, using bidirectional (non-causal) dilated convolutions to enlarge the receptive field without restricting to causal (left-only) context.

**Architecture Specifications (from Amphion/CoMoSpeech implementations):**

```
BiDilConv Decoder:
- Type: Non-causal WaveNet (bidirectional dilated convolutions)
- Residual channels: 256
- Dilation cycle: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
- Number of cycles: 2 (total 20 residual blocks)
- Kernel size: 3
- Skip channels: 256
- Gated activation: sigmoid(Wf * x) * tanh(Wg * x)
- Residual connections from each block
- Skip connections summed at output
```

**Residual Block Structure:**
```
Input
  |
  +---> Conv1D(dilated, non-causal) ---> GatedActivation
  |                                            |
  |                                      +-----+-----+
  |                                      |           |
  |                                   1x1 Conv   1x1 Conv
  |                                   (residual)  (skip)
  |                                      |           |
  +-------(add)------------------------->+           |
  |                                                  |
  Output                                        Skip Sum
```

**Key difference from standard WaveNet:** Uses non-causal (bidirectional) padding instead of causal (left-only) padding, allowing the network to see both past and future context. This is acceptable for offline/batch processing but requires modification for streaming.

### 1.4 EDM Preconditioning (Karras Noise Schedule)

**Reference:** Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (arXiv:2206.00364)

The denoiser is parameterized with sigma-dependent skip connections:

```
D_theta(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))
```

**Preconditioning Functions:**

```python
def c_skip(sigma, sigma_data=0.5):
    return sigma_data**2 / (sigma**2 + sigma_data**2)

def c_out(sigma, sigma_data=0.5):
    return sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)

def c_in(sigma, sigma_data=0.5):
    return 1.0 / torch.sqrt(sigma**2 + sigma_data**2)

def c_noise(sigma):
    return 0.25 * torch.log(sigma)
```

**Karras Noise Schedule Parameters (for CoMoSVC):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| sigma_data | 0.5 | Standard deviation of data distribution |
| sigma_min (epsilon) | 0.002 | Minimum noise level |
| sigma_max | 80.0 | Maximum noise level |
| P_mean | -1.2 | Mean of log-normal training distribution |
| P_std | 1.2 | Std of log-normal training distribution |
| rho | 7 | Schedule curvature parameter |
| n_timesteps | 40 | Number of discretization steps (teacher) |

**Training Noise Sampling:**
```python
# Sample noise levels for training (log-normal distribution)
ln_sigma = P_mean + P_std * torch.randn(batch_size)
sigma = torch.exp(ln_sigma)
```

**Sigma Schedule for Inference (Karras schedule):**
```python
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7):
    """Generate n sigma values using Karras schedule."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
```

**Training Loss with EDM Weighting:**
```python
# Loss = E[lambda(sigma) * ||D(x + sigma*z; sigma) - x||^2]
# where lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
def edm_loss(model, x_0, sigma, condition):
    noise = torch.randn_like(x_0)
    x_noisy = x_0 + sigma * noise
    prediction = model(x_noisy, sigma, condition)
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
    loss = weight * F.mse_loss(prediction, x_0)
    return loss
```

### 1.5 Two-Stage Training Procedure

#### Stage 1: Teacher Model Training (Diffusion)

The teacher model is a diffusion denoiser trained with EDM formulation:

```python
# Teacher Training Loop
for batch in dataloader:
    mel_target, content, pitch, speaker = batch

    # Sample noise level from log-normal
    sigma = torch.exp(P_mean + P_std * torch.randn(batch_size)).to(device)

    # Add noise
    noise = torch.randn_like(mel_target)
    x_noisy = mel_target + sigma.unsqueeze(-1) * noise

    # Predict clean mel with EDM preconditioning
    condition = encoder(content, pitch, speaker)
    x_pred = teacher_model(x_noisy, sigma, condition)

    # EDM-weighted MSE loss
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
    loss = (weight * (x_pred - mel_target)**2).mean()

    optimizer.step()
```

**Teacher Hyperparameters:**
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 48
- Training iterations: 1,000,000
- GPU: NVIDIA GTX 4090

#### Stage 2: Consistency Distillation (Student)

The student model is distilled from the teacher using the consistency training loss:

```python
# Consistency Distillation Training Loop
# Freeze: condition encoder + conformer
# Update: diffusion decoder via EMA

for batch in dataloader:
    mel_target, content, pitch, speaker = batch
    condition = encoder(content, pitch, speaker)  # frozen

    # Sample adjacent timesteps t_{n+1} and t_n
    n = torch.randint(0, N-1, (batch_size,))
    t_next = sigmas[n + 1]  # t_{n+1}
    t_curr = sigmas[n]      # t_n

    # Create noisy sample at t_{n+1}
    noise = torch.randn_like(mel_target)
    x_next = mel_target + t_next.unsqueeze(-1) * noise

    # One-step Euler ODE solve: x_{n+1} -> x_n using teacher
    with torch.no_grad():
        teacher_pred = teacher_model(x_next, t_next, condition)
        # Euler step
        x_curr = x_next + (t_curr - t_next).unsqueeze(-1) * (
            (x_next - teacher_pred) / t_next.unsqueeze(-1)
        )

    # Student predictions at adjacent points
    student_pred_next = student_model(x_next, t_next, condition)

    with torch.no_grad():
        # EMA target network prediction
        target_pred_curr = ema_model(x_curr, t_curr, condition)

    # Consistency Training Loss (CTLoss_D)
    loss = F.mse_loss(student_pred_next, target_pred_curr)

    optimizer.step()

    # EMA update of target network
    ema_update(ema_model, student_model, mu=0.95)
```

### 1.6 CTLoss_D: Consistency Training Loss for Distillation

The consistency distillation loss enforces the self-consistency property:

```
CTLoss_D = ||D_theta(x_{t_{n+1}}, t_{n+1}, cond) - D_{theta^-}(x_hat_{t_n}, t_n, cond)||^2
```

Where:
- `D_theta` is the online (student) network
- `D_{theta^-}` is the EMA target network
- `x_hat_{t_n}` is obtained via first-order Euler solver from the teacher
- `x_{t_{n+1}}` is the noisy sample at timestep `t_{n+1}`

### 1.7 EMA Teacher Update

```python
def ema_update(target_model, online_model, mu=0.95):
    """Exponential Moving Average update for target network."""
    for target_param, online_param in zip(
        target_model.parameters(), online_model.parameters()
    ):
        target_param.data = mu * target_param.data + (1 - mu) * online_param.data
```

**Parameters:**
- mu (momentum): 0.95
- Update rule: `theta^- <- stopgrad(mu * theta^- + (1 - mu) * theta)`

### 1.8 Audio Configuration

| Parameter | Value |
|-----------|-------|
| Sampling rate | 24 kHz |
| FFT size | 512 |
| Window size | 512 |
| Hop size | 128 |
| Mel bins | 80 |
| Feature dimension | 256 (after projection) |

### 1.9 Conformer Encoder Configuration (Amphion)

| Parameter | Value |
|-----------|-------|
| Input dimension | 384 |
| Output dimension | 100 |
| Attention heads | 2 |
| Layers | 6 |
| Filter channels | 512 |

---

## 2. Consistency Models: Mathematical Foundations

**Paper:** Song et al., "Consistency Models" (arXiv:2303.01469, March 2023, ICML 2023)
**Improved:** Song & Dhariwal, "Improved Techniques for Training Consistency Models" (arXiv:2310.14189, ICLR 2024 Oral)

### 2.1 Mathematical Formulation

#### Probability Flow ODE

Diffusion models define a forward process that gradually adds noise:
```
dx_t = sqrt(2t) * dw_t,  t in [epsilon, T]
```

The corresponding Probability Flow ODE (PF-ODE) traces deterministic trajectories:
```
dx/dt = -t * grad_x log p_t(x)
```

Each trajectory maps a noise sample `x_T ~ N(0, T^2 I)` to a clean data point `x_epsilon ~ p_data`.

#### Consistency Function Definition

A consistency function `f: (x_t, t) -> x_epsilon` maps any point on an ODE trajectory to its origin:

```
f(x_t, t) = x_epsilon  for all t in [epsilon, T]
```

**Self-Consistency Property:**
```
f(x_t, t) = f(x_t', t')  for all t, t' in [epsilon, T] on the same trajectory
```

This means all points along the same ODE trajectory must produce the same output.

#### Boundary Condition

```
f(x_epsilon, epsilon) = x_epsilon  (identity at t = epsilon)
```

### 2.2 Network Parameterization

To enforce the boundary condition, the consistency model uses skip connections:

```python
def consistency_model(x_t, t, condition):
    """Parameterization ensuring boundary condition."""
    c_skip_t = sigma_data**2 / ((t - epsilon)**2 + sigma_data**2)
    c_out_t = sigma_data * (t - epsilon) / torch.sqrt(sigma_data**2 + t**2)

    # F_theta is the neural network backbone
    F_output = F_theta(x_t, t, condition)

    return c_skip_t * x_t + c_out_t * F_output
```

**Properties:**
- At `t = epsilon`: `c_skip = 1`, `c_out = 0`, so `f(x, epsilon) = x` (boundary condition satisfied)
- At `t = T`: `c_skip ~ 0`, `c_out ~ sigma_data`, so output depends primarily on `F_theta`

### 2.3 Consistency Distillation (CD)

Given a pre-trained diffusion model (score function), distill into a consistency model:

**Loss Function:**
```
L_CD(theta, theta^-) = E_{t_n, x_0, noise}[
    lambda(t_n) * d(f_theta(x_{t_{n+1}}, t_{n+1}), f_{theta^-}(x_hat_{t_n}, t_n))
]
```

Where:
- `x_hat_{t_n}` is obtained by one ODE step from `x_{t_{n+1}}` using the teacher
- `f_{theta^-}` is the EMA target network
- `d(., .)` is a distance metric (L2, Pseudo-Huber, or LPIPS)
- `lambda(t_n)` is a weighting function

**Euler ODE Solver Step:**
```python
# One-step Euler from t_{n+1} to t_n
score = diffusion_model.score(x_t, t_next)
x_t_prev = x_t + (t_curr - t_next) * (-t_next * score)
```

### 2.4 Consistency Training (CT) - No Teacher Required

Train directly from data without a pre-trained diffusion model:

**Loss Function:**
```
L_CT(theta, theta^-) = E_{t_n, x_0, noise}[
    lambda(t_n) * d(f_theta(x_0 + t_{n+1}*z, t_{n+1}), f_{theta^-}(x_0 + t_n*z, t_n))
]
```

Where `z ~ N(0, I)` is shared noise, and both noisy samples are generated from the same `x_0`.

### 2.5 Training Schedule and Discretization

**Adaptive Schedule for N (number of timesteps):**
```python
def schedule_N(k, s0=10, s1=1280, K=800000):
    """Progressively increase discretization steps during training."""
    N = min(s0 * 2**(k // (K // math.log2(s1/s0))), s1) + 1
    return int(N)
```

**Sigma Schedule (Karras):**
```python
def get_sigmas(N, sigma_min=0.002, sigma_max=80, rho=7):
    """Discretize noise levels with Karras schedule."""
    indices = torch.arange(N)
    sigmas = (
        sigma_max**(1/rho) + indices / (N-1) * (sigma_min**(1/rho) - sigma_max**(1/rho))
    ) ** rho
    return sigmas
```

**Adaptive EMA Schedule (iCT):**
```python
def schedule_mu(N, s0=10):
    """EMA rate adapts with N."""
    return math.exp(s0 * math.log(0.95) / N)
```

### 2.6 Distance Metrics

**Pseudo-Huber Loss (recommended by iCT):**
```python
def pseudo_huber_loss(x, y, c=0.03):
    """Robust loss that is L2 near zero and L1 far from zero."""
    diff = x - y
    return torch.sqrt(diff**2 + c**2) - c
```

**Comparison of metrics:**
| Metric | Pros | Cons |
|--------|------|------|
| L2 | Simple, fast | Sensitive to outliers |
| L1 | Robust to outliers | Non-smooth at 0 |
| Pseudo-Huber | Smooth, robust | One hyperparameter (c) |
| LPIPS | Perceptually aligned | Expensive, biased |

### 2.7 Sampling

**One-Step Generation:**
```python
def sample_one_step(model, batch_size, device):
    """Generate samples in a single forward pass."""
    # Sample from N(0, T^2 I)
    x_T = torch.randn(batch_size, *data_shape, device=device) * sigma_max
    # Single model evaluation
    x_0 = model(x_T, sigma_max)
    return x_0
```

**Multi-Step Refinement (optional quality boost):**
```python
def sample_multistep(model, batch_size, sigmas, device):
    """Trade compute for quality with iterative refinement."""
    x = torch.randn(batch_size, *data_shape, device=device) * sigmas[0]
    for i in range(len(sigmas) - 1):
        # Denoise
        x_denoised = model(x, sigmas[i])
        # Re-noise to next level
        if i < len(sigmas) - 2:
            noise = torch.randn_like(x)
            x = x_denoised + sigmas[i+1] * noise
        else:
            x = x_denoised
    return x
```

### 2.8 Key Results (Original Paper)

| Dataset | Model | FID (1-step) | FID (2-step) |
|---------|-------|-------------|-------------|
| CIFAR-10 | CD (L2) | 3.55 | 2.93 |
| ImageNet 64x64 | CD (L2) | 6.20 | 4.70 |

### 2.9 Improved Consistency Training (iCT, arXiv:2310.14189)

Key improvements over original:
1. **Remove EMA from teacher**: Identified flaw where EMA teacher introduces bias
2. **Pseudo-Huber loss**: Replaces LPIPS, removes evaluation bias
3. **Lossless discretization**: Zero-padding approach for variable N
4. **Training from scratch**: No need for pre-trained diffusion model

**iCT Training Configuration:**
- Batch size: 4096
- Iterations: 800,000
- EMA: Adaptive schedule (removed from teacher)
- Loss: Pseudo-Huber (c=0.03)

---

## 3. TensorRT Optimization for Real-Time Inference on Jetson Thor

### 3.1 Jetson Thor Hardware Specifications

**Reference:** NVIDIA Jetson Thor (launched August 2025)

| Specification | Value |
|--------------|-------|
| GPU Architecture | Blackwell (SM 11.0 / compute capability 11.0) |
| CUDA Version | 13.0 |
| Dense FP4 TFLOPS | 2,070 |
| Dense FP8 / Sparse FP16 TFLOPS | 517 |
| Sparse FP8 / INT8 TFLOPS | 1,035 |
| Memory | 128 GB LPDDR5X |
| CPU | 14x Arm Neoverse V3AE cores |
| Power | 40-130 W |
| Performance vs Orin | 7.5x AI performance, 3.5x efficiency |
| Software Stack | JetPack 7.1, Isaac ROS 4.0, CUDA 13.0, TensorRT |

**Known Issues (as of January 2026):**
- TensorRT 10.13.3.9 may silently fall back to FP32 when FP8/FP4 flags are enabled on SM 11.0
- Verify engine precision after build; FP16/INT8 are reliable
- Monitor issue: https://github.com/NVIDIA/tensorrt/issues/4590

### 3.2 Complete TensorRT Pipeline for Voice Conversion

#### Step 1: ONNX Export from PyTorch

```python
import torch
import torch.onnx

def export_consistency_model_to_onnx(
    model,
    output_path="consistency_svc.onnx",
    mel_bins=80,
    max_frames=800,  # ~10s at 24kHz with hop=128
):
    """Export trained consistency model to ONNX with dynamic shapes."""
    model.eval()

    # Dummy inputs matching model signature
    batch_size = 1
    n_frames = 200  # Example: ~2.5s

    dummy_mel = torch.randn(batch_size, mel_bins, n_frames).cuda()
    dummy_sigma = torch.tensor([1.0]).cuda()
    dummy_content = torch.randn(batch_size, 256, n_frames).cuda()
    dummy_pitch = torch.randn(batch_size, 1, n_frames).cuda()
    dummy_speaker = torch.randn(batch_size, 256).cuda()

    # Export with dynamic time dimension
    torch.onnx.export(
        model,
        (dummy_mel, dummy_sigma, dummy_content, dummy_pitch, dummy_speaker),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
            'noisy_mel', 'sigma', 'content', 'pitch', 'speaker'
        ],
        output_names=['denoised_mel'],
        dynamic_axes={
            'noisy_mel': {0: 'batch', 2: 'time'},
            'content': {0: 'batch', 2: 'time'},
            'pitch': {0: 'batch', 2: 'time'},
            'denoised_mel': {0: 'batch', 2: 'time'},
        }
    )
    print(f"Exported to {output_path}")
```

**For PyTorch 2.5+ (dynamo-based export):**
```python
import torch.onnx

# New recommended approach
torch.onnx.export(
    model,
    (dummy_mel, dummy_sigma, dummy_content, dummy_pitch, dummy_speaker),
    "consistency_svc.onnx",
    dynamo=True,
    dynamic_shapes={
        'noisy_mel': {0: torch.export.Dim('batch'), 2: torch.export.Dim('time', min=1, max=800)},
        'content': {0: torch.export.Dim('batch'), 2: torch.export.Dim('time', min=1, max=800)},
        'pitch': {0: torch.export.Dim('batch'), 2: torch.export.Dim('time', min=1, max=800)},
    }
)
```

#### Step 2: trtexec Conversion with Dynamic Shapes

```bash
# FP16 engine (recommended starting point for Jetson Thor)
trtexec \
  --onnx=consistency_svc.onnx \
  --saveEngine=consistency_svc_fp16.engine \
  --fp16 \
  --minShapes=noisy_mel:1x80x10,content:1x256x10,pitch:1x1x10 \
  --optShapes=noisy_mel:1x80x200,content:1x256x200,pitch:1x1x200 \
  --maxShapes=noisy_mel:1x80x800,content:1x256x800,pitch:1x1x800 \
  --workspace=4096 \
  --verbose

# INT8 engine (requires calibration data)
trtexec \
  --onnx=consistency_svc.onnx \
  --saveEngine=consistency_svc_int8.engine \
  --int8 \
  --fp16 \
  --calib=calibration_cache.bin \
  --minShapes=noisy_mel:1x80x10,content:1x256x10,pitch:1x1x10 \
  --optShapes=noisy_mel:1x80x200,content:1x256x200,pitch:1x1x200 \
  --maxShapes=noisy_mel:1x80x800,content:1x256x800,pitch:1x1x800 \
  --workspace=4096

# Benchmark latency
trtexec \
  --loadEngine=consistency_svc_fp16.engine \
  --shapes=noisy_mel:1x80x50,content:1x256x50,pitch:1x1x50 \
  --iterations=1000 \
  --warmUp=500 \
  --avgRuns=100
```

**Chunk size mapping (24kHz, hop=128):**
| Frames | Duration | Use Case |
|--------|----------|----------|
| 10 | ~53 ms | Minimum viable |
| 50 | ~267 ms | Real-time streaming chunk |
| 200 | ~1.07 s | Optimal batch |
| 800 | ~4.27 s | Maximum (full phrase) |

#### Step 3: INT8 Calibration

```python
import tensorrt as trt
import numpy as np

class AudioCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration using representative audio samples."""

    def __init__(self, calibration_data, batch_size=1, cache_file="calib.cache"):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data = calibration_data  # List of (mel, content, pitch) tuples
        self.current_index = 0

        # Allocate device memory for calibration inputs
        self.device_inputs = []
        for tensor in self.data[0]:
            self.device_inputs.append(
                cuda.mem_alloc(tensor.nbytes)
            )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None

        batch = self.data[self.current_index]
        for i, tensor in enumerate(batch):
            cuda.memcpy_htod(self.device_inputs[i], tensor)

        self.current_index += 1
        return [int(d) for d in self.device_inputs]

    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
```

#### Step 4: Python TensorRT Engine Builder (Alternative to trtexec)

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(
    onnx_path,
    engine_path,
    fp16=True,
    int8=False,
    calibrator=None,
    max_workspace_gb=4,
    dynamic_shapes=None,
):
    """Build TensorRT engine from ONNX model."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Set workspace
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        max_workspace_gb * (1 << 30)
    )

    # Precision flags
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator:
            config.int8_calibrator = calibrator

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX parsing failed")

    # Dynamic shapes via optimization profiles
    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for name, shapes in dynamic_shapes.items():
            profile.set_shape(name, shapes['min'], shapes['opt'], shapes['max'])
        config.add_optimization_profile(profile)

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    return serialized_engine


# Usage for consistency SVC model
dynamic_shapes = {
    'noisy_mel': {
        'min': (1, 80, 10),
        'opt': (1, 80, 200),
        'max': (1, 80, 800),
    },
    'content': {
        'min': (1, 256, 10),
        'opt': (1, 256, 200),
        'max': (1, 256, 800),
    },
    'pitch': {
        'min': (1, 1, 10),
        'opt': (1, 1, 200),
        'max': (1, 1, 800),
    },
}

build_engine(
    onnx_path="consistency_svc.onnx",
    engine_path="consistency_svc_fp16.engine",
    fp16=True,
    dynamic_shapes=dynamic_shapes,
)
```

#### Step 5: Streaming Inference Pipeline

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import threading
import queue
from collections import deque

class StreamingConsistencySVC:
    """Real-time streaming voice conversion using TensorRT."""

    def __init__(
        self,
        engine_path: str,
        chunk_size_ms: int = 250,  # Processing chunk size
        sample_rate: int = 24000,
        hop_size: int = 128,
        overlap_frames: int = 4,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.chunk_frames = int(chunk_size_ms * sample_rate / (1000 * hop_size))
        self.overlap_frames = overlap_frames

        # Load TensorRT engine
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Set dynamic shapes for chunk processing
        self.context.set_input_shape('noisy_mel', (1, 80, self.chunk_frames))
        self.context.set_input_shape('content', (1, 256, self.chunk_frames))
        self.context.set_input_shape('pitch', (1, 1, self.chunk_frames))

        # Allocate GPU buffers
        self._allocate_buffers()

        # CUDA stream for async execution
        self.stream = cuda.Stream()

        # Ring buffer for overlap-add
        self.output_buffer = deque(maxlen=10)

    def _allocate_buffers(self):
        """Pre-allocate input/output GPU memory."""
        self.d_noisy_mel = cuda.mem_alloc(
            1 * 80 * self.chunk_frames * 4  # float32
        )
        self.d_content = cuda.mem_alloc(
            1 * 256 * self.chunk_frames * 4
        )
        self.d_pitch = cuda.mem_alloc(
            1 * 1 * self.chunk_frames * 4
        )
        self.d_output = cuda.mem_alloc(
            1 * 80 * self.chunk_frames * 4
        )
        self.h_output = cuda.pagelocked_empty(
            (1, 80, self.chunk_frames), dtype=np.float32
        )

    def process_chunk(
        self,
        content_features: np.ndarray,
        pitch_features: np.ndarray,
        speaker_embedding: np.ndarray,
    ) -> np.ndarray:
        """Process a single audio chunk through the consistency model.

        For one-step generation, we initialize with noise and denoise in one pass.
        """
        # Initialize with Gaussian noise (one-step generation)
        noisy_mel = np.random.randn(1, 80, self.chunk_frames).astype(np.float32)
        noisy_mel *= 80.0  # Scale by sigma_max for one-step

        # Copy inputs to GPU (async)
        cuda.memcpy_htod_async(self.d_noisy_mel, noisy_mel, self.stream)
        cuda.memcpy_htod_async(self.d_content, content_features, self.stream)
        cuda.memcpy_htod_async(self.d_pitch, pitch_features, self.stream)

        # Execute inference
        bindings = [
            int(self.d_noisy_mel),
            int(self.d_content),
            int(self.d_pitch),
            int(self.d_output),
        ]
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.stream.handle
        )

        # Copy output back (async)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.copy()

    def get_latency_ms(self) -> float:
        """Benchmark single-chunk latency."""
        import time

        dummy_content = np.random.randn(1, 256, self.chunk_frames).astype(np.float32)
        dummy_pitch = np.random.randn(1, 1, self.chunk_frames).astype(np.float32)
        dummy_speaker = np.random.randn(1, 256).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.process_chunk(dummy_content, dummy_pitch, dummy_speaker)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            self.process_chunk(dummy_content, dummy_pitch, dummy_speaker)
            times.append((time.perf_counter() - start) * 1000)

        return np.median(times)
```

### 3.3 FP16/INT8 Quantization Strategies

#### Strategy Comparison for Audio Models

| Precision | Speedup vs FP32 | Accuracy Loss | Best For |
|-----------|-----------------|---------------|----------|
| FP32 | 1x | None | Debugging |
| FP16 | 2-3x | Negligible (<0.1 dB) | Production default |
| INT8 | 3-5x | Small (0.2-0.5 dB SNR) | Latency-critical |
| FP8 (Thor) | 4-6x | Very small | When TRT supports it |
| FP4/W4A16 | 5-8x | Moderate | Large models |

#### Quantization-Aware Training (Optional)

```python
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

# Enable quantization-aware training
quant_modules.initialize()

# Replace standard layers with quantized versions
# This inserts Q/DQ nodes for INT8 export
model = QuantizedConsistencyModel(...)
model.load_state_dict(pretrained_weights)

# Calibrate
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)

# Export quantized model to ONNX
torch.onnx.export(model, dummy_input, "model_int8.onnx", opset_version=17)
```

### 3.4 Latency Targets and Benchmarks

**Target: <50ms per chunk on Jetson Thor**

Based on available benchmarks from similar systems:

| System | Hardware | Model Type | Latency | Notes |
|--------|----------|------------|---------|-------|
| StreamVC | Mobile | Voice conversion | 70.8 ms | 20ms chunks at 50Hz |
| WhisperTRT | Jetson Orin Nano | ASR | ~180 ms | 20s audio, 3x faster than PyTorch |
| TRT-SE | Desktop GPU | Speech enhancement | ~5 ms/frame | 4x faster than ONNX Runtime |
| TTS (Tacotron2+WaveGlow) | NVIDIA T4 | TTS | 1.14s total | RTF=6.2x real-time |
| **Target: CoMoSVC on Thor** | **Jetson Thor** | **SVC** | **<50 ms** | **One-step, FP16** |

**Estimated latency breakdown for 250ms audio chunk (one-step CoMoSVC):**

| Component | Estimated Latency (FP16) |
|-----------|--------------------------|
| Feature extraction (ContentVec) | ~15 ms |
| Pitch estimation (DIO) | ~5 ms |
| Consistency model (1 step) | ~10-20 ms |
| HiFi-GAN vocoder | ~5-10 ms |
| Data transfer overhead | ~2-5 ms |
| **Total** | **~37-55 ms** |

### 3.5 Streaming Pipeline Design

```
Audio Input (microphone/file)
    |
    v
+------------------+
| Ring Buffer      | <-- Accumulate samples until chunk_size
| (overlap-add)   |
+------------------+
    |
    v
+------------------+
| Feature Extract  | <-- ContentVec encoder (TensorRT)
| (GPU, async)    |     + DIO pitch (CPU)
+------------------+
    |
    v
+------------------+
| Consistency SVC  | <-- One-step denoising (TensorRT)
| (GPU, 1 step)   |     sigma_max -> denoised mel
+------------------+
    |
    v
+------------------+
| HiFi-GAN Vocoder| <-- Mel to waveform (TensorRT)
| (GPU, async)    |
+------------------+
    |
    v
+------------------+
| Overlap-Add     | <-- Cross-fade consecutive chunks
| + Output Buffer |
+------------------+
    |
    v
Audio Output (speaker/file)
```

**Key design principles for <50ms latency:**
1. **Pipeline parallelism**: Feature extraction for chunk N+1 while converting chunk N
2. **CUDA streams**: Use separate streams for data transfer and compute
3. **Pre-allocated buffers**: Avoid dynamic allocation during inference
4. **Causal modification**: Replace non-causal BiDilConv padding with causal + lookahead
5. **Chunk overlap**: 4-frame overlap with cross-fade to avoid boundary artifacts

### 3.6 ONNX Compatibility Considerations

**Common issues when exporting voice conversion models:**
1. LSTMCell must be replaced with LSTM layers
2. 1D convolutions may need conversion to 2D for some TensorRT versions
3. Dynamic control flow (if/while) must be eliminated or traced
4. Custom CUDA kernels need ONNX custom op registration
5. Dropout must use fixed masks at export time (or be disabled)

**RVC-style ONNX export pattern:**
```python
# Voice conversion models often need attention modifications
class OnnxCompatibleAttention(nn.Module):
    """Modified attention for ONNX export compatibility."""
    def forward(self, x, mask=None):
        # Replace dynamic masking with static operations
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v)
```

---

## 4. Related Work: Consistency Models in Audio/Speech

### 4.1 CoMoSpeech (arXiv:2305.06908, ACM MM 2023)

**Foundational work for consistency-based speech synthesis.**

- Authors: Zhen Ye, Wei Xue, Xu Tan, Jie Chen, Qifeng Liu, Yike Guo
- Achieves single-step speech synthesis with >150x real-time on A100
- Uses Grad-TTS as teacher model architecture
- Same WaveNet-style decoder as CoMoSVC
- GitHub: https://github.com/zhenye234/CoMoSpeech

### 4.2 FlashSpeech (arXiv:2404.14700, ACM MM 2024)

**Adversarial consistency training without teacher model.**

Key innovations:
- **Latent consistency model**: Operates on codec latent space (before RVQ)
- **Adversarial consistency training**: Uses frozen WavLM as discriminator backbone
- **No teacher required**: Trains from scratch with adversarial guidance
- **Prosody generator**: 30 non-causal WaveNet layers for diversity

**Architecture:**
```
f_theta(x_sigma, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(x_sigma, sigma)
```

**Combined Loss:**
```
L_total = L_ct + lambda_adv * L_adv
lambda_adv = ||grad_theta(L_ct)|| / ||grad_theta(L_adv)||  (adaptive weighting)
```

**Training Details:**
- Stage 1: 650k steps, 8x H800, batch 20k frames/GPU, lr=3e-4
- Adversarial training activated after 600k iterations
- Stage 2: 150k steps (prosody refinement only)
- Discretization curriculum: N(k) = min(s0 * 2^(k/K'), s1) + 1, s0=10, s1=1280
- Noise: sigma_min=0.002, sigma_max=80, rho=7
- Distance: Pseudo-Huber with a=0.03

**Performance:**
- RTF: 0.02 on NVIDIA V100 (50x real-time)
- 1-2 sampling steps
- ~20x faster than comparable zero-shot TTS systems

### 4.3 ECTSpeech (arXiv:2510.05984, ACM MM Asia 2025)

**Easy Consistency Tuning for speech synthesis.**

- Progressively tightens consistency constraints on pre-trained diffusion model
- Reduces training complexity vs. full distillation
- No separate teacher training phase needed
- Represents latest evolution (2025) of consistency approach

### 4.4 Consistency Models Made Easy (arXiv:2406.14548, ICLR 2025)

**Theoretical unification of diffusion and consistency models.**

- Reformulates consistency condition as differential equation
- Progressive tightening of consistency constraint during training
- Shows diffusion models are special case of consistency models
- Provides smooth interpolation between diffusion and consistency

### 4.5 Summary: Evolution of Consistency Models for Audio

| Year | Paper | Approach | Teacher? | Steps | Speed |
|------|-------|----------|----------|-------|-------|
| 2023 | CoMoSpeech | Distillation | Yes | 1 | 150x RT |
| 2024 | CoMoSVC | Distillation | Yes | 1 | ~50x RT |
| 2024 | FlashSpeech | Adversarial CT | No | 1-2 | 50x RT |
| 2025 | ECTSpeech | Progressive tuning | Partial | 1 | ~100x RT |

---

## 5. Implementation Roadmap for AutoVoice

### 5.1 Phase 1: Teacher Model Training

1. Implement BiDilConv decoder (20 blocks, 2 dilation cycles of [1,2,4,8,16,32,64,128,256,512])
2. Implement EDM preconditioning with Karras noise schedule
3. Train teacher diffusion model on target dataset (1M iterations)
4. Verify teacher quality with multi-step sampling (40 steps)

### 5.2 Phase 2: Consistency Distillation

1. Implement CTLoss_D with EMA target network (mu=0.95)
2. Freeze encoder, train only decoder via consistency distillation
3. Use Euler ODE solver for teacher -> student transfer
4. Validate one-step generation quality vs. teacher

### 5.3 Phase 3: TensorRT Deployment on Jetson Thor

1. Export consistency model to ONNX with dynamic shapes
2. Build FP16 TensorRT engine with trtexec
3. Implement streaming inference pipeline with overlap-add
4. INT8 calibration with representative audio data
5. Benchmark: target <50ms per 250ms chunk

### 5.4 Phase 4: Optimization

1. Profile bottlenecks (nvprof/Nsight Systems)
2. Consider causal modification for streaming compatibility
3. Implement pipeline parallelism (feature extraction || inference)
4. Explore FlashSpeech-style adversarial training to skip teacher

---

## References

1. Lu et al., "CoMoSVC: Consistency Model-based Singing Voice Conversion" (arXiv:2401.01792)
2. Song et al., "Consistency Models" (arXiv:2303.01469, ICML 2023)
3. Song & Dhariwal, "Improved Techniques for Training Consistency Models" (arXiv:2310.14189, ICLR 2024)
4. Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (arXiv:2206.00364)
5. Ye et al., "CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model" (arXiv:2305.06908, ACM MM 2023)
6. Ye et al., "FlashSpeech: Efficient Zero-Shot Speech Synthesis" (arXiv:2404.14700, ACM MM 2024)
7. Zheng et al., "ECTSpeech: Enhancing Efficient Speech Synthesis via Easy Consistency Tuning" (arXiv:2510.05984)
8. Song & Dhariwal, "Consistency Models Made Easy" (arXiv:2406.14548, ICLR 2025)
9. van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (arXiv:1609.03499)
10. NVIDIA, "Jetson Thor: The Ultimate Platform for Physical AI" (2025)
11. NVIDIA, "How to Deploy Real-Time Text-to-Speech Applications on GPUs Using TensorRT" (Developer Blog)
12. Amphion Project, "DiffComoSVC" (https://github.com/open-mmlab/Amphion)
13. Lin et al., "StreamVC: Real-Time Low-Latency Voice Conversion" (arXiv:2401.03078)
