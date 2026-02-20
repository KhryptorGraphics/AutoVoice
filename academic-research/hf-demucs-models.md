# HuggingFace: Demucs Models & Audio Separation Research

## Research Source
- HuggingFace model_search, paper_search, dataset_search, space_search
- Focus: Demucs/HTDemucs models, ONNX exports, competing architectures

---

## 1. Available Models

### HTDemucs ONNX/Optimized Exports (Priority for TensorRT)

| Model | Format | Notes | Link |
|-------|--------|-------|------|
| ModernMube/HTDemucs_onnx | ONNX | Direct ONNX export | [HF](https://hf.co/ModernMube/HTDemucs_onnx) |
| gentij/htdemucs-ort | ONNX Runtime | Optimized for ORT inference | [HF](https://hf.co/gentij/htdemucs-ort) |
| Intel/demucs-openvino | OpenVINO | Intel optimized | [HF](https://hf.co/Intel/demucs-openvino) |
| CrazeDigger/htdemucs | libtorch/CoreML | Mobile deployment | [HF](https://hf.co/CrazeDigger/htdemucs) |

### HTDemucs PyTorch Models

| Model | Notes | Link |
|-------|-------|------|
| jarredou/HTDemucs_Similarity_Extractor_by_wesleyr36 | Similarity-based | [HF](https://hf.co/jarredou/HTDemucs_Similarity_Extractor_by_wesleyr36) |
| dokodesuka/htdemucs_ft | Fine-tuned variant | [HF](https://hf.co/dokodesuka/htdemucs_ft) |
| pablebe/htdemucs | Standard, arxiv:2507.11427 | [HF](https://hf.co/pablebe/htdemucs) |

### Competing Architectures

| Model | Architecture | SDR | Notes |
|-------|-------------|-----|-------|
| Eddycrack864/Music-Source-Separation-Training | BS-RoFormer | 9.80 dB | Current SOTA |
| Awais/Audio_Source_Separation | ConvTasNet | - | Lightweight |

---

## 2. Key Papers

### Core Demucs Papers

| Paper | Year | Contribution |
|-------|------|--------------|
| Music Source Separation in the Waveform Domain (Défossez) | 2019 | Original Demucs |
| Hybrid Spectrogram and Waveform Source Separation (Défossez) | 2021 | Hybrid approach |
| Hybrid Transformers for Music Source Separation (Rouard et al.) | 2022 | HTDemucs with cross-domain attention |

### Competing & Newer Approaches

| Paper | Year | Key Result |
|-------|------|-----------|
| **BS-RoFormer** | 2023 | 9.80 dB SDR — current SOTA for music separation |
| **BSRNN** | 2023 | Band-split RNN, strong vocal separation |
| **Banquet** | 2023 | Query-based multi-source separation |
| **SAM Audio** (Meta) | Dec 2025 | Foundation model for general audio separation |
| **HS-TasNet** | Feb 2024 | Real-time low-latency (23ms) separation |

### Singing Voice Specific

| Paper | Year | Relevance |
|-------|------|-----------|
| MedleyVox (Byun et al.) | 2023 | Singing voice separation benchmark |
| JaCappella | 2023 | A cappella corpus for source separation |
| SVS-related papers | 2024-25 | Voice conversion pre-processing |

---

## 3. Datasets

No HuggingFace-hosted datasets found. Standard datasets from papers:

| Dataset | Size | Stems | Notes |
|---------|------|-------|-------|
| MUSDB18-HQ | 150 tracks | 4 | Standard benchmark |
| MoisesDB | 240+ tracks | Up to 6 | Extended stems |
| DSD100 | 100 tracks | 4 | Legacy benchmark |
| iKala | 352 clips | 2 | Singing voice focused |
| MedleyVox | - | Vocals | Multiple singers |
| JaCappella | - | Vocal parts | A cappella |

---

## 4. HuggingFace Spaces (Demos)

| Space | Likes | Interface | Link |
|-------|-------|-----------|------|
| r3gm/Audio_separator | 347 | Audio separation UI | [HF](https://hf.co/spaces/r3gm/Audio_separator) |
| abidlabs/music-separation | 247 | Gradio demo | [HF](https://hf.co/spaces/abidlabs/music-separation) |

---

## 5. Integration Strategy for AutoVoice (AV-008)

### Primary Approach: pip install demucs
```python
from demucs.api import Separator
separator = Separator(model="htdemucs", device="cuda", segment=12)
_, separated = separator.separate_audio_file("song.mp3")
vocals = separated["vocals"]
```

### TensorRT Optimization Path (for production):
1. Use `ModernMube/HTDemucs_onnx` as starting point
2. Convert ONNX → TensorRT with dynamic shapes
3. FP16 precision for Jetson Thor

```bash
# Download ONNX model
huggingface-cli download ModernMube/HTDemucs_onnx --local-dir models/onnx/

# Convert to TensorRT
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/onnx/htdemucs.onnx \
    --saveEngine=models/optimized/htdemucs.trt \
    --fp16 \
    --minShapes=input:1x2x88200 \
    --optShapes=input:1x2x441000 \
    --maxShapes=input:1x2x2646000
```

### Real-Time Alternative: HS-TasNet
- 23ms latency, suitable for streaming
- Trade-off: lower quality than HTDemucs (by ~1-2 dB SDR)
- Consider for live performance scenarios

### Future Watch: SAM Audio (Meta, Dec 2025)
- Foundation model for general audio separation
- May supersede task-specific models
- Monitor for open-source release

---

## 6. Performance Comparison

| Model | SDR (vocals) | Latency | Memory | Best For |
|-------|-------------|---------|--------|----------|
| htdemucs | 9.0 dB | ~2s/song | ~2GB | Offline quality |
| htdemucs_ft | 9.2 dB | ~8s/song | ~2GB | Maximum quality |
| BS-RoFormer | 9.8 dB | ~3s/song | ~3GB | Research SOTA |
| HS-TasNet | ~7.5 dB | 23ms | ~500MB | Real-time |
| HTDemucs TRT | ~9.0 dB | ~0.5s/song | ~1GB | Production edge |

**Recommendation**: Start with `htdemucs` via pip for correctness, then optimize to TensorRT for production latency on Jetson Thor.
