# Backend Audit & Frontend Gap Analysis

**Track:** frontend-parity_20260129
**Date:** 2026-01-29
**Status:** Complete

---

## Executive Summary

This audit identifies 47 backend capabilities without frontend UI controls, organized by priority:
- **Critical (Training)**: 11 parameters - LoRA, EWC, learning rates
- **High (Inference)**: 14 parameters - pitch, vocoder, encoder selection
- **Medium (System)**: 12 parameters - GPU metrics, model management
- **Low (Advanced)**: 10 parameters - batch, visualization, debug

---

## 1. Backend API Endpoints

### 1.1 Voice Conversion (`/api/v1/`)

| Endpoint | Method | Frontend Status | Gap |
|----------|--------|-----------------|-----|
| `/convert/song` | POST | ✅ Partial | Missing: `return_stems`, `output_quality` presets UI |
| `/convert/status/{id}` | GET | ✅ Exists | - |
| `/convert/download/{id}` | GET | ✅ Exists | - |
| `/convert/cancel/{id}` | POST | ✅ Exists | - |
| `/convert/metrics/{id}` | GET | ❌ Missing | Quality metrics display needed |

### 1.2 Voice Profiles (`/api/v1/`)

| Endpoint | Method | Frontend Status | Gap |
|----------|--------|-----------------|-----|
| `/voice/clone` | POST | ✅ Exists | - |
| `/voice/profiles` | GET | ✅ Exists | - |
| `/voice/profiles/{id}` | GET | ✅ Exists | - |
| `/voice/profiles/{id}` | DELETE | ✅ Exists | - |

### 1.3 Profile Management (`/api/v1/profiles/`)

| Endpoint | Method | Frontend Status | Gap |
|----------|--------|-----------------|-----|
| `/profiles` | POST | ❌ Missing | Profile creation with name/settings |
| `/profiles` | GET | ❌ Missing | List by user_id |
| `/profiles/{id}` | GET | ❌ Missing | Full profile details |
| `/profiles/{id}` | PUT | ❌ Missing | Update name/settings/model |
| `/profiles/{id}` | DELETE | ❌ Missing | - |
| `/profiles/{id}/samples` | POST | ❌ Missing | Upload training samples |
| `/profiles/{id}/samples` | GET | ❌ Missing | List samples |
| `/profiles/{id}/samples/{sid}` | GET | ❌ Missing | Get sample |
| `/profiles/{id}/samples/{sid}` | DELETE | ❌ Missing | Delete sample |

### 1.4 System Monitoring (`/api/v1/`)

| Endpoint | Method | Frontend Status | Gap |
|----------|--------|-----------------|-----|
| `/health` | GET | ✅ Partial | Component status not displayed |
| `/gpu/metrics` | GET | ❌ Missing | Real-time GPU metrics |
| `/kernels/metrics` | GET | ❌ Missing | CUDA kernel performance |
| `/system/info` | GET | ✅ Fixed | Now transforms data correctly |

### 1.5 Audio Devices (`/api/v1/`)

| Endpoint | Method | Frontend Status | Gap |
|----------|--------|-----------------|-----|
| `/devices/list` | GET | ✅ Exists | AudioDeviceSelector exists |
| `/devices/config` | GET | ✅ Exists | - |
| `/devices/config` | POST | ✅ Exists | - |

---

## 2. Backend Configurable Parameters

### 2.1 Training Configuration (TrainingJobManager)

**Source:** `src/auto_voice/training/job_manager.py`

| Parameter | Type | Default | Frontend UI | Gap |
|-----------|------|---------|-------------|-----|
| `lora_rank` | int | 8 | ❌ Missing | Slider 1-64 needed |
| `lora_alpha` | int | 16 | ❌ Missing | Input field needed |
| `lora_dropout` | float | 0.1 | ❌ Missing | Slider 0-0.5 needed |
| `lora_target_modules` | list | ["q_proj", "v_proj", "content_encoder"] | ❌ Missing | Multi-select needed |
| `learning_rate` | float | 1e-4 | ❌ Missing | Scientific notation input |
| `batch_size` | int | 4 | ❌ Missing | Selector needed |
| `epochs` | int | 10 | ❌ Missing | Slider 1-100 needed |
| `warmup_steps` | int | 100 | ❌ Missing | Input needed |
| `max_grad_norm` | float | 1.0 | ❌ Missing | Input needed |
| `use_ewc` | bool | True | ❌ Missing | Toggle needed |
| `ewc_lambda` | float | 1000.0 | ❌ Missing | Input needed (when EWC enabled) |
| `use_prior_preservation` | bool | False | ❌ Missing | Toggle needed |
| `prior_loss_weight` | float | 0.5 | ❌ Missing | Slider needed |

### 2.2 Conversion Configuration (SingingConversionPipeline)

**Source:** `src/auto_voice/inference/singing_conversion_pipeline.py`

| Parameter | Type | Default | Frontend UI | Gap |
|-----------|------|---------|-------------|-----|
| `vocal_volume` | float | 1.0 | ✅ Partial | Range 0-2 slider exists |
| `instrumental_volume` | float | 0.9 | ✅ Partial | Range 0-2 slider exists |
| `pitch_shift` | float | 0.0 | ✅ Partial | Range -12 to +12 exists |
| `preset` | str | "balanced" | ❌ Missing | Dropdown: draft/fast/balanced/high/studio |
| `return_stems` | bool | False | ❌ Missing | Toggle needed |
| `preserve_techniques` | bool | True | ❌ Missing | Toggle for vibrato/melisma detection |

**Preset Details:**
```python
PRESETS = {
    'draft': {'n_steps': 10, 'denoise': 0.3},
    'fast': {'n_steps': 20, 'denoise': 0.5},
    'balanced': {'n_steps': 50, 'denoise': 0.7},
    'high': {'n_steps': 100, 'denoise': 0.8},
    'studio': {'n_steps': 200, 'denoise': 0.9},
}
```

### 2.3 Model Selection

**Source:** `src/auto_voice/inference/model_manager.py`, `singing_conversion_pipeline.py`

| Parameter | Options | Frontend UI | Gap |
|-----------|---------|-------------|-----|
| `encoder_backend` | hubert, contentvec | ❌ Missing | Dropdown needed |
| `encoder_type` | linear, conformer | ❌ Missing | Dropdown needed |
| `vocoder_type` | hifigan, bigvgan | ❌ Missing | Dropdown needed |
| `speaker_id` | dynamic | ❌ Missing | Profile selector needed |

### 2.4 Audio Router Configuration

**Source:** `src/auto_voice/web/audio_router.py`

| Parameter | Type | Default | Frontend UI | Gap |
|-----------|------|---------|-------------|-----|
| `speaker_gain` | float | 1.0 | ❌ Missing | Slider 0-2 needed |
| `headphone_gain` | float | 1.0 | ❌ Missing | Slider 0-2 needed |
| `voice_gain` | float | 1.0 | ❌ Missing | Slider 0-2 needed |
| `instrumental_gain` | float | 0.8 | ❌ Missing | Slider 0-2 needed |
| `speaker_enabled` | bool | True | ❌ Missing | Toggle needed |
| `headphone_enabled` | bool | True | ❌ Missing | Toggle needed |
| `speaker_device` | int | None | ❌ Missing | Device selector needed |
| `headphone_device` | int | None | ❌ Missing | Device selector needed |

---

## 3. GPU Metrics Response Schema

**Endpoint:** `GET /api/v1/gpu/metrics`

```typescript
interface GPUMetrics {
  available: boolean
  device_count: number
  devices: Array<{
    index: number
    name: string
    memory_total_gb: number
    memory_used_gb: number
    memory_free_gb: number
    utilization_percent: number | null
    temperature_c: number | null
  }>
  note?: string  // When pynvml not available
}
```

**Frontend Gap:** Need `GPUMetricsPanel.tsx` with:
- Real-time utilization chart (1-2s refresh)
- Memory usage bar (model/cache/free breakdown)
- Temperature monitoring with alerts (>80°C warning)

---

## 4. Frontend Components Audit

### 4.1 Existing Components

| Component | Location | Status | Missing Features |
|-----------|----------|--------|------------------|
| `GPUMonitor.tsx` | components/ | ✅ Updated | Live charts, temperature |
| `AudioDeviceSelector.tsx` | components/ | ✅ Exists | Karaoke dual-channel routing |
| `AudioWaveform.tsx` | components/ | ✅ Exists | - |
| `RealtimeWaveform.tsx` | components/ | ✅ Exists | - |

### 4.2 Existing Pages

| Page | Location | Status | Missing Features |
|------|----------|--------|------------------|
| `KaraokePage.tsx` | pages/ | ✅ Exists | Training controls, preset selector |
| `ConversionHistoryPage.tsx` | pages/ | ✅ Exists | Inline playback, comparison, metrics |
| `SystemStatusPage.tsx` | pages/ | ✅ Exists | Model manager, TensorRT controls |
| `VoiceProfilePage.tsx` | pages/ | ✅ Exists | Sample upload, training config |
| `HelpPage.tsx` | pages/ | ✅ Exists | - |

### 4.3 Missing Components (Required)

| Component | Priority | Purpose |
|-----------|----------|---------|
| `TrainingConfigPanel.tsx` | Critical | LoRA/EWC parameter controls |
| `TrainingJobQueue.tsx` | Critical | Job list with progress/cancel |
| `LossCurveChart.tsx` | High | Training visualization |
| `InferenceConfigPanel.tsx` | High | Pitch, volume, encoder/vocoder |
| `QualityPresetSelector.tsx` | High | draft/fast/balanced/high/studio |
| `ModelManager.tsx` | Medium | Load/unload models |
| `TensorRTControls.tsx` | Medium | Precision, rebuild |
| `BatchProcessingQueue.tsx` | Medium | Multi-file conversion |
| `SpectrogramViewer.tsx` | Low | Before/after comparison |
| `DebugPanel.tsx` | Low | Logs, diagnostics |
| `PresetManager.tsx` | Low | Save/load presets |

---

## 5. API Service Gaps

### 5.1 Missing Methods in `api.ts`

```typescript
// Training
getTrainingConfig(profileId: string): Promise<TrainingConfig>
updateTrainingConfig(profileId: string, config: Partial<TrainingConfig>): Promise<TrainingConfig>

// GPU
getGPUMetrics(): Promise<GPUMetrics>  // Currently returns wrong endpoint
getKernelMetrics(): Promise<KernelMetric[]>

// Models
getLoadedModels(): Promise<LoadedModel[]>
loadModel(modelType: string, path: string): Promise<void>
unloadModel(modelType: string): Promise<void>
rebuildTensorRT(precision: 'fp32' | 'fp16' | 'int8'): Promise<void>

// Presets
getPresets(): Promise<Preset[]>
savePreset(name: string, config: ConversionConfig): Promise<Preset>
loadPreset(presetId: string): Promise<ConversionConfig>
deletePreset(presetId: string): Promise<void>

// Samples
uploadSample(profileId: string, audioFile: File): Promise<TrainingSample>
listSamples(profileId: string): Promise<TrainingSample[]>
deleteSample(profileId: string, sampleId: string): Promise<void>
```

### 5.2 Missing TypeScript Interfaces

```typescript
interface TrainingConfig {
  lora_rank: number
  lora_alpha: number
  lora_dropout: number
  lora_target_modules: string[]
  learning_rate: number
  batch_size: number
  epochs: number
  warmup_steps: number
  max_grad_norm: number
  use_ewc: boolean
  ewc_lambda: number
  use_prior_preservation: boolean
  prior_loss_weight: number
}

interface ConversionConfig {
  vocal_volume: number
  instrumental_volume: number
  pitch_shift: number
  preset: 'draft' | 'fast' | 'balanced' | 'high' | 'studio'
  return_stems: boolean
  preserve_techniques: boolean
  encoder_backend: 'hubert' | 'contentvec'
  vocoder_type: 'hifigan' | 'bigvgan'
}

interface TrainingSample {
  id: string
  profile_id: string
  audio_path: string
  duration_seconds: number
  sample_rate: number
  created: string
}

interface LoadedModel {
  type: string
  name: string
  path: string
  memory_mb: number
  loaded_at: string
}

interface Preset {
  id: string
  name: string
  config: ConversionConfig
  created_at: string
}

interface QualityMetrics {
  pitch_accuracy: {
    rmse_hz: number
    correlation: number
    mean_error_cents: number
  }
  speaker_similarity: {
    cosine_similarity: number
    embedding_distance: number
  }
  naturalness: {
    spectral_distortion: number
    mos_estimate: number
  }
}
```

---

## 6. Priority Matrix

### Tier 1: Critical (Training Controls)
1. TrainingConfigPanel with LoRA controls
2. EWC toggle and lambda input
3. Learning rate scientific notation input
4. Training job queue with progress
5. Loss curve visualization

### Tier 2: High (Inference Controls)
1. Quality preset selector (5 presets)
2. Encoder selector (HuBERT/ContentVec)
3. Vocoder selector (HiFiGAN/BigVGAN)
4. Stem separation toggle
5. Technique preservation toggle

### Tier 3: Medium (System Controls)
1. Real-time GPU metrics panel
2. Model load/unload manager
3. TensorRT precision selector
4. Batch processing queue
5. Audio router gain controls

### Tier 4: Low (Polish)
1. Spectrogram before/after viewer
2. Quality metrics display
3. Debug log viewer
4. Preset save/load
5. Configuration export/import

---

## 7. Recommended Implementation Order

1. **Phase 4** (API Service): Add all missing TypeScript interfaces and API methods
2. **Phase 5** (Training UI): TrainingConfigPanel, TrainingJobQueue, LossCurveChart
3. **Phase 6** (Inference UI): InferenceConfigPanel, PresetSelector, encoder/vocoder selectors
4. **Phase 7** (System UI): GPUMetricsPanel, ModelManager, TensorRTControls
5. **Phase 8** (Advanced UI): BatchProcessing, PresetManager
6. **Phase 9** (Visualization): SpectrogramViewer, quality metrics
7. **Phase 10** (Debug): DebugPanel, config export

---

## 8. Test Coverage Gaps

Backend modules needing frontend integration tests:
- Training job creation and monitoring
- Model loading/unloading
- Preset save/load
- Batch conversion
- Real-time GPU metrics polling
- WebSocket progress events

---

_Generated by conductor:implement for track frontend-parity_20260129_
