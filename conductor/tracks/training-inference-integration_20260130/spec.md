# Training-to-Inference Integration

**Track ID:** training-inference-integration_20260130
**Created:** 2026-01-30
**Priority:** High

## Problem Statement

The voice swap quality test produces high-pitched ringing noise instead of converted vocals because:
1. The SOTA inference pipeline (SOTAConversionPipeline) initializes all neural network components with **random weights**
2. The training pipeline (LoRA fine-tuning, TrainingJobManager) exists but is **not connected** to inference
3. There's no mechanism to train models from singer samples and load those weights during conversion

## Current Architecture Gap

```
[Audio Samples] → [TrainingJobManager] → [LoRA Fine-tuning] → [Trained Weights] → ???
                                                                        ↓
                                                                  NOT CONNECTED
                                                                        ↓
[Voice Profile] → [SOTAConversionPipeline] → [Random Weights] → [Noise Output]
```

## Desired Architecture

```
[Audio Samples] → [TrainingJobManager] → [LoRA Fine-tuning] → [Trained Weights]
                                                                        ↓
                                                              [Profile Storage]
                                                                        ↓
[Voice Profile] → [SOTAConversionPipeline] → [Load Weights] → [Quality Output]
```

## Requirements

### 1. Model Weight Storage
- Store trained LoRA adapter weights per voice profile
- Include model version tracking (created in Phase 4)
- Support loading specific versions for A/B comparison

### 2. Training Trigger Flow
- Given: Audio file(s) for a singer
- Action: Automatically register as training samples, create training job
- Output: Trained LoRA adapters for decoder components

### 3. Inference Weight Loading
- SOTAConversionPipeline must accept a profile_id
- Load trained LoRA weights for that profile's model
- Apply LoRA adapters to decoder during inference

### 4. Voice Swap Test Integration
- Test should: create profile → add samples → train → convert
- Verify output is actual voice conversion, not noise
- Measure quality metrics on trained conversion

## Existing Components to Connect

| Component | Location | Status |
|-----------|----------|--------|
| VoiceCloner | `inference/voice_cloner.py` | Creates profiles with embeddings |
| TrainingJobManager | `training/job_manager.py` | Creates/runs training jobs |
| LoRAAdapter | `training/fine_tuning.py` | LoRA fine-tuning implementation |
| ModelVersionManager | `training/model_versioning.py` | Tracks model versions |
| CoMoSVCDecoder | `models/svc_decoder.py` | Decoder (needs LoRA injection) |
| SOTAConversionPipeline | `inference/sota_pipeline.py` | Main inference (needs weight loading) |

## Acceptance Criteria

1. `create_voice_profile(audio)` also triggers training job creation
2. `TrainingJobManager.run_job()` produces saved LoRA weights
3. `SOTAConversionPipeline.convert(profile_id=X)` loads trained weights
4. Voice swap test produces audible voice conversion (not noise)
5. Quality metrics show meaningful values (not NaN or extreme outliers)

## Web Interface Integration

### Seamless User Flow
1. **Upload Audio** → Profile created + training auto-triggered
2. **Training Progress** → Real-time status via WebSocket (TrainingJobQueue component)
3. **Model Ready** → UI notification when training complete
4. **Convert Song** → Automatically uses trained model for that profile

### API Endpoints
- `POST /api/v1/profiles` - Create profile AND trigger training
- `GET /api/v1/profiles/{id}/model/status` - Check if model is trained
- `POST /api/v1/convert/song` - Uses trained model if available

### Frontend Components
- `VoiceProfilePage.tsx` - Show training status per profile
- `TrainingJobQueue.tsx` - Real-time training progress
- `ConversionPage.tsx` - Warn if profile has no trained model

### WebSocket Events
- `training.started` - Training job began
- `training.progress` - Epoch/loss updates
- `training.completed` - Model ready for inference
- `training.failed` - Training error with details

## Out of Scope

- Real-time training during karaoke sessions (existing continuous learning handles this)
- Multi-GPU training
- External model download/hosting

## Files to Modify

### Backend
1. `src/auto_voice/training/fine_tuning.py` - Add weight saving to profile storage
2. `src/auto_voice/inference/sota_pipeline.py` - Add LoRA loading from profile
3. `src/auto_voice/models/svc_decoder.py` - Add LoRA injection points
4. `src/auto_voice/web/api.py` - Auto-trigger training on profile creation
5. `src/auto_voice/inference/voice_cloner.py` - Connect to TrainingJobManager

### Frontend
6. `frontend/src/pages/VoiceProfilePage.tsx` - Show model training status
7. `frontend/src/components/TrainingJobQueue.tsx` - Real-time progress display
8. `frontend/src/services/api.ts` - Add model status endpoints

### Tests
9. `tests/quality_samples/run_voice_swap_test.py` - Add training step before conversion
