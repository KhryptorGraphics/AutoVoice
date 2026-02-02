# Cross-Context Dependency Analysis

**Generated:** 2026-02-01
**Coordinator:** Cross-Context Coordinator Agent
**Purpose:** Map and verify all cross-context dependencies between tracks

## Executive Summary

AutoVoice has 4 major dependency chains:
1. **Training → Inference** - Trained models used by conversion pipelines
2. **Diarization → Profiles** - Speaker identification feeds profile creation
3. **Frontend → Backend** - React UI calls Flask/WebSocket APIs
4. **SOTA Innovations → Pipelines** - New pipeline types integrate with infrastructure

**Status:** ✅ All critical dependencies verified and working
**Broken Links:** 1 (quality_shortcut pipeline not in PipelineFactory)
**Missing Tests:** 2 integration paths lack E2E coverage

---

## Dependency Chain 1: Training → Inference

### Overview
Trained LoRA adapters from `trainer.py` must be loadable by all pipeline types.

### Source Components
- **trainer.py** - Produces LoRA adapters at `data/trained_models/{profile_id}_adapter.pt`
- **AdapterManager** - Loads and manages adapters
- **VoiceProfileStore** - Stores profile metadata with `has_trained_model` flag

### Consumer Components
- **RealtimePipeline** - Uses adapters for karaoke
- **SOTAConversionPipeline** - Uses adapters for quality conversion
- **SeedVCPipeline** - Uses adapters via AdapterBridge
- **MeanVCPipeline** - Uses adapters via AdapterBridge

### Integration Points

| Integration | File | Status | Tested |
|-------------|------|--------|--------|
| LoRA save format | `training/trainer.py:L145` | ✅ Complete | ✅ Yes |
| AdapterManager.load_adapter() | `models/adapter_manager.py:L139` | ✅ Complete | ✅ Yes |
| PipelineFactory.get_pipeline() | `inference/pipeline_factory.py:L70` | ✅ Complete | ✅ Yes |
| AdapterBridge (Seed-VC) | `inference/adapter_bridge.py:L40` | ✅ Complete | ✅ Yes |
| API: POST /api/v1/convert/song | `web/api.py:L307` | ✅ Complete | ✅ Yes |
| API: GET /profiles/{id}/model | `web/api.py:L1088` | ✅ Complete | ✅ Yes |

### Data Flow
```
trainer.py
  ↓ saves
{profile_id}_adapter.pt (OrderedDict with lora_weights, speaker_embedding, config)
  ↓ loaded by
AdapterManager.load_adapter(profile_id)
  ↓ applied to
Pipeline (via .set_speaker() or AdapterBridge)
  ↓ used in
POST /api/v1/convert/song with pipeline_type param
```

### Tests Covering This Chain
- ✅ `test_adapter_manager.py` - AdapterManager CRUD
- ✅ `test_adapter_integration_e2e.py` - Full training→inference flow
- ✅ `test_pipeline_factory.py` - Pipeline creation with adapters
- ✅ `test_continuous_training_e2e.py` - Training job completion

### Verification Status: ✅ VERIFIED

All pipelines correctly load trained adapters. The `has_trained_model` flag is set on profile completion and checked by the frontend.

---

## Dependency Chain 2: Diarization → Profiles

### Overview
Speaker diarization segments audio by speaker, then assigns segments to voice profiles.

### Source Components
- **speaker_diarization.py** - Detects speakers, creates WavLM embeddings
- **SpeakerMatcher** - Clusters speakers across tracks
- **YouTubeMetadataFetcher** - Detects featured artists from titles

### Consumer Components
- **VoiceProfileStore** - Receives assigned speaker embeddings
- **DiarizationExtractor** - Extracts per-speaker audio
- **FileOrganizer** - Organizes files by identified artist

### Integration Points

| Integration | File | Status | Tested |
|-------------|------|--------|--------|
| Speaker detection | `audio/speaker_diarization.py:L150` | ✅ Complete | ✅ Yes |
| Cross-track clustering | `audio/speaker_matcher.py:L88` | ✅ Complete | ✅ Yes |
| Featured artist detection | `audio/youtube_metadata.py:L45` | ✅ Complete | ✅ Yes |
| Segment extraction | `audio/diarization_extractor.py:L60` | ✅ Complete | ✅ Yes |
| File organization | `audio/file_organizer.py:L40` | ✅ Complete | ✅ Yes |
| API: POST /speakers/identify | `web/speaker_api.py:L474` | ✅ Complete | ✅ Yes |
| API: GET /speakers/clusters | `web/speaker_api.py:L258` | ✅ Complete | ✅ Yes |

### Data Flow
```
YouTube Download
  ↓
YouTubeMetadataFetcher.detect_featured_artists()
  ↓
speaker_diarization.py → WavLM embeddings per segment
  ↓
SpeakerMatcher.cluster_speakers() → Cross-track clusters
  ↓
SpeakerMatcher.auto_match_clusters_to_artists() → Assign to featured artists
  ↓
DiarizationExtractor.extract_segments() → Per-speaker WAV files
  ↓
VoiceProfileStore.create_from_diarization() → Profile with speaker embedding
```

### Tests Covering This Chain
- ✅ `test_speaker_diarization.py` - Diarization E2E
- ✅ `test_youtube_pipeline.py` - YouTube → Diarization
- ✅ `test_frontend_integration_e2e.py` - UI diarization flow
- ❌ **MISSING:** `test_speaker_clustering_e2e.py` - Cross-track clustering full flow

### Verification Status: ⚠️ MOSTLY VERIFIED

Core diarization works. Cross-track clustering exists but lacks E2E test from multiple YouTube downloads → auto-profile creation.

**Action:** Create `test_speaker_clustering_e2e.py` to test multi-artist detection.

---

## Dependency Chain 3: Frontend → Backend

### Overview
React frontend calls Flask API and WebSocket endpoints for all features.

### API Integration Matrix

| Feature | Frontend Component | Backend Endpoint | Status | Tested |
|---------|-------------------|------------------|--------|--------|
| Profile list | VoiceProfilePage | GET /api/v1/profiles | ✅ Complete | ✅ Yes |
| Profile model status | VoiceProfilePage | GET /profiles/{id}/model | ✅ Complete | ✅ Yes |
| Training job submit | TrainingConfigPanel | POST /api/v1/training/jobs | ✅ Complete | ✅ Yes |
| Training status | LiveTrainingMonitor | GET /training/jobs/{id} | ✅ Complete | ✅ Yes |
| Song conversion | ConversionPage | POST /api/v1/convert/song | ✅ Complete | ✅ Yes |
| Pipeline selection | PipelineSelector | (param in convert) | ✅ Complete | ✅ Yes |
| Karaoke streaming | KaraokePage | WebSocket /ws/karaoke | ✅ Complete | ✅ Yes |
| Speaker diarization | SpeakerIdentificationPanel | POST /speakers/identify | ✅ Complete | ✅ Yes |
| Cluster management | SpeakerAssignmentPanel | GET /speakers/clusters | ✅ Complete | ✅ Yes |
| YouTube download | YouTubeDownloadPage | POST /api/v1/youtube/download | ✅ Complete | ✅ Yes |

### Pipeline Type Coverage

Frontend `PipelineSelector.tsx` defines 5 pipeline types:

```typescript
type PipelineType = 'realtime' | 'quality' | 'quality_seedvc' | 'realtime_meanvc' | 'quality_shortcut'
```

Backend `pipeline_factory.py` supports 4:

```python
PipelineType = Literal['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc']
```

**❌ BROKEN LINK:** `quality_shortcut` not implemented in PipelineFactory

### Tests Covering This Chain
- ✅ `test_frontend_integration_e2e.py` - Full UI → API flow
- ✅ `test_karaoke_integration.py` - WebSocket streaming
- ❌ **MISSING:** `test_pipeline_selector_e2e.py` - All 5 pipeline types

### Verification Status: ⚠️ MOSTLY VERIFIED

All API endpoints work. Frontend shows `quality_shortcut` option but backend doesn't support it yet.

**Action:** Either implement `quality_shortcut` in PipelineFactory or remove from frontend selector.

---

## Dependency Chain 4: SOTA Innovations → Pipelines

### Overview
New pipeline implementations (Seed-VC, MeanVC, Shortcut) integrate with existing infrastructure.

### Integration Points

| Pipeline | PipelineFactory | AdapterBridge | Frontend | Tests | Status |
|----------|----------------|---------------|----------|-------|--------|
| Seed-VC (quality_seedvc) | ✅ L138 | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Complete |
| MeanVC (realtime_meanvc) | ✅ L147 | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Complete |
| Shortcut (quality_shortcut) | ❌ No | ❓ N/A | ✅ Yes | ❌ No | ❌ Not Implemented |

### Shared Infrastructure Dependencies

All pipelines depend on:
- **AdapterManager** for LoRA loading → ✅ Works
- **VoiceProfileStore** for speaker embeddings → ✅ Works
- **PipelineFactory** for lazy loading → ✅ Works
- **api.py** routing via `pipeline_type` param → ✅ Works

### Tests Covering This Chain
- ✅ `test_seed_vc_pipeline.py` - Seed-VC standalone
- ✅ `test_meanvc_pipeline.py` - MeanVC standalone
- ✅ `test_adapter_bridge.py` - AdapterBridge integration
- ✅ `test_pipeline_benchmarks.py` - Multi-pipeline comparison

### Verification Status: ⚠️ MOSTLY VERIFIED

Seed-VC and MeanVC fully integrated. Shortcut pipeline missing.

**Action:** Implement `quality_shortcut` pipeline using R-VC 2-step shortcut flow matching.

---

## Cross-Context API Contract

### Voice Profile API

```typescript
interface VoiceProfile {
  id: string
  name: string
  has_trained_model: boolean  // ✅ Set by trainer.py completion
  speaker_embedding?: number[] // ✅ From diarization or training
  created_at: string
}
```

### Conversion API

```typescript
POST /api/v1/convert/song
{
  audio: File,
  profile_id: string,           // ✅ Links to trained model
  pipeline_type: PipelineType,  // ✅ Routes to correct pipeline
  settings?: { ... }
}
```

### WebSocket Events

```typescript
// Karaoke streaming
WebSocket /ws/karaoke
  client → { type: 'start', profile_id, pipeline_type }  // ✅ Uses trained model
  server → { type: 'audio_chunk', data: base64 }

// Training progress
WebSocket /ws/training
  server → { type: 'progress', job_id, epoch, loss }  // ✅ Updates LiveTrainingMonitor
  server → { type: 'complete', profile_id }           // ✅ Sets has_trained_model
```

---

## Broken Links Summary

| Link | Source | Consumer | Issue | Priority | Action |
|------|--------|----------|-------|----------|--------|
| quality_shortcut | Frontend PipelineSelector | PipelineFactory | Not implemented in backend | P2 | Implement or remove from UI |
| Cross-track clustering E2E | SpeakerMatcher | VoiceProfileStore | No E2E test | P3 | Add test |

---

## Recommendations

### Immediate (P0)
None - all critical paths work.

### Short-term (P1)
1. **Implement `quality_shortcut` pipeline** - Frontend expects it, should work
2. **Add E2E test for cross-track clustering** - Feature works but untested end-to-end

### Long-term (P2)
1. Document API contract in OpenAPI/Swagger spec
2. Add integration tests for all 5×3 = 15 combinations (5 pipelines × 3 use cases: convert, karaoke, training)
3. Create dependency diagram visual for onboarding

---

## Dependency Graph (Simplified)

```
YouTube Download
    ↓
[Metadata Fetch] → Featured Artist Detection
    ↓                       ↓
[Diarization] → WavLM Embeddings → [Clustering] → Speaker Profiles
    ↓                                                     ↓
Separated Vocals                              [VoiceProfileStore]
    ↓                                                     ↓
[Training] → LoRA Adapters ←─────────────────────────────┘
    ↓                  ↓
[AdapterManager]  [AdapterBridge]
    ↓                  ↓
[PipelineFactory] → {Realtime, Quality, Seed-VC, MeanVC, Shortcut?}
    ↓
[API Routes] ← pipeline_type param
    ↓
[WebSocket/REST] ← Frontend Components
    ↓
{VoiceProfilePage, KaraokePage, ConversionPage, TrainingConfigPanel}
```

---

## Track Metadata Updates

Updated the following track metadata.json files with cross_context fields:

### training-inference-integration_20260130
```json
{
  "cross_context": {
    "provides": ["trained_adapters", "has_trained_model_flag"],
    "consumes": ["voice_profiles", "speaker_embeddings"],
    "tested_with": ["test_adapter_integration_e2e.py", "test_continuous_training_e2e.py"]
  }
}
```

### sota-innovations_20260131
```json
{
  "cross_context": {
    "provides": ["quality_seedvc_pipeline", "realtime_meanvc_pipeline"],
    "consumes": ["adapter_manager", "voice_profile_store"],
    "broken_links": ["quality_shortcut not in PipelineFactory"],
    "tested_with": ["test_seed_vc_pipeline.py", "test_meanvc_pipeline.py"]
  }
}
```

### frontend-complete-integration_20260201
```json
{
  "cross_context": {
    "provides": ["pipeline_selector_ui", "training_monitor_ui"],
    "consumes": ["api_convert_endpoint", "websocket_karaoke"],
    "api_contract": ["GET /profiles/{id}/model", "POST /convert/song"],
    "tested_with": ["test_frontend_integration_e2e.py"]
  }
}
```

### speaker-diarization_20260130
```json
{
  "cross_context": {
    "provides": ["speaker_embeddings", "diarized_segments"],
    "consumes": ["youtube_metadata", "voice_profile_store"],
    "tested_with": ["test_speaker_diarization.py", "test_youtube_pipeline.py"]
  }
}
```

---

## Verification Checklist

- [x] Training → Inference integration verified
- [x] Diarization → Profile creation verified
- [x] Frontend → Backend API contract verified
- [x] SOTA pipelines integration verified
- [x] All track metadata updated with cross_context
- [ ] quality_shortcut pipeline implemented (BROKEN)
- [ ] Cross-track clustering E2E test added (MISSING)

---

**Next Steps:**
1. Create issue/task for `quality_shortcut` implementation
2. Create issue/task for cross-track clustering E2E test
3. Update track plans with cross-context findings
