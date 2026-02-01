# SOTA Dual-Pipeline - Deep Inspection Audit
**Date:** 2026-01-31
**Auditor:** Claude Code

## Executive Summary

The implementation is **~75% complete**. Core pipeline code is fully implemented, but integration with the web API and frontend is incomplete.

---

## Phase 1: Realtime Pipeline - ✅ CODE COMPLETE

**File:** `scripts/realtime_pipeline.py` (554 lines)

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| RealtimeVoiceConverter class | ✅ Complete | 56-486 | Full implementation |
| ContentVec encoder (HuBERT fallback) | ✅ Complete | 93-123 | FP16 support |
| RMVPE pitch extraction | ✅ Complete | 125-149 | Seed-VC fallback |
| HiFiGAN vocoder (CosyVoice) | ✅ Complete | 151-187 | Lazy loading |
| SimpleDecoder (content+pitch+speaker→mel) | ✅ Complete | 189-253 | Transformer-based |
| Streaming chunk processing | ✅ Complete | 308-446 | Crossfade overlap |
| convert_full() method | ✅ Complete | 448-475 | Non-streaming conversion |

**Missing:**
- [ ] Task 1.7: E2E test with William→Conor using HQ LoRA

---

## Phase 2: Quality Pipeline - ✅ CODE COMPLETE

**File:** `scripts/quality_pipeline.py` (598 lines)

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| QualityVoiceConverter class | ✅ Complete | 58-494 | Full implementation |
| Seed-VC model loading (DiT F0 44kHz) | ✅ Complete | 93-140 | Auto-downloads from HF |
| Whisper encoder | ✅ Complete | 141-171 | Semantic extraction |
| CAMPPlus speaker style | ✅ Complete | 183-191 | 192-dim embedding |
| CFM inference | ✅ Complete | 418-432 | FP16 autocast |
| BigVGAN vocoder | ✅ Complete | 194-200 | NVIDIA official |
| F0 conditioning with RMVPE | ✅ Complete | 365-396 | Auto pitch adjust |
| Chunked long-audio handling | ✅ Complete | 307-340 | 30s windows |

**Missing:**
- [ ] Task 2.8: E2E test with William→Conor using HQ LoRA

---

## Phase 3: HQ-SVC Enhancement - ✅ CODE COMPLETE

**File:** `src/auto_voice/inference/hq_svc_wrapper.py` (539 lines)

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| HQSVCWrapper class | ✅ Complete | 36-539 | Full implementation |
| super_resolve() method | ✅ Complete | 297-394 | 16kHz→44.1kHz |
| convert() method | ✅ Complete | 396-539 | Voice conversion |
| FACodec encoder/decoder | ✅ Complete | 167-168 | Content extraction |
| RMVPE F0 extractor | ✅ Complete | 170-171 | Pitch tracking |
| NSF-HiFiGAN vocoder | ✅ Complete | 139-147 | 44.1kHz synthesis |
| DDSP + Diffusion inference | ✅ Complete | 348-366 | dpm-solver method |

**Missing:**
- [ ] Task 3.3: Combined pipeline test (Seed-VC → HQ-SVC)
- [ ] Task 3.4: Quality vs latency benchmark

---

## Phase 4: SmoothSinger Concepts - ❌ NOT STARTED

These are optional enhancements not yet implemented.

---

## Phase 5: Web UI Integration - ⚠️ PARTIAL

### Frontend Components - ✅ COMPLETE

| File | Status | Lines | Notes |
|------|--------|-------|-------|
| PipelineSelector.tsx | ✅ Complete | 174 | Button + dropdown + badge |
| AdapterSelector.tsx | ✅ Complete | 295 | Full CRUD + metrics |

### Page Integration - ⚠️ PARTIAL

| Page | PipelineSelector | AdapterSelector | Actually Used |
|------|-----------------|-----------------|---------------|
| KaraokePage.tsx | ✅ Imported | ✅ Imported | ❌ NOT PASSED to API |
| VoiceProfilePage.tsx | ❌ Not imported | ✅ Integrated | ❌ N/A |
| ConvertPage | N/A (doesn't exist) | N/A | N/A |

**Critical Issue:** KaraokePage.tsx has PipelineSelector but:
- Line 262: `await client.startSession(uploadedSong.song_id, selectedModel)`
- The `pipeline` state variable is NOT passed to the API call!

### Backend API Integration - ⚠️ PARTIAL

| Endpoint | adapter_type | pipeline_type |
|----------|-------------|---------------|
| POST /convert/song | ✅ Supported (line 275) | ❌ NOT SUPPORTED |
| WebSocket startSession | ❌ Not supported | ❌ Not supported |

**File:** `src/auto_voice/web/api.py`
- `adapter_type` parameter exists (line 161, 275-292)
- `pipeline_type` parameter does NOT exist
- Backend always uses `singing_conversion_pipeline` - no routing to realtime/quality

---

## Phase 6: Testing - ❌ NOT STARTED

No dedicated tests written yet for dual-pipeline.

---

## Adapter Manager - ✅ COMPLETE

**File:** `src/auto_voice/models/adapter_manager.py` (396 lines)

| Feature | Status | Notes |
|---------|--------|-------|
| AdapterCache (LRU) | ✅ | 5-item default |
| load_adapter() | ✅ | From disk with validation |
| apply_adapter() | ✅ | LoRA parameter injection |
| save_adapter() | ✅ | Extract and save |
| list_available_adapters() | ✅ | Glob pattern match |
| Global instance | ✅ | get_adapter_manager() |

---

## Trained LoRAs - ✅ VERIFIED

```
data/trained_models/hq/
├── 7da05140-1303-40c6-95d9-5b6e2c3624df_hq_lora.pt (5.9MB) - William Singe
└── c572d02c-c687-4bed-8676-6ad253cf1c91_hq_lora.pt (5.9MB) - Connor

data/trained_models/nvfp4/
├── 7da05140-1303-40c6-95d9-5b6e2c3624df_nvfp4_lora.pt (100KB) - William (needs retrain)
└── c572d02c-c687-4bed-8676-6ad253cf1c91_nvfp4_lora.pt (100KB) - Connor (needs retrain)
```

---

## Remaining Work Summary

### HIGH PRIORITY (Core Integration)
1. **Backend:** Add `pipeline_type` parameter to `/convert/song` endpoint
2. **Backend:** Add `pipeline_type` parameter to WebSocket `startSession`
3. **Backend:** Implement pipeline routing logic (instantiate RealtimeVoiceConverter or QualityVoiceConverter)
4. **Frontend:** Pass `pipeline` state to `client.startSession()` in KaraokePage.tsx
5. **Frontend:** Pass `pipeline` state to conversion API calls

### MEDIUM PRIORITY (Testing)
6. E2E test: William→Conor with realtime pipeline + HQ LoRA
7. E2E test: William→Conor with quality pipeline + HQ LoRA
8. Benchmark: Memory usage comparison
9. Benchmark: Quality metrics (MCD, speaker similarity)

### LOW PRIORITY (Polish)
10. Phase 4: SmoothSinger concepts (optional)
11. Documentation: Help page updates
12. Combined pipeline: Seed-VC → HQ-SVC test
