# Specification: Frontend-Backend Parity & Granular Controls

**Track ID:** frontend-parity_20260129
**Type:** Feature
**Created:** 2026-01-29
**Status:** Draft

## Summary

Audit the backend for all implemented features, APIs, and configuration options not currently exposed in the frontend UI, then update the web interface to provide granular controls for every backend capability.

## Context

AutoVoice is a GPU-accelerated singing voice conversion system with a Flask+React stack. Multiple tracks have added significant backend capabilities (SOTA pipeline, live karaoke, voice profile training) that may not be fully represented in the frontend. Users need complete control over all system parameters for professional audio production work.

## User Story

As a developer, I want the frontend to expose granular controls for every backend capability so that users have complete control over all project features.

## Acceptance Criteria

### Core Controls
- [ ] Backend audit complete - All API endpoints documented with frontend exposure status
- [ ] Training controls exposed - LoRA rank, alpha, learning rate, epochs, EWC settings
- [ ] Inference controls exposed - Pitch shift, volume, presets, quality settings
- [ ] GPU/System metrics live - Real-time utilization, memory, temperature
- [ ] Audio device selection - Input/output device configuration
- [ ] Model management UI - View loaded models, load/unload, versioning
- [ ] Vocal separation controls - Demucs settings, stem selection
- [ ] Pitch extraction settings - CREPE/RMVPE method selection
- [ ] Training job management - Queue, progress, cancel, history/loss curves
- [ ] Conversion job queue - Pending jobs, cancel, retry, download
- [ ] Voice profile details - Samples, training history, quality scores, A/B compare
- [ ] Real-time/streaming controls - Latency vs quality tradeoffs
- [ ] Quality metrics display - Pitch RMSE, speaker similarity after conversion
- [ ] System configuration persistence - Save/load settings

### Advanced Controls
- [ ] Audio preprocessing settings - Normalization, noise reduction, silence trimming
- [ ] Encoder selection UI - HuBERT vs ContentVec toggle
- [ ] Vocoder selection UI - HiFiGAN vs BigVGAN selection
- [ ] TensorRT controls - Enable/disable, precision (FP16/INT8), rebuild
- [ ] Batch processing UI - Multi-file queue, batch settings
- [ ] Output format settings - WAV/MP3/FLAC, bitrate, sample rate
- [ ] Advanced pitch controls - Formant shift, vibrato, pitch correction
- [ ] Data augmentation settings - Pitch/time stretch, EQ for training
- [ ] Model checkpoint browser - View checkpoints, rollback, compare
- [ ] Spectrogram visualization - Before/after waveform display
- [ ] Preset management - Save/load/share custom presets
- [ ] Conversion history with playback - Listen, side-by-side compare
- [ ] Debug/logging panel - View logs, set levels, export diagnostics
- [ ] Webhook/notification settings - Job completion alerts

## Dependencies

### Backend Code
- src/auto_voice/web/api.py, audio_router.py, app.py
- src/auto_voice/inference/*.py (all pipeline files)
- src/auto_voice/training/trainer.py, job_manager.py
- src/auto_voice/gpu/memory_manager.py, latency_profiler.py
- src/auto_voice/audio/separator.py, augmentation.py, technique_detector.py
- src/auto_voice/models/*.py (encoder, vocoder, pitch, svc_decoder)
- src/auto_voice/evaluation/quality_metrics.py

### Frontend Code
- frontend/src/services/api.ts
- frontend/src/pages/*.tsx
- frontend/src/components/*.tsx
- frontend/vite.config.ts

### Configuration
- config/gpu_config.yaml, logging_config.yaml
- models/pretrained/

### Infrastructure
- PostgreSQL database schema
- WebSocket/SocketIO infrastructure
- Docker compose

### Completed Tracks
- sota-pipeline_20260124 (backend capabilities)
- live-karaoke_20260124 (streaming)

### In-Progress Track
- voice-profile-training_20260124 (partial training UI)

### Tools
- claude-flow v3.0.0-alpha.185 (updated)
- Playwright for UI verification

## Out of Scope

- Mobile app
- Cloud deployment/hosting
- Payment/subscription features
- Internationalization/localization
- New ML model architectures (expose existing only)

## Technical Notes

### Claude-Flow V3 Analysis Strategy

**Phase 1: Deep Backend Audit**
```bash
# Initialize claude-flow in project
claude-flow init

# AST analysis of backend code structure
claude-flow analyze ast src/auto_voice/ --format json > backend-analysis.json

# Extract all symbols (functions, classes, endpoints)
claude-flow analyze symbols src/auto_voice/ --type function --format json
claude-flow analyze symbols src/auto_voice/ --type class --format json

# Analyze code complexity to prioritize
claude-flow analyze complexity src/auto_voice/ --threshold 10

# Find module boundaries
claude-flow analyze modules src/auto_voice/

# Check for circular dependencies
claude-flow analyze circular src/auto_voice/
```

**Phase 2: Embeddings & Semantic Search**
```bash
# Initialize embedding system for semantic analysis
claude-flow embeddings init --model all-mpnet-base-v2

# Store analysis results in memory
claude-flow memory store "backend_endpoints" "<api analysis>" --namespace audit
claude-flow memory store "frontend_components" "<component list>" --namespace audit

# Semantic search for gaps
claude-flow embeddings search -q "training configuration"
claude-flow embeddings search -q "conversion parameters"
```

**Phase 3: Hive-Mind Development Swarm**
```bash
# Initialize hive-mind for coordinated development
claude-flow hive-mind init -t hierarchical-mesh

# Spawn specialized workers
claude-flow hive-mind spawn --claude -o "Audit backend API endpoints"
claude-flow hive-mind spawn --claude -o "Analyze frontend component coverage"
claude-flow hive-mind spawn --claude -o "Generate missing API transformations"

# Submit coordinated tasks
claude-flow hive-mind task -d "Create frontend components for missing backend features"

# Use consensus for architectural decisions
claude-flow hive-mind consensus --propose "Component structure for training controls"
```

**Phase 4: Neural Pattern Learning**
```bash
# Train patterns from existing code
claude-flow neural train -p coordination

# Analyze cognitive patterns in codebase
claude-flow neural patterns --action list

# Predict optimal component structure
claude-flow neural predict "frontend component for API endpoint"
```

**Phase 5: Development Swarm Execution**
```bash
# Full development swarm with review and testing
claude-flow swarm "Implement frontend components for backend parity" \
  --strategy development \
  --mode hierarchical \
  --max-agents 10 \
  --parallel \
  --monitor \
  --review \
  --testing \
  --verbose
```

### UI Framework Patterns
- React Query for server state management
- Component composition for reusable controls
- Lazy loading for performance
- Virtualization for large lists (react-window)

### Performance Requirements
- Initial load < 2s
- API response handling < 100ms
- Real-time updates via WebSocket (< 50ms latency)
- Debounced inputs (300ms)

### Accessibility (WCAG 2.1 AA)
- Keyboard navigation for all controls
- Screen reader support (ARIA labels)
- Focus indicators
- Color contrast ratios
- Reduced motion support

---

_Generated by Conductor. Review and edit as needed._
