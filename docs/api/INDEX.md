# AutoVoice API Documentation - Complete Index

**Last Updated:** 2026-02-01
**Version:** 1.0.0

---

## 🚀 Quick Start

| Resource | URL | Description |
|----------|-----|-------------|
| **Swagger UI** | http://localhost:5000/docs | Interactive API documentation |
| **OpenAPI JSON** | http://localhost:5000/api/v1/openapi.json | Machine-readable spec |
| **OpenAPI YAML** | http://localhost:5000/api/v1/openapi.yaml | Human-readable spec |

---

## 📖 Documentation Files

### Getting Started

| File | Description | Audience |
|------|-------------|----------|
| [README.md](./README.md) | Overview, quick reference, best practices | All users |
| [SETUP.md](./SETUP.md) | Installation, configuration, troubleshooting | Developers |
| [SUMMARY.md](./SUMMARY.md) | Implementation summary, metrics | Project managers |

### Guides and Tutorials

| File | Topics Covered | Level |
|------|----------------|-------|
| [tutorials.md](./tutorials.md) | 6 step-by-step tutorials with code examples | Beginner to Advanced |

**Tutorials:**
1. Convert a Song (Beginner)
2. Train a Voice Profile (Intermediate)
3. Live Karaoke Session (Intermediate)
4. YouTube Artist Training Pipeline (Advanced)
5. Multi-Speaker Diarization (Advanced)
6. Batch Processing (Intermediate)

### Technical References

| File | Content | Use Case |
|------|---------|----------|
| [websocket-events.md](./websocket-events.md) | WebSocket namespaces, events, schemas | Real-time features |
| [postman_collection.json](./postman_collection.json) | Postman/Insomnia collection | API testing |

---

## 🔌 API Endpoints by Category

### Conversion (7 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/convert/song` | POST | Convert song to target voice |
| `/api/v1/convert/status/{job_id}` | GET | Get conversion job status |
| `/api/v1/convert/download/{job_id}` | GET | Download conversion result |
| `/api/v1/convert/cancel/{job_id}` | POST | Cancel conversion job |
| `/api/v1/convert/metrics/{job_id}` | GET | Get quality metrics |
| `/api/v1/convert/history` | GET | List conversion history |
| `/api/v1/convert/history/{id}` | DELETE | Delete history record |

### Voice Profiles (10 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/voice/clone` | POST | Create voice profile from samples |
| `/api/v1/voice/profiles` | GET | List all voice profiles |
| `/api/v1/voice/profiles/{id}` | GET | Get profile details |
| `/api/v1/voice/profiles/{id}` | DELETE | Delete voice profile |
| `/api/v1/voice/profiles/{id}/adapters` | GET | List available adapters |
| `/api/v1/voice/profiles/{id}/model` | GET | Get model status |
| `/api/v1/voice/profiles/{id}/adapter/select` | POST | Select active adapter |
| `/api/v1/voice/profiles/{id}/adapter/metrics` | GET | Get adapter metrics |
| `/api/v1/voice/profiles/{id}/training-status` | GET | Get training status |
| `/api/v1/profiles/auto-create` | POST | Auto-create from diarization |

### Training (4 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/training/jobs` | GET | List training jobs |
| `/api/v1/training/jobs` | POST | Create training job |
| `/api/v1/training/jobs/{id}` | GET | Get job status |
| `/api/v1/training/jobs/{id}/cancel` | POST | Cancel training job |

### Profile Samples (8 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/profiles/{id}/samples` | GET | List profile samples |
| `/api/v1/profiles/{id}/samples` | POST | Upload sample |
| `/api/v1/profiles/{id}/samples/from-path` | POST | Add sample from path |
| `/api/v1/profiles/{id}/samples/{sample_id}` | GET | Get sample details |
| `/api/v1/profiles/{id}/samples/{sample_id}` | DELETE | Delete sample |
| `/api/v1/profiles/{id}/samples/{sample_id}/filter` | POST | Apply audio filter |
| `/api/v1/profiles/{id}/segments` | GET | Get speaker segments |
| `/api/v1/profiles/{id}/songs` | POST | Add song for profile |

### Audio Processing (3 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/audio/diarize` | POST | Speaker diarization |
| `/api/v1/audio/diarize/assign` | POST | Assign speakers to profiles |
| `/api/v1/audio/router/config` | GET/POST | Audio router configuration |

### YouTube (2 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/youtube/info` | POST | Get video metadata |
| `/api/v1/youtube/download` | POST | Download and process audio |

### System (10 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | System health check |
| `/api/v1/gpu/metrics` | GET | GPU metrics |
| `/api/v1/kernels/metrics` | GET | CUDA kernel metrics |
| `/api/v1/system/info` | GET | System information |
| `/api/v1/devices/list` | GET | List audio devices |
| `/api/v1/devices/config` | GET/POST | Device configuration |
| `/api/v1/config/separation` | GET/POST | Separation configuration |
| `/api/v1/config/pitch` | GET/POST | Pitch configuration |
| `/api/v1/presets` | GET/POST/PUT/DELETE | Conversion presets |
| `/api/v1/models/*` | GET/POST | Model management |

---

## 🔄 WebSocket Events

### Default Namespace (`/`)

**Server Events:**
- `job_created` - New conversion job created
- `job_progress` - Conversion progress update
- `job_complete` - Conversion completed
- `job_failed` - Conversion failed

### Karaoke Namespace (`/karaoke`)

**Client Events:**
- `join_session` - Join karaoke session
- `leave_session` - Leave karaoke session
- `start_separation` - Request track separation

**Server Events:**
- `separation_progress` - Separation progress update
- `separation_complete` - Separation completed
- `separation_failed` - Separation failed
- `track_added` - Track added to queue
- `playback_state` - Playback state changed

### Training Namespace (`/training`)

**Client Events:**
- `join_training` - Subscribe to training updates
- `leave_training` - Unsubscribe from training updates

**Server Events:**
- `training_started` - Training begun
- `training_progress` - Training progress with metrics
- `training_complete` - Training completed
- `training_failed` - Training failed
- `checkpoint_saved` - Training checkpoint saved

---

## 📊 Data Schemas

### Core Schemas

| Schema | Fields | Used In |
|--------|--------|---------|
| `Error` | error, message | All error responses |
| `JobStatus` | job_id, status, progress, message | Job status endpoints |
| `ConversionSettings` | profile_id, volumes, pitch, quality | Conversion requests |
| `ConversionResult` | audio, metadata, f0_contour | Conversion responses |
| `VoiceProfile` | profile_id, name, samples, metadata | Profile endpoints |
| `TrainingJob` | job_id, status, epoch, loss | Training endpoints |

### Specialized Schemas

| Schema | Purpose | Endpoints |
|--------|---------|-----------|
| `DiarizerSegment` | Speaker diarization segment | `/audio/diarize` |
| `DiarizationResult` | Complete diarization result | `/audio/diarize` |
| `YouTubeInfo` | Video metadata | `/youtube/info` |
| `YouTubeDownloadResult` | Download result | `/youtube/download` |
| `GPUMetrics` | GPU utilization metrics | `/gpu/metrics` |
| `HealthCheck` | System health status | `/health` |

---

## 🎯 Common Workflows

### Workflow 1: Basic Song Conversion

```
1. Upload song → POST /api/v1/convert/song
2. Get job_id → Response: {"job_id": "..."}
3. Poll status → GET /api/v1/convert/status/{job_id}
4. Download → GET /api/v1/convert/download/{job_id}
```

**Tutorial:** [Convert a Song](./tutorials.md#tutorial-1-convert-a-song)

### Workflow 2: Voice Profile Training

```
1. Create profile → POST /api/v1/voice/clone
2. Get profile_id → Response: {"profile_id": "..."}
3. Start training → POST /api/v1/training/jobs
4. Monitor progress → WebSocket /training namespace
5. Verify complete → GET /api/v1/voice/profiles/{id}/training-status
```

**Tutorial:** [Train a Voice Profile](./tutorials.md#tutorial-2-train-a-voice-profile)

### Workflow 3: YouTube to Profile

```
1. Get video info → POST /api/v1/youtube/info
2. Download audio → POST /api/v1/youtube/download
3. Diarize speakers → POST /api/v1/audio/diarize
4. Assign speakers → POST /api/v1/audio/diarize/assign
5. Extract samples → GET /api/v1/profiles/{id}/segments
6. Train model → POST /api/v1/training/jobs
```

**Tutorial:** [YouTube Artist Training](./tutorials.md#tutorial-4-youtube-artist-training-pipeline)

### Workflow 4: Live Karaoke

```
1. Separate song → POST /api/v1/karaoke/separate
2. Monitor progress → WebSocket /karaoke namespace
3. Convert vocals → POST /api/v1/convert/song (pipeline_type=realtime)
4. Download stems → GET /api/v1/separation/{job_id}/status
5. Mix locally → Client-side audio mixing
```

**Tutorial:** [Live Karaoke Session](./tutorials.md#tutorial-3-live-karaoke-session)

---

## 🛠 Development Tools

### Code Files

| File | Purpose | Language |
|------|---------|----------|
| `src/auto_voice/web/openapi_spec.py` | OpenAPI spec generator | Python |
| `src/auto_voice/web/api_docs.py` | Endpoint documentation | Python |
| `scripts/validate_openapi.py` | Validation script | Python |

### Dependencies

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt
```

**Packages:**
- `apispec>=6.0` - OpenAPI spec generation
- `apispec-webframeworks>=1.0` - Flask integration
- `flask-swagger-ui>=4.11` - Swagger UI
- `marshmallow>=3.20` - Schema validation

### Validation

```bash
# Validate OpenAPI spec
python scripts/validate_openapi.py
```

**Checks:**
- Server availability
- OpenAPI spec validity
- Swagger UI accessibility
- Endpoint coverage

---

## 🎨 Pipeline & Adapter Reference

### Pipeline Types

| Pipeline | Latency | Sample Rate | Quality | GPU Memory | Use Case |
|----------|---------|-------------|---------|------------|----------|
| `realtime` | <100ms | 16kHz | Good | Low | Live karaoke |
| `quality` | 5-10s | 24kHz | High | Medium | Song covers |
| `quality_seedvc` | 10-20s | 44kHz | SOTA | High | Studio production |

### Adapter Types

| Adapter | Precision | Quality | Speed | Memory | Compatibility |
|---------|-----------|---------|-------|--------|---------------|
| `unified` | Mixed | High | Fast | Optimized | All pipelines |
| `hq` | FP16 | Very High | Moderate | Medium | Quality pipeline |
| `nvfp4` | 4-bit | Good | Very Fast | Low | Realtime pipeline |

### Quality Presets

| Preset | Speed | Quality | Steps | Best For |
|--------|-------|---------|-------|----------|
| `draft` | Fastest | Low | 5 | Quick tests |
| `fast` | Fast | Good | 10 | Rapid iteration |
| `balanced` | Medium | High | 20 | Production (default) |
| `high` | Slow | Very High | 30 | Professional covers |
| `studio` | Slowest | Best | 50 | Studio mastering |

---

## 🔐 Security & Limits

### Authentication

**Current:** None required
**Production:** JWT/API key recommended

### Rate Limiting

**Current:** None enforced
**Production Recommendations:**
- 100 requests/minute per IP
- 10 concurrent conversions
- 1 training job per profile

### CORS

**Current:** `*` (all origins)
**Production:** Configure allowed origins

---

## 📚 Code Examples

### JavaScript (Browser)

```javascript
// Basic conversion
const response = await fetch('http://localhost:5000/api/v1/convert/song', {
  method: 'POST',
  body: formData
});
const data = await response.json();
console.log('Job ID:', data.job_id);
```

**Full Example:** [tutorials.md - Convert a Song](./tutorials.md#tutorial-1-convert-a-song)

### Python

```python
# Basic conversion
response = requests.post(
    'http://localhost:5000/api/v1/convert/song',
    files={'song': open('song.mp3', 'rb')},
    data={'profile_id': 'profile_123'}
)
job_id = response.json()['job_id']
```

**Full Example:** [tutorials.md - Batch Processing](./tutorials.md#tutorial-6-batch-processing)

### curl

```bash
# Basic conversion
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "profile_id=profile_123"
```

**Full Examples:** [tutorials.md](./tutorials.md)

---

## 📞 Support

### Getting Help

1. **Documentation:** Start with [README.md](./README.md)
2. **Tutorials:** Follow step-by-step guides in [tutorials.md](./tutorials.md)
3. **Setup Issues:** Check [SETUP.md](./SETUP.md)
4. **API Reference:** Use Swagger UI at http://localhost:5000/docs
5. **GitHub Issues:** Report bugs and request features

### Common Issues

| Issue | Solution | Reference |
|-------|----------|-----------|
| Server not starting | Check dependencies, GPU availability | [SETUP.md](./SETUP.md) |
| Profile not found | Verify profile exists, check training status | [tutorials.md](./tutorials.md#tutorial-2) |
| Conversion fails | Check audio format, GPU memory | [README.md](./README.md#troubleshooting) |
| WebSocket not connecting | Verify Socket.IO setup, check CORS | [websocket-events.md](./websocket-events.md) |

---

## 📈 Metrics & Coverage

### Documentation Coverage

- **Total Endpoints:** 60+
- **Documented:** 60+ (100%)
- **Schemas Defined:** 14
- **WebSocket Events:** 15
- **Tutorials:** 6
- **Code Examples:** 30+

### File Statistics

- **Code:** ~995 lines (Python)
- **Documentation:** ~64KB (Markdown)
- **Schemas:** 14 marshmallow classes
- **Endpoints:** 60+ with full documentation

---

## 🗺 Navigation Map

```
Starting Point → Choose Your Path
├── New User
│   ├── Read: README.md (Overview)
│   ├── Follow: SETUP.md (Installation)
│   └── Try: tutorials.md → Tutorial 1 (Convert a Song)
│
├── API Integration
│   ├── Import: postman_collection.json
│   ├── Browse: http://localhost:5000/docs (Swagger UI)
│   └── Reference: README.md (Best Practices)
│
├── Real-time Features
│   ├── Read: websocket-events.md (WebSocket Events)
│   └── Follow: tutorials.md → Tutorial 3 (Live Karaoke)
│
├── Advanced Workflows
│   ├── Follow: tutorials.md → Tutorial 4 (YouTube Pipeline)
│   └── Follow: tutorials.md → Tutorial 5 (Diarization)
│
└── Troubleshooting
    ├── Check: SETUP.md (Common Issues)
    ├── Run: validate_openapi.py (Validation)
    └── Review: README.md (Error Codes)
```

---

## 🎓 Learning Path

### Beginner (1-2 hours)

1. Read [README.md](./README.md) - Overview
2. Follow [SETUP.md](./SETUP.md) - Installation
3. Complete [Tutorial 1](./tutorials.md#tutorial-1) - Convert a Song
4. Explore [Swagger UI](http://localhost:5000/docs) - Interactive docs

### Intermediate (3-4 hours)

1. Complete [Tutorial 2](./tutorials.md#tutorial-2) - Train a Voice Profile
2. Complete [Tutorial 3](./tutorials.md#tutorial-3) - Live Karaoke Session
3. Read [websocket-events.md](./websocket-events.md) - Real-time features
4. Import [postman_collection.json](./postman_collection.json) - Test endpoints

### Advanced (5-8 hours)

1. Complete [Tutorial 4](./tutorials.md#tutorial-4) - YouTube Pipeline
2. Complete [Tutorial 5](./tutorials.md#tutorial-5) - Multi-Speaker Diarization
3. Complete [Tutorial 6](./tutorials.md#tutorial-6) - Batch Processing
4. Review [openapi_spec.py](../../src/auto_voice/web/openapi_spec.py) - Schema definitions
5. Build custom integration

---

## 📝 Quick Reference Cards

### Conversion Quick Reference

```bash
# Standard conversion
POST /api/v1/convert/song
  -F song=@file.mp3
  -F profile_id=ID
  -F output_quality=balanced
  -F pipeline_type=quality_seedvc

# Check status
GET /api/v1/convert/status/{job_id}

# Download result
GET /api/v1/convert/download/{job_id}
```

### Training Quick Reference

```bash
# Create profile
POST /api/v1/voice/clone
  -F name="Artist"
  -F samples=@sample1.wav
  -F samples=@sample2.wav

# Start training
POST /api/v1/training/jobs
  -d '{"profile_id": "ID", "epochs": 100}'

# Check status
GET /api/v1/training/jobs/{job_id}
```

### WebSocket Quick Reference

```javascript
// Connect
const socket = io('http://localhost:5000');

// Join room
socket.emit('join', job_id);

// Listen for progress
socket.on('job_progress', (data) => {
  console.log(data.progress);
});
```

---

## 🔗 External Resources

- **OpenAPI Spec:** https://swagger.io/specification/
- **Swagger UI:** https://swagger.io/tools/swagger-ui/
- **APISpec:** https://apispec.readthedocs.io/
- **Socket.IO:** https://socket.io/docs/
- **Postman:** https://www.postman.com/
- **Insomnia:** https://insomnia.rest/

---

**Last Updated:** 2026-02-01
**Maintained By:** Agent G - API Documentation Specialist
**Track ID:** api-documentation-suite_20260201
