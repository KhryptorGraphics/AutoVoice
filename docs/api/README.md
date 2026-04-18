# AutoVoice API Documentation

Comprehensive API documentation for the AutoVoice singing voice conversion system.

## Current Profile Model

The current MVP uses two profile roles:

- `source_artist`: extracted from uploaded songs after vocal separation and diarization
- `target_user`: user-owned singing profiles used for LoRA or full-model training

Only `target_user` profiles are trainable. Full-model training unlocks after `30 minutes` of clean user vocals on that target profile.

## Quick Start

### Access Swagger UI

Visit the interactive API documentation:

```
http://localhost:5000/docs
```

### Download OpenAPI Spec

- JSON: http://localhost:5000/api/v1/openapi.json
- YAML: http://localhost:5000/api/v1/openapi.yaml

### Import Postman Collection

Import `postman_collection.json` into Postman or Insomnia for ready-to-use API requests.

---

## Documentation Files

### [tutorials.md](./tutorials.md)
Step-by-step tutorials for common workflows:
- Tutorial 1: Convert a Song
- Tutorial 2: Train a Voice Profile
- Tutorial 3: Live Karaoke Session
- Tutorial 4: YouTube Artist Training Pipeline
- Tutorial 5: Multi-Speaker Diarization
- Tutorial 6: Batch Processing

### [websocket-events.md](./websocket-events.md)
Complete WebSocket event documentation:
- Connection setup
- Namespace descriptions
- Event schemas
- Usage examples (JavaScript, Python, React)

### [postman_collection.json](./postman_collection.json)
Postman/Insomnia collection with all endpoints configured and ready to use.

---

## API Overview

### Base URL

```
http://localhost:5000/api/v1
```

### API Versioning

The API uses URL path versioning (`/api/v1`). Future breaking changes will increment the version number (`/api/v2`).

### Content Types

- Request: `multipart/form-data` (file uploads), `application/json` (data)
- Response: `application/json`, `audio/wav` (downloads)

### Response Format

All JSON responses follow a consistent structure:

**Success:**
```json
{
  "status": "success",
  "data": { ... }
}
```

**Error:**
```json
{
  "error": "error_type",
  "message": "Human-readable error description"
}
```

---

## Endpoint Groups

### Conversion Endpoints

Voice conversion and processing operations.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/convert/song` | POST | Convert song to target voice |
| `/convert/status/{job_id}` | GET | Get conversion job status |
| `/convert/download/{job_id}` | GET | Download conversion result |
| `/convert/cancel/{job_id}` | POST | Cancel conversion job |
| `/convert/metrics/{job_id}` | GET | Get quality metrics |
| `/convert/history` | GET | List conversion history |

### Voice Profile Endpoints

Voice profile management and creation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/voice/clone` | POST | Create voice profile from samples |
| `/voice/profiles` | GET | List all voice profiles |
| `/voice/profiles/{id}` | GET | Get profile details |
| `/voice/profiles/{id}` | DELETE | Delete voice profile |
| `/voice/profiles/{id}/adapters` | GET | List available adapters |
| `/voice/profiles/{id}/model` | GET | Get model status |
| `/voice/profiles/{id}/adapter/select` | POST | Select active adapter |
| `/voice/profiles/{id}/adapter/metrics` | GET | Get adapter metrics |
| `/voice/profiles/{id}/training-status` | GET | Get training status |

### Training Endpoints

Model training operations.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/training/jobs` | GET | List training jobs |
| `/training/jobs` | POST | Create training job |
| `/training/jobs/{id}` | GET | Get training job status |
| `/training/jobs/{id}/cancel` | POST | Cancel training job |

### Profile Sample Endpoints

Training sample management.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/profiles/{id}/samples` | GET | List profile samples |
| `/profiles/{id}/samples` | POST | Upload sample |
| `/profiles/{id}/samples/from-path` | POST | Add sample from path |
| `/profiles/{id}/samples/{sample_id}` | GET | Get sample details |
| `/profiles/{id}/samples/{sample_id}` | DELETE | Delete sample |
| `/profiles/{id}/samples/{sample_id}/filter` | POST | Apply audio filter to sample |
| `/profiles/{id}/segments` | GET | Get speaker segments |
| `/profiles/{id}/songs` | POST | Add song for profile |

### Audio Processing Endpoints

Audio analysis and processing utilities.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/audio/diarize` | POST | Speaker diarization |
| `/audio/diarize/assign` | POST | Assign speakers to profiles |
| `/audio/router/config` | GET | Get audio router config |
| `/audio/router/config` | POST | Update audio router config |

### YouTube Endpoints

YouTube download and processing.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/youtube/info` | POST | Get video metadata |
| `/youtube/download` | POST | Download and process audio |

### System Endpoints

System health and monitoring.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/gpu/metrics` | GET | GPU metrics |
| `/kernels/metrics` | GET | CUDA kernel metrics |
| `/system/info` | GET | System information |
| `/devices/list` | GET | List audio devices |
| `/devices/config` | GET | Get device config |
| `/devices/config` | POST | Update device config |

### Configuration Endpoints

System configuration management.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config/separation` | GET | Get separation config |
| `/config/separation` | POST | Update separation config |
| `/config/pitch` | GET | Get pitch config |
| `/config/pitch` | POST | Update pitch config |
| `/presets` | GET | List conversion presets |
| `/presets` | POST | Create preset |
| `/presets/{id}` | GET | Get preset details |
| `/presets/{id}` | PUT | Update preset |
| `/presets/{id}` | DELETE | Delete preset |

### Model Management Endpoints

Model loading and TensorRT optimization.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models/loaded` | GET | List loaded models |
| `/models/load` | POST | Load model |
| `/models/unload` | POST | Unload model |
| `/models/tensorrt/status` | GET | TensorRT engine status |
| `/models/tensorrt/rebuild` | POST | Rebuild TensorRT engine |
| `/models/tensorrt/build` | POST | Build TensorRT engine |

---

## Pipeline Types

### Realtime Pipeline

**Use case:** Live karaoke, real-time voice conversion

**Characteristics:**
- Latency: <100ms
- Sample rate: 16kHz
- Quality: Good
- GPU memory: Low
- Best for: Interactive applications

**Example:**
```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@vocals.wav" \
  -F "profile_id=profile_123" \
  -F "pipeline_type=realtime" \
  -F "adapter_type=nvfp4"
```

### Quality Pipeline

**Use case:** High-quality song covers

**Characteristics:**
- Latency: 5-10 seconds
- Sample rate: 24kHz
- Quality: High
- GPU memory: Medium
- Architecture: CoMoSVC with 30-step diffusion

**Example:**
```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "profile_id=profile_123" \
  -F "pipeline_type=quality" \
  -F "output_quality=high"
```

### Quality SeedVC Pipeline (SOTA)

**Use case:** Studio-quality conversions

**Characteristics:**
- Latency: 10-20 seconds
- Sample rate: 44kHz
- Quality: State-of-the-art
- GPU memory: High
- Architecture: DiT-CFM with 5-10 step flow matching

**Example:**
```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "profile_id=profile_123" \
  -F "pipeline_type=quality_seedvc" \
  -F "output_quality=studio"
```

---

## Adapter Types

### Unified Adapter (Recommended)

**Description:** New unified adapter format combining best of both worlds.

**Characteristics:**
- Format: Single .pth file
- Quality: High
- Speed: Fast
- Memory: Optimized
- Compatibility: All pipelines

**Training:**
```bash
curl -X POST http://localhost:5000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{"profile_id": "profile_123", "adapter_type": "unified"}'
```

### HQ Adapter

**Description:** High-quality LoRA adapter with fp16 precision.

**Characteristics:**
- Format: .pth checkpoint
- Quality: Very high
- Speed: Moderate
- Memory: Medium
- Best for: Quality pipeline

### NVFP4 Adapter

**Description:** Fast 4-bit quantized adapter for real-time inference.

**Characteristics:**
- Format: Quantized .pth
- Quality: Good
- Speed: Very fast
- Memory: Low
- Best for: Realtime pipeline

---

## Quality Presets

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| **draft** | Fastest | Low | Quick tests, previews |
| **fast** | Fast | Good | Rapid iteration |
| **balanced** | Medium | High | Default, production |
| **high** | Slow | Very High | Professional covers |
| **studio** | Slowest | Best | Studio mastering |

---

## Authentication

Currently **no authentication** is required. Production deployments should implement:

- API key authentication
- JWT token-based auth
- Rate limiting per API key
- CORS configuration

### Future Authentication (Planned)

```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "song=@song.mp3" \
  -F "profile_id=profile_123"
```

---

## Rate Limiting

**Current:** No rate limiting

**Production Recommendations:**
- 100 requests/minute per IP
- 10 concurrent conversions per user
- 1 training job per profile at a time
- WebSocket: 100 events/minute per connection

---

## Error Codes

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 202 | Accepted | Job queued for async processing |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Resource not found |
| 503 | Service Unavailable | Service temporarily unavailable |
| 500 | Internal Server Error | Unexpected server error |

### Error Response Format

```json
{
  "error": "error_type",
  "message": "Detailed error description",
  "details": {
    "field": "Additional context"
  }
}
```

### Common Error Types

- `validation_error`: Invalid input parameters
- `profile_not_found`: Voice profile does not exist
- `model_not_trained`: Profile has no trained model
- `processing_error`: Error during conversion
- `insufficient_quality`: Audio quality too low
- `resource_error`: Insufficient GPU memory or disk space

---

## Best Practices

### 1. Poll Status, Don't Block

Use async mode and poll for status instead of waiting synchronously:

```python
# Good
job_id = start_conversion(song_path, profile_id)
while True:
    status = get_status(job_id)
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(2)

# Bad (blocks for entire conversion)
result = convert_song_sync(song_path, profile_id)
```

### 2. Use WebSockets for Progress

WebSocket events provide real-time updates without polling:

```javascript
socket.on('job_progress', (data) => {
  updateProgressBar(data.progress);
});
```

### 3. Select Appropriate Pipeline

Choose pipeline based on use case:
- **Live karaoke:** `realtime`
- **Quick preview:** `quality`
- **Final production:** `quality_seedvc`

### 4. Handle Errors Gracefully

Always check for error responses:

```python
response = requests.post(url, files=files, data=data)
if response.status_code == 404:
    error_msg = response.json()['message']
    if 'trained model' in error_msg:
        # Start training first
        train_profile(profile_id)
```

### 5. Clean Up Old Jobs

Delete completed jobs to free disk space:

```bash
curl -X DELETE http://localhost:5000/api/v1/convert/history/{job_id}
```

---

## Performance Tips

### Optimize for Throughput

**Batch Processing:**
- Submit multiple jobs in parallel
- Use job queue for rate limiting
- Monitor GPU utilization

**Memory Management:**
- Use `nvfp4` adapter for low memory
- Reduce batch size if OOM errors
- Unload unused models

### Optimize for Latency

**Real-time Conversion:**
- Use `realtime` pipeline
- Select `nvfp4` adapter
- Pre-load models with `/models/load`
- Keep models in GPU memory

---

## Support

### Getting Help

- GitHub Issues: https://github.com/yourusername/autovoice/issues
- Documentation: http://localhost:5000/docs
- Tutorials: [tutorials.md](./tutorials.md)

### Reporting Bugs

Include:
1. API endpoint and request details
2. Error response (if any)
3. System info from `/api/v1/system/info`
4. GPU metrics from `/api/v1/gpu/metrics`

---

## Changelog

### Version 1.0.0 (2026-02-01)

**Initial release:**
- 60+ REST endpoints
- WebSocket support for real-time updates
- 3 pipeline types (realtime, quality, quality_seedvc)
- 3 adapter types (hq, nvfp4, unified)
- Speaker diarization
- YouTube integration
- Comprehensive documentation

---

## License

MIT License - See LICENSE file for details
