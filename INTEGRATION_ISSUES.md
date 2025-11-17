# Integration Issues and Resolutions

## Critical Issues Fixed

### 1. WebSocket Handler Syntax Error
**Issue**: Incomplete `progress_callback` function definition (lines 397-398)
**Impact**: WebSocket progress updates would fail
**Fix**: Completed function body with proper state updates and event emission
**Status**: ✅ Fixed

### 2. Event Name Mismatch (conversion_failed vs conversion_error)
**Issue**: JobManager emitted `conversion_failed` but frontend expected `conversion_error`
**Impact**: Frontend error handlers would never trigger, causing silent failures
**Fix**: Changed JobManager to emit `conversion_error` with both `job_id` and `conversion_id` fields for backward compatibility
**File**: `/home/kp/repos/autovoice/src/auto_voice/web/job_manager.py` (line 270-278)
**Status**: ✅ Fixed (Comment 2, Verification Round 3)

### 3. JobManager Result Handling
**Issue**: Expected `result.audio_path` attribute but pipeline returns dict
**Impact**: All async conversions would fail
**Fix**: Updated to access dict keys and save numpy array to WAV file
**Status**: ✅ Fixed

### 4. Frontend/Backend Event Mismatch
**Issue**: Frontend expected `output_url`, backend emitted `result_path`
**Impact**: Frontend couldn't download results
**Fix**: Added `output_url` field to completion event
**Status**: ✅ Fixed

### 5. Parameter Name Inconsistency
**Issue**: Frontend sent camelCase, backend expected snake_case
**Impact**: Settings would be ignored or cause errors
**Fix**: Added parameter mapping in frontend API service
**Status**: ✅ Fixed

## Known Edge Cases

### 1. WebSocket Reconnection
**Issue**: If client disconnects during conversion, progress is lost
**Workaround**: Client can poll `/convert/status/{job_id}` for current status
**Future**: Implement reconnection with state recovery

### 2. Large File Uploads
**Issue**: Files >100MB rejected by default
**Workaround**: Increase `MAX_CONTENT_LENGTH` in config
**Future**: Implement chunked upload for large files

### 3. Concurrent Job Limit
**Issue**: JobManager limited to 4 concurrent workers by default
**Workaround**: Increase `max_workers` in job_manager config
**Future**: Implement dynamic worker scaling

### 4. Cache Invalidation
**Issue**: Cached results don't include stems even if requested
**Workaround**: Cache is bypassed when `return_stems=True`
**Future**: Implement separate cache for stem results

## Testing Coverage

### ✅ Covered
- REST API endpoint validation
- Job creation and status tracking
- WebSocket event emission
- Parameter validation
- Error handling
- Concurrent job processing

### ⚠️ Partial Coverage
- WebSocket reconnection scenarios
- Network failure recovery
- Large file handling
- Memory pressure scenarios

### ❌ Not Covered
- Load testing (>100 concurrent jobs)
- Long-running job timeout handling
- Disk space exhaustion
- GPU OOM recovery

## Deployment Considerations

### Production Checklist
- [ ] Set `CORS_ALLOWED_ORIGINS` to specific domains (not `*`)
- [ ] Configure `MAX_CONTENT_LENGTH` based on expected file sizes
- [ ] Set `job_manager.ttl_seconds` for result retention policy
- [ ] Configure `job_manager.max_workers` based on GPU capacity
- [ ] Enable Prometheus metrics for monitoring
- [ ] Set up log aggregation for error tracking
- [ ] Configure WebSocket ping/pong timeouts
- [ ] Set up health check monitoring

### Scaling Recommendations
- Use Redis for job state (instead of in-memory dict)
- Use S3/object storage for result files (instead of local disk)
- Use message queue (RabbitMQ/Kafka) for job distribution
- Deploy multiple worker instances behind load balancer
- Use sticky sessions for WebSocket connections

## Identifier Standards

### WebSocket Event Payloads
All WebSocket events use **`job_id` as the canonical identifier** with optional `conversion_id` alias for backward compatibility:

**Error Events**:
```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,              # PRIMARY - always use this
    'conversion_id': job_id,       # ALIAS - backward compatibility only
    'error': 'error message',
    'code': 'ERROR_CODE'
}, to=sid)
```

**Progress Events**:
```python
self.socketio.emit('conversion_progress', {
    'job_id': job_id,              # PRIMARY
    'conversion_id': job_id,       # ALIAS
    'progress': percent,
    'stage': stage_name,
    'timestamp': time.time()
}, to=sid)
```

**Completion Events**:
```python
self.socketio.emit('conversion_complete', {
    'job_id': job_id,              # PRIMARY
    'conversion_id': job_id,       # ALIAS
    'audio': audio_output,
    ...
}, to=sid)
```

**Frontend Usage**: Frontend code should ALWAYS reference `job_id` in WebSocket event handlers:
```typescript
this.socket.on('conversion_error', (data: { job_id: string; error: string }) => {
  const callback = this.errorCallbacks.get(data.job_id)  // Use job_id
  ...
})
```

## API Contract

### REST API
```
POST /api/v1/convert/song
  Request: multipart/form-data
    - audio: File (required)
    - target_profile_id: string (required)
    - settings: JSON string (optional)
  Response: 202 Accepted
    - status: "queued"
    - job_id: string
    - websocket_room: string
    - message: 'Join WebSocket room with job_id to receive progress updates'

GET /api/v1/convert/status/{job_id}
  Response: 200 OK
    - job_id: string
    - status: "queued" | "processing"