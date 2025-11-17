# WebSocket Error Payloads - Complete Audit

## Executive Summary

**Audit Date**: 2025-11-17
**Audit Scope**: All WebSocket error emissions in `websocket_handler.py`
**Result**: ✅ **ALL PAYLOADS STANDARDIZED**
**Standard**: `job_id` (primary) + `conversion_id` (backward compatibility alias)

## Error Payload Inventory

### File: `/home/kp/repos/autovoice/src/auto_voice/web/websocket_handler.py`

| Line | Event Type | Error Code | job_id | conversion_id | Status |
|------|------------|------------|--------|---------------|--------|
| 327-332 | `conversion_error` | `MISSING_SONG` | ✅ Primary | ✅ Alias | ✅ |
| 336-341 | `conversion_error` | `MISSING_PROFILE` | ✅ Primary | ✅ Alias | ✅ |
| 346-352 | `conversion_error` | `INVALID_PARAMS` | ✅ Primary | ✅ Alias | ✅ |
| 356-362 | `conversion_error` | `INVALID_PARAMS` | ✅ Primary | ✅ Alias | ✅ |
| 368-373 | `conversion_error` | `SERVICE_UNAVAILABLE` | ✅ Primary | ✅ Alias | ✅ |
| 572-578 | `conversion_error` | `CONVERSION_FAILED` | ✅ Primary | ✅ Alias | ✅ |

**Total Error Emissions**: 6
**Standardized**: 6 (100%)
**Non-Compliant**: 0 (0%)

## Detailed Error Payload Analysis

### 1. Missing Song Data Validation (Lines 327-332)

**Location**: `handle_convert_song_stream` event handler
**Trigger**: Client sends conversion request without `song_data`

```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,              # ✅ PRIMARY
    'conversion_id': job_id,       # ✅ ALIAS
    'error': 'Missing song data',
    'code': 'MISSING_SONG'
}, to=sid)
```

**Fields**:
- ✅ `job_id`: Primary identifier (required)
- ✅ `conversion_id`: Backward compatibility alias
- ✅ `error`: Human-readable error message
- ✅ `code`: Machine-readable error code
- ✅ `to=sid`: Sent to specific client session

---

### 2. Missing Target Profile Validation (Lines 336-341)

**Location**: `handle_convert_song_stream` event handler
**Trigger**: Client sends conversion request without `target_profile_id`

```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,                    # ✅ PRIMARY
    'conversion_id': job_id,             # ✅ ALIAS
    'error': 'Missing target profile ID',
    'code': 'MISSING_PROFILE'
}, to=sid)
```

**Fields**:
- ✅ `job_id`: Primary identifier (required)
- ✅ `conversion_id`: Backward compatibility alias
- ✅ `error`: Human-readable error message
- ✅ `code`: Machine-readable error code
- ✅ `to=sid`: Sent to specific client session

---

### 3. Vocal Volume Validation (Lines 346-352)

**Location**: `handle_convert_song_stream` event handler
**Trigger**: `vocal_volume` parameter outside valid range [0.0, 2.0]

```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,                          # ✅ PRIMARY
    'conversion_id': job_id,                   # ✅ ALIAS
    'error': 'Volume must be between 0.0 and 2.0',
    'message': 'Volume must be between 0.0 and 2.0',
    'code': 'INVALID_PARAMS'
}, to=sid)
```

**Fields**:
- ✅ `job_id`: Primary identifier (required)
- ✅ `conversion_id`: Backward compatibility alias
- ✅ `error`: Human-readable error message
- ✅ `message`: Duplicate error message (for compatibility)
- ✅ `code`: Machine-readable error code
- ✅ `to=sid`: Sent to specific client session

---

### 4. Instrumental Volume Validation (Lines 356-362)

**Location**: `handle_convert_song_stream` event handler
**Trigger**: `instrumental_volume` parameter outside valid range [0.0, 2.0]

```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,                          # ✅ PRIMARY
    'conversion_id': job_id,                   # ✅ ALIAS
    'error': 'Volume must be between 0.0 and 2.0',
    'message': 'Volume must be between 0.0 and 2.0',
    'code': 'INVALID_PARAMS'
}, to=sid)
```

**Fields**:
- ✅ `job_id`: Primary identifier (required)
- ✅ `conversion_id`: Backward compatibility alias
- ✅ `error`: Human-readable error message
- ✅ `message`: Duplicate error message (for compatibility)
- ✅ `code`: Machine-readable error code
- ✅ `to=sid`: Sent to specific client session

---

### 5. Pipeline Unavailable Error (Lines 368-373)

**Location**: `handle_convert_song_stream` event handler
**Trigger**: `singing_conversion_pipeline` not available in Flask app context

```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,                              # ✅ PRIMARY
    'conversion_id': job_id,                       # ✅ ALIAS
    'error': 'Singing conversion pipeline not available',
    'code': 'SERVICE_UNAVAILABLE'
}, to=sid)
```

**Fields**:
- ✅ `job_id`: Primary identifier (required)
- ✅ `conversion_id`: Backward compatibility alias
- ✅ `error`: Human-readable error message
- ✅ `code`: Machine-readable error code
- ✅ `to=sid`: Sent to specific client session

---

### 6. General Conversion Failure (Lines 572-578)

**Location**: `handle_convert_song_stream` exception handler
**Trigger**: Any unhandled exception during conversion process

```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,                    # ✅ PRIMARY
    'conversion_id': job_id,             # ✅ ALIAS
    'error': str(e),
    'code': 'CONVERSION_FAILED',
    'stage': self.sessions.get(f'conversion_{sid}_{job_id}', {}).get('stage', 'Unknown')
}, to=sid)
```

**Fields**:
- ✅ `job_id`: Primary identifier (required)
- ✅ `conversion_id`: Backward compatibility alias
- ✅ `error`: Human-readable error message (exception string)
- ✅ `code`: Machine-readable error code
- ✅ `stage`: Current conversion stage when error occurred
- ✅ `to=sid`: Sent to specific client session

---

## Frontend Consumption Verification

**File**: `/home/kp/repos/autovoice/frontend/src/services/websocket.ts`

### Error Handler (Lines 100-107)

```typescript
this.socket.on('conversion_error', (data: {
  job_id: string;        // ✅ Uses job_id
  error: string;
  details?: any
}) => {
  const callback = this.errorCallbacks.get(data.job_id)  // ✅ References job_id
  if (callback) {
    callback({ message: data.error, details: data.details })
  }
  this.unsubscribeFromJob(data.job_id)  // ✅ Uses job_id for cleanup
})
```

**Analysis**:
- ✅ TypeScript interface expects `job_id` field
- ✅ Error callback lookup uses `data.job_id`
- ✅ Cleanup operation uses `data.job_id`
- ✅ No references to `conversion_id` (correctly ignores backward compat field)

---

## Error Code Catalog

| Error Code | Description | HTTP Equivalent | User Action |
|------------|-------------|-----------------|-------------|
| `MISSING_SONG` | No audio file provided | 400 Bad Request | Upload audio file |
| `MISSING_PROFILE` | No target profile specified | 400 Bad Request | Select target voice |
| `INVALID_PARAMS` | Parameter validation failed | 400 Bad Request | Check volume settings (0.0-2.0) |
| `SERVICE_UNAVAILABLE` | Pipeline not initialized | 503 Service Unavailable | Retry later or contact support |
| `CONVERSION_FAILED` | Unhandled conversion error | 500 Internal Server Error | Check logs, retry, or contact support |

---

## Compliance Checklist

### Backend (Python)

- [x] All error emissions use `job_id` as primary identifier
- [x] All error emissions include `conversion_id` as backward compatibility alias
- [x] All error payloads include human-readable `error` message
- [x] All error payloads include machine-readable `code` field
- [x] All errors sent to specific client using `to=sid`
- [x] Consistent field ordering: `job_id`, `conversion_id`, `error`, `code`, [optional fields]

### Frontend (TypeScript)

- [x] Error handler types specify `job_id: string`
- [x] Error callback lookups use `data.job_id`
- [x] Cleanup operations use `data.job_id`
- [x] No legacy `conversion_id` references in new code

### Documentation

- [x] Standard pattern documented in `INTEGRATION_ISSUES.md`
- [x] Examples provided for both backend and frontend
- [x] Error code catalog maintained
- [x] Backward compatibility strategy documented

---

## Migration Notes

### For Legacy Clients

Legacy clients using `conversion_id` will continue to work because:

1. **Backward Compatibility Alias**: All events include `conversion_id: job_id`
2. **Same Value**: Both `job_id` and `conversion_id` contain the same identifier
3. **No Breaking Changes**: Existing integrations function without modification

### Recommended Migration Path

```typescript
// OLD (still works, but deprecated)
this.socket.on('conversion_error', (data) => {
  const jobId = data.conversion_id  // ⚠️ Deprecated
})

// NEW (recommended)
this.socket.on('conversion_error', (data) => {
  const jobId = data.job_id  // ✅ Preferred
})
```

---

## Future Considerations

### Phase Out conversion_id (v3.0)

When backward compatibility is no longer required:

1. Remove `conversion_id` field from all payloads
2. Update documentation to remove migration notes
3. Simplify event payload structures
4. Release as major version with breaking changes notice

### Add Structured Error Details

Consider enhancing error payloads with:

```python
self.socketio.emit('conversion_error', {
    'job_id': job_id,
    'error': {
        'message': 'Volume must be between 0.0 and 2.0',
        'code': 'INVALID_PARAMS',
        'field': 'vocal_volume',
        'value': received_value,
        'constraint': {'min': 0.0, 'max': 2.0}
    }
}, to=sid)
```

---

## Conclusion

**All WebSocket error payloads are fully standardized and compliant.**

- ✅ 6/6 error emissions follow the `job_id` (primary) + `conversion_id` (alias) pattern
- ✅ Frontend correctly consumes `job_id` in all error handlers
- ✅ Documentation updated with clear standards and examples
- ✅ Backward compatibility maintained for legacy clients
- ✅ Error codes catalogued and documented

**No action required** - implementation is correct and complete.

---

**Audit Performed By**: Code Implementation Agent
**Audit Date**: 2025-11-17
**Next Review**: Before v3.0 release (consider removing `conversion_id` alias)
