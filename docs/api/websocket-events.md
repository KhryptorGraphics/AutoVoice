# WebSocket Events Documentation

AutoVoice uses Socket.IO for real-time bidirectional communication. This document describes all available WebSocket events.

## Connection

Connect to the Socket.IO endpoint:

```javascript
const socket = io('http://localhost:5000');
```

## Namespaces

### Default Namespace (`/`)

General application events.

#### Client Events

None currently defined.

#### Server Events

**`job_created`**
Broadcast when a new conversion job is created.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Join this job room to receive progress updates"
}
```

**`job_progress`**
Emitted to job room when conversion progresses.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45,
  "stage": "Vocal separation",
  "message": "Separating vocals from instrumental..."
}
```

**`job_complete`**
Emitted when job completes successfully.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "download_url": "/api/v1/convert/download/550e8400-e29b-41d4-a716-446655440000",
  "metadata": {
    "duration": 180.5,
    "sample_rate": 44100
  }
}
```

**`job_failed`**
Emitted when job fails.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": "Conversion failed",
  "message": "Voice separation failed: insufficient audio quality"
}
```

---

### Karaoke Namespace (`/karaoke`)

Real-time karaoke and separation events.

#### Client Events

**`join_session`**
Join a karaoke session room.

```json
{
  "session_id": "session_123"
}
```

**`leave_session`**
Leave a karaoke session room.

```json
{
  "session_id": "session_123"
}
```

**`start_separation`**
Request separation of a track.

```json
{
  "job_id": "job_456",
  "settings": {
    "model": "htdemucs",
    "output_format": "wav"
  }
}
```

#### Server Events

**`separation_progress`**
Emitted during separation processing.

```json
{
  "job_id": "job_456",
  "progress": 60,
  "status": "Separating stems",
  "stage": "Processing vocals",
  "current_step": 3,
  "total_steps": 5
}
```

**`separation_complete`**
Emitted when separation completes.

```json
{
  "job_id": "job_456",
  "status": "completed",
  "stems": {
    "vocals": "/api/v1/separation/job_456/vocals.wav",
    "instrumental": "/api/v1/separation/job_456/instrumental.wav",
    "drums": "/api/v1/separation/job_456/drums.wav",
    "bass": "/api/v1/separation/job_456/bass.wav"
  },
  "metadata": {
    "duration": 210.3,
    "sample_rate": 44100,
    "model": "htdemucs"
  }
}
```

**`separation_failed`**
Emitted when separation fails.

```json
{
  "job_id": "job_456",
  "status": "failed",
  "error": "Separation failed",
  "message": "Audio file corrupted or invalid format"
}
```

**`track_added`**
Emitted when a track is added to queue.

```json
{
  "track_id": "track_789",
  "title": "Song Title",
  "artist": "Artist Name",
  "duration": 180,
  "position": 3
}
```

**`playback_state`**
Emitted when playback state changes.

```json
{
  "state": "playing",
  "track_id": "track_789",
  "position": 45.2,
  "timestamp": 1234567890
}
```

---

### Training Namespace (`/training`)

Training job progress and events.

#### Client Events

**`join_training`**
Subscribe to training job updates.

```json
{
  "job_id": "train_123"
}
```

**`leave_training`**
Unsubscribe from training job updates.

```json
{
  "job_id": "train_123"
}
```

#### Server Events

**`training_started`**
Emitted when training begins.

```json
{
  "job_id": "train_123",
  "profile_id": "profile_456",
  "status": "training",
  "config": {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "adapter_type": "unified"
  }
}
```

**`training_progress`**
Emitted during training with metrics.

```json
{
  "job_id": "train_123",
  "status": "training",
  "progress": 35,
  "epoch": 35,
  "total_epochs": 100,
  "metrics": {
    "loss": 0.0234,
    "learning_rate": 0.0001,
    "steps_per_second": 2.5
  },
  "estimated_completion": "2026-02-01T12:45:00Z"
}
```

**`training_complete`**
Emitted when training completes successfully.

```json
{
  "job_id": "train_123",
  "status": "completed",
  "progress": 100,
  "profile_id": "profile_456",
  "adapter_path": "/models/profile_456/unified_adapter.pth",
  "final_metrics": {
    "final_loss": 0.0089,
    "total_time": 3600,
    "epochs_completed": 100
  }
}
```

**`training_failed`**
Emitted when training fails.

```json
{
  "job_id": "train_123",
  "status": "failed",
  "error": "Training failed",
  "message": "Insufficient GPU memory",
  "epoch": 42
}
```

**`checkpoint_saved`**
Emitted when a training checkpoint is saved.

```json
{
  "job_id": "train_123",
  "epoch": 50,
  "checkpoint_path": "/models/profile_456/checkpoint_epoch_50.pth",
  "metrics": {
    "loss": 0.0156,
    "validation_loss": 0.0178
  }
}
```

---

## Usage Examples

### JavaScript (Browser)

```javascript
// Connect to Socket.IO
const socket = io('http://localhost:5000');

// Join job room to receive updates
const jobId = '550e8400-e29b-41d4-a716-446655440000';
socket.emit('join', jobId);

// Listen for progress
socket.on('job_progress', (data) => {
  console.log(`Progress: ${data.progress}%`);
  console.log(`Stage: ${data.stage}`);
  updateProgressBar(data.progress);
});

// Listen for completion
socket.on('job_complete', (data) => {
  console.log('Job completed!');
  downloadResult(data.download_url);
});

// Listen for errors
socket.on('job_failed', (data) => {
  console.error('Job failed:', data.message);
  showError(data.error);
});

// Karaoke namespace
const karaokeSocket = io('http://localhost:5000/karaoke');

karaokeSocket.on('separation_progress', (data) => {
  updateSeparationProgress(data.progress);
});

karaokeSocket.on('separation_complete', (data) => {
  console.log('Stems ready:', data.stems);
  loadStems(data.stems);
});

// Training namespace
const trainingSocket = io('http://localhost:5000/training');

trainingSocket.emit('join_training', { job_id: 'train_123' });

trainingSocket.on('training_progress', (data) => {
  console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
  console.log(`Loss: ${data.metrics.loss}`);
  updateTrainingChart(data.metrics);
});

trainingSocket.on('training_complete', (data) => {
  console.log('Training complete!');
  console.log('Final loss:', data.final_metrics.final_loss);
  notifyTrainingComplete();
});
```

### Python (socketio-client)

```python
import socketio

# Create client
sio = socketio.Client()

# Connect
sio.connect('http://localhost:5000')

# Join job room
job_id = '550e8400-e29b-41d4-a716-446655440000'
sio.emit('join', job_id)

# Event handlers
@sio.on('job_progress')
def on_progress(data):
    print(f"Progress: {data['progress']}%")
    print(f"Stage: {data['stage']}")

@sio.on('job_complete')
def on_complete(data):
    print("Job completed!")
    download_url = data['download_url']
    # Download result...

@sio.on('job_failed')
def on_failed(data):
    print(f"Job failed: {data['message']}")

# Wait for events
sio.wait()
```

### React (socket.io-client)

```typescript
import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

function ConversionProgress({ jobId }: { jobId: string }) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    // Connect to Socket.IO
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    // Join job room
    newSocket.emit('join', jobId);

    // Listen for progress
    newSocket.on('job_progress', (data) => {
      setProgress(data.progress);
      setStatus(data.stage);
    });

    // Listen for completion
    newSocket.on('job_complete', (data) => {
      setProgress(100);
      setStatus('Complete');
      // Handle download
      window.location.href = data.download_url;
    });

    // Listen for errors
    newSocket.on('job_failed', (data) => {
      setStatus(`Failed: ${data.message}`);
    });

    // Cleanup
    return () => {
      newSocket.emit('leave', jobId);
      newSocket.close();
    };
  }, [jobId]);

  return (
    <div>
      <div className="progress-bar" style={{ width: `${progress}%` }} />
      <p>{status}</p>
    </div>
  );
}
```

---

## Error Handling

All error events include:

- `error`: Short error identifier
- `message`: Human-readable error description
- `job_id`: Associated job identifier (if applicable)

Common error types:

- `validation_error`: Invalid input parameters
- `processing_error`: Error during processing
- `resource_error`: Insufficient resources (GPU memory, disk space)
- `timeout_error`: Operation timed out
- `not_found_error`: Resource not found

---

## Rate Limiting

WebSocket connections are rate-limited to prevent abuse:

- Max 10 connections per IP address
- Max 100 events per minute per connection
- Connections idle for >30 minutes are automatically closed

---

## Authentication

Currently no authentication required for WebSocket connections. Production deployments should implement:

- JWT token authentication
- Room-based access control
- Connection validation

Example future authentication:

```javascript
const socket = io('http://localhost:5000', {
  auth: {
    token: 'your-jwt-token'
  }
});
```
