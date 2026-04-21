# WebSocket Events

AutoVoice uses Socket.IO for realtime updates. The current single-user MVP has one canonical non-karaoke namespace: the default namespace (`/`).

## Connection

Default namespace:

```javascript
import { io } from "socket.io-client";

const socket = io("http://localhost:5000");
```

Karaoke namespace:

```javascript
import { io } from "socket.io-client";

const karaokeSocket = io("http://localhost:5000/karaoke");
```

## Namespace Map

- `/`: conversion jobs, training jobs, and generic application events
- `/karaoke`: live karaoke streaming and separation events

There is no separate `/training` namespace in the current backend.

## Default Namespace (`/`)

### Client Events

`join_job`

Join a conversion job room. This is the canonical room handshake for conversion progress isolation.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

`leave_job`

Leave a previously joined conversion job room.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Server Events

`joined_job`

Acknowledges a successful room join.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

`left_job`

Acknowledges a successful room leave.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

`job_subscription_error`

Returned when a room subscription request is invalid.

```json
{
  "message": "job_id is required"
}
```

`job_created`

Broadcast when a new conversion job is created.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "websocket_room": "550e8400-e29b-41d4-a716-446655440000"
}
```

`job_progress` and `conversion_progress`

Both events are emitted during conversion. `conversion_progress` is the frontend-facing alias.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": 45,
  "message": "Converting vocals...",
  "stage": "encoding",
  "timestamp": 1761111111.25
}
```

`job_completed` and `conversion_complete`

Both events are emitted when conversion succeeds. `conversion_complete` is the frontend-facing alias.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "output_url": "/api/v1/convert/download/550e8400-e29b-41d4-a716-446655440000",
  "download_url": "/api/v1/convert/download/550e8400-e29b-41d4-a716-446655440000",
  "requested_pipeline": "quality_seedvc",
  "resolved_pipeline": "quality_seedvc",
  "runtime_backend": "pytorch",
  "active_model_type": "adapter",
  "adapter_type": "hq"
}
```

`job_failed` and `conversion_error`

Both events are emitted when conversion fails. `conversion_error` is the frontend-facing alias.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": "Profile missing speaker embedding for realtime conversion"
}
```

### Training Events

Training events are emitted on the default namespace and should currently be filtered by `job_id` on the client.

Canonical dotted events:

- `training.started`
- `training.progress`
- `training.completed`
- `training.failed`
- `training.paused`
- `training.resumed`
- `training.cancelled`

Compatibility aliases still emitted for active UI paths:

- `training_progress`
- `training_complete`
- `training_error`
- `training_paused`
- `training_resumed`
- `training_cancelled`

`training.started`

```json
{
  "job_id": "train_abc123",
  "profile_id": "profile_550e8400",
  "sample_count": 5,
  "config": {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001
  },
  "started_at": "2026-02-01T10:05:00"
}
```

`training.progress` and `training_progress`

```json
{
  "job_id": "train_abc123",
  "profile_id": "profile_550e8400",
  "epoch": 3,
  "total_epochs": 10,
  "step": 150,
  "total_steps": 500,
  "loss": 0.45,
  "learning_rate": 0.0001,
  "progress_percent": 25.0,
  "gpu_metrics": {
    "available": false
  },
  "quality_metrics": {
    "mos_proxy": 4.2
  },
  "checkpoint_path": null,
  "is_paused": false
}
```

`training.completed` and `training_complete`

```json
{
  "job_id": "train_abc123",
  "profile_id": "profile_550e8400",
  "results": {
    "final_loss": 0.15,
    "artifact_type": "adapter"
  },
  "completed_at": "2026-02-01T11:05:00"
}
```

`training.failed` and `training_error`

```json
{
  "job_id": "train_abc123",
  "profile_id": "profile_550e8400",
  "error": "CUDA out of memory",
  "failed_at": "2026-02-01T10:17:00"
}
```

## Karaoke Namespace (`/karaoke`)

The `/karaoke` namespace is reserved for live session control, streaming, and separation workflow events.

Client events include:

- `join_session`
- `leave_session`
- `startSession`
- `stopSession`
- `switchProfile`
- `audioChunk`

Server events include:

- `session_joined`
- `session_left`
- `separation_progress`
- `separation_complete`
- `separation_failed`
- live session status and recovery events emitted by the karaoke namespace handlers

See [tutorials.md](./tutorials.md) for end-to-end usage flows.

## Examples

### Conversion Job Subscription

```javascript
const socket = io("http://localhost:5000");
const jobId = "550e8400-e29b-41d4-a716-446655440000";

socket.emit("join_job", { job_id: jobId });

socket.on("joined_job", (data) => {
  if (data.job_id === jobId) {
    console.log("Joined job room");
  }
});

socket.on("conversion_progress", (data) => {
  if (data.job_id === jobId) {
    console.log(data.stage, data.progress);
  }
});
```

### Training Progress Subscription

```javascript
const socket = io("http://localhost:5000");
const trainingJobId = "train_abc123";

socket.on("training_progress", (data) => {
  if (data.job_id === trainingJobId) {
    console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
    console.log(`Loss: ${data.loss}`);
  }
});

socket.on("training_complete", (data) => {
  if (data.job_id === trainingJobId) {
    console.log("Training complete", data.results);
  }
});
```
