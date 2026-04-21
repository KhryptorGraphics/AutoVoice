# AutoVoice API Tutorials

Comprehensive tutorials for common AutoVoice workflows.

---

## Tutorial 1: Convert a Song

Convert a song to a target voice in 3 simple steps.

### Prerequisites

- AutoVoice server running
- A voice profile ID (create one first, see Tutorial 2)
- An audio file to convert (MP3, WAV, FLAC, or OGG)

### Step 1: Upload and Convert

**Using curl:**

```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@my_song.mp3" \
  -F "profile_id=profile_123" \
  -F "vocal_volume=1.0" \
  -F "instrumental_volume=0.9" \
  -F "output_quality=high" \
  -F "pipeline_type=quality_seedvc"
```

**Response (Async Mode):**

```json
{
  "status": "queued",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "websocket_room": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Join WebSocket room with job_id to receive progress updates"
}
```

### Step 2: Poll for Status

```bash
curl http://localhost:5000/api/v1/convert/status/550e8400-e29b-41d4-a716-446655440000
```

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45,
  "message": "Converting vocals...",
  "created_at": "2026-02-01T10:00:00Z",
  "updated_at": "2026-02-01T10:02:30Z"
}
```

### Step 3: Download Result

When status is `completed`:

```bash
curl -o converted_song.wav \
  http://localhost:5000/api/v1/convert/download/550e8400-e29b-41d4-a716-446655440000
```

### Advanced Options

**Sync Mode (Inline Response):**

Disable JobManager in config to get immediate response with base64 audio:

```json
{
  "status": "success",
  "job_id": "...",
  "audio": "UklGRiQAAABXQVZFZm10...",
  "format": "wav",
  "sample_rate": 44100,
  "duration": 180.5
}
```

**With Settings JSON:**

```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@my_song.mp3" \
  -F 'settings={
    "target_profile_id": "profile_123",
    "vocal_volume": 1.2,
    "instrumental_volume": 0.8,
    "pitch_shift": -2,
    "output_quality": "studio",
    "adapter_type": "unified",
    "pipeline_type": "quality_seedvc"
  }'
```

**Return Stems:**

Get separate vocal and instrumental tracks:

```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@my_song.mp3" \
  -F "profile_id=profile_123" \
  -F "return_stems=true"
```

Response includes separate stems in metadata.

---

## Tutorial 2: Train a Voice Profile

Create and train a custom voice model.

### Step 1: Create Voice Profile

Upload 5-10 clean audio samples of the target voice:

```bash
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "name=My Artist" \
  -F "samples=@sample1.wav" \
  -F "samples=@sample2.wav" \
  -F "samples=@sample3.wav" \
  -F "samples=@sample4.wav" \
  -F "samples=@sample5.wav"
```

**Response:**

```json
{
  "profile_id": "profile_550e8400",
  "name": "My Artist",
  "created_at": "2026-02-01T10:00:00Z",
  "sample_count": 5,
  "embedding_dim": 256,
  "metadata": {
    "avg_sample_duration": 12.5,
    "total_duration": 62.5
  }
}
```

### Step 2: Start Training Job

```bash
curl -X POST http://localhost:5000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "profile_550e8400",
    "sample_ids": ["sample_a", "sample_b"],
    "config": {
      "training_mode": "lora",
      "epochs": 100,
      "batch_size": 8,
      "learning_rate": 0.0001,
      "adapter_type": "unified"
    }
  }'
```

**Response:**

```json
{
  "job_id": "train_abc123",
  "profile_id": "profile_550e8400",
  "status": "pending",
  "progress": 0,
  "created_at": "2026-02-01T10:05:00Z"
}
```

### Step 3: Monitor Training Progress

**Via REST API:**

```bash
curl http://localhost:5000/api/v1/training/jobs/train_abc123
```

**Via WebSocket:**

```javascript
const socket = io('http://localhost:5000');
const trainingJobId = 'train_abc123';

socket.on('training_progress', (data) => {
  if (data.job_id !== trainingJobId) return;
  console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
  console.log(`Loss: ${data.loss}`);
});

socket.on('training_complete', (data) => {
  if (data.job_id !== trainingJobId) return;
  console.log('Training complete!');
  console.log('Results:', data.results);
});
```

**Optional job controls:**

```bash
curl -X POST http://localhost:5000/api/v1/training/jobs/train_abc123/pause
curl -X POST http://localhost:5000/api/v1/training/jobs/train_abc123/resume
curl http://localhost:5000/api/v1/training/jobs/train_abc123/telemetry
curl -X POST http://localhost:5000/api/v1/training/preview/train_abc123
```

### Step 4: Verify Training Complete

Check training status:

```bash
curl http://localhost:5000/api/v1/voice/profiles/profile_550e8400/training-status
```

**Response:**

```json
{
  "profile_id": "profile_550e8400",
  "training_status": "completed",
  "last_training_job": "train_abc123",
  "model_ready": true,
  "adapter_type": "unified",
  "metrics": {
    "final_loss": 0.0089,
    "epochs_completed": 100,
    "training_time": 3600
  }
}
```

### Step 5: Use Trained Profile

Now you can use this profile for conversion:

```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@test_song.mp3" \
  -F "profile_id=profile_550e8400" \
  -F "adapter_type=unified"
```

---

## Tutorial 3: Live Karaoke Session

Set up a real-time karaoke session with voice conversion.

### Step 1: Separate Vocals from Song

```bash
curl -X POST http://localhost:5000/api/v1/karaoke/separate \
  -F "audio=@song.mp3" \
  -F "model=htdemucs"
```

**Response:**

```json
{
  "job_id": "sep_xyz789",
  "status": "queued"
}
```

### Step 2: Monitor Separation Progress

```javascript
const socket = io('http://localhost:5000/karaoke');

socket.on('separation_progress', (data) => {
  if (data.job_id === 'sep_xyz789') {
    console.log(`Progress: ${data.progress}%`);
  }
});

socket.on('separation_complete', (data) => {
  if (data.job_id === 'sep_xyz789') {
    console.log('Stems ready!');
    console.log('Vocals:', data.stems.vocals);
    console.log('Instrumental:', data.stems.instrumental);
    loadKaraokeTrack(data.stems);
  }
});
```

### Step 3: Get Separation Status

```bash
curl http://localhost:5000/api/v1/separation/sep_xyz789/status
```

### Step 4: Start Real-time Voice Conversion

Use realtime pipeline for low-latency conversion:

```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@vocals.wav" \
  -F "profile_id=profile_123" \
  -F "pipeline_type=realtime" \
  -F "adapter_type=nvfp4"
```

The realtime pipeline provides <100ms latency for live karaoke.

### Step 5: Mix Converted Vocals with Instrumental

Download converted vocals and mix locally, or use the API's built-in mixing:

```javascript
// Client-side mixing with Web Audio API
const audioContext = new AudioContext();

async function mixTracks(vocalsUrl, instrumentalUrl) {
  const [vocals, instrumental] = await Promise.all([
    fetch(vocalsUrl).then(r => r.arrayBuffer()),
    fetch(instrumentalUrl).then(r => r.arrayBuffer())
  ]);

  const vocalsBuffer = await audioContext.decodeAudioData(vocals);
  const instrumentalBuffer = await audioContext.decodeAudioData(instrumental);

  // Create sources
  const vocalsSource = audioContext.createBufferSource();
  const instrumentalSource = audioContext.createBufferSource();

  vocalsSource.buffer = vocalsBuffer;
  instrumentalSource.buffer = instrumentalBuffer;

  // Create gain nodes
  const vocalsGain = audioContext.createGain();
  const instrumentalGain = audioContext.createGain();

  vocalsGain.gain.value = 1.0;
  instrumentalGain.gain.value = 0.9;

  // Connect and play
  vocalsSource.connect(vocalsGain).connect(audioContext.destination);
  instrumentalSource.connect(instrumentalGain).connect(audioContext.destination);

  vocalsSource.start();
  instrumentalSource.start();
}
```

---

## Tutorial 4: YouTube Artist Training Pipeline

Download YouTube videos and train a voice model.

### Step 1: Get Video Information

```bash
curl -X POST http://localhost:5000/api/v1/youtube/info \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  }'
```

**Response:**

```json
{
  "title": "Rick Astley - Never Gonna Give You Up",
  "duration": 213.0,
  "artist": "Rick Astley",
  "is_featured_artist": true,
  "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"
}
```

### Step 2: Download Audio

```bash
curl -X POST http://localhost:5000/api/v1/youtube/download \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "profile_id": "rick_astley"
  }'
```

**Response:**

```json
{
  "job_id": "yt_download_123",
  "status": "completed",
  "audio_path": "/tmp/youtube/rick_astley_never_gonna_give_you_up.wav",
  "metadata": {
    "title": "Rick Astley - Never Gonna Give You Up",
    "duration": 213.0,
    "artist": "Rick Astley"
  }
}
```

### Step 3: Perform Speaker Diarization

Extract vocals for the target artist:

```bash
curl -X POST http://localhost:5000/api/v1/audio/diarize \
  -F "audio=@/tmp/youtube/rick_astley_never_gonna_give_you_up.wav" \
  -F "num_speakers=1"
```

**Response:**

```json
{
  "job_id": "diarize_456",
  "segments": [
    {
      "speaker_id": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 213.0,
      "confidence": 0.95
    }
  ],
  "num_speakers": 1,
  "total_duration": 213.0
}
```

### Step 4: Add Samples to Profile

```bash
curl -X POST http://localhost:5000/api/v1/profiles/rick_astley/samples/from-path \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/tmp/youtube/rick_astley_never_gonna_give_you_up.wav",
    "speaker_id": "SPEAKER_00",
    "segments": [
      {"start_time": 10.0, "end_time": 25.0},
      {"start_time": 45.0, "end_time": 60.0},
      {"start_time": 120.0, "end_time": 135.0}
    ]
  }'
```

### Step 5: Train Profile

Follow Tutorial 2 to train the profile with the collected samples.

---

## Tutorial 5: Multi-Speaker Diarization

Separate and identify multiple speakers in a recording.

### Step 1: Upload Audio

```bash
curl -X POST http://localhost:5000/api/v1/audio/diarize \
  -F "audio=@podcast_episode.mp3"
```

**Response:**

```json
{
  "job_id": "diarize_789",
  "segments": [
    {
      "speaker_id": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 15.3,
      "confidence": 0.92
    },
    {
      "speaker_id": "SPEAKER_01",
      "start_time": 15.3,
      "end_time": 42.1,
      "confidence": 0.89
    },
    {
      "speaker_id": "SPEAKER_00",
      "start_time": 42.1,
      "end_time": 58.7,
      "confidence": 0.94
    }
  ],
  "num_speakers": 2,
  "total_duration": 180.0
}
```

### Step 2: Assign Speakers to Profiles

```bash
curl -X POST http://localhost:5000/api/v1/audio/diarize/assign \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "diarize_789",
    "assignments": {
      "SPEAKER_00": "profile_host",
      "SPEAKER_01": "profile_guest"
    }
  }'
```

### Step 3: Extract Speaker Segments

```bash
curl http://localhost:5000/api/v1/profiles/profile_host/segments \
  -G -d "job_id=diarize_789"
```

**Response:**

```json
{
  "profile_id": "profile_host",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 15.3,
      "audio_path": "/tmp/segments/profile_host_seg_001.wav"
    },
    {
      "start_time": 42.1,
      "end_time": 58.7,
      "audio_path": "/tmp/segments/profile_host_seg_002.wav"
    }
  ]
}
```

---

## Tutorial 6: Batch Processing

Process multiple songs efficiently.

### Python Script

```python
import requests
import time
from pathlib import Path

API_BASE = "http://localhost:5000/api/v1"
PROFILE_ID = "profile_123"

def convert_song(song_path: Path):
    """Convert a single song."""
    with open(song_path, 'rb') as f:
        response = requests.post(
            f"{API_BASE}/convert/song",
            files={'song': f},
            data={
                'profile_id': PROFILE_ID,
                'output_quality': 'high',
                'pipeline_type': 'quality_seedvc'
            }
        )

    if response.status_code == 202:
        return response.json()['job_id']
    else:
        raise Exception(f"Conversion failed: {response.text}")

def wait_for_completion(job_id: str):
    """Poll for job completion."""
    while True:
        response = requests.get(f"{API_BASE}/convert/status/{job_id}")
        data = response.json()

        status = data['status']
        if status == 'completed':
            return True
        elif status == 'failed':
            raise Exception(f"Job failed: {data.get('message')}")

        print(f"Progress: {data['progress']}%")
        time.sleep(2)

def download_result(job_id: str, output_path: Path):
    """Download converted audio."""
    response = requests.get(f"{API_BASE}/convert/download/{job_id}")
    with open(output_path, 'wb') as f:
        f.write(response.content)

# Process all songs in directory
songs_dir = Path("./songs")
output_dir = Path("./converted")
output_dir.mkdir(exist_ok=True)

for song_file in songs_dir.glob("*.mp3"):
    print(f"Converting {song_file.name}...")

    # Start conversion
    job_id = convert_song(song_file)
    print(f"Job ID: {job_id}")

    # Wait for completion
    wait_for_completion(job_id)

    # Download result
    output_file = output_dir / f"{song_file.stem}_converted.wav"
    download_result(job_id, output_file)
    print(f"Saved to {output_file}")

print("Batch processing complete!")
```

---

## Common Patterns

### Error Handling

```python
def convert_with_retry(song_path, profile_id, max_retries=3):
    """Convert song with automatic retry."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_BASE}/convert/song",
                files={'song': open(song_path, 'rb')},
                data={'profile_id': profile_id},
                timeout=300
            )

            if response.status_code == 404:
                # Profile not found or no trained model
                print(f"Error: {response.json()['message']}")
                return None

            response.raise_for_status()
            return response.json()['job_id']

        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1} timed out, retrying...")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)

    return None
```

### Quality Preset Selection

```python
# Draft: Fastest, lowest quality (for testing)
# Fast: Good quality, moderate speed
# Balanced: Default, good balance
# High: High quality, slower
# Studio: Best quality, slowest

QUALITY_PRESETS = {
    'testing': 'draft',
    'preview': 'fast',
    'production': 'high',
    'mastering': 'studio'
}

job_id = convert_song(
    song_path,
    profile_id,
    quality=QUALITY_PRESETS['production']
)
```

### Pipeline Selection

```python
# realtime: <100ms latency, good for live karaoke
# quality: High-fidelity, 24kHz output
# quality_seedvc: SOTA quality, 44kHz output (recommended)

PIPELINE_SELECTION = {
    'live_karaoke': 'realtime',
    'song_cover': 'quality_seedvc',
    'quick_preview': 'quality'
}

job_id = convert_song(
    song_path,
    profile_id,
    pipeline=PIPELINE_SELECTION['song_cover']
)
```

---

## Additional Resources

- [OpenAPI Specification](http://localhost:5000/api/v1/openapi.json)
- [Swagger UI](http://localhost:5000/docs)
- [WebSocket Events](./websocket-events.md)
- [Postman Collection](./postman_collection.json)
