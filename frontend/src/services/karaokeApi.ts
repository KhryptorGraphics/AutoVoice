/**
 * Karaoke API client for live voice conversion.
 */

const API_BASE = '/api/v1/karaoke';

export interface UploadedSong {
  song_id: string;
  duration: number;
  sample_rate: number;
  format: string;
  status: string;
}

export interface SeparationJob {
  job_id: string;
  song_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  estimated_remaining?: number;
  vocals_ready?: boolean;
  instrumental_ready?: boolean;
  vocals_path?: string;
  instrumental_path?: string;
  error?: string;
}

export interface AudioDevice {
  index: number;
  name: string;
  channels: number;
  default_sample_rate: number;
  is_default: boolean;
}

export interface DeviceConfig {
  speaker_device: number | null;
  headphone_device: number | null;
}

export interface VoiceModel {
  id: string;
  name: string;
  type: 'pretrained' | 'extracted';
  embedding_dim?: number;
  source_song_id?: string;
}

/**
 * Upload a song for karaoke processing.
 */
export async function uploadSong(file: File): Promise<UploadedSong> {
  const formData = new FormData();
  formData.append('song', file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Upload failed');
  }

  return response.json();
}

/**
 * Get song information.
 */
export async function getSong(songId: string): Promise<UploadedSong> {
  const response = await fetch(`${API_BASE}/songs/${songId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Song not found');
  }

  return response.json();
}

/**
 * Start vocal/instrumental separation.
 */
export async function startSeparation(songId: string): Promise<SeparationJob> {
  const response = await fetch(`${API_BASE}/separate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ song_id: songId }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Separation failed to start');
  }

  return response.json();
}

/**
 * Get separation job status.
 */
export async function getSeparationStatus(jobId: string): Promise<SeparationJob> {
  const response = await fetch(`${API_BASE}/separate/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Job not found');
  }

  return response.json();
}

/**
 * List available audio output devices.
 */
export async function listDevices(): Promise<{ devices: AudioDevice[]; count: number }> {
  const response = await fetch(`${API_BASE}/devices`);

  if (!response.ok) {
    throw new Error('Failed to list devices');
  }

  return response.json();
}

/**
 * Get current output device configuration.
 */
export async function getDeviceConfig(): Promise<DeviceConfig> {
  const response = await fetch(`${API_BASE}/devices/output`);

  if (!response.ok) {
    throw new Error('Failed to get device config');
  }

  return response.json();
}

/**
 * Set output device configuration.
 */
export async function setDeviceConfig(config: Partial<DeviceConfig>): Promise<DeviceConfig> {
  const response = await fetch(`${API_BASE}/devices/output`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to set device config');
  }

  return response.json();
}

/**
 * List available voice models.
 */
export async function listVoiceModels(): Promise<{ models: VoiceModel[]; count: number }> {
  const response = await fetch(`${API_BASE}/voice-models`);

  if (!response.ok) {
    throw new Error('Failed to list voice models');
  }

  return response.json();
}

/**
 * Get voice model details.
 */
export async function getVoiceModel(modelId: string): Promise<VoiceModel> {
  const response = await fetch(`${API_BASE}/voice-models/${modelId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Voice model not found');
  }

  return response.json();
}

/**
 * Extract voice model from song vocals.
 */
export async function extractVoiceModel(
  songId: string,
  name: string
): Promise<{ model_id: string; name: string; type: string }> {
  const response = await fetch(`${API_BASE}/voice-models/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ song_id: songId, name }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to extract voice model');
  }

  return response.json();
}
