import { io, Socket } from 'socket.io-client'

const API_BASE = '/api/v1'

// WebSocket event types
export type WSEventType =
  | 'conversion_progress'
  | 'conversion_complete'
  | 'conversion_error'
  | 'training_progress'
  | 'training_complete'
  | 'training_error'
  | 'gpu_metrics'
  | 'model_loaded'
  | 'model_unloaded'

export interface WSEvent<T = unknown> {
  type: WSEventType
  timestamp: string
  data: T
}

export interface ConversionProgressEvent {
  job_id: string
  progress: number
  stage: 'separating' | 'encoding' | 'converting' | 'vocoding' | 'mixing'
  message?: string
}

export interface TrainingProgressEvent {
  job_id: string
  epoch: number
  total_epochs: number
  step: number
  total_steps: number
  loss: number
  learning_rate: number
}

export type WSEventHandler<T = unknown> = (event: WSEvent<T>) => void

export interface ConversionRecord {
  id: string
  status: 'queued' | 'processing' | 'in_progress' | 'complete' | 'completed' | 'error' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  input_file: string
  profile_id: string
  preset: string
  duration?: number
  error?: string
  // Pipeline and adapter info
  pipeline_type?: 'realtime' | 'quality' | 'quality_seedvc' | 'realtime_meanvc' | 'quality_shortcut'
  adapter_type?: 'hq' | 'nvfp4' | 'unified'
  active_model_type?: ActiveModelType
  // Quality metrics
  processing_time_seconds?: number
  rtf?: number  // Real-time factor (processing_time / audio_duration)
  audio_duration_seconds?: number
  // Output URLs
  output_url?: string
  download_url?: string
  // Additional fields used by ConversionHistoryPage
  timestamp?: Date
  isFavorite?: boolean
  targetVoice?: string
  quality?: string
  originalFileName?: string
  notes?: string
  resultUrl?: string
}

export interface YouTubeHistoryItem {
  id: string
  url: string
  title: string
  mainArtist: string | null
  featuredArtists: string[]
  hasDiarization: boolean
  numSpeakers: number
  timestamp: string
  audioPath: string | null
  filteredPath: string | null
  videoId?: string | null
}

export type ProfileRole = 'source_artist' | 'target_user'
export type ActiveModelType = 'base' | 'adapter' | 'full_model'

export interface VoiceProfile {
  profile_id: string
  user_id?: string
  name?: string
  created_at: string
  created_from?: string
  profile_role?: ProfileRole
  sample_count: number
  training_sample_count?: number
  clean_vocal_seconds?: number
  clean_vocal_minutes?: number
  full_model_unlock_seconds?: number
  full_model_remaining_seconds?: number
  full_model_remaining_minutes?: number
  full_model_eligible?: boolean
  model_version?: string
  last_trained?: string
  quality_score?: number
  training_status?: TrainingStatusType
  has_trained_model?: boolean
  has_full_model?: boolean
  has_adapter_model?: boolean
  active_model_type?: ActiveModelType
  selected_adapter?: 'hq' | 'nvfp4' | 'unified' | null
}

// Training status for voice profiles
export type TrainingStatusType = 'pending' | 'training' | 'ready' | 'failed'

export interface TrainingStatus {
  profile_id: string
  has_trained_model: boolean
  training_status: TrainingStatusType
}

// Adapter types for LoRA model selection
export type AdapterType = 'hq' | 'nvfp4'

export interface Adapter {
  type: AdapterType
  path: string
  size_kb: number
  epochs: number
  loss: number | null
  precision: string
  config: Record<string, unknown>
}

export interface AdapterListResponse {
  profile_id: string
  adapters: Adapter[]
  selected: AdapterType | null
  count: number
}

export interface AdapterMetrics {
  epochs: number
  loss: number | null
  precision: string
  trained_on: string | null
  architecture: {
    input_dim: number
    hidden_dim: number
    output_dim: number
    num_layers: number
    lora_rank: number
    lora_alpha: number
  }
  file_size_kb: number
  parameter_count: number
  parameter_count_formatted: string
  file_path: string
  modified_time: string
  performance: {
    relative_quality: string
    relative_speed: string
    memory_estimate_mb: number
  }
}

export interface AdapterMetricsResponse {
  profile_id: string
  profile_name: string
  adapters: Record<AdapterType, AdapterMetrics>
  adapter_count: number
  recommended: AdapterType | null
}

export interface TrainingJob {
  job_id: string
  profile_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  progress: number
  sample_ids: string[]
  error?: string
  results?: {
    initial_loss?: number
    final_loss?: number
    loss_curve?: number[]
    artifact_type?: 'adapter' | 'full_model'
    job_type?: 'lora' | 'full_model'
  }
}

// Full training configuration with all LoRA/EWC parameters
export interface TrainingConfig {
  // Training mode: 'lora' by default, 'full' unlocks after 30 minutes of clean user vocals
  training_mode: 'lora' | 'full'
  // LoRA configuration (only used when training_mode='lora')
  lora_rank: number
  lora_alpha: number
  lora_dropout: number
  lora_target_modules: string[]
  // Training parameters
  learning_rate: number
  batch_size: number
  epochs: number
  warmup_steps: number
  max_grad_norm: number
  // EWC configuration (prevent catastrophic forgetting)
  use_ewc: boolean
  ewc_lambda: number
  // Prior preservation
  use_prior_preservation: boolean
  prior_loss_weight: number
}

// Default training config matching backend defaults
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  training_mode: 'lora',
  lora_rank: 8,
  lora_alpha: 16,
  lora_dropout: 0.1,
  lora_target_modules: ['q_proj', 'v_proj', 'content_encoder'],
  learning_rate: 1e-4,
  batch_size: 4,
  epochs: 100,
  warmup_steps: 100,
  max_grad_norm: 1.0,
  use_ewc: true,
  ewc_lambda: 1000.0,
  use_prior_preservation: false,
  prior_loss_weight: 0.5,
}

export interface AudioDevice {
  device_id: string
  name: string
  type: 'input' | 'output'
  sample_rate: number
  channels: number
  is_default: boolean
}

export interface DeviceConfig {
  input_device_id?: string
  output_device_id?: string
  sample_rate: number
  buffer_size?: number
}

export interface GPUDevice {
  index: number
  name: string
  memory_used: number
  memory_total: number
  memory_free?: number
  memory_used_gb?: number
  memory_total_gb?: number
  memory_free_gb?: number
  utilization?: number
  utilization_percent?: number | null
  temperature?: number
  temperature_c?: number | null
}

// Full GPU metrics response from /api/v1/gpu/metrics
export interface GPUMetrics {
  available: boolean
  device_count: number
  devices: GPUDevice[]
  note?: string
}

// Conversion configuration for song conversion
export type QualityPreset = 'draft' | 'fast' | 'balanced' | 'high' | 'studio'
export type EncoderBackend = 'hubert' | 'contentvec'
export type VocoderType = 'hifigan' | 'bigvgan'

export interface ConversionConfig {
  vocal_volume: number
  instrumental_volume: number
  pitch_shift: number
  preset: QualityPreset
  return_stems: boolean
  preserve_techniques: boolean
  encoder_backend: EncoderBackend
  vocoder_type: VocoderType
}

export const DEFAULT_CONVERSION_CONFIG: ConversionConfig = {
  vocal_volume: 1.0,
  instrumental_volume: 0.9,
  pitch_shift: 0.0,
  preset: 'balanced',
  return_stems: false,
  preserve_techniques: true,
  encoder_backend: 'hubert',
  vocoder_type: 'hifigan',
}

// Quality preset details matching backend PRESETS
export const QUALITY_PRESETS: Record<QualityPreset, { n_steps: number; denoise: number; label: string }> = {
  draft: { n_steps: 10, denoise: 0.3, label: 'Draft (Fast)' },
  fast: { n_steps: 20, denoise: 0.5, label: 'Fast' },
  balanced: { n_steps: 50, denoise: 0.7, label: 'Balanced' },
  high: { n_steps: 100, denoise: 0.8, label: 'High Quality' },
  studio: { n_steps: 200, denoise: 0.9, label: 'Studio' },
}

// Training sample for voice profile
export interface TrainingSample {
  id: string
  profile_id: string
  audio_path: string
  duration_seconds: number
  sample_rate: number
  created: string
  extra_metadata?: Record<string, unknown>
}

// Quality metrics for completed conversions
export interface QualityMetrics {
  pitch_accuracy: {
    rmse_hz: number
    correlation: number
    mean_error_cents: number
  }
  speaker_similarity: {
    cosine_similarity: number
    embedding_distance: number
  }
  naturalness: {
    spectral_distortion: number
    mos_estimate: number
  }
  intelligibility?: {
    stoi: number
    pesq: number
  }
}

// Audio router configuration for karaoke dual-channel output
export interface AudioRouterConfig {
  speaker_gain: number
  headphone_gain: number
  voice_gain: number
  instrumental_gain: number
  speaker_enabled: boolean
  headphone_enabled: boolean
  speaker_device: number | null
  headphone_device: number | null
  sample_rate: number
}

export const DEFAULT_AUDIO_ROUTER_CONFIG: AudioRouterConfig = {
  speaker_gain: 1.0,
  headphone_gain: 1.0,
  voice_gain: 1.0,
  instrumental_gain: 0.8,
  speaker_enabled: true,
  headphone_enabled: true,
  speaker_device: null,
  headphone_device: null,
  sample_rate: 24000,
}

// Loaded model info
export interface LoadedModel {
  type: string
  name: string
  path?: string
  memory_mb?: number
  loaded_at?: string
}

// TensorRT engine status
export interface TensorRTStatus {
  available: boolean
  version?: string
  engines: {
    name: string
    precision: 'fp32' | 'fp16' | 'int8'
    built_at?: string
    input_shape?: number[]
    optimized?: boolean
  }[]
  build_in_progress?: boolean
}

// User preset for saving conversion configurations
export interface UserPreset {
  id: string
  name: string
  config: Partial<ConversionConfig>
  created_at: string
  updated_at?: string
}

// Model checkpoint for version control
export interface Checkpoint {
  id: string
  profile_id: string
  version: string
  created_at: string
  epochs_trained: number
  final_loss: number
  is_active: boolean
  file_size_mb: number
  training_samples: number
  notes?: string
}

// Separation configuration for Demucs vocal isolation
export interface SeparationConfig {
  model: 'htdemucs' | 'htdemucs_ft' | 'mdx_extra'
  stems: ('vocals' | 'drums' | 'bass' | 'other')[]
  shifts: number // Number of random shifts for prediction (higher = better quality, slower)
  overlap: number // Overlap between chunks (0.0-0.99)
  segment_length: number | null // Segment length in seconds (null = full track)
  device: 'cuda' | 'cpu'
}

export const DEFAULT_SEPARATION_CONFIG: SeparationConfig = {
  model: 'htdemucs',
  stems: ['vocals'],
  shifts: 1,
  overlap: 0.25,
  segment_length: null,
  device: 'cuda',
}

// Pitch extraction configuration for CREPE/RMVPE
export interface PitchConfig {
  method: 'crepe' | 'rmvpe' | 'harvest' | 'dio'
  hop_length: number
  f0_min: number
  f0_max: number
  threshold: number // Confidence threshold for pitch detection
  use_gpu: boolean
}

export const DEFAULT_PITCH_CONFIG: PitchConfig = {
  method: 'rmvpe',
  hop_length: 160,
  f0_min: 50,
  f0_max: 1100,
  threshold: 0.3,
  use_gpu: true,
}

// Conversion job response (from POST /convert/song)
export interface ConversionJobResponse {
  status: 'queued' | 'processing'
  job_id: string
  websocket_room?: string
  message?: string
  active_model_type?: ActiveModelType
  adapter_type?: 'hq' | 'nvfp4' | 'unified'
  // When sync processing is used, result may be inline
  output_url?: string
  download_url?: string
}

// Extended conversion status (from GET /convert/status/{job_id})
export interface ConversionStatusExtended extends ConversionRecord {
  // Timing info
  started_at?: string
  processing_time_seconds?: number
  // Quality metrics
  rtf?: number
  audio_duration_seconds?: number
  // Pipeline info used
  pipeline_type?: 'realtime' | 'quality' | 'quality_seedvc' | 'realtime_meanvc' | 'quality_shortcut'
  adapter_type?: 'hq' | 'nvfp4' | 'unified'
  active_model_type?: ActiveModelType
}

// CUDA kernel metrics
export interface KernelMetric {
  name: string
  calls: number
  total_time_ms: number
  avg_time_ms: number
  min_time_ms?: number
  max_time_ms?: number
}

export interface HealthStatus {
  status: string
  components: Record<string, { status: string; details?: string }>
  uptime: number
  version: string
}

export interface SystemInfo {
  device: string
  cuda_available: boolean
  gpu_count: number
  python_version: string
  torch_version: string
  // Additional fields used by GPUMonitor
  gpu_available?: boolean
  gpu_name?: string
  gpu_utilization?: number
  gpu_memory_used?: number
  gpu_memory_total?: number
  gpu_temperature?: number
  model_loaded?: boolean
  models?: string[]
  status?: string
}

// Custom error classes for better error handling
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public code?: string,
    public details?: Record<string, unknown>
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class ConversionError extends ApiError {
  constructor(
    message: string,
    statusCode: number,
    public errorType: 'missing_adapter' | 'invalid_profile' | 'pipeline_error' | 'audio_error' | 'unknown',
    details?: Record<string, unknown>
  ) {
    super(message, statusCode, errorType, details)
    this.name = 'ConversionError'
  }

  static fromResponse(error: { error: string; code?: string; details?: Record<string, unknown> }, status: number): ConversionError {
    const message = error.error || 'Conversion failed'
    let errorType: ConversionError['errorType'] = 'unknown'

    // Detect error type from message or code
    if (message.includes('adapter') || message.includes('model not found')) {
      errorType = 'missing_adapter'
    } else if (message.includes('profile') || message.includes('Profile not found')) {
      errorType = 'invalid_profile'
    } else if (message.includes('pipeline') || message.includes('Pipeline')) {
      errorType = 'pipeline_error'
    } else if (message.includes('audio') || message.includes('Audio')) {
      errorType = 'audio_error'
    }

    return new ConversionError(message, status, errorType, error.details)
  }
}

class ApiService {
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: response.statusText }))
      throw new ApiError(
        error.error || `HTTP ${response.status}`,
        response.status,
        error.code,
        error.details
      )
    }

    if (response.status === 204) {
      return undefined as T
    }

    const contentType = response.headers.get('content-type') ?? ''
    if (!contentType.includes('application/json')) {
      return undefined as T
    }

    const text = await response.text()
    return (text ? JSON.parse(text) : undefined) as T
  }

  async getHealth(): Promise<HealthStatus> {
    return this.request('/health')
  }

  async getSystemInfo(): Promise<SystemInfo> {
    return this.request('/system/info')
  }

  // Aliases for SystemStatusPage compatibility
  // Transforms nested backend response to flat structure expected by GPUMonitor
  async getSystemStatus(): Promise<SystemInfo> {
    const raw = await this.request<any>('/system/info')
    return {
      // Core fields
      device: raw.system?.platform ?? 'Unknown',
      cuda_available: raw.torch?.cuda_available ?? false,
      gpu_count: raw.torch?.device_count ?? 0,
      python_version: raw.system?.python_version ?? 'Unknown',
      torch_version: raw.torch?.version ?? 'Unknown',
      // GPUMonitor fields - transform nested torch data to flat structure
      gpu_available: raw.torch?.cuda_available ?? false,
      gpu_name: raw.torch?.device_name,
      status: raw.torch?.cuda_available ? 'ready' : 'no_gpu',
    }
  }

  async getModelsInfo(): Promise<{ models: any[] }> {
    // Return mock data until backend endpoint exists
    return { models: [] }
  }

  async healthCheck(): Promise<{ status: string; gpu_available: boolean; models_loaded: boolean; uptime: number }> {
    const health = await this.getHealth()
    // Check if torch component details indicate CUDA is available
    const torchDetails = health.components?.torch?.details ?? ''
    const gpuAvailable = torchDetails.toLowerCase().includes('cuda') || health.components?.torch?.status === 'up'
    return {
      status: health.status,
      gpu_available: gpuAvailable,
      models_loaded: health.components?.singing_pipeline?.status === 'up',
      uptime: health.uptime ?? 0,
    }
  }

  async getGPUMetrics(): Promise<GPUMetrics> {
    return this.request('/gpu/metrics')
  }

  async getKernelMetrics(): Promise<KernelMetric[]> {
    const response = await this.request<{ kernels?: KernelMetric[]; note?: string }>('/kernels/metrics')
    return response.kernels ?? []
  }

  async getConversionMetrics(jobId: string): Promise<QualityMetrics> {
    return this.request(`/convert/metrics/${jobId}`)
  }

  async createVoiceProfile(audioFile: File, name?: string, userId?: string): Promise<VoiceProfile> {
    const formData = new FormData()
    formData.append('reference_audio', audioFile)
    if (name) formData.append('name', name)
    if (userId) formData.append('user_id', userId)

    const response = await fetch(`${API_BASE}/voice/clone`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error(`Upload failed: ${response.status}`)
    return response.json()
  }

  async listProfiles(userId?: string): Promise<VoiceProfile[]> {
    const params = userId ? `?user_id=${userId}` : ''
    return this.request(`/voice/profiles${params}`)
  }

  async deleteProfile(profileId: string): Promise<void> {
    await this.request(`/voice/profiles/${profileId}`, { method: 'DELETE' })
  }

  async getProfileDetails(profileId: string): Promise<VoiceProfile & { training_history: TrainingJob[] }> {
    return this.request(`/voice/profiles/${profileId}`)
  }

  async renameProfile(profileId: string, name: string): Promise<VoiceProfile> {
    return this.request(`/voice/profiles/${profileId}`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    })
  }

  async getTrainingStatus(profileId: string): Promise<TrainingStatus> {
    return this.request(`/voice/profiles/${profileId}/training-status`)
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Adapter Methods (LoRA model selection)
  // ─────────────────────────────────────────────────────────────────────────

  async getProfileAdapters(profileId: string): Promise<AdapterListResponse> {
    return this.request(`/voice/profiles/${profileId}/adapters`)
  }

  async selectAdapter(profileId: string, adapterType: AdapterType): Promise<{ status: string; selected: AdapterType }> {
    return this.request(`/voice/profiles/${profileId}/adapter/select`, {
      method: 'POST',
      body: JSON.stringify({ adapter_type: adapterType }),
    })
  }

  async getAdapterMetrics(profileId: string): Promise<AdapterMetricsResponse> {
    return this.request(`/voice/profiles/${profileId}/adapter/metrics`)
  }

  // Training Jobs
  async listTrainingJobs(profileId?: string): Promise<TrainingJob[]> {
    const params = profileId ? `?profile_id=${profileId}` : ''
    return this.request(`/training/jobs${params}`)
  }

  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    return this.request(`/training/jobs/${jobId}`)
  }

  async createTrainingJob(profileId: string, sampleIds: string[], config?: Partial<TrainingConfig>): Promise<TrainingJob> {
    return this.request('/training/jobs', {
      method: 'POST',
      body: JSON.stringify({ profile_id: profileId, sample_ids: sampleIds, config }),
    })
  }

  async cancelTrainingJob(jobId: string): Promise<void> {
    await this.request(`/training/jobs/${jobId}/cancel`, { method: 'POST' })
  }

  // Audio Devices
  async listAudioDevices(): Promise<AudioDevice[]> {
    return this.request('/devices/list')
  }

  async getDeviceConfig(): Promise<DeviceConfig> {
    return this.request('/devices/config')
  }

  async setDeviceConfig(config: Partial<DeviceConfig>): Promise<DeviceConfig> {
    return this.request('/devices/config', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  }

  async convertSong(
    audioFile: File,
    profileId: string,
    settings?: {
      preset?: string
      vocal_volume?: number
      instrumental_volume?: number
      pitch_shift?: number
      pipeline_type?: 'realtime' | 'quality' | 'quality_seedvc' | 'realtime_meanvc' | 'quality_shortcut'
      adapter_type?: 'hq' | 'nvfp4' | 'unified'
    }
  ): Promise<ConversionJobResponse> {
    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('profile_id', profileId)
    if (settings?.preset) formData.append('preset', settings.preset)
    if (settings?.vocal_volume != null) formData.append('vocal_volume', String(settings.vocal_volume))
    if (settings?.instrumental_volume != null) formData.append('instrumental_volume', String(settings.instrumental_volume))
    if (settings?.pitch_shift != null) formData.append('pitch_shift', String(settings.pitch_shift))
    if (settings?.pipeline_type) formData.append('pipeline_type', settings.pipeline_type)
    if (settings?.adapter_type) formData.append('adapter_type', settings.adapter_type)

    const response = await fetch(`${API_BASE}/convert/song`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `Conversion failed: ${response.status}` }))
      throw ConversionError.fromResponse(errorData, response.status)
    }

    return response.json()
  }

  async getConversionStatus(jobId: string): Promise<ConversionRecord> {
    return this.request(`/convert/status/${jobId}`)
  }

  async cancelConversion(jobId: string): Promise<void> {
    await this.request(`/convert/cancel/${jobId}`, { method: 'POST' })
  }

  async downloadResult(jobId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE}/convert/download/${jobId}`)
    if (!response.ok) throw new Error(`Download failed: ${response.status}`)
    return response.blob()
  }

  // Training Samples
  async uploadSample(profileId: string, audioFile: File, metadata?: Record<string, unknown>): Promise<TrainingSample> {
    const formData = new FormData()
    formData.append('audio', audioFile)
    if (metadata) formData.append('metadata', JSON.stringify(metadata))

    const response = await fetch(`${API_BASE}/profiles/${profileId}/samples`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error(`Upload failed: ${response.status}`)
    return response.json()
  }

  async listSamples(profileId: string): Promise<TrainingSample[]> {
    return this.request(`/profiles/${profileId}/samples`)
  }

  async getSample(profileId: string, sampleId: string): Promise<TrainingSample> {
    return this.request(`/profiles/${profileId}/samples/${sampleId}`)
  }

  async deleteSample(profileId: string, sampleId: string): Promise<void> {
    await this.request(`/profiles/${profileId}/samples/${sampleId}`, { method: 'DELETE' })
  }

  // Song Upload with Auto-Split (Demucs vocal separation)
  async uploadSongWithSplit(
    profileId: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ job_id: string; song_id?: string }> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('profile_id', profileId)
    formData.append('auto_split', 'true')

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = Math.round((event.loaded / event.total) * 100)
          onProgress(progress)
        }
      })

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText)
            resolve(response)
          } catch {
            reject(new Error('Invalid response format'))
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText)
            reject(new Error(error.error || `Upload failed: ${xhr.status}`))
          } catch {
            reject(new Error(`Upload failed: ${xhr.status}`))
          }
        }
      })

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'))
      })

      xhr.addEventListener('abort', () => {
        reject(new Error('Upload aborted'))
      })

      xhr.open('POST', `${API_BASE}/profiles/${profileId}/songs`)
      xhr.send(formData)
    })
  }

  async getSeparationStatus(jobId: string): Promise<{
    status: 'pending' | 'processing' | 'complete' | 'error'
    progress: number
    message?: string
    error?: string
    vocals_path?: string
    instrumental_path?: string
  }> {
    return this.request(`/separation/status/${jobId}`)
  }

  // User Presets
  async listPresets(): Promise<UserPreset[]> {
    return this.request('/presets')
  }

  async getPreset(presetId: string): Promise<UserPreset> {
    return this.request(`/presets/${presetId}`)
  }

  async savePreset(name: string, config: Partial<ConversionConfig>): Promise<UserPreset> {
    return this.request('/presets', {
      method: 'POST',
      body: JSON.stringify({ name, config }),
    })
  }

  async updatePreset(presetId: string, updates: { name?: string; config?: Partial<ConversionConfig> }): Promise<UserPreset> {
    return this.request(`/presets/${presetId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    })
  }

  async deletePreset(presetId: string): Promise<void> {
    await this.request(`/presets/${presetId}`, { method: 'DELETE' })
  }

  // Model Management
  async getLoadedModels(): Promise<LoadedModel[]> {
    const response = await this.request<{ models: LoadedModel[] }>('/models/loaded')
    return response.models
  }

  async loadModel(modelType: string, path?: string): Promise<LoadedModel> {
    return this.request('/models/load', {
      method: 'POST',
      body: JSON.stringify({ model_type: modelType, path }),
    })
  }

  async unloadModel(modelType: string): Promise<void> {
    await this.request('/models/unload', {
      method: 'POST',
      body: JSON.stringify({ model_type: modelType }),
    })
  }

  async rebuildTensorRT(precision: 'fp32' | 'fp16' | 'int8' = 'fp16'): Promise<{ status: string; duration_seconds: number }> {
    return this.request('/models/tensorrt/rebuild', {
      method: 'POST',
      body: JSON.stringify({ precision }),
    })
  }

  async getTensorRTStatus(): Promise<TensorRTStatus> {
    return this.request('/models/tensorrt/status')
  }

  async buildTensorRTEngines(
    models: string[],
    precision: 'fp32' | 'fp16' | 'int8'
  ): Promise<{ status: string; engines_built: string[] }> {
    return this.request('/models/tensorrt/build', {
      method: 'POST',
      body: JSON.stringify({ models, precision }),
    })
  }

  // Separation Configuration
  async getSeparationConfig(): Promise<SeparationConfig> {
    return this.request('/config/separation')
  }

  async updateSeparationConfig(config: Partial<SeparationConfig>): Promise<SeparationConfig> {
    return this.request('/config/separation', {
      method: 'PATCH',
      body: JSON.stringify(config),
    })
  }

  // Pitch Extraction Configuration
  async getPitchConfig(): Promise<PitchConfig> {
    return this.request('/config/pitch')
  }

  async updatePitchConfig(config: Partial<PitchConfig>): Promise<PitchConfig> {
    return this.request('/config/pitch', {
      method: 'PATCH',
      body: JSON.stringify(config),
    })
  }

  // Audio Router Configuration (for karaoke dual-channel)
  async getAudioRouterConfig(): Promise<AudioRouterConfig> {
    return this.request('/audio/router/config')
  }

  async updateAudioRouterConfig(config: Partial<AudioRouterConfig>): Promise<AudioRouterConfig> {
    return this.request('/audio/router/config', {
      method: 'PATCH',
      body: JSON.stringify(config),
    })
  }

  // Conversion History
  async getConversionHistory(profileId?: string): Promise<ConversionRecord[]> {
    const params = profileId ? `?profile_id=${profileId}` : ''
    return this.request(`/convert/history${params}`)
  }

  async deleteConversionRecord(id: string): Promise<void> {
    await this.request(`/convert/history/${id}`, { method: 'DELETE' })
  }

  async updateConversionRecord(id: string, updates: Partial<ConversionRecord>): Promise<ConversionRecord> {
    return this.request(`/convert/history/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    })
  }

  // Checkpoints
  async getCheckpoints(profileId: string): Promise<Checkpoint[]> {
    return this.request(`/profiles/${profileId}/checkpoints`)
  }

  async rollbackToCheckpoint(profileId: string, checkpointId: string): Promise<void> {
    await this.request(`/profiles/${profileId}/checkpoints/${checkpointId}/rollback`, {
      method: 'POST',
    })
  }

  async deleteCheckpoint(profileId: string, checkpointId: string): Promise<void> {
    await this.request(`/profiles/${profileId}/checkpoints/${checkpointId}`, {
      method: 'DELETE',
    })
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Diarization Methods
  // ─────────────────────────────────────────────────────────────────────────

  async diarizeAudio(file: File, numSpeakers?: number): Promise<DiarizationResult> {
    const formData = new FormData()
    formData.append('file', file)
    if (numSpeakers) {
      formData.append('num_speakers', numSpeakers.toString())
    }

    const response = await fetch(`${API_BASE}/audio/diarize`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Diarization failed')
    }

    return response.json()
  }

  async diarizeAudioPath(audioPath: string, numSpeakers?: number): Promise<DiarizationResult> {
    return this.request<DiarizationResult>('/audio/diarize', {
      method: 'POST',
      body: JSON.stringify({ audio_path: audioPath, num_speakers: numSpeakers }),
    })
  }

  async assignDiarizationSegment(
    diarizationId: string,
    segmentIndex: number,
    profileId: string,
    extractAudio: boolean = true
  ): Promise<AssignSegmentResult> {
    return this.request<AssignSegmentResult>('/audio/diarize/assign', {
      method: 'POST',
      body: JSON.stringify({
        diarization_id: diarizationId,
        segment_index: segmentIndex,
        profile_id: profileId,
        extract_audio: extractAudio,
      }),
    })
  }

  async getProfileSegments(profileId: string): Promise<ProfileSegmentsResult> {
    return this.request<ProfileSegmentsResult>(`/profiles/${profileId}/segments`)
  }

  async autoCreateProfileFromDiarization(
    diarizationId: string,
    speakerId: string,
    name: string,
    userId?: string,
    extractSegments: boolean = true,
    options?: {
      profileRole?: ProfileRole
      metadata?: Record<string, unknown>
    }
  ): Promise<AutoCreateProfileResult> {
    return this.request<AutoCreateProfileResult>('/profiles/auto-create', {
      method: 'POST',
      body: JSON.stringify({
        diarization_id: diarizationId,
        speaker_id: speakerId,
        name,
        user_id: userId,
        extract_segments: extractSegments,
        profile_role: options?.profileRole,
        metadata: options?.metadata,
      }),
    })
  }

  async setSpeakerEmbedding(profileId: string, audioPath?: string, useSamples?: boolean): Promise<SpeakerEmbeddingResult> {
    return this.request<SpeakerEmbeddingResult>(`/profiles/${profileId}/speaker-embedding`, {
      method: 'POST',
      body: JSON.stringify({
        audio_path: audioPath,
        use_samples: useSamples,
      }),
    })
  }

  async getSpeakerEmbedding(profileId: string): Promise<SpeakerEmbeddingStatus> {
    return this.request<SpeakerEmbeddingStatus>(`/profiles/${profileId}/speaker-embedding`)
  }

  async filterSample(
    profileId: string,
    sampleId: string,
    similarityThreshold: number = 0.7
  ): Promise<FilterSampleResult> {
    return this.request<FilterSampleResult>(`/profiles/${profileId}/samples/${sampleId}/filter`, {
      method: 'POST',
      body: JSON.stringify({ similarity_threshold: similarityThreshold }),
    })
  }

  // ─────────────────────────────────────────────────────────────────────────
  // YouTube Methods
  // ─────────────────────────────────────────────────────────────────────────

  async getYouTubeVideoInfo(url: string): Promise<YouTubeVideoInfo> {
    return this.request<YouTubeVideoInfo>('/youtube/info', {
      method: 'POST',
      body: JSON.stringify({ url }),
    })
  }

  async downloadYouTubeAudio(
    url: string,
    options?: {
      format?: 'wav' | 'mp3' | 'flac'
      sample_rate?: number
      run_diarization?: boolean
      filter_to_main_artist?: boolean
    }
  ): Promise<YouTubeDownloadResult> {
    return this.request<YouTubeDownloadResult>('/youtube/download', {
      method: 'POST',
      body: JSON.stringify({
        url,
        format: options?.format ?? 'wav',
        sample_rate: options?.sample_rate ?? 44100,
        run_diarization: options?.run_diarization ?? false,
        filter_to_main_artist: options?.filter_to_main_artist ?? false,
      }),
    })
  }

  async getYouTubeHistory(limit?: number): Promise<YouTubeHistoryItem[]> {
    const params = limit ? `?limit=${limit}` : ''
    return this.request(`/youtube/history${params}`)
  }

  async saveYouTubeHistory(item: Partial<YouTubeHistoryItem>): Promise<YouTubeHistoryItem> {
    return this.request('/youtube/history', {
      method: 'POST',
      body: JSON.stringify(item),
    })
  }

  async deleteYouTubeHistoryItem(id: string): Promise<void> {
    await this.request(`/youtube/history/${id}`, { method: 'DELETE' })
  }

  async clearYouTubeHistory(): Promise<void> {
    await this.request('/youtube/history', { method: 'DELETE' })
  }
}

// Diarization types
export interface DiarizationSegment {
  start: number
  end: number
  speaker_id: string
  confidence: number
  duration: number
}

export interface DiarizationResult {
  diarization_id: string
  audio_duration: number
  num_speakers: number
  segments: DiarizationSegment[]
  speaker_durations: Record<string, number>
}

export interface AssignSegmentResult {
  status: string
  profile_id: string
  segment_index: number
  extracted_path: string | null
  segment: DiarizationSegment
}

export interface ProfileSegmentsResult {
  profile_id: string
  total_segments: number
  total_duration: number
  training_samples: Array<{
    type: string
    sample_id: string
    vocals_path: string
    duration: number
    source_file: string
    created_at: string
  }>
  diarization_assignments: Array<{
    type: string
    segment_key: string
    audio_path: string
  }>
}

export interface AutoCreateProfileResult {
  profile_id: string
  name: string
  speaker_id: string
  profile_role: ProfileRole
  num_segments: number
  total_duration: number
  embedding_dim: number
  metadata?: Record<string, unknown>
  status: string
}

export interface SpeakerEmbeddingResult {
  profile_id: string
  embedding_dim: number
  source: string
  status: string
}

export interface SpeakerEmbeddingStatus {
  profile_id: string
  has_embedding: boolean
  embedding_dim: number | null
}

export interface FilterSampleResult {
  sample_id: string
  original_path: string
  filtered_path: string
  original_duration: number
  filtered_duration: number
  num_segments: number
  purity: number
  status: string
}

// ─────────────────────────────────────────────────────────────────────────
// YouTube Types
// ─────────────────────────────────────────────────────────────────────────

export interface YouTubeVideoInfo {
  success: boolean
  title: string
  duration: number
  main_artist: string | null
  featured_artists: string[]
  is_cover: boolean
  original_artist: string | null
  song_title: string | null
  thumbnail_url: string | null
  video_id: string | null
  error: string | null
}

export interface YouTubeDownloadResult {
  success: boolean
  audio_path: string | null
  title: string
  duration: number
  main_artist: string | null
  featured_artists: string[]
  is_cover: boolean
  original_artist: string | null
  song_title: string | null
  thumbnail_url: string | null
  video_id: string | null
  error: string | null
  diarization_id?: string
  speaker_durations?: Record<string, number>
  diarization_result?: {
    diarization_id?: string
    num_speakers: number
    speaker_durations?: Record<string, number>
    segments: Array<{
      speaker_id: string
      start: number
      end: number
      duration: number
    }>
  }
  diarization_error?: string
  filtered_audio_path?: string | null
  main_speaker_id?: string | null
  filtered_duration?: number
}

export const apiService = new ApiService()

// Alias for convenience
export const api = apiService

// WebSocket Manager for real-time updates
class WebSocketManager {
  private socket: Socket | null = null
  private handlers: Map<WSEventType, Set<WSEventHandler>> = new Map()

  connect(): void {
    if (this.socket?.connected) return

    if (this.socket) {
      this.socket.connect()
      return
    }

    this.socket = io(window.location.origin, {
      path: '/socket.io',
      transports: ['websocket'],
    })

    this.socket.on('connect', () => {
      console.log('WebSocket connected')
    })

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected')
    })

    ;([
      'conversion_progress',
      'conversion_complete',
      'conversion_error',
      'training_progress',
      'training_complete',
      'training_error',
      'gpu_metrics',
      'model_loaded',
      'model_unloaded',
    ] as WSEventType[]).forEach((eventType) => {
      this.socket!.on(eventType, (data: unknown) => {
        this.dispatch({
          type: eventType,
          timestamp: new Date().toISOString(),
          data,
        })
      })
    })
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
  }

  subscribe<T = unknown>(eventType: WSEventType, handler: WSEventHandler<T>): () => void {
    this.connect()
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set())
    }
    this.handlers.get(eventType)!.add(handler as WSEventHandler)

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler as WSEventHandler)
    }
  }

  private dispatch(event: WSEvent): void {
    const handlers = this.handlers.get(event.type)
    if (handlers) {
      handlers.forEach((handler) => handler(event))
    }
  }

  // Convenience methods for common subscriptions
  onConversionProgress(jobId: string, handler: (progress: ConversionProgressEvent) => void): () => void {
    return this.subscribe<ConversionProgressEvent>('conversion_progress', (event) => {
      if (event.data.job_id === jobId) {
        handler(event.data)
      }
    })
  }

  onTrainingProgress(jobId: string, handler: (progress: TrainingProgressEvent) => void): () => void {
    return this.subscribe<TrainingProgressEvent>('training_progress', (event) => {
      if (event.data.job_id === jobId) {
        handler(event.data)
      }
    })
  }

  onGPUMetrics(handler: (metrics: GPUMetrics) => void): () => void {
    return this.subscribe<GPUMetrics>('gpu_metrics', (event) => {
      handler(event.data as GPUMetrics)
    })
  }
}

export const wsManager = new WebSocketManager()
