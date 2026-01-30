const API_BASE = '/api/v1'
const WS_BASE = `ws://${window.location.host}`

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
  status: 'queued' | 'processing' | 'complete' | 'error' | 'cancelled'
  created_at: string
  completed_at?: string
  input_file: string
  profile_id: string
  preset: string
  duration?: number
  error?: string
  // Additional fields used by ConversionHistoryPage
  timestamp?: Date
  isFavorite?: boolean
  targetVoice?: string
  quality?: string
  originalFileName?: string
  notes?: string
  resultUrl?: string
}

export interface VoiceProfile {
  profile_id: string
  user_id?: string
  name?: string
  created_at: string
  sample_count: number
  model_version?: string
  last_trained?: string
  quality_score?: number
  training_status?: TrainingStatusType
  has_trained_model?: boolean
}

// Training status for voice profiles
export type TrainingStatusType = 'pending' | 'training' | 'ready' | 'failed'

export interface TrainingStatus {
  profile_id: string
  has_trained_model: boolean
  training_status: TrainingStatusType
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
  }
}

// Full training configuration with all LoRA/EWC parameters
export interface TrainingConfig {
  // LoRA configuration
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
  lora_rank: 8,
  lora_alpha: 16,
  lora_dropout: 0.1,
  lora_target_modules: ['q_proj', 'v_proj', 'content_encoder'],
  learning_rate: 1e-4,
  batch_size: 4,
  epochs: 10,
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

class ApiService {
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: response.statusText }))
      throw new Error(error.error || `HTTP ${response.status}`)
    }
    return response.json()
  }

  async getHealth(): Promise<HealthStatus> {
    const response = await fetch('/health')
    return response.json()
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

  async createVoiceProfile(audioFile: File, userId?: string): Promise<VoiceProfile> {
    const formData = new FormData()
    formData.append('audio', audioFile)
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
    }
  ): Promise<{ job_id: string }> {
    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('profile_id', profileId)
    if (settings?.preset) formData.append('preset', settings.preset)
    if (settings?.vocal_volume != null) formData.append('vocal_volume', String(settings.vocal_volume))
    if (settings?.instrumental_volume != null) formData.append('instrumental_volume', String(settings.instrumental_volume))
    if (settings?.pitch_shift != null) formData.append('pitch_shift', String(settings.pitch_shift))

    const response = await fetch(`${API_BASE}/convert/song`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error(`Conversion failed: ${response.status}`)
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
}

export const apiService = new ApiService()

// WebSocket Manager for real-time updates
class WebSocketManager {
  private socket: WebSocket | null = null
  private handlers: Map<WSEventType, Set<WSEventHandler>> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  connect(): void {
    if (this.socket?.readyState === WebSocket.OPEN) return

    this.socket = new WebSocket(`${WS_BASE}/ws`)

    this.socket.onopen = () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
    }

    this.socket.onmessage = (event) => {
      try {
        const wsEvent = JSON.parse(event.data) as WSEvent
        this.dispatch(wsEvent)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    this.socket.onclose = () => {
      console.log('WebSocket disconnected')
      this.attemptReconnect()
    }

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)
    setTimeout(() => this.connect(), delay)
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.close()
      this.socket = null
    }
  }

  subscribe<T = unknown>(eventType: WSEventType, handler: WSEventHandler<T>): () => void {
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
