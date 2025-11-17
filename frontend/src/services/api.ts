import axios, { AxiosInstance, AxiosProgressEvent } from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 300000, // 5 minutes for long conversions
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error)
        return Promise.reject(error)
      }
    )
  }

  // Health check
  async healthCheck() {
    const response = await this.client.get('/health')
    return response.data
  }

  // Voice Profiles
  async getVoiceProfiles(userId?: string) {
    const response = await this.client.get('/voice/profiles', {
      params: { user_id: userId },
    })
    return response.data
  }

  async getVoiceProfile(profileId: string) {
    const response = await this.client.get(`/voice/profiles/${profileId}`)
    return response.data
  }

  async createVoiceProfile(formData: FormData, onProgress?: (progress: number) => void) {
    const response = await this.client.post('/voice/clone', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent: AxiosProgressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(percentCompleted)
        }
      },
    })
    return response.data
  }

  async deleteVoiceProfile(profileId: string) {
    const response = await this.client.delete(`/voice/profiles/${profileId}`)
    return response.data
  }

  // Singing Voice Conversion
  /**
   * Supports async (202) and sync (200) based on server config.
   */
  async convertSong(
    audioFile: File,
    targetProfileId: string,
    settings: {
      pitchShift?: number
      preserveOriginalPitch?: boolean
      preserveVibrato?: boolean
      preserveExpression?: boolean
      outputQuality?: string
      denoiseInput?: boolean
      enhanceOutput?: boolean
    },
    onProgress?: (progress: number) => void
  ) {
    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('target_profile_id', targetProfileId)
    
    // Map frontend settings to backend parameters
    const backendSettings = {
      pitch_shift: settings.pitchShift || 0,
      output_quality: settings.outputQuality || 'balanced',
      // Note: preserve_original_pitch is always true for singing (backend default)
      // vocal_volume and instrumental_volume use backend defaults (1.0, 0.9)
    }
    formData.append('settings', JSON.stringify(backendSettings))
    
    const response = await this.client.post('/convert/song', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent: AxiosProgressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(percentCompleted)
        }
      },
    })

    // Handle both sync and async responses
    const data = response.data
    if (response.status === 202 && data.status === 'queued') {
      return {
        job_id: data.job_id,
        status: 'queued',
        websocket_room: data.websocket_room || data.job_id  // ADDED with fallback
      }
    }

    // Sync response (status === 'success')
    return data
  }

  async getConversionStatus(jobId: string) {
    const response = await this.client.get(`/convert/status/${jobId}`)
    return response.data
  }

  async downloadConvertedAudio(jobId: string): Promise<Blob> {
    const response = await this.client.get(`/convert/download/${jobId}`, {
      responseType: 'blob',
    })
    return response.data
  }

  // System Status
  async getSystemStatus() {
    const response = await this.client.get('/gpu_status')
    return response.data
  }

  async getModelsInfo() {
    const response = await this.client.get('/models/info')
    return response.data
  }

  // Audio Analysis
  async analyzeAudio(audioFile: File) {
    const formData = new FormData()
    formData.append('audio', audioFile)

    const response = await this.client.post('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  // Voice Profile Management (Extended)
  async updateVoiceProfile(profileId: string, data: Partial<VoiceProfile>) {
    const response = await this.client.patch(`/voice/profiles/${profileId}`, data)
    return response.data
  }

  async testVoiceProfile(profileId: string, testAudioFile: File) {
    const formData = new FormData()
    formData.append('audio', testAudioFile)
    formData.append('profile_id', profileId)

    const response = await this.client.post('/voice/test', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  // Advanced Conversion with all settings
  async convertSongAdvanced(
    audioFile: File,
    targetProfileId: string,
    settings: AdvancedConversionSettings,
    onProgress?: (progress: number) => void
  ) {
    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('target_profile_id', targetProfileId)

    // Add all advanced settings
    Object.entries(settings).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        formData.append(key, String(value))
      }
    })

    const response = await this.client.post('/convert/song', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent: AxiosProgressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(percentCompleted)
        }
      },
    })

    // Handle both sync and async responses (mirror convertSong behavior)
    const data = response.data
    if (response.status === 202 && data.status === 'queued') {
      return {
        job_id: data.job_id,
        status: 'queued',
        websocket_room: data.websocket_room || data.job_id  // ADDED with fallback
      }
    }

    // Sync response (status === 'success')
    return data
  }

  // Stem Separation
  async separateStems(audioFile: File, returnStems: string[] = ['vocals', 'instrumental']) {
    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('return_stems', 'true')
    formData.append('stems', returnStems.join(','))

    const response = await this.client.post('/separate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  // Quality Metrics
  async getConversionMetrics(jobId: string): Promise<QualityMetricsResponse> {
    const response = await this.client.get<QualityMetricsResponse>(`/convert/metrics/${jobId}`)
    return response.data
  }

  // Configuration
  async getConfig() {
    const response = await this.client.get('/config')
    return response.data
  }

  async updateConfig(config: Partial<AppConfig>) {
    const response = await this.client.post('/config', config)
    return response.data
  }

  // Speakers (for TTS)
  async getSpeakers(language?: string) {
    const response = await this.client.get('/speakers', {
      params: { language },
    })
    return response.data
  }

  // Health checks
  async healthLiveness() {
    const response = await this.client.get('/health/live')
    return response.data
  }

  async healthReadiness() {
    const response = await this.client.get('/health/ready')
    return response.data
  }

  // Cancel conversion
  async cancelConversion(jobId: string) {
    const response = await this.client.post(`/convert/cancel/${jobId}`)
    return response.data
  }

  // Pause conversion (if supported)
  async pauseConversion(jobId: string) {
    const response = await this.client.post(`/convert/pause/${jobId}`)
    return response.data
  }

  // Resume conversion (if supported)
  async resumeConversion(jobId: string) {
    const response = await this.client.post(`/convert/resume/${jobId}`)
    return response.data
  }

  // Export audio with format options
  async exportAudio(
    audioUrl: string,
    format: 'mp3' | 'wav' | 'flac' | 'ogg',
    options?: {
      bitrate?: number // kbps (e.g., 128, 192, 256, 320)
      sampleRate?: number // Hz (e.g., 44100, 48000)
      channels?: 1 | 2 // mono or stereo
    }
  ): Promise<Blob> {
    const response = await this.client.post(
      '/audio/export',
      {
        audio_url: audioUrl,
        format,
        ...options,
      },
      { responseType: 'blob' }
    )
    return response.data
  }
}

export const apiService = new ApiService()

// Type definitions
export interface VoiceProfile {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
  sample_duration?: number
  user_id?: string
  vocal_range?: {
    min_note: string
    max_note: string
    min_freq: number
    max_freq: number
  }
  characteristics?: {
    timbre: string
    gender: string
    age_range: string
  }
  embedding_quality?: number
}

export interface ConversionJob {
  job_id: string
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: number
  stage: string
  created_at: number
  completed_at?: number
  error?: string
  f0_contour?: number[]
  f0_times?: number[]
}

export interface QualityMetrics {
  pitch_accuracy?: {
    rmse_hz: number
    correlation: number
    mean_error_cents: number
  }
  speaker_similarity?: {
    cosine_similarity: number
    embedding_distance: number
  }
  naturalness?: {
    spectral_distortion: number
    mos_estimate: number
  }
  intelligibility?: {
    stoi: number
    pesq: number
  }
}

export interface QualityMetricsResponse {
  metrics: QualityMetrics
  job_id: string
  calculated_at: number
  targets?: {
    min_pitch_accuracy_correlation?: number
    max_pitch_accuracy_rmse_hz?: number
    min_speaker_similarity?: number
    max_spectral_distortion?: number
    min_stoi_score?: number
    min_pesq_score?: number
    min_mos_estimate?: number
  }
}

export interface SystemStatus {
  gpu_available: boolean
  gpu_name?: string
  gpu_memory_used?: number
  gpu_memory_total?: number
  gpu_utilization?: number
  gpu_temperature?: number
  model_loaded: boolean
  status: string
  models?: {
    name: string
    loaded: boolean
    memory_usage?: number
  }[]
}

export interface AdvancedConversionSettings {
  pitch_shift?: number
  preserve_original_pitch?: boolean
  preserve_vibrato?: boolean
  preserve_expression?: boolean
  output_quality?: 'draft' | 'fast' | 'balanced' | 'high' | 'studio'
  denoise_input?: boolean
  enhance_output?: boolean
  vocal_volume?: number
  instrumental_volume?: number
  return_stems?: boolean
  formant_shift?: number
  temperature?: number
  pitch_range_min?: number
  pitch_range_max?: number
}

export interface AppConfig {
  audio?: {
    sample_rate: number
    channels: number
    formats: string[]
  }
  conversion?: {
    default_quality: string
    max_duration: number
  }
  gpu?: {
    enabled: boolean
    device_id: number
  }
}

// Pitch data interface
export interface PitchData {
  f0: number[]
  times: number[]
}

export interface ConversionProgress {
  job_id: string
  progress: number  // 0.0-1.0 range (standardized across streaming and JobManager flows)
  stage: string     // Changed from current_stage
  timestamp?: number
  // Remove stages array - backend doesn't send this
  estimated_time_remaining?: number
  // Optional pitch contour data
  f0_contour?: number[]
  f0_times?: number[]
}

/**
 * CompleteCallback for conversion completion events
 *
 * NOTE: WebSocket completion payloads have two shapes:
 * - JobManager flows emit: { job_id, output_url, status, ... }
 * - Streaming flows emit: { job_id, audio, stems?, status, ... }
 *
 * The frontend handles both by checking which fields are present.
 */
export type CompleteCallback = (result: {
  job_id: string
  // JobManager path: URL to download result
  output_url?: string
  // Streaming path: base64 audio data
  audio?: string
  stems?: {
    vocals?: string
    instrumental?: string
    bass?: string
    drums?: string
    other?: string
  }
  status: string
  duration?: number
  sample_rate?: number
  metadata?: any
  // Optional pitch contour data
  f0_contour?: number[]
  f0_times?: number[]
}) => void

// Export format options
export interface ExportOptions {
  format: 'mp3' | 'wav' | 'flac' | 'ogg'
  bitrate?: number // kbps (128, 192, 256, 320)
  sampleRate?: number // Hz (44100, 48000, 96000)
  channels?: 1 | 2 // mono or stereo
}

// Conversion history record
export interface ConversionRecord {
  id: string
  originalFileName: string
  targetVoice: string
  targetVoiceId: string
  timestamp: Date
  duration: number
  quality: string
  resultUrl?: string
  isFavorite?: boolean
  tags?: string[]
  notes?: string
}
