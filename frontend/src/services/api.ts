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
    formData.append('settings', JSON.stringify(settings))

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
    return response.data
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
    return response.data
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
  async getConversionMetrics(jobId: string) {
    const response = await this.client.get(`/convert/metrics/${jobId}`)
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
  status: 'pending' | 'processing' | 'complete' | 'error'
  progress: number
  stages: {
    id: string
    name: string
    progress: number
    status: string
    message?: string
  }[]
  result?: {
    output_path: string
    duration: number
    sample_rate: number
    quality_metrics?: QualityMetrics
  }
  error?: string
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

