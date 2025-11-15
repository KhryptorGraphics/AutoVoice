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
  }
  error?: string
}

export interface SystemStatus {
  gpu_available: boolean
  gpu_name?: string
  gpu_memory_used?: number
  gpu_memory_total?: number
  gpu_utilization?: number
  model_loaded: boolean
  status: string
}

