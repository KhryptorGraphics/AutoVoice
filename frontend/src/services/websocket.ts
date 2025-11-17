import { io, Socket } from 'socket.io-client'

const WS_URL = import.meta.env.VITE_WS_URL || 'http://localhost:5000'

export interface ConversionProgress {
  job_id: string
  progress: number        // 0.0-1.0 range (standardized across streaming and JobManager flows)
  stage: string          // Changed from current_stage to match backend
  timestamp?: number
  estimated_time_remaining?: number
}

export type ProgressCallback = (progress: ConversionProgress) => void
export type ErrorCallback = (error: {
  message: string
  details?: any
  code?: string
  job_id?: string
  conversion_id?: string
  stage?: string
  error?: string
}) => void

/**
 * Stem audio data structure from backend
 *
 * Each stem contains detailed metadata about the separated audio track
 */
export interface StemData {
  audio: string        // base64-encoded WAV audio data
  format: string       // audio format (e.g., "wav")
  sample_rate: number  // sample rate in Hz (e.g., 44100)
  duration: number     // audio duration in seconds
}

/**
 * Stems object with all possible stem types
 *
 * Standard stems from source separation:
 * - vocals: Vocal track
 * - instrumental: Combined instrumental (when vocals are separated but not further split)
 * - drums: Drum track (when fully separated)
 * - bass: Bass track (when fully separated)
 * - other: Other instruments track (when fully separated)
 */
export interface StemsStructure {
  vocals?: StemData
  instrumental?: StemData
  drums?: StemData
  bass?: StemData
  other?: StemData
  [key: string]: StemData | undefined  // Allow dynamic stem types from custom separators
}

/**
 * CompleteCallback for conversion completion events
 *
 * NOTE: WebSocket completion payloads have two shapes:
 * - JobManager flows emit: { job_id, output_url, status, ... }
 * - Streaming flows emit: { job_id, audio, stems?, status, ... }
 */
export type CompleteCallback = (result: {
  job_id: string
  output_url?: string  // JobManager path
  audio?: string       // Streaming path (base64)
  stems?: StemsStructure  // Detailed stem structure with metadata
  status: string
  duration?: number
  sample_rate?: number
  metadata?: any
  f0_contour?: number[]
  f0_times?: number[]
}) => void

class WebSocketService {
  private socket: Socket | null = null
  private progressCallbacks: Map<string, ProgressCallback> = new Map()
  private errorCallbacks: Map<string, ErrorCallback> = new Map()
  private completeCallbacks: Map<string, CompleteCallback> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve()
        return
      }

      this.socket = io(WS_URL, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: this.maxReconnectAttempts,
      })

      this.socket.on('connect', () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
        resolve()
      })

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error)
        this.reconnectAttempts++
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          reject(new Error('Failed to connect to WebSocket server'))
        }
      })

      this.socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason)
      })

      // Listen for conversion progress updates
      this.socket.on('conversion_progress', (data: { job_id: string; progress: number; stage: string; timestamp?: number }) => {
        const callback = this.progressCallbacks.get(data.job_id)
        if (callback) {
          // Backend sends progress and stage directly
          callback({
            job_id: data.job_id,
            progress: data.progress,
            stage: data.stage,
            timestamp: data.timestamp,
            estimated_time_remaining: undefined
          })
        }
      })

      // Listen for conversion completion
      // NOTE: Handles both payload shapes:
      // - JobManager flows: { job_id, output_url, status, ... }
      // - Streaming flows: { job_id, audio, stems?, status, ... }
      this.socket.on('conversion_complete', (data: {
        job_id: string
        output_url?: string  // JobManager path
        audio?: string       // Streaming path (base64)
        stems?: StemsStructure  // Detailed stem structure with metadata
        status: string
        duration?: number
        sample_rate?: number
        metadata?: any
        f0_contour?: number[]
        f0_times?: number[]
      }) => {
        const callback = this.completeCallbacks.get(data.job_id)
        if (callback) {
          callback(data)
        }
        // Clean up callbacks
        this.unsubscribeFromJob(data.job_id)
      })

      // Listen for conversion errors
      this.socket.on('conversion_error', (data: {
        job_id: string;
        conversion_id?: string;
        error: string;
        code?: string;
        stage?: string;
        details?: any
      }) => {
        const callback = this.errorCallbacks.get(data.job_id)
        if (callback) {
          callback(data)
        }
        // Clean up callbacks
        this.unsubscribeFromJob(data.job_id)
      })

      // Listen for conversion cancellation
      this.socket.on('conversion_cancelled', (data: { job_id: string; message?: string }) => {
        const jobId = data.job_id

        // Look up callbacks for this job
        const errorCallback = this.errorCallbacks.get(jobId)

        // Call error callback with cancellation message
        if (errorCallback) {
          errorCallback({
            job_id: jobId,
            conversion_id: jobId,
            error: data.message || 'Conversion cancelled by user',
            message: data.message || 'Conversion cancelled by user',
            code: 'CONVERSION_CANCELLED',
            stage: 'cancelled'
          })
        }

        // Clean up and leave room
        this.unsubscribeFromJob(jobId)
      })

      // Listen for job creation
      this.socket.on('job_created', (data: { job_id: string; status: string }) => {
        console.log('Job created:', data.job_id)
        // Optionally auto-join room if we're expecting this job
      })
    })
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    this.progressCallbacks.clear()
    this.errorCallbacks.clear()
    this.completeCallbacks.clear()
  }

  subscribeToJob(
    jobId: string,
    callbacks: {
      onProgress?: ProgressCallback
      onError?: ErrorCallback
      onComplete?: CompleteCallback
    },
    websocketRoom?: string  // ADDED optional parameter
  ): Promise<void> {
    if (!this.socket?.connected) {
      throw new Error('WebSocket not connected')
    }

    const roomId = websocketRoom || jobId  // Use custom room or fallback to jobId

    if (callbacks.onProgress) {
      this.progressCallbacks.set(jobId, callbacks.onProgress)
    }
    if (callbacks.onError) {
      this.errorCallbacks.set(jobId, callbacks.onError)
    }
    if (callbacks.onComplete) {
      this.completeCallbacks.set(jobId, callbacks.onComplete)
    }

    // Join using roomId (which might differ from jobId)
    this.socket.emit('join_job', { job_id: roomId })

    // Wait for confirmation
    return new Promise<void>((resolve) => {
      this.socket?.once('joined_job', (data: { job_id: string }) => {
        if (data.job_id === roomId) {
          console.log('Joined job room:', roomId)
          resolve()
        }
      })
    })
  }

  unsubscribeFromJob(jobId: string, websocketRoom?: string) {
    const roomId = websocketRoom || jobId

    this.progressCallbacks.delete(jobId)
    this.errorCallbacks.delete(jobId)
    this.completeCallbacks.delete(jobId)

    if (this.socket?.connected) {
      this.socket.emit('leave_job', { job_id: roomId })
    }
  }

  isConnected(): boolean {
    return this.socket?.connected ?? false
  }

  // Send custom events
  emit(event: string, data: any) {
    if (!this.socket?.connected) {
      throw new Error('WebSocket not connected')
    }
    this.socket.emit(event, data)
  }

  // Listen for custom events
  on(event: string, callback: (data: any) => void) {
    if (!this.socket) {
      throw new Error('WebSocket not initialized')
    }
    this.socket.on(event, callback)
  }

  off(event: string, callback?: (data: any) => void) {
    if (!this.socket) {
      return
    }
    if (callback) {
      this.socket.off(event, callback)
    } else {
      this.socket.off(event)
    }
  }
}

export const wsService = new WebSocketService()

