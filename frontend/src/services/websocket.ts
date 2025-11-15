import { io, Socket } from 'socket.io-client'

const WS_URL = import.meta.env.VITE_WS_URL || 'http://localhost:5000'

export interface ConversionProgress {
  job_id: string
  overall_progress: number
  current_stage: string
  stages: Array<{
    id: string
    name: string
    progress: number
    status: 'pending' | 'processing' | 'complete' | 'error'
    message?: string
    duration?: number
  }>
  estimated_time_remaining?: number
}

export type ProgressCallback = (progress: ConversionProgress) => void
export type ErrorCallback = (error: { message: string; details?: any }) => void
export type CompleteCallback = (result: { job_id: string; output_url: string }) => void

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
      this.socket.on('conversion_progress', (data: ConversionProgress) => {
        const callback = this.progressCallbacks.get(data.job_id)
        if (callback) {
          callback(data)
        }
      })

      // Listen for conversion completion
      this.socket.on('conversion_complete', (data: { job_id: string; output_url: string }) => {
        const callback = this.completeCallbacks.get(data.job_id)
        if (callback) {
          callback(data)
        }
        // Clean up callbacks
        this.unsubscribeFromJob(data.job_id)
      })

      // Listen for conversion errors
      this.socket.on('conversion_error', (data: { job_id: string; error: string; details?: any }) => {
        const callback = this.errorCallbacks.get(data.job_id)
        if (callback) {
          callback({ message: data.error, details: data.details })
        }
        // Clean up callbacks
        this.unsubscribeFromJob(data.job_id)
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
    }
  ) {
    if (!this.socket?.connected) {
      throw new Error('WebSocket not connected')
    }

    if (callbacks.onProgress) {
      this.progressCallbacks.set(jobId, callbacks.onProgress)
    }
    if (callbacks.onError) {
      this.errorCallbacks.set(jobId, callbacks.onError)
    }
    if (callbacks.onComplete) {
      this.completeCallbacks.set(jobId, callbacks.onComplete)
    }

    // Join the job room
    this.socket.emit('join_job', { job_id: jobId })
  }

  unsubscribeFromJob(jobId: string) {
    this.progressCallbacks.delete(jobId)
    this.errorCallbacks.delete(jobId)
    this.completeCallbacks.delete(jobId)

    if (this.socket?.connected) {
      this.socket.emit('leave_job', { job_id: jobId })
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

