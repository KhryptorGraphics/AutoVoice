/**
 * WebSocket audio streaming client for real-time voice conversion.
 */
import { io, Socket } from 'socket.io-client';

export interface StreamingStats {
  latencyMs: number;
  chunksProcessed: number;
  isConnected: boolean;
  isStreaming: boolean;
}

export type StreamingEventCallback = (event: string, data: unknown) => void;

export class AudioStreamingClient {
  private socket: Socket | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private isStreaming = false;
  private sessionId: string | null = null;
  private eventCallback: StreamingEventCallback | null = null;

  // Audio configuration
  private readonly sampleRate = 24000;
  private readonly chunkSize = 2400; // 100ms at 24kHz

  // Stats
  private latencyMs = 0;
  private chunksProcessed = 0;

  constructor(private serverUrl: string = '') {
    // Default to current host if not specified
    if (!serverUrl) {
      this.serverUrl = window.location.origin;
    }
  }

  /**
   * Connect to the WebSocket server.
   */
  async connect(): Promise<void> {
    if (this.socket?.connected) {
      return;
    }

    return new Promise((resolve, reject) => {
      this.socket = io(this.serverUrl, {
        path: '/socket.io',
        transports: ['websocket'],
      });

      this.socket.on('connect', () => {
        console.log('WebSocket connected');
        this.emitEvent('connected', null);
        resolve();
      });

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        this.emitEvent('error', { message: 'Connection failed' });
        reject(error);
      });

      this.socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        this.emitEvent('disconnected', null);
      });

      // Handle converted audio from server
      this.socket.on('audio_output', (data: { audio: ArrayBuffer; latency_ms: number }) => {
        this.latencyMs = data.latency_ms;
        this.chunksProcessed++;
        this.playAudio(data.audio);
        this.emitEvent('audio_received', { latencyMs: data.latency_ms });
      });

      // Handle errors
      this.socket.on('error', (error: { message: string }) => {
        console.error('Server error:', error);
        this.emitEvent('error', error);
      });

      // Handle session events
      this.socket.on('session_started', (data: { session_id: string }) => {
        this.sessionId = data.session_id;
        this.emitEvent('session_started', data);
      });

      this.socket.on('session_ended', () => {
        this.sessionId = null;
        this.emitEvent('session_ended', null);
      });
    });
  }

  /**
   * Disconnect from the server.
   */
  disconnect(): void {
    this.stopStreaming();
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  /**
   * Start a karaoke session.
   *
   * @param songId - ID of the uploaded song
   * @param voiceModelId - ID of the voice model to use
   * @param pipelineType - Pipeline type for voice conversion
   * @param options - Optional additional session options
   */
  async startSession(
    songId: string,
    voiceModelId: string,
    pipelineType: 'realtime' | 'quality' | 'quality_seedvc' | 'realtime_meanvc' | 'quality_shortcut' = 'realtime',
    options?: {
      profileId?: string;
      adapterType?: 'hq' | 'nvfp4';
      collectSamples?: boolean;
    }
  ): Promise<{ session_id: string }> {
    if (!this.socket?.connected) {
      throw new Error('Not connected to server');
    }

    return new Promise((resolve, reject) => {
      this.socket!.emit(
        'start_session',
        {
          song_id: songId,
          voice_model_id: voiceModelId,
          pipeline_type: pipelineType,
          profile_id: options?.profileId,
          adapter_type: options?.adapterType,
          collect_samples: options?.collectSamples ?? false,
        },
        (response: { session_id?: string; error?: string }) => {
          if (response.error) {
            reject(new Error(response.error));
          } else {
            this.sessionId = response.session_id!;
            resolve({ session_id: this.sessionId });
          }
        }
      );
    });
  }

  /**
   * End the current session.
   */
  async endSession(): Promise<void> {
    if (!this.socket?.connected || !this.sessionId) {
      return;
    }

    this.stopStreaming();

    return new Promise((resolve) => {
      this.socket!.emit('end_session', { session_id: this.sessionId }, () => {
        this.sessionId = null;
        resolve();
      });
    });
  }

  /**
   * Start capturing and streaming microphone audio.
   */
  async startStreaming(): Promise<void> {
    if (this.isStreaming) {
      return;
    }

    if (!this.socket?.connected) {
      throw new Error('Not connected to server');
    }

    if (!this.sessionId) {
      throw new Error('No active session');
    }

    // Request microphone access
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: this.sampleRate,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    // Create audio context
    this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);

    // Create script processor for capturing audio
    // Note: ScriptProcessorNode is deprecated but AudioWorklet requires more setup
    this.scriptProcessor = this.audioContext.createScriptProcessor(
      this.chunkSize,
      1,
      1
    );

    this.scriptProcessor.onaudioprocess = (event) => {
      if (!this.isStreaming) return;

      const inputData = event.inputBuffer.getChannelData(0);
      const audioBuffer = new Float32Array(inputData);

      // Send audio to server
      this.socket!.emit('audio_input', {
        session_id: this.sessionId,
        audio: audioBuffer.buffer,
        sample_rate: this.sampleRate,
      });

      this.emitEvent('audio_sent', { samples: audioBuffer.length });
    };

    source.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioContext.destination);

    this.isStreaming = true;
    this.emitEvent('streaming_started', null);
  }

  /**
   * Stop streaming audio.
   */
  stopStreaming(): void {
    this.isStreaming = false;

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.emitEvent('streaming_stopped', null);
  }

  /**
   * Play received audio.
   */
  private async playAudio(audioBuffer: ArrayBuffer): Promise<void> {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    }

    try {
      const floatArray = new Float32Array(audioBuffer);
      const buffer = this.audioContext.createBuffer(
        1,
        floatArray.length,
        this.sampleRate
      );
      buffer.getChannelData(0).set(floatArray);

      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext.destination);
      source.start();
    } catch (error) {
      console.error('Error playing audio:', error);
    }
  }

  /**
   * Set event callback.
   */
  onEvent(callback: StreamingEventCallback): void {
    this.eventCallback = callback;
  }

  /**
   * Emit event to callback.
   */
  private emitEvent(event: string, data: unknown): void {
    if (this.eventCallback) {
      this.eventCallback(event, data);
    }
  }

  /**
   * Get current stats.
   */
  getStats(): StreamingStats {
    return {
      latencyMs: this.latencyMs,
      chunksProcessed: this.chunksProcessed,
      isConnected: this.socket?.connected ?? false,
      isStreaming: this.isStreaming,
    };
  }

  /**
   * Check if connected.
   */
  get connected(): boolean {
    return this.socket?.connected ?? false;
  }

  /**
   * Check if streaming.
   */
  get streaming(): boolean {
    return this.isStreaming;
  }
}

// Singleton instance
let clientInstance: AudioStreamingClient | null = null;

export function getAudioStreamingClient(): AudioStreamingClient {
  if (!clientInstance) {
    clientInstance = new AudioStreamingClient();
  }
  return clientInstance;
}
