/**
 * WebSocket audio streaming client for real-time voice conversion.
 */
import { io, Socket } from 'socket.io-client';
import type { ActiveModelType, LivePipelineType } from './api';

export interface StreamingStats {
  latencyMs: number;
  chunksProcessed: number;
  isConnected: boolean;
  isStreaming: boolean;
  sessionId?: string | null;
  requestedPipeline?: LivePipelineType;
  resolvedPipeline?: LivePipelineType;
  runtimeBackend?: string | null;
  targetProfileId?: string | null;
  sourceVoiceModelId?: string | null;
  activeModelType?: ActiveModelType | string | null;
  sampleCollectionEnabled?: boolean;
  audioRouterTargets?: {
    speaker_device: number | null;
    headphone_device: number | null;
  } | null;
}

export type StreamingEventCallback = (event: string, data: unknown) => void;

type SessionStartedPayload = {
  session_id: string;
  requested_pipeline?: LivePipelineType;
  resolved_pipeline?: LivePipelineType;
  runtime_backend?: string | null;
  target_profile_id?: string | null;
  source_voice_model_id?: string | null;
  active_model_type?: ActiveModelType | string | null;
  sample_collection_enabled?: boolean;
  audio_router_targets?: {
    speaker_device: number | null;
    headphone_device: number | null;
  };
};

type TestStreamingHook = {
  connect?: () => Promise<void> | void;
  disconnect?: () => void;
  startSession?: (payload: {
    songId: string;
    voiceModelId: string;
    pipelineType: LivePipelineType;
    options?: {
      profileId?: string;
      adapterType?: 'hq' | 'nvfp4';
      collectSamples?: boolean;
      vocalsPath?: string;
      instrumentalPath?: string;
    };
  }) => Promise<SessionStartedPayload> | SessionStartedPayload;
  endSession?: () => Promise<void> | void;
  startStreaming?: () => Promise<void> | void;
  stopStreaming?: () => void;
};

function getTestStreamingHook(): TestStreamingHook | null {
  const maybeHook = (globalThis as typeof globalThis & {
    __AUTOVOICE_TEST_STREAMING__?: TestStreamingHook;
  }).__AUTOVOICE_TEST_STREAMING__;
  return maybeHook ?? null;
}

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

  emitTestEvent(event: string, data: unknown): void {
    this.emitEvent(event, data);
  }

  /**
   * Connect to the WebSocket server.
   */
  async connect(): Promise<void> {
    const testHook = getTestStreamingHook();
    if (testHook) {
      await testHook.connect?.();
      this.emitEvent('connected', null);
      return;
    }

    if (this.socket?.connected) {
      return;
    }

    return new Promise((resolve, reject) => {
      this.socket = io(`${this.serverUrl}/karaoke`, {
        path: '/socket.io',
      });

      this.socket.on('connect', () => {
        console.log('WebSocket connected');
        this.emitEvent('connected', null);
        resolve();
      });

      this.socket.on('connected', () => {
        this.emitEvent('connected', null);
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
      this.socket.on('converted_audio', (data: { audio: string; latency_ms: number }) => {
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
      this.socket.on('session_started', (data: SessionStartedPayload) => {
        this.sessionId = data.session_id;
        this.emitEvent('session_started', data);
      });

      this.socket.on('session_stopped', () => {
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
    const testHook = getTestStreamingHook();
    if (testHook) {
      testHook.disconnect?.();
      this.emitEvent('disconnected', null);
      return;
    }

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
    pipelineType: LivePipelineType = 'realtime',
    options?: {
      profileId?: string;
      adapterType?: 'hq' | 'nvfp4';
      collectSamples?: boolean;
      vocalsPath?: string;
      instrumentalPath?: string;
    }
  ): Promise<SessionStartedPayload> {
    const testHook = getTestStreamingHook();
    if (testHook) {
      const result = await testHook.startSession?.({
        songId,
        voiceModelId,
        pipelineType,
        options,
      }) ?? {
        session_id:
          globalThis.crypto?.randomUUID?.() ??
          `karaoke-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      };
      this.sessionId = result.session_id;
      this.emitEvent('session_started', result);
      return result;
    }

    if (!this.socket?.connected) {
      throw new Error('Not connected to server');
    }

    const sessionId = globalThis.crypto?.randomUUID?.() ?? `karaoke-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    return new Promise((resolve, reject) => {
      const handleStarted = (data: SessionStartedPayload) => {
        if (data.session_id !== sessionId) return;
        cleanup();
        this.sessionId = data.session_id;
        resolve(data);
      };

      const handleError = (error: { message?: string }) => {
        cleanup();
        reject(new Error(error.message || 'Failed to start session'));
      };

      const timeout = window.setTimeout(() => {
        cleanup();
        reject(new Error('Timed out waiting for karaoke session to start'));
      }, 10000);

      const cleanup = () => {
        window.clearTimeout(timeout);
        this.socket?.off('session_started', handleStarted);
        this.socket?.off('error', handleError);
      };

      this.socket!.on('session_started', handleStarted);
      this.socket!.on('error', handleError);
      this.socket!.emit('start_session', {
        session_id: sessionId,
        song_id: songId,
        voice_model_id: voiceModelId,
        pipeline_type: pipelineType,
        profile_id: options?.profileId,
        adapter_type: options?.adapterType,
        collect_samples: options?.collectSamples ?? false,
        vocals_path: options?.vocalsPath,
        instrumental_path: options?.instrumentalPath,
      });
    });
  }

  /**
   * End the current session.
   */
  async endSession(): Promise<void> {
    const testHook = getTestStreamingHook();
    if (testHook) {
      if (!this.sessionId) {
        return;
      }
      this.stopStreaming();
      await testHook.endSession?.();
      this.sessionId = null;
      this.emitEvent('session_ended', null);
      return;
    }

    if (!this.socket?.connected || !this.sessionId) {
      return;
    }

    this.stopStreaming();
    const sessionId = this.sessionId;
    this.socket!.emit('stop_session', { session_id: sessionId });
    this.sessionId = null;
  }

  /**
   * Start capturing and streaming microphone audio.
   */
  async startStreaming(): Promise<void> {
    const testHook = getTestStreamingHook();
    if (testHook) {
      if (this.isStreaming) {
        return;
      }
      if (!this.sessionId) {
        throw new Error('No active session');
      }
      await testHook.startStreaming?.();
      this.isStreaming = true;
      this.emitEvent('streaming_started', null);
      this.emitEvent('audio_sent', { samples: this.chunkSize });
      this.emitEvent('audio_received', { latencyMs: 48 });
      return;
    }

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
      this.socket!.emit('audio_chunk', {
        session_id: this.sessionId,
        audio: this.encodeFloat32ToBase64(audioBuffer),
        timestamp: Date.now(),
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
    const testHook = getTestStreamingHook();
    if (testHook) {
      if (!this.isStreaming) {
        return;
      }
      this.isStreaming = false;
      testHook.stopStreaming?.();
      this.emitEvent('streaming_stopped', null);
      return;
    }

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
  private async playAudio(audioBase64: string): Promise<void> {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    }

    try {
      const floatArray = this.decodeBase64ToFloat32(audioBase64);
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

  private encodeFloat32ToBase64(audioBuffer: Float32Array): string {
    const bytes = new Uint8Array(audioBuffer.buffer.slice(0));
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      const chunk = bytes.subarray(i, i + 0x8000);
      binary += String.fromCharCode(...chunk);
    }
    return btoa(binary);
  }

  private decodeBase64ToFloat32(audioBase64: string): Float32Array {
    const binary = atob(audioBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
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
