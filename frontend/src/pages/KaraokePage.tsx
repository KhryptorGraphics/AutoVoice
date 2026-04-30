/**
 * Live Karaoke Voice Conversion page.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Upload,
  Mic,
  MicOff,
  Play,
  Square,
  Volume2,
  Headphones,
  Music,
  User,
  Activity,
  RefreshCw,
} from 'lucide-react';
import clsx from 'clsx';
import {
  uploadSong,
  startSeparation,
  getSeparationStatus,
  listDevices,
  setDeviceConfig,
  listVoiceModels,
  extractVoiceModel,
  type UploadedSong,
  type SeparationJob,
  type AudioDevice,
  type VoiceModel,
} from '../services/karaokeApi';
import { getAudioStreamingClient, type StreamingStats } from '../services/audioStreaming';
import { apiService, VoiceProfile, type LivePipelineType } from '../services/api';
import { AudioDeviceSelector } from '../components/AudioDeviceSelector';
import {
  PipelineSelector,
  type PipelineType,
  getPreferredPipeline,
  isLivePipeline,
} from '../components/PipelineSelector';
import { AdapterDropdown } from '../components/AdapterSelector';
import { AdapterType } from '../services/api';
import { KaraokeSessionInfo } from '../components/KaraokeSessionInfo';
import { StatusBanner } from '../components/StatusBanner';
import { BrowserSingAlongCapture } from '../components/BrowserSingAlongCapture';

type Stage = 'upload' | 'separating' | 'ready' | 'performing';

export function KaraokePage() {
  // Stage management
  const [stage, setStage] = useState<Stage>('upload');

  // Pipeline selection (default to realtime for karaoke, but respect user preference)
  const [pipeline, setPipeline] = useState<LivePipelineType>(() => {
    const preferred = getPreferredPipeline();
    if (preferred && isLivePipeline(preferred)) {
      return preferred;
    }
    return 'realtime';
  });
  const [pipelineStatus, setPipelineStatus] = useState<Record<string, { loaded: boolean; memory_gb?: number; latency_target_ms?: number }>>({});

  // Song state
  const [uploadedSong, setUploadedSong] = useState<UploadedSong | null>(null);
  const [separationJob, setSeparationJob] = useState<SeparationJob | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [preflightIssues, setPreflightIssues] = useState<string[]>([]);
  const [preflightWarnings, setPreflightWarnings] = useState<string[]>([]);

  // Device state
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [speakerDevice, setSpeakerDevice] = useState<number | null>(null);
  const [headphoneDevice, setHeadphoneDevice] = useState<number | null>(null);

  // Voice model state
  const [voiceModels, setVoiceModels] = useState<VoiceModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);

  // Voice profile state (for training sample collection)
  const [voiceProfiles, setVoiceProfiles] = useState<VoiceProfile[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);
  const [collectTrainingSamples, setCollectTrainingSamples] = useState(false);
  const [selectedAdapter, setSelectedAdapter] = useState<AdapterType | null>(null);
  const isLiveReadyProfile = (profile: VoiceProfile) =>
    profile.readiness?.live_conversion?.ready ?? profile.has_trained_model;
  const trainedTargetProfiles = voiceProfiles.filter(
    (profile) => profile.profile_role !== 'source_artist' && isLiveReadyProfile(profile)
  );
  const outputDevices = devices.filter((device) => device.type !== 'input');
  const selectedProfile = selectedProfileId
    ? voiceProfiles.find((profile) => profile.profile_id === selectedProfileId) ?? null
    : null;

  // Streaming state
  const [streamingStats, setStreamingStats] = useState<StreamingStats>({
    latencyMs: 0,
    chunksProcessed: 0,
    isConnected: false,
    isStreaming: false,
  });

  // Audio levels
  const [inputLevel, setInputLevel] = useState(0);
  const [outputLevel, setOutputLevel] = useState(0);

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<number | null>(null);
  const stageRef = useRef<Stage>('upload');
  const recoveringRef = useRef(false);
  const intentionalStopRef = useRef(false);

  useEffect(() => {
    stageRef.current = stage;
  }, [stage]);

  // Load devices, models, and profiles on mount
  useEffect(() => {
    loadDevices();
    loadVoiceModels();
    loadVoiceProfiles();
    // These startup loaders intentionally run once to hydrate page state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadPipelinePreferences = async () => {
      try {
        const [settings, status] = await Promise.all([
          apiService.getAppSettings(),
          apiService.getPipelineStatus(),
        ]);
        if (cancelled) {
          return;
        }
        if (isLivePipeline(settings.preferred_live_pipeline)) {
          setPipeline(settings.preferred_live_pipeline);
        }
        setPipelineStatus(status.pipelines || {});
      } catch (error) {
        console.error('Failed to load pipeline preferences:', error);
      }
    };

    void loadPipelinePreferences();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (selectedProfile?.active_model_type === 'full_model' && selectedAdapter) {
      setSelectedAdapter(null);
    }
  }, [selectedAdapter, selectedProfile?.active_model_type]);

  useEffect(() => {
    if (
      selectedProfileId &&
      !trainedTargetProfiles.some((profile) => profile.profile_id === selectedProfileId)
    ) {
      setSelectedProfileId(trainedTargetProfiles[0]?.profile_id ?? null);
    }
  }, [selectedProfileId, trainedTargetProfiles]);

  const applySessionStarted = useCallback((data: Partial<StreamingStats> & Record<string, unknown>) => {
    setStreamingStats((stats) => ({
      ...stats,
      sessionId: typeof data.session_id === 'string' ? data.session_id : stats.sessionId,
      requestedPipeline: (data.requested_pipeline as LivePipelineType | undefined) ?? stats.requestedPipeline,
      resolvedPipeline: (data.resolved_pipeline as LivePipelineType | undefined) ?? stats.resolvedPipeline,
      runtimeBackend: typeof data.runtime_backend === 'string' ? data.runtime_backend : stats.runtimeBackend,
      targetProfileId: typeof data.target_profile_id === 'string' ? data.target_profile_id : stats.targetProfileId,
      sourceVoiceModelId: typeof data.source_voice_model_id === 'string' ? data.source_voice_model_id : stats.sourceVoiceModelId,
      activeModelType: (data.active_model_type as string | undefined) ?? stats.activeModelType,
      sampleCollectionEnabled:
        typeof data.sample_collection_enabled === 'boolean'
          ? data.sample_collection_enabled
          : stats.sampleCollectionEnabled,
      audioRouterTargets:
        (data.audio_router_targets as StreamingStats['audioRouterTargets']) ?? stats.audioRouterTargets,
    }));
  }, []);

  const recoverPerformance = useCallback(async () => {
    if (!uploadedSong || !selectedModel) {
      setStage('ready');
      setSessionError('Live session disconnected and could not be recovered.');
      recoveringRef.current = false;
      return;
    }

    try {
      const client = getAudioStreamingClient();
      await client.connect();
      const started = await client.startSession(uploadedSong.song_id, selectedModel, pipeline, {
        profileId: selectedProfileId || undefined,
        adapterType: selectedAdapter || undefined,
        collectSamples: collectTrainingSamples,
        vocalsPath: separationJob?.vocals_path,
        instrumentalPath: separationJob?.instrumental_path,
      });
      applySessionStarted(started as Record<string, unknown>);
      await client.startStreaming();
      setSessionError(null);
      setStage('performing');
    } catch (error) {
      console.error('Failed to recover performance:', error);
      setSessionError('Live session disconnected and could not be recovered.');
      setStage('ready');
    } finally {
      recoveringRef.current = false;
    }
  }, [
    applySessionStarted,
    collectTrainingSamples,
    pipeline,
    selectedAdapter,
    selectedModel,
    selectedProfileId,
    separationJob?.instrumental_path,
    separationJob?.vocals_path,
    uploadedSong,
  ]);

  // Setup streaming client events
  useEffect(() => {
    const client = getAudioStreamingClient();

    client.onEvent((event, data) => {
      if (event === 'audio_received') {
        const d = data as { latencyMs: number };
        setStreamingStats((s) => ({ ...s, latencyMs: d.latencyMs }));
        setOutputLevel(Math.random() * 0.6 + 0.2); // Simulate output level
      } else if (event === 'audio_sent') {
        setInputLevel(Math.random() * 0.5 + 0.3); // Simulate input level
      } else if (event === 'connected') {
        setStreamingStats((s) => ({ ...s, isConnected: true }));
      } else if (event === 'disconnected') {
        setStreamingStats((s) => ({ ...s, isConnected: false, isStreaming: false }));
        if (
          stageRef.current === 'performing'
          && !intentionalStopRef.current
          && !recoveringRef.current
        ) {
          recoveringRef.current = true;
          setSessionError('Connection lost. Attempting recovery...');
          void recoverPerformance();
        }
      } else if (event === 'streaming_started') {
        setStreamingStats((s) => ({ ...s, isStreaming: true }));
      } else if (event === 'streaming_stopped') {
        setStreamingStats((s) => ({ ...s, isStreaming: false }));
        setInputLevel(0);
        setOutputLevel(0);
      } else if (event === 'session_started') {
        applySessionStarted(data as Record<string, unknown>);
      }
    });

    return () => {
      client.disconnect();
    };
  }, [applySessionStarted, recoverPerformance]);

  // Poll separation status
  useEffect(() => {
    if (stage === 'separating' && separationJob) {
      pollingRef.current = window.setInterval(async () => {
        try {
          const status = await getSeparationStatus(separationJob.job_id);
          setSeparationJob(status);

          if (status.status === 'completed') {
            clearInterval(pollingRef.current!);
            setStage('ready');
          } else if (status.status === 'failed') {
            clearInterval(pollingRef.current!);
            setUploadError(status.error || 'Separation failed');
            setStage('upload');
          }
        } catch (error) {
          console.error('Error polling separation status:', error);
        }
      }, 1000);

      return () => {
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
        }
      };
    }
  }, [stage, separationJob]);

  const loadDevices = async () => {
    try {
      const result = await listDevices();
      setDevices(result.devices);
      // Set defaults
      const defaultDevice = result.devices.find((d) => d.type === 'output' && d.is_default)
        ?? result.devices.find((d) => d.type !== 'input' && d.is_default);
      if (defaultDevice) {
        setSpeakerDevice(defaultDevice.index);
        setHeadphoneDevice(defaultDevice.index);
      }
    } catch (error) {
      console.error('Failed to load devices:', error);
    }
  };

  const loadVoiceModels = async () => {
    try {
      const result = await listVoiceModels();
      setVoiceModels(result.models);
      if (result.models.length > 0) {
        setSelectedModel(result.models[0].id);
      }
    } catch (error) {
      console.error('Failed to load voice models:', error);
    }
  };

  const loadVoiceProfiles = async () => {
    try {
      const profiles = await apiService.listProfiles();
      setVoiceProfiles(profiles);
      if (!selectedProfileId) {
        const defaultProfile = profiles.find(
          (profile) => profile.profile_role !== 'source_artist' && isLiveReadyProfile(profile)
        );
        if (defaultProfile) {
          setSelectedProfileId(defaultProfile.profile_id);
        }
      }
    } catch (error) {
      console.error('Failed to load voice profiles:', error);
    }
  };

  const handleFileSelect = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setUploadError(null);
      setSessionError(null);

      try {
        // Upload song
        const song = await uploadSong(file);
        setUploadedSong(song);

        // Start separation
        const job = await startSeparation(song.song_id);
        setSeparationJob(job);
        setStage('separating');
      } catch (error) {
        setUploadError((error as Error).message);
      } finally {
        setIsUploading(false);
      }
    },
    []
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('audio/')) {
        handleFileSelect(file);
      }
    },
    [handleFileSelect]
  );

  const showSeparationStage = Boolean(separationJob) && stage !== 'upload';
  const separationStageReady = stage === 'ready' || stage === 'performing';
  const separationProgress = separationStageReady
    ? 100
    : separationJob?.progress ?? 0;

  const handlePipelineChange = useCallback((nextPipeline: PipelineType) => {
    if (!isLivePipeline(nextPipeline)) {
      return;
    }
    setPipeline(nextPipeline);
    void apiService.updateAppSettings({ preferred_live_pipeline: nextPipeline }).catch((error) => {
      console.error('Failed to persist pipeline preference:', error);
    });
  }, []);

  const handleExtractVoice = async () => {
    if (!uploadedSong) return;

    setIsExtracting(true);
    setSessionError(null);
    try {
      const result = await extractVoiceModel(
        uploadedSong.song_id,
        `Artist from ${uploadedSong.song_id.slice(0, 8)}`
      );
      await loadVoiceModels();
      setSelectedModel(result.model_id);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to extract voice'
      console.error('Failed to extract voice:', error);
      setSessionError(message);
    } finally {
      setIsExtracting(false);
    }
  };

  const handleDeviceChange = async (type: 'speaker' | 'headphone', index: number) => {
    try {
      setSessionError(null);
      if (type === 'speaker') {
        await setDeviceConfig({ speaker_device: index });
        setSpeakerDevice(index);
      } else {
        await setDeviceConfig({ headphone_device: index });
        setHeadphoneDevice(index);
      }
    } catch (error) {
      console.error('Failed to set device:', error);
      setSessionError(error instanceof Error ? error.message : 'Failed to update audio device');
    }
  };

  const startPerformance = async () => {
    if (!uploadedSong || !selectedModel) return;

    try {
      setSessionError(null);
      setPreflightIssues([]);
      setPreflightWarnings([]);
      const preflight = await apiService.karaokePreflight({
        song_id: uploadedSong.song_id,
        profile_id: selectedProfileId,
        voice_model_id: selectedModel,
        pipeline_type: pipeline,
      });
      if (!preflight.ok) {
        setPreflightIssues(preflight.issues);
        setPreflightWarnings(preflight.warnings);
        setSessionError(preflight.issues[0] || 'Karaoke preflight failed');
        return;
      }
      setPreflightWarnings(preflight.warnings);

      intentionalStopRef.current = false;
      const client = getAudioStreamingClient();
      await client.connect();
      // Pass pipeline type and optional profile/adapter for trained voice conversion
      const started = await client.startSession(uploadedSong.song_id, selectedModel, pipeline, {
        profileId: selectedProfileId || undefined,
        adapterType: selectedAdapter || undefined,
        collectSamples: collectTrainingSamples,
        vocalsPath: separationJob?.vocals_path,
        instrumentalPath: separationJob?.instrumental_path,
      });
      applySessionStarted(started as Record<string, unknown>);
      await client.startStreaming();
      setStage('performing');
    } catch (error) {
      console.error('Failed to start performance:', error);
      setSessionError(error instanceof Error ? error.message : 'Failed to start performance');
    }
  };

  const stopPerformance = async () => {
    try {
      intentionalStopRef.current = true;
      const client = getAudioStreamingClient();
      await client.endSession();
      setStage('ready');
    } catch (error) {
      console.error('Failed to stop performance:', error);
      setSessionError(error instanceof Error ? error.message : 'Failed to stop performance');
    } finally {
      intentionalStopRef.current = false;
    }
  };

  const resetSession = () => {
    intentionalStopRef.current = true;
    const client = getAudioStreamingClient();
    client.disconnect();
    setStage('upload');
    setUploadedSong(null);
    setSeparationJob(null);
    setUploadError(null);
    setSessionError(null);
    setPreflightIssues([]);
    setPreflightWarnings([]);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Live Karaoke</h1>
        {stage !== 'upload' && (
          <button
            onClick={resetSession}
            className="flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-white"
          >
            <RefreshCw size={16} />
            New Song
          </button>
        )}
      </div>

      {/* Stage: Upload */}
      {stage === 'upload' && (
        <div className="space-y-4">
          {/* Audio Device Config */}
          <div className="bg-gray-800 rounded-lg p-4">
            <AudioDeviceSelector compact />
          </div>

          <div className="bg-gray-800 rounded-lg p-8">
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              data-testid="karaoke-upload-dropzone"
              className={clsx(
                'border-2 border-dashed rounded-lg p-12 text-center transition',
                isUploading
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-gray-600 hover:border-gray-500'
              )}
            >
              <Upload className="mx-auto h-16 w-16 text-gray-500 mb-4" />
              <h2 className="text-xl font-semibold mb-2">Upload Your Song</h2>
              <p className="text-gray-400 mb-4">
                Drop an audio file here or click to select
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                data-testid="karaoke-upload-input"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileSelect(file);
                }}
                aria-label="Upload song file"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                data-testid="karaoke-select-file-button"
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-medium"
              >
                {isUploading ? 'Uploading...' : 'Select File'}
              </button>
              <p className="mt-4 text-sm text-gray-500">
                Supported: MP3, WAV, FLAC, M4A, OGG
              </p>
            </div>
            {uploadError && (
              <div className="mt-4">
                <StatusBanner
                  tone="danger"
                  title="Upload failed"
                  message={uploadError}
                  testId="karaoke-upload-error"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stage: Separating */}
      {showSeparationStage && separationJob && (
        <div className="bg-gray-800 rounded-lg p-8" data-testid="karaoke-separation-stage">
          <div className="text-center">
            <Music
              className={clsx(
                'mx-auto h-16 w-16 mb-4',
                separationStageReady ? 'text-green-500' : 'text-blue-500 animate-pulse'
              )}
            />
            <h2 className="text-xl font-semibold mb-2">
              {separationStageReady ? 'Song Ready' : 'Processing Your Song'}
            </h2>
            <p className="text-gray-400 mb-6">
              {separationStageReady
                ? 'Vocals and instrumental tracks are ready for live conversion.'
                : 'Separating vocals and instrumental tracks...'}
            </p>
            <div className="max-w-md mx-auto">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>{separationStageReady ? 'Status' : 'Progress'}</span>
                <span>{separationStageReady ? 'Complete' : `${separationProgress}%`}</span>
              </div>
              <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  data-testid="karaoke-separation-progress"
                  style={{ width: `${separationProgress}%` }}
                />
              </div>
              {!separationStageReady && separationJob.estimated_remaining !== undefined && (
                <p className="mt-2 text-sm text-gray-500">
                  ~{separationJob.estimated_remaining}s remaining
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Stage: Ready / Performing */}
      {(stage === 'ready' || stage === 'performing') && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pipeline Selection */}
          <div className="bg-gray-800 rounded-lg p-6 lg:col-span-2">
            <PipelineSelector
              value={pipeline}
              onChange={handlePipelineChange}
              context="live"
              disabled={stage === 'performing'}
              showDescription={true}
              size="md"
              statusByPipeline={pipelineStatus}
            />
          </div>

          {/* Voice Model Selection */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Music size={20} />
              Voice Model
            </h3>
            <div className="space-y-2">
              <label htmlFor="voice-model-select" className="text-sm text-gray-400">
                Select voice model
              </label>
              <select
                id="voice-model-select"
                value={selectedModel || ''}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={stage === 'performing'}
                data-testid="karaoke-voice-model-select"
                className="w-full p-3 bg-gray-700 rounded-lg"
              >
                {voiceModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.type})
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleExtractVoice}
              disabled={isExtracting || stage === 'performing'}
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm"
            >
              {isExtracting ? 'Extracting...' : 'Extract Voice from This Song'}
            </button>
          </div>

          {/* Voice Profile Selection (for trained adapter conversion) */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <User size={20} />
              Voice Profile
            </h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 mb-2 block">Target Voice Profile</label>
                <select
                  value={selectedProfileId || ''}
                  onChange={(e) => setSelectedProfileId(e.target.value || null)}
                  disabled={stage === 'performing'}
                  data-testid="karaoke-profile-select"
                  className="w-full p-3 bg-gray-700 rounded-lg"
                >
                  <option value="">No profile (use voice model only)</option>
                  {trainedTargetProfiles.map((profile) => (
                      <option key={profile.profile_id} value={profile.profile_id}>
                        {profile.name || profile.profile_id} ({profile.active_model_type === 'full_model' ? 'full model' : 'LoRA'} · {profile.sample_count} samples)
                      </option>
                    ))}
                </select>
                {trainedTargetProfiles.length === 0 && (
                  <p className="text-xs text-yellow-400 mt-2">
                    No live-ready target user profiles. Train a LoRA or dedicated full model on Voice Profiles page.
                  </p>
                )}
                {selectedProfile && (
                  <p className="text-xs text-gray-500 mt-1">
                    {selectedProfile.active_model_type === 'full_model'
                      ? 'Using the dedicated full model for this target user voice in live conversion.'
                      : 'Using the trained LoRA path for this target user voice.'}
                  </p>
                )}
              </div>

              {/* Adapter Selection (shown when profile selected) */}
              {selectedProfileId && selectedProfile?.active_model_type !== 'full_model' && (
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">LoRA Adapter</label>
                  <AdapterDropdown
                    profileId={selectedProfileId}
                    value={selectedAdapter}
                    onChange={setSelectedAdapter}
                    disabled={stage === 'performing'}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    {selectedAdapter === 'nvfp4' ? 'Fast inference (recommended for live)' : 'Maximum quality'}
                  </p>
                </div>
              )}
              {selectedProfileId && selectedProfile?.active_model_type === 'full_model' && (
                <div className="rounded-lg border border-violet-700 bg-violet-950/30 p-3 text-xs text-violet-200">
                  This target profile has a dedicated full model. Adapter selection is not used while the full model is active.
                </div>
              )}

              {/* Training sample collection option */}
              <div className="pt-3 border-t border-gray-700">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={collectTrainingSamples}
                    onChange={(e) => setCollectTrainingSamples(e.target.checked)}
                    disabled={stage === 'performing' || !selectedProfileId}
                    className="w-4 h-4 rounded bg-gray-700 border-gray-600"
                  />
                  <span className="text-sm text-gray-300">Collect training samples</span>
                </label>
                {collectTrainingSamples && selectedProfileId && (
                  <p className="text-xs text-gray-500 mt-2 ml-6">
                    Your singing will be captured to improve this profile.
                  </p>
                )}
                {!selectedProfileId && (
                  <p className="text-xs text-gray-500 mt-1 ml-6">
                    Select a profile to enable sample collection.
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Audio Devices */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Volume2 size={20} />
              Audio Output
            </h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 flex items-center gap-2 mb-2">
                  <Volume2 size={14} />
                  Speaker (Audience)
                </label>
                <select
                  value={speakerDevice ?? ''}
                  onChange={(e) => handleDeviceChange('speaker', parseInt(e.target.value))}
                  disabled={stage === 'performing'}
                  data-testid="karaoke-speaker-device-select"
                  className="w-full p-2 bg-gray-700 rounded"
                >
                  {outputDevices.map((d) => (
                    <option key={`${d.device_id ?? d.index}-speaker-${d.name}`} value={d.index}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-sm text-gray-400 flex items-center gap-2 mb-2">
                  <Headphones size={14} />
                  Headphone (You)
                </label>
                <select
                  value={headphoneDevice ?? ''}
                  onChange={(e) => handleDeviceChange('headphone', parseInt(e.target.value))}
                  disabled={stage === 'performing'}
                  data-testid="karaoke-headphone-device-select"
                  className="w-full p-2 bg-gray-700 rounded"
                >
                  {outputDevices.map((d) => (
                    <option key={`${d.device_id ?? d.index}-headphone-${d.name}`} value={d.index}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Audio Levels */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity size={20} />
              Audio Levels
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span className="flex items-center gap-1">
                    <Mic size={14} /> Input
                  </span>
                  <span>{Math.round(inputLevel * 100)}%</span>
                </div>
                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 transition-all"
                    style={{ width: `${inputLevel * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span className="flex items-center gap-1">
                    <Volume2 size={14} /> Output
                  </span>
                  <span>{Math.round(outputLevel * 100)}%</span>
                </div>
                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${outputLevel * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {uploadedSong && (
            <BrowserSingAlongCapture
              song={uploadedSong}
              profiles={voiceProfiles}
              disabled={stage === 'performing'}
              onSampleAttached={loadVoiceProfiles}
            />
          )}

          {/* Session Info with Real-time Latency */}
          <KaraokeSessionInfo
            requestedPipeline={streamingStats.requestedPipeline || pipeline}
            resolvedPipeline={streamingStats.resolvedPipeline}
            runtimeBackend={streamingStats.runtimeBackend}
            profileName={selectedProfile?.name || selectedProfileId || undefined}
            adapterType={selectedAdapter}
            modelType={selectedProfile?.active_model_type}
            latencyMs={streamingStats.latencyMs}
            isConnected={streamingStats.isConnected}
            isStreaming={streamingStats.isStreaming}
            chunksProcessed={streamingStats.chunksProcessed}
            sampleCollectionEnabled={streamingStats.sampleCollectionEnabled}
            audioRouterTargets={streamingStats.audioRouterTargets || {
              speaker_device: speakerDevice,
              headphone_device: headphoneDevice,
            }}
          />

          {(preflightIssues.length > 0 || preflightWarnings.length > 0) && (
            <StatusBanner
              tone={preflightIssues.length > 0 ? 'warning' : 'info'}
              title={preflightIssues.length > 0 ? 'Preflight attention required' : 'Preflight warnings'}
              details={[...preflightIssues, ...preflightWarnings]}
            />
          )}

          {sessionError && (
            <StatusBanner
              tone="danger"
              title="Live karaoke session error"
              message={sessionError}
              testId="karaoke-session-error"
            />
          )}
        </div>
      )}

      {/* Performance Controls */}
      {(stage === 'ready' || stage === 'performing') && (
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-center gap-4">
            {stage === 'ready' ? (
              <button
                onClick={startPerformance}
                disabled={!selectedModel}
                data-testid="karaoke-start-button"
                className="flex items-center gap-3 px-8 py-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-xl text-xl font-semibold"
              >
                <Play size={28} />
                Start Performing
              </button>
            ) : (
              <>
                <button
                  onClick={stopPerformance}
                  data-testid="karaoke-stop-button"
                  className="flex items-center gap-3 px-8 py-4 bg-red-600 hover:bg-red-700 rounded-xl text-xl font-semibold"
                >
                  <Square size={28} />
                  Stop
                </button>
                <div
                  className="flex items-center gap-2 px-6 py-4 bg-green-600/20 border border-green-500 rounded-xl"
                  data-testid="karaoke-live-indicator"
                >
                  {streamingStats.isStreaming ? (
                    <>
                      <Mic size={24} className="text-green-400 animate-pulse" />
                      <span className="text-green-400 font-semibold">LIVE</span>
                    </>
                  ) : (
                    <>
                      <MicOff size={24} className="text-gray-400" />
                      <span className="text-gray-400">Connecting...</span>
                    </>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
