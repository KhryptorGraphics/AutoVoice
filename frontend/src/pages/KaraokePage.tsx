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
import { apiService, VoiceProfile } from '../services/api';
import { AudioDeviceSelector } from '../components/AudioDeviceSelector';
import { PipelineSelector, type PipelineType, getPreferredPipeline } from '../components/PipelineSelector';
import { AdapterDropdown } from '../components/AdapterSelector';
import { AdapterType } from '../services/api';
import { KaraokeSessionInfo } from '../components/KaraokeSessionInfo';

type Stage = 'upload' | 'separating' | 'ready' | 'performing';

export function KaraokePage() {
  // Stage management
  const [stage, setStage] = useState<Stage>('upload');

  // Pipeline selection (default to realtime for karaoke, but respect user preference)
  const [pipeline, setPipeline] = useState<PipelineType>(() => {
    const preferred = getPreferredPipeline();
    // For karaoke, prefer realtime pipelines but allow quality if that's what user saved
    if (preferred === 'realtime' || preferred === 'realtime_meanvc') {
      return preferred;
    }
    return 'realtime';
  });

  // Song state
  const [uploadedSong, setUploadedSong] = useState<UploadedSong | null>(null);
  const [separationJob, setSeparationJob] = useState<SeparationJob | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

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

  // Load devices, models, and profiles on mount
  useEffect(() => {
    loadDevices();
    loadVoiceModels();
    loadVoiceProfiles();
  }, []);

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
      } else if (event === 'streaming_started') {
        setStreamingStats((s) => ({ ...s, isStreaming: true }));
      } else if (event === 'streaming_stopped') {
        setStreamingStats((s) => ({ ...s, isStreaming: false }));
        setInputLevel(0);
        setOutputLevel(0);
      }
    });

    return () => {
      client.disconnect();
    };
  }, []);

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
      const defaultDevice = result.devices.find((d) => d.is_default);
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
      if (profiles.length > 0 && !selectedProfileId) {
        setSelectedProfileId(profiles[0].profile_id);
      }
    } catch (error) {
      console.error('Failed to load voice profiles:', error);
    }
  };

  const handleFileSelect = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setUploadError(null);

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

  const handleExtractVoice = async () => {
    if (!uploadedSong) return;

    setIsExtracting(true);
    try {
      const result = await extractVoiceModel(
        uploadedSong.song_id,
        `Artist from ${uploadedSong.song_id.slice(0, 8)}`
      );
      await loadVoiceModels();
      setSelectedModel(result.model_id);
    } catch (error) {
      console.error('Failed to extract voice:', error);
    } finally {
      setIsExtracting(false);
    }
  };

  const handleDeviceChange = async (type: 'speaker' | 'headphone', index: number) => {
    try {
      if (type === 'speaker') {
        await setDeviceConfig({ speaker_device: index });
        setSpeakerDevice(index);
      } else {
        await setDeviceConfig({ headphone_device: index });
        setHeadphoneDevice(index);
      }
    } catch (error) {
      console.error('Failed to set device:', error);
    }
  };

  const startPerformance = async () => {
    if (!uploadedSong || !selectedModel) return;

    try {
      const client = getAudioStreamingClient();
      await client.connect();
      // Pass pipeline type and optional profile/adapter for trained voice conversion
      await client.startSession(uploadedSong.song_id, selectedModel, pipeline, {
        profileId: selectedProfileId || undefined,
        adapterType: selectedAdapter || undefined,
        collectSamples: collectTrainingSamples,
      });
      await client.startStreaming();
      setStage('performing');
    } catch (error) {
      console.error('Failed to start performance:', error);
    }
  };

  const stopPerformance = async () => {
    try {
      const client = getAudioStreamingClient();
      await client.endSession();
      setStage('ready');
    } catch (error) {
      console.error('Failed to stop performance:', error);
    }
  };

  const resetSession = () => {
    const client = getAudioStreamingClient();
    client.disconnect();
    setStage('upload');
    setUploadedSong(null);
    setSeparationJob(null);
    setUploadError(null);
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
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileSelect(file);
                }}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-medium"
              >
                {isUploading ? 'Uploading...' : 'Select File'}
              </button>
              <p className="mt-4 text-sm text-gray-500">
                Supported: MP3, WAV, FLAC, M4A, OGG
              </p>
            </div>
            {uploadError && (
              <div className="mt-4 p-3 bg-red-500/20 border border-red-500 rounded text-red-400">
                {uploadError}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stage: Separating */}
      {stage === 'separating' && separationJob && (
        <div className="bg-gray-800 rounded-lg p-8">
          <div className="text-center">
            <Music className="mx-auto h-16 w-16 text-blue-500 mb-4 animate-pulse" />
            <h2 className="text-xl font-semibold mb-2">Processing Your Song</h2>
            <p className="text-gray-400 mb-6">
              Separating vocals and instrumental tracks...
            </p>
            <div className="max-w-md mx-auto">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Progress</span>
                <span>{separationJob.progress}%</span>
              </div>
              <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${separationJob.progress}%` }}
                />
              </div>
              {separationJob.estimated_remaining !== undefined && (
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
              onChange={setPipeline}
              disabled={stage === 'performing'}
              showDescription={true}
              size="md"
            />
          </div>

          {/* Voice Model Selection */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Music size={20} />
              Voice Model
            </h3>
            <select
              value={selectedModel || ''}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={stage === 'performing'}
              className="w-full p-3 bg-gray-700 rounded-lg mb-4"
            >
              {voiceModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.type})
                </option>
              ))}
            </select>

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
                  className="w-full p-3 bg-gray-700 rounded-lg"
                >
                  <option value="">No profile (use voice model only)</option>
                  {voiceProfiles
                    .filter((profile) => profile.has_trained_model)
                    .map((profile) => (
                      <option key={profile.profile_id} value={profile.profile_id}>
                        {profile.name || profile.profile_id} ({profile.sample_count} samples)
                      </option>
                    ))}
                </select>
                {voiceProfiles.filter((p) => p.has_trained_model).length === 0 && (
                  <p className="text-xs text-yellow-400 mt-2">
                    No trained profiles. Train one on Voice Profiles page for better quality.
                  </p>
                )}
                {selectedProfileId && (
                  <p className="text-xs text-gray-500 mt-1">
                    Uses trained LoRA adapter for enhanced voice conversion quality.
                  </p>
                )}
              </div>

              {/* Adapter Selection (shown when profile selected) */}
              {selectedProfileId && (
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
                  className="w-full p-2 bg-gray-700 rounded"
                >
                  {devices.map((d) => (
                    <option key={d.index} value={d.index}>
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
                  className="w-full p-2 bg-gray-700 rounded"
                >
                  {devices.map((d) => (
                    <option key={d.index} value={d.index}>
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

          {/* Session Info with Real-time Latency */}
          <KaraokeSessionInfo
            pipeline={pipeline}
            profileName={selectedProfileId ? voiceProfiles.find(p => p.profile_id === selectedProfileId)?.name || selectedProfileId : undefined}
            adapterType={selectedAdapter}
            latencyMs={streamingStats.latencyMs}
            isConnected={streamingStats.isConnected}
            isStreaming={streamingStats.isStreaming}
            chunksProcessed={streamingStats.chunksProcessed}
          />
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
                className="flex items-center gap-3 px-8 py-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-xl text-xl font-semibold"
              >
                <Play size={28} />
                Start Performing
              </button>
            ) : (
              <>
                <button
                  onClick={stopPerformance}
                  className="flex items-center gap-3 px-8 py-4 bg-red-600 hover:bg-red-700 rounded-xl text-xl font-semibold"
                >
                  <Square size={28} />
                  Stop
                </button>
                <div className="flex items-center gap-2 px-6 py-4 bg-green-600/20 border border-green-500 rounded-xl">
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
