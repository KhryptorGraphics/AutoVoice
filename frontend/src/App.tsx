import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { Music, History, Activity, Mic, HelpCircle, User, Users, Youtube, Upload, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { ConversionHistoryPage } from './pages/ConversionHistoryPage'
import { SystemStatusPage } from './pages/SystemStatusPage'
import { KaraokePage } from './pages/KaraokePage'
import { VoiceProfilePage } from './pages/VoiceProfilePage'
import { DiarizationResultsPage } from './pages/DiarizationResultsPage'
import { YouTubeDownloadPage } from './pages/YouTubeDownloadPage'
import HelpPage from './pages/HelpPage'
import { AdapterSelector, AdapterBadge } from './components/AdapterSelector'
import {
  PipelineSelector,
  PipelineBadge,
  type PipelineType,
  getPreferredPipeline,
  isOfflinePipeline,
} from './components/PipelineSelector'
import { PresetManager } from './components/PresetManager'
import { BatchProcessingQueue } from './components/BatchProcessingQueue'
import { ConversionHistoryTable } from './components/ConversionHistoryTable'
import {
  apiService,
  VoiceProfile,
  AdapterType,
  ConversionRecord,
  DEFAULT_CONVERSION_CONFIG,
  type ConversionConfig,
} from './services/api'
import { ToastProvider, useToastContext } from './contexts/ToastContext'
import clsx from 'clsx'

function ConvertPage() {
  const toast = useToastContext()
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null)
  const [selectedAdapter, setSelectedAdapter] = useState<AdapterType | null>(null)
  const [config, setConfig] = useState<ConversionConfig>(() => {
    const preferred = getPreferredPipeline()
    return {
      ...DEFAULT_CONVERSION_CONFIG,
      pipeline_type: preferred && isOfflinePipeline(preferred) ? preferred : 'quality_seedvc',
    }
  })
  const [pipelineStatus, setPipelineStatus] = useState<Record<string, { loaded: boolean; memory_gb?: number; latency_target_ms?: number }>>({})
  const [file, setFile] = useState<File | null>(null)
  const [isConverting, setIsConverting] = useState(false)
  const [conversionStatus, setConversionStatus] = useState<ConversionRecord | null>(null)
  const [inlineResult, setInlineResult] = useState<{ blob: Blob; jobId: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const targetProfiles = profiles.filter(
    (profile) => profile.profile_role !== 'source_artist' && profile.has_trained_model
  )
  const selectedProfileRecord = selectedProfile
    ? targetProfiles.find((profile) => profile.profile_id === selectedProfile) ?? null
    : null

  useEffect(() => {
    apiService.listProfiles().then(setProfiles).catch(console.error)
  }, [])

  useEffect(() => {
    let cancelled = false

    const loadPipelinePreferences = async () => {
      try {
        const [settings, status] = await Promise.all([
          apiService.getAppSettings(),
          apiService.getPipelineStatus(),
        ])
        if (cancelled) {
          return
        }
        if (isOfflinePipeline(settings.preferred_offline_pipeline)) {
          setConfig((prev) => ({ ...prev, pipeline_type: settings.preferred_offline_pipeline }))
        }
        setPipelineStatus(status.pipelines || {})
      } catch (error) {
        console.error('Failed to load pipeline preferences:', error)
      }
    }

    void loadPipelinePreferences()
    return () => {
      cancelled = true
    }
  }, [])

  const handlePipelineChange = useCallback((nextPipeline: PipelineType) => {
    if (!isOfflinePipeline(nextPipeline)) return
    setConfig((prev) => ({ ...prev, pipeline_type: nextPipeline }))
    void apiService.updateAppSettings({ preferred_offline_pipeline: nextPipeline }).catch((error) => {
      console.error('Failed to persist pipeline preference:', error)
    })
  }, [])

  useEffect(() => {
    if (selectedProfileRecord?.active_model_type === 'full_model' && selectedAdapter) {
      setSelectedAdapter(null)
    }
  }, [selectedAdapter, selectedProfileRecord?.active_model_type])

  useEffect(() => {
    if (selectedProfile && !selectedProfileRecord) {
      setSelectedProfile(targetProfiles[0]?.profile_id ?? null)
    }
  }, [selectedProfile, selectedProfileRecord, targetProfiles])

  const handlePresetLoad = useCallback((presetConfig: Partial<ConversionConfig>) => {
    setConfig((prev) => ({ ...prev, ...presetConfig }))
    if (presetConfig.pipeline_type && isOfflinePipeline(presetConfig.pipeline_type)) {
      void apiService.updateAppSettings({ preferred_offline_pipeline: presetConfig.pipeline_type }).catch((error) => {
        console.error('Failed to persist preset pipeline preference:', error)
      })
    }
  }, [])

  const handleHistorySelect = useCallback((record: ConversionRecord) => {
    setSelectedProfile(record.profile_id)
    if (record.adapter_type && record.adapter_type !== 'unified') {
      setSelectedAdapter(record.adapter_type)
    }
    const historyPipeline = record.pipeline_type
    if (historyPipeline && isOfflinePipeline(historyPipeline)) {
      setConfig((prev) => ({
        ...prev,
        pipeline_type: historyPipeline,
        preset: (record.preset as ConversionConfig['preset']) || prev.preset,
      }))
    }
    toast.info(`Loaded settings from ${record.originalFileName || record.input_file}`)
  }, [toast])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setError(null)
      setConversionStatus(null)
      setInlineResult(null)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('audio/')) {
      setFile(f)
      setError(null)
      setConversionStatus(null)
      setInlineResult(null)
    }
  }, [])

  const handleConvert = async () => {
    if (!file || !selectedProfile) return

    setIsConverting(true)
    setError(null)
    setInlineResult(null)

    try {
      // Start conversion with adapter type and pipeline selection
      const result = await apiService.convertSong(file, selectedProfile, {
        preset: config.preset,
        vocal_volume: config.vocal_volume,
        instrumental_volume: config.instrumental_volume,
        pitch_shift: config.pitch_shift,
        pipeline_type: config.pipeline_type,
        adapter_type:
          selectedProfileRecord?.active_model_type === 'full_model'
            ? undefined
            : (selectedAdapter || undefined),
        return_stems: config.return_stems,
      })

      if (result.audio) {
        const binary = atob(result.audio)
        const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
        const blob = new Blob([bytes], { type: 'audio/wav' })
        setInlineResult({ blob, jobId: result.job_id })
        setConversionStatus({
          id: result.job_id,
          status: 'completed',
          created_at: new Date().toISOString(),
          completed_at: new Date().toISOString(),
          input_file: file.name,
          profile_id: selectedProfile,
          preset: config.preset,
          duration: result.duration,
          active_model_type: result.active_model_type,
          adapter_type: result.adapter_type,
          requested_pipeline: result.requested_pipeline,
          resolved_pipeline: result.resolved_pipeline,
          pipeline_type: result.resolved_pipeline || result.requested_pipeline || config.pipeline_type,
          runtime_backend: result.runtime_backend,
        })
        toast.success('Conversion completed successfully!')
        setIsConverting(false)
        return
      }

      // Poll for status
      const pollStatus = async () => {
        const status = await apiService.getConversionStatus(result.job_id)
        setConversionStatus({ ...status, id: result.job_id })

        if (status.status === 'processing' || status.status === 'queued') {
          setTimeout(pollStatus, 1000)
        } else if (status.status === 'error') {
          const errorMsg = status.error || 'Conversion failed'
          setError(errorMsg)
          toast.error(errorMsg)
          setIsConverting(false)
        } else if (status.status === 'complete' || status.status === 'completed') {
          toast.success('Conversion completed successfully!')
          setIsConverting(false)
        }
      }

      pollStatus()
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Conversion failed'
      setError(errorMsg)
      toast.error(errorMsg)
      setIsConverting(false)
    }
  }

  const triggerDownload = useCallback((blob: Blob, downloadName: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = downloadName
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  const handleDownload = async () => {
    if (!conversionStatus) return
    try {
      const blob = inlineResult?.jobId === conversionStatus.id
        ? inlineResult.blob
        : await apiService.downloadResult(conversionStatus.id)
      triggerDownload(blob, `converted_${file?.name || 'audio.wav'}`)
    } catch {
      const errorMsg = 'Download failed'
      setError(errorMsg)
      toast.error(errorMsg)
    }
  }

  const handleDownloadStem = async (variant: 'vocals' | 'instrumental') => {
    if (!conversionStatus) return
    try {
      const blob = await apiService.downloadConversionAsset(conversionStatus.id, variant)
      const baseName = file?.name?.replace(/\.[^/.]+$/, '') || 'audio'
      triggerDownload(blob, `${baseName}_${variant}.wav`)
    } catch {
      const errorMsg = `Failed to download ${variant} stem`
      setError(errorMsg)
      toast.error(errorMsg)
    }
  }

  const handleReassemble = async () => {
    if (!conversionStatus) return
    try {
      const blob = await apiService.reassembleConversion(conversionStatus.id)
      const baseName = file?.name?.replace(/\.[^/.]+$/, '') || 'audio'
      triggerDownload(blob, `${baseName}_reassembled.wav`)
    } catch {
      const errorMsg = 'Failed to reassemble converted vocals with instrumental'
      setError(errorMsg)
      toast.error(errorMsg)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Voice Conversion</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">1. Select Audio</h2>
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className={clsx(
              'border-2 border-dashed rounded-lg p-8 text-center transition',
              file ? 'border-green-500 bg-green-500/10' : 'border-gray-600 hover:border-gray-500'
            )}
          >
            {file ? (
              <div className="space-y-2">
                <CheckCircle className="mx-auto h-10 w-10 text-green-500" />
                <p className="text-white font-medium">{file.name}</p>
                <p className="text-gray-400 text-sm">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                <button
                  onClick={() => {
                    setFile(null)
                    setInlineResult(null)
                  }}
                  className="text-red-400 hover:text-red-300 text-sm"
                >
                  Remove
                </button>
              </div>
            ) : (
              <>
                <Upload className="mx-auto h-12 w-12 text-gray-500 mb-3" />
                <p className="text-gray-400 mb-3">Drop audio file here or click to upload</p>
                <input
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  id="audio-upload"
                  data-testid="conversion-audio-input"
                  onChange={handleFileSelect}
                />
                <label
                  htmlFor="audio-upload"
                  className="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer"
                >
                  Select File
                </label>
              </>
            )}
          </div>
        </div>

        {/* Profile & Adapter Selection */}
        <div className="bg-gray-800 rounded-lg p-6 space-y-6">
          <div>
            <h2 className="text-lg font-semibold mb-4">2. Select Voice Profile</h2>
            <select
              value={selectedProfile || ''}
              onChange={(e) => {
                setSelectedProfile(e.target.value || null)
                setSelectedAdapter(null) // Reset adapter when profile changes
              }}
              data-testid="voice-profile-selector"
              className="w-full p-3 bg-gray-700 rounded-lg"
            >
              <option value="">Choose a profile...</option>
              {targetProfiles.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profile.name || profile.profile_id} (
                  {profile.active_model_type === 'full_model' ? 'full model' : 'LoRA'} ·
                  {' '}
                  {profile.sample_count} samples)
                </option>
              ))}
            </select>
            {targetProfiles.length === 0 && (
              <p className="text-sm text-gray-500 mt-2">
                No trained target profiles found. <a href="/profiles" className="text-blue-400 hover:underline">Create or train one</a>
              </p>
            )}
            {selectedProfileRecord && (
              <div className="mt-3 flex items-center gap-2 text-sm text-gray-300">
                <span className="rounded-full border border-gray-600 px-2 py-0.5">
                  {selectedProfileRecord.active_model_type === 'full_model' ? 'Full model' : 'LoRA target'}
                </span>
                <span>
                  {selectedProfileRecord.clean_vocal_minutes?.toFixed(1) ?? '0.0'} min clean vocals
                </span>
              </div>
            )}
          </div>

          {/* Adapter Selection */}
          {selectedProfile && selectedProfileRecord?.active_model_type !== 'full_model' && (
            <div>
              <h2 className="text-lg font-semibold mb-4">3. Select Adapter</h2>
              <AdapterSelector
                profileId={selectedProfile}
                value={selectedAdapter}
                onChange={setSelectedAdapter}
                showMetrics={true}
                size="md"
              />
            </div>
          )}
          {selectedProfileRecord?.active_model_type === 'full_model' && (
            <div className="rounded-lg border border-violet-500/40 bg-violet-500/10 p-4 text-sm text-violet-100">
              This target profile has a dedicated full model. Offline conversion will use that model directly instead of a LoRA adapter.
            </div>
          )}

          {/* Pipeline Selection */}
          <div>
            <h2 className="text-lg font-semibold mb-4">4. Select Pipeline</h2>
            <PipelineSelector
              value={config.pipeline_type}
              onChange={handlePipelineChange}
              context="offline"
              showDescription={true}
              statusByPipeline={pipelineStatus}
            />
            <div className="mt-4">
              <PresetManager
                currentConfig={config}
                onLoadPreset={handlePresetLoad}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Convert Button & Status */}
      <div className="mt-6 bg-gray-800 rounded-lg p-6">
        {error && (
          <div className="mb-4 p-3 bg-red-500/20 border border-red-500 rounded text-red-400 flex items-center gap-2">
            <XCircle size={16} />
            {error}
          </div>
        )}

        {(conversionStatus?.status === 'complete' || conversionStatus?.status === 'completed') && (
          <div className="mb-4 p-4 bg-green-500/20 border border-green-500 rounded">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-green-400">
                <CheckCircle size={20} />
                <span className="font-semibold">Conversion Complete!</span>
                {conversionStatus.pipeline_type && (
                  <PipelineBadge pipeline={conversionStatus.pipeline_type as PipelineType} />
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {conversionStatus.stem_urls?.vocals && (
                  <button
                    onClick={() => void handleDownloadStem('vocals')}
                    className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                  >
                    Download Vocals
                  </button>
                )}
                {conversionStatus.stem_urls?.instrumental && (
                  <button
                    onClick={() => void handleDownloadStem('instrumental')}
                    className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                  >
                    Download Instrumental
                  </button>
                )}
                {conversionStatus.reassemble_url && (
                  <button
                    onClick={() => void handleReassemble()}
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                  >
                    Reassemble With Instrumental
                  </button>
                )}
                <button
                  onClick={handleDownload}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
                >
                  Download Mix
                </button>
              </div>
            </div>
            {(conversionStatus.stem_urls?.vocals || conversionStatus.stem_urls?.instrumental) && (
              <p className="mt-3 text-sm text-gray-300">
                This conversion kept separate stems, so you can download the converted voice track,
                keep the instrumental, or reassemble them into a fresh mixed output.
              </p>
            )}
            {/* Show processing metrics if available */}
            {(conversionStatus.processing_time_seconds || conversionStatus.rtf) && (
              <div className="mt-2 flex gap-4 text-sm text-gray-400">
                {conversionStatus.processing_time_seconds !== undefined && (
                  <span>Processed in {conversionStatus.processing_time_seconds.toFixed(1)}s</span>
                )}
                {conversionStatus.rtf !== undefined && (
                  <span>RTF: {conversionStatus.rtf.toFixed(2)}x</span>
                )}
              </div>
            )}
          </div>
        )}

        <button
          onClick={handleConvert}
          disabled={!file || !selectedProfile || isConverting}
          data-testid="start-conversion-button"
          className={clsx(
            'w-full flex items-center justify-center gap-3 px-6 py-4 rounded-lg text-lg font-semibold transition',
            !file || !selectedProfile || isConverting
              ? 'bg-gray-600 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          )}
        >
          {isConverting ? (
            <>
              <Loader2 className="animate-spin" size={24} />
              Converting... {conversionStatus?.status === 'processing' && '(Processing)'}
            </>
          ) : (
            <>
              <Music size={24} />
              Convert Song
              {selectedProfileRecord?.active_model_type === 'full_model' ? (
                <span className="rounded-full bg-violet-500/20 px-2 py-0.5 text-sm text-violet-200">
                  Full model
                </span>
              ) : selectedAdapter ? (
                <AdapterBadge adapterType={selectedAdapter} />
              ) : null}
            </>
          )}
        </button>

        {selectedProfile && (
          <div className="mt-6">
            <BatchProcessingQueue
              profileId={selectedProfile}
              config={config}
            />
          </div>
        )}
      </div>

      <div className="mt-6">
        <ConversionHistoryTable onSelect={handleHistorySelect} />
      </div>
    </div>
  )
}

export default function App() {
  const navItems = [
    { to: '/', label: 'Convert', icon: Music },
    { to: '/karaoke', label: 'Karaoke', icon: Mic },
    { to: '/profiles', label: 'Profiles', icon: User },
    { to: '/youtube', label: 'YouTube', icon: Youtube },
    { to: '/diarization', label: 'Diarization', icon: Users },
    { to: '/history', label: 'History', icon: History },
    { to: '/system', label: 'System', icon: Activity },
    { to: '/help', label: 'Help', icon: HelpCircle },
  ]

  return (
    <ToastProvider position="top-right">
      <div className="min-h-screen bg-gray-900 text-white">
        <nav className="bg-gray-800 border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4 flex items-center h-14">
            <span className="text-xl font-bold text-blue-400 mr-8">AutoVoice</span>
            <div className="flex gap-1">
              {navItems.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center gap-2 px-3 py-2 rounded text-sm',
                      isActive
                        ? 'bg-gray-700 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                    )
                  }
                >
                  <Icon size={16} />
                  {label}
                </NavLink>
              ))}
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<ConvertPage />} />
            <Route path="/karaoke" element={<KaraokePage />} />
            <Route path="/profiles" element={<VoiceProfilePage />} />
            <Route path="/youtube" element={<YouTubeDownloadPage />} />
            <Route path="/diarization" element={<DiarizationResultsPage />} />
            <Route path="/history" element={<ConversionHistoryPage />} />
            <Route path="/system" element={<SystemStatusPage />} />
            <Route path="/help" element={<HelpPage />} />
          </Routes>
        </main>
      </div>
    </ToastProvider>
  )
}
