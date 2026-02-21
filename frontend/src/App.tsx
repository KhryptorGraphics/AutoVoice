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
import { PipelineSelector, PipelineBadge, type PipelineType, getPreferredPipeline } from './components/PipelineSelector'
import { apiService, VoiceProfile, AdapterType, ConversionRecord } from './services/api'
import { ToastProvider } from './contexts/ToastContext'
import clsx from 'clsx'

function ConvertPage() {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null)
  const [selectedAdapter, setSelectedAdapter] = useState<AdapterType | null>(null)
  // Load preferred pipeline from localStorage, default to quality for song conversion
  const [pipeline, setPipeline] = useState<PipelineType>(() => {
    return getPreferredPipeline() || 'quality'
  })
  const [file, setFile] = useState<File | null>(null)
  const [isConverting, setIsConverting] = useState(false)
  const [conversionStatus, setConversionStatus] = useState<ConversionRecord | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    apiService.listProfiles().then(setProfiles).catch(console.error)
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setError(null)
      setConversionStatus(null)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('audio/')) {
      setFile(f)
      setError(null)
      setConversionStatus(null)
    }
  }, [])

  const handleConvert = async () => {
    if (!file || !selectedProfile) return

    setIsConverting(true)
    setError(null)

    try {
      // Start conversion with adapter type and pipeline selection
      const result = await apiService.convertSong(file, selectedProfile, {
        preset: 'balanced',
        pipeline_type: pipeline,
        adapter_type: selectedAdapter || undefined,
      })

      // Poll for status
      const pollStatus = async () => {
        const status = await apiService.getConversionStatus(result.job_id)
        setConversionStatus(status)

        if (status.status === 'processing' || status.status === 'queued') {
          setTimeout(pollStatus, 1000)
        } else if (status.status === 'error') {
          setError(status.error || 'Conversion failed')
          setIsConverting(false)
        } else if (status.status === 'complete' || status.status === 'completed') {
          setIsConverting(false)
        }
      }

      pollStatus()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Conversion failed')
      setIsConverting(false)
    }
  }

  const handleDownload = async () => {
    if (!conversionStatus) return
    try {
      const blob = await apiService.downloadResult(conversionStatus.id)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `converted_${file?.name || 'audio.wav'}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError('Download failed')
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
                  onClick={() => setFile(null)}
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
              className="w-full p-3 bg-gray-700 rounded-lg"
            >
              <option value="">Choose a profile...</option>
              {profiles.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profile.name || profile.profile_id} ({profile.sample_count} samples)
                </option>
              ))}
            </select>
            {profiles.length === 0 && (
              <p className="text-sm text-gray-500 mt-2">
                No profiles found. <a href="/profiles" className="text-blue-400 hover:underline">Create one</a>
              </p>
            )}
          </div>

          {/* Adapter Selection */}
          {selectedProfile && (
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

          {/* Pipeline Selection */}
          <div>
            <h2 className="text-lg font-semibold mb-4">4. Select Pipeline</h2>
            <PipelineSelector
              value={pipeline}
              onChange={setPipeline}
              showDescription={true}
            />
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
              <button
                onClick={handleDownload}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
              >
                Download
              </button>
            </div>
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
              {selectedAdapter && (
                <AdapterBadge adapterType={selectedAdapter} />
              )}
            </>
          )}
        </button>
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
