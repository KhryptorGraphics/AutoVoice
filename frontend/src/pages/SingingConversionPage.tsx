import { useState, useEffect } from 'react'
import { Music, Download, Play } from 'lucide-react'
import { UploadInterface } from '../components/SingingConversion/UploadInterface'
import { ConversionControls, ConversionSettings } from '../components/SingingConversion/ConversionControls'
import { ProgressDisplay } from '../components/SingingConversion/ProgressDisplay'
import { VoiceProfileSelector } from '../components/VoiceProfileSelector'
import { AudioWaveform } from '../components/AudioWaveform'
import { QualityMetricsDisplay } from '../components/QualityMetricsDisplay'
import { apiService, VoiceProfile, QualityMetrics } from '../services/api'
import { wsService, ConversionProgress } from '../services/websocket'
import { useQuery } from '@tanstack/react-query'

export function SingingConversionPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedProfile, setSelectedProfile] = useState<VoiceProfile | null>(null)
  const [isConverting, setIsConverting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [websocketRoom, setWebsocketRoom] = useState<string>('')
  const [progress, setProgress] = useState<ConversionProgress | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [originalAudioUrl, setOriginalAudioUrl] = useState<string | null>(null)
  const [convertedPitchData, setConvertedPitchData] = useState<{f0: number[], times: number[]} | null>(null)
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null)
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsError, setMetricsError] = useState<string | null>(null)
  const [waveformError, setWaveformError] = useState<string | null>(null)

  const [settings, setSettings] = useState<ConversionSettings>({
    pitchShift: 0,
    preserveOriginalPitch: true,
    preserveVibrato: true,
    preserveExpression: true,
    outputQuality: 'balanced',
    denoiseInput: false,
    enhanceOutput: false,
  })

  // Fetch voice profiles
  const { data: profiles = [], isLoading: profilesLoading } = useQuery({
    queryKey: ['voiceProfiles'],
    queryFn: () => apiService.getVoiceProfiles(),
  })

  // Connect to WebSocket on mount
  useEffect(() => {
    wsService.connect().catch((err) => {
      console.error('Failed to connect to WebSocket:', err)
    })

    return () => {
      if (jobId) {
        wsService.unsubscribeFromJob(jobId, websocketRoom)
      }
    }
  }, [jobId, websocketRoom])

  // Create audio URL when file is selected
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile)
      setOriginalAudioUrl(url)

      // Cleanup on unmount or when file changes
      return () => {
        URL.revokeObjectURL(url)
      }
    } else {
      setOriginalAudioUrl(null)
    }
  }, [selectedFile])

  // Cleanup resultUrl when it changes (blob URLs only)
  useEffect(() => {
    return () => {
      if (resultUrl && resultUrl.startsWith('blob:')) {
        URL.revokeObjectURL(resultUrl)
      }
    }
  }, [resultUrl])

  const handleStartConversion = async () => {
    if (!selectedFile || !selectedProfile) {
      setError('Please select both a song file and a voice profile')
      return
    }

    setError(null)
    setIsConverting(true)
    setProgress(null)
    setResultUrl(null)
    setConvertedPitchData(null)
    setQualityMetrics(null)
    setMetricsError(null)
    setWaveformError(null)

    try {
      // Ensure WebSocket is connected
      if (!wsService.isConnected()) {
        await wsService.connect()
      }

      // Start conversion (returns job_id for async mode)
      const response = await apiService.convertSong(
        selectedFile,
        selectedProfile.id,
        {
          pitchShift: settings.pitchShift,
          preserveOriginalPitch: settings.preserveOriginalPitch,
          preserveVibrato: settings.preserveVibrato,
          preserveExpression: settings.preserveExpression,
          outputQuality: settings.outputQuality,
          denoiseInput: settings.denoiseInput,
          enhanceOutput: settings.enhanceOutput,
        }
      )

      // Handle async response (202 with job_id)
      if (response.status === 'queued' && response.job_id) {
        const newJobId = response.job_id
        setJobId(newJobId)
        setWebsocketRoom(response.websocket_room || response.job_id)  // ADDED

        // Subscribe to WebSocket updates BEFORE job starts processing
        await wsService.subscribeToJob(newJobId, {
          onProgress: (progressData) => {
            // Backend sends progress and stage directly
            setProgress(progressData)
          },
          onComplete: async (result) => {
            // Extract pitch data if available
            if (result.f0_contour && result.f0_times) {
              setConvertedPitchData({
                f0: result.f0_contour,
                times: result.f0_times
              })
            }

            // Handle both payload shapes:
            // - JobManager flows provide output_url
            // - Streaming flows provide audio (base64)
            if (result.output_url) {
              // JobManager path: fetch audio and create blob URL
              try {
                const audioBlob = await apiService.downloadConvertedAudio(newJobId)
                const blobUrl = URL.createObjectURL(audioBlob)
                setResultUrl(blobUrl)
              } catch (err) {
                setError('Failed to load converted audio')
              }
            } else if (result.audio) {
              // Streaming path: convert base64 to blob URL
              const audioBlob = base64ToBlob(result.audio, 'audio/wav')
              const url = URL.createObjectURL(audioBlob)
              setResultUrl(url)
            }
            setIsConverting(false)

            // Fetch quality metrics after conversion completes
            if (newJobId) {
              setMetricsLoading(true)
              setMetricsError(null)
              try {
                const response = await apiService.getConversionMetrics(newJobId)
                setQualityMetrics(response.metrics)
              } catch (err: any) {
                console.error('Failed to fetch quality metrics:', err)
                setMetricsError('Failed to load quality metrics')
              } finally {
                setMetricsLoading(false)
              }
            }
          },
          onError: (err) => {
            // Handle cancellation specifically
            if (err.code === 'CONVERSION_CANCELLED') {
              setIsConverting(false)
              setProgress(null)
              setError('Conversion cancelled')
            } else {
              // Existing error handling
              setIsConverting(false)
              setError(err.message || err.error || 'Conversion failed')
            }
          },
        }, response.websocket_room)  // ADDED custom room parameter
      } else if (response.status === 'success') {
        // Handle sync response (200 with audio data)
        // Set jobId if provided by backend (needed for download functionality)
        if (response.job_id) {
          setJobId(response.job_id)
        }

        // Extract pitch data if available
        if (response.f0_contour && response.f0_times) {
          setConvertedPitchData({
            f0: response.f0_contour,
            times: response.f0_times
          })
        }

        // Extract quality metrics if available in sync response
        if (response.quality_metrics) {
          setQualityMetrics(response.quality_metrics)
        }

        // Create blob URL from base64 audio
        const audioBlob = base64ToBlob(response.audio, 'audio/wav')
        const url = URL.createObjectURL(audioBlob)
        setResultUrl(url)
        setIsConverting(false)
      }
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || 'Failed to start conversion')
      setIsConverting(false)
    }
  }

  function base64ToBlob(base64: string, mimeType: string): Blob {
    const byteCharacters = atob(base64)
    const byteNumbers = new Array(byteCharacters.length)
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }
    const byteArray = new Uint8Array(byteNumbers)
    return new Blob([byteArray], { type: mimeType })
  }

  const handleDownload = async () => {
    if (!jobId || !resultUrl) return

    try {
      // If we have a blob URL already, download it directly (client-side)
      if (resultUrl.startsWith('blob:')) {
        const a = document.createElement('a')
        a.href = resultUrl
        a.download = `converted_${selectedFile?.name || 'song.wav'}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        return
      }

      // Fallback: fetch from API (shouldn't happen with new flow)
      const blob = await apiService.downloadConvertedAudio(jobId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `converted_${selectedFile?.name || 'song.wav'}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err: any) {
      setError('Failed to download converted audio')
    }
  }

  const handleReset = () => {
    if (jobId) {
      wsService.unsubscribeFromJob(jobId, websocketRoom)
    }
    if (originalAudioUrl) {
      URL.revokeObjectURL(originalAudioUrl)
    }
    if (resultUrl && resultUrl.startsWith('blob:')) {
      URL.revokeObjectURL(resultUrl)
    }
    setSelectedFile(null)
    setSelectedProfile(null)
    setIsConverting(false)
    setJobId(null)
    setWebsocketRoom('')
    setProgress(null)
    setResultUrl(null)
    setError(null)
    setOriginalAudioUrl(null)
    setConvertedPitchData(null)
    setQualityMetrics(null)
    setMetricsError(null)
    setWaveformError(null)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Music className="w-8 h-8 text-primary-600" />
          <span>Singing Voice Conversion</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Transform any song with a different voice while preserving the original pitch, vibrato, and expression
        </p>
      </div>

      {error && (
        <div className="mb-6 bg-red-50 border-2 border-red-500 rounded-lg p-4">
          <p className="text-red-800 font-semibold">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Input and Settings */}
        <div className="space-y-6">
          {/* File Upload */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">1. Upload Song</h2>
            <UploadInterface
              onFileSelect={setSelectedFile}
              selectedFile={selectedFile}
              onClear={() => setSelectedFile(null)}
            />
          </div>

          {/* Voice Profile Selection */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">2. Select Target Voice</h2>
            <VoiceProfileSelector
              profiles={profiles}
              selectedProfile={selectedProfile}
              onSelect={setSelectedProfile}
              isLoading={profilesLoading}
            />
          </div>

          {/* Conversion Settings */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">3. Configure Settings</h2>
            <ConversionControls
              settings={settings}
              onChange={setSettings}
              disabled={isConverting}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-4">
            <button
              onClick={handleStartConversion}
              disabled={!selectedFile || !selectedProfile || isConverting}
              className="flex-1 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <Play className="w-5 h-5" />
              <span>{isConverting ? 'Converting...' : 'Start Conversion'}</span>
            </button>
            {resultUrl && (
              <button
                onClick={handleDownload}
                className="bg-green-600 hover:bg-green-700 text-white font-semibold py-4 px-6 rounded-lg transition-colors flex items-center space-x-2"
              >
                <Download className="w-5 h-5" />
                <span>Download</span>
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Progress and Results */}
        <div className="space-y-6">
          {progress && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <ProgressDisplay
                stages={[{
                  id: progress.stage,
                  name: progress.stage,
                  progress: progress.progress,
                  status: 'processing',
                }]}
                overallProgress={progress.progress}
                estimatedTimeRemaining={progress.estimated_time_remaining}
              />
            </div>
          )}

          {/* Audio Waveform Display - Side by Side */}
          {(originalAudioUrl || resultUrl) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {originalAudioUrl && (
                <AudioWaveform
                  audioUrl={originalAudioUrl}
                  title="Original Song"
                />
              )}
              {resultUrl && (
                <AudioWaveform
                  audioUrl={resultUrl}
                  title="Converted Song"
                  pitchData={convertedPitchData || undefined}
                  isLoading={isConverting}
                  error={waveformError || undefined}
                  onError={(error) => {
                    setWaveformError(error.message)
                  }}
                />
              )}
            </div>
          )}

          {/* Quality Metrics Display */}
          {resultUrl && (
            <QualityMetricsDisplay
              metrics={qualityMetrics}
              isLoading={metricsLoading}
              error={metricsError}
            />
          )}

          {resultUrl && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <button
                onClick={handleReset}
                className="w-full bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                Convert Another Song
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
