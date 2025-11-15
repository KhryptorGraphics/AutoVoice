import { useState, useEffect } from 'react'
import { Music, Download, Play, Pause } from 'lucide-react'
import { UploadInterface } from '../components/SingingConversion/UploadInterface'
import { ConversionControls, ConversionSettings } from '../components/SingingConversion/ConversionControls'
import { ProgressDisplay, PipelineStage } from '../components/SingingConversion/ProgressDisplay'
import { VoiceProfileSelector } from '../components/VoiceProfileSelector'
import { apiService, VoiceProfile } from '../services/api'
import { wsService, ConversionProgress } from '../services/websocket'
import { useQuery } from '@tanstack/react-query'

export function SingingConversionPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedProfile, setSelectedProfile] = useState<VoiceProfile | null>(null)
  const [isConverting, setIsConverting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<ConversionProgress | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

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
        wsService.unsubscribeFromJob(jobId)
      }
    }
  }, [jobId])

  const handleStartConversion = async () => {
    if (!selectedFile || !selectedProfile) {
      setError('Please select both a song file and a voice profile')
      return
    }

    setError(null)
    setIsConverting(true)
    setProgress(null)
    setResultUrl(null)

    try {
      // Start conversion
      const response = await apiService.convertSong(
        selectedFile,
        selectedProfile.id,
        {
          pitch_shift: settings.pitchShift,
          preserve_original_pitch: settings.preserveOriginalPitch,
          preserve_vibrato: settings.preserveVibrato,
          preserve_expression: settings.preserveExpression,
          output_quality: settings.outputQuality,
          denoise_input: settings.denoiseInput,
          enhance_output: settings.enhanceOutput,
        }
      )

      const newJobId = response.job_id
      setJobId(newJobId)

      // Subscribe to WebSocket updates
      wsService.subscribeToJob(newJobId, {
        onProgress: (progressData) => {
          setProgress(progressData)
        },
        onComplete: (result) => {
          setResultUrl(result.output_url)
          setIsConverting(false)
        },
        onError: (err) => {
          setError(err.message)
          setIsConverting(false)
        },
      })
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || 'Failed to start conversion')
      setIsConverting(false)
    }
  }

  const handleDownload = async () => {
    if (!jobId || !resultUrl) return

    try {
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
      wsService.unsubscribeFromJob(jobId)
    }
    setSelectedFile(null)
    setSelectedProfile(null)
    setIsConverting(false)
    setJobId(null)
    setProgress(null)
    setResultUrl(null)
    setError(null)
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
                stages={progress.stages}
                overallProgress={progress.overall_progress}
                estimatedTimeRemaining={progress.estimated_time_remaining}
              />
            </div>
          )}

          {resultUrl && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Converted Audio</h2>
              <audio controls className="w-full" src={resultUrl} />
              <button
                onClick={handleReset}
                className="mt-4 w-full bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
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

