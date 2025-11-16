import { useState } from 'react'
import { Music, Play, Download, Trash2, Plus, Pause, Play as PlayIcon, X } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { apiService, VoiceProfile } from '../services/api'
import { VoiceProfileSelector } from '../components/VoiceProfileSelector'
import { AdvancedConversionSettings } from '../components/AdvancedConversionSettings'
import { RealtimeWaveform } from '../components/RealtimeWaveform'

interface BatchFile {
  id: string
  file: File
  status: 'pending' | 'processing' | 'complete' | 'error' | 'paused'
  progress: number
  error?: string
  resultUrl?: string
  processingTime?: number
}

export function BatchConversionPage() {
  const [files, setFiles] = useState<BatchFile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<VoiceProfile | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentProcessingFile, setCurrentProcessingFile] = useState<string | null>(null)
  const [batchSettings, setBatchSettings] = useState({
    outputQuality: 'balanced' as const,
    preserveOriginalPitch: true,
    preserveVibrato: true,
    preserveExpression: true,
  })
  const [stats, setStats] = useState({
    totalFiles: 0,
    completed: 0,
    failed: 0,
    totalTime: 0,
  })

  const { data: profiles = [], isLoading: profilesLoading } = useQuery({
    queryKey: ['voiceProfiles'],
    queryFn: () => apiService.getVoiceProfiles(),
  })

  const handleAddFiles = (newFiles: FileList) => {
    const newBatchFiles = Array.from(newFiles).map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'pending' as const,
      progress: 0,
    }))
    setFiles([...files, ...newBatchFiles])
  }

  const handleRemoveFile = (id: string) => {
    if (currentProcessingFile === id && isProcessing) {
      alert('Cannot remove file while processing')
      return
    }
    setFiles(files.filter((f) => f.id !== id))
  }

  const handleCancelFile = (id: string) => {
    setFiles((prev) =>
      prev.map((f) => (f.id === id ? { ...f, status: 'pending' as const, progress: 0 } : f))
    )
  }

  const handleStartBatch = async () => {
    if (!selectedProfile || files.length === 0) return

    setIsProcessing(true)
    setIsPaused(false)
    setStats({ totalFiles: files.length, completed: 0, failed: 0, totalTime: 0 })

    const startTime = Date.now()

    for (const file of files) {
      if (file.status !== 'pending') continue

      // Check if paused
      while (isPaused && isProcessing) {
        await new Promise((resolve) => setTimeout(resolve, 100))
      }

      if (!isProcessing) break // User cancelled

      try {
        setCurrentProcessingFile(file.id)
        setFiles((prev) =>
          prev.map((f) => (f.id === file.id ? { ...f, status: 'processing' as const } : f))
        )

        const response = await apiService.convertSong(
          file.file,
          selectedProfile.id,
          batchSettings,
          (progress) => {
            setFiles((prev) =>
              prev.map((f) => (f.id === file.id ? { ...f, progress } : f))
            )
          }
        )

        const processingTime = Date.now() - startTime

        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id
              ? {
                  ...f,
                  status: 'complete' as const,
                  progress: 100,
                  resultUrl: response.output_url,
                  processingTime,
                }
              : f
          )
        )

        setStats((prev) => ({
          ...prev,
          completed: prev.completed + 1,
          totalTime: processingTime,
        }))
      } catch (error: any) {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id
              ? {
                  ...f,
                  status: 'error' as const,
                  error: error.message,
                }
              : f
          )
        )

        setStats((prev) => ({
          ...prev,
          failed: prev.failed + 1,
        }))
      }
    }

    setCurrentProcessingFile(null)
    setIsProcessing(false)
  }

  const handlePauseBatch = () => {
    setIsPaused(!isPaused)
  }

  const handleCancelBatch = () => {
    if (confirm('Cancel all pending conversions?')) {
      setIsProcessing(false)
      setIsPaused(false)
      setCurrentProcessingFile(null)
      setFiles((prev) =>
        prev.map((f) => (f.status === 'processing' ? { ...f, status: 'pending' as const, progress: 0 } : f))
      )
    }
  }

  const handleDownloadAll = async () => {
    for (const file of files) {
      if (file.status === 'complete' && file.resultUrl) {
        const response = await fetch(file.resultUrl)
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `converted_${file.file.name}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Music className="w-8 h-8 text-primary-600" />
          <span>Batch Conversion</span>
        </h1>
        <p className="text-gray-600 mt-2">Convert multiple songs at once with advanced settings</p>
      </div>

      {/* Real-time Waveform */}
      {isProcessing && (
        <div className="mb-6">
          <RealtimeWaveform isProcessing={isProcessing} progress={stats.completed * (100 / stats.totalFiles)} />
        </div>
      )}

      {/* Stats */}
      {isProcessing && (
        <div className="mb-6 grid grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-600">Total Files</p>
            <p className="text-2xl font-bold text-gray-900">{stats.totalFiles}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-600">Completed</p>
            <p className="text-2xl font-bold text-green-600">{stats.completed}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-600">Failed</p>
            <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-600">Time Elapsed</p>
            <p className="text-2xl font-bold text-gray-900">{Math.round(stats.totalTime / 1000)}s</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Settings */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Voice Profile</h2>
            <VoiceProfileSelector
              profiles={profiles}
              selectedProfile={selectedProfile}
              onSelect={setSelectedProfile}
              isLoading={profilesLoading}
            />
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Conversion Settings</h2>
            <AdvancedConversionSettings
              settings={batchSettings}
              onSettingsChange={setBatchSettings}
            />
          </div>

          {/* Control Buttons */}
          <div className="space-y-2">
            <button
              onClick={handleStartBatch}
              disabled={!selectedProfile || files.length === 0 || isProcessing}
              className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <PlayIcon className="w-5 h-5" />
              <span>{isProcessing ? 'Processing...' : 'Start Batch'}</span>
            </button>

            {isProcessing && (
              <>
                <button
                  onClick={handlePauseBatch}
                  className={`w-full font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2 ${
                    isPaused
                      ? 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-yellow-600 hover:bg-yellow-700 text-white'
                  }`}
                >
                  {isPaused ? (
                    <>
                      <PlayIcon className="w-5 h-5" />
                      <span>Resume</span>
                    </>
                  ) : (
                    <>
                      <Pause className="w-5 h-5" />
                      <span>Pause</span>
                    </>
                  )}
                </button>

                <button
                  onClick={handleCancelBatch}
                  className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
                >
                  <X className="w-5 h-5" />
                  <span>Cancel</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* File List */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Files ({files.length})</h2>
              <label className="cursor-pointer">
                <input
                  type="file"
                  multiple
                  accept="audio/*"
                  onChange={(e) => e.target.files && handleAddFiles(e.target.files)}
                  disabled={isProcessing}
                  className="hidden"
                />
                <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  isProcessing
                    ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                    : 'bg-primary-600 hover:bg-primary-700 text-white cursor-pointer'
                }`}>
                  <Plus className="w-5 h-5" />
                  <span>Add Files</span>
                </div>
              </label>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {files.length === 0 ? (
                <div className="text-center py-12">
                  <Music className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No files added yet</p>
                  <p className="text-gray-400 text-sm mt-2">Click "Add Files" to get started</p>
                </div>
              ) : (
                files.map((file) => (
                  <div
                    key={file.id}
                    className={`border rounded-lg p-4 transition-colors ${
                      currentProcessingFile === file.id
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex-1">
                        <span className="font-medium text-gray-900 truncate block">{file.file.name}</span>
                        <span className="text-xs text-gray-500">
                          {(file.file.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span
                          className={`text-xs font-medium px-2 py-1 rounded ${
                            file.status === 'complete'
                              ? 'bg-green-100 text-green-800'
                              : file.status === 'error'
                              ? 'bg-red-100 text-red-800'
                              : file.status === 'processing'
                              ? 'bg-blue-100 text-blue-800'
                              : file.status === 'paused'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-gray-100 text-gray-800'
                          }`}
                        >
                          {file.status.charAt(0).toUpperCase() + file.status.slice(1)}
                        </span>
                        {file.status === 'processing' && (
                          <button
                            onClick={() => handleCancelFile(file.id)}
                            className="p-1 text-red-600 hover:bg-red-50 rounded transition-colors"
                            title="Cancel"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        )}
                        {file.status !== 'processing' && file.status !== 'complete' && (
                          <button
                            onClick={() => handleRemoveFile(file.id)}
                            className="p-1 text-red-600 hover:bg-red-50 rounded transition-colors"
                            title="Remove"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div
                        className={`h-2 rounded-full transition-all ${
                          file.status === 'complete'
                            ? 'bg-green-500'
                            : file.status === 'error'
                            ? 'bg-red-500'
                            : file.status === 'paused'
                            ? 'bg-yellow-500'
                            : 'bg-primary-500'
                        }`}
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>

                    {/* Status Info */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">{file.progress}%</span>
                      <div className="flex items-center space-x-2">
                        {file.error && (
                          <span className="text-xs text-red-600">{file.error}</span>
                        )}
                        {file.processingTime && (
                          <span className="text-xs text-gray-500">
                            {(file.processingTime / 1000).toFixed(1)}s
                          </span>
                        )}
                        {file.status === 'complete' && file.resultUrl && (
                          <>
                            <button
                              onClick={() => {
                                const audio = new Audio(file.resultUrl)
                                audio.play()
                              }}
                              className="p-1 text-primary-600 hover:bg-primary-50 rounded transition-colors"
                              title="Play"
                            >
                              <Play className="w-4 h-4" />
                            </button>
                            <a
                              href={file.resultUrl}
                              download={`converted_${file.file.name}`}
                              className="p-1 text-green-600 hover:bg-green-50 rounded transition-colors"
                              title="Download"
                            >
                              <Download className="w-4 h-4" />
                            </a>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            {files.length > 0 && files.some((f) => f.status === 'complete') && (
              <button
                onClick={handleDownloadAll}
                className="mt-6 w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
              >
                <Download className="w-5 h-5" />
                <span>Download All Completed</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

