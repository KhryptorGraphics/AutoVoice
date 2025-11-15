import { useState } from 'react'
import { Music, Upload, Play, Download, Trash2, Plus } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { apiService, VoiceProfile } from '../services/api'
import { VoiceProfileSelector } from '../components/VoiceProfileSelector'

interface BatchFile {
  id: string
  file: File
  status: 'pending' | 'processing' | 'complete' | 'error'
  progress: number
  error?: string
  resultUrl?: string
}

export function BatchConversionPage() {
  const [files, setFiles] = useState<BatchFile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<VoiceProfile | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

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
    setFiles(files.filter((f) => f.id !== id))
  }

  const handleStartBatch = async () => {
    if (!selectedProfile || files.length === 0) return

    setIsProcessing(true)
    for (const file of files) {
      if (file.status !== 'pending') continue

      try {
        setFiles((prev) =>
          prev.map((f) => (f.id === file.id ? { ...f, status: 'processing' as const } : f))
        )

        const response = await apiService.convertSong(
          file.file,
          selectedProfile.id,
          {
            outputQuality: 'balanced',
            preserveOriginalPitch: true,
            preserveVibrato: true,
            preserveExpression: true,
          },
          (progress) => {
            setFiles((prev) =>
              prev.map((f) => (f.id === file.id ? { ...f, progress } : f))
            )
          }
        )

        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id
              ? {
                  ...f,
                  status: 'complete' as const,
                  progress: 100,
                  resultUrl: response.output_url,
                }
              : f
          )
        )
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
      }
    }
    setIsProcessing(false)
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
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Music className="w-8 h-8 text-primary-600" />
          <span>Batch Conversion</span>
        </h1>
        <p className="text-gray-600 mt-2">Convert multiple songs at once</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Settings */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Settings</h2>
            <VoiceProfileSelector
              profiles={profiles}
              selectedProfile={selectedProfile}
              onSelect={setSelectedProfile}
              isLoading={profilesLoading}
            />
            <button
              onClick={handleStartBatch}
              disabled={!selectedProfile || files.length === 0 || isProcessing}
              className="mt-6 w-full bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <Play className="w-5 h-5" />
              <span>{isProcessing ? 'Processing...' : 'Start Batch'}</span>
            </button>
          </div>
        </div>

        {/* File List */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Files ({files.length})</h2>
              <label className="cursor-pointer">
                <input
                  type="file"
                  multiple
                  accept="audio/*"
                  onChange={(e) => e.target.files && handleAddFiles(e.target.files)}
                  className="hidden"
                />
                <div className="flex items-center space-x-2 bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg">
                  <Plus className="w-5 h-5" />
                  <span>Add Files</span>
                </div>
              </label>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {files.map((file) => (
                <div key={file.id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900 truncate">{file.file.name}</span>
                    <button
                      onClick={() => handleRemoveFile(file.id)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        file.status === 'complete'
                          ? 'bg-green-500'
                          : file.status === 'error'
                          ? 'bg-red-500'
                          : 'bg-primary-500'
                      }`}
                      style={{ width: `${file.progress}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-sm text-gray-600">{file.status}</span>
                    {file.status === 'complete' && file.resultUrl && (
                      <a
                        href={file.resultUrl}
                        download
                        className="text-primary-600 hover:text-primary-700 flex items-center space-x-1"
                      >
                        <Download className="w-4 h-4" />
                        <span>Download</span>
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {files.length > 0 && files.some((f) => f.status === 'complete') && (
              <button
                onClick={handleDownloadAll}
                className="mt-6 w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
              >
                <Download className="w-5 h-5" />
                <span>Download All</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

