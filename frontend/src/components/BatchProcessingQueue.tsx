import { useState, useCallback } from 'react'
import {
  ListMusic, Upload, Trash2, Play, Pause, X, GripVertical,
  CheckCircle, AlertCircle, Loader2, Clock
} from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService, ConversionConfig } from '../services/api'
import clsx from 'clsx'

interface BatchFile {
  id: string
  file: File
  status: 'pending' | 'queued' | 'processing' | 'complete' | 'error'
  progress: number
  jobId?: string
  error?: string
  resultUrl?: string
  stemUrls?: Partial<Record<'vocals' | 'instrumental', string>>
  reassembleUrl?: string
}

interface BatchProcessingQueueProps {
  profileId: string
  config: ConversionConfig
  onComplete?: (results: BatchFile[]) => void
}

export function BatchProcessingQueue({ profileId, config, onComplete }: BatchProcessingQueueProps) {
  const [files, setFiles] = useState<BatchFile[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null)
  const queryClient = useQueryClient()

  const processMutation = useMutation({
    mutationFn: async (file: File) => {
      return apiService.convertSong(file, profileId, {
        preset: config.preset,
        vocal_volume: config.vocal_volume,
        instrumental_volume: config.instrumental_volume,
        pitch_shift: config.pitch_shift,
        pipeline_type: config.pipeline_type,
        return_stems: config.return_stems,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversionHistory'] })
    },
  })

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      f => f.type.startsWith('audio/') || f.name.match(/\.(mp3|wav|flac|ogg|m4a)$/i)
    )
    addFiles(droppedFiles)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files))
    }
  }

  const addFiles = (newFiles: File[]) => {
    const batchFiles: BatchFile[] = newFiles.map(file => ({
      id: `${file.name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      file,
      status: 'pending',
      progress: 0,
    }))
    setFiles(prev => [...prev, ...batchFiles])
  }

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id))
  }

  const clearCompleted = () => {
    setFiles(prev => prev.filter(f => f.status !== 'complete' && f.status !== 'error'))
  }

  const handleDragStart = (index: number) => {
    setDraggedIndex(index)
  }

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    if (draggedIndex === null || draggedIndex === index) return

    setFiles(prev => {
      const newFiles = [...prev]
      const [removed] = newFiles.splice(draggedIndex, 1)
      newFiles.splice(index, 0, removed)
      return newFiles
    })
    setDraggedIndex(index)
  }

  const handleDragEnd = () => {
    setDraggedIndex(null)
  }

  const processQueue = async () => {
    setIsProcessing(true)
    const pendingFiles = files.filter(f => f.status === 'pending')

    for (const batchFile of pendingFiles) {
      setFiles(prev =>
        prev.map(f =>
          f.id === batchFile.id ? { ...f, status: 'processing', progress: 0 } : f
        )
      )

      try {
        const submission = await processMutation.mutateAsync(batchFile.file)
        if (submission.audio) {
          setFiles(prev =>
            prev.map(f =>
              f.id === batchFile.id
                ? {
                    ...f,
                    status: 'complete',
                    progress: 100,
                    jobId: submission.job_id,
                    stemUrls: submission.stem_urls,
                    reassembleUrl: submission.reassemble_url,
                  }
                : f
            )
          )
          continue
        }
        setFiles(prev =>
          prev.map(f =>
            f.id === batchFile.id
              ? { ...f, status: 'queued', progress: 5, jobId: submission.job_id }
              : f
          )
        )

        let status = await apiService.getConversionStatus(submission.job_id)
        while (status.status === 'queued' || status.status === 'processing' || status.status === 'in_progress') {
          setFiles(prev =>
            prev.map(f =>
              f.id === batchFile.id
                ? {
                    ...f,
                    status: status.status === 'queued' ? 'queued' : 'processing',
                    progress: Math.max(10, status.progress ?? f.progress),
                    jobId: submission.job_id,
                  }
                : f
            )
          )
          await new Promise((resolve) => window.setTimeout(resolve, 1000))
          status = await apiService.getConversionStatus(submission.job_id)
        }

        if (status.status === 'complete' || status.status === 'completed') {
          setFiles(prev =>
            prev.map(f =>
              f.id === batchFile.id
                ? {
                    ...f,
                    status: 'complete',
                    progress: 100,
                    jobId: submission.job_id,
                    resultUrl: status.resultUrl ?? status.output_url ?? status.download_url,
                    stemUrls: status.stem_urls,
                    reassembleUrl: status.reassemble_url,
                  }
                : f
            )
          )
        } else {
          throw new Error(status.error || 'Batch conversion failed')
        }
      } catch (err) {
        setFiles(prev =>
          prev.map(f =>
            f.id === batchFile.id
              ? { ...f, status: 'error', error: (err as Error).message }
              : f
          )
        )
      }
    }

    setIsProcessing(false)
    onComplete?.(
      files.map((item) => item)
    )
  }

  const stopProcessing = () => {
    setIsProcessing(false)
  }

  const pendingCount = files.filter(f => f.status === 'pending').length
  const processingFile = files.find(f => f.status === 'processing')
  const completedCount = files.filter(f => f.status === 'complete').length
  const errorCount = files.filter(f => f.status === 'error').length

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ListMusic size={18} className="text-purple-400" />
          <h3 className="font-semibold">Batch Processing</h3>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <span>{files.length} files</span>
          {completedCount > 0 && (
            <span className="text-green-400">{completedCount} done</span>
          )}
          {errorCount > 0 && (
            <span className="text-red-400">{errorCount} failed</span>
          )}
        </div>
      </div>

      {/* Drop Zone */}
      <div
        onDragOver={e => e.preventDefault()}
        onDrop={handleFileDrop}
        className={clsx(
          'border-2 border-dashed rounded-lg p-6 text-center transition-colors',
          'border-gray-700 hover:border-gray-600'
        )}
      >
        <Upload size={32} className="mx-auto text-gray-500 mb-2" />
        <p className="text-sm text-gray-400 mb-2">
          Drag & drop audio files here, or
        </p>
        <label className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer transition-colors">
          <Upload size={16} />
          Browse Files
          <input
            type="file"
            multiple
            accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a"
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>
      </div>

      {/* File Queue */}
      {files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Queue</span>
            {completedCount > 0 && (
              <button
                onClick={clearCompleted}
                className="text-gray-500 hover:text-white transition-colors"
              >
                Clear completed
              </button>
            )}
          </div>

          <div className="max-h-64 overflow-y-auto space-y-1">
            {files.map((file, index) => (
              <div
                key={file.id}
                draggable={file.status === 'pending' && !isProcessing}
                onDragStart={() => handleDragStart(index)}
                onDragOver={e => handleDragOver(e, index)}
                onDragEnd={handleDragEnd}
                className={clsx(
                  'flex items-center gap-3 p-2 rounded-lg transition-all',
                  file.status === 'processing' && 'bg-blue-900/30 border border-blue-800',
                  file.status === 'complete' && 'bg-green-900/20 border border-green-800/50',
                  file.status === 'error' && 'bg-red-900/20 border border-red-800/50',
                  file.status === 'pending' && 'bg-gray-750 hover:bg-gray-700',
                  draggedIndex === index && 'opacity-50'
                )}
              >
                {/* Drag Handle */}
                {file.status === 'pending' && !isProcessing && (
                  <GripVertical size={14} className="text-gray-500 cursor-grab" />
                )}

                {/* Status Icon */}
                {(file.status === 'pending' || file.status === 'queued') && <Clock size={14} className="text-gray-500" />}
                {file.status === 'processing' && (
                  <Loader2 size={14} className="text-blue-400 animate-spin" />
                )}
                {file.status === 'complete' && (
                  <CheckCircle size={14} className="text-green-400" />
                )}
                {file.status === 'error' && (
                  <AlertCircle size={14} className="text-red-400" />
                )}

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <div className="text-sm truncate">{file.file.name}</div>
                  {(file.status === 'processing' || file.status === 'queued') && (
                    <div className="h-1 bg-gray-700 rounded-full mt-1 overflow-hidden">
                      <div
                        className="h-full bg-blue-500 transition-all"
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>
                  )}
                  {file.error && (
                    <div className="text-xs text-red-400 truncate">{file.error}</div>
                  )}
                  {file.jobId && (
                    <div className="text-xs text-gray-500 truncate">Job {file.jobId}</div>
                  )}
                </div>

                {/* Size */}
                <span className="text-xs text-gray-500">
                  {(file.file.size / 1024 / 1024).toFixed(1)} MB
                </span>

                {/* Remove Button */}
                {file.status !== 'processing' && (
                  <button
                    onClick={() => removeFile(file.id)}
                    className="p-1 text-gray-500 hover:text-red-400 transition-colors"
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Progress Summary */}
      {isProcessing && processingFile && (
        <div className="bg-gray-750 rounded-lg p-3">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-gray-400">Processing</span>
            <span>{completedCount + 1} / {files.length}</span>
          </div>
          <div className="text-sm truncate mb-1">{processingFile.file.name}</div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${((completedCount + errorCount) / files.length) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2">
        {!isProcessing ? (
          <button
            onClick={processQueue}
            disabled={pendingCount === 0}
            className={clsx(
              'flex-1 flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-all',
              pendingCount > 0
                ? 'bg-purple-600 hover:bg-purple-700 text-white'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            )}
          >
            <Play size={18} />
            Process {pendingCount} Files
          </button>
        ) : (
          <button
            onClick={stopProcessing}
            className="flex-1 flex items-center justify-center gap-2 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-all"
          >
            <Pause size={18} />
            Stop Processing
          </button>
        )}

        {files.length > 0 && !isProcessing && (
          <button
            onClick={() => setFiles([])}
            className="px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            <Trash2 size={18} />
          </button>
        )}
      </div>
    </div>
  )
}
