import { useState, useRef } from 'react'
import { Plus, Upload, Music, Loader2, CheckCircle, AlertCircle, Scissors } from 'lucide-react'
import clsx from 'clsx'
import { apiService } from '../services/api'
import { useToastContext } from '../contexts/ToastContext'

interface AddSongButtonProps {
  profileId: string
  onSongAdded?: () => void
  className?: string
}

type UploadStatus = 'idle' | 'uploading' | 'splitting' | 'complete' | 'error'

interface UploadProgress {
  status: UploadStatus
  progress: number
  message: string
  songName?: string
}

export function AddSongButton({ profileId, onSongAdded, className }: AddSongButtonProps) {
  const [uploadState, setUploadState] = useState<UploadProgress>({
    status: 'idle',
    progress: 0,
    message: '',
  })
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const songName = file.name.replace(/\.[^/.]+$/, '') // Remove extension

    try {
      // Upload phase
      setUploadState({
        status: 'uploading',
        progress: 0,
        message: 'Uploading song...',
        songName,
      })

      // Create form data for upload
      const formData = new FormData()
      formData.append('file', file)
      formData.append('profile_id', profileId)
      formData.append('auto_split', 'true')

      // Upload with progress tracking
      const response = await apiService.uploadSongWithSplit(profileId, file, (progress) => {
        if (progress < 100) {
          setUploadState({
            status: 'uploading',
            progress,
            message: `Uploading: ${progress}%`,
            songName,
          })
        } else {
          setUploadState({
            status: 'splitting',
            progress: 0,
            message: 'Separating vocals and instrumental...',
            songName,
          })
        }
      })

      // Poll for separation status
      if (response.job_id) {
        await pollSeparationStatus(response.job_id, songName)
      } else {
        // If no job_id, assume immediate success
        setUploadState({
          status: 'complete',
          progress: 100,
          message: 'Song added successfully!',
          songName,
        })
      }

      // Notify parent
      onSongAdded?.()

      // Reset after delay
      setTimeout(() => {
        setUploadState({ status: 'idle', progress: 0, message: '' })
      }, 2000)

    } catch (error) {
      console.error('Failed to add song:', error)
      setUploadState({
        status: 'error',
        progress: 0,
        message: `Failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        songName,
      })
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const pollSeparationStatus = async (jobId: string, songName: string) => {
    const maxAttempts = 60 // 5 minutes max
    let attempts = 0

    while (attempts < maxAttempts) {
      try {
        const status = await apiService.getSeparationStatus(jobId)

        if (status.status === 'complete') {
          setUploadState({
            status: 'complete',
            progress: 100,
            message: 'Vocals extracted successfully!',
            songName,
          })
          return
        }

        if (status.status === 'error') {
          throw new Error(status.error || 'Separation failed')
        }

        // Update progress
        setUploadState({
          status: 'splitting',
          progress: status.progress || Math.min(attempts * 2, 95),
          message: status.message || 'Separating vocals and instrumental...',
          songName,
        })

        await new Promise(resolve => setTimeout(resolve, 5000))
        attempts++

      } catch (error) {
        throw error instanceof Error ? error : new Error('Separation failed')
      }
    }

    throw new Error('Separation timed out')
  }

  const statusConfig = {
    idle: { icon: Plus, color: 'text-gray-400' },
    uploading: { icon: Upload, color: 'text-blue-400' },
    splitting: { icon: Scissors, color: 'text-yellow-400' },
    complete: { icon: CheckCircle, color: 'text-green-400' },
    error: { icon: AlertCircle, color: 'text-red-400' },
  }

  const { icon: StatusIcon, color } = statusConfig[uploadState.status]

  return (
    <div className={clsx('relative', className)}>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileSelect}
        className="hidden"
        aria-label="Select song file to upload"
      />

      {uploadState.status === 'idle' ? (
        <button
          onClick={() => fileInputRef.current?.click()}
          className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
            'bg-violet-600 hover:bg-violet-500 text-white',
            'focus:outline-none focus:ring-2 focus:ring-violet-500 focus:ring-offset-2 focus:ring-offset-gray-900'
          )}
        >
          <Music className="w-4 h-4" />
          <span>Add Song</span>
        </button>
      ) : (
        <div className={clsx(
          'flex items-center gap-3 px-4 py-2 rounded-lg',
          'bg-gray-800 border border-gray-700'
        )}>
          {uploadState.status === 'uploading' || uploadState.status === 'splitting' ? (
            <Loader2 className={clsx('w-4 h-4 animate-spin', color)} />
          ) : (
            <StatusIcon className={clsx('w-4 h-4', color)} />
          )}

          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-200 truncate">
                {uploadState.songName || 'Song'}
              </span>
              {uploadState.status !== 'error' && uploadState.status !== 'complete' && (
                <span className="text-xs text-gray-500 ml-2">
                  {uploadState.progress}%
                </span>
              )}
            </div>
            <div className="text-xs text-gray-500 truncate">
              {uploadState.message}
            </div>
            {(uploadState.status === 'uploading' || uploadState.status === 'splitting') && (
              <div className="mt-1 h-1 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    'h-full transition-all duration-300 rounded-full',
                    uploadState.status === 'uploading' ? 'bg-blue-500' : 'bg-yellow-500'
                  )}
                  style={{ width: `${uploadState.progress}%` }}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// Compact version for toolbar use
export function AddSongCompact({ profileId, onSongAdded }: AddSongButtonProps) {
  const toast = useToastContext()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    try {
      await apiService.uploadSongWithSplit(profileId, file)
      onSongAdded?.()
    } catch (error) {
      console.error('Failed to add song:', error)
      toast.error(`Failed to add song: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileSelect}
        className="hidden"
        aria-label="Select song file to upload"
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={uploading}
        className={clsx(
          'p-2 rounded-lg transition-colors',
          'bg-gray-800 hover:bg-gray-700 text-gray-300',
          'focus:outline-none focus:ring-2 focus:ring-violet-500',
          uploading && 'opacity-50 cursor-not-allowed'
        )}
        title="Add song with auto-split"
      >
        {uploading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Plus className="w-4 h-4" />
        )}
      </button>
    </>
  )
}
