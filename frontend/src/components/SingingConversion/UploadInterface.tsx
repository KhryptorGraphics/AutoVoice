import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Music, FileAudio, X } from 'lucide-react'
import clsx from 'clsx'

interface UploadInterfaceProps {
  onFileSelect: (file: File) => void
  selectedFile: File | null
  onClear: () => void
}

export function UploadInterface({ onFileSelect, selectedFile, onClear }: UploadInterfaceProps) {
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(null)
    
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0]
      if (rejection.errors[0]?.code === 'file-too-large') {
        setError('File is too large. Maximum size is 100MB.')
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        setError('Invalid file type. Please upload MP3, WAV, FLAC, or OGG files.')
      } else {
        setError('Failed to upload file. Please try again.')
      }
      return
    }

    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0])
    }
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/mpeg': ['.mp3'],
      'audio/wav': ['.wav'],
      'audio/flac': ['.flac'],
      'audio/ogg': ['.ogg'],
      'audio/x-m4a': ['.m4a'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false,
  })

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div className="space-y-4">
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={clsx(
            'border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all',
            'hover:border-primary-500 hover:bg-primary-50/50',
            isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300',
            error && 'border-red-500'
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center space-y-4">
            <div className={clsx(
              'p-4 rounded-full',
              isDragActive ? 'bg-primary-100' : 'bg-gray-100'
            )}>
              {isDragActive ? (
                <Upload className="w-12 h-12 text-primary-600" />
              ) : (
                <Music className="w-12 h-12 text-gray-400" />
              )}
            </div>
            
            <div>
              <p className="text-lg font-semibold text-gray-700">
                {isDragActive ? 'Drop your song here' : 'Drag & drop your song file'}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                or click to browse
              </p>
            </div>

            <div className="text-xs text-gray-400 space-y-1">
              <p>Supported formats: MP3, WAV, FLAC, OGG, M4A</p>
              <p>Maximum file size: 100MB</p>
              <p>Sample rates: 16kHz - 48kHz</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="border-2 border-primary-500 rounded-lg p-6 bg-primary-50/30">
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-4 flex-1">
              <div className="p-3 bg-primary-100 rounded-lg">
                <FileAudio className="w-8 h-8 text-primary-600" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-semibold text-gray-900 truncate">
                  {selectedFile.name}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {formatFileSize(selectedFile.size)} • {selectedFile.type || 'audio file'}
                </p>
                <div className="mt-2 flex items-center space-x-2">
                  <div className="h-2 flex-1 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-primary-500 w-full"></div>
                  </div>
                  <span className="text-xs text-gray-500">Ready</span>
                </div>
              </div>
            </div>
            <button
              onClick={onClear}
              className="ml-4 p-2 hover:bg-red-100 rounded-lg transition-colors"
              title="Remove file"
            >
              <X className="w-5 h-5 text-red-600" />
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}
    </div>
  )
}

