import { useState, useCallback, useRef } from 'react'
import { Upload, CheckCircle, XCircle, Music } from 'lucide-react'
import clsx from 'clsx'

interface DragDropUploadZoneProps {
  onFileSelect: (file: File) => void
  accept?: string[]
  maxSizeMB?: number
  disabled?: boolean
  className?: string
  selectedFile?: File | null
}

export function DragDropUploadZone({
  onFileSelect,
  accept = ['audio/*', '.mp3', '.wav', '.flac', '.ogg', '.m4a'],
  maxSizeMB,
  disabled = false,
  className = '',
  selectedFile = null,
}: DragDropUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isValidFile, setIsValidFile] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)
  const dragCounter = useRef(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const validateFile = useCallback(
    (file: File): { valid: boolean; error?: string } => {
      // Check file type
      const audioExtensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
      const fileName = file.name.toLowerCase()
      const isAudioType = file.type.startsWith('audio/')
      const hasValidExtension = audioExtensions.some((ext) => fileName.endsWith(ext))

      if (!isAudioType && !hasValidExtension) {
        return {
          valid: false,
          error: 'Invalid file type. Please select an audio file.',
        }
      }

      // Check file size if limit is set
      if (maxSizeMB && file.size > maxSizeMB * 1024 * 1024) {
        return {
          valid: false,
          error: `File too large. Maximum size is ${maxSizeMB}MB.`,
        }
      }

      return { valid: true }
    },
    [maxSizeMB]
  )

  const handleDragEnter = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()

      if (disabled) return

      dragCounter.current++

      if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
        setIsDragging(true)

        // Check if file type is valid during drag
        const item = e.dataTransfer.items[0]
        if (item.kind === 'file') {
          const fileType = item.type
          const isValid = fileType.startsWith('audio/')
          setIsValidFile(isValid)
          if (!isValid) {
            setError('Invalid file type')
          } else {
            setError(null)
          }
        }
      }
    },
    [disabled]
  )

  const handleDragLeave = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()

      if (disabled) return

      dragCounter.current--

      if (dragCounter.current === 0) {
        setIsDragging(false)
        setIsValidFile(null)
        setError(null)
      }
    },
    [disabled]
  )

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
    },
    []
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()

      if (disabled) return

      dragCounter.current = 0
      setIsDragging(false)
      setIsValidFile(null)

      const files = e.dataTransfer.files
      if (files && files.length > 0) {
        const file = files[0]
        const validation = validateFile(file)

        if (validation.valid) {
          setError(null)
          onFileSelect(file)
        } else {
          setError(validation.error || 'Invalid file')
        }
      }
    },
    [disabled, onFileSelect, validateFile]
  )

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (files && files.length > 0) {
        const file = files[0]
        const validation = validateFile(file)

        if (validation.valid) {
          setError(null)
          onFileSelect(file)
        } else {
          setError(validation.error || 'Invalid file')
        }
      }
    },
    [onFileSelect, validateFile]
  )

  const handleClick = useCallback(() => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click()
    }
  }, [disabled])

  const getIcon = () => {
    if (selectedFile) {
      return <Music className="mx-auto h-12 w-12 text-green-500" />
    }

    if (isDragging) {
      if (isValidFile === false) {
        return <XCircle className="mx-auto h-12 w-12 text-red-500" />
      } else if (isValidFile === true) {
        return <CheckCircle className="mx-auto h-12 w-12 text-green-500" />
      } else {
        return <Upload className="mx-auto h-12 w-12 text-blue-500 animate-pulse" />
      }
    }

    return <Upload className="mx-auto h-12 w-12 text-gray-500" />
  }

  const getBorderClass = () => {
    if (disabled) {
      return 'border-gray-700 cursor-not-allowed opacity-50'
    }

    if (selectedFile) {
      return 'border-green-500 bg-green-500/10'
    }

    if (isDragging) {
      if (isValidFile === false) {
        return 'border-red-500 bg-red-500/10'
      } else if (isValidFile === true) {
        return 'border-green-500 bg-green-500/10 scale-105 shadow-lg shadow-green-500/20'
      } else {
        return 'border-blue-500 bg-blue-500/10 scale-105 shadow-lg shadow-blue-500/20'
      }
    }

    return 'border-gray-600 hover:border-gray-500 cursor-pointer'
  }

  const acceptString = accept.join(',')
  const formatBadge = 'MP3, WAV, FLAC, OGG, M4A'

  return (
    <div className={clsx('relative', className)}>
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
        className={clsx(
          'border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ease-in-out',
          getBorderClass()
        )}
      >
        {getIcon()}

        <div className="mt-3 space-y-2">
          {selectedFile ? (
            <>
              <p className="text-white font-medium">{selectedFile.name}</p>
              <p className="text-gray-400 text-sm">
                {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
              </p>
            </>
          ) : (
            <>
              <p className="text-gray-400 mb-3">
                {isDragging
                  ? isValidFile === false
                    ? 'Invalid file type'
                    : 'Drop file here'
                  : 'Drag & drop audio file here, or click to upload'}
              </p>
              <div className="text-xs text-gray-500 space-y-1">
                <p>Accepted formats: {formatBadge}</p>
                {maxSizeMB && <p>Maximum size: {maxSizeMB}MB</p>}
              </div>
            </>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept={acceptString}
          onChange={handleFileInputChange}
          disabled={disabled}
          className="hidden"
        />
      </div>

      {error && (
        <div className="mt-2 flex items-center gap-2 text-sm text-red-400">
          <XCircle size={16} />
          <span>{error}</span>
        </div>
      )}
    </div>
  )
}
