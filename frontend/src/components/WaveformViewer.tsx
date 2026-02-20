import { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, Volume2, VolumeX, ZoomIn, ZoomOut } from 'lucide-react'
import clsx from 'clsx'

interface WaveformViewerProps {
  audioUrl?: string
  height?: number
  color?: string
  showPlaybackControls?: boolean
  onTimeUpdate?: (currentTime: number) => void
}

export function WaveformViewer({
  audioUrl,
  height = 100,
  color = '#3b82f6',
  showPlaybackControls = true,
  onTimeUpdate,
}: WaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [waveformData, setWaveformData] = useState<number[]>([])
  const [loading, setLoading] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)
  const [zoom, setZoom] = useState(1)

  const extractWaveform = useCallback(async (url: string) => {
    setLoading(true)
    try {
      const audioContext = new AudioContext()
      const response = await fetch(url)
      const arrayBuffer = await response.arrayBuffer()
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

      const channelData = audioBuffer.getChannelData(0)
      const samples = 1000 // Number of samples for visualization
      const blockSize = Math.floor(channelData.length / samples)
      const peaks: number[] = []

      for (let i = 0; i < samples; i++) {
        let max = 0
        for (let j = 0; j < blockSize; j++) {
          const idx = i * blockSize + j
          if (idx < channelData.length) {
            max = Math.max(max, Math.abs(channelData[idx]))
          }
        }
        peaks.push(max)
      }

      await audioContext.close()
      setWaveformData(peaks)
      setLoading(false)
    } catch (err) {
      console.error('Waveform extraction failed:', err)
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (audioUrl) {
      extractWaveform(audioUrl)
    }
  }, [audioUrl, extractWaveform])

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || waveformData.length === 0) return

    const ctx = canvas.getContext('2d')!
    const width = canvas.offsetWidth
    const canvasHeight = height

    canvas.width = width * window.devicePixelRatio
    canvas.height = canvasHeight * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    ctx.clearRect(0, 0, width, canvasHeight)

    const barWidth = (width / waveformData.length) * zoom
    const progressX = duration > 0 ? (currentTime / duration) * width * zoom : 0

    waveformData.forEach((peak, i) => {
      const x = i * barWidth
      const barHeight = peak * canvasHeight * 0.9

      // Determine if this bar is before or after the playhead
      const isPast = x < progressX

      ctx.fillStyle = isPast ? color : `${color}40`
      ctx.fillRect(
        x,
        (canvasHeight - barHeight) / 2,
        Math.max(1, barWidth - 1),
        barHeight
      )
    })

    // Draw playhead
    if (duration > 0) {
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(progressX - 1, 0, 2, canvasHeight)
    }
  }, [waveformData, currentTime, duration, height, color, zoom])

  useEffect(() => {
    drawWaveform()
  }, [drawWaveform])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime)
      onTimeUpdate?.(audio.currentTime)
    }

    const handleDurationChange = () => {
      setDuration(audio.duration)
    }

    const handleEnded = () => {
      setIsPlaying(false)
    }

    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('durationchange', handleDurationChange)
    audio.addEventListener('ended', handleEnded)

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('durationchange', handleDurationChange)
      audio.removeEventListener('ended', handleEnded)
    }
  }, [onTimeUpdate])

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return

    if (isPlaying) {
      audio.pause()
    } else {
      audio.play()
    }
    setIsPlaying(!isPlaying)
  }

  const toggleMute = () => {
    const audio = audioRef.current
    if (!audio) return

    audio.muted = !isMuted
    setIsMuted(!isMuted)
  }

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current
    if (!audio) return

    const newVolume = parseFloat(e.target.value)
    audio.volume = newVolume
    setVolume(newVolume)
    setIsMuted(newVolume === 0)
  }

  const handleSeek = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const audio = audioRef.current
    const canvas = canvasRef.current
    if (!audio || !canvas || duration === 0) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = x / rect.width / zoom
    audio.currentTime = percentage * duration
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  if (!audioUrl) {
    return (
      <div
        className="bg-gray-800 rounded-lg flex items-center justify-center text-gray-500"
        style={{ height }}
      >
        No audio loaded
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Hidden audio element */}
      <audio ref={audioRef} src={audioUrl} preload="metadata" />

      {/* Waveform canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          className={clsx(
            'w-full cursor-pointer',
            loading && 'opacity-50'
          )}
          style={{ height }}
          onClick={handleSeek}
        />

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-gray-400 text-sm">Loading waveform...</div>
          </div>
        )}
      </div>

      {/* Playback controls */}
      {showPlaybackControls && (
        <div className="flex items-center gap-3 p-3 border-t border-gray-700">
          <button
            onClick={togglePlay}
            className="p-2 bg-blue-600 hover:bg-blue-700 rounded-full transition-colors"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
          </button>

          <div className="flex items-center gap-2 text-sm font-mono text-gray-400">
            <span>{formatTime(currentTime)}</span>
            <span>/</span>
            <span>{formatTime(duration)}</span>
          </div>

          <div className="flex-1" />

          {/* Volume controls */}
          <button
            onClick={toggleMute}
            className="p-1.5 text-gray-400 hover:text-white rounded"
          >
            {isMuted || volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="w-20 h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
          />

          {/* Zoom controls */}
          <div className="flex items-center gap-1 ml-2 border-l border-gray-700 pl-2">
            <button
              onClick={() => setZoom(z => Math.max(0.5, z * 0.8))}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            >
              <ZoomOut size={14} />
            </button>
            <span className="text-xs text-gray-500 w-10 text-center">
              {(zoom * 100).toFixed(0)}%
            </span>
            <button
              onClick={() => setZoom(z => Math.min(4, z * 1.25))}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            >
              <ZoomIn size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
