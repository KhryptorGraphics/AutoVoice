import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { Play, Pause, Volume2, VolumeX, Loader2, AlertCircle } from 'lucide-react'
import WaveSurfer from 'wavesurfer.js'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface AudioWaveformProps {
  audioUrl: string
  title?: string
  pitchData?: { f0: number[], times: number[] }
  isLoading?: boolean
  error?: string
  onError?: (error: Error) => void
}

export function AudioWaveform({
  audioUrl,
  title,
  pitchData,
  isLoading = false,
  error,
  onError
}: AudioWaveformProps) {
  const waveformRef = useRef<HTMLDivElement>(null)
  const wavesurfer = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [internalLoading, setInternalLoading] = useState(false)
  const [internalError, setInternalError] = useState<string | null>(null)

  useEffect(() => {
    // Reset playback-related state when audioUrl changes
    setIsPlaying(false)
    setIsMuted(false)
    setCurrentTime(0)
    setDuration(0)
    setInternalError(null)
    setInternalLoading(false)
    // Volume is preserved across audio changes for better UX

    // Skip if no audioUrl or no container
    if (!audioUrl || !waveformRef.current) {
      wavesurfer.current?.destroy()
      wavesurfer.current = null
      return
    }

    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#9333ea',
      progressColor: '#7c3aed',
      cursorColor: '#6d28d9',
      barWidth: 2,
      barRadius: 3,
      cursorWidth: 1,
      height: 80,
      barGap: 2,
    })

    // Set initial volume to match React state
    wavesurfer.current.setVolume(volume)

    // Enhanced error handling
    try {
      setInternalLoading(true)
      wavesurfer.current.load(audioUrl)
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to load audio')
      setInternalError(error.message)
      if (onError) onError(error)
      setInternalLoading(false)
    }

    // Loading event listeners
    wavesurfer.current.on('loading', (percent) => {
      setInternalLoading(percent < 100)
    })

    wavesurfer.current.on('ready', () => {
      setDuration(wavesurfer.current?.getDuration() || 0)
      // Sync volume with React state in ready event
      wavesurfer.current?.setVolume(volume)
      setInternalLoading(false)
    })

    // Error event listener
    wavesurfer.current.on('error', (err) => {
      const error = err instanceof Error ? err : new Error('Wavesurfer error')
      setInternalError(error.message)
      if (onError) onError(error)
      setInternalLoading(false)
    })

    wavesurfer.current.on('audioprocess', () => {
      setCurrentTime(wavesurfer.current?.getCurrentTime() || 0)
    })

    wavesurfer.current.on('play', () => setIsPlaying(true))
    wavesurfer.current.on('pause', () => setIsPlaying(false))

    return () => {
      wavesurfer.current?.destroy()
    }
  }, [audioUrl, onError])

  const handlePlayPause = () => {
    wavesurfer.current?.playPause()
  }

  const handleMute = () => {
    if (wavesurfer.current) {
      const newMuted = !isMuted
      wavesurfer.current.setVolume(newMuted ? 0 : volume)
      setIsMuted(newMuted)
    }
  }

  const handleVolumeChange = (e: ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value)
    setVolume(newVolume)
    if (wavesurfer.current && !isMuted) {
      wavesurfer.current.setVolume(newVolume)
    }
  }

  const handleRetry = () => {
    setInternalError(null)
    if (wavesurfer.current) {
      try {
        setInternalLoading(true)
        wavesurfer.current.load(audioUrl)
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Failed to load audio')
        setInternalError(error.message)
        if (onError) onError(error)
        setInternalLoading(false)
      }
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const displayError = error || internalError
  const displayLoading = isLoading || internalLoading

  // Down-sample pitch data for performance if too dense (max 5000 points for visualization)
  const MAX_PITCH_POINTS = 5000;
  let sampledLen = 0;
  let sampledData: Array<{x: number, y: number}> = [];

  if (pitchData) {
    const rawLen = Math.min(pitchData.f0.length, pitchData.times.length);
    if (rawLen > 0) {
      if (rawLen > MAX_PITCH_POINTS) {
        const step = Math.ceil(rawLen / MAX_PITCH_POINTS);
        sampledLen = Math.floor(rawLen / step);
        sampledData = Array.from({ length: sampledLen }, (_, i) => ({
          x: pitchData.times[i * step],
          y: pitchData.f0[i * step]
        }));
      } else {
        sampledLen = rawLen;
        sampledData = Array.from({ length: sampledLen }, (_, i) => ({
          x: pitchData.times[i],
          y: pitchData.f0[i]
        }));
      }
    }
  }

  const pitchChartData = sampledLen > 0 ? {
    datasets: [
      {
        label: 'Pitch (Hz)',
        data: sampledData,
        borderColor: 'rgba(124, 58, 237, 0.6)',
        backgroundColor: 'rgba(124, 58, 237, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.4,
        pointRadius: 0,
      },
    ],
  } : null

  // Calculate max time safely
  const maxTime = sampledLen > 0 && pitchData?.times && pitchData.times[sampledLen - 1] !== undefined ? pitchData.times[sampledLen - 1] : 1

  const pitchChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
        callbacks: {
          label: (context: any) => `${context.parsed.y.toFixed(1)} Hz`,
        },
      },
    },
    scales: {
      x: {
        type: 'linear' as const,
        display: false,
        min: pitchData && pitchData.times && pitchData.times.length > 0 ? pitchData.times[0] : 0,
        max: maxTime,
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'Hz',
          font: {
            size: 10,
          },
        },
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          font: {
            size: 10,
          },
        },
      },
    },
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {title && <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>}

      {/* Loading State */}
      {displayLoading && (
        <div className="mb-4 bg-gray-200 rounded-lg h-20 animate-pulse flex items-center justify-center">
          <div className="flex items-center space-x-2 text-gray-600">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Loading waveform...</span>
          </div>
        </div>
      )}

      {/* Error State */}
      {displayError && (
        <div className="mb-4 bg-red-50 border-2 border-red-500 rounded-lg p-4">
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-5 h-5 text-red-800 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-semibold text-red-800">Failed to load audio</h4>
              <p className="text-sm text-red-700 mt-1">{displayError}</p>
              <button
                onClick={handleRetry}
                className="mt-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Waveform Container */}
      {!displayLoading && !displayError && (
        <div className="relative mb-4">
          <div ref={waveformRef} />

          {/* Pitch Contour Overlay */}
          {pitchData && pitchChartData && (
            <div className="absolute top-0 left-0 right-0" style={{ height: '80px', pointerEvents: 'none' }}>
              <Line data={pitchChartData} options={pitchChartOptions} />
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      {!displayError && (
        <div className="flex items-center space-x-4">
          <button
            onClick={handlePlayPause}
            disabled={displayLoading}
            className="p-3 bg-primary-600 hover:bg-primary-700 text-white rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>

          <div className="flex-1">
            <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
            <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-600 transition-all"
                style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
              />
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleMute}
              disabled={displayLoading}
              className="p-2 text-gray-600 hover:text-gray-900 transition-colors disabled:opacity-50"
            >
              {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              disabled={displayLoading}
              className="w-24"
            />
          </div>
        </div>
      )}
    </div>
  )
}
