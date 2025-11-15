import { useState, useRef, useEffect } from 'react'
import { Play, Pause, RotateCcw, Volume2 } from 'lucide-react'
import { AudioWaveform } from './AudioWaveform'

interface ABComparisonProps {
  originalAudio: string | Blob // URL or Blob
  convertedAudio: string | Blob // URL or Blob
  originalLabel?: string
  convertedLabel?: string
  className?: string
}

export function ABComparison({
  originalAudio,
  convertedAudio,
  originalLabel = 'Original',
  convertedLabel = 'Converted',
  className = '',
}: ABComparisonProps) {
  const [activeTrack, setActiveTrack] = useState<'original' | 'converted'>('original')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1.0)

  const originalAudioRef = useRef<HTMLAudioElement>(null)
  const convertedAudioRef = useRef<HTMLAudioElement>(null)

  useEffect(() => {
    const originalEl = originalAudioRef.current
    const convertedEl = convertedAudioRef.current

    if (!originalEl || !convertedEl) return

    const handleTimeUpdate = () => {
      const activeEl = activeTrack === 'original' ? originalEl : convertedEl
      setCurrentTime(activeEl.currentTime)
    }

    const handleLoadedMetadata = () => {
      const activeEl = activeTrack === 'original' ? originalEl : convertedEl
      setDuration(activeEl.duration)
    }

    const handleEnded = () => {
      setIsPlaying(false)
      setCurrentTime(0)
    }

    originalEl.addEventListener('timeupdate', handleTimeUpdate)
    convertedEl.addEventListener('timeupdate', handleTimeUpdate)
    originalEl.addEventListener('loadedmetadata', handleLoadedMetadata)
    convertedEl.addEventListener('loadedmetadata', handleLoadedMetadata)
    originalEl.addEventListener('ended', handleEnded)
    convertedEl.addEventListener('ended', handleEnded)

    return () => {
      originalEl.removeEventListener('timeupdate', handleTimeUpdate)
      convertedEl.removeEventListener('timeupdate', handleTimeUpdate)
      originalEl.removeEventListener('loadedmetadata', handleLoadedMetadata)
      convertedEl.removeEventListener('loadedmetadata', handleLoadedMetadata)
      originalEl.removeEventListener('ended', handleEnded)
      convertedEl.removeEventListener('ended', handleEnded)
    }
  }, [activeTrack])

  const togglePlayPause = () => {
    const originalEl = originalAudioRef.current
    const convertedEl = convertedAudioRef.current

    if (!originalEl || !convertedEl) return

    if (isPlaying) {
      originalEl.pause()
      convertedEl.pause()
      setIsPlaying(false)
    } else {
      const activeEl = activeTrack === 'original' ? originalEl : convertedEl
      activeEl.play()
      setIsPlaying(true)
    }
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = Number(e.target.value)
    const originalEl = originalAudioRef.current
    const convertedEl = convertedAudioRef.current

    if (originalEl) originalEl.currentTime = newTime
    if (convertedEl) convertedEl.currentTime = newTime
    setCurrentTime(newTime)
  }

  const handleReset = () => {
    const originalEl = originalAudioRef.current
    const convertedEl = convertedAudioRef.current

    if (originalEl) originalEl.currentTime = 0
    if (convertedEl) convertedEl.currentTime = 0
    setCurrentTime(0)
    setIsPlaying(false)
  }

  const switchTrack = (track: 'original' | 'converted') => {
    const wasPlaying = isPlaying
    
    // Pause current track
    if (originalAudioRef.current) originalAudioRef.current.pause()
    if (convertedAudioRef.current) convertedAudioRef.current.pause()
    
    setActiveTrack(track)
    setIsPlaying(false)

    // If was playing, start new track
    if (wasPlaying) {
      setTimeout(() => {
        const newEl = track === 'original' ? originalAudioRef.current : convertedAudioRef.current
        if (newEl) {
          newEl.play()
          setIsPlaying(true)
        }
      }, 100)
    }
  }

  const handleVolumeChange = (newVolume: number) => {
    setVolume(newVolume)
    if (originalAudioRef.current) originalAudioRef.current.volume = newVolume
    if (convertedAudioRef.current) convertedAudioRef.current.volume = newVolume
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getAudioUrl = (audio: string | Blob) => {
    if (typeof audio === 'string') return audio
    return URL.createObjectURL(audio)
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      <h3 className="text-lg font-bold text-gray-900 mb-4">A/B Comparison</h3>

      {/* Hidden audio elements */}
      <audio ref={originalAudioRef} src={getAudioUrl(originalAudio)} preload="auto" />
      <audio ref={convertedAudioRef} src={getAudioUrl(convertedAudio)} preload="auto" />

      {/* Track Selector */}
      <div className="flex space-x-2 mb-6">
        <button
          onClick={() => switchTrack('original')}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            activeTrack === 'original'
              ? 'bg-blue-600 text-white shadow-lg'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          {originalLabel}
        </button>
        <button
          onClick={() => switchTrack('converted')}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            activeTrack === 'converted'
              ? 'bg-purple-600 text-white shadow-lg'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          {convertedLabel}
        </button>
      </div>

      {/* Waveform Visualization */}
      <div className="mb-6">
        <AudioWaveform
          audioUrl={activeTrack === 'original' ? getAudioUrl(originalAudio) : getAudioUrl(convertedAudio)}
          height={120}
        />
      </div>

      {/* Playback Controls */}
      <div className="space-y-4">
        <div className="flex items-center space-x-4">
          <button
            onClick={togglePlayPause}
            className={`p-3 rounded-full transition-colors ${
              activeTrack === 'original'
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-purple-600 hover:bg-purple-700 text-white'
            }`}
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          <button
            onClick={handleReset}
            className="p-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors"
            title="Reset to beginning"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
          <div className="flex-1">
            <input
              type="range"
              min="0"
              max={duration || 100}
              step="0.1"
              value={currentTime}
              onChange={handleSeek}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>
        </div>

        {/* Volume Control */}
        <div className="flex items-center space-x-3">
          <Volume2 className="w-5 h-5 text-gray-400" />
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={volume}
            onChange={(e) => handleVolumeChange(Number(e.target.value))}
            className="flex-1"
          />
          <span className="text-sm text-gray-600 w-12 text-right">
            {(volume * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Quick Switch Hint */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <p className="text-sm text-gray-600 text-center">
          💡 Click the buttons above to instantly switch between original and converted audio
        </p>
      </div>
    </div>
  )
}

