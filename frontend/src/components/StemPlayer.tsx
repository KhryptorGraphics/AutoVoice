import { useState, useRef, useEffect } from 'react'
import { Play, Pause, Volume2, VolumeX, Download, Music, Mic } from 'lucide-react'

interface StemPlayerProps {
  stems: {
    vocals?: string // base64 encoded audio
    instrumental?: string // base64 encoded audio
    drums?: string
    bass?: string
    other?: string
  }
  format?: string
  sampleRate?: number
  className?: string
}

export function StemPlayer({ stems, format = 'wav', className = '' }: StemPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volumes, setVolumes] = useState<Record<string, number>>({
    vocals: 1.0,
    instrumental: 0.9,
    drums: 1.0,
    bass: 1.0,
    other: 1.0,
  })
  const [muted, setMuted] = useState<Record<string, boolean>>({})

  const audioContextRef = useRef<AudioContext | null>(null)
  const audioBuffersRef = useRef<Record<string, AudioBuffer>>({})
  const sourceNodesRef = useRef<Record<string, AudioBufferSourceNode>>({})
  const gainNodesRef = useRef<Record<string, GainNode>>({})
  const startTimeRef = useRef<number>(0)
  const pauseTimeRef = useRef<number>(0)

  useEffect(() => {
    // Initialize audio context
    audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
    
    // Load all stems
    loadStems()

    return () => {
      stopPlayback()
      audioContextRef.current?.close()
    }
  }, [stems])

  const loadStems = async () => {
    if (!audioContextRef.current) return

    const stemEntries = Object.entries(stems).filter(([_, data]) => data)
    
    for (const [name, base64Data] of stemEntries) {
      try {
        // Decode base64 to array buffer
        const binaryString = atob(base64Data as string)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i)
        }
        
        // Decode audio data
        const audioBuffer = await audioContextRef.current.decodeAudioData(bytes.buffer)
        audioBuffersRef.current[name] = audioBuffer
        
        if (duration === 0) {
          setDuration(audioBuffer.duration)
        }
      } catch (error) {
        console.error(`Failed to load ${name} stem:`, error)
      }
    }
  }

  const togglePlayPause = () => {
    if (isPlaying) {
      pausePlayback()
    } else {
      startPlayback()
    }
  }

  const startPlayback = () => {
    if (!audioContextRef.current) return

    const ctx = audioContextRef.current
    const startOffset = pauseTimeRef.current

    // Create and connect nodes for each stem
    Object.entries(audioBuffersRef.current).forEach(([name, buffer]) => {
      const source = ctx.createBufferSource()
      const gain = ctx.createGain()
      
      source.buffer = buffer
      gain.gain.value = muted[name] ? 0 : volumes[name]
      
      source.connect(gain)
      gain.connect(ctx.destination)
      
      source.start(0, startOffset)
      
      sourceNodesRef.current[name] = source
      gainNodesRef.current[name] = gain
    })

    startTimeRef.current = ctx.currentTime - startOffset
    setIsPlaying(true)

    // Update current time
    const updateTime = () => {
      if (audioContextRef.current && isPlaying) {
        const elapsed = audioContextRef.current.currentTime - startTimeRef.current
        setCurrentTime(elapsed)
        
        if (elapsed >= duration) {
          stopPlayback()
        } else {
          requestAnimationFrame(updateTime)
        }
      }
    }
    requestAnimationFrame(updateTime)
  }

  const pausePlayback = () => {
    if (!audioContextRef.current) return

    pauseTimeRef.current = audioContextRef.current.currentTime - startTimeRef.current
    
    Object.values(sourceNodesRef.current).forEach(source => {
      source.stop()
    })
    
    sourceNodesRef.current = {}
    setIsPlaying(false)
  }

  const stopPlayback = () => {
    pausePlayback()
    pauseTimeRef.current = 0
    setCurrentTime(0)
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = Number(e.target.value)
    pauseTimeRef.current = newTime
    setCurrentTime(newTime)
    
    if (isPlaying) {
      pausePlayback()
      startPlayback()
    }
  }

  const handleVolumeChange = (stem: string, volume: number) => {
    setVolumes(prev => ({ ...prev, [stem]: volume }))
    if (gainNodesRef.current[stem] && !muted[stem]) {
      gainNodesRef.current[stem].gain.value = volume
    }
  }

  const toggleMute = (stem: string) => {
    const newMuted = !muted[stem]
    setMuted(prev => ({ ...prev, [stem]: newMuted }))
    if (gainNodesRef.current[stem]) {
      gainNodesRef.current[stem].gain.value = newMuted ? 0 : volumes[stem]
    }
  }

  const downloadStem = (stem: string, data: string) => {
    const link = document.createElement('a')
    link.href = `data:audio/${format};base64,${data}`
    link.download = `${stem}.${format}`
    link.click()
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const stemIcons: Record<string, React.ReactNode> = {
    vocals: <Mic className="w-4 h-4" />,
    instrumental: <Music className="w-4 h-4" />,
    drums: <Music className="w-4 h-4" />,
    bass: <Music className="w-4 h-4" />,
    other: <Music className="w-4 h-4" />,
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
        <Music className="w-5 h-5 text-purple-600" />
        <span>Separated Stems</span>
      </h3>

      {/* Playback Controls */}
      <div className="mb-6">
        <div className="flex items-center space-x-4 mb-3">
          <button
            onClick={togglePlayPause}
            className="p-3 bg-purple-600 hover:bg-purple-700 text-white rounded-full transition-colors"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          <div className="flex-1">
            <input
              type="range"
              min="0"
              max={duration}
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
      </div>

      {/* Stem Controls */}
      <div className="space-y-4">
        {Object.entries(stems).filter(([_, data]) => data).map(([name, data]) => (
          <div key={name} className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <div className="text-purple-600">{stemIcons[name]}</div>
                <span className="font-medium text-gray-900 capitalize">{name}</span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => toggleMute(name)}
                  className="p-2 text-gray-600 hover:text-purple-600 rounded-lg hover:bg-purple-50 transition-colors"
                >
                  {muted[name] ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => downloadStem(name, data as string)}
                  className="p-2 text-gray-600 hover:text-purple-600 rounded-lg hover:bg-purple-50 transition-colors"
                  title="Download stem"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Volume2 className="w-4 h-4 text-gray-400" />
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={volumes[name] || 1.0}
                onChange={(e) => handleVolumeChange(name, Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm text-gray-600 w-12 text-right">
                {((volumes[name] || 1.0) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

