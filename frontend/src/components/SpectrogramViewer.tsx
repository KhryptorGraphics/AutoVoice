import { useState, useRef, useEffect, useCallback } from 'react'
import { ZoomIn, ZoomOut, RotateCcw, Columns, Loader2 } from 'lucide-react'
import clsx from 'clsx'

interface SpectrogramViewerProps {
  audioUrl?: string
  compareUrl?: string // For before/after comparison
  height?: number
  colorScale?: 'viridis' | 'magma' | 'inferno' | 'plasma'
}

export function SpectrogramViewer({
  audioUrl,
  compareUrl,
  height = 200,
  colorScale = 'viridis',
}: SpectrogramViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const compareCanvasRef = useRef<HTMLCanvasElement>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [zoom, setZoom] = useState(1)
  const [scrollX, setScrollX] = useState(0)
  const [showComparison, setShowComparison] = useState(false)
  const [spectrogramData, setSpectrogramData] = useState<ImageData | null>(null)
  const [compareData, setCompareData] = useState<ImageData | null>(null)

  const colorMaps: Record<string, (t: number) => [number, number, number]> = {
    viridis: (t) => {
      const r = Math.floor(68 + 187 * Math.pow(t, 0.5))
      const g = Math.floor(1 + 254 * t)
      const b = Math.floor(84 + 171 * (1 - t))
      return [Math.min(255, r), Math.min(255, g), Math.min(255, b)]
    },
    magma: (t) => {
      const r = Math.floor(252 * Math.pow(t, 0.6))
      const g = Math.floor(100 * t)
      const b = Math.floor(180 * (1 - Math.pow(t, 0.8)) + 75 * t)
      return [r, g, b]
    },
    inferno: (t) => {
      const r = Math.floor(252 * Math.pow(t, 0.5))
      const g = Math.floor(160 * t)
      const b = Math.floor(255 * (1 - Math.pow(t, 0.5)))
      return [r, g, b]
    },
    plasma: (t) => {
      const r = Math.floor(240 * Math.pow(t, 0.4))
      const g = Math.floor(100 + 155 * t)
      const b = Math.floor(220 * (1 - t * 0.5))
      return [r, g, b]
    },
  }

  const generateSpectrogram = useCallback(async (url: string): Promise<ImageData | null> => {
    try {
      const audioContext = new AudioContext()
      const response = await fetch(url)
      const arrayBuffer = await response.arrayBuffer()
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

      const channelData = audioBuffer.getChannelData(0)
      // audioBuffer.sampleRate available for frequency axis scaling if needed
      const fftSize = 2048
      const hopSize = 512
      const numFrames = Math.floor((channelData.length - fftSize) / hopSize)
      const numBins = fftSize / 2

      // Create offscreen canvas for spectrogram
      const canvas = document.createElement('canvas')
      canvas.width = numFrames
      canvas.height = numBins
      const ctx = canvas.getContext('2d')!
      const imageData = ctx.createImageData(numFrames, numBins)

      // Simple FFT magnitude calculation (simplified for demo)
      const colorMap = colorMaps[colorScale]
      for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * hopSize
        const window = new Float32Array(fftSize)

        // Apply Hann window
        for (let i = 0; i < fftSize; i++) {
          const hannWindow = 0.5 * (1 - Math.cos((2 * Math.PI * i) / fftSize))
          window[i] = (channelData[start + i] || 0) * hannWindow
        }

        // Simple DFT approximation for visualization
        for (let bin = 0; bin < numBins; bin++) {
          let real = 0, imag = 0
          for (let n = 0; n < fftSize; n += 8) { // Subsampled for performance
            const angle = (2 * Math.PI * bin * n) / fftSize
            real += window[n] * Math.cos(angle)
            imag -= window[n] * Math.sin(angle)
          }
          const magnitude = Math.sqrt(real * real + imag * imag)
          const db = 20 * Math.log10(magnitude + 1e-10)
          const normalized = Math.max(0, Math.min(1, (db + 80) / 80))

          const [r, g, b] = colorMap(normalized)
          const y = numBins - 1 - bin
          const idx = (y * numFrames + frame) * 4
          imageData.data[idx] = r
          imageData.data[idx + 1] = g
          imageData.data[idx + 2] = b
          imageData.data[idx + 3] = 255
        }
      }

      await audioContext.close()
      return imageData
    } catch (err) {
      console.error('Spectrogram generation failed:', err)
      return null
    }
  }, [colorScale])

  useEffect(() => {
    if (!audioUrl) return

    setLoading(true)
    setError(null)

    generateSpectrogram(audioUrl)
      .then(data => {
        setSpectrogramData(data)
        setLoading(false)
      })
      .catch(() => {
        setError('Failed to generate spectrogram')
        setLoading(false)
      })
  }, [audioUrl, generateSpectrogram])

  useEffect(() => {
    if (!compareUrl) return

    generateSpectrogram(compareUrl).then(setCompareData)
  }, [compareUrl, generateSpectrogram])

  useEffect(() => {
    if (!canvasRef.current || !spectrogramData) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!

    // Create temp canvas with spectrogram data
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = spectrogramData.width
    tempCanvas.height = spectrogramData.height
    const tempCtx = tempCanvas.getContext('2d')!
    tempCtx.putImageData(spectrogramData, 0, 0)

    // Draw scaled to main canvas
    canvas.width = canvas.offsetWidth
    canvas.height = height
    ctx.imageSmoothingEnabled = false

    const scaledWidth = spectrogramData.width * zoom
    ctx.drawImage(
      tempCanvas,
      -scrollX, 0,
      scaledWidth, height
    )
  }, [spectrogramData, zoom, scrollX, height])

  useEffect(() => {
    if (!compareCanvasRef.current || !compareData) return

    const canvas = compareCanvasRef.current
    const ctx = canvas.getContext('2d')!

    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = compareData.width
    tempCanvas.height = compareData.height
    const tempCtx = tempCanvas.getContext('2d')!
    tempCtx.putImageData(compareData, 0, 0)

    canvas.width = canvas.offsetWidth
    canvas.height = height
    ctx.imageSmoothingEnabled = false

    const scaledWidth = compareData.width * zoom
    ctx.drawImage(
      tempCanvas,
      -scrollX, 0,
      scaledWidth, height
    )
  }, [compareData, zoom, scrollX, height])

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setZoom(z => Math.max(0.5, Math.min(10, z * delta)))
    } else {
      setScrollX(x => Math.max(0, x + e.deltaX))
    }
  }

  const handleZoomIn = () => setZoom(z => Math.min(10, z * 1.25))
  const handleZoomOut = () => setZoom(z => Math.max(0.5, z * 0.8))
  const handleReset = () => { setZoom(1); setScrollX(0) }

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

  if (loading) {
    return (
      <div
        className="bg-gray-800 rounded-lg flex items-center justify-center"
        style={{ height }}
      >
        <Loader2 className="animate-spin text-gray-500" />
        <span className="ml-2 text-gray-400 text-sm">Generating spectrogram...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="bg-gray-800 rounded-lg flex items-center justify-center text-red-400"
        style={{ height }}
      >
        {error}
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Controls */}
      <div className="flex items-center justify-between p-2 border-b border-gray-700">
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomOut}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Zoom out"
          >
            <ZoomOut size={14} />
          </button>
          <span className="text-xs text-gray-500 w-12 text-center">
            {(zoom * 100).toFixed(0)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Zoom in"
          >
            <ZoomIn size={14} />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Reset view"
          >
            <RotateCcw size={14} />
          </button>
        </div>

        {compareUrl && (
          <button
            onClick={() => setShowComparison(!showComparison)}
            className={clsx(
              'flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors',
              showComparison ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
            )}
          >
            <Columns size={12} />
            Compare
          </button>
        )}

        <select
          value={colorScale}
          className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
        >
          <option value="viridis">Viridis</option>
          <option value="magma">Magma</option>
          <option value="inferno">Inferno</option>
          <option value="plasma">Plasma</option>
        </select>
      </div>

      {/* Spectrogram Display */}
      <div
        className={clsx('flex', showComparison && compareUrl ? 'gap-1' : '')}
        onWheel={handleWheel}
      >
        <div className="flex-1 relative">
          <canvas
            ref={canvasRef}
            className="w-full cursor-grab active:cursor-grabbing"
            style={{ height }}
          />
          {showComparison && (
            <div className="absolute bottom-2 left-2 text-xs bg-black/50 px-2 py-1 rounded">
              Original
            </div>
          )}
        </div>

        {showComparison && compareUrl && (
          <div className="flex-1 relative">
            <canvas
              ref={compareCanvasRef}
              className="w-full cursor-grab active:cursor-grabbing"
              style={{ height }}
            />
            <div className="absolute bottom-2 left-2 text-xs bg-black/50 px-2 py-1 rounded">
              Converted
            </div>
          </div>
        )}
      </div>

      {/* Frequency Axis */}
      <div className="flex justify-between text-xs text-gray-500 px-2 py-1 border-t border-gray-700">
        <span>0 Hz</span>
        <span>~22 kHz</span>
      </div>
    </div>
  )
}
