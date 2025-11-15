import { useEffect, useRef, useState } from 'react'

interface RealtimeWaveformProps {
  isProcessing: boolean
  progress: number
  audioUrl?: string
}

export function RealtimeWaveform({ isProcessing, progress, audioUrl }: RealtimeWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null)
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)

  useEffect(() => {
    if (!audioUrl || !isProcessing) return

    const initAudio = async () => {
      try {
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)()
        const response = await fetch(audioUrl)
        const arrayBuffer = await response.arrayBuffer()
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer)

        const analyserNode = ctx.createAnalyser()
        analyserNode.fftSize = 256

        const source = ctx.createBufferSource()
        source.buffer = audioBuffer
        source.connect(analyserNode)
        analyserNode.connect(ctx.destination)
        source.start(0)

        setAudioContext(ctx)
        setAnalyser(analyserNode)
      } catch (err) {
        console.error('Failed to initialize audio:', err)
      }
    }

    initAudio()
  }, [audioUrl, isProcessing])

  useEffect(() => {
    if (!canvasRef.current || !analyser) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dataArray = new Uint8Array(analyser.frequencyBinCount)

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw)

      analyser.getByteFrequencyData(dataArray)

      ctx.fillStyle = 'rgb(249, 250, 251)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw frequency bars
      const barWidth = (canvas.width / dataArray.length) * 2.5
      let x = 0

      for (let i = 0; i < dataArray.length; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height

        // Gradient color based on frequency
        const hue = (i / dataArray.length) * 360
        ctx.fillStyle = `hsl(${hue}, 100%, 50%)`
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)

        x += barWidth + 1
      }

      // Draw progress indicator
      ctx.fillStyle = 'rgba(124, 58, 237, 0.3)'
      ctx.fillRect(0, 0, (canvas.width * progress) / 100, canvas.height)

      // Draw progress text
      ctx.fillStyle = 'rgb(107, 114, 128)'
      ctx.font = '12px sans-serif'
      ctx.fillText(`${progress}%`, 10, 20)
    }

    draw()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [analyser, progress])

  // Fallback visualization when not playing audio
  if (!isProcessing || !analyser) {
    return (
      <div className="w-full h-32 bg-gray-50 rounded-lg border border-gray-200 flex items-center justify-center">
        <div className="text-center">
          <div className="flex justify-center space-x-1 mb-2">
            {[0, 1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="w-1 bg-primary-600 rounded-full"
                style={{
                  height: `${20 + i * 10}px`,
                  animation: `pulse 0.6s ease-in-out ${i * 0.1}s infinite`,
                }}
              />
            ))}
          </div>
          <p className="text-sm text-gray-600">Processing: {progress}%</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      <canvas
        ref={canvasRef}
        width={800}
        height={128}
        className="w-full h-32 bg-gray-50 rounded-lg border border-gray-200"
      />
      <style>{`
        @keyframes pulse {
          0%, 100% { transform: scaleY(0.5); }
          50% { transform: scaleY(1); }
        }
      `}</style>
    </div>
  )
}

