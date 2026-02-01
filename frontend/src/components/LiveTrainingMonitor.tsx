/**
 * Live Training Monitor - Real-time training progress with streaming loss curve
 *
 * Subscribes to WebSocket training_progress events and displays:
 * - Live loss curve updating in real-time
 * - Current epoch/step progress
 * - Learning rate display
 * - Audio preview of current model output
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { Activity, Pause, Play, Volume2, TrendingDown, Zap, RefreshCw } from 'lucide-react'
import { wsManager, TrainingProgressEvent } from '../services/api'
import clsx from 'clsx'

interface LiveTrainingMonitorProps {
  jobId: string
  profileId: string
  onComplete?: () => void
}

interface LiveStats {
  epoch: number
  totalEpochs: number
  step: number
  totalSteps: number
  loss: number
  learningRate: number
  lossHistory: number[]
  startTime: number
  lastUpdateTime: number
}

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function formatLoss(value: number): string {
  if (value < 0.001) return value.toExponential(2)
  return value.toFixed(4)
}

export function LiveTrainingMonitor({ jobId, profileId, onComplete }: LiveTrainingMonitorProps) {
  const [stats, setStats] = useState<LiveStats>({
    epoch: 0,
    totalEpochs: 0,
    step: 0,
    totalSteps: 0,
    loss: 0,
    learningRate: 0,
    lossHistory: [],
    startTime: Date.now(),
    lastUpdateTime: Date.now(),
  })
  const [isConnected, setIsConnected] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null)
  const [loadingPreview, setLoadingPreview] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)
  const chartRef = useRef<SVGSVGElement>(null)

  // Subscribe to WebSocket training progress events
  useEffect(() => {
    wsManager.connect()
    setIsConnected(true)

    const unsubProgress = wsManager.onTrainingProgress(jobId, (progress: TrainingProgressEvent) => {
      if (isPaused) return

      setStats(prev => ({
        epoch: progress.epoch,
        totalEpochs: progress.total_epochs,
        step: progress.step,
        totalSteps: progress.total_steps,
        loss: progress.loss,
        learningRate: progress.learning_rate,
        lossHistory: [...prev.lossHistory, progress.loss].slice(-500), // Keep last 500 points
        startTime: prev.startTime,
        lastUpdateTime: Date.now(),
      }))
    })

    const unsubComplete = wsManager.subscribe('training_complete', (event) => {
      if ((event.data as any).job_id === jobId) {
        onComplete?.()
      }
    })

    return () => {
      unsubProgress()
      unsubComplete()
    }
  }, [jobId, isPaused, onComplete])

  // Generate audio preview from current model checkpoint
  const handleGeneratePreview = useCallback(async () => {
    setLoadingPreview(true)
    try {
      // Request audio preview from backend
      const response = await fetch(`/api/v1/training/preview/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: profileId }),
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        setAudioPreviewUrl(url)
      }
    } catch (error) {
      console.error('Failed to generate preview:', error)
    } finally {
      setLoadingPreview(false)
    }
  }, [jobId, profileId])

  // Calculate derived stats
  const elapsedTime = (Date.now() - stats.startTime) / 1000
  const epochProgress = stats.totalEpochs > 0 ? (stats.epoch / stats.totalEpochs) * 100 : 0
  const stepProgress = stats.totalSteps > 0 ? (stats.step / stats.totalSteps) * 100 : 0
  const overallProgress = stats.totalEpochs > 0
    ? ((stats.epoch - 1 + stepProgress / 100) / stats.totalEpochs) * 100
    : stepProgress

  // Loss curve SVG path
  const lossCurvePath = stats.lossHistory.length > 1 ? (() => {
    const min = Math.min(...stats.lossHistory)
    const max = Math.max(...stats.lossHistory)
    const range = max - min || 1

    return stats.lossHistory.map((loss, i) => {
      const x = (i / (stats.lossHistory.length - 1)) * 100
      const y = 100 - ((loss - min) / range) * 100
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
    }).join(' ')
  })() : ''

  // Loss improvement
  const lossImprovement = stats.lossHistory.length > 1
    ? ((stats.lossHistory[0] - stats.loss) / stats.lossHistory[0]) * 100
    : 0

  return (
    <div className="bg-gray-800 rounded-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={clsx(
            'w-3 h-3 rounded-full',
            isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          )} />
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Activity size={20} />
            Live Training
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPaused(!isPaused)}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              isPaused ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-700 hover:bg-gray-600'
            )}
            title={isPaused ? 'Resume updates' : 'Pause updates'}
          >
            {isPaused ? <Play size={16} /> : <Pause size={16} />}
          </button>
        </div>
      </div>

      {/* Progress bars */}
      <div className="space-y-4">
        {/* Overall progress */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">Overall Progress</span>
            <span className="text-white font-medium">{overallProgress.toFixed(1)}%</span>
          </div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-500 to-purple-500 transition-all duration-300"
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>

        {/* Epoch progress */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-500">Epoch</span>
              <span className="text-gray-300">{stats.epoch} / {stats.totalEpochs}</span>
            </div>
            <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${epochProgress}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-500">Step</span>
              <span className="text-gray-300">{stats.step} / {stats.totalSteps}</span>
            </div>
            <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-500 transition-all duration-300"
                style={{ width: `${stepProgress}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Live loss curve */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingDown size={16} className="text-gray-400" />
            <span className="text-sm text-gray-400">Loss Curve (Live)</span>
          </div>
          {lossImprovement > 0 && (
            <span className="text-xs text-green-400 bg-green-900/30 px-2 py-0.5 rounded">
              -{lossImprovement.toFixed(1)}% improvement
            </span>
          )}
        </div>

        <div className="relative h-32 bg-gray-900 rounded-lg p-2">
          {stats.lossHistory.length < 2 ? (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
              Waiting for training data...
            </div>
          ) : (
            <svg
              ref={chartRef}
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              className="w-full h-full"
            >
              {/* Grid */}
              <defs>
                <pattern id="liveGrid" width="10" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 10 0 L 0 0 0 20" fill="none" stroke="rgba(75, 85, 99, 0.2)" strokeWidth="0.5" />
                </pattern>
              </defs>
              <rect width="100" height="100" fill="url(#liveGrid)" />

              {/* Loss curve with glow effect */}
              <defs>
                <linearGradient id="liveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#ef4444" />
                  <stop offset="50%" stopColor="#eab308" />
                  <stop offset="100%" stopColor="#22c55e" />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>

              <path
                d={lossCurvePath}
                fill="none"
                stroke="url(#liveGradient)"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#glow)"
                vectorEffect="non-scaling-stroke"
              />

              {/* Current point indicator */}
              {stats.lossHistory.length > 0 && (
                <circle
                  cx="100"
                  cy={100 - ((stats.loss - Math.min(...stats.lossHistory)) / (Math.max(...stats.lossHistory) - Math.min(...stats.lossHistory) || 1)) * 100}
                  r="4"
                  fill="#22c55e"
                  filter="url(#glow)"
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </svg>
          )}
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-white font-mono">
            {formatLoss(stats.loss)}
          </div>
          <div className="text-xs text-gray-500">Current Loss</div>
        </div>
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-cyan-400 font-mono">
            {stats.learningRate.toExponential(1)}
          </div>
          <div className="text-xs text-gray-500">Learning Rate</div>
        </div>
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-yellow-400 font-mono">
            {formatTime(elapsedTime)}
          </div>
          <div className="text-xs text-gray-500">Elapsed</div>
        </div>
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-green-400 font-mono">
            {stats.lossHistory.length}
          </div>
          <div className="text-xs text-gray-500">Data Points</div>
        </div>
      </div>

      {/* Audio preview section */}
      <div className="border-t border-gray-700 pt-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Volume2 size={16} className="text-gray-400" />
            <span className="text-sm text-gray-400">Audio Preview</span>
          </div>
          <button
            onClick={handleGeneratePreview}
            disabled={loadingPreview || stats.epoch < 1}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
              loadingPreview || stats.epoch < 1
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-violet-600 hover:bg-violet-700 text-white'
            )}
          >
            {loadingPreview ? (
              <>
                <RefreshCw size={14} className="animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Zap size={14} />
                Generate Preview
              </>
            )}
          </button>
        </div>

        {audioPreviewUrl && (
          <audio
            ref={audioRef}
            src={audioPreviewUrl}
            controls
            className="w-full"
          />
        )}

        {!audioPreviewUrl && stats.epoch < 1 && (
          <p className="text-xs text-gray-500">
            Audio preview available after first epoch completes
          </p>
        )}
      </div>
    </div>
  )
}

// Compact version for embedding in job cards
export function LiveLossMini({ jobId }: { jobId: string }) {
  const [losses, setLosses] = useState<number[]>([])
  const [currentLoss, setCurrentLoss] = useState<number>(0)

  useEffect(() => {
    wsManager.connect()
    const unsub = wsManager.onTrainingProgress(jobId, (progress) => {
      setCurrentLoss(progress.loss)
      setLosses(prev => [...prev, progress.loss].slice(-50))
    })
    return unsub
  }, [jobId])

  if (losses.length < 2) return null

  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const range = max - min || 1

  const points = losses.map((loss, i) => {
    const x = (i / (losses.length - 1)) * 100
    const y = 100 - ((loss - min) / range) * 100
    return `${x},${y}`
  }).join(' ')

  return (
    <div className="flex items-center gap-2">
      <svg viewBox="0 0 100 100" className="w-20 h-6" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke="#22c55e"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <span className="text-xs font-mono text-green-400">{formatLoss(currentLoss)}</span>
    </div>
  )
}
