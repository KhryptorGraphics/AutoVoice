/**
 * Live Training Monitor - real backend telemetry and controls.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Activity,
  Pause,
  Play,
  Volume2,
  TrendingDown,
  Zap,
  RefreshCw,
  Square,
  HardDrive,
  Gauge,
} from 'lucide-react'
import { apiService, wsManager, TrainingProgressEvent } from '../services/api'
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
  gpuMemoryGb?: number
  gpuUtilizationPercent?: number
  mosProxy?: number
  speakerSimilarityProxy?: number
  checkpointPath?: string | null
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

function applyProgressEvent(prev: LiveStats, progress: TrainingProgressEvent): LiveStats {
  return {
    epoch: progress.epoch,
    totalEpochs: progress.total_epochs,
    step: progress.step,
    totalSteps: progress.total_steps,
    loss: progress.loss,
    learningRate: progress.learning_rate,
    lossHistory: [...prev.lossHistory, progress.loss].slice(-500),
    startTime: prev.startTime,
    lastUpdateTime: Date.now(),
    gpuMemoryGb: progress.gpu_metrics?.memory_used_gb,
    gpuUtilizationPercent: progress.gpu_metrics?.utilization_percent,
    mosProxy: progress.quality_metrics?.mos_proxy,
    speakerSimilarityProxy: progress.quality_metrics?.speaker_similarity_proxy,
    checkpointPath: progress.checkpoint_path ?? prev.checkpointPath ?? null,
  }
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
    checkpointPath: null,
  })
  const [isConnected, setIsConnected] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null)
  const [loadingPreview, setLoadingPreview] = useState(false)
  const [updatingState, setUpdatingState] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)
  const chartRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    wsManager.connect()
    setIsConnected(true)

    let cancelled = false

    const loadTelemetry = async () => {
      try {
        const telemetry = await apiService.getTrainingTelemetry(jobId)
        if (cancelled) {
          return
        }

        setIsPaused(Boolean(telemetry.job.is_paused))
        setStats((prev) => ({
          ...prev,
          epoch: telemetry.runtime_metrics.epoch ?? telemetry.job.results?.current_epoch ?? 0,
          totalEpochs: telemetry.runtime_metrics.total_epochs ?? 0,
          step: telemetry.runtime_metrics.step ?? 0,
          totalSteps: telemetry.runtime_metrics.total_steps ?? 0,
          loss: telemetry.runtime_metrics.loss ?? telemetry.job.results?.current_loss ?? 0,
          learningRate: telemetry.runtime_metrics.learning_rate ?? 0,
          gpuMemoryGb:
            telemetry.runtime_metrics.gpu_metrics?.memory_used_gb ??
            telemetry.job.results?.gpu_metrics?.memory_used_gb,
          gpuUtilizationPercent:
            telemetry.runtime_metrics.gpu_metrics?.utilization_percent ??
            telemetry.job.results?.gpu_metrics?.utilization_percent,
          mosProxy:
            telemetry.runtime_metrics.quality_metrics?.mos_proxy ??
            telemetry.job.results?.quality_metrics?.mos_proxy,
          speakerSimilarityProxy:
            telemetry.runtime_metrics.quality_metrics?.speaker_similarity_proxy ??
            telemetry.job.results?.quality_metrics?.speaker_similarity_proxy,
          checkpointPath:
            telemetry.runtime_metrics.checkpoint_path ??
            telemetry.job.results?.latest_checkpoint ??
            null,
          lastUpdateTime: Date.now(),
        }))
      } catch (error) {
        console.error('Failed to load training telemetry:', error)
      }
    }

    void loadTelemetry()

    const unsubProgress = wsManager.onTrainingProgress(jobId, (progress: TrainingProgressEvent) => {
      setIsPaused(Boolean(progress.is_paused))
      setStats((prev) => applyProgressEvent(prev, progress))
    })

    const unsubComplete = wsManager.subscribe('training_complete', (event) => {
      if ((event.data as { job_id?: string }).job_id === jobId) {
        onComplete?.()
      }
    })

    const unsubPaused = wsManager.subscribe('training_paused', (event) => {
      if ((event.data as { job_id?: string }).job_id === jobId) {
        setIsPaused(true)
      }
    })

    const unsubResumed = wsManager.subscribe('training_resumed', (event) => {
      if ((event.data as { job_id?: string }).job_id === jobId) {
        setIsPaused(false)
      }
    })

    const unsubCancelled = wsManager.subscribe('training_cancelled', (event) => {
      if ((event.data as { job_id?: string }).job_id === jobId) {
        onComplete?.()
      }
    })

    return () => {
      cancelled = true
      unsubProgress()
      unsubComplete()
      unsubPaused()
      unsubResumed()
      unsubCancelled()
    }
  }, [jobId, onComplete])

  useEffect(() => {
    return () => {
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl)
      }
    }
  }, [audioPreviewUrl])

  const handlePauseResume = useCallback(async () => {
    setUpdatingState(true)
    try {
      if (isPaused) {
        await apiService.resumeTrainingJob(jobId)
        setIsPaused(false)
      } else {
        await apiService.pauseTrainingJob(jobId)
        setIsPaused(true)
      }
    } catch (error) {
      console.error('Failed to update training state:', error)
    } finally {
      setUpdatingState(false)
    }
  }, [isPaused, jobId])

  const handleCancel = useCallback(async () => {
    setCancelling(true)
    try {
      await apiService.cancelTrainingJob(jobId)
      onComplete?.()
    } catch (error) {
      console.error('Failed to cancel training job:', error)
    } finally {
      setCancelling(false)
    }
  }, [jobId, onComplete])

  const handleGeneratePreview = useCallback(async () => {
    setLoadingPreview(true)
    try {
      const blob = await apiService.getTrainingPreview(jobId, {
        profile_id: profileId,
        duration_seconds: 4,
      })
      const url = URL.createObjectURL(blob)
      setAudioPreviewUrl((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev)
        }
        return url
      })
    } catch (error) {
      console.error('Failed to generate preview:', error)
    } finally {
      setLoadingPreview(false)
    }
  }, [jobId, profileId])

  const elapsedTime = (Date.now() - stats.startTime) / 1000
  const epochProgress = stats.totalEpochs > 0 ? (stats.epoch / stats.totalEpochs) * 100 : 0
  const stepProgress = stats.totalSteps > 0 ? (stats.step / stats.totalSteps) * 100 : 0
  const overallProgress = stats.totalEpochs > 0
    ? ((Math.max(stats.epoch - 1, 0) + stepProgress / 100) / stats.totalEpochs) * 100
    : stepProgress

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

  const lossImprovement = stats.lossHistory.length > 1
    ? ((stats.lossHistory[0] - stats.loss) / stats.lossHistory[0]) * 100
    : 0

  return (
    <div className="bg-gray-800 rounded-lg p-6 space-y-6" data-testid="live-training-monitor">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={clsx(
              'w-3 h-3 rounded-full',
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            )}
          />
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Activity size={20} />
            Live Training
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handlePauseResume}
            disabled={updatingState || cancelling}
            data-testid={isPaused ? 'resume-training-button' : 'pause-training-button'}
            className={clsx(
              'p-2 rounded-lg transition-colors disabled:opacity-50',
              isPaused ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-700 hover:bg-gray-600'
            )}
            title={isPaused ? 'Resume training' : 'Pause training'}
          >
            {updatingState ? (
              <RefreshCw size={16} className="animate-spin" />
            ) : isPaused ? (
              <Play size={16} />
            ) : (
              <Pause size={16} />
            )}
          </button>
          <button
            onClick={handleCancel}
            disabled={cancelling || updatingState}
            data-testid="cancel-training-button"
            className="p-2 rounded-lg bg-red-600 hover:bg-red-700 transition-colors disabled:opacity-50"
            title="Cancel training"
          >
            {cancelling ? <RefreshCw size={16} className="animate-spin" /> : <Square size={16} />}
          </button>
        </div>
      </div>

      {isPaused && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
          Training is paused. Resume to continue producing checkpoints and updated previews.
        </div>
      )}

      <div className="space-y-4">
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
              <defs>
                <pattern id="liveGrid" width="10" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 10 0 L 0 0 0 20" fill="none" stroke="rgba(75, 85, 99, 0.2)" strokeWidth="0.5" />
                </pattern>
                <linearGradient id="liveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#ef4444" />
                  <stop offset="50%" stopColor="#eab308" />
                  <stop offset="100%" stopColor="#22c55e" />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <rect width="100" height="100" fill="url(#liveGrid)" />
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
              {stats.lossHistory.length > 0 && (
                <circle
                  cx="100"
                  cy={
                    100 - (
                      (stats.loss - Math.min(...stats.lossHistory)) /
                      (Math.max(...stats.lossHistory) - Math.min(...stats.lossHistory) || 1)
                    ) * 100
                  }
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

      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-white font-mono">
            {formatLoss(stats.loss)}
          </div>
          <div className="text-xs text-gray-500">Current Loss</div>
        </div>
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-cyan-400 font-mono">
            {stats.learningRate > 0 ? stats.learningRate.toExponential(1) : 'n/a'}
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
            {stats.mosProxy?.toFixed(2) ?? 'n/a'}
          </div>
          <div className="text-xs text-gray-500">MOS Proxy</div>
        </div>
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-violet-300 font-mono">
            {stats.gpuMemoryGb != null ? `${stats.gpuMemoryGb.toFixed(1)}G` : 'n/a'}
          </div>
          <div className="text-xs text-gray-500 flex items-center justify-center gap-1">
            <HardDrive size={12} />
            GPU Memory
          </div>
        </div>
        <div className="bg-gray-750 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-emerald-300 font-mono">
            {stats.gpuUtilizationPercent != null ? `${stats.gpuUtilizationPercent.toFixed(0)}%` : 'n/a'}
          </div>
          <div className="text-xs text-gray-500 flex items-center justify-center gap-1">
            <Gauge size={12} />
            GPU Util
          </div>
        </div>
      </div>

      {stats.checkpointPath && (
        <div className="rounded-lg bg-gray-750 px-3 py-2 text-xs text-gray-400" data-testid="training-checkpoint-path">
          Latest checkpoint: <span className="text-gray-200">{stats.checkpointPath}</span>
        </div>
      )}

      <div className="border-t border-gray-700 pt-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Volume2 size={16} className="text-gray-400" />
            <span className="text-sm text-gray-400">Audio Preview</span>
          </div>
          <button
            onClick={handleGeneratePreview}
            disabled={loadingPreview || stats.epoch < 1}
            data-testid="generate-training-preview"
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
            data-testid="training-preview-audio"
          />
        )}

        {!audioPreviewUrl && stats.epoch < 1 && (
          <p className="text-xs text-gray-500">
            Audio preview available after the first epoch emits progress.
          </p>
        )}
      </div>
    </div>
  )
}

export function LiveLossMini({ jobId }: { jobId: string }) {
  const [losses, setLosses] = useState<number[]>([])
  const [currentLoss, setCurrentLoss] = useState<number>(0)

  useEffect(() => {
    wsManager.connect()
    const unsub = wsManager.onTrainingProgress(jobId, (progress) => {
      setCurrentLoss(progress.loss)
      setLosses((prev) => [...prev, progress.loss].slice(-50))
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
