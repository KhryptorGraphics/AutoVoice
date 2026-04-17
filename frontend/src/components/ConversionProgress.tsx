/**
 * Conversion Progress Component - Real-time conversion tracking with quality metrics
 *
 * Displays:
 * - Progress bar with stage indicators
 * - Real-time updates via WebSocket
 * - Quality metrics after completion (RTF, processing time)
 * - Pipeline and adapter info
 * - Error handling for missing adapters
 */
import { useState, useEffect, useCallback } from 'react'
import {
  Loader2, CheckCircle, XCircle, AlertCircle, Clock, Zap,
  Music, Wand2, Mic2, Volume2, Timer, Cpu, Download
} from 'lucide-react'
import { wsManager, ConversionProgressEvent, apiService, AdapterType } from '../services/api'
import { PipelineBadge, PipelineType } from './PipelineSelector'
import { AdapterBadge } from './AdapterSelector'
import clsx from 'clsx'

// Extended conversion status from backend
export interface ConversionStatus {
  job_id: string
  status: 'queued' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'error' | 'cancelled'
  progress: number
  stage?: ConversionStage
  message?: string
  error?: string
  // Metadata
  profile_id?: string
  pipeline_type?: PipelineType
  adapter_type?: AdapterType
  // Timing metrics
  created_at?: string
  started_at?: string
  completed_at?: string
  processing_time_seconds?: number
  // Quality metrics
  rtf?: number  // Real-time factor
  audio_duration_seconds?: number
  output_url?: string
  download_url?: string
}

export type ConversionStage = 'separating' | 'encoding' | 'converting' | 'vocoding' | 'mixing'

interface ConversionProgressProps {
  jobId: string
  profileId?: string
  pipelineType?: PipelineType
  adapterType?: AdapterType
  onComplete?: (status: ConversionStatus) => void
  onError?: (error: string) => void
  showDownload?: boolean
  compact?: boolean
}

const stageConfig: Record<ConversionStage, { label: string; icon: React.ReactNode; order: number }> = {
  separating: { label: 'Separating Vocals', icon: <Music size={14} />, order: 1 },
  encoding: { label: 'Encoding Content', icon: <Wand2 size={14} />, order: 2 },
  converting: { label: 'Converting Voice', icon: <Mic2 size={14} />, order: 3 },
  vocoding: { label: 'Vocoding', icon: <Volume2 size={14} />, order: 4 },
  mixing: { label: 'Final Mix', icon: <Music size={14} />, order: 5 },
}

const stages: ConversionStage[] = ['separating', 'encoding', 'converting', 'vocoding', 'mixing']

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  }
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}m ${secs}s`
}

function formatRTF(rtf: number): string {
  if (rtf < 1) {
    return `${(rtf * 100).toFixed(0)}% realtime`
  }
  return `${rtf.toFixed(1)}x realtime`
}

export function ConversionProgress({
  jobId,
  profileId: _profileId,
  pipelineType,
  adapterType,
  onComplete,
  onError,
  showDownload = true,
  compact = false,
}: ConversionProgressProps) {
  // _profileId reserved for future use (profile-specific styling)
  void _profileId

  const [status, setStatus] = useState<ConversionStatus>({
    job_id: jobId,
    status: 'queued',
    progress: 0,
  })
  const [isPolling, setIsPolling] = useState(true)

  // Fetch initial status
  const fetchStatus = useCallback(async () => {
    try {
      const data = await apiService.getConversionStatus(jobId) as unknown as ConversionStatus
      const normalizedStatus =
        data.status === 'in_progress' ? 'processing' :
        data.status === 'failed' ? 'error' :
        data.status
      setStatus(prev => ({
        ...prev,
        ...data,
        status: normalizedStatus,
        pipeline_type: data.pipeline_type || pipelineType,
        adapter_type: data.adapter_type || adapterType,
      }))

      if (normalizedStatus === 'completed') {
        setIsPolling(false)
        onComplete?.({ ...data, status: normalizedStatus })
      } else if (normalizedStatus === 'error') {
        setIsPolling(false)
        onError?.(data.error || 'Conversion failed')
      }
    } catch (err) {
      console.error('Failed to fetch status:', err)
    }
  }, [jobId, pipelineType, adapterType, onComplete, onError])

  // Subscribe to WebSocket updates
  useEffect(() => {
    const unsubscribe = wsManager.onConversionProgress(jobId, (event: ConversionProgressEvent) => {
      setStatus(prev => ({
        ...prev,
        progress: event.progress,
        stage: event.stage,
        message: event.message,
        status: event.progress >= 100 ? 'completed' : 'processing',
      }))
    })

    // Also subscribe to completion/error events
    const unsubComplete = wsManager.subscribe('conversion_complete', (event) => {
      const data = event.data as { job_id: string; output_url?: string }
      if (data.job_id === jobId) {
        fetchStatus() // Get full status with metrics
      }
    })

    const unsubError = wsManager.subscribe('conversion_error', (event) => {
      const data = event.data as { job_id: string; error: string }
      if (data.job_id === jobId) {
        setStatus(prev => ({
          ...prev,
          status: 'error',
          error: data.error,
        }))
        setIsPolling(false)
        onError?.(data.error)
      }
    })

    return () => {
      unsubscribe()
      unsubComplete()
      unsubError()
    }
  }, [jobId, fetchStatus, onError])

  // Poll for status updates as fallback
  useEffect(() => {
    fetchStatus()

    if (isPolling) {
      const interval = setInterval(fetchStatus, 2000)
      return () => clearInterval(interval)
    }
  }, [fetchStatus, isPolling])

  const currentStageIndex = status.stage ? stages.indexOf(status.stage) : -1

  const StatusIcon = () => {
    switch (status.status) {
      case 'completed':
        return <CheckCircle size={20} className="text-green-400" />
      case 'error':
        return <XCircle size={20} className="text-red-400" />
      case 'cancelled':
        return <AlertCircle size={20} className="text-gray-400" />
      case 'processing':
        return <Loader2 size={20} className="text-blue-400 animate-spin" />
      default:
        return <Clock size={20} className="text-yellow-400" />
    }
  }

  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 bg-gray-800 rounded-lg">
        <StatusIcon />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium truncate">
              {status.status === 'completed' ? 'Complete' :
               status.status === 'error' ? 'Failed' :
               status.stage ? stageConfig[status.stage].label : 'Queued'}
            </span>
            <span className="text-xs text-gray-400">{status.progress}%</span>
          </div>
          <div className="h-1.5 bg-gray-700 rounded-full mt-1.5 overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full transition-all duration-300',
                status.status === 'completed' ? 'bg-green-500' :
                status.status === 'error' ? 'bg-red-500' :
                'bg-blue-500'
              )}
              style={{ width: `${status.progress}%` }}
            />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <StatusIcon />
            <div>
              <h3 className="font-semibold">
                {status.status === 'completed' ? 'Conversion Complete' :
                 status.status === 'error' ? 'Conversion Failed' :
                 status.status === 'cancelled' ? 'Conversion Cancelled' :
                 'Converting...'}
              </h3>
              <p className="text-sm text-gray-400">
                {status.message || `Job: ${jobId.slice(0, 8)}...`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {status.pipeline_type && <PipelineBadge pipeline={status.pipeline_type} />}
            {status.adapter_type && <AdapterBadge adapterType={status.adapter_type} />}
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="p-4">
        {/* Stage indicators */}
        <div className="flex items-center justify-between mb-4">
          {stages.map((stage, idx) => (
            <div key={stage} className="flex items-center">
              <div className={clsx(
                'flex items-center justify-center w-8 h-8 rounded-full border-2 transition-all',
                idx < currentStageIndex
                  ? 'bg-green-500 border-green-500 text-white'
                  : idx === currentStageIndex
                  ? 'bg-blue-500 border-blue-500 text-white'
                  : 'bg-gray-700 border-gray-600 text-gray-500'
              )}>
                {idx < currentStageIndex ? (
                  <CheckCircle size={14} />
                ) : idx === currentStageIndex && status.status === 'processing' ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  stageConfig[stage].icon
                )}
              </div>
              {idx < stages.length - 1 && (
                <div className={clsx(
                  'w-8 sm:w-12 h-0.5 mx-1',
                  idx < currentStageIndex ? 'bg-green-500' : 'bg-gray-700'
                )} />
              )}
            </div>
          ))}
        </div>

        {/* Stage labels (hidden on mobile) */}
        <div className="hidden sm:flex items-center justify-between text-xs text-gray-500 mb-4">
          {stages.map((stage, idx) => (
            <span key={stage} className={clsx(
              'w-8 text-center',
              idx === currentStageIndex && 'text-blue-400 font-medium'
            )}>
              {stageConfig[stage].label.split(' ')[0]}
            </span>
          ))}
        </div>

        {/* Progress bar */}
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={clsx(
              'h-full rounded-full transition-all duration-300',
              status.status === 'completed' ? 'bg-green-500' :
              status.status === 'error' ? 'bg-red-500' :
              'bg-blue-500'
            )}
            style={{ width: `${status.progress}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>{status.progress}%</span>
          {status.status === 'processing' && status.stage && (
            <span>{stageConfig[status.stage].label}</span>
          )}
        </div>
      </div>

      {/* Error display */}
      {status.status === 'error' && status.error && (
        <div className="px-4 pb-4">
          <div className="p-3 bg-red-900/30 border border-red-500/30 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertCircle size={16} className="text-red-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm text-red-300 font-medium">Conversion Error</p>
                <p className="text-xs text-red-400 mt-1">{status.error}</p>
                {status.error.includes('adapter') && (
                  <p className="text-xs text-gray-400 mt-2">
                    Tip: Train a model for this profile or select a different adapter type.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Completion metrics */}
      {status.status === 'completed' && (
        <div className="px-4 pb-4 space-y-4">
          {/* Quality Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {status.processing_time_seconds != null && (
              <div className="bg-gray-750 rounded-lg p-3">
                <div className="flex items-center gap-1.5 text-gray-400 text-xs mb-1">
                  <Timer size={12} />
                  Processing Time
                </div>
                <div className="font-mono text-lg">
                  {formatDuration(status.processing_time_seconds)}
                </div>
              </div>
            )}
            {status.rtf != null && (
              <div className="bg-gray-750 rounded-lg p-3">
                <div className="flex items-center gap-1.5 text-gray-400 text-xs mb-1">
                  <Zap size={12} />
                  Speed (RTF)
                </div>
                <div className={clsx(
                  'font-mono text-lg',
                  status.rtf < 1 ? 'text-green-400' : 'text-yellow-400'
                )}>
                  {formatRTF(status.rtf)}
                </div>
              </div>
            )}
            {status.audio_duration_seconds != null && (
              <div className="bg-gray-750 rounded-lg p-3">
                <div className="flex items-center gap-1.5 text-gray-400 text-xs mb-1">
                  <Music size={12} />
                  Audio Length
                </div>
                <div className="font-mono text-lg">
                  {formatDuration(status.audio_duration_seconds)}
                </div>
              </div>
            )}
            {status.pipeline_type && (
              <div className="bg-gray-750 rounded-lg p-3">
                <div className="flex items-center gap-1.5 text-gray-400 text-xs mb-1">
                  <Cpu size={12} />
                  Pipeline
                </div>
                <div className="mt-1">
                  <PipelineBadge pipeline={status.pipeline_type} />
                </div>
              </div>
            )}
          </div>

          {/* Download button */}
          {showDownload && status.download_url && (
            <a
              href={status.download_url}
              download
              className="flex items-center justify-center gap-2 w-full px-4 py-2.5 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
            >
              <Download size={16} />
              Download Converted Audio
            </a>
          )}
        </div>
      )}
    </div>
  )
}

// Compact inline progress for lists
export function ConversionProgressInline({
  stage,
  progress,
}: {
  jobId?: string  // Reserved for linking to full progress view
  stage?: ConversionStage
  progress: number
}) {
  const stageLabel = stage ? stageConfig[stage].label : undefined

  return (
    <div className="flex items-center gap-2" title={stageLabel}>
      {progress < 100 ? (
        <Loader2 size={14} className="text-blue-400 animate-spin" />
      ) : (
        <CheckCircle size={14} className="text-green-400" />
      )}
      <div className="flex-1">
        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
      <span className="text-xs text-gray-400 w-8 text-right">{progress}%</span>
    </div>
  )
}

// Metrics summary badge for completed conversions
export function ConversionMetricsBadge({
  rtf,
  processingTime,
  pipelineType,
}: {
  rtf?: number
  processingTime?: number
  pipelineType?: PipelineType
}) {
  return (
    <div className="inline-flex items-center gap-2 px-2 py-1 bg-gray-750 rounded text-xs">
      {pipelineType && <PipelineBadge pipeline={pipelineType} />}
      {rtf != null && (
        <span className={clsx(
          'font-mono',
          rtf < 1 ? 'text-green-400' : 'text-yellow-400'
        )}>
          {formatRTF(rtf)}
        </span>
      )}
      {processingTime != null && (
        <span className="text-gray-400">
          {formatDuration(processingTime)}
        </span>
      )}
    </div>
  )
}
