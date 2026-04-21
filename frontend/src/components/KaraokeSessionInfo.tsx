/**
 * KaraokeSessionInfo - Displays current session information during live performance.
 * Shows pipeline type, voice profile, adapter, and real-time latency metrics.
 */
import { useState, useEffect, useRef } from 'react'
import { Activity, User, Timer, Wifi, WifiOff, Cpu, Route, Database } from 'lucide-react'
import clsx from 'clsx'
import { PipelineBadge, type PipelineType } from './PipelineSelector'
import { AdapterBadge } from './AdapterSelector'
import { AdapterType, ActiveModelType } from '../services/api'

interface KaraokeSessionInfoProps {
  requestedPipeline: PipelineType
  resolvedPipeline?: PipelineType
  runtimeBackend?: string | null
  profileName?: string
  adapterType?: AdapterType | null
  modelType?: ActiveModelType
  latencyMs: number
  isConnected: boolean
  isStreaming: boolean
  chunksProcessed: number
  sampleCollectionEnabled?: boolean
  audioRouterTargets?: {
    speaker_device: number | null
    headphone_device: number | null
  } | null
}

// Real-time latency tracker with rolling average
function useLatencyStats(latencyMs: number) {
  const [stats, setStats] = useState({
    current: 0,
    average: 0,
    min: Infinity,
    max: 0,
    jitter: 0,
  })
  const samplesRef = useRef<number[]>([])
  const maxSamples = 100 // Rolling window

  useEffect(() => {
    if (latencyMs <= 0) return

    samplesRef.current.push(latencyMs)
    if (samplesRef.current.length > maxSamples) {
      samplesRef.current.shift()
    }

    const samples = samplesRef.current
    const avg = samples.reduce((a, b) => a + b, 0) / samples.length
    const min = Math.min(...samples)
    const max = Math.max(...samples)

    // Calculate jitter (standard deviation)
    const variance = samples.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / samples.length
    const jitter = Math.sqrt(variance)

    setStats({
      current: latencyMs,
      average: avg,
      min,
      max,
      jitter,
    })
  }, [latencyMs])

  return stats
}

// Latency quality indicator
function getLatencyQuality(latencyMs: number): { label: string; color: string } {
  if (latencyMs < 50) return { label: 'Excellent', color: 'text-green-400' }
  if (latencyMs < 100) return { label: 'Good', color: 'text-green-300' }
  if (latencyMs < 150) return { label: 'Acceptable', color: 'text-yellow-400' }
  if (latencyMs < 250) return { label: 'High', color: 'text-orange-400' }
  return { label: 'Poor', color: 'text-red-400' }
}

export function KaraokeSessionInfo({
  requestedPipeline,
  resolvedPipeline,
  runtimeBackend,
  profileName,
  adapterType,
  modelType,
  latencyMs,
  isConnected,
  isStreaming,
  chunksProcessed,
  sampleCollectionEnabled = false,
  audioRouterTargets,
}: KaraokeSessionInfoProps) {
  const latencyStats = useLatencyStats(latencyMs)
  const quality = getLatencyQuality(latencyStats.average)

  return (
    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Activity size={16} className="text-violet-400" />
          Session Info
        </h4>
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Wifi size={16} className="text-green-400" />
          ) : (
            <WifiOff size={16} className="text-red-400" />
          )}
          <span className={clsx(
            'text-xs font-medium px-2 py-0.5 rounded',
            isStreaming
              ? 'bg-green-600/30 text-green-400'
              : 'bg-gray-700 text-gray-400'
          )}>
            {isStreaming ? 'LIVE' : 'Ready'}
          </span>
        </div>
      </div>

      {/* Pipeline and Profile badges */}
      <div className="flex flex-wrap gap-2 mb-4">
        <PipelineBadge pipeline={requestedPipeline} />
        {resolvedPipeline && resolvedPipeline !== requestedPipeline && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-cyan-900/50 text-cyan-300">
            <Route size={12} />
            Resolved {resolvedPipeline}
          </span>
        )}
        {runtimeBackend && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-gray-700 text-gray-200">
            <Cpu size={12} />
            {runtimeBackend}
          </span>
        )}
        {profileName && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-blue-900/50 text-blue-300">
            <User size={12} />
            {profileName}
          </span>
        )}
        {modelType && (
          <span className={clsx(
            'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
            modelType === 'full_model'
              ? 'bg-violet-900/50 text-violet-300'
              : modelType === 'adapter'
                ? 'bg-emerald-900/50 text-emerald-300'
                : 'bg-gray-700 text-gray-300'
          )}>
            {modelType === 'full_model' ? 'Full model' : modelType === 'adapter' ? 'LoRA target' : 'Base'}
          </span>
        )}
        {adapterType && <AdapterBadge adapterType={adapterType} />}
        {sampleCollectionEnabled && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-900/50 text-emerald-300">
            <Database size={12} />
            Sample capture on
          </span>
        )}
      </div>

      {/* Real-time latency display */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400 flex items-center gap-1">
            <Timer size={14} />
            Latency
          </span>
          <div className="text-right">
            <span className={clsx('text-2xl font-bold', quality.color)}>
              {latencyStats.current.toFixed(0)}
            </span>
            <span className="text-gray-500 text-sm ml-1">ms</span>
          </div>
        </div>

        {/* Latency bar visualization */}
        <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={clsx(
              'h-full transition-all duration-100',
              latencyStats.average < 100 ? 'bg-green-500' :
              latencyStats.average < 200 ? 'bg-yellow-500' : 'bg-red-500'
            )}
            style={{ width: `${Math.min(100, (latencyStats.current / 300) * 100)}%` }}
          />
          {/* Target line at 100ms */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/30"
            style={{ left: `${(100 / 300) * 100}%` }}
          />
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-4 gap-2 text-xs">
          <div className="text-center">
            <div className="text-gray-500">Avg</div>
            <div className="text-gray-200">{latencyStats.average.toFixed(0)}ms</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Min</div>
            <div className="text-green-400">{latencyStats.min === Infinity ? '-' : `${latencyStats.min.toFixed(0)}ms`}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Max</div>
            <div className="text-orange-400">{latencyStats.max === 0 ? '-' : `${latencyStats.max.toFixed(0)}ms`}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Jitter</div>
            <div className="text-gray-300">{latencyStats.jitter.toFixed(1)}ms</div>
          </div>
        </div>

        {/* Quality indicator */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Quality</span>
          <span className={quality.color}>{quality.label}</span>
        </div>

        {/* Chunks processed */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Chunks processed</span>
          <span className="text-gray-300">{chunksProcessed.toLocaleString()}</span>
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Routing</span>
          <span className="text-gray-300">
            SPK {audioRouterTargets?.speaker_device ?? 'sys'} · HP {audioRouterTargets?.headphone_device ?? 'sys'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default KaraokeSessionInfo
