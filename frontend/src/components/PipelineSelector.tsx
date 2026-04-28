import { useState } from 'react'
import { Zap, Sparkles, HelpCircle, Crown, Radio, Rocket, BarChart3, X } from 'lucide-react'
import clsx from 'clsx'
import { STORAGE_KEYS } from '../hooks/usePersistedState'
import type { LivePipelineType, OfflinePipelineType, PipelineType } from '../services/api'

interface PipelineRuntimeStatus {
  loaded?: boolean
  memory_gb?: number
  latency_target_ms?: number
  sample_rate?: number
  description?: string
}

// LocalStorage key for persisting user's preferred pipeline
const PIPELINE_PREFERENCE_KEY = STORAGE_KEYS.PIPELINE_PREFERENCE

export function getPreferredPipeline(): PipelineType | null {
  try {
    const saved = localStorage.getItem(PIPELINE_PREFERENCE_KEY)
    if (saved && isPipelineType(saved)) {
      return saved as PipelineType
    }
  } catch (e) {
    console.error('Failed to read pipeline preference:', e)
  }
  return null
}

export function savePreferredPipeline(pipeline: PipelineType): void {
  try {
    localStorage.setItem(PIPELINE_PREFERENCE_KEY, pipeline)
  } catch (e) {
    console.error('Failed to save pipeline preference:', e)
  }
}

export type PipelineSelectionContext = 'offline' | 'live'

export function isPipelineType(value: string): value is PipelineType {
  return (
    value === 'realtime'
    || value === 'quality'
    || value === 'quality_seedvc'
    || value === 'realtime_meanvc'
    || value === 'quality_shortcut'
  )
}

export function isOfflinePipeline(value: string): value is OfflinePipelineType {
  return (
    value === 'realtime'
    || value === 'quality'
    || value === 'quality_seedvc'
    || value === 'quality_shortcut'
  )
}

export function isLivePipeline(value: string): value is LivePipelineType {
  return value === 'realtime' || value === 'realtime_meanvc'
}

interface PipelineSelectorProps {
  value: PipelineType
  onChange: (value: PipelineType) => void
  context?: PipelineSelectionContext
  disabled?: boolean
  showDescription?: boolean
  size?: 'sm' | 'md' | 'lg'
  statusByPipeline?: Partial<Record<PipelineType, PipelineRuntimeStatus>>
  showCompareButton?: boolean
}

interface PipelineInfo {
  id: PipelineType
  name: string
  icon: React.ReactNode
  description: string
  latency: string
  quality: string
  sampleRate: string
  bestFor: string
  rtf?: string // Real-time factor (lower = faster)
  speedScore: number // 1-5 scale for comparison
  qualityScore: number // 1-5 scale for comparison
}

const pipelines: PipelineInfo[] = [
  {
    id: 'realtime',
    name: 'Realtime',
    icon: <Zap className="w-4 h-4" />,
    description: 'Low-latency pipeline optimized for live karaoke',
    latency: '<100ms',
    quality: 'Good',
    sampleRate: '22kHz',
    bestFor: 'Live performance, karaoke',
    rtf: '0.1x',
    speedScore: 5,
    qualityScore: 3,
  },
  {
    id: 'quality',
    name: 'Quality',
    icon: <Sparkles className="w-4 h-4" />,
    description: 'High-fidelity CoMoSVC with 30-step diffusion',
    latency: '~2-5s',
    quality: 'Excellent',
    sampleRate: '24kHz',
    bestFor: 'Song conversion, production',
    rtf: '0.3x',
    speedScore: 2,
    qualityScore: 4,
  },
  {
    id: 'quality_seedvc',
    name: 'SOTA (Seed-VC)',
    icon: <Crown className="w-4 h-4" />,
    description: 'State-of-the-art DiT-CFM with in-context learning',
    latency: '~1-3s',
    quality: 'Maximum',
    sampleRate: '44.1kHz',
    bestFor: 'Best quality, reference-based conversion',
    rtf: '0.5-0.6x',
    speedScore: 2,
    qualityScore: 5,
  },
  {
    id: 'realtime_meanvc',
    name: 'Streaming (MeanVC)',
    icon: <Radio className="w-4 h-4" />,
    description: 'CPU-friendly mean flow inference with KV-cache streaming',
    latency: '<350ms',
    quality: 'Good',
    sampleRate: '16kHz',
    bestFor: 'CPU-friendly live conversion and edge validation',
    rtf: '<2.0x',
    speedScore: 3,
    qualityScore: 3,
  },
  {
    id: 'quality_shortcut',
    name: 'Fast Quality',
    icon: <Rocket className="w-4 h-4" />,
    description: '2-step shortcut inference - 2.83x faster than standard, 92%+ quality retention',
    latency: '~0.5-1s',
    quality: 'Very Good',
    sampleRate: '44.1kHz',
    bestFor: 'Quick high-quality conversion, batch processing',
    rtf: '0.2x',
    speedScore: 4,
    qualityScore: 4,
  },
]

export function getSelectablePipelines(context: PipelineSelectionContext): PipelineInfo[] {
  if (context === 'live') {
    return pipelines.filter((pipeline) => isLivePipeline(pipeline.id))
  }
  return pipelines.filter((pipeline) => isOfflinePipeline(pipeline.id))
}

/**
 * Get pipeline info by ID
 */
export function getPipelineInfo(pipelineId: PipelineType): PipelineInfo | undefined {
  return pipelines.find(p => p.id === pipelineId)
}

export function PipelineSelector({
  value,
  onChange,
  context = 'offline',
  disabled = false,
  showDescription = true,
  size = 'md',
  statusByPipeline,
  showCompareButton = true,
}: PipelineSelectorProps) {
  const [showTooltip, setShowTooltip] = useState<PipelineType | null>(null)
  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false)
  const selectablePipelines = getSelectablePipelines(context)

  // Save preference when user changes pipeline
  const handleChange = (newPipeline: PipelineType) => {
    if (disabled) return
    savePreferredPipeline(newPipeline)
    onChange(newPipeline)
  }

  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-2 text-sm',
    lg: 'px-4 py-3 text-base',
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-gray-300">Pipeline</label>
        <div
          className="relative"
          onMouseEnter={() => setShowTooltip(value)}
          onMouseLeave={() => setShowTooltip(null)}
        >
          <HelpCircle className="w-4 h-4 text-gray-500 cursor-help" />
          {showTooltip && (
            <div className="absolute z-10 w-72 p-3 text-xs bg-gray-800 border border-gray-700 rounded-lg shadow-xl left-6 top-0">
              <div className="font-medium text-white mb-1">
                {pipelines.find(p => p.id === showTooltip)?.name} Pipeline
              </div>
              <div className="text-gray-400">
                {pipelines.find(p => p.id === showTooltip)?.description}
              </div>
              <div className="mt-2 grid grid-cols-2 gap-1 text-gray-500">
                <div>Latency: <span className="text-gray-300">{pipelines.find(p => p.id === showTooltip)?.latency}</span></div>
                <div>Quality: <span className="text-gray-300">{pipelines.find(p => p.id === showTooltip)?.quality}</span></div>
                <div>Sample Rate: <span className="text-gray-300">{pipelines.find(p => p.id === showTooltip)?.sampleRate}</span></div>
                {pipelines.find(p => p.id === showTooltip)?.rtf && (
                  <div>RTF: <span className="text-gray-300">{pipelines.find(p => p.id === showTooltip)?.rtf}</span></div>
                )}
              </div>
            </div>
          )}
        </div>
        {showCompareButton && (
          <button
            onClick={() => setShowBenchmarkModal(true)}
            className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1"
          >
            <BarChart3 className="w-3 h-3" />
            Compare
          </button>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {selectablePipelines.map((pipeline) => {
          const runtimeStatus = statusByPipeline?.[pipeline.id]
          return (
            <button
              key={pipeline.id}
              onClick={() => handleChange(pipeline.id)}
              disabled={disabled}
              data-testid={`pipeline-option-${pipeline.id}`}
              className={clsx(
                'flex items-center gap-2 rounded-lg border transition-all',
                sizeClasses[size],
                value === pipeline.id
                  ? 'bg-violet-600 border-violet-500 text-white'
                  : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-600',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              {pipeline.icon}
              <span className="font-medium">{pipeline.name}</span>
              {runtimeStatus?.loaded !== undefined && (
                <span
                  className={clsx(
                    'h-2 w-2 rounded-full',
                    runtimeStatus.loaded ? 'bg-emerald-400' : 'bg-gray-500'
                  )}
                  aria-label={runtimeStatus.loaded ? 'Pipeline loaded' : 'Pipeline not loaded'}
                />
              )}
            </button>
          )
        })}
      </div>

      {showDescription && (
        <div className="space-y-1" data-testid="pipeline-status-summary">
          <p className="text-xs text-gray-500">
            {pipelines.find(p => p.id === value)?.bestFor}
          </p>
          {statusByPipeline?.[value] && (
            <p className="text-xs text-gray-400">
              {statusByPipeline[value]?.loaded ? 'Loaded' : 'Standby'}
              {statusByPipeline[value]?.memory_gb != null && ` · ${statusByPipeline[value]?.memory_gb?.toFixed(1)} GB`}
              {statusByPipeline[value]?.latency_target_ms != null && ` · ${statusByPipeline[value]?.latency_target_ms} ms target`}
            </p>
          )}
        </div>
      )}

      {/* Benchmark Comparison Modal */}
      {showBenchmarkModal && (
        <PipelineBenchmarkModal
          currentPipeline={value}
          context={context}
          onSelect={(p) => {
            handleChange(p)
            setShowBenchmarkModal(false)
          }}
          onClose={() => setShowBenchmarkModal(false)}
        />
      )}
    </div>
  )
}

// Benchmark comparison modal component
interface BenchmarkModalProps {
  currentPipeline: PipelineType
  context: PipelineSelectionContext
  onSelect: (pipeline: PipelineType) => void
  onClose: () => void
}

function PipelineBenchmarkModal({ currentPipeline, context, onSelect, onClose }: BenchmarkModalProps) {
  const selectablePipelines = getSelectablePipelines(context)
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-2xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Pipeline Comparison</h3>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-white rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-2 px-2">Pipeline</th>
                <th className="text-center py-2 px-2">Speed</th>
                <th className="text-center py-2 px-2">Quality</th>
                <th className="text-center py-2 px-2">Latency</th>
                <th className="text-center py-2 px-2">Sample Rate</th>
                <th className="text-right py-2 px-2">Best For</th>
              </tr>
            </thead>
            <tbody>
              {selectablePipelines.map((pipeline) => (
                <tr
                  key={pipeline.id}
                  onClick={() => onSelect(pipeline.id)}
                  className={clsx(
                    'cursor-pointer border-b border-gray-700/50 transition-colors',
                    currentPipeline === pipeline.id
                      ? 'bg-violet-600/20'
                      : 'hover:bg-gray-700/50'
                  )}
                >
                  <td className="py-3 px-2">
                    <div className="flex items-center gap-2">
                      <span className={clsx(
                        'p-1 rounded',
                        currentPipeline === pipeline.id ? 'bg-violet-600' : 'bg-gray-700'
                      )}>
                        {pipeline.icon}
                      </span>
                      <div>
                        <div className="font-medium text-white">{pipeline.name}</div>
                        {currentPipeline === pipeline.id && (
                          <div className="text-xs text-violet-400">Current</div>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="py-3 px-2 text-center">
                    <div className="flex justify-center gap-0.5">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className={clsx(
                            'w-2 h-4 rounded-sm',
                            i < pipeline.speedScore ? 'bg-green-500' : 'bg-gray-700'
                          )}
                        />
                      ))}
                    </div>
                  </td>
                  <td className="py-3 px-2 text-center">
                    <div className="flex justify-center gap-0.5">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className={clsx(
                            'w-2 h-4 rounded-sm',
                            i < pipeline.qualityScore ? 'bg-amber-500' : 'bg-gray-700'
                          )}
                        />
                      ))}
                    </div>
                  </td>
                  <td className="py-3 px-2 text-center text-gray-300">{pipeline.latency}</td>
                  <td className="py-3 px-2 text-center text-gray-300">{pipeline.sampleRate}</td>
                  <td className="py-3 px-2 text-right text-gray-400 text-xs">{pipeline.bestFor}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mt-4 p-3 bg-gray-700/50 rounded-lg text-xs text-gray-400">
            <p className="font-medium text-gray-300 mb-1">Understanding the metrics:</p>
            <ul className="space-y-1">
              <li><strong>Speed (RTF)</strong>: Real-Time Factor - lower is faster. 0.1x means 10x faster than real-time.</li>
              <li><strong>Quality</strong>: Perceptual quality rating from listening tests and objective metrics.</li>
              <li><strong>Latency</strong>: Time from input to output - critical for live/realtime applications.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// Compact dropdown version for toolbar use
interface PipelineDropdownProps {
  value: PipelineType
  onChange: (value: PipelineType) => void
  context?: PipelineSelectionContext
  disabled?: boolean
}

export function PipelineDropdown({ value, onChange, context = 'offline', disabled }: PipelineDropdownProps) {
  const selectablePipelines = getSelectablePipelines(context)
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as PipelineType)}
      disabled={disabled}
      className={clsx(
        'px-3 py-1.5 rounded-lg border text-sm',
        'bg-gray-800 border-gray-700 text-gray-200',
        'focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      {selectablePipelines.map((pipeline) => (
        <option key={pipeline.id} value={pipeline.id}>
          {pipeline.name} ({pipeline.latency})
        </option>
      ))}
    </select>
  )
}

// Display-only badge showing current pipeline
interface PipelineBadgeProps {
  pipeline: PipelineType
  showLatency?: boolean
}

export function PipelineBadge({ pipeline, showLatency = false }: PipelineBadgeProps) {
  const info = pipelines.find(p => p.id === pipeline)
  if (!info) return null

  const colorClasses: Record<PipelineType, string> = {
    realtime: 'bg-yellow-900/50 text-yellow-300',
    quality: 'bg-violet-900/50 text-violet-300',
    quality_seedvc: 'bg-amber-900/50 text-amber-300',
    realtime_meanvc: 'bg-cyan-900/50 text-cyan-300',
    quality_shortcut: 'bg-emerald-900/50 text-emerald-300',
  }

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
        colorClasses[pipeline] || 'bg-gray-900/50 text-gray-300'
      )}
    >
      {info.icon}
      {info.name}
      {showLatency && (
        <span className="opacity-75 ml-1">({info.latency})</span>
      )}
    </span>
  )
}

// Export all pipeline types for use in other components
export { pipelines }
export type { PipelineInfo, PipelineType, OfflinePipelineType, LivePipelineType }
