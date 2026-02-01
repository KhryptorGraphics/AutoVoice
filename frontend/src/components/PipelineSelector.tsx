import { useState } from 'react'
import { Zap, Sparkles, HelpCircle, Crown, Radio } from 'lucide-react'
import clsx from 'clsx'

export type PipelineType = 'realtime' | 'quality' | 'quality_seedvc' | 'realtime_meanvc'

interface PipelineSelectorProps {
  value: PipelineType
  onChange: (value: PipelineType) => void
  disabled?: boolean
  showDescription?: boolean
  size?: 'sm' | 'md' | 'lg'
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
  },
  {
    id: 'realtime_meanvc',
    name: 'Streaming (MeanVC)',
    icon: <Radio className="w-4 h-4" />,
    description: 'Single-step mean flow inference with KV-cache streaming',
    latency: '<80ms',
    quality: 'Good',
    sampleRate: '16kHz',
    bestFor: 'True streaming, real-time voice chat, low latency',
  },
]

export function PipelineSelector({
  value,
  onChange,
  disabled = false,
  showDescription = true,
  size = 'md',
}: PipelineSelectorProps) {
  const [showTooltip, setShowTooltip] = useState<PipelineType | null>(null)

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
            <div className="absolute z-10 w-64 p-3 text-xs bg-gray-800 border border-gray-700 rounded-lg shadow-xl left-6 top-0">
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
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        {pipelines.map((pipeline) => (
          <button
            key={pipeline.id}
            onClick={() => !disabled && onChange(pipeline.id)}
            disabled={disabled}
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
          </button>
        ))}
      </div>

      {showDescription && (
        <p className="text-xs text-gray-500">
          {pipelines.find(p => p.id === value)?.bestFor}
        </p>
      )}
    </div>
  )
}

// Compact dropdown version for toolbar use
interface PipelineDropdownProps {
  value: PipelineType
  onChange: (value: PipelineType) => void
  disabled?: boolean
}

export function PipelineDropdown({ value, onChange, disabled }: PipelineDropdownProps) {
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
      {pipelines.map((pipeline) => (
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
}

export function PipelineBadge({ pipeline }: PipelineBadgeProps) {
  const info = pipelines.find(p => p.id === pipeline)
  if (!info) return null

  const colorClasses = {
    realtime: 'bg-yellow-900/50 text-yellow-300',
    quality: 'bg-violet-900/50 text-violet-300',
    quality_seedvc: 'bg-amber-900/50 text-amber-300',
    realtime_meanvc: 'bg-cyan-900/50 text-cyan-300',
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
    </span>
  )
}
