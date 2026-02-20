import { useState } from 'react'
import { Split, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { SeparationConfig, DEFAULT_SEPARATION_CONFIG } from '../services/api'
import clsx from 'clsx'

interface SeparationConfigPanelProps {
  config: SeparationConfig
  onChange: (config: SeparationConfig) => void
  disabled?: boolean
}

const modelOptions = [
  { value: 'htdemucs', label: 'HTDemucs', description: 'Best overall quality' },
  { value: 'htdemucs_ft', label: 'HTDemucs Fine-tuned', description: 'Fine-tuned for vocals' },
  { value: 'mdx_extra', label: 'MDX Extra', description: 'Good for complex mixes' },
] as const

const stemOptions = [
  { value: 'vocals', label: 'Vocals', icon: '🎤' },
  { value: 'drums', label: 'Drums', icon: '🥁' },
  { value: 'bass', label: 'Bass', icon: '🎸' },
  { value: 'other', label: 'Other', icon: '🎹' },
] as const

export function SeparationConfigPanel({ config, onChange, disabled }: SeparationConfigPanelProps) {
  const [expanded, setExpanded] = useState(false)

  const update = <K extends keyof SeparationConfig>(key: K, value: SeparationConfig[K]) => {
    onChange({ ...config, [key]: value })
  }

  const toggleStem = (stem: SeparationConfig['stems'][number]) => {
    const stems = config.stems.includes(stem)
      ? config.stems.filter(s => s !== stem)
      : [...config.stems, stem]
    // Ensure at least one stem is selected
    if (stems.length > 0) {
      update('stems', stems as SeparationConfig['stems'])
    }
  }

  const resetToDefaults = () => {
    onChange(DEFAULT_SEPARATION_CONFIG)
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Split size={18} className="text-gray-400" />
          <h3 className="font-semibold">Vocal Separation</h3>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-sm text-gray-400 hover:text-white"
        >
          {expanded ? 'Collapse' : 'Advanced'}
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Model Selection */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">Separation Model</label>
        <div className="grid grid-cols-3 gap-2">
          {modelOptions.map(opt => (
            <button
              key={opt.value}
              onClick={() => update('model', opt.value)}
              disabled={disabled}
              title={opt.description}
              className={clsx(
                'px-3 py-2 text-xs rounded transition-colors text-center',
                config.model === opt.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stem Selection */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">Extract Stems</label>
        <div className="grid grid-cols-4 gap-2">
          {stemOptions.map(opt => (
            <button
              key={opt.value}
              onClick={() => toggleStem(opt.value)}
              disabled={disabled}
              className={clsx(
                'px-3 py-2 text-xs rounded transition-colors flex flex-col items-center gap-1',
                config.stems.includes(opt.value)
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <span>{opt.icon}</span>
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {expanded && (
        <div className="space-y-4 pt-2 border-t border-gray-700">
          {/* Shifts */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <label className="text-sm text-gray-400 flex items-center gap-1">
                Random Shifts
                <span title="Number of random shifts for prediction averaging (higher = better quality but slower)" className="cursor-help">
                  <Info size={12} className="text-gray-500" />
                </span>
              </label>
              <span className="text-sm font-mono">{config.shifts}</span>
            </div>
            <input
              type="range"
              value={config.shifts}
              onChange={e => update('shifts', Number(e.target.value))}
              min={1}
              max={10}
              step={1}
              disabled={disabled}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-blue-500"
            />
          </div>

          {/* Overlap */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <label className="text-sm text-gray-400 flex items-center gap-1">
                Chunk Overlap
                <span title="Overlap between audio chunks (higher = smoother but slower)" className="cursor-help">
                  <Info size={12} className="text-gray-500" />
                </span>
              </label>
              <span className="text-sm font-mono">{(config.overlap * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              value={config.overlap}
              onChange={e => update('overlap', Number(e.target.value))}
              min={0}
              max={0.5}
              step={0.05}
              disabled={disabled}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-blue-500"
            />
          </div>

          {/* Segment Length */}
          <div className="space-y-1">
            <label className="text-sm text-gray-400 flex items-center gap-1">
              Segment Length
              <span title="Process in segments (null = process full track, lower values use less memory)" className="cursor-help">
                <Info size={12} className="text-gray-500" />
              </span>
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => update('segment_length', null)}
                disabled={disabled}
                className={clsx(
                  'px-3 py-1.5 text-xs rounded transition-colors',
                  config.segment_length === null
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                  disabled && 'opacity-50 cursor-not-allowed'
                )}
              >
                Full Track
              </button>
              {[30, 60, 120].map(len => (
                <button
                  key={len}
                  onClick={() => update('segment_length', len)}
                  disabled={disabled}
                  className={clsx(
                    'px-3 py-1.5 text-xs rounded transition-colors',
                    config.segment_length === len
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                    disabled && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  {len}s
                </button>
              ))}
            </div>
          </div>

          {/* Device */}
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Processing Device</label>
            <div className="flex gap-2">
              {(['cuda', 'cpu'] as const).map(device => (
                <button
                  key={device}
                  onClick={() => update('device', device)}
                  disabled={disabled}
                  className={clsx(
                    'px-4 py-1.5 text-xs rounded transition-colors uppercase',
                    config.device === device
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                    disabled && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  {device}
                </button>
              ))}
            </div>
          </div>

          {/* Reset Button */}
          <div className="pt-2">
            <button
              onClick={resetToDefaults}
              disabled={disabled}
              className="text-sm text-gray-400 hover:text-white disabled:opacity-50"
            >
              Reset to Defaults
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
