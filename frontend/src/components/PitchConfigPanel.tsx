import { useState } from 'react'
import { Music, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { PitchConfig, DEFAULT_PITCH_CONFIG } from '../services/api'
import clsx from 'clsx'

interface PitchConfigPanelProps {
  config: PitchConfig
  onChange: (config: PitchConfig) => void
  disabled?: boolean
}

const methodOptions = [
  { value: 'rmvpe', label: 'RMVPE', description: 'Best quality, GPU optimized' },
  { value: 'crepe', label: 'CREPE', description: 'High accuracy, slower' },
  { value: 'harvest', label: 'Harvest', description: 'CPU-friendly, good quality' },
  { value: 'dio', label: 'DIO', description: 'Fast, lower quality' },
] as const

export function PitchConfigPanel({ config, onChange, disabled }: PitchConfigPanelProps) {
  const [expanded, setExpanded] = useState(false)

  const update = <K extends keyof PitchConfig>(key: K, value: PitchConfig[K]) => {
    onChange({ ...config, [key]: value })
  }

  const resetToDefaults = () => {
    onChange(DEFAULT_PITCH_CONFIG)
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Music size={18} className="text-gray-400" />
          <h3 className="font-semibold">Pitch Extraction</h3>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-sm text-gray-400 hover:text-white"
        >
          {expanded ? 'Collapse' : 'Advanced'}
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Method Selection */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">Extraction Method</label>
        <div className="grid grid-cols-4 gap-2">
          {methodOptions.map(opt => (
            <button
              key={opt.value}
              onClick={() => update('method', opt.value)}
              disabled={disabled}
              title={opt.description}
              className={clsx(
                'px-3 py-2 text-xs rounded transition-colors text-center',
                config.method === opt.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="text-xs text-gray-500">
          {methodOptions.find(m => m.value === config.method)?.description}
        </div>
      </div>

      {/* GPU Toggle */}
      <div className="flex items-center justify-between">
        <label className="text-sm text-gray-400 flex items-center gap-1">
          Use GPU
          <span title="Enable GPU acceleration for pitch extraction" className="cursor-help">
            <Info size={12} className="text-gray-500" />
          </span>
        </label>
        <button
          type="button"
          onClick={() => update('use_gpu', !config.use_gpu)}
          disabled={disabled}
          className={clsx(
            'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
            config.use_gpu ? 'bg-blue-600' : 'bg-gray-600',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <span
            className={clsx(
              'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
              config.use_gpu ? 'translate-x-6' : 'translate-x-1'
            )}
          />
        </button>
      </div>

      {expanded && (
        <div className="space-y-4 pt-2 border-t border-gray-700">
          {/* F0 Range */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-300">Frequency Range (Hz)</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="flex justify-between items-center">
                  <label className="text-sm text-gray-400">Min F0</label>
                  <span className="text-sm font-mono">{config.f0_min} Hz</span>
                </div>
                <input
                  type="range"
                  value={config.f0_min}
                  onChange={e => update('f0_min', Number(e.target.value))}
                  min={20}
                  max={200}
                  step={5}
                  disabled={disabled}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-blue-500"
                />
              </div>
              <div className="space-y-1">
                <div className="flex justify-between items-center">
                  <label className="text-sm text-gray-400">Max F0</label>
                  <span className="text-sm font-mono">{config.f0_max} Hz</span>
                </div>
                <input
                  type="range"
                  value={config.f0_max}
                  onChange={e => update('f0_max', Number(e.target.value))}
                  min={400}
                  max={2000}
                  step={50}
                  disabled={disabled}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-blue-500"
                />
              </div>
            </div>
            <div className="text-xs text-gray-500">
              Typical range: Male 85-255 Hz, Female 165-500 Hz
            </div>
          </div>

          {/* Hop Length */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <label className="text-sm text-gray-400 flex items-center gap-1">
                Hop Length
                <span title="Frame shift in samples (lower = more precise but slower)" className="cursor-help">
                  <Info size={12} className="text-gray-500" />
                </span>
              </label>
              <span className="text-sm font-mono">{config.hop_length}</span>
            </div>
            <div className="flex gap-2">
              {[80, 160, 320, 512].map(hop => (
                <button
                  key={hop}
                  onClick={() => update('hop_length', hop)}
                  disabled={disabled}
                  className={clsx(
                    'flex-1 px-2 py-1.5 text-xs rounded transition-colors',
                    config.hop_length === hop
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                    disabled && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  {hop}
                </button>
              ))}
            </div>
          </div>

          {/* Confidence Threshold */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <label className="text-sm text-gray-400 flex items-center gap-1">
                Confidence Threshold
                <span title="Minimum confidence to accept pitch detection (higher = fewer false positives)" className="cursor-help">
                  <Info size={12} className="text-gray-500" />
                </span>
              </label>
              <span className="text-sm font-mono">{(config.threshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              value={config.threshold}
              onChange={e => update('threshold', Number(e.target.value))}
              min={0.1}
              max={0.9}
              step={0.05}
              disabled={disabled}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-blue-500"
            />
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
