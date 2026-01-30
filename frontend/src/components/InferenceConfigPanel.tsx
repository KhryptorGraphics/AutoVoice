import { useState } from 'react'
import { Sliders, Info, ChevronDown, ChevronUp, Zap } from 'lucide-react'
import {
  ConversionConfig,
  DEFAULT_CONVERSION_CONFIG,
  QualityPreset,
  EncoderBackend,
  VocoderType,
  QUALITY_PRESETS,
} from '../services/api'
import clsx from 'clsx'

interface InferenceConfigPanelProps {
  config: ConversionConfig
  onChange: (config: ConversionConfig) => void
  disabled?: boolean
  compact?: boolean
}

interface SliderInputProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  unit?: string
  tooltip?: string
  disabled?: boolean
  formatValue?: (v: number) => string
}

function SliderInput({ label, value, onChange, min, max, step, unit, tooltip, disabled, formatValue }: SliderInputProps) {
  const displayValue = formatValue ? formatValue(value) : `${value}${unit || ''}`

  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <label className="text-sm text-gray-400 flex items-center gap-1">
          {label}
          {tooltip && (
            <span title={tooltip} className="cursor-help">
              <Info size={12} className="text-gray-500" />
            </span>
          )}
        </label>
        <span className="text-sm font-mono">{displayValue}</span>
      </div>
      <input
        type="range"
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed accent-blue-500"
      />
    </div>
  )
}

interface SelectInputProps<T extends string> {
  label: string
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string; description?: string }[]
  tooltip?: string
  disabled?: boolean
}

function SelectInput<T extends string>({ label, value, onChange, options, tooltip, disabled }: SelectInputProps<T>) {
  return (
    <div className="space-y-1">
      <label className="text-sm text-gray-400 flex items-center gap-1">
        {label}
        {tooltip && (
          <span title={tooltip} className="cursor-help">
            <Info size={12} className="text-gray-500" />
          </span>
        )}
      </label>
      <select
        value={value}
        onChange={e => onChange(e.target.value as T)}
        disabled={disabled}
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
      >
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  )
}

interface ToggleProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  tooltip?: string
  disabled?: boolean
}

function Toggle({ label, checked, onChange, tooltip, disabled }: ToggleProps) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-sm text-gray-400 flex items-center gap-1">
        {label}
        {tooltip && (
          <span title={tooltip} className="cursor-help">
            <Info size={12} className="text-gray-500" />
          </span>
        )}
      </label>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        disabled={disabled}
        className={clsx(
          'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
          checked ? 'bg-blue-600' : 'bg-gray-600',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <span
          className={clsx(
            'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
            checked ? 'translate-x-6' : 'translate-x-1'
          )}
        />
      </button>
    </div>
  )
}

const presetOptions: { value: QualityPreset; label: string; description: string }[] = [
  { value: 'draft', label: 'Draft', description: 'Fast preview (10 steps)' },
  { value: 'fast', label: 'Fast', description: 'Quick conversion (20 steps)' },
  { value: 'balanced', label: 'Balanced', description: 'Good quality (50 steps)' },
  { value: 'high', label: 'High Quality', description: 'High fidelity (100 steps)' },
  { value: 'studio', label: 'Studio', description: 'Maximum quality (200 steps)' },
]

const encoderOptions: { value: EncoderBackend; label: string; description: string }[] = [
  { value: 'hubert', label: 'HuBERT-Soft', description: 'Better for singing' },
  { value: 'contentvec', label: 'ContentVec', description: 'Better speaker separation' },
]

const vocoderOptions: { value: VocoderType; label: string; description: string }[] = [
  { value: 'hifigan', label: 'HiFi-GAN', description: 'Fast, good quality' },
  { value: 'bigvgan', label: 'BigVGAN', description: 'Better singing quality' },
]

export function InferenceConfigPanel({ config, onChange, disabled, compact }: InferenceConfigPanelProps) {
  const [expanded, setExpanded] = useState(!compact)

  const update = <K extends keyof ConversionConfig>(key: K, value: ConversionConfig[K]) => {
    onChange({ ...config, [key]: value })
  }

  const resetToDefaults = () => {
    onChange(DEFAULT_CONVERSION_CONFIG)
  }

  const currentPreset = QUALITY_PRESETS[config.preset]

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders size={18} className="text-gray-400" />
          <h3 className="font-semibold">Conversion Settings</h3>
        </div>
        {compact && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-sm text-gray-400 hover:text-white"
          >
            {expanded ? 'Collapse' : 'Expand'}
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        )}
      </div>

      {/* Quality Preset - Always Visible */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">Quality Preset</label>
        <div className="grid grid-cols-5 gap-1">
          {presetOptions.map(opt => (
            <button
              key={opt.value}
              onClick={() => update('preset', opt.value)}
              disabled={disabled}
              title={opt.description}
              className={clsx(
                'px-2 py-2 text-xs rounded transition-colors',
                config.preset === opt.value
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
          {currentPreset.n_steps} steps · {(currentPreset.denoise * 100).toFixed(0)}% denoise
        </div>
      </div>

      {(!compact || expanded) && (
        <>
          {/* Volume Controls */}
          <div className="space-y-4 pt-2 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300">Volume Mix</h4>
            <SliderInput
              label="Vocal Volume"
              value={config.vocal_volume}
              onChange={v => update('vocal_volume', v)}
              min={0}
              max={2}
              step={0.05}
              formatValue={v => `${(v * 100).toFixed(0)}%`}
              tooltip="Volume of converted vocals"
              disabled={disabled}
            />
            <SliderInput
              label="Instrumental Volume"
              value={config.instrumental_volume}
              onChange={v => update('instrumental_volume', v)}
              min={0}
              max={2}
              step={0.05}
              formatValue={v => `${(v * 100).toFixed(0)}%`}
              tooltip="Volume of backing track"
              disabled={disabled}
            />
          </div>

          {/* Pitch Control */}
          <div className="space-y-4 pt-2 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300">Pitch</h4>
            <SliderInput
              label="Pitch Shift"
              value={config.pitch_shift}
              onChange={v => update('pitch_shift', v)}
              min={-12}
              max={12}
              step={1}
              unit=" semitones"
              tooltip="Shift pitch up or down (0 = no change)"
              disabled={disabled}
            />
          </div>

          {/* Model Selection */}
          <div className="space-y-4 pt-2 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300">Model Selection</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <SelectInput
                label="Encoder"
                value={config.encoder_backend}
                onChange={v => update('encoder_backend', v)}
                options={encoderOptions}
                tooltip="Content encoder for feature extraction"
                disabled={disabled}
              />
              <SelectInput
                label="Vocoder"
                value={config.vocoder_type}
                onChange={v => update('vocoder_type', v)}
                options={vocoderOptions}
                tooltip="Neural vocoder for audio synthesis"
                disabled={disabled}
              />
            </div>
          </div>

          {/* Advanced Options */}
          <div className="space-y-4 pt-2 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300">Advanced</h4>
            <Toggle
              label="Return Stems"
              checked={config.return_stems}
              onChange={v => update('return_stems', v)}
              tooltip="Return separated vocals and instrumentals as individual files"
              disabled={disabled}
            />
            <Toggle
              label="Preserve Techniques"
              checked={config.preserve_techniques}
              onChange={v => update('preserve_techniques', v)}
              tooltip="Preserve singing techniques like vibrato and melisma"
              disabled={disabled}
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
        </>
      )}
    </div>
  )
}

// Compact preset selector for inline use
export function PresetSelector({
  value,
  onChange,
  disabled,
}: {
  value: QualityPreset
  onChange: (preset: QualityPreset) => void
  disabled?: boolean
}) {
  return (
    <div className="flex items-center gap-2">
      <Zap size={14} className="text-gray-400" />
      <select
        value={value}
        onChange={e => onChange(e.target.value as QualityPreset)}
        disabled={disabled}
        className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
      >
        {presetOptions.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  )
}
