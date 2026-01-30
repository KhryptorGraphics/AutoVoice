import { useState } from 'react'
import { Wand2, Settings, HelpCircle } from 'lucide-react'
import clsx from 'clsx'

export interface AugmentationConfig {
  enabled: boolean
  // Pitch variation
  pitch_shift_enabled: boolean
  pitch_shift_min: number // semitones
  pitch_shift_max: number // semitones
  // Time stretching
  time_stretch_enabled: boolean
  time_stretch_min: number // ratio (0.8 = 80% speed)
  time_stretch_max: number // ratio (1.2 = 120% speed)
  // EQ augmentation
  eq_enabled: boolean
  eq_variation: 'subtle' | 'moderate' | 'aggressive'
  // Noise injection
  noise_enabled: boolean
  noise_level: number // 0-100
  // Room simulation
  reverb_enabled: boolean
  reverb_amount: number // 0-100
}

interface AugmentationSettingsProps {
  config: AugmentationConfig
  onChange: (config: AugmentationConfig) => void
  disabled?: boolean
}

function SliderWithValue({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
  disabled,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  unit?: string
  onChange: (value: number) => void
  disabled?: boolean
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-400">{label}</span>
        <span className="font-mono text-xs">
          {value.toFixed(step < 1 ? 1 : 0)}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500 disabled:opacity-50"
      />
    </div>
  )
}

function Toggle({
  label,
  checked,
  onChange,
  disabled,
  tooltip,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  tooltip?: string
}) {
  return (
    <label className={clsx(
      'flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors',
      checked ? 'bg-blue-900/30' : 'bg-gray-750',
      disabled && 'opacity-50 cursor-not-allowed'
    )}>
      <div className="flex items-center gap-2">
        <span className="text-sm">{label}</span>
        {tooltip && (
          <span title={tooltip} className="text-gray-500 cursor-help">
            <HelpCircle size={12} />
          </span>
        )}
      </div>
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
          disabled={disabled}
          className="sr-only"
        />
        <div className={clsx(
          'w-10 h-6 rounded-full transition-colors',
          checked ? 'bg-blue-600' : 'bg-gray-600'
        )}>
          <div className={clsx(
            'absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform',
            checked && 'translate-x-4'
          )} />
        </div>
      </div>
    </label>
  )
}

export function AugmentationSettings({ config, onChange, disabled = false }: AugmentationSettingsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const updateConfig = (updates: Partial<AugmentationConfig>) => {
    onChange({ ...config, ...updates })
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Wand2 size={18} className="text-pink-400" />
          <h3 className="font-semibold">Data Augmentation</h3>
        </div>
        <Toggle
          label=""
          checked={config.enabled}
          onChange={enabled => updateConfig({ enabled })}
          disabled={disabled}
        />
      </div>

      {config.enabled && (
        <div className="space-y-4">
          <p className="text-xs text-gray-500">
            Augmentation creates variations of training samples to improve model generalization.
          </p>

          {/* Pitch Variation */}
          <div className={clsx(
            'p-3 rounded-lg border transition-colors',
            config.pitch_shift_enabled ? 'border-blue-500/50 bg-blue-900/10' : 'border-gray-700'
          )}>
            <Toggle
              label="Pitch Variation"
              checked={config.pitch_shift_enabled}
              onChange={pitch_shift_enabled => updateConfig({ pitch_shift_enabled })}
              disabled={disabled}
              tooltip="Randomly shift pitch within range to improve pitch robustness"
            />
            {config.pitch_shift_enabled && (
              <div className="mt-3 space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <SliderWithValue
                    label="Min Shift"
                    value={config.pitch_shift_min}
                    min={-12}
                    max={0}
                    step={0.5}
                    unit=" st"
                    onChange={pitch_shift_min => updateConfig({ pitch_shift_min })}
                    disabled={disabled}
                  />
                  <SliderWithValue
                    label="Max Shift"
                    value={config.pitch_shift_max}
                    min={0}
                    max={12}
                    step={0.5}
                    unit=" st"
                    onChange={pitch_shift_max => updateConfig({ pitch_shift_max })}
                    disabled={disabled}
                  />
                </div>
                <div className="text-xs text-gray-500">
                  Range: {config.pitch_shift_min} to +{config.pitch_shift_max} semitones
                </div>
              </div>
            )}
          </div>

          {/* Time Stretching */}
          <div className={clsx(
            'p-3 rounded-lg border transition-colors',
            config.time_stretch_enabled ? 'border-blue-500/50 bg-blue-900/10' : 'border-gray-700'
          )}>
            <Toggle
              label="Time Stretching"
              checked={config.time_stretch_enabled}
              onChange={time_stretch_enabled => updateConfig({ time_stretch_enabled })}
              disabled={disabled}
              tooltip="Randomly adjust tempo without affecting pitch"
            />
            {config.time_stretch_enabled && (
              <div className="mt-3 space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <SliderWithValue
                    label="Min Speed"
                    value={config.time_stretch_min}
                    min={0.5}
                    max={1.0}
                    step={0.05}
                    unit="x"
                    onChange={time_stretch_min => updateConfig({ time_stretch_min })}
                    disabled={disabled}
                  />
                  <SliderWithValue
                    label="Max Speed"
                    value={config.time_stretch_max}
                    min={1.0}
                    max={2.0}
                    step={0.05}
                    unit="x"
                    onChange={time_stretch_max => updateConfig({ time_stretch_max })}
                    disabled={disabled}
                  />
                </div>
                <div className="text-xs text-gray-500">
                  Range: {(config.time_stretch_min * 100).toFixed(0)}% to {(config.time_stretch_max * 100).toFixed(0)}% speed
                </div>
              </div>
            )}
          </div>

          {/* EQ Augmentation */}
          <div className={clsx(
            'p-3 rounded-lg border transition-colors',
            config.eq_enabled ? 'border-blue-500/50 bg-blue-900/10' : 'border-gray-700'
          )}>
            <Toggle
              label="EQ Variation"
              checked={config.eq_enabled}
              onChange={eq_enabled => updateConfig({ eq_enabled })}
              disabled={disabled}
              tooltip="Apply random EQ curves to simulate different recording environments"
            />
            {config.eq_enabled && (
              <div className="mt-3">
                <label className="text-sm text-gray-400 mb-2 block">Intensity</label>
                <div className="grid grid-cols-3 gap-2">
                  {(['subtle', 'moderate', 'aggressive'] as const).map(level => (
                    <button
                      key={level}
                      onClick={() => updateConfig({ eq_variation: level })}
                      disabled={disabled}
                      className={clsx(
                        'px-3 py-2 rounded-lg border text-sm capitalize transition-all',
                        config.eq_variation === level
                          ? 'border-blue-500 bg-blue-500/20'
                          : 'border-gray-700 hover:border-gray-600',
                        disabled && 'opacity-50 cursor-not-allowed'
                      )}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Advanced Options */}
          <div className="pt-2 border-t border-gray-700">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
            >
              <Settings size={14} />
              Advanced Augmentation
              <span className={clsx('transition-transform', showAdvanced && 'rotate-180')}>▼</span>
            </button>

            {showAdvanced && (
              <div className="mt-3 space-y-3">
                {/* Noise Injection */}
                <div className={clsx(
                  'p-3 rounded-lg border transition-colors',
                  config.noise_enabled ? 'border-blue-500/50 bg-blue-900/10' : 'border-gray-700'
                )}>
                  <Toggle
                    label="Background Noise"
                    checked={config.noise_enabled}
                    onChange={noise_enabled => updateConfig({ noise_enabled })}
                    disabled={disabled}
                    tooltip="Add subtle background noise for robustness"
                  />
                  {config.noise_enabled && (
                    <div className="mt-3">
                      <SliderWithValue
                        label="Noise Level"
                        value={config.noise_level}
                        min={0}
                        max={20}
                        step={1}
                        unit="%"
                        onChange={noise_level => updateConfig({ noise_level })}
                        disabled={disabled}
                      />
                    </div>
                  )}
                </div>

                {/* Reverb Simulation */}
                <div className={clsx(
                  'p-3 rounded-lg border transition-colors',
                  config.reverb_enabled ? 'border-blue-500/50 bg-blue-900/10' : 'border-gray-700'
                )}>
                  <Toggle
                    label="Room Reverb"
                    checked={config.reverb_enabled}
                    onChange={reverb_enabled => updateConfig({ reverb_enabled })}
                    disabled={disabled}
                    tooltip="Simulate different room acoustics"
                  />
                  {config.reverb_enabled && (
                    <div className="mt-3">
                      <SliderWithValue
                        label="Reverb Amount"
                        value={config.reverb_amount}
                        min={0}
                        max={50}
                        step={5}
                        unit="%"
                        onChange={reverb_amount => updateConfig({ reverb_amount })}
                        disabled={disabled}
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Summary */}
          <div className="bg-gray-750 rounded-lg p-3 text-xs text-gray-400">
            <span className="font-medium text-gray-300">Active augmentations: </span>
            {[
              config.pitch_shift_enabled && 'Pitch',
              config.time_stretch_enabled && 'Time',
              config.eq_enabled && 'EQ',
              config.noise_enabled && 'Noise',
              config.reverb_enabled && 'Reverb',
            ].filter(Boolean).join(', ') || 'None'}
          </div>
        </div>
      )}
    </div>
  )
}

export const DEFAULT_AUGMENTATION_CONFIG: AugmentationConfig = {
  enabled: true,
  pitch_shift_enabled: true,
  pitch_shift_min: -2,
  pitch_shift_max: 2,
  time_stretch_enabled: true,
  time_stretch_min: 0.9,
  time_stretch_max: 1.1,
  eq_enabled: true,
  eq_variation: 'subtle',
  noise_enabled: false,
  noise_level: 5,
  reverb_enabled: false,
  reverb_amount: 10,
}
