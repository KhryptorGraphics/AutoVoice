import { useState } from 'react'
import { Settings, ChevronDown, ChevronUp, Info } from 'lucide-react'
import { AdvancedConversionSettings as SettingsType } from '../services/api'

interface AdvancedConversionSettingsProps {
  settings: SettingsType
  onChange: (settings: SettingsType) => void
  className?: string
}

export function AdvancedConversionSettings({ settings, onChange, className = '' }: AdvancedConversionSettingsProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const updateSetting = <K extends keyof SettingsType>(key: K, value: SettingsType[K]) => {
    onChange({ ...settings, [key]: value })
  }

  const qualityPresets = [
    { value: 'draft', label: 'Draft', description: '2 steps, 4.0x speed, ~7.5s per 30s' },
    { value: 'fast', label: 'Fast', description: '4 steps, 2.0x speed, ~15s per 30s' },
    { value: 'balanced', label: 'Balanced', description: '4 steps, 1.0x speed, ~30s per 30s' },
    { value: 'high', label: 'High', description: '8 steps, 0.5x speed, ~60s per 30s' },
    { value: 'studio', label: 'Studio', description: '16 steps, 0.25x speed, ~120s per 30s' },
  ]

  return (
    <div className={`bg-white rounded-lg shadow-lg ${className}`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-6 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center space-x-3">
          <Settings className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-900">Advanced Settings</h3>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {isExpanded && (
        <div className="px-6 pb-6 space-y-6 border-t border-gray-200 pt-6">
          {/* Quality Preset */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quality Preset
            </label>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
              {qualityPresets.map((preset) => (
                <button
                  key={preset.value}
                  onClick={() => updateSetting('output_quality', preset.value as any)}
                  className={`p-3 border-2 rounded-lg text-left transition-all ${
                    settings.output_quality === preset.value
                      ? 'border-purple-600 bg-purple-50'
                      : 'border-gray-200 hover:border-purple-300'
                  }`}
                  title={preset.description}
                >
                  <div className="font-semibold text-sm">{preset.label}</div>
                  <div className="text-xs text-gray-500 mt-1">{preset.description.split(',')[0]}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Pitch Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center space-x-2">
                <span>Pitch Shift (semitones)</span>
                <Tooltip text="Shift the pitch up or down by semitones (-12 to +12)" />
              </label>
              <input
                type="range"
                min="-12"
                max="12"
                step="1"
                value={settings.pitch_shift || 0}
                onChange={(e) => updateSetting('pitch_shift', Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-600 mt-1">
                <span>-12</span>
                <span className="font-medium text-gray-900">{settings.pitch_shift || 0}</span>
                <span>+12</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center space-x-2">
                <span>Formant Shift</span>
                <Tooltip text="Adjust vocal formants (0.8 to 1.2)" />
              </label>
              <input
                type="range"
                min="0.8"
                max="1.2"
                step="0.05"
                value={settings.formant_shift || 1.0}
                onChange={(e) => updateSetting('formant_shift', Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-600 mt-1">
                <span>0.8</span>
                <span className="font-medium text-gray-900">{(settings.formant_shift || 1.0).toFixed(2)}</span>
                <span>1.2</span>
              </div>
            </div>
          </div>

          {/* Volume Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Vocal Volume
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={settings.vocal_volume || 1.0}
                onChange={(e) => updateSetting('vocal_volume', Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-600 mt-1">
                <span>0%</span>
                <span className="font-medium text-gray-900">{((settings.vocal_volume || 1.0) * 100).toFixed(0)}%</span>
                <span>200%</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Instrumental Volume
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={settings.instrumental_volume || 0.9}
                onChange={(e) => updateSetting('instrumental_volume', Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-600 mt-1">
                <span>0%</span>
                <span className="font-medium text-gray-900">{((settings.instrumental_volume || 0.9) * 100).toFixed(0)}%</span>
                <span>200%</span>
              </div>
            </div>
          </div>

          {/* Expression Controls */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center space-x-2">
              <span>Temperature (Expressiveness)</span>
              <Tooltip text="Control expressiveness and variation (0.5 to 1.5)" />
            </label>
            <input
              type="range"
              min="0.5"
              max="1.5"
              step="0.1"
              value={settings.temperature || 1.0}
              onChange={(e) => updateSetting('temperature', Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-sm text-gray-600 mt-1">
              <span>Conservative</span>
              <span className="font-medium text-gray-900">{(settings.temperature || 1.0).toFixed(1)}</span>
              <span>Expressive</span>
            </div>
          </div>

          {/* Toggle Options */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ToggleOption
              label="Preserve Original Pitch"
              description="Maintain the original pitch contour"
              checked={settings.preserve_original_pitch ?? true}
              onChange={(checked) => updateSetting('preserve_original_pitch', checked)}
            />
            <ToggleOption
              label="Preserve Vibrato"
              description="Keep vibrato patterns intact"
              checked={settings.preserve_vibrato ?? true}
              onChange={(checked) => updateSetting('preserve_vibrato', checked)}
            />
            <ToggleOption
              label="Preserve Expression"
              description="Maintain dynamics and expression"
              checked={settings.preserve_expression ?? true}
              onChange={(checked) => updateSetting('preserve_expression', checked)}
            />
            <ToggleOption
              label="Denoise Input"
              description="Remove background noise"
              checked={settings.denoise_input ?? false}
              onChange={(checked) => updateSetting('denoise_input', checked)}
            />
            <ToggleOption
              label="Enhance Output"
              description="Apply audio enhancement"
              checked={settings.enhance_output ?? false}
              onChange={(checked) => updateSetting('enhance_output', checked)}
            />
            <ToggleOption
              label="Return Stems"
              description="Get separated vocal and instrumental tracks"
              checked={settings.return_stems ?? false}
              onChange={(checked) => updateSetting('return_stems', checked)}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function ToggleOption({ label, description, checked, onChange }: {
  label: string
  description: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="flex items-start space-x-3 cursor-pointer p-3 rounded-lg hover:bg-gray-50 transition-colors">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
      />
      <div className="flex-1">
        <div className="font-medium text-gray-900 text-sm">{label}</div>
        <div className="text-xs text-gray-500 mt-0.5">{description}</div>
      </div>
    </label>
  )
}

function Tooltip({ text }: { text: string }) {
  return (
    <div className="group relative inline-block">
      <Info className="w-4 h-4 text-gray-400 cursor-help" />
      <div className="invisible group-hover:visible absolute z-10 w-64 p-2 bg-gray-900 text-white text-xs rounded-lg -top-2 left-6">
        {text}
      </div>
    </div>
  )
}

