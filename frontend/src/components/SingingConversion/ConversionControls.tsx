import { useState } from 'react'
import { Settings, Sliders, Music2, Sparkles } from 'lucide-react'
import clsx from 'clsx'

export interface ConversionSettings {
  pitchShift: number
  preserveOriginalPitch: boolean
  preserveVibrato: boolean
  preserveExpression: boolean
  outputQuality: 'fast' | 'balanced' | 'high' | 'studio'
  denoiseInput: boolean
  enhanceOutput: boolean
}

interface ConversionControlsProps {
  settings: ConversionSettings
  onChange: (settings: ConversionSettings) => void
  disabled?: boolean
}

export function ConversionControls({ settings, onChange, disabled = false }: ConversionControlsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const updateSetting = <K extends keyof ConversionSettings>(
    key: K,
    value: ConversionSettings[K]
  ) => {
    onChange({ ...settings, [key]: value })
  }

  const qualityOptions = [
    { value: 'fast', label: 'Fast', description: '~10s per song' },
    { value: 'balanced', label: 'Balanced', description: '~20s per song' },
    { value: 'high', label: 'High', description: '~30s per song' },
    { value: 'studio', label: 'Studio', description: '~60s per song' },
  ] as const

  return (
    <div className="space-y-6">
      {/* Pitch Shift Control */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="flex items-center space-x-2 text-sm font-medium text-gray-700">
            <Sliders className="w-4 h-4" />
            <span>Pitch Shift</span>
          </label>
          <span className="text-sm font-semibold text-primary-600">
            {settings.pitchShift > 0 ? '+' : ''}{settings.pitchShift} semitones
          </span>
        </div>
        <input
          type="range"
          min="-12"
          max="12"
          step="0.5"
          value={settings.pitchShift}
          onChange={(e) => updateSetting('pitchShift', parseFloat(e.target.value))}
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>-12 (1 octave down)</span>
          <span>0 (original)</span>
          <span>+12 (1 octave up)</span>
        </div>
      </div>

      {/* Preservation Options */}
      <div className="space-y-3">
        <label className="flex items-center space-x-2 text-sm font-medium text-gray-700">
          <Music2 className="w-4 h-4" />
          <span>Preservation Settings</span>
        </label>
        
        <div className="space-y-2">
          <label className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer">
            <input
              type="checkbox"
              checked={settings.preserveOriginalPitch}
              onChange={(e) => updateSetting('preserveOriginalPitch', e.target.checked)}
              disabled={disabled}
              className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
            />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Preserve Original Pitch</p>
              <p className="text-xs text-gray-500">Keep the exact pitch contour from the original</p>
            </div>
          </label>

          <label className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer">
            <input
              type="checkbox"
              checked={settings.preserveVibrato}
              onChange={(e) => updateSetting('preserveVibrato', e.target.checked)}
              disabled={disabled}
              className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
            />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Preserve Vibrato</p>
              <p className="text-xs text-gray-500">Maintain vibrato rate and depth</p>
            </div>
          </label>

          <label className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer">
            <input
              type="checkbox"
              checked={settings.preserveExpression}
              onChange={(e) => updateSetting('preserveExpression', e.target.checked)}
              disabled={disabled}
              className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
            />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Preserve Expression</p>
              <p className="text-xs text-gray-500">Keep dynamics and emotional nuances</p>
            </div>
          </label>
        </div>
      </div>

      {/* Output Quality */}
      <div className="space-y-3">
        <label className="flex items-center space-x-2 text-sm font-medium text-gray-700">
          <Sparkles className="w-4 h-4" />
          <span>Output Quality</span>
        </label>
        
        <div className="grid grid-cols-2 gap-2">
          {qualityOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => updateSetting('outputQuality', option.value)}
              disabled={disabled}
              className={clsx(
                'p-3 rounded-lg border-2 transition-all text-left',
                settings.outputQuality === option.value
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:border-gray-300',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <p className="text-sm font-medium text-gray-900">{option.label}</p>
              <p className="text-xs text-gray-500">{option.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings Toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center space-x-2 text-sm text-primary-600 hover:text-primary-700 font-medium"
      >
        <Settings className="w-4 h-4" />
        <span>{showAdvanced ? 'Hide' : 'Show'} Advanced Settings</span>
      </button>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div className="space-y-2 pt-2 border-t border-gray-200">
          <label className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer">
            <input
              type="checkbox"
              checked={settings.denoiseInput}
              onChange={(e) => updateSetting('denoiseInput', e.target.checked)}
              disabled={disabled}
              className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
            />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Denoise Input</p>
              <p className="text-xs text-gray-500">Remove background noise before conversion</p>
            </div>
          </label>

          <label className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer">
            <input
              type="checkbox"
              checked={settings.enhanceOutput}
              onChange={(e) => updateSetting('enhanceOutput', e.target.checked)}
              disabled={disabled}
              className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
            />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Enhance Output</p>
              <p className="text-xs text-gray-500">Apply post-processing for better quality</p>
            </div>
          </label>
        </div>
      )}
    </div>
  )
}

