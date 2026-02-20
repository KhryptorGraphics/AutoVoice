import { FileAudio, Settings } from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

export type OutputFormat = 'wav' | 'mp3' | 'flac'

export interface OutputFormatConfig {
  format: OutputFormat
  sampleRate: number
  bitDepth?: number // WAV/FLAC only
  bitrate?: number  // MP3 only
}

interface OutputFormatSelectorProps {
  config: OutputFormatConfig
  onChange: (config: OutputFormatConfig) => void
  disabled?: boolean
  compact?: boolean
}

const FORMAT_OPTIONS: {
  value: OutputFormat
  label: string
  ext: string
  description: string
  icon: string
}[] = [
  {
    value: 'wav',
    label: 'WAV',
    ext: '.wav',
    description: 'Lossless, largest file size',
    icon: '🎵',
  },
  {
    value: 'mp3',
    label: 'MP3',
    ext: '.mp3',
    description: 'Compressed, universal compatibility',
    icon: '🎧',
  },
  {
    value: 'flac',
    label: 'FLAC',
    ext: '.flac',
    description: 'Lossless compression, smaller than WAV',
    icon: '💿',
  },
]

const SAMPLE_RATES = [
  { value: 22050, label: '22.05 kHz' },
  { value: 44100, label: '44.1 kHz' },
  { value: 48000, label: '48 kHz' },
  { value: 96000, label: '96 kHz' },
]

const BIT_DEPTHS = [
  { value: 16, label: '16-bit' },
  { value: 24, label: '24-bit' },
  { value: 32, label: '32-bit float' },
]

const MP3_BITRATES = [
  { value: 128, label: '128 kbps', description: 'Good' },
  { value: 192, label: '192 kbps', description: 'Better' },
  { value: 256, label: '256 kbps', description: 'High' },
  { value: 320, label: '320 kbps', description: 'Best' },
]

export function OutputFormatSelector({
  config,
  onChange,
  disabled = false,
  compact = false,
}: OutputFormatSelectorProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleFormatChange = (format: OutputFormat) => {
    const newConfig: OutputFormatConfig = {
      ...config,
      format,
    }
    // Set defaults based on format
    if (format === 'mp3') {
      newConfig.bitrate = config.bitrate || 320
      delete newConfig.bitDepth
    } else {
      newConfig.bitDepth = config.bitDepth || 24
      delete newConfig.bitrate
    }
    onChange(newConfig)
  }

  if (compact) {
    return (
      <div className="flex items-center gap-3">
        <FileAudio size={14} className="text-gray-400" />
        <select
          value={config.format}
          onChange={e => handleFormatChange(e.target.value as OutputFormat)}
          disabled={disabled}
          className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
        >
          {FORMAT_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        {config.format === 'mp3' && (
          <select
            value={config.bitrate || 320}
            onChange={e => onChange({ ...config, bitrate: parseInt(e.target.value) })}
            disabled={disabled}
            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
          >
            {MP3_BITRATES.map(opt => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Format Selection */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">Output Format</label>
        <div className="grid grid-cols-3 gap-2">
          {FORMAT_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => handleFormatChange(opt.value)}
              disabled={disabled}
              className={clsx(
                'p-3 rounded-lg border text-center transition-all',
                config.format === opt.value
                  ? 'border-blue-500 bg-blue-500/20'
                  : 'border-gray-700 hover:border-gray-600 bg-gray-750',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <span className="text-xl">{opt.icon}</span>
              <div className="font-medium text-sm mt-1">{opt.label}</div>
              <div className="text-xs text-gray-500">{opt.ext}</div>
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500">
          {FORMAT_OPTIONS.find(o => o.value === config.format)?.description}
        </p>
      </div>

      {/* Format-Specific Options */}
      {config.format === 'mp3' && (
        <div className="space-y-2">
          <label className="text-sm text-gray-400">Bitrate</label>
          <div className="grid grid-cols-4 gap-2">
            {MP3_BITRATES.map(opt => (
              <button
                key={opt.value}
                onClick={() => onChange({ ...config, bitrate: opt.value })}
                disabled={disabled}
                className={clsx(
                  'p-2 rounded-lg border text-center transition-all',
                  config.bitrate === opt.value
                    ? 'border-blue-500 bg-blue-500/20'
                    : 'border-gray-700 hover:border-gray-600 bg-gray-750',
                  disabled && 'opacity-50 cursor-not-allowed'
                )}
              >
                <div className="font-mono text-sm">{opt.label}</div>
                <div className="text-xs text-gray-500">{opt.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {(config.format === 'wav' || config.format === 'flac') && (
        <div className="space-y-2">
          <label className="text-sm text-gray-400">Bit Depth</label>
          <div className="grid grid-cols-3 gap-2">
            {BIT_DEPTHS.map(opt => (
              <button
                key={opt.value}
                onClick={() => onChange({ ...config, bitDepth: opt.value })}
                disabled={disabled}
                className={clsx(
                  'p-2 rounded-lg border text-center transition-all',
                  config.bitDepth === opt.value
                    ? 'border-blue-500 bg-blue-500/20'
                    : 'border-gray-700 hover:border-gray-600 bg-gray-750',
                  disabled && 'opacity-50 cursor-not-allowed'
                )}
              >
                <div className="font-medium text-sm">{opt.label}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Advanced Settings */}
      <div className="pt-2 border-t border-gray-700">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <Settings size={14} />
          Advanced Settings
          <span className={clsx('transition-transform', showAdvanced && 'rotate-180')}>▼</span>
        </button>

        {showAdvanced && (
          <div className="mt-3 space-y-3 bg-gray-750 rounded-lg p-3">
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Sample Rate</label>
              <div className="grid grid-cols-4 gap-2">
                {SAMPLE_RATES.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => onChange({ ...config, sampleRate: opt.value })}
                    disabled={disabled}
                    className={clsx(
                      'px-2 py-1.5 rounded border text-xs transition-all',
                      config.sampleRate === opt.value
                        ? 'border-blue-500 bg-blue-500/20'
                        : 'border-gray-700 hover:border-gray-600',
                      disabled && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {config.format === 'flac' && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Compression Level</span>
                <select
                  value={5}
                  className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                  disabled={disabled}
                >
                  <option value={0}>0 (Fastest)</option>
                  <option value={5}>5 (Balanced)</option>
                  <option value={8}>8 (Smallest)</option>
                </select>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Estimated Size */}
      <div className="text-xs text-gray-500 flex items-center justify-between">
        <span>Estimated file size (1 min):</span>
        <span className="font-mono">
          {config.format === 'mp3'
            ? `~${((config.bitrate || 320) * 60 / 8 / 1024).toFixed(1)} MB`
            : config.format === 'wav'
            ? `~${((config.sampleRate * (config.bitDepth || 24) / 8 * 2 * 60) / 1024 / 1024).toFixed(1)} MB`
            : `~${((config.sampleRate * (config.bitDepth || 24) / 8 * 2 * 60 * 0.6) / 1024 / 1024).toFixed(1)} MB`}
        </span>
      </div>
    </div>
  )
}

export const DEFAULT_OUTPUT_FORMAT: OutputFormatConfig = {
  format: 'wav',
  sampleRate: 44100,
  bitDepth: 24,
}
