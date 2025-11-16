import { useState } from 'react'
import { Download, Settings } from 'lucide-react'
import { ExportOptions as ExportOptionsType } from '../services/api'

interface ExportOptionsProps {
  audioUrl: string
  fileName: string
  onExport?: (options: ExportOptionsType) => void
}

export function ExportOptions({ audioUrl, fileName, onExport }: ExportOptionsProps) {
  const [showOptions, setShowOptions] = useState(false)
  const [options, setOptions] = useState<ExportOptionsType>({
    format: 'wav',
    bitrate: 320,
    sampleRate: 48000,
    channels: 2,
  })

  const formatDescriptions: Record<string, string> = {
    mp3: 'MP3 - Compressed, widely compatible',
    wav: 'WAV - Lossless, high quality',
    flac: 'FLAC - Lossless, smaller than WAV',
    ogg: 'OGG - Compressed, open format',
  }

  const sampleRateOptions = [44100, 48000, 96000]

  const handleExport = async () => {
    if (onExport) {
      onExport(options)
    } else {
      // Default export behavior
      const link = document.createElement('a')
      link.href = audioUrl
      link.download = `${fileName.replace(/\.[^/.]+$/, '')}.${options.format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
    setShowOptions(false)
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowOptions(!showOptions)}
        className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
      >
        <Settings className="w-5 h-5" />
        <span>Export Options</span>
      </button>

      {showOptions && (
        <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-xl border border-gray-200 p-4 z-50">
          <h3 className="font-semibold text-gray-900 mb-4">Export Settings</h3>

          {/* Format Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Format</label>
            <div className="space-y-2">
              {(['mp3', 'wav', 'flac', 'ogg'] as const).map((fmt) => (
                <label key={fmt} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="format"
                    value={fmt}
                    checked={options.format === fmt}
                    onChange={(e) => setOptions({ ...options, format: e.target.value as any })}
                    className="w-4 h-4"
                  />
                  <span className="text-sm text-gray-700">{formatDescriptions[fmt]}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Bitrate Selection */}
          {options.format !== 'wav' && options.format !== 'flac' && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Bitrate: {options.bitrate} kbps
              </label>
              <input
                type="range"
                min="128"
                max="320"
                step="64"
                value={options.bitrate}
                onChange={(e) => setOptions({ ...options, bitrate: parseInt(e.target.value) })}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>128 kbps</span>
                <span>320 kbps</span>
              </div>
            </div>
          )}

          {/* Sample Rate Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Sample Rate</label>
            <select
              value={options.sampleRate}
              onChange={(e) => setOptions({ ...options, sampleRate: parseInt(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            >
              {sampleRateOptions.map((rate) => (
                <option key={rate} value={rate}>
                  {rate / 1000}kHz
                </option>
              ))}
            </select>
          </div>

          {/* Channels Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Channels</label>
            <div className="flex space-x-2">
              {[1, 2].map((ch) => (
                <button
                  key={ch}
                  onClick={() => setOptions({ ...options, channels: ch as 1 | 2 })}
                  className={`flex-1 py-2 rounded-lg transition-colors ${
                    options.channels === ch
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {ch === 1 ? 'Mono' : 'Stereo'}
                </button>
              ))}
            </div>
          </div>

          {/* Export Button */}
          <div className="flex space-x-2">
            <button
              onClick={() => setShowOptions(false)}
              className="flex-1 px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleExport}
              className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

