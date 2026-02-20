import { useState, useEffect } from 'react'
import { Mic, Volume2, RefreshCw, CheckCircle, AlertCircle, Loader2, Settings } from 'lucide-react'
import { apiService, AudioDevice, DeviceConfig } from '../services/api'
import clsx from 'clsx'

interface AudioDeviceSelectorProps {
  onConfigChange?: (config: DeviceConfig) => void
  compact?: boolean
  showSampleRate?: boolean
}

const SAMPLE_RATES = [
  { value: 16000, label: '16 kHz', description: 'Phone quality' },
  { value: 22050, label: '22.05 kHz', description: 'AM radio quality' },
  { value: 44100, label: '44.1 kHz', description: 'CD quality' },
  { value: 48000, label: '48 kHz', description: 'Professional audio' },
  { value: 96000, label: '96 kHz', description: 'High-resolution' },
]

export function AudioDeviceSelector({ onConfigChange, compact = false, showSampleRate = true }: AudioDeviceSelectorProps) {
  const [devices, setDevices] = useState<AudioDevice[]>([])
  const [config, setConfig] = useState<DeviceConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const fetchDevices = async () => {
    setLoading(true)
    setError(null)
    try {
      const [deviceList, currentConfig] = await Promise.all([
        apiService.listAudioDevices(),
        apiService.getDeviceConfig(),
      ])
      setDevices(deviceList)
      setConfig(currentConfig)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load devices')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDevices()
  }, [])

  const handleDeviceChange = async (type: 'input' | 'output', deviceId: string) => {
    if (!config) return

    setSaving(true)
    setError(null)
    setSuccess(false)

    try {
      const newConfig = await apiService.setDeviceConfig({
        ...config,
        [type === 'input' ? 'input_device_id' : 'output_device_id']: deviceId || undefined,
      })
      setConfig(newConfig)
      setSuccess(true)
      onConfigChange?.(newConfig)
      setTimeout(() => setSuccess(false), 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update device')
    } finally {
      setSaving(false)
    }
  }

  const handleSampleRateChange = async (sampleRate: number) => {
    if (!config) return

    setSaving(true)
    setError(null)
    setSuccess(false)

    try {
      const newConfig = await apiService.setDeviceConfig({
        ...config,
        sample_rate: sampleRate,
      })
      setConfig(newConfig)
      setSuccess(true)
      onConfigChange?.(newConfig)
      setTimeout(() => setSuccess(false), 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update sample rate')
    } finally {
      setSaving(false)
    }
  }

  const inputDevices = devices.filter(d => d.type === 'input')
  const outputDevices = devices.filter(d => d.type === 'output')

  if (loading) {
    return (
      <div className={clsx('flex items-center gap-2 text-gray-400', compact ? 'text-sm' : '')}>
        <Loader2 className="animate-spin" size={compact ? 14 : 16} />
        Loading devices...
      </div>
    )
  }

  if (compact) {
    return (
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <Mic size={14} className="text-gray-400" />
          <select
            value={config?.input_device_id || ''}
            onChange={e => handleDeviceChange('input', e.target.value)}
            disabled={saving}
            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500"
          >
            <option value="">Default Input</option>
            {inputDevices.map(d => (
              <option key={d.device_id} value={d.device_id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <Volume2 size={14} className="text-gray-400" />
          <select
            value={config?.output_device_id || ''}
            onChange={e => handleDeviceChange('output', e.target.value)}
            disabled={saving}
            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500"
          >
            <option value="">Default Output</option>
            {outputDevices.map(d => (
              <option key={d.device_id} value={d.device_id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        {saving && <Loader2 className="animate-spin" size={14} />}
        {success && <CheckCircle size={14} className="text-green-400" />}
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Audio Devices</h3>
        <button
          onClick={fetchDevices}
          disabled={loading}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
          title="Refresh devices"
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-red-400 text-sm">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {success && (
        <div className="flex items-center gap-2 text-green-400 text-sm">
          <CheckCircle size={16} />
          Device configuration saved
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
            <Mic size={16} />
            Input Device (Microphone)
          </label>
          <select
            value={config?.input_device_id || ''}
            onChange={e => handleDeviceChange('input', e.target.value)}
            disabled={saving}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
          >
            <option value="">System Default</option>
            {inputDevices.map(device => (
              <option key={device.device_id} value={device.device_id}>
                {device.name}
                {device.is_default && ' (Default)'}
                {device.sample_rate && ` - ${device.sample_rate}Hz`}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
            <Volume2 size={16} />
            Output Device (Speakers)
          </label>
          <select
            value={config?.output_device_id || ''}
            onChange={e => handleDeviceChange('output', e.target.value)}
            disabled={saving}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
          >
            <option value="">System Default</option>
            {outputDevices.map(device => (
              <option key={device.device_id} value={device.device_id}>
                {device.name}
                {device.is_default && ' (Default)'}
                {device.sample_rate && ` - ${device.sample_rate}Hz`}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Sample Rate Selector */}
      {showSampleRate && (
        <div className="pt-4 border-t border-gray-700">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors mb-3"
          >
            <Settings size={14} />
            Advanced Settings
            <span className={clsx('transition-transform', showAdvanced && 'rotate-180')}>▼</span>
          </button>

          {showAdvanced && (
            <div className="space-y-3 bg-gray-750 rounded-lg p-3">
              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Sample Rate
                </label>
                <div className="grid grid-cols-5 gap-2">
                  {SAMPLE_RATES.map(rate => (
                    <button
                      key={rate.value}
                      onClick={() => handleSampleRateChange(rate.value)}
                      disabled={saving}
                      className={clsx(
                        'px-2 py-2 rounded text-xs transition-all',
                        config?.sample_rate === rate.value
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
                        saving && 'opacity-50 cursor-not-allowed'
                      )}
                      title={rate.description}
                    >
                      {rate.label}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Current: {config?.sample_rate ? `${(config.sample_rate / 1000).toFixed(1)} kHz` : 'Default'}
                  {config?.sample_rate && SAMPLE_RATES.find(r => r.value === config.sample_rate)?.description &&
                    ` (${SAMPLE_RATES.find(r => r.value === config.sample_rate)?.description})`}
                </p>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Buffer Size</span>
                <select
                  value={config?.buffer_size || 512}
                  onChange={e => {
                    if (config) {
                      apiService.setDeviceConfig({
                        ...config,
                        buffer_size: parseInt(e.target.value),
                      }).then(setConfig)
                    }
                  }}
                  disabled={saving}
                  className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                >
                  <option value={128}>128 samples (low latency)</option>
                  <option value={256}>256 samples</option>
                  <option value={512}>512 samples (balanced)</option>
                  <option value={1024}>1024 samples</option>
                  <option value={2048}>2048 samples (stable)</option>
                </select>
              </div>
            </div>
          )}
        </div>
      )}

      {devices.length === 0 && !error && (
        <p className="text-gray-500 text-sm">
          No audio devices found. Make sure your microphone is connected.
        </p>
      )}
    </div>
  )
}
