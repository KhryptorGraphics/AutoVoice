import { useState, useEffect } from 'react'
import { Sparkles, Zap, HelpCircle, AlertCircle, Loader2 } from 'lucide-react'
import clsx from 'clsx'
import { api, AdapterType, AdapterListResponse, AdapterMetricsResponse } from '../services/api'

interface AdapterSelectorProps {
  profileId: string
  value?: AdapterType | null
  onChange?: (value: AdapterType) => void
  disabled?: boolean
  showMetrics?: boolean
  size?: 'sm' | 'md' | 'lg'
  onLoadingChange?: (loading: boolean) => void
}

interface AdapterInfo {
  id: AdapterType
  name: string
  icon: React.ReactNode
  description: string
  quality: string
  speed: string
  memoryMb: number
  colorClass: string
  bgClass: string
  borderClass: string
}

const adapterConfigs: Record<AdapterType, Omit<AdapterInfo, 'id'>> = {
  hq: {
    name: 'High Quality',
    icon: <Sparkles className="w-4 h-4" />,
    description: 'Maximum fidelity with 5M+ parameters',
    quality: 'Excellent',
    speed: 'Moderate',
    memoryMb: 150,
    colorClass: 'text-violet-300',
    bgClass: 'bg-violet-600',
    borderClass: 'border-violet-500',
  },
  nvfp4: {
    name: 'Fast (nvfp4)',
    icon: <Zap className="w-4 h-4" />,
    description: '4-bit quantized for speed',
    quality: 'Good',
    speed: 'Fast',
    memoryMb: 20,
    colorClass: 'text-yellow-300',
    bgClass: 'bg-yellow-600',
    borderClass: 'border-yellow-500',
  },
}

export function AdapterSelector({
  profileId,
  value,
  onChange,
  disabled = false,
  showMetrics = true,
  size = 'md',
  onLoadingChange,
}: AdapterSelectorProps) {
  const [adapters, setAdapters] = useState<AdapterListResponse | null>(null)
  const [metrics, setMetrics] = useState<AdapterMetricsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showTooltip, setShowTooltip] = useState<AdapterType | null>(null)

  useEffect(() => {
    const loadAdapters = async () => {
      try {
        setLoading(true)
        onLoadingChange?.(true)
        const [adapterList, adapterMetrics] = await Promise.all([
          api.getProfileAdapters(profileId),
          showMetrics ? api.getAdapterMetrics(profileId) : Promise.resolve(null),
        ])
        setAdapters(adapterList)
        setMetrics(adapterMetrics)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load adapters')
      } finally {
        setLoading(false)
        onLoadingChange?.(false)
      }
    }

    if (profileId) {
      loadAdapters()
    }
  }, [profileId, showMetrics, onLoadingChange])

  const handleSelect = async (adapterType: AdapterType) => {
    if (disabled || !adapters?.adapters.some(a => a.type === adapterType)) return

    try {
      await api.selectAdapter(profileId, adapterType)
      onChange?.(adapterType)
      // Update local state
      setAdapters(prev => prev ? { ...prev, selected: adapterType } : null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to select adapter')
    }
  }

  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-2 text-sm',
    lg: 'px-4 py-3 text-base',
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-gray-400">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">Loading adapters...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-red-400">
        <AlertCircle className="w-4 h-4" />
        <span className="text-sm">{error}</span>
      </div>
    )
  }

  const selectedAdapter = value ?? adapters?.selected
  const availableTypes = adapters?.adapters.map(a => a.type) ?? []

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-gray-300">Adapter</label>
        <div
          className="relative"
          onMouseEnter={() => setShowTooltip(selectedAdapter ?? 'hq')}
          onMouseLeave={() => setShowTooltip(null)}
        >
          <HelpCircle className="w-4 h-4 text-gray-500 cursor-help" />
          {showTooltip && (
            <div className="absolute z-10 w-72 p-3 text-xs bg-gray-800 border border-gray-700 rounded-lg shadow-xl left-6 top-0">
              <div className="font-medium text-white mb-1">
                {adapterConfigs[showTooltip].name}
              </div>
              <div className="text-gray-400 mb-2">
                {adapterConfigs[showTooltip].description}
              </div>
              {metrics?.adapters[showTooltip] && (
                <div className="grid grid-cols-2 gap-1 text-gray-500 border-t border-gray-700 pt-2">
                  <div>Epochs: <span className="text-gray-300">{metrics.adapters[showTooltip].epochs}</span></div>
                  <div>Loss: <span className="text-gray-300">{metrics.adapters[showTooltip].loss?.toFixed(4) ?? 'N/A'}</span></div>
                  <div>Memory: <span className="text-gray-300">{metrics.adapters[showTooltip].performance.memory_estimate_mb}MB</span></div>
                  <div>Params: <span className="text-gray-300">{metrics.adapters[showTooltip].parameter_count_formatted}</span></div>
                </div>
              )}
            </div>
          )}
        </div>
        {adapters?.count === 0 && (
          <span className="text-xs text-gray-500">(No trained adapters)</span>
        )}
      </div>

      <div className="flex gap-2">
        {(['hq', 'nvfp4'] as AdapterType[]).map((adapterType) => {
          const config = adapterConfigs[adapterType]
          const available = availableTypes.includes(adapterType)
          const isSelected = selectedAdapter === adapterType
          const adapterData = adapters?.adapters.find(a => a.type === adapterType)

          return (
            <button
              key={adapterType}
              onClick={() => handleSelect(adapterType)}
              disabled={disabled || !available}
              className={clsx(
                'flex flex-col items-start gap-1 rounded-lg border transition-all',
                sizeClasses[size],
                isSelected
                  ? `${config.bgClass} ${config.borderClass} text-white`
                  : available
                    ? 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-600'
                    : 'bg-gray-900 border-gray-800 text-gray-600',
                (disabled || !available) && 'opacity-50 cursor-not-allowed'
              )}
            >
              <div className="flex items-center gap-2">
                {config.icon}
                <span className="font-medium">{config.name}</span>
              </div>
              {showMetrics && adapterData && (
                <div className="text-xs opacity-80">
                  Loss: {adapterData.loss?.toFixed(4) ?? 'N/A'} · {adapterData.epochs} epochs
                </div>
              )}
              {!available && (
                <div className="text-xs text-gray-500">Not trained</div>
              )}
            </button>
          )
        })}
      </div>

      {selectedAdapter && (
        <p className="text-xs text-gray-500">
          {adapterConfigs[selectedAdapter].quality} quality · {adapterConfigs[selectedAdapter].speed} inference
        </p>
      )}
    </div>
  )
}

// Compact dropdown version for toolbar use
interface AdapterDropdownProps {
  profileId: string
  value?: AdapterType | null
  onChange?: (value: AdapterType) => void
  disabled?: boolean
}

export function AdapterDropdown({ profileId, value, onChange, disabled }: AdapterDropdownProps) {
  const [adapters, setAdapters] = useState<AdapterListResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getProfileAdapters(profileId)
      .then(setAdapters)
      .catch(() => setAdapters(null))
      .finally(() => setLoading(false))
  }, [profileId])

  const handleChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newValue = e.target.value as AdapterType
    try {
      await api.selectAdapter(profileId, newValue)
      onChange?.(newValue)
    } catch (err) {
      console.error('Failed to select adapter:', err)
    }
  }

  const selectedAdapter = value ?? adapters?.selected
  const availableTypes = adapters?.adapters.map(a => a.type) ?? []

  return (
    <select
      value={selectedAdapter ?? ''}
      onChange={handleChange}
      disabled={disabled || loading || availableTypes.length === 0}
      className={clsx(
        'px-3 py-1.5 rounded-lg border text-sm',
        'bg-gray-800 border-gray-700 text-gray-200',
        'focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent',
        (disabled || loading) && 'opacity-50 cursor-not-allowed'
      )}
    >
      {loading && <option value="">Loading...</option>}
      {!loading && availableTypes.length === 0 && <option value="">No adapters</option>}
      {availableTypes.map((adapterType) => (
        <option key={adapterType} value={adapterType}>
          {adapterConfigs[adapterType].name}
        </option>
      ))}
    </select>
  )
}

// Display-only badge showing current adapter
interface AdapterBadgeProps {
  adapterType: AdapterType | null | undefined
}

export function AdapterBadge({ adapterType }: AdapterBadgeProps) {
  if (!adapterType) return null
  const config = adapterConfigs[adapterType]

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
        adapterType === 'hq'
          ? 'bg-violet-900/50 text-violet-300'
          : 'bg-yellow-900/50 text-yellow-300'
      )}
    >
      {config.icon}
      {config.name}
    </span>
  )
}
