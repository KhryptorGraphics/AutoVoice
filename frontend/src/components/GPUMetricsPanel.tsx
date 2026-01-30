import { useState, useEffect, useMemo } from 'react'
import { Cpu, Activity, Thermometer, HardDrive, AlertTriangle, RefreshCw } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { apiService, wsManager } from '../services/api'
import clsx from 'clsx'

interface GPUMetricsPanelProps {
  refreshInterval?: number
  showChart?: boolean
  compact?: boolean
}

interface HistoryPoint {
  timestamp: number
  utilization: number
  memoryUsed: number
  temperature: number
}

const MAX_HISTORY = 60 // Keep 60 data points (2 minutes at 2s refresh)

function MiniChart({ data, color, max }: { data: number[]; color: string; max: number }) {
  if (data.length < 2) return null

  const width = 100
  const height = 30
  const points = data.map((value, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height - (value / max) * height
    return `${x},${y}`
  }).join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-8" preserveAspectRatio="none">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}

function MemoryBreakdown({
  used,
  total,
  reserved,
}: {
  used: number
  total: number
  reserved?: number
}) {
  const usedPct = (used / total) * 100
  const reservedPct = reserved ? ((reserved - used) / total) * 100 : 0
  const freePct = 100 - usedPct - reservedPct

  return (
    <div className="space-y-2">
      <div className="h-3 bg-gray-700 rounded-full overflow-hidden flex">
        <div
          className="bg-blue-500 transition-all"
          style={{ width: `${usedPct}%` }}
          title={`Used: ${formatGB(used)}`}
        />
        {reserved && (
          <div
            className="bg-yellow-500 transition-all"
            style={{ width: `${reservedPct}%` }}
            title={`Reserved: ${formatGB(reserved - used)}`}
          />
        )}
        <div
          className="bg-gray-600"
          style={{ width: `${freePct}%` }}
          title={`Free: ${formatGB(total - (reserved || used))}`}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 bg-blue-500 rounded" /> Used: {formatGB(used)}
        </span>
        {reserved && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-yellow-500 rounded" /> Cache: {formatGB(reserved - used)}
          </span>
        )}
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 bg-gray-600 rounded" /> Free: {formatGB(total - (reserved || used))}
        </span>
      </div>
    </div>
  )
}

function formatGB(bytes: number): string {
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function TemperatureAlert({ temp }: { temp: number }) {
  if (temp < 75) return null

  return (
    <div className={clsx(
      'flex items-center gap-2 px-3 py-2 rounded text-sm',
      temp >= 85 ? 'bg-red-900/50 text-red-300' : 'bg-yellow-900/50 text-yellow-300'
    )}>
      <AlertTriangle size={16} />
      <span>
        {temp >= 85 ? 'Critical temperature! Consider reducing load.' : 'High temperature warning.'}
      </span>
    </div>
  )
}

export function GPUMetricsPanel({ refreshInterval = 2000, showChart = true, compact: _compact = false }: GPUMetricsPanelProps) {
  void _compact // Reserved for future compact mode
  const [history, setHistory] = useState<HistoryPoint[]>([])

  const { data: metrics, isLoading, error, refetch } = useQuery({
    queryKey: ['gpuMetrics'],
    queryFn: () => apiService.getGPUMetrics(),
    refetchInterval: refreshInterval,
  })

  // Update history when metrics change
  useEffect(() => {
    if (metrics?.devices?.[0]) {
      const device = metrics.devices[0]
      setHistory(prev => {
        const newPoint: HistoryPoint = {
          timestamp: Date.now(),
          utilization: device.utilization_percent ?? 0,
          memoryUsed: device.memory_used_gb ?? device.memory_used / 1024 / 1024 / 1024,
          temperature: device.temperature_c ?? device.temperature ?? 0,
        }
        const updated = [...prev, newPoint]
        return updated.slice(-MAX_HISTORY)
      })
    }
  }, [metrics])

  // WebSocket real-time updates
  useEffect(() => {
    wsManager.connect()
    const unsub = wsManager.onGPUMetrics(() => {
      // Real-time updates handled by react-query refetch
    })
    return unsub
  }, [])

  const utilizationHistory = useMemo(() => history.map(h => h.utilization), [history])
  const temperatureHistory = useMemo(() => history.map(h => h.temperature), [history])

  if (isLoading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4" />
        <div className="h-24 bg-gray-700 rounded" />
      </div>
    )
  }

  if (error || !metrics) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 text-red-400">
          <AlertTriangle size={20} />
          <div>
            <div className="font-medium">Failed to load GPU metrics</div>
            <div className="text-sm text-gray-500">{(error as Error)?.message}</div>
          </div>
        </div>
      </div>
    )
  }

  if (!metrics.available || metrics.devices.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <Cpu size={40} className="mx-auto text-gray-500 mb-3" />
        <div className="font-medium mb-1">No GPU Available</div>
        <div className="text-sm text-gray-500">Running on CPU - processing will be slower</div>
        {metrics.note && <div className="text-xs text-gray-600 mt-2">{metrics.note}</div>}
      </div>
    )
  }

  const device = metrics.devices[0]
  const utilization = device.utilization_percent ?? 0
  const memoryUsedGB = device.memory_used_gb ?? device.memory_used / 1024 / 1024 / 1024
  const memoryTotalGB = device.memory_total_gb ?? device.memory_total / 1024 / 1024 / 1024
  const temperature = device.temperature_c ?? device.temperature ?? 0

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu size={18} className="text-purple-400" />
          <h3 className="font-semibold">GPU Metrics</h3>
        </div>
        <button
          onClick={() => refetch()}
          className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
          title="Refresh"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      {/* GPU Name */}
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-400">Device</span>
        <span className="font-medium">{device.name}</span>
      </div>

      {/* Temperature Alert */}
      {temperature > 0 && <TemperatureAlert temp={temperature} />}

      {/* Utilization */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Activity size={14} />
            Utilization
          </div>
          <span className={clsx(
            'font-mono font-medium',
            utilization > 80 ? 'text-red-400' : utilization > 50 ? 'text-yellow-400' : 'text-green-400'
          )}>
            {utilization.toFixed(0)}%
          </span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={clsx(
              'h-full transition-all',
              utilization > 80 ? 'bg-red-500' : utilization > 50 ? 'bg-yellow-500' : 'bg-green-500'
            )}
            style={{ width: `${utilization}%` }}
          />
        </div>
        {showChart && utilizationHistory.length > 1 && (
          <div className="bg-gray-750 rounded p-2">
            <MiniChart
              data={utilizationHistory}
              color={utilization > 80 ? '#ef4444' : utilization > 50 ? '#eab308' : '#22c55e'}
              max={100}
            />
          </div>
        )}
      </div>

      {/* Memory */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <HardDrive size={14} />
            Memory
          </div>
          <span className="font-mono text-sm">
            {memoryUsedGB.toFixed(1)} / {memoryTotalGB.toFixed(1)} GB
          </span>
        </div>
        <MemoryBreakdown
          used={device.memory_used}
          total={device.memory_total}
          reserved={undefined} // Would need torch.cuda.memory_reserved() from backend
        />
      </div>

      {/* Temperature */}
      {temperature > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Thermometer size={14} />
              Temperature
            </div>
            <span className={clsx(
              'font-mono font-medium',
              temperature > 80 ? 'text-red-400' : temperature > 70 ? 'text-yellow-400' : 'text-green-400'
            )}>
              {temperature}°C
            </span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={clsx(
                'h-full transition-all',
                temperature > 80 ? 'bg-red-500' : temperature > 70 ? 'bg-yellow-500' : 'bg-green-500'
              )}
              style={{ width: `${Math.min(temperature, 100)}%` }}
            />
          </div>
          {showChart && temperatureHistory.length > 1 && (
            <div className="bg-gray-750 rounded p-2">
              <MiniChart
                data={temperatureHistory}
                color={temperature > 80 ? '#ef4444' : temperature > 70 ? '#eab308' : '#22c55e'}
                max={100}
              />
            </div>
          )}
        </div>
      )}

      {/* Multi-GPU support */}
      {metrics.devices.length > 1 && (
        <div className="pt-2 border-t border-gray-700">
          <div className="text-xs text-gray-500 mb-2">{metrics.device_count} GPUs detected</div>
          <div className="grid grid-cols-2 gap-2">
            {metrics.devices.slice(1).map((gpu) => (
              <div key={gpu.index} className="bg-gray-750 rounded p-2 text-xs">
                <div className="font-medium truncate">{gpu.name}</div>
                <div className="text-gray-500">
                  {(gpu.utilization_percent ?? 0).toFixed(0)}% · {((gpu.memory_used_gb ?? gpu.memory_used / 1024 / 1024 / 1024)).toFixed(1)}GB
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
