import { Cpu, Activity, Thermometer, HardDrive, AlertCircle } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { apiService } from '../services/api'

interface GPUMonitorProps {
  refreshInterval?: number // milliseconds
  className?: string
}

export function GPUMonitor({ refreshInterval = 2000, className = '' }: GPUMonitorProps) {
  const { data: status, isLoading, error } = useQuery({
    queryKey: ['gpuStatus'],
    queryFn: () => apiService.getSystemStatus(),
    refetchInterval: refreshInterval,
  })

  if (isLoading) {
    return (
      <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-1/3"></div>
          <div className="h-20 bg-gray-200 rounded"></div>
        </div>
      </div>
    )
  }

  if (error || !status) {
    return (
      <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
        <div className="flex items-start space-x-3 text-red-600">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold">Failed to load GPU status</h3>
            <p className="text-sm text-red-500 mt-1">{(error as Error)?.message}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
        <Cpu className="w-5 h-5 text-purple-600" />
        <span>System Status</span>
      </h3>

      {/* GPU Status */}
      {status.gpu_available ? (
        <div className="space-y-4">
          {/* GPU Name */}
          {status.gpu_name && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">GPU:</span>
              <span className="font-medium text-gray-900">{status.gpu_name}</span>
            </div>
          )}

          {/* GPU Utilization */}
          {status.gpu_utilization !== undefined && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Utilization:</span>
                </div>
                <span className="font-medium text-gray-900">{status.gpu_utilization.toFixed(1)}%</span>
              </div>
              <ProgressBar
                value={status.gpu_utilization}
                max={100}
                color={getUtilizationColor(status.gpu_utilization)}
              />
            </div>
          )}

          {/* GPU Memory */}
          {status.gpu_memory_used !== undefined && status.gpu_memory_total !== undefined && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <HardDrive className="w-4 h-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Memory:</span>
                </div>
                <span className="font-medium text-gray-900">
                  {formatBytes(status.gpu_memory_used)} / {formatBytes(status.gpu_memory_total)}
                </span>
              </div>
              <ProgressBar
                value={status.gpu_memory_used}
                max={status.gpu_memory_total}
                color={getMemoryColor((status.gpu_memory_used / status.gpu_memory_total) * 100)}
              />
            </div>
          )}

          {/* GPU Temperature */}
          {status.gpu_temperature !== undefined && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Thermometer className="w-4 h-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Temperature:</span>
                </div>
                <span className="font-medium text-gray-900">{status.gpu_temperature}°C</span>
              </div>
              <ProgressBar
                value={status.gpu_temperature}
                max={100}
                color={getTemperatureColor(status.gpu_temperature)}
              />
            </div>
          )}

          {/* Model Status */}
          <div className="pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Model Status:</span>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  status.model_loaded
                    ? 'bg-green-100 text-green-800'
                    : 'bg-yellow-100 text-yellow-800'
                }`}
              >
                {status.model_loaded ? 'Loaded' : 'Not Loaded'}
              </span>
            </div>
          </div>

          {/* Individual Models */}
          {status.models && status.models.length > 0 && (
            <div className="pt-4 border-t border-gray-200 space-y-2">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Models:</h4>
              {status.models.map((model: any, index: number) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        model.loaded ? 'bg-green-500' : 'bg-gray-300'
                      }`}
                    />
                    <span className="text-gray-700">{model.name}</span>
                  </div>
                  {model.memory_usage && (
                    <span className="text-gray-500">{formatBytes(model.memory_usage)}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-8">
          <div className="p-4 bg-yellow-100 rounded-full inline-block mb-3">
            <AlertCircle className="w-8 h-8 text-yellow-600" />
          </div>
          <h4 className="font-semibold text-gray-900 mb-1">GPU Not Available</h4>
          <p className="text-sm text-gray-600">
            Running on CPU. Processing will be slower.
          </p>
        </div>
      )}

      {/* Overall Status */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">System Status:</span>
          <span
            className={`px-3 py-1 rounded-full text-xs font-medium ${
              status.status === 'ready'
                ? 'bg-green-100 text-green-800'
                : status.status === 'busy'
                ? 'bg-yellow-100 text-yellow-800'
                : 'bg-red-100 text-red-800'
            }`}
          >
            {status.status.toUpperCase()}
          </span>
        </div>
      </div>
    </div>
  )
}

function ProgressBar({ value, max, color }: { value: number; max: number; color: string }) {
  const percentage = (value / max) * 100

  return (
    <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
      <div
        className={`h-full ${color} transition-all duration-300`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}

function getUtilizationColor(utilization: number): string {
  if (utilization < 50) return 'bg-green-500'
  if (utilization < 80) return 'bg-yellow-500'
  return 'bg-red-500'
}

function getMemoryColor(percentage: number): string {
  if (percentage < 60) return 'bg-blue-500'
  if (percentage < 85) return 'bg-yellow-500'
  return 'bg-red-500'
}

function getTemperatureColor(temp: number): string {
  if (temp < 60) return 'bg-green-500'
  if (temp < 75) return 'bg-yellow-500'
  return 'bg-red-500'
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

