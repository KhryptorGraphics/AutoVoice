import { useQuery } from '@tanstack/react-query'
import { Activity, Package, Zap, CheckCircle, XCircle, Clock } from 'lucide-react'
import { apiService } from '../services/api'
import { GPUMonitor } from '../components/GPUMonitor'
import clsx from 'clsx'

export function SystemStatusPage() {
  const { data: _status, isLoading, error } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => apiService.getSystemStatus(),
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  const { data: modelsInfo } = useQuery({
    queryKey: ['modelsInfo'],
    queryFn: () => apiService.getModelsInfo(),
    refetchInterval: 10000,
  })

  const { data: healthStatus } = useQuery({
    queryKey: ['healthCheck'],
    queryFn: () => apiService.healthCheck(),
    refetchInterval: 5000,
  })

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Activity className="w-8 h-8 text-purple-600" />
          <span>System Status</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Monitor GPU utilization, models, and system performance in real-time
        </p>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-2 border-red-500 rounded-lg p-6">
          <p className="text-red-800 font-semibold">Failed to load system status</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* GPU Monitor - Full width on mobile, 2 cols on desktop */}
        <div className="lg:col-span-2">
          <GPUMonitor refreshInterval={2000} />
        </div>

        {/* Health Status */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
            <Zap className="w-5 h-5 text-purple-600" />
            <span>Health Status</span>
          </h3>

          {healthStatus && (
            <div className="space-y-3">
              <StatusItem
                label="API Server"
                status={healthStatus.status === 'healthy' ? 'online' : 'offline'}
              />
              <StatusItem
                label="GPU"
                status={healthStatus.gpu_available ? 'available' : 'unavailable'}
              />
              <StatusItem
                label="Models"
                status={healthStatus.models_loaded ? 'loaded' : 'not loaded'}
              />
              <StatusItem
                label="WebSocket"
                status="connected"
              />

              {healthStatus.uptime && (
                <div className="pt-3 border-t border-gray-200">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-2">
                      <Clock className="w-4 h-4 text-gray-500" />
                      <span className="text-gray-600">Uptime:</span>
                    </div>
                    <span className="font-medium text-gray-900">
                      {formatUptime(healthStatus.uptime)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Models Information */}
        {modelsInfo && (
          <div className="lg:col-span-3 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
              <Package className="w-5 h-5 text-purple-600" />
              <span>Loaded Models</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {modelsInfo.models?.map((model: any, index: number) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-semibold text-gray-900">{model.name}</h4>
                    {model.loaded ? (
                      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-5 h-5 text-gray-400 flex-shrink-0" />
                    )}
                  </div>

                  {model.version && (
                    <p className="text-xs text-gray-500 mb-2">v{model.version}</p>
                  )}

                  {model.size && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Size:</span>
                      <span className="font-medium text-gray-900">{model.size}</span>
                    </div>
                  )}

                  {model.memory_usage && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Memory:</span>
                      <span className="font-medium text-gray-900">
                        {formatBytes(model.memory_usage)}
                      </span>
                    </div>
                  )}

                  {model.device && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Device:</span>
                      <span className="font-medium text-gray-900">{model.device}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* System Configuration */}
        <div className="lg:col-span-3 bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">System Configuration</h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold text-gray-700 mb-3">Audio Settings</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Sample Rate:</span>
                  <span className="font-medium text-gray-900">44.1 kHz</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Channels:</span>
                  <span className="font-medium text-gray-900">Stereo</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Bit Depth:</span>
                  <span className="font-medium text-gray-900">16-bit</span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-gray-700 mb-3">Conversion Settings</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Default Quality:</span>
                  <span className="font-medium text-gray-900">Balanced</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Max Duration:</span>
                  <span className="font-medium text-gray-900">10 minutes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Batch Size:</span>
                  <span className="font-medium text-gray-900">4 files</span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-gray-700 mb-3">Performance</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">GPU Acceleration:</span>
                  <span className="font-medium text-green-600">Enabled</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Mixed Precision:</span>
                  <span className="font-medium text-green-600">Enabled</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Caching:</span>
                  <span className="font-medium text-green-600">Enabled</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatusItem({ label, status }: { label: string; status: string }) {
  const isOnline = status === 'online' || status === 'available' || status === 'loaded' || status === 'connected'

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-gray-600">{label}:</span>
      <div className="flex items-center space-x-2">
        <div className={clsx(
          'w-2 h-2 rounded-full',
          isOnline ? 'bg-green-500' : 'bg-gray-400'
        )} />
        <span className={clsx(
          'text-sm font-medium',
          isOnline ? 'text-green-600' : 'text-gray-500'
        )}>
          {status}
        </span>
      </div>
    </div>
  )
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)

  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

