import { useQuery } from '@tanstack/react-query'
import { Activity, Cpu, HardDrive, Zap } from 'lucide-react'
import { apiService } from '../services/api'
import clsx from 'clsx'

export function SystemStatusPage() {
  const { data: status, isLoading, error } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => apiService.getSystemStatus(),
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Activity className="w-8 h-8 text-primary-600" />
          <span>System Status</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Monitor GPU utilization and system performance
        </p>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-2 border-red-500 rounded-lg p-6">
          <p className="text-red-800 font-semibold">Failed to load system status</p>
        </div>
      )}

      {status && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* GPU Status */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <div className={clsx(
                'p-3 rounded-lg',
                status.gpu_available ? 'bg-green-100' : 'bg-gray-100'
              )}>
                <Cpu className={clsx(
                  'w-6 h-6',
                  status.gpu_available ? 'text-green-600' : 'text-gray-400'
                )} />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">GPU Status</h2>
                <p className={clsx(
                  'text-sm font-medium',
                  status.gpu_available ? 'text-green-600' : 'text-gray-500'
                )}>
                  {status.gpu_available ? 'Available' : 'Not Available'}
                </p>
              </div>
            </div>

            {status.gpu_available && status.gpu_name && (
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-600">GPU Name</p>
                  <p className="font-semibold text-gray-900">{status.gpu_name}</p>
                </div>

                {status.gpu_memory_total && (
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Memory Usage</span>
                      <span className="font-semibold text-gray-900">
                        {status.gpu_memory_used?.toFixed(1)} / {status.gpu_memory_total.toFixed(1)} GB
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary-600 transition-all"
                        style={{
                          width: `${((status.gpu_memory_used || 0) / status.gpu_memory_total) * 100}%`
                        }}
                      />
                    </div>
                  </div>
                )}

                {status.gpu_utilization !== undefined && (
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">GPU Utilization</span>
                      <span className="font-semibold text-gray-900">
                        {status.gpu_utilization}%
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-accent-600 transition-all"
                        style={{ width: `${status.gpu_utilization}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Model Status */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <div className={clsx(
                'p-3 rounded-lg',
                status.model_loaded ? 'bg-green-100' : 'bg-yellow-100'
              )}>
                <HardDrive className={clsx(
                  'w-6 h-6',
                  status.model_loaded ? 'text-green-600' : 'text-yellow-600'
                )} />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Model Status</h2>
                <p className={clsx(
                  'text-sm font-medium',
                  status.model_loaded ? 'text-green-600' : 'text-yellow-600'
                )}>
                  {status.model_loaded ? 'Loaded' : 'Not Loaded'}
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-sm text-gray-600">
                {status.model_loaded
                  ? 'All models are loaded and ready for conversion'
                  : 'Models will be loaded on first conversion request'}
              </p>
            </div>
          </div>

          {/* System Status */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <div className="p-3 bg-primary-100 rounded-lg">
                <Zap className="w-6 h-6 text-primary-600" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">System Status</h2>
                <p className="text-sm font-medium text-primary-600">
                  {status.status || 'Running'}
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Backend</span>
                <span className="font-semibold text-green-600">Online</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">WebSocket</span>
                <span className="font-semibold text-green-600">Connected</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

